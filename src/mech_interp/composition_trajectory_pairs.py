import os
import pickle
import argparse
import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.models import StoneStateDecoderClassifier
from patching_utils import split_support_query, decode_sequence, compute_patching_score_batched
from tqdm import tqdm
from collections import defaultdict
import itertools
import random

def build_chemistry_potion_seq_sets(dataset, input_word2idx, min_qhop=2):
    """
    Groups samples by (chemistry, potion_sequence).
    Returns mapping from (chemistry, potion_seq) -> list of (sample, query)
    """
    item_sep_id = input_word2idx.get('<item_sep>', 23)
    
    if hasattr(dataset, 'data'):
        samples = dataset.data
    else:
        samples = dataset

    groups = defaultdict(list)
    for sample in samples:
        enc_inps = sample['encoder_input_ids']
        support, query = split_support_query(enc_inps, item_sep_id)
        if len(query) < 4 + min_qhop: # 4 stone features + potions + potentially reward if present? Wait, query format is [color, size, shape, reward, P1, P2...]
            continue
            
        # Potion sequence is everything from index 4 to end
        potion_seq = tuple(query[4:])
        support_key = tuple(support)
        groups[(support_key, potion_seq)].append((sample, query))
        
    return groups

def extract_reachable_sets(groups):
    """
    Extract reachable target sets for each starting stone, grouped by chemistry and potion sequence.
    Returns:
    - chemistry_to_starting_stones: dict mapping (support_key, potion_seq) -> set of starting stones (tuples of 4 features)
    - reachable_sets: dict mapping (support_key, potion_seq, starting_stone) -> set of target_class_ids
    """
    chemistry_to_starting_stones = defaultdict(set)
    reachable_sets = defaultdict(set)
    
    for (support_key, potion_seq), samples in groups.items():
        for sample, query in samples:
            starting_stone = tuple(query[:4])
            target = sample.get('target_class_id')
            if target is not None:
                chemistry_to_starting_stones[(support_key, potion_seq)].add(starting_stone)
                reachable_sets[(support_key, potion_seq, starting_stone)].add(target)
                
    # Convert sets to frozensets
    for k in reachable_sets:
        reachable_sets[k] = frozenset(reachable_sets[k])
        
    return chemistry_to_starting_stones, reachable_sets

def build_last_potion_pairs(dataset, input_word2idx, max_pairs=None):
    """
    Experiment 1: Same support, same query stone, same first k-1 potions, different last potion.
    """
    item_sep_id = input_word2idx.get('<item_sep>', 23)
    
    if hasattr(dataset, 'data'):
        samples = dataset.data
    else:
        samples = dataset

    # Group by (support, query_stone, first_k_minus_1_potions)
    groups = defaultdict(list)
    for sample in samples:
        enc_inps = sample['encoder_input_ids']
        support, query = split_support_query(enc_inps, item_sep_id)
        if len(query) < 5:
            continue
        
        query_stone = tuple(query[:4])
        first_k_minus_1_potions = tuple(query[4:-1])
        last_potion = query[-1]
        
        group_key = (tuple(support), query_stone, first_k_minus_1_potions)
        groups[group_key].append((sample, query, last_potion))
        
    all_pairs = []
    for group_key, items in tqdm(groups.items(), desc="Building Exp 1 Pairs"):
        for i, j in itertools.combinations(range(len(items)), 2):
            sample_A, query_A, last_pot_A = items[i]
            sample_B, query_B, last_pot_B = items[j]
            
            if last_pot_A == last_pot_B:
                continue
                
            target_A = sample_A.get('target_class_id')
            target_B = sample_B.get('target_class_id')
            if target_A is None or target_B is None or target_A == target_B:
                continue
                
            clean_tensor = torch.tensor([sample_A['encoder_input_ids']], dtype=torch.long)
            corrupt_tensor = torch.tensor([sample_B['encoder_input_ids']], dtype=torch.long)
            
            all_pairs.append((clean_tensor, corrupt_tensor, target_A, frozenset([target_A]), frozenset([target_B])))
            all_pairs.append((corrupt_tensor, clean_tensor, target_B, frozenset([target_B]), frozenset([target_A])))
            
    random.seed(42)
    random.shuffle(all_pairs)
    
    if max_pairs is not None and max_pairs > 0:
        all_pairs = all_pairs[:max_pairs]
        
    return all_pairs

def build_query_stone_pairs(dataset, input_word2idx, max_pairs=None):
    """
    Experiment 2: Same support, same full potion sequence, different query stone.
    Returns t_clean and t_corrupt as reachable sets.
    """
    groups = build_chemistry_potion_seq_sets(dataset, input_word2idx)
    chemistry_to_starting_stones, reachable_sets = extract_reachable_sets(groups)
    
    all_pairs = []
    for (support_key, potion_seq), items in tqdm(groups.items(), desc="Building Exp 2 Pairs"):
        for i, j in itertools.combinations(range(len(items)), 2):
            sample_A, query_A = items[i]
            sample_B, query_B = items[j]
            
            stone_A = tuple(query_A[:4])
            stone_B = tuple(query_B[:4])
            
            if stone_A == stone_B:
                continue
                
            target_A = sample_A.get('target_class_id')
            target_B = sample_B.get('target_class_id')
            if target_A is None or target_B is None or target_A == target_B:
                continue
                
            t_clean_set = reachable_sets[(support_key, potion_seq, stone_A)]
            t_corrupt_set = reachable_sets[(support_key, potion_seq, stone_B)]
            
            clean_tensor = torch.tensor([sample_A['encoder_input_ids']], dtype=torch.long)
            corrupt_tensor = torch.tensor([sample_B['encoder_input_ids']], dtype=torch.long)
            
            all_pairs.append((clean_tensor, corrupt_tensor, target_A, t_clean_set, t_corrupt_set))
            all_pairs.append((corrupt_tensor, clean_tensor, target_B, t_corrupt_set, t_clean_set))
            
    random.seed(42)
    random.shuffle(all_pairs)
    
    if max_pairs is not None and max_pairs > 0:
        all_pairs = all_pairs[:max_pairs]
        
    return all_pairs

def run_full_sweep(model, all_pairs, input_word2idx, setup='noising', patch_mlp=False, batch_size=128, experiment_type='last_potion', overlap_strategy='disjoint'):
    components = [
        'embedding',
        'layer_0_head_0', 'layer_0_head_1', 'layer_0_head_2', 'layer_0_head_3',
        'layer_1_head_0', 'layer_1_head_1', 'layer_1_head_2', 'layer_1_head_3',
        'layer_2_head_0', 'layer_2_head_1', 'layer_2_head_2', 'layer_2_head_3',
        'layer_3_head_0', 'layer_3_head_1', 'layer_3_head_2', 'layer_3_head_3',
    ]
    if patch_mlp:
        components.extend([
            'layer_0_mlp_out', 'layer_1_mlp_out', 'layer_2_mlp_out', 'layer_3_mlp_out'
        ])
    
    item_sep_id = input_word2idx.get('<item_sep>', 23)
    results = {}
    
    for comp in tqdm(components, desc="Sweeping components"):
        softmax_list = []
        lse_list = []
        raw_lse_list = []
        
        for i in tqdm(range(0, len(all_pairs), batch_size), desc=f"Batches for {comp}", leave=False):
            batch = all_pairs[i:i+batch_size]
            
            clean_tensors = []
            corrupt_tensors = []
            masks = []
            t_clean_ids_list = []
            t_corrupt_ids_list = []
            
            for pair in batch:
                c_in, corr_in, target_class_id, t_clean, t_corr = pair
                c_in_flat = c_in.squeeze(0)
                corr_in_flat = corr_in.squeeze(0)
                
                clean_tensors.append(c_in_flat)
                corrupt_tensors.append(corr_in_flat)
                t_clean_ids_list.append(t_clean)
                t_corrupt_ids_list.append(t_corr)
                
                support, query = split_support_query(c_in_flat.tolist(), item_sep_id)
                mask = torch.zeros(len(c_in_flat), dtype=torch.bool)
                
                if experiment_type == 'last_potion':
                    # Mask the final token of the sequence (which is the last potion)
                    mask[-1] = True
                elif experiment_type == 'query_stone':
                    # Mask the first 4 tokens of the query
                    start_idx = len(support)
                    mask[start_idx:start_idx+4] = True
                
                masks.append(mask)
                
            from torch.nn.utils.rnn import pad_sequence
            pad_id = input_word2idx.get('<pad>', 19)
            
            clean_batch = pad_sequence(clean_tensors, batch_first=True, padding_value=pad_id)
            corrupt_batch = pad_sequence(corrupt_tensors, batch_first=True, padding_value=pad_id)
            mask_batch = pad_sequence(masks, batch_first=True, padding_value=False)
            
            max_len = max(clean_batch.shape[1], corrupt_batch.shape[1], mask_batch.shape[1])
            
            if clean_batch.shape[1] < max_len:
                clean_batch = F.pad(clean_batch, (0, max_len - clean_batch.shape[1]), value=pad_id)
            if corrupt_batch.shape[1] < max_len:
                corrupt_batch = F.pad(corrupt_batch, (0, max_len - corrupt_batch.shape[1]), value=pad_id)
            if mask_batch.shape[1] < max_len:
                mask_batch = F.pad(mask_batch, (0, max_len - mask_batch.shape[1]), value=False)
                
            scores = compute_patching_score_batched(
                model=model,
                clean_batch=clean_batch,
                corrupt_batch=corrupt_batch,
                component_name=comp,
                target_token_masks=mask_batch,
                t_clean_ids_list=t_clean_ids_list,
                t_corrupt_ids_list=t_corrupt_ids_list,
                setup=setup,
                overlap_strategy=overlap_strategy
            )
            
            softmax_list.extend(scores["softmax_scores"])
            lse_list.extend(scores["lse_scores"])
            raw_lse_list.extend(scores["raw_lse_scores"])
            
        softmax_mean = sum(softmax_list) / len(softmax_list) if len(softmax_list) > 0 else 0.0
        lse_mean = sum(lse_list) / len(lse_list) if len(lse_list) > 0 else 0.0
        raw_lse_mean = sum(raw_lse_list) / len(raw_lse_list) if len(raw_lse_list) > 0 else 0.0
        
        results[comp] = {
            "softmax_mean": float(softmax_mean),
            "lse_mean": float(lse_mean),
            "raw_lse_mean": float(raw_lse_mean),
            "softmax_all": softmax_list,
            "lse_all": lse_list,
            "raw_lse_all": raw_lse_list
        }
        
    sorted_comps = sorted(components, key=lambda c: results[c]["raw_lse_mean"], reverse=True)
    
    print("\n" + "="*100)
    print(f"{'Component Name':<25} | {'Mean Softmax Score':<22} | {'Mean LSE Score':<20} | {'Raw LSE Score':<20}")
    print("-"*100)
    for comp in sorted_comps:
        print(f"{comp:<25} | {results[comp]['softmax_mean']:<22.6f} | {results[comp]['lse_mean']:<20.6f} | {results[comp]['raw_lse_mean']:<20.6f}")
    print("="*100 + "\n")
    if overlap_strategy == 'subtract' and experiment_type == 'query_stone':
        print("\nNote: For Experiment 2 (Reachable Set), overlapping targets between clean and corrupt reachable sets were subtracted before scoring.")
    elif overlap_strategy == 'allow' and experiment_type == 'query_stone':
        print("\nNote: For Experiment 2 (Reachable Set), overlapping targets were ALLOWED to remain in the target sets.")
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Composition Task Activation Patching Experiments")
    parser.add_argument("--experiment", type=str, required=True, choices=["last_potion", "query_stone"],
                        help="Experiment to run: 'last_potion' (Exp 1) or 'query_stone' (Exp 2)")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to model checkpoint .pt file or dir")
    parser.add_argument("--val_data", type=str, required=True, help="Path to validation data pickle")
    parser.add_argument("--calibration_exp", type=str, default="True", choices=["True", "False"])
    parser.add_argument("--max_pairs", type=int, default=2000)
    parser.add_argument("--setup", type=str, default="noising", choices=["noising", "denoising"])
    parser.add_argument("--epoch_range", type=str, default=None, help="e.g. '50-150'")
    parser.add_argument("--patch_mlp", type=str, default="True", choices=["True", "False"])
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--overlap_strategy", type=str, default="subtract", choices=["disjoint", "subtract", "allow"],
                        help="How to handle overlapping reachable sets in Exp 2. Default is subtract (remove intersection).")
    
    args = parser.parse_args()
    args.calibration_exp = str(args.calibration_exp).lower() == "true"
    args.patch_mlp = str(args.patch_mlp).lower() == "true"
    
    import glob, re
    
    checkpoint_dir = args.checkpoint_path if os.path.isdir(args.checkpoint_path) else os.path.dirname(args.checkpoint_path)
    if os.path.isdir(args.checkpoint_path):
        pt_files = glob.glob(os.path.join(checkpoint_dir, "best_model_epoch_*_classification_xsmall.pt"))
        if not pt_files:
            print(f"Error: No checkpoints found in directory {checkpoint_dir}")
            return
        args.checkpoint_path = sorted(pt_files)[-1]
    
    wandb.init(project="mech_interp_alchemy", config=vars(args))

    vocab_path = args.val_data.replace('_data.pkl', '_vocab.pkl')
    
    if not os.path.exists(args.val_data) or not os.path.exists(vocab_path):
        print(f"Error: Could not locate data or vocab pickle files.")
        return
        
    print(f"Loading validation data from: {args.val_data}")
    with open(args.val_data, 'rb') as f:
        dataset = pickle.load(f)
        
    print(f"Loading vocabulary from: {vocab_path}")
    with open(vocab_path, 'rb') as f:
        vocab_data = pickle.load(f)
        
    input_word2idx = vocab_data.get('input_word2idx', vocab_data.get('word2idx'))
    idx2word = {v: k for k, v in input_word2idx.items()}

    print(f"\nBuilding {args.experiment} pairs...")
    if args.experiment == 'last_potion':
        pairs = build_last_potion_pairs(dataset, input_word2idx, max_pairs=args.max_pairs)
    else:
        pairs = build_query_stone_pairs(dataset, input_word2idx, max_pairs=args.max_pairs)
        
    print(f"Successfully generated {len(pairs)} pairs.")
    
    num_to_print = min(3, len(pairs))
    print(f"\n--- Printing {num_to_print} sample pairs for manual inspection ---")
    for idx in range(num_to_print):
        clean_tensor, corrupt_tensor, target_class_id, t_clean_ids, t_corrupt_ids = pairs[idx]
        print(f"\nPair {idx + 1}:")
        print(f"  Target Class ID: {target_class_id}")
        print(f"  Clean Targets: {t_clean_ids}")
        print(f"  Corrupt Targets: {t_corrupt_ids}")
        if args.overlap_strategy == 'subtract':
            intersection = t_clean_ids & t_corrupt_ids
            if intersection:
                print(f"  Intersection to be subtracted: {intersection}")
        print(f"  Clean Decoded:   {decode_sequence(clean_tensor, idx2word)}")
        print(f"  Corrupt Decoded: {decode_sequence(corrupt_tensor, idx2word)}")
        
    # --- Loading Model Checkpoint ---
    print(f"\nLoading checkpoint from: {args.checkpoint_path}")
    checkpoint = torch.load(args.checkpoint_path, map_location='cpu', weights_only=False)
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    
    detected_max_len = state_dict.get("positional_encoding.pe", torch.zeros(5000)).shape[0]
    detected_num_classes = state_dict.get("classification_head.weight", torch.zeros(80)).shape[0]
    detected_src_vocab_size = state_dict.get("src_tok_emb.weight", torch.zeros(len(input_word2idx))).shape[0]
        
    model_config = {
        "num_decoder_layers": 4, "emb_size": 256, "nhead": 4, "dim_feedforward": 512,
        "dropout": 0.1, "src_vocab_size": detected_src_vocab_size,
        "num_classes": detected_num_classes, "use_flash_attention": True, "max_len": detected_max_len
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = StoneStateDecoderClassifier(**model_config)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    if args.calibration_exp:
        print("\n--- Running Calibration Experiment ---")
        # Just use a tiny subset for calibration
        calib_pairs = pairs[:args.batch_size*2] if len(pairs) > args.batch_size*2 else pairs
        
        sum_scores_clean_softmax = 0.0
        sum_scores_clean_lse = 0.0
        
        item_sep_id = input_word2idx.get('<item_sep>', 23)
        for i in tqdm(range(0, len(calib_pairs), args.batch_size), desc="Calibration"):
            batch = calib_pairs[i:i+args.batch_size]
            clean_tensors, masks, t_clean_ids_list, t_corrupt_ids_list = [], [], [], []
            for pair in batch:
                c_in, corr_in, target_class_id, t_clean, t_corr = pair
                c_in_flat = c_in.squeeze(0)
                clean_tensors.append(c_in_flat)
                t_clean_ids_list.append(t_clean)
                t_corrupt_ids_list.append(t_corr)
                
                support, query = split_support_query(c_in_flat.tolist(), item_sep_id)
                mask = torch.zeros(len(c_in_flat), dtype=torch.bool)
                if args.experiment == 'last_potion':
                    mask[-1] = True
                else:
                    start_idx = len(support)
                    mask[start_idx:start_idx+4] = True
                masks.append(mask)
                
            from torch.nn.utils.rnn import pad_sequence
            pad_id = input_word2idx.get('<pad>', 19)
            clean_batch = pad_sequence(clean_tensors, batch_first=True, padding_value=pad_id)
            mask_batch = pad_sequence(masks, batch_first=True, padding_value=False)
            full_mask = torch.ones_like(mask_batch, dtype=torch.bool)
            
            scores = compute_patching_score_batched(
                model=model, clean_batch=clean_batch, corrupt_batch=clean_batch,
                component_name='embedding', target_token_masks=full_mask,
                t_clean_ids_list=t_clean_ids_list, t_corrupt_ids_list=t_corrupt_ids_list,
                setup=args.setup, overlap_strategy=args.overlap_strategy
            )
            sum_scores_clean_softmax += sum(scores["softmax_scores"])
            sum_scores_clean_lse += sum(scores["lse_scores"])
            
        N = len(calib_pairs)
        print(f"Mean Score for Clean-on-Clean patching (Softmax): {sum_scores_clean_softmax/N:.6f}")
        print(f"Mean Score for Clean-on-Clean patching (LSE): {sum_scores_clean_lse/N:.6f}")

    if args.epoch_range:
        try:
            start_ep, end_ep = map(int, args.epoch_range.split('-'))
        except ValueError:
            print("Invalid epoch_range format.")
            return
            
        pt_files = glob.glob(os.path.join(checkpoint_dir, "best_model_epoch_*_classification_xsmall.pt"))
        epoch_files = []
        for f in pt_files:
            m = re.search(r'best_model_epoch_(\d+)_', os.path.basename(f))
            if m:
                ep = int(m.group(1))
                if start_ep <= ep <= end_ep:
                    epoch_files.append((ep, f))
        epoch_files.sort()
    else:
        m = re.search(r'best_model_epoch_(\d+)_', os.path.basename(args.checkpoint_path))
        ep = int(m.group(1)) if m else 0
        epoch_files = [(ep, args.checkpoint_path)]
        
    print(f"\nFound {len(epoch_files)} checkpoints to process.")
    
    all_epochs_results = {}
    rel_match = re.search(r'saved_models/(.*?init_seed_\d+)', args.checkpoint_path)
    suffix = rel_match.group(1) if rel_match else "default_run"
            
    base_save_dir = f"/home/rsaha/projects/aip-afyshe/rsaha/dm_alchemy/mech_interp_results/composition/{suffix}/{args.experiment}"
    os.makedirs(base_save_dir, exist_ok=True)
    
    out_pkl_path = os.path.join(base_save_dir, f"layer_head_sweep_results_{args.setup}.pkl")
    if args.epoch_range:
        out_pkl_path = os.path.join(base_save_dir, f"layer_head_sweep_results_epochs_{args.epoch_range}_{args.setup}.pkl")
        
    for epoch, cp_path in epoch_files:
        print(f"\n=== Processing Epoch {epoch} ===")
        if len(epoch_files) > 1:
            checkpoint = torch.load(cp_path, map_location='cpu', weights_only=False)
            model.load_state_dict(checkpoint.get('model_state_dict', checkpoint))
            model.eval()
            
        sweep_results = run_full_sweep(model, pairs, input_word2idx, setup=args.setup, patch_mlp=args.patch_mlp, batch_size=args.batch_size, experiment_type=args.experiment, overlap_strategy=args.overlap_strategy)
        all_epochs_results[epoch] = sweep_results
        
    with open(out_pkl_path, 'wb') as f:
        pickle.dump(all_epochs_results, f)
    print(f"\nResults saved to {out_pkl_path}")

if __name__ == "__main__":
    main()
