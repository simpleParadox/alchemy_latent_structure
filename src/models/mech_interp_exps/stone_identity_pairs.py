import os
import pickle
import argparse
import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.models import StoneStateDecoderClassifier
from activation_cache import ActivationCacheManager
from tqdm import tqdm

_printed_attn_shapes = set()

def split_support_query(encoder_input_ids, item_sep_id):
    """
    Splits encoder_input_ids sequence into support prefix and query suffix at the last item_sep_id.
    """
    enc_inps_list = encoder_input_ids.tolist() if hasattr(encoder_input_ids, "tolist") else list(encoder_input_ids)
    try:
        # Find index of last item_sep_id in sequence
        last_sep_idx = len(enc_inps_list) - 1 - enc_inps_list[::-1].index(item_sep_id)
        query = enc_inps_list[last_sep_idx + 1:]
        support = enc_inps_list[:last_sep_idx + 1]
    except ValueError:
        # No separator found (empty support set)
        query = enc_inps_list
        support = []
    return support, query

def extract_stones_from_support(support_tokens, io_id, item_sep_id, potion_ids, special_ids):
    """
    Extract unique stone 4-tuples (color, size, shape, reward) from support tokens.
    Each support item has the structure:
        [color size shape reward POTION <io> color size shape reward <item_sep>]
    Returns a frozenset of stone 4-tuples (as tuples of token IDs).
    """
    stones = set()
    # Split by item_sep to get individual items
    items = []
    current = []
    for t in support_tokens:
        if t == item_sep_id:
            items.append(current)
            current = []
        else:
            current.append(t)
    if current:
        items.append(current)
    
    for item in items:
        if io_id not in item:
            continue
        io_idx = item.index(io_id)
        input_part = item[:io_idx]
        output_part = item[io_idx + 1:]
        
        # Input stone: first 4 non-potion, non-special tokens (color size shape reward)
        input_stone_tokens = []
        for t in input_part:
            if t not in potion_ids and t not in special_ids:
                input_stone_tokens.append(t)
                if len(input_stone_tokens) == 4:
                    break
        
        # Output stone: first 4 non-special tokens after <io>
        output_stone_tokens = []
        for t in output_part:
            if t not in special_ids:
                output_stone_tokens.append(t)
                if len(output_stone_tokens) == 4:
                    break
        
        if len(input_stone_tokens) == 4:
            stones.add(tuple(input_stone_tokens))
        if len(output_stone_tokens) == 4:
            stones.add(tuple(output_stone_tokens))
    
    return frozenset(stones)

def build_all_stone_identity_pairs(dataset, input_word2idx=None, max_pairs=None):
    """
    Returns a list of (clean_input, corrupt_input, target_class_id, t_clean, t_corrupt) tuples.
    Pairs are selected within the same chemistry (support) where query stones have:
    - Same potion
    - Same reward
    - Different perceptual features
    - Different target class IDs
    """
    hardcoded_vocab = {
        '-1': 0, '-3': 1, '1': 2, '3': 3, 'CYAN': 4, 'GREEN': 5, 'ORANGE': 6, 'PINK': 7, 'RED': 8, 'YELLOW': 9,
        'blue': 10, 'large': 11, 'medium': 12, 'medium_round': 13, 'pointy': 14, 'purple': 15, 'red': 16,
        'round': 17, 'small': 18, '<pad>': 19, '<sos>': 20, '<eos>': 21, '<io>': 22, '<item_sep>': 23, '<unk>': 24
    }
    vocab = input_word2idx if input_word2idx is not None else hardcoded_vocab
    item_sep_id = vocab.get('<item_sep>', 23)
    
    if hasattr(dataset, 'data'):
        samples = dataset.data
    else:
        samples = dataset
        
    from collections import defaultdict
    import itertools
    
    # Group samples by support (chemistry)
    support_to_samples = defaultdict(list)
    for sample in samples:
        enc_inps = sample['encoder_input_ids']
        support, query = split_support_query(enc_inps, item_sep_id)
        if len(query) < 5:
            continue
        support_to_samples[tuple(support)].append((sample, query))
        
    all_pairs = []
    
    for support_key, group in tqdm(support_to_samples.items(), desc="Building pairs within chemistries"):
        for i, j in itertools.combinations(range(len(group)), 2):
            sample_A, query_A = group[i]
            sample_B, query_B = group[j]
            
            # Same potion?
            if query_A[-1] != query_B[-1]:
                continue
                
            # Same reward?
            if query_A[-2] != query_B[-2]:
                continue
                
            # Different perceptual features?
            perceptual_A = query_A[:-2]
            perceptual_B = query_B[:-2]
            if perceptual_A == perceptual_B:
                continue
                
            # Different targets?
            target_A = sample_A.get('target_class_id')
            target_B = sample_B.get('target_class_id')
            if target_A is None or target_B is None or target_A == target_B:
                continue
                
            clean_input = sample_A['encoder_input_ids']
            corrupt_input = sample_B['encoder_input_ids']
            
            clean_tensor = torch.tensor([clean_input], dtype=torch.long)
            corrupt_tensor = torch.tensor([corrupt_input], dtype=torch.long)
            
            # Return target lists containing the single class ID so metrics work seamlessly
            all_pairs.append((clean_tensor, corrupt_tensor, target_A, frozenset([target_A]), frozenset([target_B])))
            # Add reverse pair as well for robustness
            all_pairs.append((corrupt_tensor, clean_tensor, target_B, frozenset([target_B]), frozenset([target_A])))
            
    import random
    random.seed(42)
    random.shuffle(all_pairs)
            
    if max_pairs is not None and max_pairs > 0:
        all_pairs = all_pairs[:max_pairs]
        
    return all_pairs

def decode_sequence(token_ids, idx2word):
    """
    Decodes a tensor or list of token IDs into a space-separated string of token names.
    """
    if torch.is_tensor(token_ids):
        token_ids = token_ids.cpu().numpy()
    import numpy as np
    flat_ids = np.array(token_ids).flatten()
    return " ".join([idx2word.get(int(tid), f"<unk_{tid}>") for tid in flat_ids])

def make_new_fwd(fwd, self_attn_mod, layer_idx, cache_dict=None, patch_config=None):
    def new_fwd(query, key, value, *args, **kwargs):
        orig_weight = self_attn_mod.out_proj.weight
        orig_bias = self_attn_mod.out_proj.bias
        device = orig_weight.device
        dtype = orig_weight.dtype
        d_model = orig_weight.shape[0]
        
        self_attn_mod.out_proj.weight = nn.Parameter(torch.eye(d_model, device=device, dtype=dtype))
        if orig_bias is not None:
            self_attn_mod.out_proj.bias = nn.Parameter(torch.zeros(d_model, device=device, dtype=dtype))
            
        pre_proj, weights = fwd(query, key, value, *args, **kwargs)
        
        self_attn_mod.out_proj.weight = orig_weight
        self_attn_mod.out_proj.bias = orig_bias
        
        batch_first = getattr(self_attn_mod, 'batch_first', True)
        act_to_process = pre_proj
        if not batch_first:
            act_to_process = pre_proj.transpose(0, 1)
            
        assert act_to_process.shape[-1] == d_model, f"Expected self_attn_output shape[-1] to be {d_model}, got {act_to_process.shape}"
        
        global _printed_attn_shapes
        layer_attn_name = f"layer_{layer_idx}_self_attn"
        if layer_attn_name not in _printed_attn_shapes:
            print(f"[First forward pass] {layer_attn_name} pre-projection output shape: {list(act_to_process.shape)}")
            _printed_attn_shapes.add(layer_attn_name)
            
        if cache_dict is not None:
            cache_dict[layer_attn_name] = act_to_process.detach()
            
        if patch_config is not None and patch_config.get('layer_idx') == layer_idx:
            patched_act = act_to_process.clone()
            t_mask = patch_config.get('target_token_masks')
            c_act = patch_config.get('corrupt_activation')
            head_idx = patch_config.get('head_idx')
            
            if not batch_first:
                mask_expanded = t_mask.transpose(0, 1).unsqueeze(-1)
            else:
                mask_expanded = t_mask.unsqueeze(-1)
            
            if head_idx is not None:
                d_head = d_model // getattr(self_attn_mod, 'num_heads', 4)
                h_start = head_idx * d_head
                h_end = (head_idx + 1) * d_head
                
                p_slice = patched_act[:, :, h_start:h_end]
                c_slice = c_act[:, :, h_start:h_end]
                patched_act[:, :, h_start:h_end] = torch.where(mask_expanded, c_slice, p_slice)
            else:
                patched_act = torch.where(mask_expanded, c_act, patched_act)
                    
            if not batch_first:
                pre_proj = patched_act.transpose(0, 1)
            else:
                pre_proj = patched_act
                
        projected = F.linear(pre_proj, orig_weight, orig_bias)
        return projected, weights
    return new_fwd

def patch_attention_modules(model, cache_dict, patch_config=None):
    original_forwards = {}
    for i, layer in enumerate(model.transformer_encoder.layers):
        orig_fwd = layer.self_attn.forward
        original_forwards[i] = orig_fwd
        layer.self_attn.forward = make_new_fwd(orig_fwd, layer.self_attn, i, cache_dict=cache_dict, patch_config=patch_config)
    return original_forwards

def unpatch_attention_modules(model, original_forwards):
    for i, orig_fwd in original_forwards.items():
        model.transformer_encoder.layers[i].self_attn.forward = orig_fwd

def compute_patching_score_batched(model, clean_batch, corrupt_batch, component_name, target_token_masks, t_clean_ids_list, t_corrupt_ids_list, setup='noising'):
    """
    Computes the patching scores using both softmax delta and logsumexp diff metrics across a batch.
    """
    for clean, corrupt in zip(t_clean_ids_list, t_corrupt_ids_list):
        assert clean & corrupt == frozenset(), f"Target class overlap detected in scoring: {clean & corrupt}"
    device = next(model.parameters()).device
    
    if setup == 'noising':
        base_input = clean_batch.to(device)
        patch_input = corrupt_batch.to(device)
    elif setup == 'denoising':
        base_input = corrupt_batch.to(device)
        patch_input = clean_batch.to(device)
    else:
        raise ValueError(f"Unknown setup: {setup}")
        
    target_token_masks = target_token_masks.to(device)
    
    # 1. Base run forward pass and cache activations
    base_attn_cache = {}
    orig_forwards = patch_attention_modules(model, base_attn_cache)
    try:
        with ActivationCacheManager(model) as base_cache:
            f_base = model(base_input)
            base_acts = base_cache.get_activations()
    finally:
        unpatch_attention_modules(model, orig_forwards)
        
    for k, v in base_attn_cache.items():
        base_acts[k] = v
        
    # 2. Patch source run forward pass and cache activations
    source_attn_cache = {}
    orig_forwards = patch_attention_modules(model, source_attn_cache)
    try:
        with ActivationCacheManager(model) as source_cache:
            f_source = model(patch_input)
            source_acts = source_cache.get_activations()
    finally:
        unpatch_attention_modules(model, orig_forwards)
        
    for k, v in source_attn_cache.items():
        source_acts[k] = v
        
    if setup == 'noising':
        f_clean = f_base
        f_corrupt = f_source
    else:
        f_clean = f_source
        f_corrupt = f_base
        
    if component_name.startswith('layer_') and '_head_' in component_name:
        layer_idx = int(component_name.split('_')[1])
        head_idx = int(component_name.split('_')[3])
        
        patch_config = {
            'layer_idx': layer_idx,
            'head_idx': head_idx,
            'target_token_masks': target_token_masks,
            'corrupt_activation': source_acts.get(f"layer_{layer_idx}_self_attn")
        }
        
        orig_forwards = patch_attention_modules(model, None, patch_config=patch_config)
        try:
            f_patched = model(base_input)
        except Exception as e:
            raise Exception(f"Errored out forward pass for head patching. Error: {e}")
        finally:
            unpatch_attention_modules(model, orig_forwards)
            
    else:
        if component_name == 'embedding':
            target_module = model.src_tok_emb
        elif component_name.startswith('layer_') and component_name.endswith('_output'):
            layer_idx = int(component_name.split('_')[1])
            target_module = model.transformer_encoder.layers[layer_idx]
        elif component_name.startswith('layer_') and component_name.endswith('_mlp_out'):
            layer_idx = int(component_name.split('_')[1])
            target_module = model.transformer_encoder.layers[layer_idx].linear2
        else:
            raise ValueError(f"Unknown component_name '{component_name}'")
            
        corrupt_act = source_acts.get(component_name)
        if corrupt_act is None:
            raise ValueError(f"Component '{component_name}' not found in corrupt cached activations. Keys are: {list(source_acts.keys())}")
            
        def make_patch_hook(c_act, t_mask):
            def hook(module, inp, outp):
                is_tuple = isinstance(outp, tuple)
                out_tensor = outp[0] if is_tuple else outp
                patched = out_tensor.clone()
                
                batch_first = getattr(module, 'batch_first', True)
                if not batch_first:
                    mask_expanded = t_mask.transpose(0, 1).unsqueeze(-1)
                else:
                    mask_expanded = t_mask.unsqueeze(-1)
                    
                patched = torch.where(mask_expanded, c_act, patched)
                        
                if is_tuple:
                    return (patched,) + outp[1:]
                return patched
            return hook
            
        handle = target_module.register_forward_hook(make_patch_hook(corrupt_act, target_token_masks))
        
        try:
            f_patched = model(base_input)
        except Exception as e:
            raise Exception(f"Errored out forward pass for the third run, removing hook. Error: {e}")
        finally:
            handle.remove()
            
    softmax_scores = []
    lse_scores = []
    raw_lse_scores = []
    
    probs_clean = torch.nn.functional.softmax(f_clean, dim=-1)
    probs_corrupt = torch.nn.functional.softmax(f_corrupt, dim=-1)
    probs_patched = torch.nn.functional.softmax(f_patched, dim=-1)
    
    B = f_clean.shape[0]
    for b in range(B):
        c_ids = list(t_clean_ids_list[b])
        corr_ids = list(t_corrupt_ids_list[b])
        
        # Softmax delta
        def s_delta(probs):
            return probs[b, c_ids].sum().item() - probs[b, corr_ids].sum().item()
            
        sd_clean = s_delta(probs_clean)
        sd_corrupt = s_delta(probs_corrupt)
        sd_patched = s_delta(probs_patched)
        
        denom_softmax = sd_clean - sd_corrupt
        
        # LSE diff
        def l_diff(logits):
            return torch.logsumexp(logits[b, c_ids], dim=0).item() - torch.logsumexp(logits[b, corr_ids], dim=0).item()
            
        ld_clean = l_diff(f_clean)
        ld_corrupt = l_diff(f_corrupt)
        ld_patched = l_diff(f_patched)
        
        denom_lse = ld_clean - ld_corrupt
        
        if setup == 'noising':
            raw_lse_score = ld_clean - ld_patched
            softmax_score = (sd_clean - sd_patched) / denom_softmax if abs(denom_softmax) > 1e-7 else 0.0
            lse_score = raw_lse_score / denom_lse if abs(denom_lse) > 1e-7 else 0.0
        elif setup == 'denoising':
            raw_lse_score = ld_patched - ld_corrupt
            softmax_score = (sd_patched - sd_corrupt) / denom_softmax if abs(denom_softmax) > 1e-7 else 0.0
            lse_score = raw_lse_score / denom_lse if abs(denom_lse) > 1e-7 else 0.0
            
        softmax_scores.append(softmax_score)
        lse_scores.append(lse_score)
        raw_lse_scores.append(raw_lse_score)
        
    return {
        "softmax_scores": softmax_scores,
        "lse_scores": lse_scores,
        "raw_lse_scores": raw_lse_scores
    }

def run_full_sweep(model, all_pairs, input_word2idx, setup='noising', patch_mlp=False, batch_size=128):
    components = [
        'embedding',
        # 'layer_0_output', 'layer_1_output', 'layer_2_output', 'layer_3_output',
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
                
                start_idx = len(support)
                end_idx = len(c_in_flat) - 1
                mask[start_idx:end_idx] = True
                
                assert c_in_flat[end_idx - 1] == corr_in_flat[end_idx - 1], "Reward token mismatch!"
                
                masks.append(mask)
                
            from torch.nn.utils.rnn import pad_sequence
            import torch.nn.functional as F
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
                setup=setup
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
        
    # Print results table sorted by mean raw LSE score descending
    sorted_comps = sorted(components, key=lambda c: results[c]["raw_lse_mean"], reverse=True)
    
    print("\n" + "="*100)
    print(f"{'Component Name':<25} | {'Mean Softmax Score':<22} | {'Mean LSE Score':<20} | {'Raw LSE Score':<20}")
    print("-"*100)
    for comp in sorted_comps:
        print(f"{comp:<25} | {results[comp]['softmax_mean']:<22.6f} | {results[comp]['lse_mean']:<20.6f} | {results[comp]['raw_lse_mean']:<20.6f}")
    print("="*100 + "\n")
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Chemistry Identity Pairs and Activation Patching Experiments")
    parser.add_argument("--checkpoint_path", type=str, 
                        default='/home/rsaha/projects/aip-afyshe/rsaha/dm_alchemy/src/saved_models/held_out_color_exp/held_out_edges_4/all_graphs/scheduler_cosine/wd_0.01_lr_0.0001/eta_min_7e-05/xsmall/decoder/classification/input_features/output_stone_states/shop_1_qhop_1/seed_0/init_seed_3/best_model_epoch_100_classification_xsmall.pt', 
                        help="Path to model checkpoint .pt file")
    parser.add_argument("--val_data", type=str, 
                        default='src/data/shuffled_held_out_exps_preprocessed_separate_enhanced/compositional_chemistry_samples_167424_80_unique_stones_val_shop_1_qhop_1_single_held_out_color_4_edges_exp_seed_0_classification_filter_True_input_features_output_stone_states_data.pkl', 
                        help="Path to validation data pickle")
    parser.add_argument("--calibration_exp", type=str, default="True", choices=["True", "False"],
                        help="Run the embedding-level calibration sanity check (defaults to True for verification)")
    parser.add_argument("--max_pairs", type=int, default=2000, help="Maximum number of pairs to use for experiments (set to 0 for full dataset)")
    parser.add_argument("--setup", type=str, default="noising", choices=["noising", "denoising"],
                        help="Experiment setup: 'noising' patches clean with corrupt, 'denoising' patches corrupt with clean.")
    parser.add_argument("--epoch_range", type=str, default=None,
                        help="Range of epochs to run over in the checkpoint directory, e.g. '100-200'. Runs single checkpoint if None.")
    parser.add_argument("--patch_mlp", type=str, default="True", choices=["True", "False"],
                        help="Include MLP layer outputs in the component sweep.")
    parser.add_argument("--batch_size", type=int, default=128,
                        help="Batch size for forward passes")
    
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
        # Use the last checkpoint for the calibration check if a directory was passed
        args.checkpoint_path = sorted(pt_files)[-1]
    
    # Initialize W&B run
    wandb.init(project="mech_interp_alchemy", config=vars(args))

    
    vocab_path = args.val_data.replace('_data.pkl', '_vocab.pkl')
    
    if not os.path.exists(args.val_data) or not os.path.exists(vocab_path):
        print(f"Error: Could not locate data or vocab pickle files.")
        print(f"Data path: {args.val_data}")
        print(f"Vocab path: {vocab_path}")
        return
        
    print(f"Loading validation data from: {args.val_data}")
    with open(args.val_data, 'rb') as f:
        dataset = pickle.load(f)
        
    print(f"Loading vocabulary from: {vocab_path}")
    with open(vocab_path, 'rb') as f:
        vocab_data = pickle.load(f)
        
    if 'input_word2idx' in vocab_data:
        input_word2idx = vocab_data['input_word2idx']
    else:
        print("Could not load input_word2idx from the vocabulary, trying 'word2idx'.")
        input_word2idx = vocab_data.get('word2idx', None)
        print("Loaded input_word2idx: ", input_word2idx)
        
    idx2word = {v: k for k, v in input_word2idx.items()}

    # 1. Print sample pairs and decoded sequences
    print("\nBuilding stone identity pairs...")
    pairs = build_all_stone_identity_pairs(dataset, input_word2idx, max_pairs=3)
    print(f"Successfully generated {len(pairs)} pairs.")
    
    num_to_print = min(3, len(pairs))
    print(f"\n--- Printing {num_to_print} sample pairs for manual inspection ---")
    for idx in range(num_to_print):
        clean_tensor, corrupt_tensor, target_class_id, t_clean_ids, t_corrupt_ids = pairs[idx]
        print(f"\nPair {idx + 1}:")
        print(f"  Target Class ID: {target_class_id}")
        print(f"  Overlap of Stone Sets: {t_clean_ids & t_corrupt_ids}")
        print(f"  Clean Decoded:   {decode_sequence(clean_tensor, idx2word)}")
        print(f"  Corrupt Decoded: {decode_sequence(corrupt_tensor, idx2word)}")
        
    # 2. Calibration experiment if requested
    if args.calibration_exp:
        if not os.path.exists(args.checkpoint_path):
            print(f"\nWARNING: Checkpoint path {args.checkpoint_path} not found. Skipping calibration check.")
            return
            
        print(f"\nLoading checkpoint from: {args.checkpoint_path}")
        checkpoint = torch.load(args.checkpoint_path, map_location='cpu', weights_only=False)
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        
        # Dynamic parameter detection from the checkpoint state_dict
        detected_max_len = 5000
        if "positional_encoding.pe" in state_dict:
            detected_max_len = state_dict["positional_encoding.pe"].shape[0]
            
        detected_num_classes = 80
        if "classification_head.weight" in state_dict:
            detected_num_classes = state_dict["classification_head.weight"].shape[0]
            
        detected_src_vocab_size = len(input_word2idx)
        if "src_tok_emb.weight" in state_dict:
            detected_src_vocab_size = state_dict["src_tok_emb.weight"].shape[0]
            
        # Instantiate model based on standard configuration
        model_config = {
            "num_decoder_layers": 4,
            "emb_size": 256,
            "nhead": 4,
            "dim_feedforward": 512,
            "dropout": 0.1,
            "src_vocab_size": detected_src_vocab_size,
            "num_classes": detected_num_classes,
            "use_flash_attention": True,
            "max_len": detected_max_len
        }
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        model = StoneStateDecoderClassifier(**model_config)
        model.load_state_dict(state_dict)
        model = model.to(device)
        model.eval()
        
        print("\n--- Running Calibration Experiment ---")
        
        sum_scores_clean_softmax = 0.0
        sum_scores_clean_lse = 0.0
        sum_scores_corrupt_softmax = 0.0
        sum_scores_corrupt_lse = 0.0

        item_sep_id = input_word2idx.get('<item_sep>', 23)
        
        for i in tqdm(range(0, len(pairs), args.batch_size), desc="Calibration"):
            batch = pairs[i:i+args.batch_size]
            
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
                
                start_idx = len(support)
                end_idx = len(c_in_flat) - 1
                mask[start_idx:end_idx] = True
                
                assert c_in_flat[end_idx - 1] == corr_in_flat[end_idx - 1], "Reward token mismatch!"
                
                masks.append(mask)
                
            from torch.nn.utils.rnn import pad_sequence
            import torch.nn.functional as F
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
                
            full_mask = torch.ones_like(mask_batch, dtype=torch.bool)
            
            # Clean-on-clean calibration check
            scores_clean_on_clean = compute_patching_score_batched(
                model=model,
                clean_batch=clean_batch,
                corrupt_batch=clean_batch, # clean-on-clean
                component_name='embedding',
                target_token_masks=full_mask,
                t_clean_ids_list=t_clean_ids_list,
                t_corrupt_ids_list=t_corrupt_ids_list,
                setup=args.setup
            )
            sum_scores_clean_softmax += sum(scores_clean_on_clean["softmax_scores"])
            sum_scores_clean_lse += sum(scores_clean_on_clean["lse_scores"])
            
            # Corrupt-on-clean calibration check
            scores_corrupt_on_clean = compute_patching_score_batched(
                model=model,
                clean_batch=clean_batch,
                corrupt_batch=corrupt_batch,
                component_name='embedding',
                target_token_masks=mask_batch,
                t_clean_ids_list=t_clean_ids_list,
                t_corrupt_ids_list=t_corrupt_ids_list,
                setup=args.setup
            )
            sum_scores_corrupt_softmax += sum(scores_corrupt_on_clean["softmax_scores"])
            sum_scores_corrupt_lse += sum(scores_corrupt_on_clean["lse_scores"])
            
        N = len(pairs)
        print(f"Mean Score for Clean-on-Clean patching (Softmax): {sum_scores_clean_softmax/N:.6f}")
        print(f"Mean Score for Clean-on-Clean patching (LSE): {sum_scores_clean_lse/N:.6f}")
        print(f"Mean Score for Corrupt-on-Clean support patching (Softmax): {sum_scores_corrupt_softmax/N:.6f}")
        print(f"Mean Score for Corrupt-on-Clean support patching (LSE): {sum_scores_corrupt_lse/N:.6f}")

        # Run full sweep
        all_pairs = build_all_stone_identity_pairs(dataset, input_word2idx, max_pairs=args.max_pairs)

        print(f"Successfully built {len(all_pairs)} pairs for the full experiment.")
        
        # Build epoch files list
        if args.epoch_range:
            try:
                start_ep, end_ep = map(int, args.epoch_range.split('-'))
            except ValueError:
                print("Invalid epoch_range format. Expected 'START-END' (e.g. '100-200').")
                return
                
            checkpoint_dir = os.path.dirname(args.checkpoint_path)
            import glob, re
            pt_files = glob.glob(os.path.join(checkpoint_dir, "best_model_epoch_*_classification_xsmall.pt"))
            epoch_files = []
            for f in pt_files:
                m = re.search(r'best_model_epoch_(\d+)_', os.path.basename(f))
                if m:
                    ep = int(m.group(1))
                    if start_ep <= ep <= end_ep:
                        epoch_files.append((ep, f))
            epoch_files.sort()
            if not epoch_files:
                print(f"No checkpoint files found in {checkpoint_dir} matching epoch range {args.epoch_range}.")
                return
        else:
            import re
            m = re.search(r'best_model_epoch_(\d+)_', os.path.basename(args.checkpoint_path))
            ep = int(m.group(1)) if m else 0
            epoch_files = [(ep, args.checkpoint_path)]
            
        print(f"\nFound {len(epoch_files)} checkpoints to process.")
        
        all_epochs_results = {}
        # Local storage logic
        rel_match = re.search(r'saved_models/(.*?init_seed_\d+)', args.checkpoint_path)
        if rel_match:
            suffix = rel_match.group(1)
        else:
            match = re.search(r'(.*?init_seed_\d+)', args.checkpoint_path)
            if match:
                suffix = os.path.basename(match.group(1))
            else:
                suffix = "default_run"
                
        base_save_dir = f"/home/rsaha/projects/aip-afyshe/rsaha/dm_alchemy/mech_interp_results/{suffix}"
        os.makedirs(base_save_dir, exist_ok=True)
        
        out_pkl_path = os.path.join(base_save_dir, f"layer_head_sweep_results_{args.setup}.pkl")
        if args.epoch_range:
            out_pkl_path = os.path.join(base_save_dir, f"layer_head_sweep_results_epochs_{args.epoch_range}_{args.setup}.pkl")
            
        for epoch, cp_path in epoch_files:
            print(f"\n=== Processing Epoch {epoch} ===")
            if len(epoch_files) > 1:
                checkpoint = torch.load(cp_path, map_location='cpu', weights_only=False)
                state_dict = checkpoint.get('model_state_dict', checkpoint)
                model.load_state_dict(state_dict)
                model.eval()
                
            sweep_results = run_full_sweep(model, all_pairs, input_word2idx, setup=args.setup, patch_mlp=args.patch_mlp, batch_size=args.batch_size)
            all_epochs_results[epoch] = sweep_results
            
            with open(out_pkl_path, 'wb') as f:
                pickle.dump(all_epochs_results, f)
                
            print(f"Saved incremental sweep results for epoch {epoch} to {out_pkl_path}")
            
            # Log metrics to W&B
            metrics_to_log = {"epoch": epoch}
            for comp_name, comp_res in sweep_results.items():
                metrics_to_log[f"{comp_name}/softmax_mean"] = comp_res["softmax_mean"]
                metrics_to_log[f"{comp_name}/lse_mean"] = comp_res["lse_mean"]
            wandb.log(metrics_to_log, step=epoch)

if __name__ == '__main__':
    main()
