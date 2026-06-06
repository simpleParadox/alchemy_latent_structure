import os
import pickle
import argparse
import torch
import torch.nn as nn
from src.models.models import StoneStateDecoderClassifier

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

def build_chemistry_identity_pairs(dataset, chemistry_id_1, chemistry_id_2, input_word2idx=None):
    """
    Returns a list of (clean_input, corrupt_input, target_class_id) tuples.
    For each query item from chemistry_id_1, constructs a corrupt input identical in structure
    but with support examples drawn from chemistry_id_2 instead.
    
    Args:
        dataset: List of dicts (loaded from dataset pickle) or AlchemyDataset object.
        chemistry_id_1: Integer index in sorted unique support prefixes, or the support prefix tuple of token IDs.
        chemistry_id_2: Integer index in sorted unique support prefixes, or the support prefix tuple of token IDs.
        input_word2idx: Dictionary mapping token names to indices. If None, hardcoded vocabulary is used.
        
    Returns:
        list of tuples: [(clean_tensor, corrupt_tensor, target_class_id), ...] where each tensor has shape [1, seq_len].
    """
    # Hardcoded known vocabulary as a fallback
    hardcoded_vocab = {
        '-1': 0, '-3': 1, '1': 2, '3': 3, 'CYAN': 4, 'GREEN': 5, 'ORANGE': 6, 'PINK': 7, 'RED': 8, 'YELLOW': 9,
        'blue': 10, 'large': 11, 'medium': 12, 'medium_round': 13, 'pointy': 14, 'purple': 15, 'red': 16,
        'round': 17, 'small': 18, '<pad>': 19, '<sos>': 20, '<eos>': 21, '<io>': 22, '<item_sep>': 23, '<unk>': 24
    }
    
    vocab = input_word2idx if input_word2idx is not None else hardcoded_vocab
    item_sep_id = vocab.get('<item_sep>', 23)
    
    # Extract actual data list
    if hasattr(dataset, 'data'):
        samples = dataset.data
    else:
        samples = dataset
        
    # Extract all unique support prefixes and sort them deterministically to assign integer IDs
    support_keys = []
    seen = set()
    for sample in samples:
        enc_inps = sample['encoder_input_ids']
        support, _ = split_support_query(enc_inps, item_sep_id)
        support_key = tuple(support)
        if support_key not in seen:
            seen.add(support_key)
            support_keys.append(support_key)
            
    # Deterministic sorting of support keys (tuples of ints)
    sorted_support_keys = sorted(support_keys)
    
    # Helper to map chemistry_id (int or tuple) to the actual support key tuple
    def resolve_chemistry_id(chem_id, name):
        if isinstance(chem_id, int):
            if chem_id < 0 or chem_id >= len(sorted_support_keys):
                raise ValueError(f"{name} index {chem_id} out of range (total unique support keys: {len(sorted_support_keys)})")
            return sorted_support_keys[chem_id]
        else:
            key_tuple = tuple(chem_id)
            if key_tuple not in seen:
                raise ValueError(f"{name} tuple not found in dataset support keys.")
            return key_tuple

    sk1 = resolve_chemistry_id(chemistry_id_1, "chemistry_id_1")
    sk2 = resolve_chemistry_id(chemistry_id_2, "chemistry_id_2")
    
    pairs = []
    for sample in samples:
        enc_inps = sample['encoder_input_ids']
        support, query = split_support_query(enc_inps, item_sep_id)
        target_class_id = sample.get('target_class_id', None)
        
        # If the sample matches the clean chemistry (chemistry_id_1)
        if tuple(support) == sk1:
            clean_input = enc_inps
            corrupt_input = list(sk2) + list(query)
            
            clean_tensor = torch.tensor([clean_input], dtype=torch.long)
            corrupt_tensor = torch.tensor([corrupt_input], dtype=torch.long)
            pairs.append((clean_tensor, corrupt_tensor, target_class_id))
            
    return pairs

def decode_sequence(token_ids, idx2word):
    """
    Decodes a tensor or list of token IDs into a space-separated string of token names.
    """
    if torch.is_tensor(token_ids):
        token_ids = token_ids.cpu().numpy()
    import numpy as np
    flat_ids = np.array(token_ids).flatten()
    return " ".join([idx2word.get(int(tid), f"<unk_{tid}>") for tid in flat_ids])

def compute_patching_score(model, clean_input, corrupt_input, component_name, target_token_position, y_clean=None):
    """
    Computes the logit-difference score under activation patching.
    
    Returns:
        float: (f_clean(y_clean) - f_patched(y_clean)) / (f_clean(y_clean) - f_corrupt(y_clean))
    """
    from activation_cache import ActivationCacheManager
    
    # 1. Clean run forward pass and cache activations
    with ActivationCacheManager(model) as clean_cache:
        f_clean = model(clean_input)
        clean_acts = clean_cache.get_activations()
        
    # If y_clean is not provided, default to the class that the model predicted on clean input
    if y_clean is None:
        print("No y_clean provided, using model's prediction on clean input. Prediction: ", f_clean.argmax(dim=-1).item())
        y_clean = f_clean.argmax(dim=-1).item()
        
    # 2. Corrupt run forward pass and cache activations
    with ActivationCacheManager(model) as corrupt_cache:
        f_corrupt = model(corrupt_input)
        corrupt_acts = corrupt_cache.get_activations()
        
    f_clean_y = f_clean[0, y_clean].item()
    f_corrupt_y = f_corrupt[0, y_clean].item()
    
    # If the clean and corrupt outputs are virtually identical for y_clean, logit difference is zero
    denom = f_clean_y - f_corrupt_y
    import pdb; pdb.set_trace()
    if abs(denom) < 1e-7:
        return 0.0
        
    # 3. Find the target module to register hook on
    if component_name == 'embedding':
        target_module = model.src_tok_emb
    elif component_name.startswith('layer_') and component_name.endswith('_output'):
        try:
            layer_idx = int(component_name.split('_')[1])
            target_module = model.transformer_encoder.layers[layer_idx]
        except (ValueError, IndexError) as e:
            raise ValueError(f"Invalid layer index in component_name '{component_name}': {e}")
    else:
        raise ValueError(f"Unknown component_name '{component_name}'")
        
    corrupt_act = corrupt_acts.get(component_name)
    if corrupt_act is None:
        raise ValueError(f"Component '{component_name}' not found in corrupt cached activations. Keys are: {list(corrupt_acts.keys())}")
        
    # Define patching hook closure
    def make_patch_hook(c_act, t_pos):
        def hook(module, inp, outp):
            is_tuple = isinstance(outp, tuple)
            out_tensor = outp[0] if is_tuple else outp
            patched = out_tensor.clone()
            
            batch_first = getattr(module, 'batch_first', True)
            if batch_first:
                if t_pos == 'all' or t_pos is None:
                    patched[:, :, :] = c_act[:, :, :]
                elif isinstance(t_pos, (list, tuple, range)):
                    for pos in t_pos:
                        patched[:, pos, :] = c_act[:, pos, :]
                else:
                    patched[:, t_pos, :] = c_act[:, t_pos, :]
            else:
                # batch_first = False
                c_act_perm = c_act.transpose(0, 1)
                if t_pos == 'all' or t_pos is None:
                    patched[:, :, :] = c_act_perm[:, :, :]
                elif isinstance(t_pos, (list, tuple, range)):
                    for pos in t_pos:
                        patched[pos, :, :] = c_act_perm[pos, :, :]
                else:
                    patched[t_pos, :, :] = c_act_perm[t_pos, :, :]
                    
            if is_tuple:
                return (patched,) + outp[1:]
            return patched
        return hook
        
    # Register the patching hook
    handle = target_module.register_forward_hook(make_patch_hook(corrupt_act, target_token_position))
    
    try:
        # Run third forward pass (patched clean run)
        f_patched = model(clean_input)
    except Exception as e:
        raise Exception(f"Errored out forward pass for the third run, removing hook. Error: {e}")
    finally:
        # Ensure hook is removed even if forward pass errors out
        handle.remove()
        
    f_patched_y = f_patched[0, y_clean].item()
    print(f"f_clean_y: {f_clean_y}, f_corrupt_y: {f_corrupt_y}, f_patched_y: {f_patched_y}")
    import pdb; pdb.set_trace()
    

    
    score = (f_clean_y - f_patched_y) / denom
    return score

def main():
    parser = argparse.ArgumentParser(description="Chemistry Identity Pairs and Activation Patching Experiments")
    parser.add_argument("--checkpoint_path", type=str, 
                        default='/home/rsaha/projects/aip-afyshe/rsaha/dm_alchemy/src/saved_models/held_out_color_exp/held_out_edges_4/all_graphs/scheduler_cosine/wd_0.01_lr_0.0001/eta_min_7e-05/xsmall/decoder/classification/input_features/output_stone_states/shop_1_qhop_1/seed_0/init_seed_3/best_model_epoch_100_classification_xsmall.pt', 
                        help="Path to model checkpoint .pt file")
    parser.add_argument("--val_data", type=str, 
                        default='src/data/shuffled_held_out_exps_preprocessed_separate_enhanced/compositional_chemistry_samples_167424_80_unique_stones_val_shop_1_qhop_1_single_held_out_color_4_edges_exp_seed_0_classification_filter_True_input_features_output_stone_states_data.pkl', 
                        help="Path to validation data pickle")
    parser.add_argument("--calibration_exp", action="store_true", default=True,
                        help="Run the embedding-level calibration sanity check (defaults to True for verification)")
    
    args = parser.parse_args()
    
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
    print("\nBuilding chemistry identity pairs for chemistry_id_1=0 and chemistry_id_2=1...")
    pairs = build_chemistry_identity_pairs(dataset, 0, 1, input_word2idx)
    print(f"Successfully generated {len(pairs)} pairs.")
    
    num_to_print = min(3, len(pairs))
    print(f"\n--- Printing {num_to_print} sample pairs for manual inspection ---")
    for idx in range(num_to_print):
        clean_tensor, corrupt_tensor, target_class_id = pairs[idx]
        print(f"\nPair {idx + 1} (Target Class: {target_class_id}):")
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
            # "vocab": input_word2idx,
            "use_flash_attention": True,
            "max_len": detected_max_len
        }
        model = StoneStateDecoderClassifier(**model_config)
        model.load_state_dict(state_dict)
        model.eval()
        
        print("\n--- Running Calibration Experiment ---")
        clean_input, corrupt_input, target_class_id = pairs[0]
        
        # Clean-on-clean calibration check
        # We hook 'embedding' and patch with clean activations (which are clean_cache['embedding'])
        from activation_cache import ActivationCacheManager
        with ActivationCacheManager(model) as clean_cache:
            _ = model(clean_input)
            clean_acts = clean_cache.get_activations()
            
        # Register clean-on-clean hook manually to verify compute_patching_score
        score_clean_on_clean = compute_patching_score(
            model=model,
            clean_input=clean_input,
            corrupt_input=clean_input, # clean-on-clean
            component_name='embedding',
            target_token_position='all',
            y_clean=target_class_id
        )
        print(f"Score for Clean-on-Clean patching (should be 0.0): {score_clean_on_clean:.6f}")
        
        # Corrupt-on-clean calibration check on all support tokens (0 to 175)
        # Note: clean and corrupt differ only in support examples, so patching all support tokens
        # should copy the entire difference, resulting in a score very close to 1.0.
        support_token_positions = list(range(176))
        score_corrupt_on_clean = compute_patching_score(
            model=model,
            clean_input=clean_input,
            corrupt_input=corrupt_input,
            component_name='embedding',
            target_token_position=support_token_positions,
            y_clean=target_class_id
        )
        print(f"Score for Corrupt-on-Clean support patching (should be ~1.0): {score_corrupt_on_clean:.6f}")

if __name__ == '__main__':
    main()
