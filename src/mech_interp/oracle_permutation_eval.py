#!/usr/bin/env python3
"""
Oracle Permutation Evaluation for Potion Remapping.
This script loads a model trained on Pairing 0, permutes its potion token embeddings
using the target permutation, and evaluates the model zero-shot on both the fixed
and buggy target validation sets.
"""

import os
import sys
import argparse
import pickle
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from functools import partial
import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.models import StoneStateDecoderClassifier
from src.models.data_loaders import AlchemyDataset, collate_fn

# Define the pairing mappings exactly as in the generator
PAIRINGS = [
    ['RED', 'GREEN', 'ORANGE', 'YELLOW', 'PINK', 'CYAN'],  # 0
    ['RED', 'GREEN', 'ORANGE', 'PINK', 'YELLOW', 'CYAN'],  # 1
    ['RED', 'GREEN', 'ORANGE', 'CYAN', 'YELLOW', 'PINK'],  # 2
    ['RED', 'ORANGE', 'GREEN', 'YELLOW', 'PINK', 'CYAN'],  # 3
    ['RED', 'ORANGE', 'GREEN', 'PINK', 'YELLOW', 'CYAN'],  # 4
    ['RED', 'ORANGE', 'GREEN', 'CYAN', 'YELLOW', 'PINK'],  # 5
    ['RED', 'YELLOW', 'GREEN', 'ORANGE', 'PINK', 'CYAN'],  # 6
    ['RED', 'YELLOW', 'GREEN', 'PINK', 'ORANGE', 'CYAN'],  # 7
    ['RED', 'YELLOW', 'GREEN', 'CYAN', 'ORANGE', 'PINK'],  # 8
    ['RED', 'PINK', 'GREEN', 'ORANGE', 'YELLOW', 'CYAN'],  # 9
    ['RED', 'PINK', 'GREEN', 'YELLOW', 'ORANGE', 'CYAN'],  # 10
    ['RED', 'PINK', 'GREEN', 'CYAN', 'ORANGE', 'YELLOW'],  # 11
    ['RED', 'CYAN', 'GREEN', 'ORANGE', 'YELLOW', 'PINK'],  # 12
    ['RED', 'CYAN', 'GREEN', 'YELLOW', 'ORANGE', 'PINK'],  # 13
    ['RED', 'CYAN', 'GREEN', 'PINK', 'ORANGE', 'YELLOW'],  # 14
]

def load_preprocessed_dataset(data_dir, pairing_index):
    """Load the preprocessed dataset and vocab for the given pairing index."""
    import glob
    # Search for files matching any number of held-out edges
    pattern_data = os.path.join(
        data_dir, 
        f"compositional_chemistry_samples_167424_80_unique_stones_val_shop_1_qhop_1_single_held_out_color_*_edges_exp_seed_0_pairing_index_{pairing_index}_classification_filter_True_input_features_output_stone_states_data.pkl"
    )
    matching_data_files = glob.glob(pattern_data)
    
    # Fallback for D1 if not explicitly indexed
    if not matching_data_files and pairing_index == 0:
        pattern_data = os.path.join(
            data_dir, 
            "compositional_chemistry_samples_167424_80_unique_stones_val_shop_1_qhop_1_single_held_out_color_*_edges_exp_seed_0_classification_filter_True_input_features_output_stone_states_data.pkl"
        )
        matching_data_files = glob.glob(pattern_data)
        
    if not matching_data_files:
        raise FileNotFoundError(f"No matching preprocessed data file found for pairing index {pairing_index} in {data_dir}")
        
    # Sort by modification time descending to select the most recently created file
    matching_data_files.sort(key=os.path.getmtime, reverse=True)
    
    data_path = matching_data_files[0]
    vocab_path = data_path.replace("_data.pkl", "_vocab.pkl")
    
    print(f"Found files for pairing_index={pairing_index}: {[os.path.basename(f) for f in matching_data_files]}")
    print(f"Loading data from: {data_path}")
    with open(data_path, "rb") as f:
        data = pickle.load(f)
        
    print(f"Loading vocab from: {vocab_path}")
    with open(vocab_path, "rb") as f:
        vocab = pickle.load(f)
        
    return data, vocab

def evaluate_model(model, val_dataset, pad_token_id, batch_size, device):
    """Run evaluation and return classification accuracy."""
    model.eval()
    
    # Custom collate fn
    custom_collate = partial(
        collate_fn,
        pad_token_id=pad_token_id,
        eos_token_id=None,
        task_type="classification",
        model_architecture="decoder",
        sos_token_id=None,
        prediction_type=None,
        max_seq_len=1024,
        truncate=False
    )
    
    dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=custom_collate,
        num_workers=0
    )
    
    from tqdm import tqdm
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            encoder_input_ids = batch["encoder_input_ids"].to(device)
            target_class_ids = batch["target_class_id"].to(device)
            
            src_padding_mask = (encoder_input_ids == pad_token_id)
            
            output_logits = model(encoder_input_ids, src_padding_mask=src_padding_mask)
            preds = output_logits.argmax(dim=-1)
            
            correct += (preds == target_class_ids).sum().item()
            total += target_class_ids.size(0)
            
    accuracy = correct / total if total > 0 else 0.0
    return accuracy, correct, total

def main():
    parser = argparse.ArgumentParser(description="Evaluate permuted model zero-shot on remapped datasets.")
    parser.add_argument("--checkpoint_path", type=str, required=True,
                        help="Path to the D1 (pairing_index=0) checkpoint .pt file.")
    parser.add_argument("--target_index", type=int, default=6,
                        help="Target pairing index for fixed D2 evaluation (default: 6).")
    parser.add_argument("--buggy_index", type=int, default=1,
                        help="Target pairing index for buggy D2 evaluation (default: 1).")
    parser.add_argument("--data_dir", type=str,
                        default="src/data/chemistry_pickles/original_reward_potion_remap_preprocessed_data",
                        help="Directory containing preprocessed data/vocab pickles.")
    parser.add_argument("--batch_size", type=int, default=256,
                        help="Batch size for evaluation (default: 256).")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use for evaluation (cuda or cpu).")
    
    args = parser.parse_args()
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # 1. Load checkpoints
    print(f"Loading checkpoint from: {args.checkpoint_path}")
    checkpoint = torch.load(args.checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    checkpoint_args = checkpoint.get("args")
    
    # Get vocabulary info and mapping from vocab
    _, vocab0 = load_preprocessed_dataset(args.data_dir, 0)
    input_word2idx = vocab0["input_word2idx"]
    pad_token_id = vocab0["pad_token_id"]
    
    detected_max_len = state_dict.get("positional_encoding.pe", torch.zeros(5000)).shape[0]
    detected_num_classes = state_dict.get("classification_head.weight", torch.zeros(80)).shape[0]
    detected_src_vocab_size = state_dict.get("src_tok_emb.weight", torch.zeros(len(input_word2idx))).shape[0]
    
    model_config = {
        "num_decoder_layers": 4,
        "emb_size": 256,
        "nhead": 4,
        "dim_feedforward": 512,
        "dropout": 0.1,
        "src_vocab_size": detected_src_vocab_size,
        "num_classes": detected_num_classes,
        "use_flash_attention": True if args.device == "cuda" else False,
        "max_len": detected_max_len,
        "vocab": input_word2idx
    }
    
    # Instantiate and load model
    model = StoneStateDecoderClassifier(**model_config)
    model.load_state_dict(state_dict)
    model = model.to(device)
    
    # Load D1 validation set
    d1_data_pkl, _ = load_preprocessed_dataset(args.data_dir, 0)
    
    # We construct AlchemyDataset wrapping the loaded pickle data
    # Note: AlchemyDataset expects json_file_path, but if we pass use_preprocessed=True,
    # we can bypass it by creating a dummy subclass or mock dataset.
    # To keep it extremely simple and avoid parsing logic, let's create a custom Dataset wrapper:
    class PreprocessedPklDataset(Dataset):
        def __init__(self, data_list, vocab):
            self.data = data_list
            self.input_word2idx = vocab["input_word2idx"]
            self.pad_token_id = vocab["pad_token_id"]
            self.item_sep_token_id = vocab["item_sep_token_id"]
            
        def __len__(self):
            return len(self.data)
            
        def __getitem__(self, idx):
            item = self.data[idx].copy()
            if not isinstance(item["encoder_input_ids"], torch.Tensor):
                item["encoder_input_ids"] = torch.tensor(item["encoder_input_ids"], dtype=torch.long)
            return item
            
    d1_dataset = PreprocessedPklDataset(d1_data_pkl, vocab0)
    
    print("\n==================== ARM 1: D1 checkpoint on D1 val ====================")
    acc1, corr1, tot1 = evaluate_model(model, d1_dataset, pad_token_id, args.batch_size, device)
    print(f"D1 Accuracy: {acc1 * 100:.4f}% ({corr1}/{tot1})")
    
    # --- FIXED D2 (Pairing index target_index) ---
    print(f"\nLoading fixed D2 dataset (index {args.target_index})...")
    d2_fixed_data, vocab_fixed = load_preprocessed_dataset(args.data_dir, args.target_index)
    d2_fixed_dataset = PreprocessedPklDataset(d2_fixed_data, vocab_fixed)
    
    print("\n==================== ARM 2: D1 checkpoint (unpermuted) on FIXED D2 ====================")
    acc2, corr2, tot2 = evaluate_model(model, d2_fixed_dataset, pad_token_id, args.batch_size, device)
    print(f"Unpermuted on Fixed D2 Accuracy: {acc2 * 100:.4f}% ({corr2}/{tot2})")
    
    # --- PERMUTE EMBEDDINGS ---
    print(f"\nApplying permutation mapping 0 -> {args.target_index}...")
    L0 = PAIRINGS[0]
    Lk = PAIRINGS[args.target_index]
    
    # Clone embedding layer
    E = model.src_tok_emb.weight.data
    E_new = E.clone()
    for i, c in enumerate(L0):
        src_idx = input_word2idx[c]
        tgt_idx = input_word2idx[Lk[i]]
        E_new[tgt_idx] = E[src_idx]
    E.copy_(E_new)
    
    print("\n==================== ARM 3: D1 checkpoint (permuted) on FIXED D2 ====================")
    acc3, corr3, tot3 = evaluate_model(model, d2_fixed_dataset, pad_token_id, args.batch_size, device)
    print(f"Permuted on Fixed D2 Accuracy: {acc3 * 100:.4f}% ({corr3}/{tot3})")
    
    # --- BUGGY D2 (Pairing index buggy_index) ---
    # We reload the clean model to test against buggy dataset
    model_buggy = StoneStateDecoderClassifier(**model_config)
    model_buggy.load_state_dict(state_dict)
    model_buggy = model_buggy.to(device)
    
    print(f"\nLoading buggy D2 dataset (index {args.buggy_index})...")
    d2_buggy_data, vocab_buggy = load_preprocessed_dataset(args.data_dir, args.buggy_index)
    d2_buggy_dataset = PreprocessedPklDataset(d2_buggy_data, vocab_buggy)
    
    print("\n==================== CONTROL: D1 checkpoint (unpermuted) on BUGGY D2 ====================")
    acc_buggy_unprem, corr_buggy_unprem, tot_buggy_unprem = evaluate_model(model_buggy, d2_buggy_dataset, pad_token_id, args.batch_size, device)
    print(f"Unpermuted on Buggy D2 Accuracy: {acc_buggy_unprem * 100:.4f}% ({corr_buggy_unprem}/{tot_buggy_unprem})")
    
    # Apply permutation for index buggy_index
    print(f"Applying permutation mapping 0 -> {args.buggy_index} to test on Buggy D2...")
    Lk_buggy = PAIRINGS[args.buggy_index]
    E_buggy = model_buggy.src_tok_emb.weight.data
    E_new_buggy = E_buggy.clone()
    for i, c in enumerate(L0):
        src_idx = input_word2idx[c]
        tgt_idx = input_word2idx[Lk_buggy[i]]
        E_new_buggy[tgt_idx] = E_buggy[src_idx]
    E_buggy.copy_(E_new_buggy)
    
    print("\n==================== CONTROL: D1 checkpoint (permuted) on BUGGY D2 ====================")
    acc_buggy_prem, corr_buggy_prem, tot_buggy_prem = evaluate_model(model_buggy, d2_buggy_dataset, pad_token_id, args.batch_size, device)
    print(f"Permuted on Buggy D2 Accuracy: {acc_buggy_prem * 100:.4f}% ({corr_buggy_prem}/{tot_buggy_prem})")
    
    # --- LOG RESULTS TO CSV ---
    csv_dir = "csv_files"
    os.makedirs(csv_dir, exist_ok=True)
    csv_path = os.path.join(csv_dir, "oracle_permutation_eval.csv")
    
    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a" if file_exists else "w") as f:
        if not file_exists:
            f.write("checkpoint,target_index,arm1_d1_acc,arm2_unperm_fixed_acc,arm3_perm_fixed_acc,control_unperm_buggy_acc,control_perm_buggy_acc\n")
        f.write(f"{os.path.basename(args.checkpoint_path)},{args.target_index},{acc1:.6f},{acc2:.6f},{acc3:.6f},{acc_buggy_unprem:.6f},{acc_buggy_prem:.6f}\n")
    
    print(f"\nResults appended to {csv_path}")

if __name__ == "__main__":
    main()
