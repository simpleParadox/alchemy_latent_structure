#!/usr/bin/env python3
"""
Baseline B2/B3 evals for the half-edge potion pairing experiment (CL_experiment_handoff.md, sec 5).

B2: D1 checkpoint (unmodified) evaluated zero-shot on D2 val (pairing_index=6).
    Expected: well below ceiling. This is the headline number -- confirms the pairing
    is actually stored in weights (not solvable by in-context elimination).
B3: D1 checkpoint with potion embedding rows swapped by SIGMA=(GREEN YELLOW),
    evaluated on D2 val. Expected: ~ceiling (within noise of B1), NOT exactly 100% --
    the invariance confirmed in Step 0 holds over the merged train+val multiset, not
    per-split, so this is a sanity check, not a bit-exact identity.

Runs B2+B3 for all three D1 seeds' winning checkpoints (see
csv_files/half_edge_held_out_D1_winning_checkpoints.csv), applying the embedding swap
independently to each seed's own checkpoint.

Per-potion breakdown (GREEN vs YELLOW query) uses the fact that the query potion is
always the last token of encoder_input_ids in this dataset (confirmed empirically:
532/532 GREEN/YELLOW split on D1 val with no other token appearing there).
"""
import argparse
import csv
import os
import pickle
from functools import partial

import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.models import StoneStateDecoderClassifier
from src.models.data_loaders import collate_fn

SIGMA = {"GREEN": "YELLOW", "YELLOW": "GREEN"}

PREPROCESSED_DIR = "src/data/chemistry_pickles/half_edge_held_out_preprocessed_data"
FILENAME_TEMPLATE = (
    "compositional_chemistry_samples_167424_80_unique_stones_val_shop_1_qhop_1_"
    "half_edge_held_out_GREEN_YELLOW_exp_seed_0_pairing_index_{idx}_"
    "classification_filter_True_input_features_output_stone_states_{suffix}.pkl"
)

# (seed, weight_decay, eta_min) winning checkpoints, from the earlier ceiling analysis.
WINNING_CHECKPOINTS = {
    1: {"wd": 0.01, "eta_min": "9.5e-05"},
    3: {"wd": 0.1, "eta_min": "1e-05"},
    42: {"wd": 0.1, "eta_min": "7e-05"},
}

CHECKPOINT_TEMPLATE = (
    "src/saved_models/continual/half_edge_held_out/all_graphs/scheduler_cosine/"
    "wd_{wd}_lr_0.0001/eta_min_{eta_min}/xsmall/decoder/classification/input_features/"
    "output_stone_states/continual_seq_0/seed_0/init_seed_{seed}/"
    "model_cycle_1_task_0_pairing_index_0.pt"
)


def load_preprocessed(pairing_index):
    data_path = os.path.join(PREPROCESSED_DIR, FILENAME_TEMPLATE.format(idx=pairing_index, suffix="data"))
    vocab_path = os.path.join(PREPROCESSED_DIR, FILENAME_TEMPLATE.format(idx=pairing_index, suffix="vocab"))
    with open(data_path, "rb") as f:
        data = pickle.load(f)
    with open(vocab_path, "rb") as f:
        vocab = pickle.load(f)
    return data, vocab


class PreprocessedPklDataset(Dataset):
    def __init__(self, data_list):
        self.data = data_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx].copy()
        if not isinstance(item["encoder_input_ids"], torch.Tensor):
            item["encoder_input_ids"] = torch.tensor(item["encoder_input_ids"], dtype=torch.long)
        return item


def evaluate_with_potion_breakdown(model, dataset, pad_token_id, idx2word, batch_size, device):
    """Returns (overall_acc, per_potion_acc_dict, correct, total)."""
    model.eval()
    custom_collate = partial(
        collate_fn,
        pad_token_id=pad_token_id,
        eos_token_id=None,
        task_type="classification",
        model_architecture="decoder",
        sos_token_id=None,
        prediction_type=None,
        max_seq_len=1024,
        truncate=False,
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate, num_workers=0)

    correct = 0
    total = 0
    per_potion_correct = {}
    per_potion_total = {}

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            encoder_input_ids = batch["encoder_input_ids"].to(device)
            target_class_ids = batch["target_class_id"].to(device)
            src_padding_mask = (encoder_input_ids == pad_token_id)

            output_logits = model(encoder_input_ids, src_padding_mask=src_padding_mask)
            preds = output_logits.argmax(dim=-1)
            is_correct = (preds == target_class_ids)

            correct += is_correct.sum().item()
            total += target_class_ids.size(0)

            # Query potion = last non-pad token of each (unpadded) row.
            for row_ids, row_correct in zip(encoder_input_ids.cpu(), is_correct.cpu()):
                non_pad = row_ids[row_ids != pad_token_id]
                potion = idx2word[non_pad[-1].item()]
                per_potion_total[potion] = per_potion_total.get(potion, 0) + 1
                per_potion_correct[potion] = per_potion_correct.get(potion, 0) + int(row_correct.item())

    overall_acc = correct / total if total > 0 else 0.0
    per_potion_acc = {
        p: (per_potion_correct[p] / per_potion_total[p] if per_potion_total[p] > 0 else 0.0)
        for p in per_potion_total
    }
    return overall_acc, per_potion_acc, correct, total, per_potion_total


def build_model(state_dict, input_word2idx, device):
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
        "use_flash_attention": device.type == "cuda",
        "max_len": detected_max_len,
        "vocab": input_word2idx,
    }
    model = StoneStateDecoderClassifier(**model_config)
    model.load_state_dict(state_dict)
    return model.to(device)


def apply_sigma_swap(model, input_word2idx):
    E = model.src_tok_emb.weight.data
    E_new = E.clone()
    for src, tgt in SIGMA.items():
        E_new[input_word2idx[tgt]] = E[input_word2idx[src]]
    E.copy_(E_new)


def main():
    parser = argparse.ArgumentParser(description="B2/B3 baseline evals for the half-edge potion pairing experiment.")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    device = torch.device(args.device)
    print(f"Using device: {device}")

    d1_data, vocab0 = load_preprocessed(0)
    d2_data, vocab6 = load_preprocessed(6)

    assert vocab0["input_word2idx"] == vocab6["input_word2idx"], "Vocab mismatch between pairing_index 0 and 6!"
    assert vocab0["pad_token_id"] == vocab6["pad_token_id"], "pad_token_id mismatch between pairing_index 0 and 6!"
    print("Vocab assert passed: pairing_index 0 and 6 share the same vocab.")

    input_word2idx = vocab0["input_word2idx"]
    input_idx2word = vocab0["input_idx2word"]
    pad_token_id = vocab0["pad_token_id"]

    d2_dataset = PreprocessedPklDataset(d2_data)

    rows = []
    for seed, hp in WINNING_CHECKPOINTS.items():
        ckpt_path = CHECKPOINT_TEMPLATE.format(wd=hp["wd"], eta_min=hp["eta_min"], seed=seed)
        print(f"\n{'='*70}\nseed={seed}  wd={hp['wd']}  eta_min={hp['eta_min']}\n{ckpt_path}\n{'='*70}")
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state_dict = checkpoint.get("model_state_dict", checkpoint)

        # --- B2: unmodified D1 checkpoint on D2 val ---
        model = build_model(state_dict, input_word2idx, device)
        b2_acc, b2_per_potion, b2_corr, b2_tot, b2_counts = evaluate_with_potion_breakdown(
            model, d2_dataset, pad_token_id, input_idx2word, args.batch_size, device
        )
        print(f"B2 (unmodified, D2 val): {b2_acc*100:.4f}% ({b2_corr}/{b2_tot})")
        for p, acc in b2_per_potion.items():
            print(f"  B2 potion={p}: {acc*100:.4f}% (n={b2_counts[p]})")

        # --- B3: SIGMA-permuted D1 checkpoint on D2 val ---
        apply_sigma_swap(model, input_word2idx)
        b3_acc, b3_per_potion, b3_corr, b3_tot, b3_counts = evaluate_with_potion_breakdown(
            model, d2_dataset, pad_token_id, input_idx2word, args.batch_size, device
        )
        print(f"B3 (SIGMA-permuted, D2 val): {b3_acc*100:.4f}% ({b3_corr}/{b3_tot})")
        for p, acc in b3_per_potion.items():
            print(f"  B3 potion={p}: {acc*100:.4f}% (n={b3_counts[p]})")

        rows.append({
            "seed": seed, "weight_decay": hp["wd"], "eta_min": hp["eta_min"],
            "B2_overall": b2_acc, "B2_GREEN": b2_per_potion.get("GREEN"), "B2_YELLOW": b2_per_potion.get("YELLOW"),
            "B3_overall": b3_acc, "B3_GREEN": b3_per_potion.get("GREEN"), "B3_YELLOW": b3_per_potion.get("YELLOW"),
        })

    os.makedirs("csv_files", exist_ok=True)
    csv_path = "csv_files/half_edge_held_out_B2_B3_baselines.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved results to {csv_path}")


if __name__ == "__main__":
    main()
