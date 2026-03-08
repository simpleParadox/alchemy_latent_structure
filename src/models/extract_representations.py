"""
extract_representations.py — Extract per-layer hidden representations from
trained checkpoints for downstream analyses (CKA, linear probes, etc.).

For each checkpoint (epoch), this script:
  1. Loads the model weights.
  2. Runs a forward pass over the validation data.
  3. Hooks into each Transformer layer to capture the last-token representation
     (the same representation fed to the classification head).
  4. Also computes a binary "in-support" label per example:
       1 if target_class_id ∈ support_stone_ids, else 0.
  5. Saves a dict per checkpoint:
       {
         "epoch": int,
         "layer_representations": {
             "embedding": np.ndarray (N, d_model),
             "layer_0":   np.ndarray (N, d_model),
             "layer_1":   np.ndarray (N, d_model),
             "layer_2":   np.ndarray (N, d_model),
             "layer_3":   np.ndarray (N, d_model),
         },
         "in_support_labels":  np.ndarray (N,) binary,
         "target_class_ids":   np.ndarray (N,),
         "predicted_class_ids": np.ndarray (N,),
       }

Usage:
    python src/models/extract_representations.py \
        --checkpoint_dir /path/to/checkpoints \
        --val_data_path src/data/.../val_shop_1_qhop_3.json \
        --output_dir representations_output/ \
        --epoch_stride 50
"""

import argparse
import gc
import glob
import os
import re
import math

import numpy as np
import torch
import torch.nn as nn
from collections import defaultdict
from functools import partial
from torch.utils.data import DataLoader
from tqdm import tqdm

from cluster_profile import cluster
from data_loaders import AlchemyDataset, collate_fn
from models import (
    create_decoder_classifier_model,
)
from train import set_seed, worker_init_fn


# ─────────────────────────────────────────────────────────────────────────────
#  Argument parsing
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract per-layer representations from trained checkpoints."
    )

    # ── Checkpoint ───────────────────────────────────────────────────────
    parser.add_argument("--checkpoint_dir", type=str, required=True,
                        help="Directory containing checkpoint .pt files.")
    parser.add_argument("--epoch_stride", type=int, default=50,
                        help="Only process every N-th epoch (to save disk/time). "
                             "Set to 1 to process all.")
    parser.add_argument("--specific_epochs", type=str, default=None,
                        help="Comma-separated list of specific epochs to extract "
                             "(overrides epoch_stride). E.g. '0,50,100,200,500,999'")

    # ── Data ─────────────────────────────────────────────────────────────
    parser.add_argument("--val_data_path", type=str, required=True,
                        help="Path to the validation JSON data file.")
    parser.add_argument("--data_split_seed", type=int, default=0)

    # ── Model (must match the training config) ───────────────────────────
    parser.add_argument("--model_size", type=str, default="xsmall")
    parser.add_argument("--model_architecture", type=str, default="decoder")
    parser.add_argument("--max_seq_len", type=int, default=2048)
    parser.add_argument("--task_type", type=str, default="classification")
    parser.add_argument("--input_format", type=str, default="features")
    parser.add_argument("--output_format", type=str, default="stone_states")
    parser.add_argument("--prediction_type", type=str, default="default")
    parser.add_argument("--override_num_classes", type=int, default=108)
    parser.add_argument("--padding_side", type=str, default="right")
    parser.add_argument("--use_flash_attention", type=str, default="True")
    parser.add_argument("--filter_query_from_support", type=str, default="True")
    parser.add_argument("--use_truncation", type=str, default="False")

    # ── Runtime ──────────────────────────────────────────────────────────
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)

    # ── Data pre-processing ──────────────────────────────────────────────
    parser.add_argument("--preprocessed_dir", type=str,
                        default="src/data/complete_graph_composition_fully_shuffled_balanced_grouped_by_unique_end_state_preprocessed")
    parser.add_argument("--use_preprocessed", type=str, default="True")

    # ── Output ───────────────────────────────────────────────────────────
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Where to save extracted representations. "
                             "Defaults to <checkpoint_dir>/representations/")

    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────

def resolve_base_path():
    if cluster == "vulcan":
        return "/home/rsaha/projects/def-afyshe-ab/rsaha/dm_alchemy/"
    elif cluster == "cc":
        return "/home/rsaha/projects/def-afyshe-ab/rsaha/projects/dm_alchemy/"
    elif cluster == "cirrus":
        return "/home/rsaha/projects/dm_alchemy/"
    elif cluster == "rorqual":
        return "/home/rsaha/links/projects/def-afyshe-ab/rsaha/projects/dm_alchemy/"
    elif cluster == "killarney":
        return "/home/rsaha/projects/aip-afyshe/rsaha/dm_alchemy/"
    else:
        raise RuntimeError(f"Unknown cluster: {cluster}")


def discover_checkpoints(checkpoint_dir, task_type, model_size):
    pattern = os.path.join(
        checkpoint_dir,
        f"best_model_epoch_*_{task_type}_{model_size}.pt",
    )
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"No checkpoints matching: {pattern}")

    regex = re.compile(r"best_model_epoch_(\d+)_")
    pairs = []
    for f in files:
        m = regex.search(os.path.basename(f))
        if m:
            pairs.append((int(m.group(1)), f))
    pairs.sort(key=lambda x: x[0])
    return pairs


def extract_support_stone_ids(encoder_input_ids, input_vocab_idx2word, stone_state_to_id, hop):
    """
    Parse support segment of encoder_input_ids (features format) to find
    the set of stone class IDs that appear in the support set.

    The query portion occupies the last (hop + 4) tokens:
        4 feature tokens + hop potion tokens.
    For shop_1, query = 4 feature tokens + 1 potion = 5 tokens.

    More robustly: support is everything before the last <item_sep>.
    """
    # Find the position of the last <item_sep> token — everything before it
    # is support, everything after is the query input.
    item_sep_tok = "<item_sep>"
    io_tok = "<io>"

    # Build reverse vocab
    idx2word = input_vocab_idx2word if isinstance(input_vocab_idx2word, dict) else {v: k for k, v in input_vocab_idx2word.items()}
    
    # Convert encoder_input_ids 

    tokens = [idx2word.get(t.item(), 'unk') for t in encoder_input_ids]

    assert '<unk>' not in tokens, "Found unk token in encoder_input_ids — check vocab mapping!"

    # Find last <item_sep> — that separates support from query
    last_sep_idx = -1
    for i in range(len(tokens) - 1, -1, -1):
        if tokens[i] == item_sep_tok:
            last_sep_idx = i
            break

    if last_sep_idx == -1:
        # No separator found — can't extract support
        return set()

    support_tokens = tokens[:last_sep_idx]

    # Parse support transitions: pattern is
    #   feat1 feat2 feat3 feat4 potion <io> feat1 feat2 feat3 feat4 <item_sep> ...
    stone_ids = set()
    i = 0
    while i + 4 <= len(support_tokens):
        # Look for a group of 4 feature tokens followed by something
        # Try to find <io> separator
        if i + 5 < len(support_tokens) and support_tokens[i + 5] == io_tok:
            # This is: feat1..feat4 potion <io> ...
            in_feat = support_tokens[i:i + 4]
            in_state_str = (
                f"{{color: {in_feat[0]}, size: {in_feat[1]}, "
                f"roundness: {in_feat[2]}, reward: {in_feat[3]}}}"
            )
            in_id = stone_state_to_id.get(in_state_str)
            if in_id is not None:
                stone_ids.add(in_id)

            # Output features are at i+6..i+9
            if i + 10 <= len(support_tokens):
                out_feat = support_tokens[i + 6:i + 10]
                out_state_str = (
                    f"{{color: {out_feat[0]}, size: {out_feat[1]}, "
                    f"roundness: {out_feat[2]}, reward: {out_feat[3]}}}"
                )
                out_id = stone_state_to_id.get(out_state_str)
                if out_id is not None:
                    stone_ids.add(out_id)

            # Advance past this transition (4+1+1+4 = 10) + possible <item_sep>
            i += 10
            if i < len(support_tokens) and support_tokens[i] == item_sep_tok:
                i += 1
        elif i + 4 < len(support_tokens) and support_tokens[i + 4] == io_tok:
            # This is: feat1..feat4 <io> feat1..feat4 (no potion — stone_states format?)
            # Handle the no-potion case
            in_feat = support_tokens[i:i + 4]
            in_state_str = (
                f"{{color: {in_feat[0]}, size: {in_feat[1]}, "
                f"roundness: {in_feat[2]}, reward: {in_feat[3]}}}"
            )
            in_id = stone_state_to_id.get(in_state_str)
            if in_id is not None:
                stone_ids.add(in_id)

            if i + 9 <= len(support_tokens):
                out_feat = support_tokens[i + 5:i + 9]
                out_state_str = (
                    f"{{color: {out_feat[0]}, size: {out_feat[1]}, "
                    f"roundness: {out_feat[2]}, reward: {out_feat[3]}}}"
                )
                out_id = stone_state_to_id.get(out_state_str)
                if out_id is not None:
                    stone_ids.add(out_id)
            i += 9
            if i < len(support_tokens) and support_tokens[i] == item_sep_tok:
                i += 1
        else:
            i += 1

    return stone_ids


# ─────────────────────────────────────────────────────────────────────────────
#  Hook-based representation extractor
# ─────────────────────────────────────────────────────────────────────────────

class RepresentationExtractor:
    """
    Attaches forward hooks to the embedding + each TransformerEncoderLayer
    to capture the output at every layer.

    For the decoder-classifier, we collect the *last valid token*
    representation at each layer (same position used for classification).
    """

    def __init__(self, model, padding_side="right"):
        self.model = model
        self.padding_side = padding_side
        self.hooks = []
        self.layer_outputs = {}  # layer_name → list of (batch_size, d_model)
        self._attach_hooks()

    def _attach_hooks(self):
        """Register forward hooks on the embedding + PE and each encoder layer."""
        # We need to unwrap if the model is wrapped by Accelerator/DDP
        m = self.model
        if hasattr(m, "module"):
            m = m.module

        # 1. After embedding + positional encoding
        #    The PE is applied inside forward(), so we hook the positional_encoding module.
        def embed_hook(module, input, output):
            # output shape: (seq_len, batch, d_model) — PE uses seq-first
            self.layer_outputs.setdefault("embedding", []).append(
                output.detach().permute(1, 0, 2)  # → (batch, seq, d_model)
            )

        h = m.positional_encoding.register_forward_hook(embed_hook)
        self.hooks.append(h)

        # 2. Each TransformerEncoderLayer
        for layer_idx, layer_module in enumerate(m.transformer_encoder.layers):
            name = f"layer_{layer_idx}"

            def layer_hook(module, input, output, _name=name):
                # output shape: (batch, seq, d_model) — batch_first=True
                self.layer_outputs.setdefault(_name, []).append(
                    output.detach()
                )

            h = layer_module.register_forward_hook(layer_hook)
            self.hooks.append(h)

    def clear(self):
        self.layer_outputs = {}

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []

    def collect_last_token_representations(self, src_padding_mask):
        """
        From the captured full-sequence outputs, extract the last valid
        token representation for each example in the batch.

        Parameters
        ----------
        src_padding_mask : (batch, seq_len) bool tensor, True = padding

        Returns
        -------
        dict[str, torch.Tensor]  — layer_name → (batch, d_model) on CPU
        """
        if src_padding_mask is not None:
            seq_lengths = (~src_padding_mask).sum(dim=1) - 1  # 0-indexed
            seq_lengths = torch.clamp(seq_lengths, min=0)
        else:
            seq_lengths = None

        result = {}
        for name, output_list in self.layer_outputs.items():
            # output_list has one entry per batch; concatenate if needed
            # (we call this once per batch, so there should be exactly 1)
            full_output = output_list[-1]  # (batch, seq, d_model)

            if seq_lengths is not None:
                batch_idx = torch.arange(full_output.size(0), device=full_output.device)
                last_tok = full_output[batch_idx, seq_lengths.to(full_output.device), :]
            else:
                last_tok = full_output[:, -1, :]

            result[name] = last_tok.cpu()

        return result


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # ── Boolean conversions ──────────────────────────────────────────────
    args.use_flash_attention = str(args.use_flash_attention).lower() == "true"
    args.use_truncation = str(args.use_truncation).lower() == "true"
    args.filter_query_from_support = str(args.filter_query_from_support).lower() == "true"
    args.use_preprocessed = str(args.use_preprocessed).lower() == "true"

    # ── Paths ────────────────────────────────────────────────────────────
    base_path = resolve_base_path()
    args.val_data_path = f"{args.val_data_path.split('.json')[0]}_seed_{args.data_split_seed}.json"
    args.val_data_path = os.path.join(base_path, args.val_data_path)
    args.preprocessed_dir = os.path.join(base_path, args.preprocessed_dir)

    if args.output_dir is None:
        args.output_dir = os.path.join(args.checkpoint_dir, "representations")
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Checkpoint dir : {args.checkpoint_dir}")
    print(f"Val data       : {args.val_data_path}")
    print(f"Output dir     : {args.output_dir}")

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Discover checkpoints ─────────────────────────────────────────────
    epoch_file_pairs = discover_checkpoints(
        args.checkpoint_dir, args.task_type, args.model_size
    )
    print(f"Found {len(epoch_file_pairs)} total checkpoints")

    # Filter to requested epochs
    if args.specific_epochs is not None:
        target_epochs = set(int(e) for e in args.specific_epochs.split(","))
        epoch_file_pairs = [(e, f) for e, f in epoch_file_pairs if e in target_epochs]
    else:
        epoch_file_pairs = [(e, f) for e, f in epoch_file_pairs
                            if e % args.epoch_stride == 0 or e == epoch_file_pairs[-1][0]]

    print(f"Will process {len(epoch_file_pairs)} checkpoints after filtering")

    # ── Load vocab from first checkpoint ─────────────────────────────────
    first_ckpt = torch.load(epoch_file_pairs[0][1], map_location="cpu", weights_only=False)
    vocab_word2idx = first_ckpt["src_vocab_word2idx"]
    vocab_idx2word = first_ckpt.get("src_vocab_idx2word", {v: k for k, v in vocab_word2idx.items()})
    stone_state_to_id = first_ckpt["stone_state_to_id"]
    src_vocab_size = len(vocab_word2idx)
    num_classes = args.override_num_classes if args.override_num_classes else len(stone_state_to_id)
    del first_ckpt
    gc.collect()

    # ── Build validation dataset ─────────────────────────────────────────
    val_dataset = AlchemyDataset(
        json_file_path=args.val_data_path,
        task_type=args.task_type,
        vocab_word2idx=vocab_word2idx,
        vocab_idx2word=vocab_idx2word,
        stone_state_to_id=stone_state_to_id,
        filter_query_from_support=args.filter_query_from_support,
        num_workers=args.num_workers,
        preprocessed_dir=args.preprocessed_dir,
        use_preprocessed=args.use_preprocessed,
        input_format=args.input_format,
        output_format=args.output_format,
        model_architecture=args.model_architecture,
    )
    pad_token_id = val_dataset.pad_token_id
    eos_token_id = val_dataset.eos_token_id if hasattr(val_dataset, "eos_token_id") else None
    sos_token_id = val_dataset.sos_token_id if hasattr(val_dataset, "sos_token_id") else None

    # ── Pre-compute in-support labels for every example ──────────────────
    print("Computing in-support labels...")
    in_support_labels = []
    target_class_ids_all = []
    for item in tqdm(val_dataset, desc="Labelling in-support"):
        encoder_ids = item["encoder_input_ids"]
        target_id = item["target_class_id"]
        target_class_ids_all.append(target_id)

        support_stones = extract_support_stone_ids(
            encoder_ids, vocab_idx2word, stone_state_to_id, hop=None  # hop unused, we find last <item_sep>
        )
        assert len(support_stones) == 8, f"Expected 8 support stones, got {len(support_stones)}: {support_stones}"
        label = 1 if target_id in support_stones else 0
        in_support_labels.append(label)

    in_support_labels = np.array(in_support_labels, dtype=np.int64)
    target_class_ids_all = np.array(target_class_ids_all, dtype=np.int64)
    print(f"In-support rate: {in_support_labels.mean():.4f} "
          f"({in_support_labels.sum()}/{len(in_support_labels)})")

    # Save labels (same for all epochs)
    np.savez(
        os.path.join(args.output_dir, "labels.npz"),
        in_support_labels=in_support_labels,
        target_class_ids=target_class_ids_all,
    )

    # ── Max seq len ──────────────────────────────────────────────────────
    max_length = max(len(item["encoder_input_ids"]) for item in val_dataset)
    if max_length > args.max_seq_len and not args.use_truncation:
        args.max_seq_len = max_length

    # ── DataLoader ───────────────────────────────────────────────────────
    custom_collate = partial(
        collate_fn,
        pad_token_id=pad_token_id,
        eos_token_id=eos_token_id,
        task_type=args.task_type,
        model_architecture=args.model_architecture,
        sos_token_id=sos_token_id,
        prediction_type=args.prediction_type,
        max_seq_len=args.max_seq_len,
        truncate=args.use_truncation,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=custom_collate,
        num_workers=args.num_workers,
        worker_init_fn=worker_init_fn,
        generator=torch.Generator().manual_seed(args.seed),
    )

    # ── Build model ──────────────────────────────────────────────────────
    model = create_decoder_classifier_model(
        config_name=args.model_size,
        src_vocab_size=src_vocab_size,
        num_classes=num_classes,
        device=device,
        max_len=args.max_seq_len,
        prediction_type=args.prediction_type,
        padding_side=args.padding_side,
        use_flash_attention=args.use_flash_attention,
        batch_size=args.batch_size,
    )

    # ── Attach hooks ─────────────────────────────────────────────────────
    extractor = RepresentationExtractor(model, padding_side=args.padding_side)

    # ── Main loop over checkpoints ───────────────────────────────────────
    for epoch_num, ckpt_path in tqdm(epoch_file_pairs, desc="Checkpoints"):
        out_file = os.path.join(args.output_dir, f"representations_epoch_{epoch_num}.npz")
        if os.path.exists(out_file):
            print(f"  Epoch {epoch_num}: already extracted, skipping.")
            continue

        # Load weights
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        del ckpt

        # Collect representations
        all_layer_reps = defaultdict(list)  # layer_name → list of (batch, d_model)
        all_preds = []

        with torch.no_grad():
            for batch in tqdm(val_dataloader, desc=f"Epoch {epoch_num}", leave=False):
                batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
                encoder_input_ids = batch["encoder_input_ids"]
                src_padding_mask = (encoder_input_ids == pad_token_id)

                extractor.clear()

                # Forward pass (triggers hooks)
                output_logits = model(encoder_input_ids, src_padding_mask=src_padding_mask)
                preds = output_logits.argmax(dim=-1).cpu().numpy()
                all_preds.append(preds)

                # Extract last-token representations from hooks
                layer_reps = extractor.collect_last_token_representations(src_padding_mask)
                for name, rep in layer_reps.items():
                    all_layer_reps[name].append(rep.numpy())

        # Concatenate across batches
        save_dict = {"epoch": epoch_num}
        for name in sorted(all_layer_reps.keys()):
            save_dict[name] = np.concatenate(all_layer_reps[name], axis=0)

        save_dict["predicted_class_ids"] = np.concatenate(all_preds, axis=0)

        np.savez_compressed(out_file, **save_dict)
        print(f"  Epoch {epoch_num}: saved {out_file} "
              f"(shapes: {', '.join(f'{k}={v.shape}' for k, v in save_dict.items() if isinstance(v, np.ndarray))})")

        # Cleanup
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    extractor.remove_hooks()
    print("\nDone extracting representations.")


if __name__ == "__main__":
    main()
