"""
val.py — Checkpoint-based validation script.

Given a checkpoint directory (containing per-epoch checkpoints from a training run),
this script iterates over every checkpoint file found, loads the model weights,
runs validation on the provided val_data_path, and logs per-epoch metrics to wandb.

No training is performed — this is purely an evaluation / inference script.

Usage (standalone):
    python src/models/val.py \
        --checkpoint_dir /path/to/checkpoints \
        --val_data_path src/data/.../val_shop_1_qhop_3.json \
        --model_size xsmall --model_architecture decoder ...

Usage (via wandb sweep):
    wandb sweep wandb_sweep_val_config.yaml
    wandb agent <sweep_id>
"""

import argparse
import gc
import glob
import os
import math
import random
import re
import time

import numpy as np
import torch
import torch.nn as nn
from functools import partial
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb
from accelerate import Accelerator
from cluster_profile import cluster

# Reuse components from train.py
from data_loaders import AlchemyDataset, collate_fn
from models import (
    create_transformer_model,
    create_classifier_model,
    create_decoder_classifier_model,
    create_linear_model,
)
from train import (
    set_seed,
    worker_init_fn,
    validate_epoch,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Validate Alchemy model checkpoints (no training)"
    )

    # ── Checkpoint location ──────────────────────────────────────────────
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        required=True,
        help="Directory containing checkpoint .pt files to evaluate.",
    )

    # ── Data ─────────────────────────────────────────────────────────────
    parser.add_argument(
        "--val_data_path",
        type=str,
        required=True,
        help="Path to the validation JSON data file.",
    )
    parser.add_argument(
        "--data_split_seed",
        type=int,
        default=0,
        help="Seed appended to the data path to load the appropriate split.",
    )

    # ── Model ────────────────────────────────────────────────────────────
    parser.add_argument(
        "--model_size",
        type=str,
        default="xsmall",
        choices=[
            "tiny", "xsmall", "xsmall_modified", "xsmall_wide",
            "xsmall_deep", "small", "medium", "large",
        ],
    )
    parser.add_argument(
        "--model_architecture",
        type=str,
        default="decoder",
        choices=["encoder", "decoder", "linear"],
    )
    parser.add_argument("--max_seq_len", type=int, default=2048)

    # ── Task ─────────────────────────────────────────────────────────────
    parser.add_argument(
        "--task_type",
        type=str,
        default="classification",
        choices=[
            "seq2seq", "classification",
            "classification_multi_label", "seq2seq_stone_state",
        ],
    )
    parser.add_argument("--input_format", type=str, default="features",
                        choices=["stone_states", "features"])
    parser.add_argument("--output_format", type=str, default="stone_states",
                        choices=["stone_states", "features"])
    parser.add_argument("--prediction_type", type=str, default="default",
                        choices=["default", "feature", "autoregressive"])
    parser.add_argument("--multi_label_reduction", type=str, default="mean",
                        choices=["mean", "sum", "none"])

    # ── Runtime ──────────────────────────────────────────────────────────
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--override_num_classes", type=int, default=108)
    parser.add_argument("--pooling_strategy", type=str, default="last_token",
                        choices=["global", "query_only", "last_token"])
    parser.add_argument("--padding_side", type=str, default="right",
                        choices=["left", "right"])
    parser.add_argument("--use_flash_attention", type=str, default="True",
                        choices=["True", "False"])
    parser.add_argument("--use_truncation", type=str, default="False",
                        choices=["True", "False"])
    parser.add_argument("--fp16", type=str, default="False",
                        choices=["True", "False"])
    parser.add_argument("--filter_query_from_support", type=str, default="True",
                        choices=["True", "False"])
    parser.add_argument("--include_nonlinearity", type=str, default="True",
                        choices=["True", "False"])
    parser.add_argument("--flatten_linear_model_input", type=str, default="False",
                        choices=["True", "False"])

    # ── Data pre-processing ──────────────────────────────────────────────
    parser.add_argument("--preprocessed_dir", type=str,
                        default="src/data/complete_graph_composition_fully_shuffled_balanced_grouped_by_unique_end_state_preprocessed")
    parser.add_argument("--use_preprocessed", type=str, default="True",
                        choices=["True", "False"])

    # ── Predictions ──────────────────────────────────────────────────────
    parser.add_argument("--store_predictions", type=str, default="True",
                        choices=["True", "False"])
    parser.add_argument("--log_interval", type=int, default=50)

    # ── W&B ──────────────────────────────────────────────────────────────
    parser.add_argument("--wandb_project", type=str,
                        default="alchemy-meta-learning-cross")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--wandb_mode", type=str, default="online",
                        choices=["online", "offline"])

    # ── Train data path (needed to build vocabulary from the training set) ──
    parser.add_argument(
        "--train_data_path",
        type=str,
        default=None,
        help=(
            "Path to the training JSON data file. Required to build the "
            "vocabulary / stone_state_to_id mapping that the checkpoint was "
            "trained with. If not provided, these will be loaded from the "
            "first checkpoint."
        ),
    )

    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────

def discover_checkpoints(checkpoint_dir, task_type, model_size):
    """
    Scan *checkpoint_dir* for files matching the naming convention
    ``best_model_epoch_{N}_{task_type}_{model_size}.pt`` and return them
    sorted by epoch number (ascending).

    Returns
    -------
    list[tuple[int, str]]
        Sorted list of ``(epoch_number, full_path)`` pairs.
    """
    pattern = os.path.join(
        checkpoint_dir,
        f"best_model_epoch_*_{task_type}_{model_size}.pt",
    )
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(
            f"No checkpoint files found matching pattern:\n  {pattern}"
        )

    epoch_file_pairs = []
    regex = re.compile(r"best_model_epoch_(\d+)_")
    for fpath in files:
        m = regex.search(os.path.basename(fpath))
        if m:
            epoch_file_pairs.append((int(m.group(1)), fpath))

    epoch_file_pairs.sort(key=lambda x: x[0])
    return epoch_file_pairs


def resolve_base_path():
    """Return the workspace-root base path depending on the cluster profile."""
    if cluster == "cirrus":
        return "/home/rsaha/projects/dm_alchemy/"
    elif cluster == "cc":
        return "/home/rsaha/projects/def-afyshe-ab/rsaha/projects/dm_alchemy/"
    elif cluster == "rorqual":
        return "/home/rsaha/links/projects/def-afyshe-ab/rsaha/projects/dm_alchemy/"
    elif cluster == "vulcan":
        return "/home/rsaha/projects/def-afyshe-ab/rsaha/dm_alchemy/"
    elif cluster == "killarney":
        return "/home/rsaha/projects/aip-afyshe/rsaha/dm_alchemy/"
    else:
        raise RuntimeError(f"Unknown cluster profile: {cluster}")


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # ── Boolean conversions ──────────────────────────────────────────────
    args.use_flash_attention = str(args.use_flash_attention).lower() == "true"
    args.use_truncation = str(args.use_truncation).lower() == "true"
    args.fp16 = str(args.fp16).lower() == "true"
    args.filter_query_from_support = str(args.filter_query_from_support).lower() == "true"
    args.store_predictions = str(args.store_predictions).lower() == "true"
    args.use_preprocessed = str(args.use_preprocessed).lower() == "true"
    args.flatten_linear_model_input = str(args.flatten_linear_model_input).lower() == "true"
    args.include_nonlinearity = str(args.include_nonlinearity).lower() == "true"

    # ── Cluster-aware paths ──────────────────────────────────────────────
    base_path = resolve_base_path()
    print(f"Base path: {base_path}")
    print(f"Cluster profile: {cluster}")

    # Append data_split_seed to val_data_path
    args.val_data_path = f"{args.val_data_path.split('.json')[0]}_seed_{args.data_split_seed}.json"
    args.val_data_path = os.path.join(base_path, args.val_data_path)
    print(f"Validation data path: {args.val_data_path}")

    # Preprocessed dir
    args.preprocessed_dir = os.path.join(base_path, args.preprocessed_dir)
    print(f"Preprocessed directory: {args.preprocessed_dir}")

    # ── Task-specific output format fixups ────────────────────────────────
    if args.task_type == "classification":
        args.output_format = "stone_states"
    elif args.task_type == "classification_multi_label":
        args.output_format = "features"

    # ── Accelerator ──────────────────────────────────────────────────────
    if args.fp16:
        accelerator = Accelerator(mixed_precision="fp16")
    else:
        accelerator = Accelerator()

    set_seed(args.seed)
    print(f"Using device: {accelerator.device}")

    # ── Discover checkpoints ─────────────────────────────────────────────
    epoch_file_pairs = discover_checkpoints(
        args.checkpoint_dir, args.task_type, args.model_size
    )
    print(f"Found {len(epoch_file_pairs)} checkpoint(s) in {args.checkpoint_dir}")
    print(f"Epoch range: {epoch_file_pairs[0][0]} – {epoch_file_pairs[-1][0]}")

    # ── Load vocabulary / mappings from the first checkpoint ─────────────
    # We need the vocabulary that the model was trained with so that we can
    # tokenise the validation data identically.
    first_ckpt = torch.load(epoch_file_pairs[0][1], map_location="cpu", weights_only=False)

    vocab_word2idx = first_ckpt.get("src_vocab_word2idx", None)
    vocab_idx2word = first_ckpt.get("src_vocab_idx2word", None)
    stone_state_to_id = first_ckpt.get("stone_state_to_id", None)
    
    # 1. Extract target vocabularies if they exist (crucial for seq2seq tasks)
    tgt_vocab_word2idx = first_ckpt.get("tgt_vocab_word2idx", None)
    tgt_vocab_idx2word = first_ckpt.get("tgt_vocab_idx2word", None)

    checkpoint_args = first_ckpt.get("args", None)

    if vocab_word2idx is None:
        raise ValueError(
            "Checkpoint does not contain 'src_vocab_word2idx'. "
            "Cannot build validation dataset without the training vocabulary."
        )

    # 2. Strict validation to ensure we don't silently create a new state mapping
    if args.task_type == "classification" and stone_state_to_id is None:
        raise ValueError(
            "Checkpoint does not contain 'stone_state_to_id'. "
            "This is required to ensure classification targets map to the correct IDs during cross-evaluation!"
        )

    # If train_data_path was provided, we can also build a full training
    # dataset for vocabulary (like the training script does). But the
    # checkpoint already stores the vocab, so we use that directly.

    # ── Build validation dataset ─────────────────────────────────────────
    print(f"Loading validation data from: {args.val_data_path}")
    val_dataset = AlchemyDataset(
        json_file_path=args.val_data_path,
        task_type=args.task_type,
        vocab_word2idx=vocab_word2idx,
        vocab_idx2word=vocab_idx2word,
        stone_state_to_id=stone_state_to_id if args.task_type == "classification" else None,
        filter_query_from_support=args.filter_query_from_support,
        num_workers=args.num_workers,
        preprocessed_dir=args.preprocessed_dir,
        use_preprocessed=args.use_preprocessed,
        input_format=args.input_format,
        output_format=args.output_format,
        model_architecture=args.model_architecture,
    )

    pad_token_id = val_dataset.pad_token_id
    src_vocab_size = len(vocab_word2idx)
    eos_token_id = val_dataset.eos_token_id if hasattr(val_dataset, "eos_token_id") else None
    sos_token_id = val_dataset.sos_token_id if hasattr(val_dataset, "sos_token_id") else None

    print(f"Source vocabulary size: {src_vocab_size}")
    print(f"Pad token ID: {pad_token_id}")

    # ── Adjust max_seq_len if needed ─────────────────────────────────────
    max_length = max(len(item["encoder_input_ids"]) for item in val_dataset)
    print(f"Max sequence length in validation data: {max_length}")
    if max_length > args.max_seq_len:
        if args.use_truncation:
            print(f"Truncation enabled — keeping max_seq_len={args.max_seq_len}")
        else:
            args.max_seq_len = max_length
            print(f"Adjusted max_seq_len to {args.max_seq_len}")

    # ── DataLoader ───────────────────────────────────────────────────────
    custom_collate_val = partial(
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
        collate_fn=custom_collate_val,
        num_workers=args.num_workers,
        worker_init_fn=worker_init_fn,
        generator=torch.Generator().manual_seed(args.seed),
    )

    # ── Determine number of classes / vocab sizes ────────────────────────
    if args.task_type in ("seq2seq", "seq2seq_stone_state"):
        if args.task_type == "seq2seq":
            tgt_vocab_size = src_vocab_size
        else:
            tgt_vocab_size = len(first_ckpt.get("tgt_vocab_word2idx", vocab_word2idx))
    elif args.task_type == "classification":
        num_classes = (
            args.override_num_classes
            if args.override_num_classes is not None
            else len(stone_state_to_id)
        )
        print(f"Number of classes: {num_classes}")
    elif args.task_type == "classification_multi_label":
        num_output_features = first_ckpt.get(
            "num_output_features", val_dataset.num_output_features
        )
        print(f"Number of output features: {num_output_features}")

    # ── Build model skeleton (weights will be loaded per checkpoint) ──────
    if args.task_type in ("seq2seq", "seq2seq_stone_state"):
        model = create_transformer_model(
            config_name=args.model_size,
            src_vocab_size=src_vocab_size,
            tgt_vocab_size=tgt_vocab_size,
            device=accelerator.device,
            max_len=args.max_seq_len,
        )
        criterion = nn.CrossEntropyLoss(
            ignore_index=pad_token_id, reduction=args.multi_label_reduction
        )
    elif args.task_type == "classification":
        if args.model_architecture == "decoder":
            model = create_decoder_classifier_model(
                config_name=args.model_size,
                src_vocab_size=src_vocab_size,
                num_classes=num_classes,
                device=accelerator.device,
                max_len=args.max_seq_len,
                prediction_type=args.prediction_type,
                padding_side=args.padding_side,
                use_flash_attention=args.use_flash_attention,
                batch_size=args.batch_size,
            )
        elif args.model_architecture == "linear":
            model = create_linear_model(
                config_name=args.model_size,
                input_size=src_vocab_size,
                num_classes=num_classes,
                device=accelerator.device,
                max_len=args.max_seq_len,
                io_sep_token_id=val_dataset.io_sep_token_id if hasattr(val_dataset, "io_sep_token_id") else None,
                item_sep_token_id=val_dataset.item_sep_token_id if hasattr(val_dataset, "item_sep_token_id") else None,
                pooling_strategy=args.pooling_strategy,
                batch_size=args.batch_size,
                use_flash_attention=args.use_flash_attention,
                padding_side=args.padding_side,
                include_nonlinearity=args.include_nonlinearity,
                flatten_input=args.flatten_linear_model_input,
            )
        else:
            model = create_classifier_model(
                config_name=args.model_size,
                src_vocab_size=src_vocab_size,
                num_classes=num_classes,
                device=accelerator.device,
                max_len=args.max_seq_len,
                io_sep_token_id=val_dataset.io_sep_token_id if hasattr(val_dataset, "io_sep_token_id") else None,
                item_sep_token_id=val_dataset.item_sep_token_id if hasattr(val_dataset, "item_sep_token_id") else None,
                pooling_strategy=args.pooling_strategy,
            )
        criterion = nn.CrossEntropyLoss(reduction=args.multi_label_reduction)
    elif args.task_type == "classification_multi_label":
        if args.model_architecture == "decoder":
            model = create_decoder_classifier_model(
                config_name=args.model_size,
                src_vocab_size=src_vocab_size,
                num_classes=num_output_features,
                device=accelerator.device,
                max_len=args.max_seq_len,
                prediction_type=args.prediction_type,
                padding_side=args.padding_side,
            )
        else:
            model = create_classifier_model(
                config_name=args.model_size,
                src_vocab_size=src_vocab_size,
                num_classes=num_output_features,
                device=accelerator.device,
                max_len=args.max_seq_len,
                io_sep_token_id=val_dataset.io_sep_token_id if hasattr(val_dataset, "io_sep_token_id") else None,
                item_sep_token_id=val_dataset.item_sep_token_id if hasattr(val_dataset, "item_sep_token_id") else None,
                pooling_strategy=args.pooling_strategy,
            )
        criterion = nn.CrossEntropyLoss(reduction=args.multi_label_reduction)
    else:
        raise ValueError(f"Unknown task_type: {args.task_type}")

    # Prepare model with accelerator (no optimizer / scheduler needed)
    model = accelerator.prepare(model)

    # ── Extract info for save_dir (used by validate_epoch for predictions) ──
    val_hop_match = re.search(r"shop_(\d+)_qhop_(\d+)", args.val_data_path)
    val_shop = val_hop_match.group(1) if val_hop_match else "?"
    val_qhop = val_hop_match.group(2) if val_hop_match else "?"

    ckpt_hop_match = re.search(r"shop_(\d+)_qhop_(\d+)", args.checkpoint_dir)
    ckpt_shop = ckpt_hop_match.group(1) if ckpt_hop_match else "?"
    ckpt_qhop = ckpt_hop_match.group(2) if ckpt_hop_match else "?"

    # Build a save_dir for predictions
    args.save_dir = os.path.join(
        args.checkpoint_dir,
        f"val_predictions_shop_{val_shop}_qhop_{val_qhop}",
    )
    if accelerator.is_local_main_process:
        os.makedirs(args.save_dir, exist_ok=True)

    # ── W&B init ─────────────────────────────────────────────────────────
    if accelerator.is_local_main_process:
        run_name = args.wandb_run_name or (
            f"val_ckpt_shop{ckpt_shop}_q{ckpt_qhop}"
            f"__on_shop{val_shop}_q{val_qhop}"
            f"__{args.model_size}_{time.strftime('%Y%m%d-%H%M%S')}"
        )
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            config=vars(args),
            mode=args.wandb_mode,
            name=run_name,
        )

    # ── Main evaluation loop ─────────────────────────────────────────────
    best_val_acc = 0.0
    best_val_loss = float("inf")
    best_epoch = -1

    print(f"\n{'='*60}")
    print(f"Starting validation over {len(epoch_file_pairs)} checkpoints")
    print(f"Checkpoint dir : {args.checkpoint_dir}")
    print(f"Val data       : {args.val_data_path}")
    print(f"{'='*60}\n")

    for epoch_num, ckpt_path in tqdm(
        epoch_file_pairs,
        desc="Evaluating checkpoints",
        disable=not accelerator.is_local_main_process,
    ):
        # Load checkpoint weights
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.load_state_dict(checkpoint["model_state_dict"])

        # Run validation
        with accelerator.autocast():
            if args.task_type == "seq2seq":
                val_loss, val_acc, val_batch_accs = validate_epoch(
                    model, val_dataloader, criterion, accelerator,
                    epoch_num, pad_token_id, args,
                )
            else:
                val_loss, val_acc = validate_epoch(
                    model, val_dataloader, criterion, accelerator,
                    epoch_num, pad_token_id, args,
                )

        # Log to wandb
        if accelerator.is_local_main_process:
            print(
                f"Epoch {epoch_num:4d} | "
                f"Val Loss: {val_loss:.6f} | Val Acc: {val_acc:.6f}"
            )
            wandb.log({
                "epoch": epoch_num,
                "val_loss": val_loss,
                "val_accuracy": val_acc,
            })

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch_num
            if val_loss < best_val_loss:
                best_val_loss = val_loss

        # Free checkpoint memory
        del checkpoint
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    # ── Summary ──────────────────────────────────────────────────────────
    if accelerator.is_local_main_process:
        print(f"\n{'='*60}")
        print(f"Validation complete.")
        print(f"Best val accuracy : {best_val_acc:.6f} (epoch {best_epoch})")
        print(f"Best val loss     : {best_val_loss:.6f}")
        print(f"{'='*60}")

        wandb.summary["best_val_accuracy"] = best_val_acc
        wandb.summary["best_val_loss"] = best_val_loss
        wandb.summary["best_epoch"] = best_epoch
        wandb.finish()

    accelerator.end_training()


if __name__ == "__main__":
    main()
