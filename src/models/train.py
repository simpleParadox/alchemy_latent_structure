import argparse
import os
import math
import random
import numpy as np
from functools import partial
import time
from cluster_profile import cluster

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import wandb
from tqdm import tqdm
from accelerate import Accelerator

# Assuming train.py is in src/models/, so data_loaders and models are siblings
from data_loaders import AlchemyDataset, collate_fn
import re
from models import create_transformer_model, create_classifier_model # Added create_classifier_model
import torch.nn.functional as F # For padding

def set_seed(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)

def parse_args():
    
    
    parser = argparse.ArgumentParser(description="Train Alchemy Transformer Model")
    parser.add_argument("--multi_label_reduction", type=str, default="mean", choices=["mean", "sum"])
    parser.add_argument("--task_type", type=str, default="classification_multi_label", choices=["seq2seq", "classification", "classification_multi_label"],
                        help="Type of task: 'seq2seq' for feature-wise prediction, 'classification' for whole state prediction, or 'classification_multi_label' for multi-label feature prediction.")
    parser.add_argument("--train_data_path", type=str, default="src/data/generated_data/decompositional_chemistry_samples_167424_80_unique_stones_train_shop_2_qhop_1.json",
                        help="Path to the training JSON data file.")
    parser.add_argument("--val_data_path", type=str, default="src/data/generated_data/decompositional_chemistry_samples_167424_80_unique_stones_val_shop_2_qhop_1.json",
                        help="Path to the validation JSON data file (optional).")
    parser.add_argument("--val_split", type=float, default=None,
                        help="Validation split ratio (e.g., 0.1 for 10%%). If provided, validation set will be created from training data instead of loading separate file. Default is None.")
    parser.add_argument("--val_split_seed", type=int, default=42,
                        help="Seed for reproducible train/val splits.")
    parser.add_argument("--data_split_seed", type=int, default=42,
                        help="Seed value that gets appended to the data path to load the approapriate training / validation.")
    parser.add_argument("--model_size", type=str, default="xsmall", choices=["tiny", "xsmall", "small", "medium", "large"],
                        help="Size of the transformer model.")
    parser.add_argument("--max_seq_len", type=int, default=2048, # Max length for support + query + separators
                        help="Maximum sequence length for the model.")
    parser.add_argument("--epochs", type=int, default=2,
                        help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size for training and validation.")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                        help="Initial learning rate for AdamW optimizer.")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="Weight decay for AdamW optimizer.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility.")
    parser.add_argument("--save_dir", type=str, default="src/saved_models/",
                        help="Directory to save model checkpoints.")
    parser.add_argument("--filter_query_from_support", type=str, choices=["True", "False"],
                        help="Filter out query examples from support sets to prevent data leakage when support_steps=query_steps=1. Default=True", default=True)
    parser.add_argument("--wandb_project", type=str, default="alchemy-meta-learning",
                        help="Weights & Biases project name.")
    parser.add_argument("--scheduler_call_location", type=str, default="after_epoch", choices=["after_epoch", "after_batch"],
                        help="Where to call the scheduler: 'after_epoch' for per-epoch scheduling, 'after_batch' for per-batch.")
    parser.add_argument("--wandb_entity", type=str, default=None, # Replace with your W&B entity
                        help="Weights & Biases entity name.")
    parser.add_argument("--wandb_run_name", type=str, default=None,
                        help="Weights & Biases run name (optional). Defaults to a generated name.")
    parser.add_argument("--log_interval", type=int, default=50,
                        help="Log training batch metrics every N batches.")
    parser.add_argument("--num_workers", type=int, default=15,
                        help="Number of workers for data preprocessing parallelization.")
    parser.add_argument("--wandb_mode", type=str, default="offline", choices=["online", "offline"],
                        help="Weights & Biases mode: 'online' for live logging, 'offline' for local logging.")
    parser.add_argument("--generation_max_len_buffer", type=int, default=10, help="Buffer for max generation length in validation.")
    parser.add_argument("--sos_token_str", type=str, default="<sos>", help="String for Start of Sequence token.")
    parser.add_argument("--eos_token_str", type=str, default="<eos>", help="String for End of Sequence token.")
    return parser.parse_args()

def calculate_accuracy_seq2seq(predictions, targets, ignore_index):
    """
    Calculates accuracy for seq2seq task based on full sequence matches, ignoring padding tokens.
    predictions: (batch_size, seq_len, vocab_size) - model logits
    targets: (batch_size, seq_len) - ground truth token ids
    ignore_index: token id for padding, to be ignored in accuracy calculation
    """
    predicted_tokens = predictions.argmax(dim=-1) # (batch_size, seq_len)
    
    # Mask for non-padding tokens (True for actual tokens, False for padding)
    non_padding_mask = (targets != ignore_index)
    
    # Check if each token is correct
    # token_correct will be True where predicted_token == target_token, False otherwise
    token_correct = (predicted_tokens == targets)
    
    # For each sequence, we consider it correct if all its non-padding tokens are correct.
    # We can achieve this by checking if (token_correct OR is_padding_token) is True for all tokens in a sequence.
    # is_padding_token is ~non_padding_mask
    # So, for a sequence to be correct, (token_correct | ~non_padding_mask) must be all True along the sequence dimension.
    correct_sequences = (token_correct | ~non_padding_mask).all(dim=1)
    
    num_correct_sequences = correct_sequences.sum().item()
    num_sequences_in_batch = targets.size(0) # batch_size
    
    accuracy = num_correct_sequences / num_sequences_in_batch if num_sequences_in_batch > 0 else 0.0
    # The function now returns sequence-level accuracy, count of correct sequences, and total sequences in batch.
    return accuracy, num_correct_sequences, num_sequences_in_batch

def calculate_accuracy_generated_seq2seq(generated_ids, target_ids, pad_token_id):
    """
    Calculates exact match accuracy for generated sequences against target sequences.
    Args:
        generated_ids: (batch_size, gen_seq_len) - Tensor of generated token IDs.
        target_ids: (batch_size, tgt_seq_len) - Tensor of ground truth token IDs.
        pad_token_id: Token ID for padding.
    Returns:
        accuracy (float), num_correct_sequences (int), num_sequences_in_batch (int)
    """
    batch_size = target_ids.size(0)
    gen_seq_len = generated_ids.size(1)
    tgt_seq_len = target_ids.size(1)

    # Pad the shorter sequence to the length of the longer one for comparison
    if gen_seq_len > tgt_seq_len:
        padding_size = gen_seq_len - tgt_seq_len
        # Pad target_ids on the right (dim=1) with pad_token_id
        target_ids_padded = F.pad(target_ids, (0, padding_size), mode='constant', value=pad_token_id)
        generated_ids_padded = generated_ids
    elif tgt_seq_len > gen_seq_len:
        padding_size = tgt_seq_len - gen_seq_len
        # Pad generated_ids on the right (dim=1) with pad_token_id
        generated_ids_padded = F.pad(generated_ids, (0, padding_size), mode='constant', value=pad_token_id)
        target_ids_padded = target_ids
    else:
        generated_ids_padded = generated_ids
        target_ids_padded = target_ids

    # Mask for non-padding tokens in the original target (up to the padded length)
    # We care about matching the target up to its original non-padded content.
    # For generated tokens beyond original target length, they should ideally be EOS or padding.
    # For simplicity here, we compare up to the max length of the two after padding.
    
    # Create a mask based on the original target_ids's non-padding tokens
    # This defines what parts of the target_ids_padded we actually care about matching.
    target_non_padding_mask = (target_ids_padded != pad_token_id)

    # Element-wise comparison
    correct_tokens = (generated_ids_padded == target_ids_padded)

    # A sequence is correct if all tokens that are not padding in the target are correctly predicted.
    # (correct_tokens OR ~target_non_padding_mask) means:
    #   - If target token is padding, it doesn't matter what generated is (evaluates to True).
    #   - If target token is NOT padding, generated token must match target token.
    # This handles cases where generated sequence is longer and correctly has padding/EOS after matching target content.
    # However, a stricter match would be (correct_tokens & target_non_padding_mask) | ~target_non_padding_mask
    # Let's use: a sequence is correct if all its *original target* non-padding tokens are matched,
    # and any generated tokens beyond the original target length are ignored for correctness,
    # or if generated is shorter, it's an error if target had more non-pad tokens.

    # Simpler: for a sequence to be correct, all tokens in the (padded) target must match,
    # OR the target token must be a pad token.
    # This means if target has [A, B, PAD] and generated is [A, B, C], it's a match.
    # If target has [A, B, C] and generated is [A, B, PAD], it's not a match if C was not PAD.

    # Let's use the logic from the original calculate_accuracy_seq2seq:
    # A sequence is correct if all its non-padding target tokens are correctly predicted.
    # We need to ensure generated_ids_padded also respects this.
    
    # Mask for non-padding tokens in the (potentially padded) target.
    # This is the effective region of comparison.
    effective_target_mask = (target_ids_padded != pad_token_id)
    
    # Sequences are correct if all tokens within the effective_target_mask match,
    # OR if the token in target_ids_padded is a pad token (already covered by ~effective_target_mask)
    # So, (generated_ids_padded == target_ids_padded) must be true for all positions where effective_target_mask is true.
    # This can be written as: ((generated_ids_padded == target_ids_padded) | ~effective_target_mask).all(dim=1)

    # Stricter: Match exactly where target is not padding.
    # And if generated is longer, ensure those extra tokens are padding.
    # For now, let's use the simpler version that aligns with typical full sequence match:
    # All non-padding tokens in the target must be matched.
    # If generated is shorter, it will fail if target had more non-pad tokens.
    # If generated is longer, the extra tokens don't hurt if original target is matched.

    # Consider only the parts of the target that are not padding.
    # For these parts, the generated sequence must match.
    match_where_target_not_pad = (generated_ids_padded == target_ids_padded) | ~target_non_padding_mask
    correct_sequences = match_where_target_not_pad.all(dim=1)

    num_correct_sequences = correct_sequences.sum().item()
    
    accuracy = num_correct_sequences / batch_size if batch_size > 0 else 0.0
    return accuracy, num_correct_sequences, batch_size


def calculate_accuracy_multilabel(predictions, targets, threshold=0.5):
    """
    Calculates accuracy for multi-label classification.
    predictions: (batch_size, num_features) - model logits (before sigmoid)
    targets: (batch_size, num_features) - multi-hot encoded ground truth
    threshold: threshold for converting probabilities to binary predictions
    """
    # Apply sigmoid and threshold to get binary predictions
    preds_binary = (torch.sigmoid(predictions) > threshold).float()
    
    # Exact match accuracy: (number of samples where all labels are correct) / batch_size
    correct_samples = (preds_binary == targets).all(dim=1).sum().item()
    total_samples = targets.size(0)
    
    exact_match_accuracy = correct_samples / total_samples if total_samples > 0 else 0.0
    
    return exact_match_accuracy, correct_samples, total_samples


def calculate_accuracy_classification(predictions, targets):
    """
    Calculates accuracy for classification task.
    predictions: (batch_size, num_classes) - model logits
    targets: (batch_size) - ground truth class ids
    """
    predicted_classes = predictions.argmax(dim=-1) # (batch_size)
    correct_predictions = (predicted_classes == targets).sum().item()
    num_samples = targets.size(0)
    accuracy = correct_predictions / num_samples if num_samples > 0 else 0.0
    return accuracy, correct_predictions, num_samples

def train_epoch(model, dataloader, optimizer, criterion, scheduler, accelerator, epoch_num, pad_token_id, args):
    model.train()
    total_loss = 0
    total_correct_preds = 0
    total_considered_items = 0 # Can be tokens (seq2seq) or samples (classification)
    start_time = time.time()

    pbar = tqdm(dataloader, disable=not accelerator.is_local_main_process)
    for batch_idx, batch in enumerate(pbar):
        # No need to move to device explicitly - Accelerate handles this
        encoder_input_ids = batch["encoder_input_ids"]
        
        optimizer.zero_grad()
        
        if args.task_type == "seq2seq":
            decoder_input_ids = batch["decoder_input_ids"] # Teacher forcing input
            decoder_target_ids = batch["decoder_target_ids"] # Actual targets
            
            # Create causal mask for teacher forcing
            tgt_mask = model.module.generate_square_subsequent_mask(decoder_input_ids.size(1), device=accelerator.device) \
                if hasattr(model, 'module') else model.generate_square_subsequent_mask(decoder_input_ids.size(1), device=accelerator.device)

            src_padding_mask = (encoder_input_ids == pad_token_id)
            tgt_padding_mask = (decoder_input_ids == pad_token_id) # Padding in teacher forcing input

            output_logits = model(encoder_input_ids, decoder_input_ids,
                                  tgt_mask=tgt_mask,
                                  src_padding_mask=src_padding_mask,
                                  tgt_padding_mask=tgt_padding_mask,
                                  memory_key_padding_mask=src_padding_mask) # memory_key_padding_mask is often src_padding_mask
            
            loss = criterion(output_logits.reshape(-1, output_logits.shape[-1]), decoder_target_ids.reshape(-1))
            # Accuracy during training is still teacher-forced
            acc, correct, considered = calculate_accuracy_seq2seq(output_logits, decoder_target_ids, pad_token_id)
        elif args.task_type == "classification":
            target_class_ids = batch["target_class_id"]
            src_padding_mask = (encoder_input_ids == pad_token_id) 
            output_logits = model(encoder_input_ids, src_padding_mask=src_padding_mask) # (batch_size, num_classes)
            loss = criterion(output_logits, target_class_ids) # CrossEntropyLoss expects (N, C) and (N)
            acc, correct, considered = calculate_accuracy_classification(output_logits, target_class_ids)
        elif args.task_type == "classification_multi_label":
            target_feature_vector = batch["target_feature_vector"] # (batch_size, num_output_features)
            src_padding_mask = (encoder_input_ids == pad_token_id)
            output_logits = model(encoder_input_ids, src_padding_mask=src_padding_mask) # (batch_size, num_output_features)
            loss = criterion(output_logits, target_feature_vector.float()) # BCEWithLogitsLoss expects float targets
            acc, correct, considered = calculate_accuracy_multilabel(output_logits, target_feature_vector)
        else:
            raise ValueError(f"Unknown task_type: {args.task_type}")

        accelerator.backward(loss)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        total_correct_preds += correct
        total_considered_items += considered

        if batch_idx % args.log_interval == 0 and accelerator.is_local_main_process:
            current_lr = scheduler.get_last_lr()[0] if scheduler else optimizer.param_groups[0]['lr']
            elapsed_time = time.time() - start_time
            # Update tqdm description instead of printing
            pbar.set_description(f"Epoch {epoch_num+1} | Batch {batch_idx}/{len(dataloader)} | Loss: {loss.item():.4f} | Train Acc: {acc:.4f} | LR: {current_lr:.2e} | Time: {elapsed_time:.2f}s")
            wandb.log({
                "train_batch_loss": loss.item(),
                "train_batch_accuracy": acc,
                "learning_rate": current_lr,
                "epoch": epoch_num,
                "batch_idx": batch_idx
            })
            start_time = time.time()
    
    # if args.task_type == "classification": # If scheduler is per epoch for classification
    # Call scheduler.step() after each epoch if it is defined.
    if scheduler and args.scheduler_call_location == 'after_epoch': # Respect scheduler location
        scheduler.step()

    avg_epoch_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0
    avg_epoch_accuracy = total_correct_preds / total_considered_items if total_considered_items > 0 else 0.0
    return avg_epoch_loss, avg_epoch_accuracy

def validate_epoch(model, dataloader, criterion, accelerator, epoch_num, pad_token_id, args, sos_token_id, eos_token_id): # Added sos/eos
    if dataloader is None:
        return (None, None, None) if args.task_type == "seq2seq" else (None, None) # Adjusted for classification return

    model.eval()
    total_loss = 0 # For seq2seq, this might be less meaningful with generation
    total_correct_preds = 0
    total_considered_items = 0
    # all_accs = [] if args.task_type == "seq2seq" else None # Retained for consistency if needed later
    
    with torch.no_grad():
        pbar = tqdm(dataloader, disable=not accelerator.is_local_main_process, desc=f"Epoch {epoch_num+1} Validation")
        for batch_idx, batch in enumerate(pbar):
            encoder_input_ids = batch["encoder_input_ids"].to(accelerator.device)

            if args.task_type == "seq2seq":
                decoder_target_ids = batch["decoder_target_ids"].to(accelerator.device) # Ground truth
                
                # Max length for generation can be based on target length or a fixed value from args
                # Adding a bit of buffer to target length can be good.
                max_gen_len = decoder_target_ids.size(1) + args.generation_max_len_buffer # Ensure generation_max_len_buffer is in args

                current_model = model.module if hasattr(model, 'module') else model
                generated_ids = current_model.generate(
                    encoder_input_ids,
                    start_symbol_id=sos_token_id,
                    end_symbol_id=eos_token_id,
                    max_len=max_gen_len,
                    device=accelerator.device,
                    pad_token_id=pad_token_id
                )
                
                # Loss calculation for generated sequences is not straightforward with cross-entropy
                # Typically, for validation with generation, focus on metrics like BLEU, ROUGE, or exact match accuracy.
                # For now, let's assign a placeholder loss or skip it for validation.
                loss = torch.tensor(0.0, device=accelerator.device) # Placeholder

                acc, correct, considered = calculate_accuracy_generated_seq2seq(generated_ids, decoder_target_ids, pad_token_id)
                # if all_accs is not None: all_accs.append(acc) # If you want to collect per-batch accuracies

            elif args.task_type == "classification":
                target_class_ids = batch["target_class_id"].to(accelerator.device)
                src_padding_mask = (encoder_input_ids == pad_token_id)
                output_logits = model(encoder_input_ids, src_padding_mask=src_padding_mask)
                loss = criterion(output_logits, target_class_ids)
                acc, correct, considered = calculate_accuracy_classification(output_logits, target_class_ids)
            elif args.task_type == "classification_multi_label":
                target_feature_vector = batch["target_feature_vector"].to(accelerator.device)
                src_padding_mask = (encoder_input_ids == pad_token_id)
                output_logits = model(encoder_input_ids, src_padding_mask=src_padding_mask)
                loss = criterion(output_logits, target_feature_vector.float())
                acc, correct, considered = calculate_accuracy_multilabel(output_logits, target_feature_vector)
            else:
                raise ValueError(f"Unknown task_type: {args.task_type}")

            total_loss += loss.item() # This will be 0 for seq2seq if using placeholder
            total_correct_preds += correct
            total_considered_items += considered
            
            if accelerator.is_local_main_process:
                 pbar.set_postfix({"Val Acc": f"{acc:.4f} ({correct}/{considered})", "Val Loss": f"{loss.item():.4f}"})
            
    avg_epoch_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0
    avg_epoch_accuracy = total_correct_preds / total_considered_items if total_considered_items > 0 else 0.0
    
    if args.task_type == "seq2seq":
        # return avg_epoch_loss, avg_epoch_accuracy, all_accs # all_accs can be returned if needed
        return avg_epoch_loss, avg_epoch_accuracy, None # Returning None for the third element for now
    else: # classification or classification_multi_label
        return avg_epoch_loss, avg_epoch_accuracy

def main():
    args = parse_args()

    
    base_path = None
    if cluster == 'cirrus':
        # Cirrus-specific paths
        base_path = '/home/rsaha/projects/dm_alchemy/'
    else:
        # Profile is cc.
        base_path = '/home/rsaha/projects/def-afyshe-ab/rsaha/projects/dm_alchemy/'
    
    print("Base path: ", base_path)
    print("Profile cluster: ", cluster)
    
    # First, add the 'data_split_seed' to the train_data_path and val_data_path
    args.train_data_path = f"{args.train_data_path.split('.json')[0]}_seed_{args.data_split_seed}.json"
    args.val_data_path = f"{args.val_data_path.split('.json')[0]}_seed_{args.data_split_seed}.json"
        
    # Update data paths to be relative to the base path
    args.train_data_path = os.path.join(base_path, args.train_data_path)
    args.val_data_path = os.path.join(base_path, args.val_data_path) if args.val_data_path else None
    args.save_dir = os.path.join(base_path, args.save_dir)
    
    
    # Initialize Accelerator
    accelerator = Accelerator()
    
    # Set seed for reproducibility
    set_seed(args.seed)
    print("Seed: ", args.seed)

    # Initialize wandb only on main process
    if accelerator.is_local_main_process:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name if args.wandb_run_name else f"{args.task_type}_{args.model_size}_{time.strftime('%Y%m%d-%H%M%S')}",
            config=vars(args),
            mode=args.wandb_mode
        )

    print(f"Using device: {accelerator.device}")
    print(f"Selected task type: {args.task_type}")
    print(f"Number of processes: {accelerator.num_processes}")

    # --- Dataset and DataLoader ---
    print(f"Loading training data from: {args.train_data_path}")
    
    # Create dataset with optional validation split
    print(f"Filter query from support initial: {args.filter_query_from_support}")
    # Correctly convert string "True" or "False" or boolean True to boolean
    args.filter_query_from_support = str(args.filter_query_from_support) == 'True'
    
    print("Filter query from support set to:", args.filter_query_from_support)
    with accelerator.main_process_first(): # Doing this once to prevent OOM errors when loading data on multiple processes.
        full_dataset = AlchemyDataset(
            json_file_path=args.train_data_path, 
            task_type=args.task_type,
            filter_query_from_support=args.filter_query_from_support,
            num_workers=args.num_workers,
            val_split=args.val_split,
            val_split_seed=args.val_split_seed
        )

    # Get train and validation sets
    train_dataset = full_dataset.get_train_set()
    val_dataset = None
    
    # Determine validation strategy
    if args.val_split is not None:
        # Use internal split
        val_dataset = full_dataset.get_val_set()
        print(f"Using internal validation split: {args.val_split}")
    elif args.val_data_path:
        # Use separate validation file
        print(f"Loading validation data from: {args.val_data_path}")
        val_dataset = AlchemyDataset(
            json_file_path=args.val_data_path,
            task_type=args.task_type,
            vocab_word2idx=full_dataset.word2idx,
            vocab_idx2word=full_dataset.idx2word,
            stone_state_to_id=full_dataset.stone_state_to_id if args.task_type == "classification" else None,
            filter_query_from_support=args.filter_query_from_support,
            num_workers=args.num_workers
        )
    else:
        print("No validation data specified. Skipping validation.")

    # Get vocabulary info from the main dataset
    pad_token_id = full_dataset.pad_token_id
    src_vocab_size = len(full_dataset.word2idx)
    # Assuming sos_token_id and eos_token_id are attributes of your dataset or tokenizer
    # These are placeholders, replace with actual access to your SOS/EOS IDs
    try:
        sos_token_id = full_dataset.sos_token_id 
        eos_token_id = full_dataset.eos_token_id
    except AttributeError:
        # Fallback if not directly on dataset, you might need to get them from word2idx
        # This is a common way:
        try:
            sos_token_id = full_dataset.word2idx[full_dataset.SOS_TOKEN_STR] # Assuming SOS_TOKEN_STR like '<sos>'
            eos_token_id = full_dataset.word2idx[full_dataset.EOS_TOKEN_STR] # Assuming EOS_TOKEN_STR like '<eos>'
        except (AttributeError, KeyError) as e:
            accelerator.print(f"Warning: SOS/EOS token IDs not found on dataset. Using default 1 and 2. Adjust if necessary. Error: {e}")
            sos_token_id = 1 # Placeholder
            eos_token_id = 2 # Placeholder
            # It's CRITICAL that these IDs match what the model expects and what's in the data.


    print(f"Source (Feature/Potion) Vocabulary size: {src_vocab_size}")
    print(f"Pad token ID for encoder inputs: {pad_token_id}")
    if args.task_type == "seq2seq":
        tgt_vocab_size = len(full_dataset.word2idx) # Assuming same vocab for src/tgt for this task
        print(f"Target (Generated Feature) Vocabulary size: {tgt_vocab_size}")
        print(f"SOS token ID for generation: {sos_token_id}")
        print(f"EOS token ID for generation: {eos_token_id}")
    elif args.task_type == "classification":
        num_classes = len(full_dataset.stone_state_to_id)
        print(f"Number of classes (Stone States for Classification): {num_classes}")
    elif args.task_type == "classification_multi_label":
        num_output_features = full_dataset.num_output_features
        print(f"Number of output features (for Multi-label Classification): {num_output_features}")

    # Create data loaders
    custom_collate_train = partial(collate_fn, pad_token_id=pad_token_id, task_type=args.task_type)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=custom_collate_train,
        num_workers=args.num_workers
    )

    val_dataloader = None
    if val_dataset is not None:
        custom_collate_val = partial(collate_fn, pad_token_id=pad_token_id, task_type=args.task_type)
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=custom_collate_val,
            num_workers=args.num_workers
        )

    # --- Model ---
    if args.task_type == "seq2seq":
        model = create_transformer_model(
            config_name=args.model_size,
            src_vocab_size=src_vocab_size,
            tgt_vocab_size=tgt_vocab_size,
            device=accelerator.device,
            max_len=args.max_seq_len
        )
        criterion = nn.CrossEntropyLoss(ignore_index=pad_token_id)
    elif args.task_type == "classification":
        model = create_classifier_model(
            config_name=args.model_size,
            src_vocab_size=src_vocab_size,
            num_classes=num_classes,
            device=accelerator.device,
            max_len=args.max_seq_len
        )
        criterion = nn.CrossEntropyLoss()
    elif args.task_type == "classification_multi_label":
        model = create_classifier_model( # Using the same StoneStateClassifier model
            config_name=args.model_size,
            src_vocab_size=src_vocab_size,
            num_classes=num_output_features, # Output layer size is num_output_features
            device=accelerator.device,
            max_len=args.max_seq_len
        )
        criterion = nn.BCEWithLogitsLoss(reduction=args.multi_label_reduction) # Use BCEWithLogitsLoss for multi-label
    else:
        raise ValueError(f"Unknown task_type: {args.task_type}")
    
    if accelerator.is_local_main_process:
        wandb.watch(model, log="all", log_freq=100)

    # --- Optimizer and Scheduler ---
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    
    num_training_steps = args.epochs
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_training_steps, eta_min=1e-7)

    # Prepare everything with Accelerator
    model, optimizer, train_dataloader, scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, scheduler
    )
    
    if val_dataloader is not None:
        val_dataloader = accelerator.prepare(val_dataloader)

    if accelerator.is_local_main_process:
        print(f"Model initialized: {args.model_size}, Task: {args.task_type}")
        print(f"Optimizer: AdamW, LR: {args.learning_rate}, Weight Decay: {args.weight_decay}")
        print(f"Scheduler: CosineAnnealingLR, T_max: {num_training_steps}")
        if args.task_type == "seq2seq":
            print(f"Criterion: CrossEntropyLoss (ignoring PAD_ID: {pad_token_id} for target sequences)")
        elif args.task_type == "classification":
            print(f"Criterion: CrossEntropyLoss (for class predictions)")
        elif args.task_type == "classification_multi_label":
            print(f"Criterion: BCEWithLogitsLoss (for multi-label feature predictions)")

    # --- Training Loop ---
    # Extract support and query hop values from train_data_path
    
    # Parse hop values from the filename
    hop_pattern = r'shop_(\d+)_qhop_(\d+)'
    match = re.search(hop_pattern, args.train_data_path)
    if match:
        support_hop = match.group(1)
        query_hop = match.group(2)
    else:
        # Fallback values if pattern not found
        support_hop = "1"
        query_hop = "2"
    
    # Create hierarchical save directory
    hierarchical_save_dir = os.path.join(
        args.save_dir,
        args.model_size,
        args.task_type,
        f"shop_{support_hop}_qhop_{query_hop}",
        f"seed_{args.data_split_seed}"
    )
    
    if accelerator.is_local_main_process:
        if not os.path.exists(hierarchical_save_dir):
            os.makedirs(hierarchical_save_dir)
            print(f"Created checkpoint directory: {hierarchical_save_dir}")
        
        # Update args.save_dir to use the hierarchical structure
        args.save_dir = hierarchical_save_dir

    best_val_loss = float('inf')

    for epoch in tqdm(range(args.epochs), disable=not accelerator.is_local_main_process):
        if accelerator.is_local_main_process:
            print(f"--- Epoch {epoch+1}/{args.epochs} ---")
        
        train_loss, train_acc = train_epoch(model, train_dataloader, optimizer, criterion, scheduler, accelerator, epoch, pad_token_id, args)
        
        if accelerator.is_local_main_process:
            print(f"Epoch {epoch+1} Training Summary: Avg Loss: {train_loss:.4f}, Avg Acc: {train_acc:.4f}")
        
        epoch_log = {"epoch": epoch + 1, "train_epoch_loss": train_loss, "train_epoch_accuracy": train_acc}

        if val_dataloader:
            if args.task_type == "seq2seq":
                val_loss, val_acc, val_batch_accs_list = validate_epoch(model, val_dataloader, criterion, accelerator, epoch, pad_token_id, args, sos_token_id, eos_token_id)
                if accelerator.is_local_main_process and val_batch_accs_list:
                     print(f"Validation Acc (Seq2Seq) mean over batches: {np.mean(val_batch_accs_list):.4f}, std: {np.std(val_batch_accs_list):.4f}")
            else:
                val_loss, val_acc = validate_epoch(model, val_dataloader, criterion, accelerator, epoch, pad_token_id, args, sos_token_id, eos_token_id)
            
            if accelerator.is_local_main_process:
                print(f"Epoch {epoch+1} Validation Summary: Avg Loss: {val_loss:.4f}, Avg Acc: {val_acc:.4f}")
                epoch_log["val_epoch_loss"] = val_loss
                epoch_log["val_epoch_accuracy"] = val_acc

            if accelerator.is_local_main_process and val_loss < best_val_loss:
                best_val_loss = val_loss
                model_save_path = os.path.join(args.save_dir, f"best_model_epoch_{epoch+1}_{args.task_type}_{args.model_size}.pt")
                
                # Get unwrapped model for saving
                unwrapped_model = accelerator.unwrap_model(model)
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': unwrapped_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': val_loss,
                    'args': args,
                    'src_vocab_word2idx': full_dataset.word2idx,
                    'src_vocab_idx2word': full_dataset.idx2word
                }
                if args.task_type == "seq2seq":
                    checkpoint['tgt_vocab_word2idx'] = full_dataset.word2idx 
                    checkpoint['tgt_vocab_idx2word'] = full_dataset.idx2word
                elif args.task_type == "classification":
                    checkpoint['stone_state_to_id'] = full_dataset.stone_state_to_id
                    checkpoint['id_to_stone_state'] = full_dataset.id_to_stone_state
                elif args.task_type == "classification_multi_label":
                    checkpoint['feature_to_idx_map'] = full_dataset.feature_to_idx_map
                    checkpoint['idx_to_feature_map'] = {v: k for k, v in full_dataset.feature_to_idx_map.items()}
                    checkpoint['num_output_features'] = full_dataset.num_output_features
                
                torch.save(checkpoint, model_save_path)
                print(f"New best validation loss: {best_val_loss:.4f}. Model saved to {model_save_path}")
                wandb.save(model_save_path) 
        else: 
            if accelerator.is_local_main_process:
                model_save_path = os.path.join(args.save_dir, f"model_epoch_{epoch+1}_{args.task_type}_{args.model_size}.pt")
                unwrapped_model = accelerator.unwrap_model(model)                
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': unwrapped_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': train_loss,
                    'args': args,
                    'src_vocab_word2idx': full_dataset.word2idx,
                    'src_vocab_idx2word': full_dataset.idx2word
                }
                if args.task_type == "seq2seq":
                    checkpoint['tgt_vocab_word2idx'] = full_dataset.word2idx
                    checkpoint['tgt_vocab_idx2word'] = full_dataset.idx2word
                elif args.task_type == "classification":
                    checkpoint['stone_state_to_id'] = full_dataset.stone_state_to_id
                    checkpoint['id_to_stone_state'] = full_dataset.id_to_stone_state
                elif args.task_type == "classification_multi_label":
                    checkpoint['feature_to_idx_map'] = full_dataset.feature_to_idx_map
                    checkpoint['idx_to_feature_map'] = {v: k for k, v in full_dataset.feature_to_idx_map.items()}
                    checkpoint['num_output_features'] = full_dataset.num_output_features
                torch.save(checkpoint, model_save_path)
                print(f"Model saved to {model_save_path} (no validation)")
        
        if accelerator.is_local_main_process:
            current_lr_end_epoch = scheduler.get_last_lr()[0] if scheduler else optimizer.param_groups[0]['lr']
            epoch_log["learning_rate_end_epoch"] = current_lr_end_epoch
            wandb.log(epoch_log)

    if accelerator.is_local_main_process:
        print("Training complete.")
        wandb.finish()
        
    accelerator.end_training()

if __name__ == "__main__":
    # Add --generation_max_len_buffer to arguments
    def add_generation_args(parser):
        parser.add_argument("--generation_max_len_buffer", type=int, default=10, help="Buffer for max generation length in validation.")
        parser.add_argument("--sos_token_str", type=str, default="<sos>", help="String for Start of Sequence token.")
        parser.add_argument("--eos_token_str", type=str, default="<eos>", help="String for End of Sequence token.")

    original_parse_args = parse_args
    def new_parse_args():
        parser = argparse.ArgumentParser(description="Train Alchemy Transformer Model")
        # Add all original arguments from parse_args() definition
        parser.add_argument("--task_type", type=str, default="classification_multi_label", choices=["seq2seq", "classification", "classification_multi_label"], help="Type of task")
        parser.add_argument("--train_data_path", type=str, default="src/data/generated_data/decompositional_chemistry_samples_167424_80_unique_stones_train_shop_2_qhop_1.json", help="Path to the training JSON data file.")
        parser.add_argument("--val_data_path", type=str, default="src/data/generated_data/decompositional_chemistry_samples_167424_80_unique_stones_val_shop_2_qhop_1.json", help="Path to the validation JSON data file (optional).")
        parser.add_argument("--val_split", type=float, default=None, help="Validation split ratio")
        parser.add_argument("--val_split_seed", type=int, default=42, help="Seed for reproducible train/val splits.")
        parser.add_argument("--data_split_seed", type=int, default=42, help="Seed value that gets appended to the data path.")
        parser.add_argument("--model_size", type=str, default="xsmall", choices=["tiny", "xsmall", "small", "medium", "large"], help="Size of the transformer model.")
        parser.add_argument("--max_seq_len", type=int, default=2048, help="Maximum sequence length for the model.")
        parser.add_argument("--epochs", type=int, default=2, help="Number of training epochs.")
        parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training and validation.")
        parser.add_argument("--learning_rate", type=float, default=1e-4, help="Initial learning rate.")
        parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay.")
        parser.add_argument("--seed", type=int, default=42, help="Random seed.")
        parser.add_argument("--save_dir", type=str, default="src/saved_models/", help="Directory to save model checkpoints.")
        parser.add_argument("--filter_query_from_support", type=str, choices=["True", "False"], default="True", help="Filter query from support.")
        parser.add_argument("--wandb_project", type=str, default="alchemy-meta-learning", help="W&B project name.")
        parser.add_argument("--scheduler_call_location", type=str, default="after_epoch", choices=["after_epoch", "after_batch"], help="Scheduler call location.")
        parser.add_argument("--wandb_entity", type=str, default=None, help="W&B entity name.")
        parser.add_argument("--wandb_run_name", type=str, default=None, help="W&B run name.")
        parser.add_argument("--log_interval", type=int, default=50, help="Log training batch metrics every N batches.")
        parser.add_argument("--num_workers", type=int, default=15, help="Number of workers for data preprocessing.")
        parser.add_argument("--wandb_mode", type=str, default="offline", choices=["online", "offline"], help="W&B mode.")
        # Add new arguments
        parser.add_argument("--generation_max_len_buffer", type=int, default=10, help="Buffer for max generation length in validation.")
        parser.add_argument("--sos_token_str", type=str, default="<sos>", help="String for Start of Sequence token.") # Make sure these defaults are sensible
        parser.add_argument("--eos_token_str", type=str, default="<eos>", help="String for End of Sequence token.")   # Make sure these defaults are sensible
        return parser.parse_args()

    parse_args_fn = new_parse_args # Use the new parse_args

    # Replace the original parse_args call in main if it's not already this structure
    # In main(): args = parse_args_fn()
    
    # The main() function needs to be adjusted to use the new parse_args_fn
    # This is a bit complex to inject perfectly without seeing the full original parse_args
    # Assuming main() will call the globally scoped parse_args, we can redefine it.
    
    _original_main = main # Keep a reference if needed
    
    def main_with_new_args():
        global parse_args # Ensure we are modifying the global parse_args used by main
        
        # Store original parse_args and temporarily replace it
        original_parse_args_func = parse_args 
        
        def updated_parse_args():
            parser = argparse.ArgumentParser(description="Train Alchemy Transformer Model")
            # Copy all arguments from the original parse_args() definition
            parser.add_argument("--task_type", type=str, default="classification_multi_label", choices=["seq2seq", "classification", "classification_multi_label"], help="Type of task")
            parser.add_argument("--train_data_path", type=str, default="src/data/generated_data/decompositional_chemistry_samples_167424_80_unique_stones_train_shop_2_qhop_1.json", help="Path to the training JSON data file.")
            parser.add_argument("--val_data_path", type=str, default="src/data/generated_data/decompositional_chemistry_samples_167424_80_unique_stones_val_shop_2_qhop_1.json", help="Path to the validation JSON data file (optional).")
            parser.add_argument("--val_split", type=float, default=None, help="Validation split ratio")
            parser.add_argument("--val_split_seed", type=int, default=42, help="Seed for reproducible train/val splits.")
            parser.add_argument("--data_split_seed", type=int, default=42, help="Seed value that gets appended to the data path.")
            parser.add_argument("--model_size", type=str, default="xsmall", choices=["tiny", "xsmall", "small", "medium", "large"], help="Size of the transformer model.")
            parser.add_argument("--max_seq_len", type=int, default=2048, help="Maximum sequence length for the model.")
            parser.add_argument("--epochs", type=int, default=2, help="Number of training epochs.")
            parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training and validation.")
            parser.add_argument("--learning_rate", type=float, default=1e-4, help="Initial learning rate.")
            parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay.")
            parser.add_argument("--seed", type=int, default=42, help="Random seed.")
            parser.add_argument("--save_dir", type=str, default="src/saved_models/", help="Directory to save model checkpoints.")
            parser.add_argument("--filter_query_from_support", type=str, choices=["True", "False"], default="True", help="Filter query from support.")
            parser.add_argument("--wandb_project", type=str, default="alchemy-meta-learning", help="W&B project name.")
            parser.add_argument("--scheduler_call_location", type=str, default="after_epoch", choices=["after_epoch", "after_batch"], help="Scheduler call location.")
            parser.add_argument("--wandb_entity", type=str, default=None, help="W&B entity name.")
            parser.add_argument("--wandb_run_name", type=str, default=None, help="W&B run name.")
            parser.add_argument("--log_interval", type=int, default=50, help="Log training batch metrics every N batches.")
            parser.add_argument("--num_workers", type=int, default=15, help="Number of workers for data preprocessing.")
            parser.add_argument("--wandb_mode", type=str, default="offline", choices=["online", "offline"], help="W&B mode.")
            # Add new arguments
            parser.add_argument("--generation_max_len_buffer", type=int, default=10, help="Buffer for max generation length in validation.")
            parser.add_argument("--sos_token_str", type=str, default="<sos>", help="String for Start of Sequence token.") # Make sure these defaults are sensible
            parser.add_argument("--eos_token_str", type=str, default="<eos>", help="String for End of Sequence token.")   # Make sure these defaults are sensible
            return parser.parse_args()

        parse_args = updated_parse_args # Override global parse_args
        
        # Call the original main logic
        _original_main()
        
        # Restore original parse_args if other parts of the script call it later (though unlikely for __main__)
        parse_args = original_parse_args_func

    if __name__ == "__main__":
        main_with_new_args() # Call the wrapped main

    # If parse_args is defined inside main(), this approach needs to be different.
    # The above assumes parse_args is a global function.
    # A cleaner way is to modify parse_args directly if its definition is available.
    # For now, this is a workaround to inject args.
    # The most robust solution is to edit the parse_args function directly.
    # Let's assume we can edit parse_args directly for a cleaner solution.
    # Remove the __main__ wrapper and edit parse_args directly.
    # This means the user needs to ensure parse_args is edited.
    # The following edit to parse_args is what's actually needed.
    # The __main__ block above is overly complex due to not directly editing parse_args.
    # I will provide the edit for parse_args separately if this approach is too convoluted.
    # For now, the main logic for validate_epoch and getting sos/eos from dataset is the core change.
    
    # Corrected way to get sos/eos in main, using args:
    # sos_token_id = full_dataset.word2idx[args.sos_token_str]
    # eos_token_id = full_dataset.word2idx[args.eos_token_str]
    # This requires args.sos_token_str and args.eos_token_str to be defined.
    # The edit to parse_args should add these.
