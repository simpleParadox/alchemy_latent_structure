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

def set_seed(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)

def parse_args():
    
    parser = argparse.ArgumentParser(description="Train Alchemy Transformer Model")
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
            decoder_input_ids = batch["decoder_input_ids"]
            decoder_target_ids = batch["decoder_target_ids"]
            output_logits = model(encoder_input_ids, decoder_input_ids)
            loss = criterion(output_logits.view(-1, output_logits.shape[-1]), decoder_target_ids.view(-1))
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
    if scheduler: scheduler.step()

    avg_epoch_loss = total_loss / len(dataloader)
    avg_epoch_accuracy = total_correct_preds / total_considered_items if total_considered_items > 0 else 0.0
    return avg_epoch_loss, avg_epoch_accuracy

def validate_epoch(model, dataloader, criterion, accelerator, epoch_num, pad_token_id, args):
    if dataloader is None:
        return None, None, None if args.task_type == "seq2seq" else None # Adjusted for classification return

    model.eval()
    total_loss = 0
    total_correct_preds = 0
    total_considered_items = 0
    all_accs = [] if args.task_type == "seq2seq" else None 
    with torch.no_grad():
        pbar = tqdm(dataloader, disable=not accelerator.is_local_main_process)
        for batch_idx, batch in enumerate(pbar):
            encoder_input_ids = batch["encoder_input_ids"]

            if args.task_type == "seq2seq":
                decoder_input_ids = batch["decoder_input_ids"]
                decoder_target_ids = batch["decoder_target_ids"]
                output_logits = model(encoder_input_ids, decoder_input_ids)
                loss = criterion(output_logits.view(-1, output_logits.shape[-1]), decoder_target_ids.view(-1))
                acc, correct, considered = calculate_accuracy_seq2seq(output_logits, decoder_target_ids, pad_token_id)
                if all_accs is not None: all_accs.append(acc)
            elif args.task_type == "classification":
                target_class_ids = batch["target_class_id"]
                src_padding_mask = (encoder_input_ids == pad_token_id)
                output_logits = model(encoder_input_ids, src_padding_mask=src_padding_mask)
                loss = criterion(output_logits, target_class_ids)
                acc, correct, considered = calculate_accuracy_classification(output_logits, target_class_ids)
            elif args.task_type == "classification_multi_label":
                target_feature_vector = batch["target_feature_vector"]
                src_padding_mask = (encoder_input_ids == pad_token_id)
                output_logits = model(encoder_input_ids, src_padding_mask=src_padding_mask)
                loss = criterion(output_logits, target_feature_vector.float())
                acc, correct, considered = calculate_accuracy_multilabel(output_logits, target_feature_vector)
            else:
                raise ValueError(f"Unknown task_type: {args.task_type}")

            total_loss += loss.item()
            total_correct_preds += correct
            total_considered_items += considered
            
    avg_epoch_loss = total_loss / len(dataloader)
    avg_epoch_accuracy = total_correct_preds / total_considered_items if total_considered_items > 0 else 0.0
    
    if args.task_type == "seq2seq":
        return avg_epoch_loss, avg_epoch_accuracy, all_accs # all_accs is specific to seq2seq here
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

    print(f"Source (Feature/Potion) Vocabulary size: {src_vocab_size}")
    print(f"Pad token ID for encoder inputs: {pad_token_id}")

    if args.task_type == "seq2seq":
        sos_token_id = full_dataset.sos_token_id
        eos_token_id = full_dataset.eos_token_id
        tgt_vocab_size = src_vocab_size
        print(f"Target (Feature) Vocabulary size (for Seq2Seq): {tgt_vocab_size}")
        print(f"SOS ID: {sos_token_id}, EOS ID: {eos_token_id}")
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
        criterion = nn.BCEWithLogitsLoss() # Use BCEWithLogitsLoss for multi-label
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
                val_loss, val_acc, val_batch_accs_list = validate_epoch(model, val_dataloader, criterion, accelerator, epoch, pad_token_id, args)
                if accelerator.is_local_main_process and val_batch_accs_list:
                     print(f"Validation Acc (Seq2Seq) mean over batches: {np.mean(val_batch_accs_list):.4f}, std: {np.std(val_batch_accs_list):.4f}")
            else:
                val_loss, val_acc = validate_epoch(model, val_dataloader, criterion, accelerator, epoch, pad_token_id, args)
            
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
    main()
