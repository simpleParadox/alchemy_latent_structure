import argparse
import os
import math
import random
import numpy as np
from functools import partial
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import wandb
from tqdm import tqdm

# Assuming train.py is in src/models/, so data_loaders and models are siblings
from data_loaders import AlchemyDataset, collate_fn
from models import create_transformer_model, create_classifier_model # Added create_classifier_model

def set_seed(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)

def parse_args():
    parser = argparse.ArgumentParser(description="Train Alchemy Transformer Model")
    parser.add_argument("--task_type", type=str, default="classification", choices=["seq2seq", "classification"],
                        help="Type of task: 'seq2seq' for feature-wise prediction or 'classification' for whole state prediction.")
    parser.add_argument("--train_data_path", type=str, default="/home/rsaha/projects/dm_alchemy/src/data/generated_data/compositional_chemistry_samples_train_shop_1_qhop_1.json",
                        help="Path to the training JSON data file.")
    parser.add_argument("--val_data_path", type=str, default="/home/rsaha/projects/dm_alchemy/src/data/generated_data/compositional_chemistry_samples_val_shop_1_qhop_1.json",
                        help="Path to the validation JSON data file (optional).")
    parser.add_argument("--model_size", type=str, default="tiny", choices=["tiny", "xsmall", "small", "medium", "large"],
                        help="Size of the transformer model.")
    parser.add_argument("--max_seq_len", type=int, default=2048, # Max length for support + query + separators
                        help="Maximum sequence length for the model.")
    parser.add_argument("--epochs", type=int, default=2,
                        help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for training and validation.")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                        help="Initial learning rate for AdamW optimizer.")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="Weight decay for AdamW optimizer.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility.")
    parser.add_argument("--save_dir", type=str, default="./checkpoints",
                        help="Directory to save model checkpoints.")
    parser.add_argument("--filter_query_from_support", action="store_true",
                        help="Filter out query examples from support sets to prevent data leakage when support_steps=query_steps=1.", default=True)
    parser.add_argument("--wandb_project", type=str, default="alchemy-meta-learning",
                        help="Weights & Biases project name.")
    parser.add_argument("--wandb_entity", type=str, default=None, # Replace with your W&B entity
                        help="Weights & Biases entity name.")
    parser.add_argument("--wandb_run_name", type=str, default=None,
                        help="Weights & Biases run name (optional). Defaults to a generated name.")
    parser.add_argument("--log_interval", type=int, default=50,
                        help="Log training batch metrics every N batches.")
    return parser.parse_args()

def calculate_accuracy_seq2seq(predictions, targets, ignore_index):
    """
    Calculates accuracy for seq2seq task, ignoring padding tokens.
    predictions: (batch_size, seq_len, vocab_size) - model logits
    targets: (batch_size, seq_len) - ground truth token ids
    ignore_index: token id for padding, to be ignored in accuracy calculation
    """
    predicted_tokens = predictions.argmax(dim=-1) # (batch_size, seq_len)
    mask = (targets != ignore_index)
    
    correct_predictions = ((predicted_tokens == targets) & mask).sum().item()
    num_tokens_to_consider = mask.sum().item()
    
    accuracy = correct_predictions / num_tokens_to_consider if num_tokens_to_consider > 0 else 0.0
    return accuracy, correct_predictions, num_tokens_to_consider

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

def train_epoch(model, dataloader, optimizer, criterion, scheduler, device, epoch_num, pad_token_id, args):
    model.train()
    total_loss = 0
    total_correct_preds = 0
    total_considered_items = 0 # Can be tokens (seq2seq) or samples (classification)
    start_time = time.time()

    for batch_idx, batch in enumerate(tqdm(dataloader)):
        encoder_input_ids = batch["encoder_input_ids"].to(device)
        
        optimizer.zero_grad()
        
        if args.task_type == "seq2seq":
            decoder_input_ids = batch["decoder_input_ids"].to(device)
            decoder_target_ids = batch["decoder_target_ids"].to(device)
            output_logits = model(encoder_input_ids, decoder_input_ids)
            loss = criterion(output_logits.view(-1, output_logits.shape[-1]), decoder_target_ids.view(-1))
            acc, correct, considered = calculate_accuracy_seq2seq(output_logits, decoder_target_ids, pad_token_id)
        elif args.task_type == "classification":
            target_class_ids = batch["target_class_id"].to(device)
            # For classification, generate src_padding_mask if model expects it
            # Assuming pad_token_id is consistent for encoder inputs
            src_padding_mask = (encoder_input_ids == pad_token_id) 
            output_logits = model(encoder_input_ids, src_padding_mask=src_padding_mask) # (batch_size, num_classes)
            loss = criterion(output_logits, target_class_ids) # CrossEntropyLoss expects (N, C) and (N)
            acc, correct, considered = calculate_accuracy_classification(output_logits, target_class_ids)
        else:
            raise ValueError(f"Unknown task_type: {args.task_type}")

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if args.task_type == "seq2seq": # Scheduler might be per step or per epoch depending on setup
            scheduler.step() 

        total_loss += loss.item()
        total_correct_preds += correct
        total_considered_items += considered

        if batch_idx % args.log_interval == 0:
            current_lr = scheduler.get_last_lr()[0] if scheduler else optimizer.param_groups[0]['lr']
            elapsed_time = time.time() - start_time
            print(f"Epoch {epoch_num+1} | Batch {batch_idx}/{len(dataloader)} | Loss: {loss.item():.4f} | Train Acc: {acc:.4f} | LR: {current_lr:.2e} | Time: {elapsed_time:.2f}s")
            wandb.log({
                "train_batch_loss": loss.item(),
                "train_batch_accuracy": acc,
                "learning_rate": current_lr,
                "epoch": epoch_num,
                "batch_idx": batch_idx
            })
            start_time = time.time()
    
    if args.task_type == "classification": # If scheduler is per epoch for classification
        if scheduler: scheduler.step()

    avg_epoch_loss = total_loss / len(dataloader)
    avg_epoch_accuracy = total_correct_preds / total_considered_items if total_considered_items > 0 else 0.0
    return avg_epoch_loss, avg_epoch_accuracy

def validate_epoch(model, dataloader, criterion, device, epoch_num, pad_token_id, args):
    if dataloader is None:
        return None, None, None if args.task_type == "seq2seq" else None # Adjusted for classification return

    model.eval()
    total_loss = 0
    total_correct_preds = 0
    total_considered_items = 0
    all_accs = [] if args.task_type == "seq2seq" else None # all_accs might not be relevant for classification like this
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader)):
            encoder_input_ids = batch["encoder_input_ids"].to(device)

            if args.task_type == "seq2seq":
                decoder_input_ids = batch["decoder_input_ids"].to(device)
                decoder_target_ids = batch["decoder_target_ids"].to(device)
                output_logits = model(encoder_input_ids, decoder_input_ids)
                loss = criterion(output_logits.view(-1, output_logits.shape[-1]), decoder_target_ids.view(-1))
                acc, correct, considered = calculate_accuracy_seq2seq(output_logits, decoder_target_ids, pad_token_id)
                if all_accs is not None: all_accs.append(acc)
            elif args.task_type == "classification":
                target_class_ids = batch["target_class_id"].to(device)
                src_padding_mask = (encoder_input_ids == pad_token_id)
                output_logits = model(encoder_input_ids, src_padding_mask=src_padding_mask)
                loss = criterion(output_logits, target_class_ids)
                acc, correct, considered = calculate_accuracy_classification(output_logits, target_class_ids)
                # For classification, `acc` is already the batch accuracy.
                # If we want a list of batch accuracies, we can append `acc`.
                # For now, we just calculate the overall average.
            else:
                raise ValueError(f"Unknown task_type: {args.task_type}")

            total_loss += loss.item()
            total_correct_preds += correct
            total_considered_items += considered
            
    avg_epoch_loss = total_loss / len(dataloader)
    avg_epoch_accuracy = total_correct_preds / total_considered_items if total_considered_items > 0 else 0.0
    
    if args.task_type == "seq2seq":
        return avg_epoch_loss, avg_epoch_accuracy, all_accs # all_accs is specific to seq2seq here
    else: # classification
        return avg_epoch_loss, avg_epoch_accuracy # No third element like all_accs needed for now

def main():
    args = parse_args()
    set_seed(args.seed)

    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_run_name if args.wandb_run_name else f"{args.task_type}_{args.model_size}_{time.strftime('%Y%m%d-%H%M%S')}",
        config=vars(args),
        mode="offline" # Consider changing to "online" or "disabled" as needed
    )

    device = 'cpu' # torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Selected task type: {args.task_type}")

    # --- Dataset and DataLoader ---
    print(f"Loading training data from: {args.train_data_path}")
    # Pass task_type to AlchemyDataset
    train_dataset = AlchemyDataset(
        json_file_path=args.train_data_path, 
        task_type=args.task_type,
        filter_query_from_support=args.filter_query_from_support
    )
    
    pad_token_id = train_dataset.pad_token_id # Used for encoder input padding in both tasks
    src_vocab_size = len(train_dataset.word2idx) # For encoder features/potions
    print(f"Source (Feature/Potion) Vocabulary size: {src_vocab_size}")
    print(f"Pad token ID for encoder inputs: {pad_token_id}")

    if args.task_type == "seq2seq":
        sos_token_id = train_dataset.sos_token_id
        eos_token_id = train_dataset.eos_token_id
        tgt_vocab_size = src_vocab_size # Assuming shared vocab for seq2seq target features
        print(f"Target (Feature) Vocabulary size (for Seq2Seq): {tgt_vocab_size}")
        print(f"SOS ID: {sos_token_id}, EOS ID: {eos_token_id}")
    elif args.task_type == "classification":
        num_classes = len(train_dataset.stone_state_to_id)
        print(f"Number of classes (Stone States for Classification): {num_classes}")
        # We don't need sos/eos for classification targets

    # Pass task_type to collate_fn
    custom_collate_train = partial(collate_fn, pad_token_id=pad_token_id, task_type=args.task_type)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=custom_collate_train,
        num_workers=4 # Adjust based on your system
    )

    val_dataloader = None
    if args.val_data_path:
        print(f"Loading validation data from: {args.val_data_path}")
        val_dataset = AlchemyDataset(
            json_file_path=args.val_data_path,
            task_type=args.task_type,
            vocab_word2idx=train_dataset.word2idx, # Use train vocab for features/potions
            vocab_idx2word=train_dataset.idx2word,
            stone_state_to_id=train_dataset.stone_state_to_id if args.task_type == "classification" else None, # Use train stone state mapping
            filter_query_from_support=args.filter_query_from_support
        )
        custom_collate_val = partial(collate_fn, pad_token_id=pad_token_id, task_type=args.task_type)
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=custom_collate_val,
            num_workers=4 # Adjust based on your system
        )
    else:
        print("No validation data path provided. Skipping validation.")

    # --- Model ---
    if args.task_type == "seq2seq":
        model = create_transformer_model(
            config_name=args.model_size,
            src_vocab_size=src_vocab_size,
            tgt_vocab_size=tgt_vocab_size, # Defined above for seq2seq
            device=device
        )
        criterion = nn.CrossEntropyLoss(ignore_index=pad_token_id) # pad_token_id is for seq2seq targets
    elif args.task_type == "classification":
        model = create_classifier_model(
            config_name=args.model_size,
            src_vocab_size=src_vocab_size,
            num_classes=num_classes, # Defined above for classification
            device=device,
            max_len=args.max_seq_len # Pass max_seq_len for positional encoding in classifier
        )
        criterion = nn.CrossEntropyLoss() # No ignore_index for classification targets (target is class ID)
    else:
        raise ValueError(f"Unknown task_type: {args.task_type}")

    wandb.watch(model, log="all", log_freq=100) # Log model gradients and parameters

    # --- Optimizer, Criterion, Scheduler ---
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    
    # Scheduler setup - CosineAnnealingLR is often per step for Transformers
    # For classification, it might be per epoch. Let's assume per step for now for consistency,
    # but adjust train_epoch if scheduler.step() should be per epoch for classification.
    num_training_steps = args.epochs * len(train_dataloader)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_training_steps, eta_min=1e-7)
    # If using a scheduler that updates per epoch (e.g. StepLR), move scheduler.step() in train_epoch accordingly.

    print(f"Model initialized: {args.model_size}, Task: {args.task_type}")
    print(f"Optimizer: AdamW, LR: {args.learning_rate}, Weight Decay: {args.weight_decay}")
    print(f"Scheduler: CosineAnnealingLR, T_max: {num_training_steps}")
    if args.task_type == "seq2seq":
        print(f"Criterion: CrossEntropyLoss (ignoring PAD_ID: {pad_token_id} for target sequences)")
    else:
        print(f"Criterion: CrossEntropyLoss (for class predictions)")

    # --- Training Loop ---
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
        print(f"Created checkpoint directory: {args.save_dir}")

    best_val_loss = float('inf')

    for epoch in tqdm(range(args.epochs)):
        print(f"--- Epoch {epoch+1}/{args.epochs} ---")
        # Training model by calling train_epoch.
        train_loss, train_acc = train_epoch(model, train_dataloader, optimizer, criterion, scheduler, device, epoch, pad_token_id, args)
        print(f"Epoch {epoch+1} Training Summary: Avg Loss: {train_loss:.4f}, Avg Acc: {train_acc:.4f}")
        
        epoch_log = {"epoch": epoch + 1, "train_epoch_loss": train_loss, "train_epoch_accuracy": train_acc}

        if val_dataloader:
            if args.task_type == "seq2seq":
                val_loss, val_acc, val_batch_accs_list = validate_epoch(model, val_dataloader, criterion, device, epoch, pad_token_id, args)
                if val_batch_accs_list: # Can be None if dataloader was None initially
                     print(f"Validation Acc (Seq2Seq) mean over batches: {np.mean(val_batch_accs_list):.4f}, std: {np.std(val_batch_accs_list):.4f}")
            else: # classification
                # val_loss, val_acc = validate_epoch(model, val_dataloader, criterion, device, epoch, pad_token_id, args)
                pass
            
            print(f"Epoch {epoch+1} Validation Summary: Avg Loss: {val_loss:.4f}, Avg Acc: {val_acc:.4f}")
            epoch_log["val_epoch_loss"] = val_loss
            epoch_log["val_epoch_accuracy"] = val_acc

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                # Save logic needs to be aware of task_type if vocab/args differ significantly for checkpoint loading
                model_save_path = os.path.join(args.save_dir, f"best_model_{args.task_type}_{args.model_size}.pt")
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': val_loss,
                    'args': args, # Save args to know task_type, model_size etc. on load
                    'src_vocab_word2idx': train_dataset.word2idx, # For encoder
                    'src_vocab_idx2word': train_dataset.idx2word  # For encoder
                }
                if args.task_type == "seq2seq":
                    # For seq2seq, target vocab is same as source, but good to be explicit if it could change
                    checkpoint['tgt_vocab_word2idx'] = train_dataset.word2idx 
                    checkpoint['tgt_vocab_idx2word'] = train_dataset.idx2word
                elif args.task_type == "classification":
                    checkpoint['stone_state_to_id'] = train_dataset.stone_state_to_id
                    checkpoint['id_to_stone_state'] = train_dataset.id_to_stone_state
                
                torch.save(checkpoint, model_save_path)
                print(f"New best validation loss: {best_val_loss:.4f}. Model saved to {model_save_path}")
                wandb.save(model_save_path) 
        else: 
            model_save_path = os.path.join(args.save_dir, f"model_epoch_{epoch+1}_{args.task_type}_{args.model_size}.pt")
            # Similar save logic as above for consistency
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': train_loss, 
                'args': args,
                'src_vocab_word2idx': train_dataset.word2idx,
                'src_vocab_idx2word': train_dataset.idx2word
            }
            if args.task_type == "seq2seq":
                checkpoint['tgt_vocab_word2idx'] = train_dataset.word2idx
                checkpoint['tgt_vocab_idx2word'] = train_dataset.idx2word
            elif args.task_type == "classification":
                checkpoint['stone_state_to_id'] = train_dataset.stone_state_to_id
                checkpoint['id_to_stone_state'] = train_dataset.id_to_stone_state
            torch.save(checkpoint, model_save_path)
            print(f"Model saved to {model_save_path} (no validation)")
        
        current_lr_end_epoch = scheduler.get_last_lr()[0] if scheduler else optimizer.param_groups[0]['lr']
        epoch_log["learning_rate_end_epoch"] = current_lr_end_epoch
        wandb.log(epoch_log)

    print("Training complete.")
    wandb.finish()

if __name__ == "__main__":
    main()
