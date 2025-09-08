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
from models import create_transformer_model, create_classifier_model, create_decoder_classifier_model

def set_seed(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)

def parse_args():
    parser = argparse.ArgumentParser(description="Train Alchemy Transformer Model")
    parser.add_argument("--multi_label_reduction", type=str, default="mean", choices=["mean", "sum", 'none'],
                        help="Reduction method for multi-label classification: 'mean' or 'sum'. Default is 'mean'.")
    parser.add_argument("--task_type", type=str, default="classification", choices=["seq2seq", "classification", "classification_multi_label", "seq2seq_stone_state"],
                        help="Type of task: 'seq2seq' for feature-wise prediction, 'classification' for whole state prediction, or 'classification_multi_label' for multi-label feature prediction.")
    parser.add_argument("--train_data_path", type=str, default="src/data/held_out_exps_generated_data/compositional_chemistry_samples_167424_80_unique_stones_train_shop_1_qhop_1_single_held_out_color_1_edges_exp.json",
                        help="Path to the training JSON data file.")
    parser.add_argument("--val_data_path", type=str, default="src/data/held_out_exps_generated_data/compositional_chemistry_samples_167424_80_unique_stones_val_shop_1_qhop_1_single_held_out_color_1_edges_exp.json",
                        help="Path to the validation JSON data file (optional).")
    parser.add_argument("--val_split", type=float, default=None,
                        help="Validation split ratio (e.g., 0.1 for 10%%). If provided, validation set will be created from training data instead of loading separate file. Default is None.")
    parser.add_argument("--val_split_seed", type=int, default=42,
                        help="Seed for reproducible train/val splits.")
    parser.add_argument("--data_split_seed", type=int, default=0,
                        help="Seed value that gets appended to the data path to load the approapriate training / validation.")

    parser.add_argument("--model_size", type=str, default="xsmall", choices=["tiny", "xsmall", "xsmall_modified", "xsmall_deep", "small", "medium", "large"],
                        help="Size of the transformer model.")
    parser.add_argument("--model_architecture", type=str, default="encoder", choices=["encoder", "decoder"],
                        help="Model architecture: 'encoder' for encoder-only classifier, 'decoder' for decoder-only classifier.")
    parser.add_argument("--max_seq_len", type=int, default=2048, # Max length for support + query + separators
                        help="Maximum sequence length for the model.")
    parser.add_argument("--epochs", type=int, default=60,
                        help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=256,
                        help="Batch size for training and validation.")

    parser.add_argument("--learning_rate", type=float, default=1e-4,
                        help="Initial learning rate for AdamW optimizer.")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="Weight decay for the optimizer.")
    parser.add_argument("--optimizer", type=str, 
                        default='adamw', choices=['adam', 'adamw', 'rmsprop', 'adagrad'],
                        help="Optimizer to use: 'adam', 'adamw', 'rmsprop', or 'adafactor'. Default is 'adamw'.")
    # NOTE: pytorch 2.4 does not have Adafactor implemented. So not adding it here.


    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility.")
    parser.add_argument("--save_dir", type=str, default="src/saved_models/",
                        help="Directory to save model checkpoints.")
    parser.add_argument("--filter_query_from_support", type=str, choices=["True", "False"],
                        help="Filter out query examples from support sets to prevent data leakage when support_steps=query_steps=1. Default=True", default=True)
    parser.add_argument("--wandb_project", type=str, default="alchemy-meta-learning",
                        help="Weights & Biases project name.")
    parser.add_argument("--use_scheduler", type=str, default="True", choices=["True", "False"],
                        help="Use learning rate scheduler. Default is True.")

    parser.add_argument("--scheduler_type", type=str, default="cosine", 
                        choices=["cosine", "exponential", "cosine_restarts", "none"],
                        help="Type of learning rate scheduler: 'cosine', 'exponential', or 'none'.")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="Multiplicative factor for ExponentialLR scheduler.")
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
    parser.add_argument("--use_autoregressive_training", type=str, default="False", choices=["True", "False"],
                        help="Use autoregressive generation during training instead of teacher forcing. This makes training consistent with validation but may be slower. Default is False.")
    parser.add_argument("--store_predictions", type=str, default="True", choices=["True", "False"],
                        help="Store predictions during training and validation. Default is True.")
    
    # Add new preprocessing arguments
    parser.add_argument("--preprocessed_dir", type=str, default="src/data/preprocessed_separate_held_out_exps",
                        help="Directory to look for/store preprocessed data files.")
    parser.add_argument("--use_preprocessed", type=str, default="True", choices=["True", "False"],
                        help="Whether to use preprocessed data if available. Default is True.")
    
    parser.add_argument("--input_format", type=str, default='features', choices=["stone_states", "features"],
                        help="Input format: 'stone_states' for complete states as tokens, 'features' for individual features as tokens. Default inferred from task_type.")
    parser.add_argument("--output_format", type=str, default='stone_states', choices=["stone_states", "features"],
                        help="Output format: 'stone_states' for classification targets, 'features' for multi-hot vectors. Default inferred from task_type.")
    
    parser.add_argument("--save_checkpoints", type=str, default="False", choices=["True", "False"],
                        help="Whether to save model checkpoints during training. Default is True.")
    parser.add_argument("--is_held_out_color_exp", type=str, default="True", choices=["True", "False"],
                        help="Whether the dataset is a held-out color experiment. Default is True.")
    parser.add_argument("--prediction_type", type=str, default="default", choices=["default", "feature", "autoregressive"],
                        help="Type of prediction: 'default' for standard full stone state classification, 'feature' for feature-wise classification, 'autoregressive' for autoregressive generation.")

    parser.add_argument('--padding_side', type=str, default='right', choices=['left', 'right'],
                        help="Padding side for sequences: 'left' or 'right'. Default is 'right' for both encoder and decoder.")
    
    
    parser.add_argument("--override_num_classes", type=int, default=108,
                        help="Override the number of classes for classification tasks. If None, will use dataset's class count.")
    
    parser.add_argument("--pooling_strategy", type=str, default="global", choices=["global", "query_only"],
                        help="Pooling strategy for encoder-only models: 'global' for global average pooling, 'query_only' for pooling only over query tokens. Default is 'global'.")

    parser.add_argument("--use_truncation", type=str, default="False", choices=["True", "False"],
                        help="Whether to truncate sequences longer than max_seq_len. Default is True.")
    
    parser.add_argument("--fp16", type=str, default="False", choices=["True", "False"])

    # Add resume arguments
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                        help="Path to checkpoint file to resume training from.")
    parser.add_argument("--resume_wandb_run_id", type=str, default=None,
                        help="wandb run ID to resume. If provided, will continue the same wandb run.")
    parser.add_argument("--allow_data_path_mismatch", type=str, default="False", choices=["True", "False"],
                        help="Allow resuming even if data paths don't match checkpoint.")
    
    return parser.parse_args()

def calculate_accuracy_seq2seq(predictions, targets, pad_token_id, eos_token_id=None):
    """
    Calculates accuracy for seq2seq task, ignoring both padding and EOS tokens.
    """
    predicted_tokens = predictions.argmax(dim=-1)
    
    # Create mask for tokens we want to evaluate
    if eos_token_id is not None:
        # Ignore both padding and EOS tokens
        valid_token_mask = (targets != pad_token_id) & (targets != eos_token_id)
    else:
        # Only ignore padding tokens
        valid_token_mask = (targets != pad_token_id)
    
    token_correct = (predicted_tokens == targets)
    
    # For each sequence, check if all valid tokens are correct
    correct_sequences = (token_correct | ~valid_token_mask).all(dim=1)
    
    num_correct_sequences = correct_sequences.sum().item()
    num_sequences_in_batch = targets.size(0)
    
    accuracy = num_correct_sequences / num_sequences_in_batch if num_sequences_in_batch > 0 else 0.0
    # Need to check the generated accuracy as well.
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

def calculate_accuracy_generated_seq2seq(generated_ids, targets, pad_token_id):
    """
    Calculates accuracy for seq2seq task by comparing generated sequences with targets.
    
    Args:
        generated_ids: Generated token sequences, shape (batch_size, generated_seq_len)
        targets: Target token sequences, shape (batch_size, target_seq_len)
        pad_token_id: Token ID for padding, to be ignored in comparison
        
    Returns:
        accuracy: Sequence-level accuracy (fraction of exactly matching sequences)
        correct: Number of correctly generated sequences
        considered: Total number of sequences in batch
    """
    batch_size = targets.size(0)
    
    # Find the maximum length between generated and target sequences
    max_gen_len = generated_ids.size(1)
    max_tgt_len = targets.size(1)
    max_len = max(max_gen_len, max_tgt_len)
    
    # Pad both sequences to the same length
    if max_gen_len < max_len:
        # Pad generated sequences
        padding = torch.full((batch_size, max_len - max_gen_len), pad_token_id, 
                           dtype=generated_ids.dtype, device=generated_ids.device)
        generated_ids_padded = torch.cat([generated_ids, padding], dim=1)
    else:
        generated_ids_padded = generated_ids
    
    if max_tgt_len < max_len:
        # Pad target sequences
        padding = torch.full((batch_size, max_len - max_tgt_len), pad_token_id, 
                           dtype=targets.dtype, device=targets.device)
        targets_padded = torch.cat([targets, padding], dim=1)
    else:
        targets_padded = targets
    
    # Create mask for non-padding tokens in targets
    non_padding_mask = (targets_padded != pad_token_id)
    
    # Compare tokens element-wise
    token_matches = (generated_ids_padded == targets_padded)
    
    # For sequence to be correct, all non-padding positions must match
    # (token_matches | ~non_padding_mask) will be True for:
    # 1. Positions where tokens match, OR
    # 2. Positions where target is padding (we don't care about these)
    correct_sequences = (token_matches | ~non_padding_mask).all(dim=1)
    
    num_correct = correct_sequences.sum().item()
    accuracy = num_correct / batch_size if batch_size > 0 else 0.0
    
    return accuracy, num_correct, batch_size

def train_epoch(model, dataloader, optimizer, criterion, scheduler, accelerator, epoch_num, pad_token_id, args):
    model.train()
    total_loss = 0
    total_correct_preds = 0
    total_considered_items = 0 # Can be tokens (seq2seq) or samples (classification)
    start_time = time.time()

    pbar = tqdm(dataloader, disable=not accelerator.is_local_main_process)
    iters = len(dataloader)
    for batch_idx, batch in enumerate(pbar):
        # No need to move to device explicitly - Accelerate handles this
        encoder_input_ids = batch["encoder_input_ids"]
        
        optimizer.zero_grad()
        
        if args.task_type == "seq2seq":
            decoder_input_ids = batch["decoder_input_ids"]
            decoder_target_ids = batch["decoder_target_ids"]
            
            # Convert string arguments to boolean
            use_autoregressive = args.use_autoregressive_training == "True"
            
            if use_autoregressive:
                # Use autoregressive training (consistent with validation)
                # Get necessary token IDs
                sos_token_id = dataloader.dataset.sos_token_id if hasattr(dataloader.dataset, 'sos_token_id') else None
                eos_token_id = dataloader.dataset.eos_token_id if hasattr(dataloader.dataset, 'eos_token_id') else None
                
                # Get the unwrapped model to access the generate method
                unwrapped_model = accelerator.unwrap_model(model)
                
                # Generate sequences autoregressively
                max_target_len = decoder_target_ids.size(1) + 4
                generated_ids = unwrapped_model.generate(
                    src=encoder_input_ids,
                    start_symbol_id=sos_token_id,
                    end_symbol_id=eos_token_id,
                    max_len=max_target_len,
                    device=encoder_input_ids.device,
                    pad_token_id=pad_token_id
                )
                
                # Remove SOS token for comparison
                generated_ids = generated_ids[:, 1:]
                
                # Calculate loss based on token error rate (same as validation)
                batch_size = decoder_target_ids.size(0)
                min_len = min(generated_ids.size(1), decoder_target_ids.size(1))
                max_len = max(generated_ids.size(1), decoder_target_ids.size(1))

                # Pad both to the same length
                if generated_ids.size(1) < max_len:
                    padding = torch.full((batch_size, max_len - generated_ids.size(1)), pad_token_id, 
                                       dtype=generated_ids.dtype, device=generated_ids.device)
                    generated_padded = torch.cat([generated_ids, padding], dim=1)
                else:
                    generated_padded = generated_ids

                if decoder_target_ids.size(1) < max_len:
                    padding = torch.full((batch_size, max_len - decoder_target_ids.size(1)), pad_token_id, 
                                       dtype=decoder_target_ids.dtype, device=decoder_target_ids.device)
                    target_padded = torch.cat([decoder_target_ids, padding], dim=1)
                else:
                    target_padded = decoder_target_ids

                # Calculate token error rate
                valid_mask = (target_padded != pad_token_id)
                mismatches = (generated_padded != target_padded) & valid_mask
                total_valid_tokens = valid_mask.sum().float()

                if total_valid_tokens > 0:
                    token_error_rate = mismatches.sum().float() / total_valid_tokens
                else:
                    token_error_rate = torch.tensor(0.0, device=encoder_input_ids.device)

                # Convert to log scale for training stability
                epsilon = 1e-8
                loss = -torch.log(1.0 - token_error_rate + epsilon)
                
                # NOTE: Need to use the criterion for calculating loss so that backward pass works correctly.
                # And accelerator.backward(loss) will work correctly.
                
                # Calculate accuracy using generated sequences
                acc, correct, considered = calculate_accuracy_generated_seq2seq(generated_ids, decoder_target_ids, pad_token_id)
                
            else:
                # Use teacher forcing (original behavior)
                output_logits = model(encoder_input_ids, decoder_input_ids)
                
                # The criterion will automatically ignore pad_token_id because it was initialized with ignore_index.
                # Also, for the matrix shapes, look at Karpahty's video on implementing GPT from scratch.
                loss = criterion(output_logits.view(-1, output_logits.shape[-1]), decoder_target_ids.view(-1)) # Reshaping such that the output_logits is a 2d tensor of shape (batch_size * seq_len, vocab_size) and decoder_target_ids is a 1d tensor of shape (batch_size * seq_len).
                
                acc, correct, considered = calculate_accuracy_seq2seq(output_logits, decoder_target_ids, pad_token_id, eos_token_id=None)
            
        elif args.task_type == "classification":
            target_class_ids = batch["target_class_id"]
               
            # This is a design decision and is generally how it is done in the literature.
            src_padding_mask = (encoder_input_ids == pad_token_id) 
            
            # For decoder only model, the model will only take the last token to predict the class.
            output_logits = model(encoder_input_ids, src_padding_mask=src_padding_mask) # (batch_size, num_classes)
            
            # For decoder-only architecture, we need to use the encoder_input_ids as input
            # and treat the target_class_ids as the output.
            # The model is designed to handle both cases - encoder and decoder.
            # For both encoder and decoder, the loss is calculated on the prediction.
            # This is unlike how generally we would train a decoder-only model to do next token prediction (which is shifting the logits by 1).
            loss = criterion(output_logits, target_class_ids) # CrossEntropyLoss expects (N, C) and (N)
            acc, correct, considered = calculate_accuracy_classification(output_logits, target_class_ids)
            
            
            
        elif args.task_type == "classification_multi_label":
            
            if args.prediction_type == "autoregressive":
                # TODO: Needs testing.
                # We need to concatenate the encoder input ids and decoder input ids
                # But we will only calculate the loss on the output of the query example.
                # There will be some masking that's going to happen.
                decoder_input_ids = batch["decoder_input_ids"]
                encoder_input_ids = torch.cat([encoder_input_ids, decoder_input_ids], dim=1) 
            else:
                target_feature_vector = batch["target_feature_vector"] # (batch_size, num_output_features)
                src_padding_mask = (encoder_input_ids == pad_token_id)
            
            output_logits = model(encoder_input_ids, src_padding_mask=src_padding_mask) # (batch_size, num_output_features)
            
            
            # 1. Define the feature gorups.
            feature_groups = [(0, 3), (3, 6), (6, 9), (9, 13)]
            
            
            # 2. Initialize CrossEntropyLoss with ignore_index for group level multi-class classification.
            loss_criterion = nn.CrossEntropyLoss(ignore_index=pad_token_id) # Default reduction is 'mean'.
            
            total_loss = 0.0
            num_losses = 0
            
            all_groups_correct = torch.ones(target_feature_vector.size(0), dtype=torch.bool, device=target_feature_vector.device)  # Start with all samples considered correct, and then set 0 where they are not correct.

            for start_idx, end_idx in feature_groups:
                logits_group = output_logits[:, start_idx:end_idx]  # Get logits for the current group
                target_group_one_hot = target_feature_vector[:, start_idx:end_idx]  # Get corresponding target features for that group.
                
                # Convert one-hot to class indices
                target_group_indices = target_group_one_hot.argmax(dim=-1) 
                
                # Calculate loss for this group
                group_loss = loss_criterion(logits_group, target_group_indices)
                
                total_loss += group_loss
                num_losses += 1
               
                # --- Accuracy Calculation ---
                # Get predicted class index for this group
                predicted_indices_group = logits_group.argmax(dim=-1)
                # Check which samples are correct for this group
                correct_in_group = (predicted_indices_group == target_group_indices)
                # Update the overall correctness. A sample is only correct if it has been
                # correct for all previous groups AND is correct for the current one.
                all_groups_correct &= correct_in_group
            
            
                
            # Average the loss across all groups
            loss = total_loss # / num_losses
                
            # acc, correct, considered = calculate_accuracy_multilabel(output_logits, target_feature_vector)
            
            # Calculate accuracy based on the overall correctness across all groups
            correct = all_groups_correct.sum().item()
            considered = target_feature_vector.size(0)
            acc = correct / considered if considered > 0 else 0.0
            
           
            
        elif args.task_type == "seq2seq_stone_state":
            decoder_input_ids = batch["decoder_input_ids"]
            decoder_target_ids = batch["decoder_target_ids"]
            
            output_logits = model(encoder_input_ids, decoder_input_ids)
            
            # The criterion will automatically ignore pad_token_id because it was initialized with ignore_index.
            loss = criterion(output_logits.view(-1, output_logits.shape[-1]), decoder_target_ids.view(-1))
            
            acc, correct, considered = calculate_accuracy_seq2seq(output_logits, decoder_target_ids, pad_token_id, eos_token_id=None)
        else:
            raise ValueError(f"Unknown task_type: {args.task_type}")

        accelerator.backward(loss)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()


        if scheduler and args.scheduler_call_location == 'after_batch':
            # But for later versions of python, it might be possible to call scheduler.step() after the 
            # epoch and still have no cycles. NOTE: If you use a different pytorch version, please verify this behavior
            # as it may not hold.
            if args.scheduler_type == 'cosine':
                scheduler.step() # For python 2.4.x (which is the one being used here) this is where you call the scheduler. 
            elif args.scheduler_type == 'cosine_restarts':
                scheduler.step(epoch_num + batch_idx / iters) # For python 2.4.x (which is the one being used here) this is where you call the scheduler.

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
        
    
    if scheduler and args.scheduler_call_location == 'after_epoch':
        scheduler.step()
    # Call scheduler after each epoch for ExponentialLR
    # if scheduler and args.scheduler_type == "exponential":
    #     scheduler.step()

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
    sos_token_id = dataloader.dataset.sos_token_id if hasattr(dataloader.dataset, 'sos_token_id') else None
    
    # Initialize storage for predictions and targets if requested
    all_predictions = []
    all_targets = []
    all_encoder_inputs = [] if args.store_predictions else None  # Optional: store inputs for analysis
    
    if args.task_type in ["seq2seq", "seq2seq_stone_state"]:
        assert sos_token_id is not None, "SOS token ID must be defined for seq2seq tasks."
        eos_token_id = dataloader.dataset.eos_token_id if hasattr(dataloader.dataset, 'eos_token_id') else None
        assert eos_token_id is not None, "EOS token ID must be defined for seq2seq tasks."
    
    with torch.no_grad():
        pbar = tqdm(dataloader, disable=not accelerator.is_local_main_process)
        for _, batch in enumerate(pbar):
            # Move batch to device manually since dataloader is not prepared with accelerator
            batch = {k: v.to(accelerator.device) if torch.is_tensor(v) else v for k, v in batch.items()}
            encoder_input_ids = batch["encoder_input_ids"]
            
            if args.task_type == "seq2seq":
                decoder_input_ids = batch["decoder_input_ids"]
                decoder_target_ids = batch["decoder_target_ids"]

                # Get the unwrapped model to access the generate method
                unwrapped_model = accelerator.unwrap_model(model)

                # Use autoregressive generation instead of teacher forcing
                max_target_len = decoder_target_ids.size(1) + 4  # Allow some extra length
                
                generated_ids = unwrapped_model.generate(
                    src=encoder_input_ids,
                    start_symbol_id=sos_token_id,  
                    end_symbol_id=eos_token_id,    
                    max_len=max_target_len,
                    device=encoder_input_ids.device,
                    pad_token_id=pad_token_id
                )
                
                # Remove the SOS token from the generated sequences
                generated_ids = generated_ids[:, 1:]

                # Store predictions and targets if requested
                if args.store_predictions:
                    all_predictions.append(generated_ids)
                    all_targets.append(decoder_target_ids)
                    if all_encoder_inputs is not None:
                        all_encoder_inputs.append(encoder_input_ids)

                # Calculate loss and accuracy (existing logic)
                batch_size = decoder_target_ids.size(0)
                max_len = max(generated_ids.size(1), decoder_target_ids.size(1))

                # Pad both to the same length
                if generated_ids.size(1) < max_len:
                    padding = torch.full((batch_size, max_len - generated_ids.size(1)), pad_token_id, 
                                       dtype=generated_ids.dtype, device=generated_ids.device)
                    generated_padded = torch.cat([generated_ids, padding], dim=1)
                else:
                    generated_padded = generated_ids

                if decoder_target_ids.size(1) < max_len:
                    padding = torch.full((batch_size, max_len - decoder_target_ids.size(1)), pad_token_id, 
                                       dtype=decoder_target_ids.dtype, device=decoder_target_ids.device)
                    target_padded = torch.cat([decoder_target_ids, padding], dim=1)
                else:
                    target_padded = decoder_target_ids

                # Calculate error rate on all non-padding tokens (INCLUDING EOS)
                valid_mask = (target_padded != pad_token_id)
                mismatches = (generated_padded != target_padded) & valid_mask
                total_valid_tokens = valid_mask.sum().float()

                if total_valid_tokens > 0:
                    token_error_rate = mismatches.sum().float() / total_valid_tokens
                else:
                    token_error_rate = torch.tensor(0.0, device=encoder_input_ids.device)

                epsilon = 1e-8
                loss = -torch.log(1.0 - token_error_rate + epsilon)

                acc, correct, considered = calculate_accuracy_generated_seq2seq(generated_ids, decoder_target_ids, pad_token_id)
                if all_accs is not None: all_accs.append(acc)
                
            elif args.task_type == "classification":
                # if accelerator.is_local_main_process:
                #     import pdb; pdb.set_trace()
                target_class_ids = batch["target_class_id"]
                src_padding_mask = (encoder_input_ids == pad_token_id)
                output_logits = model(encoder_input_ids, src_padding_mask=src_padding_mask)
                loss = criterion(output_logits, target_class_ids)
                acc, correct, considered = calculate_accuracy_classification(output_logits, target_class_ids)
                
                # Store predictions and targets if requested
                if args.store_predictions:
                    predicted_classes = output_logits.argmax(dim=-1)  # Get predicted class IDs
                    all_predictions.append(predicted_classes)
                    all_targets.append(target_class_ids)
                    if all_encoder_inputs is not None:
                        all_encoder_inputs.append(encoder_input_ids)
                
            elif args.task_type == "classification_multi_label":
                target_feature_vector = batch["target_feature_vector"]
                src_padding_mask = (encoder_input_ids == pad_token_id)
                output_logits = model(encoder_input_ids, src_padding_mask=src_padding_mask)
                
                
                feature_groups = [(0, 3), (3, 6), (6, 9), (9, 13)]
                
                # Initialize CrossEntropyLoss with ignore_index for group level multi-class classification.
                loss_criterion = nn.CrossEntropyLoss(ignore_index=pad_token_id)
                
                total_loss = 0.0
                num_losses = 0
                
                all_groups_correct = torch.ones(target_feature_vector.size(0), dtype=torch.bool, device=target_feature_vector.device)  # Start with all samples considered correct, and then set 0 where they are not correct.
                for start_idx, end_idx in feature_groups:
                    logits_group = output_logits[:, start_idx:end_idx]
                    target_group_one_hot = target_feature_vector[:, start_idx:end_idx]  # Get corresponding target features for that group.
                    
                    # Convert one-hot to class indices
                    target_group_indices = target_group_one_hot.argmax(dim=-1)
                    # Calculate loss for this group
                    group_loss = loss_criterion(logits_group, target_group_indices)
                    total_loss += group_loss
                    num_losses += 1
                    
                    # --- Accuracy Calculation ---
                    # Get predicted class index for this group
                    predicted_indices_group = logits_group.argmax(dim=-1)
                    # Check which samples are correct for this group
                    correct_in_group = (predicted_indices_group == target_group_indices)
                    # Update the overall correctness. A sample is only correct if it has been
                    # correct for all previous groups AND is correct for the current one.
                    all_groups_correct &= correct_in_group
                    
                # Average the loss across all groups
                loss = total_loss / num_losses
                
                # Calculate accuracy based on the overall correctness across all groups
                correct = all_groups_correct.sum().item()
                considered = target_feature_vector.size(0)
                acc = correct / considered if considered > 0 else 0.0
                
                
                # Store predictions and targets if requested
                if args.store_predictions:
                    # To store predictions, we need to convert the group-wise logits back to a single multi-hot vector
                    predicted_multi_hot = torch.zeros_like(target_feature_vector)
                    for start_idx, end_idx in feature_groups:
                        logits_group = output_logits[:, start_idx:end_idx]
                        predicted_indices_group = logits_group.argmax(dim=-1)
                        # Place a '1' at the predicted index for each sample in the group
                        predicted_multi_hot.scatter_(1, (start_idx + predicted_indices_group).unsqueeze(1), 1)

                    all_predictions.append(predicted_multi_hot)
                    all_targets.append(target_feature_vector)
                    if all_encoder_inputs is not None:
                        all_encoder_inputs.append(encoder_input_ids)
                
            elif args.task_type == "seq2seq_stone_state":
                decoder_input_ids = batch["decoder_input_ids"]
                decoder_target_ids = batch["decoder_target_ids"]
                
                output_logits = model(encoder_input_ids, decoder_input_ids)
                loss = criterion(output_logits.view(-1, output_logits.shape[-1]), decoder_target_ids.view(-1))
                acc, correct, considered = calculate_accuracy_seq2seq(output_logits, decoder_target_ids, pad_token_id, eos_token_id=None)
                
                # Store predictions and targets if requested
                if args.store_predictions:
                    predicted_tokens = output_logits.argmax(dim=-1)  # Get predicted token IDs
                    all_predictions.append(predicted_tokens)
                    all_targets.append(decoder_target_ids)
                    if all_encoder_inputs is not None:
                        all_encoder_inputs.append(encoder_input_ids)
                        
            else:
                raise ValueError(f"Unknown task_type: {args.task_type}")

            total_loss += loss.item()
            total_correct_preds += correct
            total_considered_items += considered

    # Convert predictions and targets to numpy arrays for single GPU validation
    if args.store_predictions:
        # Convert tensor lists to numpy arrays directly
        all_predictions = [item.cpu().numpy() for item in all_predictions]
        all_targets = [item.cpu().numpy() for item in all_targets]

        if all_encoder_inputs is not None:
            all_encoder_inputs = [item.cpu().numpy() for item in all_encoder_inputs]

    # Save predictions and targets if requested and we're on the main process
    if args.store_predictions and accelerator.is_local_main_process:
        predictions_dir = os.path.join(args.save_dir, "predictions")
        if not os.path.exists(predictions_dir):
            os.makedirs(predictions_dir)
        
        # Create filenames with epoch and task type
        epoch_str = f"epoch_{epoch_num+1:03d}"
        pred_filename = f"predictions_{args.task_type}_{epoch_str}.npz"
        target_filename = f"targets_{args.task_type}_{epoch_str}.npz"
        input_filename = f"inputs_{args.task_type}_{epoch_str}.npz"
        
        pred_path = os.path.join(predictions_dir, pred_filename)
        target_path = os.path.join(predictions_dir, target_filename)
        input_path = os.path.join(predictions_dir, input_filename)
        # Convert lists to numpy arrays
        # predictions_array = np.array(all_predictions, dtype=object) if args.task_type == "seq2seq" else np.array(all_predictions)
        # targets_array = np.array(all_targets, dtype=object) if args.task_type in ["seq2seq", "seq2seq_stone_state"] else np.array(all_targets)
        
        predictions_array = np.concatenate(all_predictions, axis=0) if all_predictions else np.array([])
        targets_array = np.concatenate(all_targets, axis=0) if all_targets else np.array([])
        # if all_encoder_inputs is not None:
        #     all_encoder_inputs = np.concatenate(all_encoder_inputs, axis=0) if all_encoder_inputs else np.array([])
        
        print("Shape of predictions array:", predictions_array.shape)
        print("Shape of targets array:", targets_array.shape)
        
        # Save using numpy compressed format
        np.savez_compressed(pred_path, predictions=predictions_array)
        np.savez_compressed(target_path, targets=targets_array)
    
        if all_encoder_inputs is not None:
            # if accelerator.is_local_main_process:
            #     print("Shape of inputs array:", len(all_encoder_inputs), all_encoder_inputs[0].shape if len(all_encoder_inputs) > 0 else "N/A")
            #     import pdb; pdb.set_trace()
            inputs_array = np.array(all_encoder_inputs, dtype=object)
            np.savez_compressed(input_path, inputs=inputs_array)
        
        print(f"Saved predictions to: {pred_path}")
        print(f"Saved targets to: {target_path}")
        if all_encoder_inputs is not None:
            print(f"Saved inputs to: {input_path}")
        
        # Log file paths to wandb for easy access
        # wandb.log({
        #     f"predictions_file_epoch_{epoch_num+1}": pred_path,
        #     f"targets_file_epoch_{epoch_num+1}": target_path,
        # })
        # if all_encoder_inputs is not None:
        #     wandb.log({f"inputs_file_epoch_{epoch_num+1}": input_path})
            
    avg_epoch_loss = total_loss / len(dataloader)
    avg_epoch_accuracy = total_correct_preds / total_considered_items if total_considered_items > 0 else 0.0
    
    if args.task_type == "seq2seq":
        return avg_epoch_loss, avg_epoch_accuracy, all_accs # all_accs is specific to seq2seq here
    else: # classification or classification_multi_label
        return avg_epoch_loss, avg_epoch_accuracy

def load_checkpoint(checkpoint_path, model, optimizer, scheduler, accelerator):
    """Load checkpoint and return starting epoch and best validation loss."""
    print(f"Loading checkpoint from: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Load model state
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load optimizer state
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Load scheduler state if available
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        print("Loaded scheduler state from checkpoint")
    
    start_epoch = checkpoint['epoch'] + 1
    best_val_loss = checkpoint.get('best_val_loss', checkpoint.get('loss', float('inf')))
    
    print(f"Resumed from epoch {checkpoint['epoch']}, starting epoch {start_epoch}")
    print(f"Previous best validation loss: {best_val_loss}")
    
    return start_epoch, best_val_loss, checkpoint.get('args', None)

def validate_resume_compatibility(checkpoint_args, current_args, allow_mismatch=False):
    """Validate that resumed training is compatible with current arguments."""
    if not checkpoint_args:
        print("No checkpoint args found, skipping validation")
        return
    
    # Critical parameters that must match
    critical_params = ['task_type', 'model_size', 'model_architecture']
    # Data parameters that should ideally match
    data_params = ['train_data_path', 'val_data_path', 'data_split_seed']
    
    errors = []
    warnings = []
    
    # Check critical parameters
    for param in critical_params:
        if hasattr(checkpoint_args, param) and hasattr(current_args, param):
            checkpoint_val = getattr(checkpoint_args, param)
            current_val = getattr(current_args, param)
            if checkpoint_val != current_val:
                errors.append(f"{param}: checkpoint={checkpoint_val}, current={current_val}")
    
    # Check data parameters
    for param in data_params:
        if hasattr(checkpoint_args, param) and hasattr(current_args, param):
            checkpoint_val = getattr(checkpoint_args, param)
            current_val = getattr(current_args, param)
            if checkpoint_val != current_val:
                warnings.append(f"{param}: checkpoint={checkpoint_val}, current={current_val}")
    
    if errors:
        raise ValueError(f"Critical parameter mismatch - cannot resume: {errors}")
    
    if warnings:
        print(f"WARNING: Parameter differences detected: {warnings}")
        if not allow_mismatch:
            print("Use --allow_data_path_mismatch to override this warning and continue")
            raise ValueError("Data path mismatch detected. Use --allow_data_path_mismatch to continue anyway.")
        else:
            print("Continuing with different data paths as requested")

def main():
    args = parse_args()
    
    # ADD THIS SECTION - Handle resume logic early
    start_epoch = 0
    best_val_loss = float('inf')
    resume_checkpoint_path = None
    checkpoint_args = None
    
    if args.resume_from_checkpoint and args.resume_from_checkpoint != '':
        resume_checkpoint_path = args.resume_from_checkpoint
        if not os.path.exists(resume_checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {resume_checkpoint_path}")
        
        # Load checkpoint to get original args for validation
        print(f"Loading checkpoint for resume: {resume_checkpoint_path}")
        checkpoint = torch.load(resume_checkpoint_path, map_location='cpu')
        checkpoint_args = checkpoint.get('args', None)
        
        # Validate compatibility
        validate_resume_compatibility(
            checkpoint_args, 
            args, 
            allow_mismatch=args.allow_data_path_mismatch
        )
        
        print(f"Resuming training from epoch {checkpoint['epoch'] + 1}")
        if checkpoint_args:
            print(f"Original train data: {getattr(checkpoint_args, 'train_data_path', 'N/A')}")
            print(f"Current train data: {args.train_data_path}")

    args.allow_data_path_mismatch = str(args.allow_data_path_mismatch) == 'True'  or str(args.allow_data_path_mismatch) == 'true'

    
    args.is_held_out_color_exp = str(args.is_held_out_color_exp) == 'True' or str(args.is_held_out_color_exp) == 'true'  # Convert to boolean 
    if args.is_held_out_color_exp:  
        print("Running held-out color experiment.")
        edges_value = args.train_data_path.split('_edges_exp.json')[0].split('_')[-1]
        print("Held out edges value for held-out color experiment: ", edges_value)
        args.num_held_out_edges = int(edges_value)
        
    args.save_checkpoints = str(args.save_checkpoints) == 'True'  # Convert to boolean
    if args.save_checkpoints:
        print("Saving checkpoints during training.")
    
    base_path = None
    if cluster == 'cirrus':
        # Cirrus-specific paths
        base_path = '/home/rsaha/projects/dm_alchemy/'
    elif cluster == 'cc':
        # Profile is cc.
        base_path = '/home/rsaha/projects/def-afyshe-ab/rsaha/projects/dm_alchemy/'
    elif cluster == 'rorqual':
        base_path = '/home/rsaha/links/projects/def-afyshe-ab/rsaha/projects/dm_alchemy/'
    elif cluster == 'vulcan':
        base_path = '/home/rsaha/projects/def-afyshe-ab/rsaha/dm_alchemy/'

    
    print("Base path: ", base_path)
    print("Profile cluster: ", cluster)
    
    # First, add the 'data_split_seed' to the train_data_path and val_data_path
    args.train_data_path = f"{args.train_data_path.split('.json')[0]}_seed_{args.data_split_seed}.json"
    args.val_data_path = f"{args.val_data_path.split('.json')[0]}_seed_{args.data_split_seed}.json"
        
    # Update data paths to be relative to the base path
    args.train_data_path = os.path.join(base_path, args.train_data_path)
    args.val_data_path = os.path.join(base_path, args.val_data_path) if args.val_data_path else None
    args.save_dir = os.path.join(base_path, args.save_dir)
    print("Updated train data path: ", args.train_data_path)
    print("Updated validation data path: ", args.val_data_path if args.val_data_path else "None")
    
    # Extract the 'shop' and 'qhop' length from the args.train_data_path.
    # the train_data_path is expected to be in the format:
    # src/data/generated_data/decompositional_chemistry_samples_167424_80_unique_stones_train_shop_2_qhop_1.json'
    match = re.search(r'shop_(\d+)_qhop_(\d+)', args.train_data_path)
    if match:
        args.shop_length = int(match.group(1))
        args.qhop_length = int(match.group(2))
        if args.shop_length < args.qhop_length:
            args.way = 'composition'
        elif args.shop_length > args.qhop_length:
            args.way = 'decomposition'
        else:
            args.way = 'equal'
        

    
    # Initialize Accelerator
    args.fp16 = str(args.fp16) == 'True'  # Convert to boolean
    if args.fp16:
        # Use mixed precision training if specified
        accelerator = Accelerator(mixed_precision='fp16')
        print("Using mixed precision training (fp16).")
    else:
        accelerator = Accelerator()
        print("Using full precision training (fp32).")
    # Set seed for reproducibility
    set_seed(args.seed)
    print("Seed: ", args.seed)
    
    # Scale learning rate linearly with number of processes (GPUs) (https://huggingface.co/docs/accelerate/concept_guides/performance#learning-rates).
    # NOTE: This is at the discretion of the user and is not a requirement.
    

    # Initialize wandb only on main process
    if accelerator.is_local_main_process:
        # REPLACE YOUR EXISTING wandb.init() with this:
        wandb_kwargs = {
            "project": args.wandb_project,
            "entity": args.wandb_entity,
            "config": vars(args),
            "mode": args.wandb_mode
        }
        
        if args.resume_wandb_run_id:
            # Resume specific wandb run
            wandb_kwargs["id"] = args.resume_wandb_run_id
            wandb_kwargs["resume"] = "must"
            wandb_kwargs["name"] = None  # Don't override name when resuming
            print(f"Resuming wandb run: {args.resume_wandb_run_id}")
        else:
            # New wandb run
            wandb_kwargs["name"] = args.wandb_run_name if args.wandb_run_name else f"{args.task_type}_{args.model_size}_{time.strftime('%Y%m%d-%H%M%S')}"
        
        wandb.init(**wandb_kwargs)

    print(f"Using device: {accelerator.device}")
    print(f"Selected task type: {args.task_type}")
    print(f"Number of processes: {accelerator.num_processes}")

    # --- Dataset and DataLoader ---
    print(f"Loading training data from: {args.train_data_path}")
    
    # Create dataset with optional validation split
    print(f"Filter query from support initial: {args.filter_query_from_support}")
    # Correctly convert string "True" or "False" or boolean True to boolean
    args.filter_query_from_support = str(args.filter_query_from_support) == 'True'
    args.store_predictions = str(args.store_predictions) == 'True'
    args.use_preprocessed = str(args.use_preprocessed) == 'True'  # Add this line
    
    print("Filter query from support set to:", args.filter_query_from_support)
    print("Store predictions:", args.store_predictions)
    print("Use preprocessed data:", args.use_preprocessed)  # Add this line
    
    # Update paths to be absolute
    args.preprocessed_dir = os.path.join(base_path, args.preprocessed_dir)  # Add this line
    print("Preprocessed directory path: ", args.preprocessed_dir)  # Add this line
    
    if args.task_type == 'classification' and (args.input_format is not None):
        args.output_format = 'stone_states'
    elif args.task_type == 'classification_multi_label' and (args.input_format is not None):
        args.output_format = 'features'
    
    with accelerator.main_process_first():
        full_dataset = AlchemyDataset(
            json_file_path=args.train_data_path, 
            task_type=args.task_type,
            filter_query_from_support=args.filter_query_from_support,
            num_workers=args.num_workers,
            val_split=args.val_split,
            val_split_seed=args.val_split_seed,
            preprocessed_dir=args.preprocessed_dir,  # Add this line
            use_preprocessed=args.use_preprocessed,   # Add this line
            input_format=args.input_format,
            output_format=args.output_format,
            model_architecture=args.model_architecture
        )

    # Get train and validation sets
    train_dataset = full_dataset.get_train_set()
    val_dataset = None
   
   
    # Doing some changes to the sequence length based on the training data. The default sequence length is 2048 but
    # if the sequence length is longer than that, we will double the max_seq_len.
    args.use_truncation = str(args.use_truncation) == 'True'  # Convert to boolean

    max_length = max(len(item['encoder_input_ids']) for item in train_dataset)
    # all_lengths = [len(item['encoder_input_ids']) for item in train_dataset]
    print(f"Maximum sequence length in training data: {max_length}")
    if max_length > args.max_seq_len:
        if args.use_truncation:
            print("Truncation will be applied to sequences longer than args.max_seq_len.")
            print("Using truncation as per args.max_seq_len = ", args.max_seq_len)
        else:
            print("Increasing args.max_seq_len to accommodate longer sequences without truncation.")
            args.max_seq_len = max_length
            print(f"Adjusted args.max_seq_len: {args.max_seq_len}")
    else:
        print(f"Maximum sequence length {max_length} is within args.max_seq_len {args.max_seq_len}. No adjustment needed.")

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
            vocab_word2idx=full_dataset.input_word2idx,  # Use input vocabulary
            vocab_idx2word=full_dataset.input_idx2word,
            stone_state_to_id=full_dataset.stone_state_to_id if args.task_type == "classification" else None,
            filter_query_from_support=args.filter_query_from_support,
            num_workers=args.num_workers,
            preprocessed_dir=args.preprocessed_dir,  # Add this line
            use_preprocessed=args.use_preprocessed,   # Add this line
            input_format=args.input_format,
            output_format=args.output_format,
            model_architecture=args.model_architecture
        )
    else:
        print("No validation data specified. Skipping validation.")

    # Get vocabulary info from the main dataset
    pad_token_id = full_dataset.pad_token_id
    src_vocab_size = len(full_dataset.input_word2idx)  # Use input vocabulary size

    print(f"Source (Input) Vocabulary size: {src_vocab_size}")
    print(f"Pad token ID for encoder inputs: {pad_token_id}")
    
    eos_token_id = full_dataset.eos_token_id if hasattr(full_dataset, 'eos_token_id') else None
    sos_token_id = full_dataset.sos_token_id if hasattr(full_dataset, 'sos_token_id') else None

    if args.task_type == "seq2seq" or args.task_type == "seq2seq_stone_state":
        sos_token_id = full_dataset.sos_token_id
        eos_token_id = full_dataset.eos_token_id
        if args.task_type == "seq2seq":
            tgt_vocab_size = src_vocab_size  # Features for both input and output
        else:  # seq2seq_stone_state
            tgt_vocab_size = len(full_dataset.output_word2idx)  # Use output vocabulary size
        print(f"Target (Output) Vocabulary size: {tgt_vocab_size}")
        print(f"SOS ID: {sos_token_id}, EOS ID: {eos_token_id}")
    elif args.task_type == "classification":
        if args.override_num_classes is None:
            num_classes = len(full_dataset.stone_state_to_id)  # Based on output stone states
        else:
            print(f"override_num_classes is set. Using a value of {args.override_num_classes} for classification.")
            num_classes = args.override_num_classes
        print(f"Number of classes (Stone States): {num_classes}")
    elif args.task_type == "classification_multi_label":
        num_output_features = full_dataset.num_output_features
        print(f"Number of ungrouped output features (for Multi-label Classification): {num_output_features}")

    # Create data loaders
    custom_collate_train = partial(collate_fn, pad_token_id=pad_token_id, eos_token_id = eos_token_id, 
                                   task_type=args.task_type, model_architecture=args.model_architecture, 
                                   sos_token_id=sos_token_id, prediction_type=args.prediction_type,
                                   max_seq_len=args.max_seq_len, truncate=args.use_truncation, padding_side=args.padding_side)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=custom_collate_train,
        num_workers=args.num_workers
    )

    val_dataloader = None
    if val_dataset is not None:
        custom_collate_val = partial(collate_fn, pad_token_id=pad_token_id, eos_token_id = eos_token_id, 
                                     task_type=args.task_type, model_architecture=args.model_architecture, 
                                     sos_token_id=sos_token_id, prediction_type=args.prediction_type,
                                     max_seq_len=args.max_seq_len, truncate=args.use_truncation)
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=custom_collate_val,
            num_workers=args.num_workers
        )

    # --- Model ---
    if args.task_type == "seq2seq" or args.task_type == "seq2seq_stone_state":
        model = create_transformer_model(
            config_name=args.model_size,
            src_vocab_size=src_vocab_size,
            tgt_vocab_size=tgt_vocab_size,
            device=accelerator.device,
            max_len=args.max_seq_len
        )
        criterion = nn.CrossEntropyLoss(ignore_index=pad_token_id, reduction=args.multi_label_reduction) # Use CrossEntropyLoss for seq2seq, ignoring PAD_ID
    elif args.task_type == "classification":
        if args.model_architecture == "decoder":
            print("Using decoder architecture for classification task.")
            model = create_decoder_classifier_model(
                config_name=args.model_size,
                src_vocab_size=src_vocab_size,
                num_classes=num_classes,
                device=accelerator.device,
                max_len=args.max_seq_len,
                prediction_type=args.prediction_type,
                padding_side=args.padding_side
            )
        else:  # encoder architecture
            model = create_classifier_model(
                config_name=args.model_size,
                src_vocab_size=src_vocab_size,
                num_classes=num_classes,
                device=accelerator.device,
                max_len=args.max_seq_len,
                io_sep_token_id=full_dataset.io_sep_token_id if hasattr(full_dataset, 'io_sep_token_id') else None,
                item_sep_token_id=full_dataset.item_sep_token_id if hasattr(full_dataset, 'item_sep_token_id') else None,
                pooling_strategy=args.pooling_strategy
            )
        criterion = nn.CrossEntropyLoss(reduction=args.multi_label_reduction) # Use CrossEntropyLoss for classification
    elif args.task_type == "classification_multi_label":
        if args.model_architecture == "decoder":
            print("Using decoder architecture for multi-label classification task.")
            print("Prediction type:", args.prediction_type)
            model = create_decoder_classifier_model(
                config_name=args.model_size,
                src_vocab_size=src_vocab_size,
                num_classes=num_output_features, # Output layer size is num_output_features
                device=accelerator.device,
                max_len=args.max_seq_len,
                prediction_type=args.prediction_type,
                padding_side=args.padding_side
            )
        else:  # encoder architecture
            model = create_classifier_model( # Using the same StoneStateClassifier model
                config_name=args.model_size,
                src_vocab_size=src_vocab_size,
                num_classes=num_output_features, # Output layer size is num_output_features
                device=accelerator.device,
                max_len=args.max_seq_len,
                io_sep_token_id=full_dataset.io_sep_token_id if hasattr(full_dataset, 'io_sep_token_id') else None,
                item_sep_token_id=full_dataset.item_sep_token_id if hasattr(full_dataset, 'item_sep_token_id') else None,
                pooling_strategy=args.pooling_strategy, 
            )
        criterion = nn.CrossEntropyLoss(reduction=args.multi_label_reduction) # Use BCEWithLogitsLoss for multi-label
    else:
        raise ValueError(f"Unknown task_type: {args.task_type}")
    
    if accelerator.is_local_main_process:
        wandb.watch(model, log="all", log_freq=100)

    # --- Optimizer and Scheduler ---
    print(f"Base learning rate: {args.learning_rate }, Weight decay: {args.weight_decay}")


    if args.optimizer == "adamw":
        optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        print("Using AdamW optimizer.")
    elif args.optimizer == "adam":
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=0)
        print("Using Adam optimizer with no weight decay (0) - hardcoded.")
    elif args.optimizer == 'rmsprop':
        optimizer = optim.RMSprop(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        print("Using RMSprop optimizer.")
    elif args.optimizer == 'adagrad':
        optimizer = optim.Adagrad(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        print("Using Adagrad optimizer.")
    # elif args.optimizer == 'adafactor':
    #     optimizer = optim.Adafactor(
    #         model.parameters(),
    #         lr=args.learning_rate,
    #         scale_parameter=False,
    #         relative_step=False,
    #         weight_decay=args.weight_decay,
    #         warmup_init=False
    #     )
        # print("Using Adafactor optimizer.")
    
    use_scheduler = str(args.use_scheduler) == "True" or str(args.use_scheduler) == "true"
    print("Use scheduler: ", use_scheduler)
    scheduler = None
    
    if use_scheduler and args.scheduler_type != "none":
        if args.scheduler_type == "cosine":
            # For cosine annealing no cycles, T_max should be total number of batches
            # NOTE: The way you define the num_training_steps, it changes whether you have cyclic behaviour or not - but also depends on where you call scheduler.step().
            if args.scheduler_call_location == "after_batch":
                num_training_steps = args.epochs * len(train_dataloader)
                print(f"Using CosineAnnealingLR with T_max={num_training_steps} (called per batch)")
            elif args.scheduler_call_location == "after_epoch":
                num_training_steps = args.epochs
                print(f"Using CosineAnnealingLR with T_max={num_training_steps} (called per epoch)")
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_training_steps, eta_min=1e-5)
            print(f"Using CosineAnnealingLR with T_max={num_training_steps}")
        elif args.scheduler_type == 'cosine_restarts':
            # For cosine annealing with restarts, T_0 is the number of epochs
            t_0 = 20
            if accelerator.is_local_main_process:
                wandb.log({"T_0": t_0})
            print(f"Using CosineAnnealingWarmRestarts with T_0={t_0} (called per batch but will restart after every T_0 epochs)")
            scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=t_0, eta_min=1e-5)
            print(f"Using CosineAnnealingWarmRestarts with T_0={t_0}")
        elif args.scheduler_type == "exponential":
            scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.gamma)
            print(f"Using ExponentialLR with gamma={args.gamma}")
    else:
        print("No scheduler will be used")

    # Prepare everything with Accelerator
    model, optimizer, train_dataloader, scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, scheduler
    )
    
    # ADD THIS SECTION - Load checkpoint after preparation if resuming
    if resume_checkpoint_path:
        start_epoch, best_val_loss, _ = load_checkpoint(
            resume_checkpoint_path, model, optimizer, scheduler, accelerator
        )
    # END OF ADDED SECTION
    
    # Note: val_dataloader is NOT prepared with accelerator for single GPU validation

    if accelerator.is_local_main_process:
        print(f"Model initialized: {args.model_size}, Architecture: {args.model_architecture}, Task: {args.task_type}")
        print(f"Optimizer: AdamW, LR: {args.learning_rate}, Weight Decay: {args.weight_decay}")
        
        if scheduler:
            if args.scheduler_type == "cosine":
                # num_training_steps = args.epochs * len(train_dataloader)
                num_training_steps = args.epochs
                print(f"Scheduler: CosineAnnealingLR, T_max: {num_training_steps} (called per batch)")
            elif args.scheduler_type == "exponential":
                print(f"Scheduler: ExponentialLR, gamma: {args.gamma} (called per epoch)")
        else:
            print("Scheduler: ", scheduler)
            
        if args.task_type == "seq2seq":
            print(f"Criterion: CrossEntropyLoss (ignoring PAD_ID: {pad_token_id} for target sequences)")
        elif args.task_type == "classification":
            print(f"Criterion: CrossEntropyLoss (for class predictions)")
        elif args.task_type == "classification_multi_label":
            print(f"Criterion: CrossEntropyLoss (for group level multi-class predictions)")

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
    
    if args.is_held_out_color_exp:
        # The train_data_path is of the form 'src/data/held_out_exps_generated_data/compositional_chemistry_samples_167424_80_unique_stones_train_shop_1_qhop_1_single_held_out_color_1_edges_exp.json'
        # You can see that towards the end, there's a '1_edges_exp' part. The number should also be a part of the args.save_dir.
        # Let's first extract that number and then add it to the args.save_dir using os.path.join.
        
        
        # Extract the held-out color number from the train_data_path
        held_out_edge_match = re.search(r'_held_out_color_(\d+)_edges_exp', args.train_data_path)
        if held_out_edge_match:
            held_out_edge_number = held_out_edge_match.group(1)
            print(f"Held-out color number extracted: {held_out_edge_number}")
        args.save_dir = os.path.join(args.save_dir, f"held_out_color_exp")
        args.save_dir = os.path.join(args.save_dir, f"held_out_edges_{held_out_edge_number}")
        if 'complete_graph' in args.train_data_path:
            args.save_dir = os.path.join(args.save_dir, f"complete_graph")
        
    hierarchical_save_dir = os.path.join(
        args.save_dir,
        args.model_size,
        args.model_architecture,
        args.task_type,
        f"input_{args.input_format or 'default'}",
        f"output_{args.output_format or 'default'}",
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

    for epoch in tqdm(range(start_epoch, args.epochs), disable=not accelerator.is_local_main_process):
        if accelerator.is_local_main_process:
            print(f"--- Epoch {epoch+1}/{args.epochs} ---")
        with accelerator.autocast():
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
                    'best_val_loss': best_val_loss,  
                    'loss': val_loss,  # or train_loss
                    'args': args,
                    'src_vocab_word2idx': full_dataset.word2idx,
                    'src_vocab_idx2word': full_dataset.idx2word
                }
                if args.task_type == "seq2seq" or args.task_type == "seq2seq_stone_state":
                    checkpoint['tgt_vocab_word2idx'] = full_dataset.word2idx 
                    checkpoint['tgt_vocab_idx2word'] = full_dataset.idx2word
                elif args.task_type == "classification":
                    checkpoint['stone_state_to_id'] = full_dataset.stone_state_to_id
                    checkpoint['id_to_stone_state'] = full_dataset.id_to_stone_state
                elif args.task_type == "classification_multi_label":
                    checkpoint['feature_to_idx_map_input'] = full_dataset.feature_to_idx_map_input
                    checkpoint['feature_to_idx_map_output'] = full_dataset.feature_to_idx_map_output
                    checkpoint['num_output_features'] = full_dataset.num_output_features
                
                # ADD this section after creating the checkpoint dict:
                if scheduler is not None:
                    checkpoint['scheduler_state_dict'] = scheduler.state_dict()
                    
                if args.save_checkpoints:
                    torch.save(checkpoint, model_save_path)
                print(f"New best validation loss: {best_val_loss:.4f}. Model saved to {model_save_path}")
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
                if args.task_type == "seq2seq" or args.task_type == "seq2seq_stone_state":
                    checkpoint['tgt_vocab_word2idx'] = full_dataset.word2idx
                    checkpoint['tgt_vocab_idx2word'] = full_dataset.idx2word
                elif args.task_type == "classification":
                    checkpoint['stone_state_to_id'] = full_dataset.stone_state_to_id
                    checkpoint['id_to_stone_state'] = full_dataset.id_to_stone_state
                elif args.task_type == "classification_multi_label":
                    checkpoint['feature_to_idx_map'] = full_dataset.feature_to_idx_map
                    checkpoint['idx_to_feature_map'] = {v: k for k, v in full_dataset.feature_to_idx_map.items()}
                    checkpoint['num_output_features'] = full_dataset.num_output_features
                if args.save_checkpoints:
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