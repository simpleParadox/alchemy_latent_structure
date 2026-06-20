import argparse
import gc
import glob
import json
import os
import math
import random
import re
import numpy as np
from functools import partial
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import wandb
from tqdm import tqdm
from accelerate import Accelerator

def format_time(seconds):
    seconds = int(seconds)
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    if h > 0:
        return f"{h}h {m}m {s}s"
    elif m > 0:
        return f"{m}m {s}s"
    else:
        return f"{s}s"

# Reuse standard components directly from train.py
from train import (
    build_parser,
    set_seed,
    worker_init_fn,
    train_epoch,
    validate_epoch,
    _prefer_aip_output_root,
    _apply_freeze_layers_in_place,
    cluster
)

# Sibling imports
from data_loaders import AlchemyDataset, collate_fn
from models import (
    create_transformer_model,
    create_classifier_model,
    create_decoder_classifier_model,
    create_linear_model
)

def parse_continual_args():
    parser = build_parser()
    parser.set_defaults(continual="True", model_architecture="decoder")
    
    # Override type of --task_sequence to str to prevent argparse int casting crashes on string/list sweep inputs
    for action in parser._actions:
        if action.dest == 'task_sequence':
            action.type = str
            
    args = parser.parse_args()
    
    # Parse task_sequence into list of integers
    if args.task_sequence:
        parsed_seq = []
        for item in args.task_sequence:
            if isinstance(item, str):
                cleaned = item.replace('[', '').replace(']', '').replace(',', ' ').strip()
                for part in cleaned.split():
                    try:
                        parsed_seq.append(int(part))
                    except ValueError:
                        pass
            else:
                try:
                    parsed_seq.append(int(item))
                except (ValueError, TypeError):
                    pass
        args.task_sequence = parsed_seq
        
    return args

def main():
    args = parse_continual_args()
    
    # Always use StoneStateDecoderClassifier (decoder architecture) for classification
    args.model_architecture = "decoder"
    
    # ----------------------------------------------------
    # Pre-setup configuration
    # ----------------------------------------------------
    # Convert string boolean flags to Python bools
    args.continual = str(args.continual) == 'True'
    args.reset_optimizer = str(args.reset_optimizer) == 'True'
    args.filter_query_from_support = str(args.filter_query_from_support) == 'True'
    args.store_predictions = str(args.store_predictions) == 'True'
    args.use_preprocessed = str(args.use_preprocessed) == 'True'
    args.use_scheduler = str(args.use_scheduler) == 'True'
    args.save_checkpoints = str(args.save_checkpoints) == 'True'
    args.log_continual_csv = str(args.log_continual_csv) == 'True'
    args.enable_auto_stop = str(args.enable_auto_stop) == 'True'
    args.use_truncation = str(args.use_truncation) == 'True'
    
    # Multi-GPU / Precision Setup
    if args.fp16 == 'True':
        accelerator = Accelerator(mixed_precision='fp16')
    else:
        accelerator = Accelerator()
    
    # Set seed
    set_seed(args.seed)
    
    # Base path resolution (mirrors train.py logic)
    base_path = '/home/rsaha/projects/aip-afyshe/rsaha/dm_alchemy/'
    if cluster == 'beluga':
        base_path = '/home/rsaha/projects/def-afyshe-ab/rsaha/projects/dm_alchemy/'
        args.store_in_scratch = True
    elif cluster == 'rorqual':
        base_path = '/home/rsaha/links/projects/def-afyshe-ab/rsaha/projects/dm_alchemy/'
    elif cluster == 'vulcan':
        base_path = '/home/rsaha/projects/def-afyshe-ab/rsaha/dm_alchemy/'
    elif cluster == 'killarney':
        base_path = '/home/rsaha/projects/aip-afyshe/rsaha/dm_alchemy/'
        
    if accelerator.is_local_main_process:
        print("Base path: ", base_path)
        print("Profile cluster: ", cluster)
        print(f"Using device: {accelerator.device}")
        print(f"Selected task type: {args.task_type}")
        print(f"Number of processes: {accelerator.num_processes}")
        
    if args.val_data_path is None or args.val_data_path == 'None':
        args.val_data_path = args.train_data_path.replace('train', 'val')
        
    original_train_data_path = args.train_data_path
    original_val_data_path = args.val_data_path
    
    # Continual run folders
    train_data_path_template = original_train_data_path
    val_data_path_template = original_val_data_path
    
    # Parse initial hop values
    match = re.search(r'shop_(\d+)_qhop_(\d+)', train_data_path_template)
    if match:
        support_hop_init = match.group(1)
        query_hop_init = match.group(2)
    else:
        support_hop_init = "1"
        query_hop_init = "2"
        
    if args.save_continual_run_id:
        continual_folder_name = f"continual_{args.save_continual_run_id}"
    else:
        seq_str = "_".join(map(str, args.task_sequence))
        continual_folder_name = f"continual_seq_{seq_str}"
        
    # Build save directory structure matching train.py standard
    save_dir_base = args.save_dir
    if "src/saved_models/" in save_dir_base:
        save_dir_base = save_dir_base.replace("src/saved_models/", "src/saved_models/continual/")
    else:
        save_dir_base = os.path.join(save_dir_base, "continual")
    if args.is_held_out_color_exp == 'True' or args.is_held_out_color_exp is True:
        held_out_edge_match = re.search(r'_held_out_color_(\d+)_edges_exp', train_data_path_template)
        held_out_edge_number = held_out_edge_match.group(1) if held_out_edge_match else "1"
        save_dir_base = os.path.join(save_dir_base, f"held_out_color_exp")
        if 'same_reward' in args.preprocessed_dir:
            save_dir_base = os.path.join(save_dir_base, f"same_reward_held_out_color_{held_out_edge_number}")
        else:
            save_dir_base = os.path.join(save_dir_base, f"held_out_edges_{held_out_edge_number}")
            
    if ('subsampled_complete_graph' in train_data_path_template) or ('subsampled_complete_graph' in args.preprocessed_dir):
        save_dir_base = os.path.join(save_dir_base, f"subsampled_complete_graph")
    elif ('complete_graph' in train_data_path_template) or ('complete_graph' in args.preprocessed_dir):
        save_dir_base = os.path.join(save_dir_base, f"complete_graph")
    else:
        save_dir_base = os.path.join(save_dir_base, f"all_graphs")
        
    if 'fully_shuffled' in train_data_path_template:
        save_dir_base = os.path.join(save_dir_base, f"fully_shuffled")
        
    if args.use_scheduler and args.scheduler_type != "none":
        save_dir_base = os.path.join(save_dir_base, f"scheduler_{args.scheduler_type}")
    else:
        save_dir_base = os.path.join(save_dir_base, f"no_scheduler")
        
    save_dir_base = os.path.join(save_dir_base, f"wd_{args.weight_decay}_lr_{args.learning_rate}")
    if args.scheduler_type == "cosine" or args.scheduler_type == "cosine_restarts":
        save_dir_base = os.path.join(save_dir_base, f"eta_min_{args.eta_min}")
    if args.scheduler_type == 'step_lr' or args.scheduler_type == 'multi_step_lr':
        save_dir_base = os.path.join(save_dir_base, f"step_size_{args.step_size}_gamma_{args.reduce_factor}")
        
    continual_save_dir = os.path.join(
        save_dir_base,
        args.model_size,
        args.model_architecture,
        args.task_type,
        f"input_{args.input_format or 'default'}",
        f"output_{args.output_format or 'default'}",
        continual_folder_name,
        f"seed_{args.data_split_seed}",
        f"init_seed_{args.seed}"
    )
    
    # Store normalized run-level path preference
    continual_save_dir = _prefer_aip_output_root(continual_save_dir)
    if accelerator.is_local_main_process:
        print(f"Continual learning output base: {continual_save_dir}")
        os.makedirs(continual_save_dir, exist_ok=True)
        
    # Model configuration
    model = None
    optimizer = None
    scheduler = None
    
    # Weights & Biases Config
    if accelerator.is_local_main_process:
        wandb_mode = args.wandb_mode
        wandb_config = vars(args).copy()
        wandb_config["slurm_job_id"] = os.environ.get("SLURM_JOB_ID", "")
        wandb_config["slurm_job_name"] = os.environ.get("SLURM_JOB_NAME", "")
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name or f"continual_{continual_folder_name}_seed_{args.seed}",
            config=wandb_config,
            mode=wandb_mode
        )
        
    # ----------------------------------------------------
    # Task Sequence Loop with Cycles
    # ----------------------------------------------------
    global_epoch_counter = 0
    num_cycles = getattr(args, "num_cycles", 1)
    start_time = time.time()
    
    for cycle_idx in range(1, num_cycles + 1):
        for task_idx, hop_length in enumerate(args.task_sequence):
            # 1. Resolve path for this hop length
            if args.continual_mode == "composition":
                hop_pattern_src = f"shop_{support_hop_init}_qhop_{query_hop_init}"
                hop_pattern_dst = f"shop_1_qhop_{hop_length}"
            else:  # decomposition
                hop_pattern_src = f"shop_{support_hop_init}_qhop_{query_hop_init}"
                hop_pattern_dst = f"shop_{hop_length}_qhop_1"
                
            current_train_data_path = re.sub(hop_pattern_src, hop_pattern_dst, train_data_path_template)
            current_val_data_path = re.sub(hop_pattern_src, hop_pattern_dst, val_data_path_template)
            
            current_train_data_path = f"{current_train_data_path.split('.json')[0]}_seed_{args.data_split_seed}.json"
            current_val_data_path = f"{current_val_data_path.split('.json')[0]}_seed_{args.data_split_seed}.json"
            
            current_train_data_path = os.path.join(base_path, current_train_data_path)
            current_val_data_path = os.path.join(base_path, current_val_data_path)
            
            if accelerator.is_local_main_process:
                print(f"\n==========================================")
                print(f"CYCLE {cycle_idx} | TASK {task_idx}: Hop Length = {hop_length}")
                print(f"Loading training data from: {current_train_data_path}")
                print(f"Loading validation data from: {current_val_data_path}")
                print(f"==========================================\n")
                
            # Instantiate dataset and dataloaders
            train_dataset = AlchemyDataset(
                json_file_path=current_train_data_path,
                task_type=args.task_type,
                filter_query_from_support=args.filter_query_from_support,
                num_workers=args.num_workers,
                preprocessed_dir=args.preprocessed_dir,
                use_preprocessed=args.use_preprocessed,
                input_format=args.input_format,
                output_format=args.output_format,
                model_architecture=args.model_architecture
            )
            
            val_dataset = AlchemyDataset(
                json_file_path=current_val_data_path,
                task_type=args.task_type,
                vocab_word2idx=train_dataset.input_word2idx,  # Use input vocabulary
                vocab_idx2word=train_dataset.input_idx2word,
                stone_state_to_id=train_dataset.stone_state_to_id if args.task_type == "classification" else None,
                filter_query_from_support=args.filter_query_from_support,
                num_workers=args.num_workers,
                preprocessed_dir=args.preprocessed_dir,
                use_preprocessed=args.use_preprocessed,
                input_format=args.input_format,
                output_format=args.output_format,
                model_architecture=args.model_architecture
            )
            
            pad_token_id = train_dataset.pad_token_id
            eos_token_id = train_dataset.eos_token_id if hasattr(train_dataset, 'eos_token_id') else None
            sos_token_id = train_dataset.sos_token_id if hasattr(train_dataset, 'sos_token_id') else None

            custom_collate_train = partial(
                collate_fn,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                task_type=args.task_type,
                model_architecture=args.model_architecture,
                sos_token_id=sos_token_id,
                prediction_type=args.prediction_type,
                max_seq_len=args.max_seq_len,
                truncate=args.use_truncation,
                padding_side=args.padding_side
            )

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
                padding_side=args.padding_side
            )

            train_dataloader = DataLoader(
                train_dataset,
                batch_size=args.batch_size,
                shuffle=True,
                collate_fn=custom_collate_train,
                num_workers=args.num_workers,
                worker_init_fn=worker_init_fn,
                generator=torch.Generator().manual_seed(args.seed)
            )
            
            val_dataloader = DataLoader(
                val_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                collate_fn=custom_collate_val,
                num_workers=1,
                worker_init_fn=worker_init_fn,
                generator=torch.Generator().manual_seed(args.seed)
            )
            
            # Reset streak counter for early stopping
            val_acc_streak = 0
            
            # 2. Build model if not built (built once at sequence start)
            if model is None:
                if args.override_num_classes is not None:
                    num_classes = args.override_num_classes
                elif hasattr(train_dataset, "stone_state_to_id") and train_dataset.stone_state_to_id is not None:
                    num_classes = len(train_dataset.stone_state_to_id)
                else:
                    num_classes = 108

                if args.model_architecture == "encoder":
                    model = create_classifier_model(
                        config_name=args.model_size,
                        src_vocab_size=len(train_dataset.word2idx),
                        num_classes=num_classes,
                        device=accelerator.device,
                        max_len=args.max_seq_len,
                        io_sep_token_id=train_dataset.io_sep_token_id if hasattr(train_dataset, 'io_sep_token_id') else None,
                        item_sep_token_id=train_dataset.item_sep_token_id if hasattr(train_dataset, 'item_sep_token_id') else None,
                        pooling_strategy=args.pooling_strategy
                    )
                elif args.model_architecture == "decoder":
                    model = create_decoder_classifier_model(
                        config_name=args.model_size,
                        src_vocab_size=len(train_dataset.word2idx),
                        num_classes=num_classes,
                        device=accelerator.device,
                        max_len=args.max_seq_len,
                        prediction_type=args.prediction_type,
                        padding_side=args.padding_side,
                        use_flash_attention=(args.use_flash_attention == 'True' or args.use_flash_attention is True),
                        batch_size=args.batch_size,
                        vocab=train_dataset.input_word2idx,
                        use_pre_norm=(args.use_pre_norm == 'True' or args.use_pre_norm is True)
                    )
                elif args.model_architecture == "linear":
                    model = create_linear_model(
                        config_name=args.model_size,
                        input_size=len(train_dataset.word2idx),
                        num_classes=num_classes,
                        device=accelerator.device,
                        max_len=args.max_seq_len,
                        io_sep_token_id=train_dataset.io_sep_token_id if hasattr(train_dataset, 'io_sep_token_id') else None,
                        item_sep_token_id=train_dataset.item_sep_token_id if hasattr(train_dataset, 'item_sep_token_id') else None,
                        pooling_strategy=args.pooling_strategy,
                        batch_size=args.batch_size,
                        use_flash_attention=(args.use_flash_attention == 'True' or args.use_flash_attention is True),
                        padding_side=args.padding_side,
                        include_nonlinearity=(args.include_nonlinearity == 'True' or args.include_nonlinearity is True),
                        flatten_input=(args.flatten_linear_model_input == 'True' or args.flatten_linear_model_input is True)
                    )
                else:
                    raise ValueError(f"Unknown architecture: {args.model_architecture}")
                    
                if args.freeze_layers:
                    _apply_freeze_layers_in_place(model, args.freeze_layers)
                    
                # First-time model preparation with accelerator
                model = accelerator.prepare(model)
                
            # 3. Setup optimizer and scheduler
            # Re-initialize on boundary if requested, or initialize if first task
            if (cycle_idx == 1 and task_idx == 0) or args.reset_optimizer:
                # Recreate optimizer using current optimizer parameter groupings
                if args.optimizer == 'adamw':
                    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
                elif args.optimizer == 'adam':
                    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
                elif args.optimizer == 'rmsprop':
                    optimizer = optim.RMSprop(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
                elif args.optimizer == 'adagrad':
                    optimizer = optim.Adagrad(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
                else:
                    raise ValueError(f"Unknown optimizer: {args.optimizer}")
                    
                # Recreate scheduler
                epochs_to_run = args.epochs_per_task if args.epochs_per_task is not None else args.epochs
                if args.use_scheduler:
                    if args.scheduler_type == "cosine":
                        if args.scheduler_call_location == "after_batch":
                            num_training_steps = epochs_to_run * len(train_dataloader)
                        else:
                            num_training_steps = epochs_to_run
                        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_training_steps, eta_min=args.eta_min)
                    elif args.scheduler_type == "exponential":
                        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.gamma)
                    else:
                        scheduler = None
                else:
                    scheduler = None
                    
                # Prepare optimizer & scheduler with Accelerator
                optimizer = accelerator.prepare(optimizer)
                if scheduler is not None:
                    scheduler = accelerator.prepare(scheduler)
                    
            # Prepare dataloaders for Accelerator
            # Note: val_dataloader is NOT prepared for evaluation mapping logic consistency
            train_dataloader = accelerator.prepare(train_dataloader)
            
            # Loss criterion
            if args.task_type == "seq2seq" or args.task_type == "seq2seq_stone_state":
                criterion = nn.CrossEntropyLoss(ignore_index=pad_token_id)
            elif args.task_type == "classification":
                criterion = nn.CrossEntropyLoss()
            elif args.task_type == "classification_multi_label":
                criterion = nn.CrossEntropyLoss(reduction=args.multi_label_reduction)
            else:
                raise ValueError(f"Unknown task type: {args.task_type}")
                
            # Target epochs budget
            epochs_limit = args.epochs_per_task if args.epochs_per_task is not None else args.epochs
            
            # Per-task predictions output subdirectory
            task_pred_dir = os.path.join(continual_save_dir, f"cycle_{cycle_idx}_task_{task_idx}_hop_{hop_length}", "predictions")
            if accelerator.is_local_main_process:
                os.makedirs(task_pred_dir, exist_ok=True)
                
            # ----------------------------------------------------
            # Epoch loop for this task
            # ----------------------------------------------------
            for epoch in range(epochs_limit):
                if accelerator.is_local_main_process:
                    print(f"Cycle {cycle_idx} | Task {task_idx} | Epoch {epoch + 1}/{epochs_limit}")
                    
                # Train one epoch
                train_loss, train_acc = train_epoch(
                    model=model,
                    dataloader=train_dataloader,
                    optimizer=optimizer,
                    criterion=criterion,
                    scheduler=scheduler,
                    accelerator=accelerator,
                    epoch_num=epoch,
                    pad_token_id=pad_token_id,
                    args=args
                )
                
                # Validate one epoch
                val_loss, val_metrics = validate_epoch(
                    model=model,
                    dataloader=val_dataloader,
                    criterion=criterion,
                    accelerator=accelerator,
                    epoch_num=epoch,
                    pad_token_id=pad_token_id,
                    args=args,
                    task_idx=task_idx,
                    hop_length=hop_length
                )
                
                # Save predictions at epoch boundary if configured
                if args.store_predictions and accelerator.is_local_main_process:
                    pred_path = os.path.join(task_pred_dir, f"epoch_{epoch + 1}.json")
                    with open(pred_path, "w") as f:
                        json.dump(val_metrics.get("predictions", []), f, indent=2)
                        
                # Logging with Task and Hop prefixes
                if accelerator.is_local_main_process:
                    current_lr = scheduler.get_last_lr()[0] if scheduler else optimizer.param_groups[0]['lr']
                    epoch_log = {}
                    prefix = f"continual/cycle_{cycle_idx}_task_{task_idx}_hop_{hop_length}/"
                    epoch_log[f"{prefix}train_loss"] = train_loss
                    epoch_log[f"{prefix}train_accuracy"] = train_acc
                    epoch_log[f"{prefix}val_loss"] = val_loss
                    
                    # Log generic current task validation metrics
                    epoch_log["continual/current_val_loss"] = val_loss
                    if "accuracy" in val_metrics:
                        epoch_log["continual/current_val_accuracy"] = val_metrics["accuracy"]
                        
                    # Also log other generic current metrics if present
                    for metric_name in ["P_A", "P_B_given_A", "P_C_given_AB"]:
                        if metric_name in val_metrics:
                            epoch_log[f"continual/current_{metric_name}"] = val_metrics[metric_name]
                            
                    # Flat global progress tracking metrics (for single continuous curves in W&B)
                    epoch_log["continual/global_epoch"] = global_epoch_counter
                    epoch_log["continual/learning_rate"] = current_lr
                    epoch_log["continual/current_train_loss"] = train_loss
                    epoch_log["continual/current_train_accuracy"] = train_acc
                    
                    # Copy accuracy flags and other scalar evaluation metrics
                    for key, val in val_metrics.items():
                        if key != "predictions" and isinstance(val, (int, float)):
                            epoch_log[f"{prefix}{key}"] = val
                    
                    # Hierarchical timeline logging
                    epoch_within_task_1indexed = epoch + 1
                    if "P_A" in val_metrics:
                        # epoch_log[f"cycle{cycle_idx}_task{task_idx}_h{hop_length}/epoch{epoch_within_task_1indexed}/P_A"] = val_metrics["P_A"]
                        # epoch_log[f"cycle{cycle_idx}_task{task_idx}_h{hop_length}/epoch{epoch_within_task_1indexed}/P_B_given_A"] = val_metrics["P_B_given_A"]
                        # epoch_log[f"cycle{cycle_idx}_task{task_idx}_h{hop_length}/epoch{epoch_within_task_1indexed}/P_C_given_AB"] = val_metrics["P_C_given_AB"]
                        
                        # # Also log with prefix just in case
                        # epoch_log[f"continual/cycle{cycle_idx}_task{task_idx}_h{hop_length}/epoch{epoch_within_task_1indexed}/P_A"] = val_metrics["P_A"]
                        # epoch_log[f"continual/cycle{cycle_idx}_task{task_idx}_h{hop_length}/epoch{epoch_within_task_1indexed}/P_B_given_A"] = val_metrics["P_B_given_A"]
                        # epoch_log[f"continual/cycle{cycle_idx}_task{task_idx}_h{hop_length}/epoch{epoch_within_task_1indexed}/P_C_given_AB"] = val_metrics["P_C_given_AB"]
                        
                        # Flat CSV Logging
                        if getattr(args, "log_continual_csv", True):
                            run_id = "unknown"
                            if wandb.run is not None:
                                run_id = wandb.run.id
                            elif getattr(args, "wandb_run_name", None) is not None:
                                run_id = args.wandb_run_name
                                
                            for csv_dir in [".", getattr(args, "save_dir", ".")]:
                                if csv_dir:
                                    from train import log_continual_metrics_csv
                                    log_continual_metrics_csv(
                                        os.path.join(csv_dir, "continual_metrics.csv"),
                                        run_id=run_id,
                                        task_idx=task_idx,
                                        hop_length=hop_length,
                                        global_epoch=global_epoch_counter + 1,
                                        epoch_within_task=epoch_within_task_1indexed,
                                        P_A=val_metrics["P_A"],
                                        P_B_given_A=val_metrics["P_B_given_A"],
                                        P_C_given_AB=val_metrics["P_C_given_AB"],
                                        cycle_idx=cycle_idx
                                    )
                            
                    epoch_log[f"{prefix}learning_rate"] = current_lr
                    epoch_log[f"{prefix}global_epoch"] = global_epoch_counter
                    epoch_log[f"{prefix}epoch_within_task"] = epoch
                    wandb.log(epoch_log)
                    
                    print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}")
                    if "accuracy" in val_metrics:
                        print(f"Val Accuracy: {val_metrics['accuracy']:.4f}")
                        
                    # Calculate and print ETA and remaining epoch stats
                    elapsed = time.time() - start_time
                    epochs_completed = global_epoch_counter + 1
                    
                    # Remaining epochs calculations
                    task_remaining = max(0, epochs_limit - (epoch + 1))
                    cycle_tasks_remaining = len(args.task_sequence) - (task_idx + 1)
                    cycle_remaining = max(0, cycle_tasks_remaining * epochs_limit + task_remaining)
                    global_remaining = max(0, (num_cycles - cycle_idx) * len(args.task_sequence) * epochs_limit + cycle_remaining)
                    
                    avg_time_per_epoch = elapsed / epochs_completed
                    eta = global_remaining * avg_time_per_epoch
                    
                    print(f"Time Elapsed: {format_time(elapsed)} | Estimated Remaining: {format_time(eta)}")
                    print(f"Remaining Epochs -> Task: {task_remaining} | Cycle {cycle_idx}: {cycle_remaining} | Global: {global_remaining}")
                
                # Increment global epoch counter after logging
                global_epoch_counter += 1
                        
                torch.cuda.empty_cache()
                gc.collect()
                
                # Check early stopping convergence
                convergence_stop_triggered_local = False
                if args.enable_auto_stop:
                    val_acc = val_metrics.get("accuracy", 0.0) if val_metrics is not None else 0.0
                    if val_acc >= float(args.auto_stop_val_acc_threshold):
                        val_acc_streak += 1
                    else:
                        val_acc_streak = 0
                    
                    if val_acc_streak >= int(args.auto_stop_val_acc_patience):
                        convergence_stop_triggered_local = True

                stop_flag = torch.tensor(
                    1 if convergence_stop_triggered_local else 0,
                    device=accelerator.device,
                    dtype=torch.int32,
                )
                stop_flag = accelerator.reduce(stop_flag, reduction="max")
                if int(stop_flag.item()) > 0:
                    if accelerator.is_local_main_process:
                        print(f"Stopping current task early due to val-accuracy convergence (streak={val_acc_streak}).")
                    break
                
            # End of task checkpointing
            if accelerator.is_local_main_process:
                unwrapped_model = accelerator.unwrap_model(model)
                checkpoint = {
                    'task_idx': task_idx,
                    'hop_length': hop_length,
                    'model_state_dict': unwrapped_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'args': args,
                    'src_vocab_word2idx': train_dataset.word2idx,
                    'src_vocab_idx2word': train_dataset.idx2word,
                    'continual_meta': {
                        'run_type': 'continual',
                        'cycle_idx': cycle_idx,
                        'task_idx': task_idx,
                        'hop_length': hop_length,
                        'epoch_within_task': epochs_limit - 1,
                        'global_epoch': global_epoch_counter - 1
                    }
                }
                if scheduler is not None:
                    checkpoint['scheduler_state_dict'] = scheduler.state_dict()
                    
                task_ckpt_path = os.path.join(continual_save_dir, f"model_cycle_{cycle_idx}_task_{task_idx}_hop_{hop_length}.pt")
                torch.save(checkpoint, task_ckpt_path)
                print(f"Saved task-specific checkpoint to {task_ckpt_path}")
                
            # Clear accelerator registries to prevent memory / worker process leak
            accelerator._dataloaders.clear()
            if args.reset_optimizer:
                accelerator._optimizers.clear()
                accelerator._schedulers.clear()
            
            # Explicitly delete dataloaders and datasets to free memory
            del train_dataloader
            del val_dataloader
            del train_dataset
            del val_dataset
            
            # Reclaim GPU & CPU memory
            torch.cuda.empty_cache()
            gc.collect()
            
    if accelerator.is_local_main_process:
        print("Continual training sequence complete.")
        wandb.finish()
        
    accelerator.end_training()

if __name__ == "__main__":
    main()
