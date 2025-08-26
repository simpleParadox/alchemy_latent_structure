#!/usr/bin/env python3
"""
Train a GPT-2 model from scratch on the Alchemy dataset for classification.
Uses HuggingFace Transformers and Trainer API with a custom tokenizer trained on the dataset.
"""

import os
import json
import pickle
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    GPT2Config, 
    GPT2LMHeadModel,
    GPT2TokenizerFast,
    Trainer, 
    TrainingArguments,
    EarlyStoppingCallback,
    DataCollatorWithPadding
)
from transformers.modeling_outputs import SequenceClassifierOutput
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
from typing import Dict, List, Any, Optional
import wandb
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
import torch.distributed as dist


def set_seed(seed_value):
    """Set seed for reproducibility."""
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)


class GPT2ForClassification(GPT2LMHeadModel):
    """GPT-2 model adapted for classification tasks."""
    
    def __init__(self, config, num_labels):
        super().__init__(config)
        self.num_labels = num_labels
        
        # Replace the language modeling head with a classification head
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        
        # Initialize the classifier weights
        self.classifier.weight.data.normal_(mean=0.0, std=config.initializer_range)
        self.classifier.bias.data.zero_()
        
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        **kwargs
    ):
        # Get the base model outputs
        transformer_outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs
        )
        
        hidden_states = transformer_outputs[0]  # (batch_size, seq_len, hidden_size)
        
        # Use the last token's hidden state for classification (decoder-style)
        # For left-padded sequences in decoder models, the last non-padded token is always at the end
        if attention_mask is not None:
            # With left padding, the last non-padded token is at the rightmost position where attention_mask == 1
            # We need to find the last position where attention_mask is 1 for each sequence
            batch_size = input_ids.shape[0]
            seq_len = input_ids.shape[1]
            
            # For left padding, find the last position with attention_mask == 1
            # This should be the rightmost 1 in each row
            sequence_lengths = []
            for i in range(batch_size):
                # Find the last position where attention_mask is 1
                mask_positions = (attention_mask[i] == 1).nonzero(as_tuple=True)[0]
                if len(mask_positions) > 0:
                    # last_pos = mask_positions[-1].item()  # Last position with attention = 1
                    last_pos = mask_positions[-2].item()
                else:
                    last_pos = seq_len - 1  # Fallback to last position
                sequence_lengths.append(last_pos)
            
            sequence_lengths = torch.tensor(sequence_lengths, device=hidden_states.device)
            # Extract the hidden state of the last token for each sequence
            pooled_output = hidden_states[range(batch_size), sequence_lengths]
        else:
            # If no attention mask, use the last token
            pooled_output = hidden_states[:, -1]
        
        # Pass through classifier
        logits = self.classifier(pooled_output)
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
        
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )


class AlchemyDatasetForGPT2(Dataset):
    """Dataset wrapper for Alchemy data compatible with HuggingFace Trainer."""
    
    def __init__(self, data_path: str, vocab_path: str, tokenizer, max_length: int = 512):
        """
        Args:
            data_path: Path to preprocessed data pickle file
            vocab_path: Path to vocabulary pickle file  
            tokenizer: HuggingFace tokenizer
            max_length: Maximum sequence length
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Load preprocessed data
        with open(data_path, 'rb') as f:
            self.data = pickle.load(f)
            
        # Load vocabulary info for class mapping
        with open(vocab_path, 'rb') as f:
            vocab_data = pickle.load(f)
            
        self.stone_state_to_id = vocab_data['stone_state_to_id']
        self.input_word2idx = vocab_data.get('input_word2idx', vocab_data['word2idx'])
        self.input_idx2word = vocab_data.get('input_idx2word', vocab_data['idx2word'])
        
        print(f"Loaded {len(self.data)} samples")
        print(f"Number of classes: {len(self.stone_state_to_id)}")
        print(f"Input vocabulary size: {len(self.input_word2idx)}")
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Convert token IDs back to text for the HuggingFace tokenizer
        encoder_input_ids = item['encoder_input_ids']
        input_text = ' '.join([self.input_idx2word.get(token_id, '<unk>') 
                              for token_id in encoder_input_ids if token_id != 0])  # Skip padding
        
        # Tokenize with HuggingFace tokenizer
        encoding = self.tokenizer(
            input_text,
            truncation=False,
            padding=False,  # We'll pad in the collate function
            max_length=self.max_length,
            padding_side='left',  # Left padding for decoder models
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(item['target_class_id'], dtype=torch.long)
        }


class CustomDataCollator:
    """Custom data collator that handles left-padding for decoder models."""
    
    def __init__(self, tokenizer, padding_side='left'):
        self.tokenizer = tokenizer
        self.padding_side = padding_side
        
    def __call__(self, features):
        # Extract input_ids, attention_mask, and labels
        input_ids = [f['input_ids'] for f in features]
        attention_masks = [f['attention_mask'] for f in features]
        labels = torch.stack([f['labels'] for f in features])
        
        # Find max length in the batch
        max_length = max(len(ids) for ids in input_ids)
        
        # Pad sequences
        padded_input_ids = []
        padded_attention_masks = []
        
        for ids, mask in zip(input_ids, attention_masks):
            padding_length = max_length - len(ids)
            
            if self.padding_side == 'left':
                # Left padding for decoder models
                padded_ids = torch.cat([
                    torch.full((padding_length,), self.tokenizer.pad_token_id, dtype=ids.dtype),
                    ids
                ])
                padded_mask = torch.cat([
                    torch.zeros(padding_length, dtype=mask.dtype),
                    mask
                ])
            else:
                # Right padding
                padded_ids = torch.cat([
                    ids,
                    torch.full((padding_length,), self.tokenizer.pad_token_id, dtype=ids.dtype)
                ])
                padded_mask = torch.cat([
                    mask,
                    torch.zeros(padding_length, dtype=mask.dtype)
                ])
                
            padded_input_ids.append(padded_ids)
            padded_attention_masks.append(padded_mask)
        
        return {
            'input_ids': torch.stack(padded_input_ids),
            'attention_mask': torch.stack(padded_attention_masks),
            'labels': labels
        }


def build_tokenizer_from_alchemy_data(train_data_path: str, vocab_path: str, save_path: str):
    """Build and train a BPE tokenizer from the Alchemy dataset."""
    
    # Load vocabulary info
    with open(vocab_path, 'rb') as f:
        vocab_data = pickle.load(f)
    
    input_idx2word = vocab_data.get('input_idx2word', vocab_data['idx2word'])
    
    # Load training data
    with open(train_data_path, 'rb') as f:
        data = pickle.load(f)
    
    # Extract all text sequences for training the tokenizer
    texts = []
    for item in tqdm(data):
        encoder_input_ids = item['encoder_input_ids']
        # Convert back to text (skip padding tokens)
        text = ' '.join([input_idx2word.get(token_id, '<unk>') 
                        for token_id in encoder_input_ids if token_id != 0])
        if text.strip():  # Only add non-empty texts
            texts.append(text)
    
    print(f"Training tokenizer on {len(texts)} text sequences...")
    
    # Create a BPE tokenizer
    tokenizer = Tokenizer(BPE())
    tokenizer.pre_tokenizer = Whitespace()
    
    # Train the tokenizer
    trainer = BpeTrainer(
        vocab_size=1000,  # Reasonable vocab size for this domain
        min_frequency=2,
        special_tokens=["<pad>", "<sos>", "<eos>", "<unk>", "<io>", "<item_sep>"]
    )
    
    tokenizer.train_from_iterator(texts, trainer)
    
    # Add post-processing
    tokenizer.post_processor = TemplateProcessing(
        single="<sos> $A <eos>",
        special_tokens=[
            ("<sos>", tokenizer.token_to_id("<sos>")),
            ("<eos>", tokenizer.token_to_id("<eos>")),
        ],
    )
    
    # Save the tokenizer
    tokenizer.save(save_path)
    print(f"Tokenizer saved to {save_path}")
    
    return tokenizer


def compute_metrics(eval_pred):
    """Compute accuracy and F1 score for evaluation."""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average='weighted')
    
    return {
        'accuracy': accuracy,
        'f1': f1
    }


def is_main_process():
    """Check if this is the main process in distributed training."""
    return not dist.is_initialized() or dist.get_rank() == 0

def wait_for_everyone():
    """Wait for all processes to reach this point."""
    if dist.is_initialized():
        dist.barrier()


def main():
    # Set CUDA devices to use only GPUs 0,1,2,3
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    
    parser = argparse.ArgumentParser(description="Train GPT-2 from scratch on Alchemy dataset")
    parser.add_argument("--train_data_path", type=str, 
                       default="src/data/preprocessed_separate_enhanced_qnodes_in_snodes/decompositional_chemistry_samples_167424_80_unique_stones_train_shop_2_qhop_1_seed_0_classification_filter_True_input_features_output_stone_states_data.pkl",
                       help="Path to training data")
    parser.add_argument("--train_vocab_path", type=str,
                       default="src/data/preprocessed_separate_enhanced_qnodes_in_snodes/decompositional_chemistry_samples_167424_80_unique_stones_train_shop_2_qhop_1_seed_0_classification_filter_True_input_features_output_stone_states_vocab.pkl", 
                       help="Path to training vocabulary")
    parser.add_argument("--val_data_path", type=str,
                       default="src/data/preprocessed_separate_enhanced_qnodes_in_snodes/decompositional_chemistry_samples_167424_80_unique_stones_val_shop_2_qhop_1_seed_0_classification_filter_True_input_features_output_stone_states_data.pkl",
                       help="Path to validation data")
    parser.add_argument("--val_vocab_path", type=str,
                       default="src/data/preprocessed_separate_enhanced_qnodes_in_snodes/decompositional_chemistry_samples_167424_80_unique_stones_val_shop_2_qhop_1_seed_0_classification_filter_True_input_features_output_stone_states_vocab.pkl",
                       help="Path to validation vocabulary")
    parser.add_argument("--output_dir", type=str, default="src/saved_models/gpt2_alchemy_classification",
                       help="Output directory for model and tokenizer")
    parser.add_argument("--epochs", type=int, default=32, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Training batch size")
    parser.add_argument("--eval_batch_size", type=int, default=2, help="Evaluation batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--max_length", type=int, default=4096, help="Maximum sequence length")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--model_size", type=str, default="tiny", choices=["tiny", "small", "medium"],
                       help="Model size configuration")
    parser.add_argument("--use_wandb", action="store_true", help="Use Weights & Biases for logging")
    parser.add_argument("--wandb_project", type=str, default="gpt2-alchemy-classification",
                       help="W&B project name")
    
    args = parser.parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Create output directory (only main process)
    if is_main_process():
        os.makedirs(args.output_dir, exist_ok=True)
    
    # Wait for main process to create directory
    wait_for_everyone()
    
    # Initialize W&B (only main process)
    if args.use_wandb and is_main_process():
        wandb.init(project=args.wandb_project, config=args)
    
    # Build and save tokenizer (only main process)
    tokenizer_path = os.path.join(args.output_dir, "tokenizer.json")
    if is_main_process():
        if not os.path.exists(tokenizer_path):
            print("Building tokenizer from training data...")
            build_tokenizer_from_alchemy_data(args.train_data_path, args.train_vocab_path, tokenizer_path)
        else:
            print(f"Loading existing tokenizer from {tokenizer_path}")
    
    # Wait for main process to finish tokenizer creation/loading
    wait_for_everyone()
    
    # Load the tokenizer (all processes)
    tokenizer = GPT2TokenizerFast(tokenizer_file=tokenizer_path)
    tokenizer.pad_token = tokenizer.unk_token  # GPT-2 doesn't have a pad token by default
    
    if is_main_process():
        print(f"Tokenizer vocabulary size: {len(tokenizer)}")
    
    # Model configuration based on size
    model_configs = {
        "tiny": {
            "vocab_size": len(tokenizer),
            "n_positions": args.max_length,
            "n_embd": 256,
            "n_layer": 4,
            "n_head": 4,
            "n_inner": 512,
            "activation_function": "gelu_new",
            "resid_pdrop": 0.1,
            "embd_pdrop": 0.1,
            "attn_pdrop": 0.1,
            "layer_norm_epsilon": 1e-5,
            "initializer_range": 0.02,
        },
        "small": {
            "vocab_size": len(tokenizer),
            "n_positions": args.max_length,
            "n_embd": 512,
            "n_layer": 6,
            "n_head": 8,
            "n_inner": 2048,
            "activation_function": "gelu_new",
            "resid_pdrop": 0.1,
            "embd_pdrop": 0.1,
            "attn_pdrop": 0.1,
            "layer_norm_epsilon": 1e-5,
            "initializer_range": 0.02,
        },
        "medium": {
            "vocab_size": len(tokenizer),
            "n_positions": args.max_length,
            "n_embd": 768,
            "n_layer": 12,
            "n_head": 12,
            "n_inner": 3072,
            "activation_function": "gelu_new",
            "resid_pdrop": 0.1,
            "embd_pdrop": 0.1,
            "attn_pdrop": 0.1,
            "layer_norm_epsilon": 1e-5,
            "initializer_range": 0.02,
        }
    }
    
    config = GPT2Config(**model_configs[args.model_size])
    
    # Load number of classes from vocabulary (only main process for printing)
    with open(args.train_vocab_path, 'rb') as f:
        vocab_data = pickle.load(f)
    num_classes = 108 #len(vocab_data['stone_state_to_id'])
    
    if is_main_process():
        print(f"Creating GPT-2 model with {args.model_size} configuration")
        print(f"Number of classes: {num_classes}")
    
    # Create the model
    model = GPT2ForClassification(config, num_classes)
    
    # Print parameter count (only main process)
    if is_main_process():
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
    
    # Create datasets (all processes, but suppress prints for non-main)
    if is_main_process():
        print("Loading datasets...")
    
    # Temporarily redirect prints for non-main processes during dataset loading
    import sys
    import io
    original_stdout = sys.stdout
    if not is_main_process():
        sys.stdout = io.StringIO()  # Suppress prints for non-main processes
    
    train_dataset = AlchemyDatasetForGPT2(
        args.train_data_path, 
        args.train_vocab_path, 
        tokenizer, 
        args.max_length
    )
    
    val_dataset = AlchemyDatasetForGPT2(
        args.val_data_path,
        args.val_vocab_path,
        tokenizer,
        args.max_length
    )
    
    # Restore stdout for non-main processes
    if not is_main_process():
        sys.stdout = original_stdout
    
    # Create custom data collator with left padding for decoder model
    data_collator = CustomDataCollator(tokenizer, padding_side='left')
    
    # Training arguments with DDP support
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        logging_dir=os.path.join(args.output_dir, 'logs'),
        logging_steps=100,
        eval_steps=500,
        save_steps=500,
        eval_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=False,
        metric_for_best_model="eval_accuracy",
        greater_is_better=True,
        save_total_limit=10,
        report_to="wandb" if args.use_wandb else None,
        run_name=f"gpt2-{args.model_size}-alchemy-classification",
        dataloader_num_workers=10,
        remove_unused_columns=False,
        seed=args.seed,
        # DDP settings
        ddp_find_unused_parameters=False,
        dataloader_pin_memory=True,
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    
    # Train the model
    if is_main_process():
        print("Starting training...")
    trainer.train()
    
    # Save the final model and tokenizer (only main process)
    if is_main_process():
        print("Saving final model and tokenizer...")
        trainer.save_model()
        tokenizer.save_pretrained(args.output_dir)
    
    # Final evaluation (only main process)
    if is_main_process():
        print("Running final evaluation...")
        eval_results = trainer.evaluate()
        print("Final evaluation results:")
        for key, value in eval_results.items():
            print(f"  {key}: {value:.4f}")
        
        # Save evaluation results
        with open(os.path.join(args.output_dir, 'eval_results.json'), 'w') as f:
            json.dump(eval_results, f, indent=2)
        
        print(f"Training completed! Model saved to {args.output_dir}")


if __name__ == "__main__":
    main()
