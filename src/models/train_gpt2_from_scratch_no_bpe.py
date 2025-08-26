#!/usr/bin/env python3
"""
Train a GPT-2 model from scratch on the Alchemy dataset for classification.
Uses HuggingFace Transformers and Trainer API with the existing custom vocabulary.
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
    Trainer, 
    TrainingArguments,
    EarlyStoppingCallback,
)
from transformers.modeling_outputs import SequenceClassifierOutput
from typing import Dict, List, Any, Optional
import wandb
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm


def set_seed(seed_value):
    """Set seed for reproducibility."""
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)


class SimpleTokenizer:
    """Simple tokenizer that uses the existing word2idx mapping from AlchemyDataset."""
    
    def __init__(self, word2idx: Dict[str, int], idx2word: Dict[int, str]):
        self.word2idx = word2idx
        self.idx2word = idx2word
        self.vocab_size = len(word2idx)
        
        # Set special token IDs based on existing vocabulary
        self.pad_token_id = word2idx.get("<pad>", 0)
        self.sos_token_id = word2idx.get("<sos>", 1)
        self.eos_token_id = word2idx.get("<eos>", 2)
        self.unk_token_id = word2idx.get("<unk>", 3)
        
        # For HuggingFace compatibility
        self.pad_token = "<pad>"
        self.sos_token = "<sos>"
        self.eos_token = "<eos>"
        self.unk_token = "<unk>"
        
    def __len__(self):
        return self.vocab_size
    
    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """Encode text to token IDs."""
        if isinstance(text, str):
            tokens = text.split()
        else:
            tokens = text
            
        token_ids = [self.word2idx.get(token, self.unk_token_id) for token in tokens]
        
        if add_special_tokens:
            token_ids = [self.sos_token_id] + token_ids + [self.eos_token_id]
            
        return token_ids
    
    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """Decode token IDs to text."""
        tokens = []
        for token_id in token_ids:
            token = self.idx2word.get(token_id, self.unk_token)
            if skip_special_tokens and token in [self.pad_token, self.sos_token, self.eos_token]:
                continue
            tokens.append(token)
        return " ".join(tokens)
    
    def convert_tokens_to_ids(self, tokens):
        """Convert tokens to IDs (for HuggingFace compatibility)."""
        if isinstance(tokens, str):
            return self.word2idx.get(tokens, self.unk_token_id)
        return [self.word2idx.get(token, self.unk_token_id) for token in tokens]
    
    def convert_ids_to_tokens(self, ids):
        """Convert IDs to tokens (for HuggingFace compatibility)."""
        if isinstance(ids, int):
            return self.idx2word.get(ids, self.unk_token)
        return [self.idx2word.get(id, self.unk_token) for id in ids]


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
        # For left-padded sequences, the last non-padded token is at the rightmost position
        if attention_mask is not None:
            # With left padding, find the last position with attention_mask == 1
            batch_size = input_ids.shape[0]
            seq_len = input_ids.shape[1]
            
            sequence_lengths = []
            for i in range(batch_size):
                # Find the last position where attention_mask is 1
                mask_positions = (attention_mask[i] == 1).nonzero(as_tuple=True)[0]
                if len(mask_positions) > 0:
                    last_pos = mask_positions[-1].item()  # Last position with attention = 1
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
    
    def __init__(self, data_path: str, vocab_path: str, tokenizer, max_length: int = 2048):
        """
        Args:
            data_path: Path to preprocessed data pickle file
            vocab_path: Path to vocabulary pickle file  
            tokenizer: Simple tokenizer using existing vocabulary
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
        
        # Use existing token IDs directly (no need to convert to text and back)
        encoder_input_ids = item['encoder_input_ids']
        
        # Filter out padding tokens and create attention mask
        non_pad_mask = [1 if token_id != 0 else 0 for token_id in encoder_input_ids]
        
        # Truncate if necessary
        if len(encoder_input_ids) > self.max_length:
            encoder_input_ids = encoder_input_ids[:self.max_length]
            non_pad_mask = non_pad_mask[:self.max_length]
        
        return {
            'input_ids': torch.tensor(encoder_input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(non_pad_mask, dtype=torch.long),
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


def compute_metrics(eval_pred):
    """Compute accuracy and F1 score for evaluation."""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average='weighted')
    
    # Save the predictions and targets to disk.
    output_dir = os.path.join(os.getcwd(), 'src/models/predictions')
    os.makedirs(output_dir, exist_ok=True)
    np.savez_compressed(os.path.join(output_dir, 'predictions.npz'), predictions)
    np.savez_compressed(os.path.join(output_dir, 'labels.npz'), labels)
    print(f"Saved predictions and labels to {output_dir}")
    print(f"Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}")
    
    return {
        'accuracy': accuracy,
        'f1': f1
    }


def main():
    # Set CUDA devices to use only GPUs 0,1,2,3
    # os.environ["CUDA_VISIBLE_DEVICES"] = "4" # For debugging on a single GPU.
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    
    parser = argparse.ArgumentParser(description="Train GPT-2 from scratch on Alchemy dataset")
    parser.add_argument("--train_data_path", type=str, 
                       default="src/data/preprocessed_separate/decompositional_chemistry_samples_167424_80_unique_stones_train_shop_2_qhop_1_seed_0_classification_filter_True_input_features_output_stone_states_data.pkl",
                       help="Path to training data")
    parser.add_argument("--train_vocab_path", type=str,
                       default="src/data/preprocessed_separate/decompositional_chemistry_samples_167424_80_unique_stones_train_shop_2_qhop_1_seed_0_classification_filter_True_input_features_output_stone_states_vocab.pkl", 
                       help="Path to training vocabulary")
    parser.add_argument("--val_data_path", type=str,
                       default="src/data/preprocessed_separate/decompositional_chemistry_samples_167424_80_unique_stones_val_shop_2_qhop_1_seed_0_classification_filter_True_input_features_output_stone_states_data.pkl",
                       help="Path to validation data")
    parser.add_argument("--val_vocab_path", type=str,
                       default="src/data/preprocessed_separate/decompositional_chemistry_samples_167424_80_unique_stones_val_shop_2_qhop_1_seed_0_classification_filter_True_input_features_output_stone_states_vocab.pkl",
                       help="Path to validation vocabulary")
    parser.add_argument("--output_dir", type=str, default="src/saved_models/gpt2_alchemy_classification_no_bpe",
                       help="Output directory for model and tokenizer")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Training batch size")
    parser.add_argument("--eval_batch_size", type=int, default=64, help="Evaluation batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--model_size", type=str, default="small", choices=["tiny", "small", "medium"],
                       help="Model size configuration")
    parser.add_argument("--use_wandb", action="store_true", help="Use Weights & Biases for logging")
    parser.add_argument("--wandb_project", type=str, default="gpt2-alchemy-classification-no-bpe",
                       help="W&B project name")
    
    args = parser.parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize W&B if requested
    if args.use_wandb:
        wandb.init(project=args.wandb_project, config=args)
    
    # Load vocabulary from the preprocessed data
    print("Loading vocabulary from preprocessed data...")
    with open(args.train_vocab_path, 'rb') as f:
        vocab_data = pickle.load(f)
    
    input_word2idx = vocab_data.get('input_word2idx', vocab_data['word2idx'])
    input_idx2word = vocab_data.get('input_idx2word', vocab_data['idx2word'])
    
    # Create simple tokenizer using existing vocabulary
    tokenizer = SimpleTokenizer(input_word2idx, input_idx2word)
    
    print(f"Tokenizer vocabulary size: {len(tokenizer)}")
    print(f"Special tokens - PAD: {tokenizer.pad_token_id}, SOS: {tokenizer.sos_token_id}, EOS: {tokenizer.eos_token_id}, UNK: {tokenizer.unk_token_id}")
    
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
    
    # Load number of classes from vocabulary
    num_classes = len(vocab_data['stone_state_to_id'])
    
    print(f"Creating GPT-2 model with {args.model_size} configuration")
    print(f"Number of classes: {num_classes}")
    
    # Create the model
    model = GPT2ForClassification(config, num_classes)
    
    # Print parameter count
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Create datasets
    print("Loading datasets...")
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
    
    # Create custom data collator with left padding for decoder model
    data_collator = CustomDataCollator(tokenizer, padding_side='right')
    
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
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=False,
        metric_for_best_model="eval_accuracy",
        greater_is_better=True,
        save_total_limit=3,
        report_to="wandb" if args.use_wandb else None,
        run_name=f"gpt2-{args.model_size}-alchemy-classification-no-bpe",
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
        compute_metrics=compute_metrics
    )
    
    # Train the model
    print("Starting training...")
    trainer.train()
    
    # Save the final model and vocabulary
    print("Saving final model and vocabulary...")
    trainer.save_model()
    
    # Save the tokenizer vocabulary for later use
    tokenizer_save_path = os.path.join(args.output_dir, 'tokenizer_vocab.pkl')
    with open(tokenizer_save_path, 'wb') as f:
        pickle.dump({
            'word2idx': tokenizer.word2idx,
            'idx2word': tokenizer.idx2word,
            'vocab_size': tokenizer.vocab_size,
            'special_tokens': {
                'pad_token_id': tokenizer.pad_token_id,
                'sos_token_id': tokenizer.sos_token_id,
                'eos_token_id': tokenizer.eos_token_id,
                'unk_token_id': tokenizer.unk_token_id,
            }
        }, f)
    
    # Final evaluation
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
