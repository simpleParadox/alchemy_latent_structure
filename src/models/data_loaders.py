import torch
from torch.utils.data import Dataset, DataLoader, Subset
import json
import re
from typing import List, Dict, Tuple, Any, Set, Optional
from functools import partial
from tqdm import tqdm
import multiprocessing as mp
from multiprocessing import Pool
import random
import numpy as np
from collections import defaultdict
import pickle
import hashlib
import os

def process_episode_worker(args):
    """Worker function for processing a single episode. Must be at module level for multiprocessing."""
    (episode_id, episode_content, task_type, input_word2idx, output_word2idx, stone_state_to_id, 
     special_token_ids, filter_query_from_support, 
     all_output_features_list, feature_to_idx_map, input_format, output_format, model_architecture) = args
    
    if not episode_content:
        return []
    
    # Unpack special token IDs
    io_sep_token_id, item_sep_token_id, sos_token_id, eos_token_id, unk_token_id = special_token_ids
    
    support_examples_str = episode_content.get("support", [])
    query_examples_str = episode_content.get("query", [])
    processed_data = []
    
    for query_ex_str in query_examples_str:
        # Filter support examples. Essentially, we want to remove the query example from the support set if filter_query_from_support is True.
        filtered_support_examples_str = filter_support_examples_helper(
            support_examples_str, query_ex_str, filter_query_from_support)
        
        # Tokenize query example using input vocabulary
        query_tokens = tokenize_single_example_str_helper(query_ex_str, input_word2idx, unk_token_id, io_sep_token_id)
        _, query_input_tokens, query_output_feature_tokens, query_initial_state_str, query_output_state_str = query_tokens
        
        if task_type == "seq2seq":
            # For seq2seq, both input and output use features vocabulary
            tokenized_support_full_for_encoder = []
            for sup_ex_str in filtered_support_examples_str:
                full_sup_tokens, _, _, _, _ = tokenize_single_example_str_helper(sup_ex_str, input_word2idx, unk_token_id, io_sep_token_id)
                tokenized_support_full_for_encoder.append(full_sup_tokens)
            
            encoder_input_ids = []
            # Add item_sep_token_id between support examples.
            for i, sup_tokens in enumerate(tokenized_support_full_for_encoder):
                encoder_input_ids.extend(sup_tokens)
                if i < len(tokenized_support_full_for_encoder) - 1:
                    encoder_input_ids.append(item_sep_token_id)
            
            if tokenized_support_full_for_encoder:
                encoder_input_ids.append(item_sep_token_id)  # Separator before query
            encoder_input_ids.extend(query_input_tokens)

            decoder_input_ids = [sos_token_id] + query_output_feature_tokens
            decoder_target_ids = query_output_feature_tokens + [eos_token_id]
            processed_data.append({
                "encoder_input_ids": encoder_input_ids,
                "decoder_input_ids": decoder_input_ids,
                "decoder_target_ids": decoder_target_ids
            })
            
        elif task_type == "classification":
            # Process support examples based on input_format
            tokenized_support_for_encoder = []
            for sup_ex_str in filtered_support_examples_str:
                if input_format == "stone_states":
                    # Original classification approach: stone states as single tokens
                    parts = sup_ex_str.split(" -> ")
                    input_part_str = parts[0]
                    output_state_str = parts[1]
                    
                    # Extract input stone state and potions
                    first_brace_end = input_part_str.find("}")
                    input_stone_state = input_part_str[:first_brace_end+1]
                    potions_str = input_part_str[first_brace_end+1:].strip()
                    
                    # Convert input stone state to single token
                    input_state_token_id = input_word2idx.get(input_stone_state, unk_token_id)
                    
                    # Convert potions to tokens 
                    tokenized_potions = []
                    if potions_str:
                        potions = potions_str.split(" ")
                        tokenized_potions = [input_word2idx.get(p, unk_token_id) for p in potions]
                    
                    # For support examples, we still need to show the output for context
                    # Use input vocabulary for output stone state too (since it's part of input context)
                    output_state_token_id = input_word2idx.get(output_state_str, unk_token_id)
                    
                    # Build full support example: input_state + potions + separator + output_state
                    full_sup_tokens = [input_state_token_id] + tokenized_potions + [io_sep_token_id] + [output_state_token_id]
                    tokenized_support_for_encoder.append(full_sup_tokens)
                    
                elif input_format == "features":
                    # Feature input: use input vocabulary for features
                    full_sup_tokens, _, _, _, _ = tokenize_single_example_str_helper(sup_ex_str, input_word2idx, unk_token_id, io_sep_token_id)
                    tokenized_support_for_encoder.append(full_sup_tokens)
            
            # Process query input based on input_format
            if input_format == "stone_states":
                # Original classification query processing
                query_parts = query_ex_str.split(" -> ")
                query_input_part_str = query_parts[0]
                query_first_brace_end = query_input_part_str.find("}")
                query_potions_str = query_input_part_str[query_first_brace_end+1:].strip()
                
                # Convert query input stone state to single token
                query_input_state_token_id = input_word2idx.get(query_initial_state_str, unk_token_id)
                
                # Convert query potions to tokens
                query_tokenized_potions = []
                if query_potions_str:
                    query_potions = query_potions_str.split(" ")
                    query_tokenized_potions = [input_word2idx.get(p, unk_token_id) for p in query_potions]
                
                query_input_tokens_classification = [query_input_state_token_id] + query_tokenized_potions
                
            elif input_format == "features":
                # Feature-based query processing
                query_input_tokens_classification = query_input_tokens  # Already feature-based
            
            # Build encoder input
            encoder_input_ids = []
            for i, sup_tokens in enumerate(tokenized_support_for_encoder):
                encoder_input_ids.extend(sup_tokens)
                if i < len(tokenized_support_for_encoder) - 1:
                    encoder_input_ids.append(item_sep_token_id)
            
            if tokenized_support_for_encoder:
                encoder_input_ids.append(item_sep_token_id)  # Separator before query
            encoder_input_ids.extend(query_input_tokens_classification)
            

            # Output is always stone state classification
            target_class_id = stone_state_to_id.get(query_output_state_str, -1)
            if target_class_id == -1:
                print(f"Warning: Output state '{query_output_state_str}' not found in stone_state_to_id mapping for episode {episode_id}. Using -1 as target class ID.")
                continue 
            processed_data.append({
                "encoder_input_ids": encoder_input_ids,
                "target_class_id": target_class_id
            })
            
        elif task_type == "classification_multi_label":
            # Process support examples based on input_format
            tokenized_support_for_encoder = []
            for sup_ex_str in filtered_support_examples_str:
                if input_format == "features":
                    # Original multi-label approach: features as individual tokens
                    full_sup_tokens, _, _, _, _ = tokenize_single_example_str_helper(sup_ex_str, input_word2idx, unk_token_id, io_sep_token_id)
                    tokenized_support_for_encoder.append(full_sup_tokens)
                    
                elif input_format == "stone_states":
                    # Stone state approach: stone states as single tokens
                    parts = sup_ex_str.split(" -> ")
                    input_part_str = parts[0]
                    output_state_str = parts[1]
                    
                    # Extract input stone state and potions
                    first_brace_end = input_part_str.find("}")
                    input_stone_state = input_part_str[:first_brace_end+1]
                    potions_str = input_part_str[first_brace_end+1:].strip()
                    
                    # Convert input stone state to single token
                    input_state_token_id = input_word2idx.get(input_stone_state, unk_token_id)
                    
                    # Convert potions to tokens 
                    tokenized_potions = []
                    if potions_str:
                        potions = potions_str.split(" ")
                        tokenized_potions = [input_word2idx.get(p, unk_token_id) for p in potions]
                    
                    # For context, include output in input vocabulary
                    output_state_token_id = input_word2idx.get(output_state_str, unk_token_id)
                    
                    # Build full support example: input_state + potions + separator + output_state
                    full_sup_tokens = [input_state_token_id] + tokenized_potions + [io_sep_token_id] + [output_state_token_id]
                    tokenized_support_for_encoder.append(full_sup_tokens)
            
            # Process query input based on input_format
            if input_format == "features":
                # Feature-based query processing (original)
                query_input_tokens_multi_label = query_input_tokens  # Already feature-based
                
            elif input_format == "stone_states":
                # Stone state query processing
                query_parts = query_ex_str.split(" -> ")
                query_input_part_str = query_parts[0]
                query_first_brace_end = query_input_part_str.find("}")
                query_potions_str = query_input_part_str[query_first_brace_end+1:].strip()
                
                # Convert query input stone state to single token
                query_input_state_token_id = input_word2idx.get(query_initial_state_str, unk_token_id)
                
                # Convert query potions to tokens
                query_tokenized_potions = []
                if query_potions_str:
                    query_potions = query_potions_str.split(" ")
                    query_tokenized_potions = [input_word2idx.get(p, unk_token_id) for p in query_potions]
                
                query_input_tokens_multi_label = [query_input_state_token_id] + query_tokenized_potions
            
            # Build encoder input
            encoder_input_ids = []
            for i, sup_tokens in enumerate(tokenized_support_for_encoder):
                encoder_input_ids.extend(sup_tokens)
                if i < len(tokenized_support_for_encoder) - 1:
                    encoder_input_ids.append(item_sep_token_id)
            
            if tokenized_support_for_encoder:
                encoder_input_ids.append(item_sep_token_id)  # Separator before query
            encoder_input_ids.extend(query_input_tokens_multi_label)

            output_feature_strings = re.findall(r':\s*([\w-]+)', query_output_state_str)
            target_feature_vector = [0] * len(all_output_features_list) # Should be 13 features.
            feature_to_idx_map_input, feature_to_idx_map_output = feature_to_idx_map[0], feature_to_idx_map[1]
            for feature_str in output_feature_strings:
                if feature_str in feature_to_idx_map_output:
                    target_feature_vector[feature_to_idx_map_output[feature_str]] = 1
            
            # Validation checks for feature groups
            groups = [(0, 3), (3, 6), (6, 9), (9, 13)]
            for start, end in groups:
                group_features = target_feature_vector[start:end]
                if sum(group_features) > 1:
                    print(f"Warning: More than one feature activated in group {start}-{end} for episode {episode_id}. Features: {group_features}")
                if sum(group_features) == 0:
                    print(f"Warning: No features activated in group {start}-{end} for episode {episode_id}. Features: {group_features}")
                    
            target_feature_vector_autoregressive = []
            # For the autoregressive target vector, we convert features to tokens based on the 'input' vocabulary, not the output.
            for feature_str in output_feature_strings:
                target_feature_vector_autoregressive.append(
                    input_word2idx.get(feature_str, unk_token_id)
                )
            # Ensure no unk tokens in target vector. If present, throw warning.
            if unk_token_id in target_feature_vector_autoregressive:
                print(f"Warning: UNK token ({unk_token_id}) found in target feature vector for episode {episode_id}. Features: {target_feature_vector}")
                
            
            processed_data.append({
                "encoder_input_ids": encoder_input_ids,
                "target_feature_vector": target_feature_vector,
                "target_feature_vector_autoregressive": target_feature_vector_autoregressive
            })

        elif task_type == "seq2seq_stone_state":
            # For seq2seq_stone_state, use stone states as tokens (like classification)
            # but with seq2seq decoder structure
            tokenized_support_for_encoder = []
            for sup_ex_str in filtered_support_examples_str:
                # Parse the support example (same as classification)
                parts = sup_ex_str.split(" -> ")
                input_part_str = parts[0]
                output_state_str = parts[1]
                
                # Extract input stone state and potions
                first_brace_end = input_part_str.find("}")
                input_stone_state = input_part_str[:first_brace_end+1]
                potions_str = input_part_str[first_brace_end+1:].strip()
                
                # Convert input stone state to single token
                input_state_token_id = input_word2idx.get(input_stone_state, unk_token_id)
                
                # Convert potions to tokens 
                tokenized_potions = []
                if potions_str:
                    potions = potions_str.split(" ")
                    tokenized_potions = [input_word2idx.get(p, unk_token_id) for p in potions]
                
                # Convert output stone state to single token
                output_state_token_id = input_word2idx.get(output_state_str, unk_token_id)
                
                # Build full support example: input_state + potions + separator + output_state
                full_sup_tokens = [input_state_token_id] + tokenized_potions + [io_sep_token_id] + [output_state_token_id]
                tokenized_support_for_encoder.append(full_sup_tokens)
    
            # Process query input similarly 
            query_parts = query_ex_str.split(" -> ")
            query_input_part_str = query_parts[0]
            query_first_brace_end = query_input_part_str.find("}")
            query_potions_str = query_input_part_str[query_first_brace_end+1:].strip()            # Convert query input stone state to single token
            query_input_state_token_id = input_word2idx.get(query_initial_state_str, unk_token_id)

            # Convert query potions to tokens
            query_tokenized_potions = []
            if query_potions_str:
                query_potions = query_potions_str.split(" ")
                query_tokenized_potions = [input_word2idx.get(p, unk_token_id) for p in query_potions]
    
            query_input_tokens_stone_state = [query_input_state_token_id] + query_tokenized_potions
    
            # Build encoder input
            encoder_input_ids = []
            for i, sup_tokens in enumerate(tokenized_support_for_encoder):
                encoder_input_ids.extend(sup_tokens)
                if i < len(tokenized_support_for_encoder) - 1:
                    encoder_input_ids.append(item_sep_token_id)
    
            if tokenized_support_for_encoder:
                encoder_input_ids.append(item_sep_token_id)  # Separator before query
            encoder_input_ids.extend(query_input_tokens_stone_state)

            # For decoder output: use output vocabulary for stone state
            output_state_token_id = output_word2idx.get(query_output_state_str, unk_token_id)

            decoder_input_ids = [sos_token_id, output_state_token_id]  # [<sos>, stone_state]
            decoder_target_ids = [output_state_token_id, eos_token_id]  # [stone_state, <eos>]

            processed_data.append({
                "encoder_input_ids": encoder_input_ids,
                "decoder_input_ids": decoder_input_ids,
                "decoder_target_ids": decoder_target_ids
            })

    # Debugging: Check for UNK token in encoder_input_ids
    for item in processed_data:
        if "encoder_input_ids" in item and unk_token_id in item["encoder_input_ids"]:
            print(f"DEBUG: UNK token ({unk_token_id}) found in encoder_input_ids for episode {episode_id}. Example: {item['encoder_input_ids']}")

    return processed_data


def filter_support_examples_helper(support_examples, query_example, filter_query_from_support):
    """Helper function for filtering support examples."""
    if not filter_query_from_support:
        return support_examples
    
    filtered_support = []
    for sup_ex in support_examples:
        if sup_ex != query_example:
            filtered_support.append(sup_ex)
    return filtered_support


def tokenize_single_example_str_helper(example_str, word2idx, unk_token_id, io_sep_token_id):
    """Helper function for tokenizing a single example string."""
    parts = example_str.split(" -> ")
    input_part_str = parts[0]
    output_state_str = parts[1]

    first_brace_end = input_part_str.find("}")
    initial_state_str = input_part_str[:first_brace_end+1]
    potions_str = input_part_str[first_brace_end+1:].strip()

    # Parse stone string to tokens (extract features)
    features = re.findall(r':\s*([\w-]+)', initial_state_str)
    tokenized_initial_state = [word2idx.get(f, unk_token_id) for f in features]
    
    tokenized_potions = []
    if potions_str:
        potions = potions_str.split(" ")
        tokenized_potions = [word2idx.get(p, unk_token_id) for p in potions]

    # Parse output stone features
    features_output = re.findall(r':\s*([\w-]+)', output_state_str)
    tokenized_output_feature_tokens = [word2idx.get(f, unk_token_id) for f in features_output]
    
    full_example_tokens = tokenized_initial_state + tokenized_potions + [io_sep_token_id] + tokenized_output_feature_tokens
    query_input_tokens = tokenized_initial_state + tokenized_potions # Might be confusing because I do not add the io_sep_token_id here, but this is how it was done in Brenden Lake's paper.
    
    return full_example_tokens, query_input_tokens, tokenized_output_feature_tokens, initial_state_str, output_state_str


class AlchemyDataset(Dataset):
    def __init__(self, json_file_path: str, 
                 task_type: str = "classification",
                 vocab_word2idx: Dict[str, int] = None, 
                 vocab_idx2word: Dict[int, str] = None,
                 stone_state_to_id: Dict[str, int] = None,
                 filter_query_from_support: bool = False,
                 num_workers: int = 4,
                 chunk_size: int = 10000,
                 val_split: Optional[float] = None,
                 val_split_seed: int = 42,
                 preprocessed_dir: str = "src/data/preprocessed", 
                 use_preprocessed: bool = True,
                 input_format: Optional[str] = None,
                 output_format: Optional[str] = None,
                 model_architecture: str = "encoder"):
        
        self.task_type = task_type
        self.filter_query_from_support = filter_query_from_support
        self.num_workers = max(1, num_workers)
        self.chunk_size = max(1, chunk_size)
        self.val_split = val_split
        self.val_split_seed = val_split_seed
        self.preprocessed_dir = preprocessed_dir 
        self.use_preprocessed = use_preprocessed 
        self.architecture = model_architecture
        
        # Set default input/output formats based on task_type if not specified
        if input_format is None:
            if task_type in ["classification", "seq2seq_stone_state"]:
                self.input_format = "stone_states"
            elif task_type in ["seq2seq", "classification_multi_label"]:
                self.input_format = "features"
            else:
                raise ValueError(f"Unknown task_type: {task_type}")
        else:
            self.input_format = input_format
            
        if output_format is None:
            if task_type in ["classification", "seq2seq_stone_state"]:
                self.output_format = "stone_states"
            elif task_type in ["seq2seq", "classification_multi_label"]:
                self.output_format = "features"
            else:
                raise ValueError(f"Unknown task_type: {task_type}")
        else:
            self.output_format = output_format
        
        # Validate format combinations
        if self.input_format not in ["stone_states", "features"]:
            raise ValueError(f"input_format must be 'stone_states' or 'features', got {self.input_format}")
        if self.output_format not in ["stone_states", "features"]:
            raise ValueError(f"output_format must be 'stone_states' or 'features', got {self.output_format}")
        
        # Validate task_type and output_format compatibility
        if task_type == "classification" and self.output_format != "stone_states":
            raise ValueError(f"task_type 'classification' requires output_format 'stone_states', got {self.output_format}")
        if task_type == "classification_multi_label" and self.output_format != "features":
            raise ValueError(f"task_type 'classification_multi_label' requires output_format 'features', got {self.output_format}")
        
        print(f"Dataset configuration: task_type={task_type}, input_format={self.input_format}, output_format={self.output_format}")
        
        self.PAD_TOKEN_STR = "<pad>"
        self.SOS_TOKEN_STR = "<sos>"
        self.EOS_TOKEN_STR = "<eos>"
        self.IO_SEP_TOKEN_STR = "<io>"
        self.ITEM_SEP_TOKEN_STR = "<item_sep>"
        self.UNK_TOKEN_STR = "<unk>"

        self.special_tokens = [self.PAD_TOKEN_STR, self.SOS_TOKEN_STR, self.EOS_TOKEN_STR, 
                               self.IO_SEP_TOKEN_STR, self.ITEM_SEP_TOKEN_STR, self.UNK_TOKEN_STR]

        # Try to load preprocessed data first
        if self.use_preprocessed and self._load_preprocessed_data(json_file_path):
            print("Successfully loaded preprocessed data!")
        else:
            # Fallback to original preprocessing
            if self.use_preprocessed:
                print("Preprocessed data not found, falling back to original preprocessing...")
            else:
                print("Using original preprocessing (preprocessed data disabled)...")
            
            self._initialize_from_scratch(json_file_path, vocab_word2idx, vocab_idx2word, stone_state_to_id)

        # Create train/val splits if requested
        self.train_set = None
        self.val_set = None
        if self.val_split is not None:
            self._create_train_val_splits()

    def _generate_preprocessed_filename(self, json_file_path: str) -> tuple:
        """Generate filenames for preprocessed data based on parameters."""
        base_name = os.path.splitext(os.path.basename(json_file_path))[0]
        
        # Create a unique suffix based on parameters including new formats
        if (self.input_format is not None) or (self.output_format is not None):
            print("self.input_format:", self.input_format)
            print("self.output_format:", self.output_format)
            suffix_parts = [
                self.task_type,
                f"filter_{self.filter_query_from_support}",
                f"input_{self.input_format}",
                f"output_{self.output_format}",
            ]
        else:
            print("no input/output format specified, using defaults")
            suffix_parts = [
                self.task_type,
                f"filter_{self.filter_query_from_support}",
            ]
        suffix = "_".join(suffix_parts)
        
        data_file = os.path.join(self.preprocessed_dir, f"{base_name}_{suffix}_data.pkl")
        vocab_file = os.path.join(self.preprocessed_dir, f"{base_name}_{suffix}_vocab.pkl")
        metadata_file = os.path.join(self.preprocessed_dir, f"{base_name}_{suffix}_metadata.json")
        
        return data_file, vocab_file, metadata_file

    def _load_preprocessed_data(self, json_file_path: str) -> bool:
        """
        Try to load preprocessed data from disk.
        Returns True if successful, False otherwise.
        """
        data_file, vocab_file, metadata_file = self._generate_preprocessed_filename(json_file_path)
        
        # Check if all required files exist
        if not all(os.path.exists(f) for f in [data_file, vocab_file, metadata_file]):
            return False
        
        try:
            # Load and verify metadata
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            # Verify compatibility including new format parameters
            if (metadata['task_type'] != self.task_type or 
                metadata['filter_query_from_support'] != self.filter_query_from_support or
                metadata.get('input_format', 'default') != self.input_format or
                metadata.get('output_format', 'default') != self.output_format):
                print(f"Preprocessed data parameters don't match current settings:")
                print(f"  Expected: task_type={self.task_type}, filter={self.filter_query_from_support}, input_format={self.input_format}, output_format={self.output_format}")
                print(f"  Found: task_type={metadata['task_type']}, filter={metadata['filter_query_from_support']}, input_format={metadata.get('input_format', 'default')}, output_format={metadata.get('output_format', 'default')}")
                return False
            
            print(f"Loading preprocessed data from {data_file}")
            print(f"  Created: {metadata['created_at']}")
            print(f"  Samples: {metadata['num_samples']}")
            print(f"  Vocab size: {metadata['vocab_size']}")
            
            # Load vocabulary
            with open(vocab_file, 'rb') as f:
                vocab_data = pickle.load(f)
            
            # Handle both old (unified) and new (separate) vocabulary formats
            if 'input_word2idx' in vocab_data:
                # New format with separate vocabularies
                print("Using separate input/output vocabularies")
                self.input_word2idx = vocab_data['input_word2idx']
                self.input_idx2word = vocab_data['input_idx2word']
                self.output_word2idx = vocab_data['output_word2idx']
                self.output_idx2word = vocab_data['output_idx2word']
                # For backward compatibility
                self.word2idx = self.input_word2idx
                self.idx2word = self.input_idx2word
            else:
                # Old format with unified vocabulary
                self.word2idx = vocab_data['word2idx']
                self.idx2word = vocab_data['idx2word']
                # For new separate vocabulary approach, use unified vocab for both
                self.input_word2idx = self.word2idx
                self.input_idx2word = self.idx2word
                self.output_word2idx = self.word2idx
                self.output_idx2word = self.idx2word
            
            self.pad_token_id = vocab_data['pad_token_id']
            self.sos_token_id = vocab_data['sos_token_id']
            self.eos_token_id = vocab_data['eos_token_id']
            self.io_sep_token_id = vocab_data['io_sep_token_id']
            self.item_sep_token_id = vocab_data['item_sep_token_id']
            self.unk_token_id = vocab_data['unk_token_id']
            
            # Load task-specific data
            self.stone_state_to_id = vocab_data.get('stone_state_to_id')
            self.id_to_stone_state = vocab_data.get('id_to_stone_state')
            self.all_output_features_list = vocab_data.get('all_output_features_list')
            if self.output_format == "features":
                self.feature_to_idx_map_input = vocab_data.get('feature_to_idx_map_input')
                self.feature_to_idx_map_output = vocab_data.get('feature_to_idx_map_output')
            else:
                self.feature_to_idx_map = vocab_data.get('feature_to_idx_map')
            self.num_output_features = vocab_data.get('num_output_features')
            
            # Load preprocessed data
            with open(data_file, 'rb') as f:
                self.data = pickle.load(f)
            
            print(f"Successfully loaded {len(self.data)} preprocessed samples")
            return True
            
        except Exception as e:
            print(f"Error loading preprocessed data: {e}")
            return False

    def _initialize_from_scratch(self, json_file_path: str, vocab_word2idx: Dict[str, int] = None, 
                                vocab_idx2word: Dict[int, str] = None, stone_state_to_id: Dict[str, int] = None):
        """Initialize dataset from scratch with separate input/output vocabularies."""
        
        if vocab_word2idx and vocab_idx2word:
            # Use provided vocabulary for both input and output (unified approach)
            self.input_word2idx = vocab_word2idx
            self.input_idx2word = vocab_idx2word
            self.output_word2idx = vocab_word2idx  # Use same vocabulary for output when provided
            self.output_idx2word = vocab_idx2word
            # For backward compatibility, also set the old names
            self.word2idx = vocab_word2idx  
            self.idx2word = vocab_idx2word
        else:
            # Build separate input and output vocabularies
            self._build_separate_vocabularies(json_file_path)

        # Set token IDs from input vocabulary (since these are used for processing)
        self.pad_token_id = self.input_word2idx[self.PAD_TOKEN_STR]
        self.sos_token_id = self.input_word2idx[self.SOS_TOKEN_STR]
        self.eos_token_id = self.input_word2idx[self.EOS_TOKEN_STR]
        self.io_sep_token_id = self.input_word2idx[self.IO_SEP_TOKEN_STR]
        self.item_sep_token_id = self.input_word2idx[self.ITEM_SEP_TOKEN_STR]
        self.unk_token_id = self.input_word2idx[self.UNK_TOKEN_STR]

        # Initialize task-specific attributes
        self.stone_state_to_id = None
        self.id_to_stone_state = None
        if self.output_format == "stone_states":
            if stone_state_to_id:
                self.stone_state_to_id = stone_state_to_id
                self.id_to_stone_state = {v: k for k, v in stone_state_to_id.items()}
            else:
                # Build from output vocabulary
                self.stone_state_to_id = {state: idx for idx, state in enumerate(
                    [state for state in self.output_word2idx.keys() 
                     if state.startswith('{') and state.endswith('}')]
                )}
                self.id_to_stone_state = {v: k for k, v in self.stone_state_to_id.items()}
        
        # For multi-label features output
        self.all_output_features_list = None
        self.feature_to_idx_map_input = None
        self.feature_to_idx_map_output = None
        self.num_output_features = None
        if self.output_format == "features":
            # Build feature mappings from data
            all_features_in_dataset: Set[str] = set()
            with open(json_file_path, 'r') as f:
                raw_data_for_features = json.load(f)
            for episode_id, episode_content in raw_data_for_features["episodes"].items():
                if not episode_content: continue
                all_example_strings = episode_content.get("support", []) + episode_content.get("query", [])
                for example_str in all_example_strings:
                    try:
                        output_state_str = example_str.split(" -> ")[1]
                        features_output = re.findall(r':\s*([\w-]+)', output_state_str)
                        all_features_in_dataset.update(features_output)
                    except IndexError:
                        print(f"Warning: Malformed example string in episode {episode_id}: {example_str}")
            
            self.all_input_features_list = sorted(list(all_features_in_dataset))
            self.feature_to_idx_map_input = {feature: idx for idx, feature in enumerate(self.all_input_features_list)}
            
            # Hardcoded for multi-label classification to ensure consistent feature grouping
            self.feature_to_idx_map_output = {'blue': 0, 'red': 1, 'purple': 2,
                                              'small': 3, 'medium': 4, 'large': 5,
                                              'pointy': 6, 'round': 7, 'medium_round': 8,
                                              '-1': 9, '-3': 10, '1': 11, '3': 12}
            
            self.num_output_features = len(self.feature_to_idx_map_output)
            print(f"Built output feature mapping for multi-label classification: {self.num_output_features} unique features.")

        self.data = self._load_and_preprocess_data(json_file_path, self.num_workers, self.chunk_size)

    def _parse_stone_string_to_tokens(self, stone_str: str) -> List[int]:
        features = re.findall(r':\s*([\w-]+)', stone_str) 
        return [self.word2idx.get(f, self.unk_token_id) for f in features]

    def _get_output_stone_str(self, example_str: str) -> str:
        # Example: "{state_A} P1 P2 -> {state_B}"
        # Returns "{state_B}"
        return example_str.split(" -> ")[1]

    def _tokenize_single_example_str(self, example_str: str) -> Tuple[List[int], List[int], List[int], str]:
        # Returns: full_example_tokens, query_input_tokens, query_output_feature_tokens, query_output_state_str
        parts = example_str.split(" -> ")
        input_part_str = parts[0]
        output_state_str = parts[1] # This is the raw string for classification target

        first_brace_end = input_part_str.find("}")
        initial_state_str = input_part_str[:first_brace_end+1]
        potions_str = input_part_str[first_brace_end+1:].strip()

        tokenized_initial_state = self._parse_stone_string_to_tokens(initial_state_str)
        
        tokenized_potions: List[int] = []
        if potions_str:
            potions = potions_str.split(" ")
            tokenized_potions = [self.word2idx.get(p, self.unk_token_id) for p in potions]

        tokenized_output_feature_tokens = self._parse_stone_string_to_tokens(output_state_str)
        
        full_example_tokens = tokenized_initial_state + tokenized_potions + [self.io_sep_token_id] + tokenized_output_feature_tokens
        query_input_tokens = tokenized_initial_state + tokenized_potions
        
        return full_example_tokens, query_input_tokens, tokenized_output_feature_tokens, initial_state_str, output_state_str 

    def _build_feature_potion_vocab(self, json_file_path: str) -> Tuple[Dict[str, int], Dict[int, str]]:
        # Renamed from _build_vocab to be specific
        all_words: Set[str] = set()
        with open(json_file_path, 'r') as f:
            raw_data = json.load(f)
        for episode_id, episode_content in raw_data["episodes"].items():
            if not episode_content: continue # Skip empty episodes
            all_example_strings = episode_content.get("support", []) + episode_content.get("query", [])
            for example_str in all_example_strings:
                parts = example_str.split(" -> ")
                input_part_str = parts[0]
                output_state_str = parts[1]

                first_brace_end = input_part_str.find("}")
                initial_state_str = input_part_str[:first_brace_end+1]
                features_initial = re.findall(r':\s*([\w-]+)', initial_state_str)
                all_words.update(features_initial)

                potions_str = input_part_str[first_brace_end+1:].strip()
                if potions_str:
                    potions = potions_str.split(" ")
                    all_words.update(potions)
                
                features_output = re.findall(r':\s*([\w-]+)', output_state_str) # Also from output for completeness
                all_words.update(features_output)
        
        sorted_words = sorted(list(all_words))
        word2idx = {word: i for i, word in enumerate(sorted_words + self.special_tokens)}
        idx2word = {i: word for word, i in word2idx.items()}
        return word2idx, idx2word

    def _build_classification_vocab(self, json_file_path: str) -> Tuple[Dict[str, int], Dict[int, str]]:
        """Build vocabulary for classification: special tokens + potions + stone states (NO individual features)"""
        potions: Set[str] = set()
        unique_stone_states: Set[str] = set()
        
        with open(json_file_path, 'r') as f:
            raw_data = json.load(f)
        
        for episode_id, episode_content in raw_data["episodes"].items():
            if not episode_content: continue
            all_example_strings = episode_content.get("support", []) + episode_content.get("query", [])
            
            for example_str in all_example_strings:
                parts = example_str.split(" -> ")
                if len(parts) != 2:
                    continue
                    
                input_part_str = parts[0]
                output_state_str = parts[1]

                # Extract potions (but NOT individual features)
                first_brace_end = input_part_str.find("}")
                initial_state_str = input_part_str[:first_brace_end+1]
                potions_str = input_part_str[first_brace_end+1:].strip()
                
                if potions_str:
                    potion_list = potions_str.split(" ")
                    potions.update(potion_list)
                
                # Extract stone states (complete states as single tokens)
                unique_stone_states.add(initial_state_str)
                unique_stone_states.add(output_state_str)
        
        # Build vocabulary: special_tokens + potions + stone_states
        sorted_potions = sorted(list(potions))
        sorted_states = sorted(list(unique_stone_states))
        
        all_tokens = sorted_states + sorted_potions + self.special_tokens  # Order matters: states first. This is because later, the num_classes will be based on the stone states.
        word2idx = {token: i for i, token in enumerate(all_tokens)}
        idx2word = {i: token for token, i in word2idx.items()}
        
        print(f"Built classification vocabulary:")
        print(f"  Special tokens: {len(self.special_tokens)}")
        print(f"  Potions: {len(sorted_potions)}")
        print(f"  Stone states: {len(sorted_states)}")
        print(f"  Total vocabulary size: {len(word2idx)}")
        
        return word2idx, idx2word

    def _build_mixed_vocab(self, json_file_path: str) -> Tuple[Dict[str, int], Dict[int, str]]:
        """Build vocabulary that includes both individual features AND complete stone states."""
        all_features: Set[str] = set()
        all_potions: Set[str] = set()
        unique_stone_states: Set[str] = set()
        
        with open(json_file_path, 'r') as f:
            raw_data = json.load(f)
        
        for episode_id, episode_content in raw_data["episodes"].items():
            if not episode_content: continue
            all_example_strings = episode_content.get("support", []) + episode_content.get("query", [])
            
            for example_str in all_example_strings:
                parts = example_str.split(" -> ")
                if len(parts) != 2:
                    continue
                    
                input_part_str = parts[0]
                output_state_str = parts[1]

                # Extract features from input stone state
                first_brace_end = input_part_str.find("}")
                initial_state_str = input_part_str[:first_brace_end+1]
                features_initial = re.findall(r':\s*([\w-]+)', initial_state_str)
                all_features.update(features_initial)
                
                # Extract potions
                potions_str = input_part_str[first_brace_end+1:].strip()
                if potions_str:
                    potion_list = potions_str.split(" ")
                    all_potions.update(potion_list)
                
                # Extract features from output stone state
                features_output = re.findall(r':\s*([\w-]+)', output_state_str)
                all_features.update(features_output)
                
                # Extract complete stone states
                unique_stone_states.add(initial_state_str)
                unique_stone_states.add(output_state_str)
        
        # Build vocabulary: features + potions + stone_states + special_tokens
        sorted_features = sorted(list(all_features))
        sorted_potions = sorted(list(all_potions))
        sorted_states = sorted(list(unique_stone_states))
        
        all_tokens = sorted_features + sorted_potions + sorted_states + self.special_tokens
        word2idx = {token: i for i, token in enumerate(all_tokens)}
        idx2word = {i: token for token, i in word2idx.items()}
        
        print(f"Built mixed vocabulary:")
        print(f"  Features: {len(sorted_features)}")
        print(f"  Potions: {len(sorted_potions)}")
        print(f"  Stone states: {len(sorted_states)}")
        print(f"  Special tokens: {len(self.special_tokens)}")
        print(f"  Total vocabulary size: {len(word2idx)}")
        
        return word2idx, idx2word

    def _build_separate_vocabularies(self, json_file_path: str):
        """Build separate input and output vocabularies based on formats."""
        
        # Build input vocabulary
        if self.input_format == "stone_states":
            self.input_word2idx, self.input_idx2word = self._build_stone_state_potion_vocab(json_file_path, for_input=True)
        elif self.input_format == "features":
            self.input_word2idx, self.input_idx2word = self._build_feature_potion_vocab(json_file_path) # This adds the potions and the special tokens.
        
        # Build output vocabulary  
        if self.output_format == "stone_states":
            self.output_word2idx, self.output_idx2word = self._build_stone_state_vocab(json_file_path)
        elif self.output_format == "features":
            self.output_word2idx, self.output_idx2word = self._build_feature_vocab(json_file_path) # Only stone features, no potions or special tokens.
        
        # For backward compatibility, use input vocabulary as main vocabulary
        self.word2idx = self.input_word2idx
        self.idx2word = self.input_idx2word
        
        print(f"Built separate vocabularies:")
        print(f"  Input vocabulary ({self.input_format}): {len(self.input_word2idx)} tokens")
        print(f"  Output vocabulary ({self.output_format}): {len(self.output_word2idx)} tokens")

    def _build_stone_state_potion_vocab(self, json_file_path: str, for_input: bool = True) -> Tuple[Dict[str, int], Dict[int, str]]:
        """Build vocabulary for stone states and potions (used for input when input_format='stone_states')."""
        potions: Set[str] = set()
        unique_stone_states: Set[str] = set()
        
        with open(json_file_path, 'r') as f:
            raw_data = json.load(f)
        
        for episode_id, episode_content in raw_data["episodes"].items():
            if not episode_content: continue
            all_example_strings = episode_content.get("support", []) + episode_content.get("query", [])
            
            for example_str in all_example_strings:
                parts = example_str.split(" -> ")
                if len(parts) != 2:
                    continue
                    
                input_part_str = parts[0]
                output_state_str = parts[1]

                # Extract potions
                first_brace_end = input_part_str.find("}")
                initial_state_str = input_part_str[:first_brace_end+1]
                potions_str = input_part_str[first_brace_end+1:].strip()
                
                if potions_str:
                    potion_list = potions_str.split(" ")
                    potions.update(potion_list)
                
                # Extract stone states (for input, we need both input and output states for context)
                unique_stone_states.add(initial_state_str)
                if for_input:  # For input vocab, include output states for support examples
                    unique_stone_states.add(output_state_str)
        
        # Build vocabulary: stone_states + potions + special_tokens
        sorted_potions = sorted(list(potions))
        sorted_states = sorted(list(unique_stone_states))
        
        all_tokens = sorted_states + sorted_potions + self.special_tokens
        word2idx = {token: i for i, token in enumerate(all_tokens)}
        idx2word = {i: token for token, i in word2idx.items()}
        
        return word2idx, idx2word

    def _build_stone_state_vocab(self, json_file_path: str) -> Tuple[Dict[str, int], Dict[int, str]]:
        """Build vocabulary for stone states only (used for output when output_format='stone_states')."""
        unique_stone_states: Set[str] = set()
        
        with open(json_file_path, 'r') as f:
            raw_data = json.load(f)
        
        for episode_id, episode_content in raw_data["episodes"].items():
            if not episode_content: continue
            all_example_strings = episode_content.get("support", []) + episode_content.get("query", [])
            
            for example_str in all_example_strings:
                parts = example_str.split(" -> ")
                if len(parts) != 2:
                    continue
                    
                output_state_str = parts[1]
                unique_stone_states.add(output_state_str)
        
        # Build vocabulary: only stone_states (no special tokens needed for output vocab)
        sorted_states = sorted(list(unique_stone_states))
        word2idx = {state: i for i, state in enumerate(sorted_states)}
        idx2word = {i: state for state, i in word2idx.items()}
        
        return word2idx, idx2word

    def _build_feature_vocab(self, json_file_path: str) -> Tuple[Dict[str, int], Dict[int, str]]:
        """Build vocabulary for features only (used for output when output_format='features')."""
        all_features: Set[str] = set()
        
        with open(json_file_path, 'r') as f:
            raw_data = json.load(f)
        
        for episode_id, episode_content in raw_data["episodes"].items():
            if not episode_content: continue
            all_example_strings = episode_content.get("support", []) + episode_content.get("query", [])
            
            for example_str in all_example_strings:
                parts = example_str.split(" -> ")
                if len(parts) != 2:
                    continue
                    
                output_state_str = parts[1]
                features_output = re.findall(r':\s*([\w-]+)', output_state_str)
                all_features.update(features_output)
        
        # Build vocabulary: only features (no special tokens needed for output vocab)
        sorted_features = sorted(list(all_features))
        word2idx = {feature: i for i, feature in enumerate(sorted_features)}
        idx2word = {i: feature for feature, i in word2idx.items()}
        
        return word2idx, idx2word

    def _filter_support_examples(self, support_examples: List[str], query_examples: List[str]) -> List[str]:
        """
        Filter out support examples that appear in the query set to prevent data leakage.
        This is particularly important when support_steps=1 and query_steps=1.
        """
        if not support_examples or not query_examples:
            return support_examples
        
       
        # First check if the query example is a single string.
        if isinstance(query_examples, str):
            filtered_support = [ex for ex in support_examples if ex != query_examples]
        else:
            query_set = set(query_examples)
            filtered_support = [ex for ex in support_examples if ex not in query_set]
        
        # Log the filtering if any examples were removed
        # if len(filtered_support) < len(support_examples):
        #     removed_count = len(support_examples) - len(filtered_support)
            # print(f"Filtered {removed_count} support examples that appeared in query set "
            #       f"(original: {len(support_examples)}, filtered: {len(filtered_support)})")
        
        return filtered_support

    def _load_and_preprocess_data(self, json_file_path: str, num_workers: int, chunk_size: int) -> List[Dict[str, Any]]:
        with open(json_file_path, 'r') as f:
            raw_data = json.load(f)
        
        # Prepare arguments for multiprocessing
        special_token_ids = (self.io_sep_token_id, self.item_sep_token_id, 
                           self.sos_token_id, self.eos_token_id, self.unk_token_id)
        
        # Add feature mapping for multi-label task to worker args
        all_output_features_list_arg = list(self.feature_to_idx_map_output.keys()) if self.output_format == "features" else None
        feature_to_idx_map_arg = (self.feature_to_idx_map_input, self.feature_to_idx_map_output) if self.output_format == "features" else None

        episode_args = []
        for episode_id, episode_content in raw_data["episodes"].items():
            if episode_content:  # Skip empty episodes
                episode_args.append((
                    episode_id, episode_content, self.task_type, 
                    self.input_word2idx, self.output_word2idx,  # Pass separate vocabularies
                    self.stone_state_to_id, special_token_ids, self.filter_query_from_support,
                    all_output_features_list_arg, feature_to_idx_map_arg,
                    self.input_format, self.output_format, self.architecture
                ))
        
        # Use multiprocessing to process episodes in parallel
        processed_data: List[Dict[str, Any]] = []
        
        if num_workers > 1 and len(episode_args) > 1:
            # Process in chunks to manage memory usage
            for i in range(0, len(episode_args), chunk_size):
                chunk = episode_args[i:i + chunk_size]
                chunk_num = i // chunk_size + 1
                total_chunks = (len(episode_args) + chunk_size - 1) // chunk_size
                
                print(f"Processing chunk {chunk_num}/{total_chunks} ({len(chunk)} episodes) using {num_workers} workers...")
                
                with Pool(processes=min(num_workers, len(chunk))) as pool:
                    # Process episodes in parallel
                    results = list(tqdm(
                        pool.imap(process_episode_worker, chunk),
                        total=len(chunk),
                        desc=f"Processing chunk {chunk_num}/{total_chunks} ({self.task_type})",
                        unit="episode"
                    ))
                    
                    # Flatten results from all episodes in this chunk
                    for episode_results in results:
                        processed_data.extend(episode_results)
                
                print(f"Chunk {chunk_num}/{total_chunks} completed. Total examples so far: {len(processed_data)}")
        else:
            # Fallback to sequential processing for single worker or single episode
            print(f"Processing {len(episode_args)} episodes sequentially...")
            for args in tqdm(episode_args, desc=f"Processing episodes ({self.task_type})", unit="episode"):
                episode_results = process_episode_worker(args)
                processed_data.extend(episode_results)
        
        print(f"Total processed examples: {len(processed_data)}")
        return processed_data
    
    def _create_train_val_splits(self):
        """Create balanced train and validation splits from the loaded data."""
        if not (0.0 < self.val_split < 1.0):
            raise ValueError(f"val_split must be between 0 and 1, got {self.val_split}")
        
        if self.task_type == "classification": # Only classification needs balanced splits for now
            self._create_balanced_classification_splits()
        else: # seq2seq and classification_multi_label use random splits
            self._create_random_splits()
    
    def _create_balanced_classification_splits(self):
        """Create stratified splits for classification to ensure class balance."""
        # Set seed for reproducible splits
        random_state = np.random.RandomState(self.val_split_seed)
        
        # Group data by target class
        class_to_indices = defaultdict(list)
        for idx, item in enumerate(self.data):
            target_class_id = item["target_class_id"]
            class_to_indices[target_class_id].append(idx)
        
        train_indices = []
        val_indices = []
        
        print(f"Creating balanced splits with {self.val_split} validation ratio:")
        
        # For each class, split proportionally
        for class_id, indices in class_to_indices.items():
            random_state.shuffle(indices)
            
            val_size = max(1, int(len(indices) * self.val_split))  # Ensure at least 1 sample in val
            train_size = len(indices) - val_size
            
            train_indices.extend(indices[:train_size])
            val_indices.extend(indices[train_size:])
            
            stone_state = self.id_to_stone_state.get(class_id, f"Unknown({class_id})")
            print(f"  Class {class_id} ({stone_state}): {train_size} train, {val_size} val (total: {len(indices)})")
        
        # Create Subset objects
        self.train_set = Subset(self, train_indices)
        self.val_set = Subset(self, val_indices)
        
        # Verify class balance
        self._verify_class_balance()
        
        print(f"Created balanced train/val splits:")
        print(f"  Total samples: {len(self.data)}")
        print(f"  Train samples: {len(self.train_set)} ({len(self.train_set)/len(self.data)*100:.1f}%)")
        print(f"  Val samples: {len(self.val_set)} ({len(self.val_set)/len(self.data)*100:.1f}%)")
    
    def _create_random_splits(self):
        """Create random splits for seq2seq tasks."""
        random_state = np.random.RandomState(self.val_split_seed)
        
        indices = list(range(len(self.data)))
        random_state.shuffle(indices)
        
        val_size = int(len(self.data) * self.val_split)
        train_size = len(self.data) - val_size
        
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]
        
        self.train_set = Subset(self, train_indices)
        self.val_set = Subset(self, val_indices)
        
        print(f"Created random train/val splits:")
        print(f"  Total samples: {len(self.data)}")
        print(f"  Train samples: {len(self.train_set)} ({len(self.train_set)/len(self.data)*100:.1f}%)")
        print(f"  Val samples: {len(self.val_set)} ({len(self.val_set)/len(self.data)*100:.1f}%)")
    
    def _verify_class_balance(self):
        """Verify that all classes in train set are also present in validation set."""
        if self.task_type != "classification":
            return
        
        # Get class distributions
        train_classes = set()
        val_classes = set()
        
        for idx in self.train_set.indices:
            target_class_id = self.data[idx]["target_class_id"]
            train_classes.add(target_class_id)
        
        for idx in self.val_set.indices:
            target_class_id = self.data[idx]["target_class_id"]
            val_classes.add(target_class_id)
        
        # Check for class imbalance
        missing_in_val = train_classes - val_classes
        missing_in_train = val_classes - train_classes
        
        if missing_in_val:
            print(f"WARNING: {len(missing_in_val)} classes present in train but missing in validation:")
            for class_id in missing_in_val:
                stone_state = self.id_to_stone_state.get(class_id, f"Unknown({class_id})")
                print(f"  Class {class_id}: {stone_state}")
        
        if missing_in_train:
            print(f"WARNING: {len(missing_in_train)} classes present in validation but missing in train:")
            for class_id in missing_in_train:
                stone_state = self.id_to_stone_state.get(class_id, f"Unknown({class_id})")
                print(f"  Class {class_id}: {stone_state}")
        
        if not missing_in_val and not missing_in_train:
            print("✓ Class balance verified: All classes present in both train and validation sets")
        else:
            raise ValueError("Class imbalance detected! Train and validation sets must contain all the same classes.")
    
    def get_train_set(self):
        """Get the training subset. Returns self if no split was created."""
        return self.train_set if self.train_set is not None else self
    
    def get_val_set(self):
        """Get the validation subset. Returns None if no split was created."""
        return self.val_set
    
    def has_val_split(self) -> bool:
        """Check if validation split was created."""
        return self.val_set is not None

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single sample from the dataset."""
        if idx >= len(self.data):
            raise IndexError(f"Index {idx} is out of range for dataset of size {len(self.data)}")
        
        sample = self.data[idx]
        
        # Convert lists to tensors for the sample
        result = {}
        for key, value in sample.items():
            if isinstance(value, list):
                result[key] = torch.tensor(value, dtype=torch.long)
            else:
                result[key] = value
        
        return result

def collate_fn(batch: List[Dict[str, torch.Tensor]], pad_token_id: int, sos_token_id: int, eos_token_id: int, task_type: str = "seq2seq", model_architecture: str = "encoder", prediction_type: str = None,
               dataset=None, max_seq_len=2048) -> Dict[str, torch.Tensor]:
    """
    Collates data samples into a batch.
    Args:
        batch: A list of data samples.
        pad_token_id: The ID of the padding token.
        task_type: The type of task ("seq2seq", "classification", etc.).
        model_architecture: The model architecture ("encoder" or "decoder"). 
                           "decoder" uses left-padding, "encoder" uses right-padding.
    """
    encoder_inputs = [item["encoder_input_ids"] for item in batch]
    
    
    # Truncate sequences if they exceed max_seq_len
    if max_seq_len > 0:
        encoder_inputs = [seq[:max_seq_len] for seq in encoder_inputs]
    
    # Use right-padding for both encoder and decoder models.
    if model_architecture == "decoder":
        # Right-padding for decoder models
        # encoder_inputs = [torch.cat([torch.tensor([sos_token_id], dtype=seq.dtype), seq]) for seq in encoder_inputs] # First prepend the SOS token.
        max_len = max(len(seq) for seq in encoder_inputs)
        
        padded_sequences = []
        for seq in encoder_inputs:
            num_pads = max_len - len(seq)
            padded_seq = torch.cat([seq, torch.full((num_pads,), pad_token_id, dtype=seq.dtype)])
            padded_sequences.append(padded_seq)
        padded_encoder_inputs = torch.stack(padded_sequences)
    else:
        # Right-padding for encoder models (default behavior)
        padded_encoder_inputs = torch.nn.utils.rnn.pad_sequence(encoder_inputs, batch_first=True, padding_value=pad_token_id)
    
    if task_type == "seq2seq" or task_type == "seq2seq_stone_state":  # Handle both seq2seq variants
        decoder_inputs = [item["decoder_input_ids"] for item in batch]
        decoder_targets = [item["decoder_target_ids"] for item in batch]
        padded_decoder_inputs = torch.nn.utils.rnn.pad_sequence(decoder_inputs, batch_first=True, padding_value=pad_token_id)
        padded_decoder_targets = torch.nn.utils.rnn.pad_sequence(decoder_targets, batch_first=True, padding_value=pad_token_id)
        return {
            "encoder_input_ids": padded_encoder_inputs,
            "decoder_input_ids": padded_decoder_inputs,
            "decoder_target_ids": padded_decoder_targets
        }
    elif task_type == "classification":
        target_class_ids = torch.tensor([item["target_class_id"] for item in batch]) # Stack to form a batch of IDs
        return {
            "encoder_input_ids": padded_encoder_inputs,
            "target_class_id": target_class_ids
        }
    elif task_type == "classification_multi_label":
        # Target is a multi-hot vector, already a tensor in __getitem__
        # We just need to stack them into a batch tensor.
        if model_architecture == "decoder" and prediction_type == 'autoregressive':
            target_feature_vectors = torch.stack([item["target_feature_vector"] for item in batch])
            target_feature_vectors_autoregressive = [item['target_feature_vector_autoregressive'] for item in batch]
            
            # Append the EOS token to each target feature vector.
            target_feature_vectors_autoregressive = [torch.cat([vec, torch.tensor([eos_token_id], dtype=vec.dtype)]) for vec in target_feature_vectors_autoregressive]
            return {
                "encoder_input_ids": padded_encoder_inputs,
                "target_feature_vector_autoregressive": torch.nn.utils.rnn.pad_sequence(target_feature_vectors_autoregressive, batch_first=True, padding_value=pad_token_id)
            }
            
        return {
            "encoder_input_ids": padded_encoder_inputs,
            "target_feature_vector": target_feature_vectors # This is for the general multi-label classification task.
        }
    else:
        raise ValueError(f"Unknown task_type in collate_fn: {task_type}")

if __name__ == '__main__':
    # FILE_PATH = "/home/rsaha/projects/dm_alchemy/src/data/chemistry_samples.json" 
    # Using the provided context file for testing
    FILE_PATH = "/home/rsaha/projects/dm_alchemy/src/data/generated_data/decompositional_chemistry_samples_167424_80_unique_stones_train_shop_2_qhop_1_seed_0.json"
    
    # Test Seq2Seq
    # print("\n--- Testing Seq2Seq Task ---")
    # try:
    #     dataset_s2s = AlchemyDataset(json_file_path=FILE_PATH, task_type="seq2seq")
    #     print(f"Dataset (Seq2Seq) initialized. Number of samples: {len(dataset_s2s)}")
    #     print(f"Feature/Potion Vocabulary size: {len(dataset_s2s.word2idx)}")
        
    #     custom_collate_s2s = partial(collate_fn, pad_token_id=dataset_s2s.pad_token_id, task_type="seq2seq")
    #     dataloader_s2s = DataLoader(dataset_s2s, batch_size=2, shuffle=False, collate_fn=custom_collate_s2s)
    #     print("DataLoader (Seq2Seq) created.")

    #     for i, batch_data in enumerate(dataloader_s2s):
    #         if i >= 1: break
    #         print(f"\n--- Seq2Seq Batch {i+1} --- ")
    #         print("Encoder Inputs Shape:", batch_data["encoder_input_ids"].shape)
    #         print("Decoder Inputs Shape:", batch_data["decoder_input_ids"].shape)
    #         print("Decoder Targets Shape:", batch_data["decoder_target_ids"].shape)
    #         if len(batch_data["encoder_input_ids"]) > 0:
    #             print("Encoder input (first sample):", batch_data["encoder_input_ids"][0])
    #             print("Decoder target (first sample):", batch_data["decoder_target_ids"][0])
    #     if len(dataset_s2s) == 0: print("WARNING: Seq2Seq dataset is empty.")
    # except Exception as e:
    #     print(f"Error during Seq2Seq DataLoader testing: {e}")
    #     import traceback
    #     traceback.print_exc()

    # # Test Classification
    # print("\n\n--- Testing Classification Task ---")
    # try:
    #     dataset_clf = AlchemyDataset(json_file_path=FILE_PATH, task_type="classification")
    #     print(f"Dataset (Classification) initialized. Number of samples: {len(dataset_clf)}")
    #     print(f"Feature/Potion Vocabulary size: {len(dataset_clf.word2idx)}")
    #     if dataset_clf.stone_state_to_id:
    #         print(f"Stone State Vocabulary (Num Classes): {len(dataset_clf.stone_state_to_id)}")
    #         # print("Sample stone_state_to_id:", list(dataset_clf.stone_state_to_id.items())[:5])

    #     custom_collate_clf = partial(collate_fn, pad_token_id=dataset_clf.pad_token_id, task_type="classification")
    #     dataloader_clf = DataLoader(dataset_clf, batch_size=2, shuffle=False, collate_fn=custom_collate_clf)
    #     print("DataLoader (Classification) created.")

    #     for i, batch_data in enumerate(dataloader_clf):
    #         if i >= 1: break
    #         print(f"\n--- Classification Batch {i+1} --- ")
    #         print("Encoder Inputs Shape:", batch_data["encoder_input_ids"].shape)
    #         print("Target Class ID Shape:", batch_data["target_class_id"].shape)
    #         if len(batch_data["encoder_input_ids"]) > 0:
    #              print("Encoder input (first sample):", batch_data["encoder_input_ids"][0])
    #              print("Target class ID (first sample):", batch_data["target_class_id"][0])

    #     if len(dataset_clf) == 0: print("WARNING: Classification dataset is empty.")

    # except FileNotFoundError:
    #     print(f"ERROR: JSON data file not found at {FILE_PATH}. Please ensure the file exists.")
    # except Exception as e:
    #     print(f"An error occurred during Classification DataLoader testing: {e}")
    #     import traceback
    #     traceback.print_exc()

    # Test Classification Multi-Label
    print("\n\n--- Testing Classification Multi-Label Task ---")
    try:
        dataset_ml = AlchemyDataset(json_file_path=FILE_PATH, task_type="classification_multi_label", num_workers=20, use_preprocessed=False) # Using 1 worker for easier debugging if needed
        print(f"Dataset (Multi-Label) initialized. Number of samples: {len(dataset_ml)}")
        print(f"Feature/Potion Vocabulary size: {len(dataset_ml.word2idx)}")
        if dataset_ml.num_output_features:
            print(f"Number of output features for multi-label: {dataset_ml.num_output_features}")
            # print(f"  Feature to index map (sample): {list(dataset_ml.feature_to_idx_map.items())[:5]}")

        custom_collate_ml = partial(collate_fn, pad_token_id=dataset_ml.pad_token_id, task_type="classification_multi_label")
        dataloader_ml = DataLoader(dataset_ml, batch_size=2, shuffle=False, collate_fn=custom_collate_ml)
        print("DataLoader (Multi-Label) created.")

        for i, batch_data in enumerate(dataloader_ml):
            if i >= 1: break
            print(f"\n--- Multi-Label Batch {i+1} --- ")
            print("Encoder Inputs Shape:", batch_data["encoder_input_ids"].shape)
            print("Target Feature Vector Shape:", batch_data["target_feature_vector"].shape)
            if len(batch_data["encoder_input_ids"]) > 0:
                 print("Encoder input (first sample):", batch_data["encoder_input_ids"][0])
                 print("Target feature vector (first sample):", batch_data["target_feature_vector"][0])
                 # Find indices where target is 1
                 target_indices = (batch_data["target_feature_vector"][0] == 1).nonzero(as_tuple=True)[0]
                 if dataset_ml.all_output_features_list: # Ensure list is not None
                    target_features = [dataset_ml.all_output_features_list[idx.item()] for idx in target_indices]
                    print(f"  Target features (first sample): {target_features}")


        if len(dataset_ml) == 0: print("WARNING: Multi-Label dataset is empty.")

    except FileNotFoundError:
        print(f"ERROR: JSON data file not found at {FILE_PATH}. Please ensure the file exists.")
    except Exception as e:
        print(f"An error occurred during Multi-Label DataLoader testing: {e}")
        import traceback
        traceback.print_exc()

