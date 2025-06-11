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


def process_episode_worker(args):
    """Worker function for processing a single episode. Must be at module level for multiprocessing."""
    (episode_id, episode_content, task_type, word2idx, stone_state_to_id, 
     special_token_ids, filter_query_from_support, 
     all_output_features_list, feature_to_idx_map) = args # Added new args for multi-label
    
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
        
        # Tokenize query example
        query_tokens = tokenize_single_example_str_helper(query_ex_str, word2idx, unk_token_id, io_sep_token_id)
        _, query_input_tokens, query_output_feature_tokens, query_initial_state_str, query_output_state_str = query_tokens
        
        if task_type == "seq2seq":
            # For seq2seq, use existing tokenization (features as individual tokens)
            tokenized_support_full_for_encoder = []
            for sup_ex_str in filtered_support_examples_str:
                full_sup_tokens, _, _, _, _ = tokenize_single_example_str_helper(sup_ex_str, word2idx, unk_token_id, io_sep_token_id)
                tokenized_support_full_for_encoder.append(full_sup_tokens)
            
            encoder_input_ids = []
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
            # For classification, convert stone states to single tokens
            tokenized_support_for_encoder = []
            for sup_ex_str in filtered_support_examples_str:
                # Parse the support example
                parts = sup_ex_str.split(" -> ")
                input_part_str = parts[0]
                output_state_str = parts[1]
                
                # Extract input stone state and potions
                first_brace_end = input_part_str.find("}")
                input_stone_state = input_part_str[:first_brace_end+1]
                potions_str = input_part_str[first_brace_end+1:].strip()
                
                # Convert input stone state to single token
                input_state_token_id = word2idx.get(input_stone_state, unk_token_id)
                
                # Convert potions to tokens 
                tokenized_potions = []
                if potions_str:
                    potions = potions_str.split(" ")
                    tokenized_potions = [word2idx.get(p, unk_token_id) for p in potions]
                
                # Convert output stone state to single token
                output_state_token_id = word2idx.get(output_state_str, unk_token_id)
                
                # Build full support example: input_state + potions + separator + output_state
                full_sup_tokens = [input_state_token_id] + tokenized_potions + [io_sep_token_id] + [output_state_token_id]
                tokenized_support_for_encoder.append(full_sup_tokens)
            
            # Process query input similarly - convert input state to single token
            query_parts = query_ex_str.split(" -> ")
            query_input_part_str = query_parts[0]
            query_first_brace_end = query_input_part_str.find("}")
            query_potions_str = query_input_part_str[query_first_brace_end+1:].strip()
            
            # Convert query input stone state to single token
            query_input_state_token_id = word2idx.get(query_initial_state_str, unk_token_id)
            
            # Convert query potions to tokens
            query_tokenized_potions = []
            if query_potions_str:
                query_potions = query_potions_str.split(" ")
                query_tokenized_potions = [word2idx.get(p, unk_token_id) for p in query_potions]
            
            query_input_tokens_classification = [query_input_state_token_id] + query_tokenized_potions
            
            # Build encoder input
            encoder_input_ids = []
            for i, sup_tokens in enumerate(tokenized_support_for_encoder):
                encoder_input_ids.extend(sup_tokens)
                if i < len(tokenized_support_for_encoder) - 1:
                    encoder_input_ids.append(item_sep_token_id)
            
            if tokenized_support_for_encoder:
                encoder_input_ids.append(item_sep_token_id)  # Separator before query
            encoder_input_ids.extend(query_input_tokens_classification)

            target_class_id = stone_state_to_id.get(query_output_state_str, -1)
            if target_class_id == -1:
                continue 
            processed_data.append({
                "encoder_input_ids": encoder_input_ids,
                "target_class_id": target_class_id
            })
            
        elif task_type == "classification_multi_label":
            # For multi-label classification, tokenize inputs like seq2seq (features as individual tokens)
            # but target is a multi-hot vector of output features.
            tokenized_support_full_for_encoder = []
            for sup_ex_str in filtered_support_examples_str:
                full_sup_tokens, _, _, _, _ = tokenize_single_example_str_helper(sup_ex_str, word2idx, unk_token_id, io_sep_token_id)
                tokenized_support_full_for_encoder.append(full_sup_tokens)
            
            encoder_input_ids = []
            for i, sup_tokens in enumerate(tokenized_support_full_for_encoder):
                encoder_input_ids.extend(sup_tokens)
                if i < len(tokenized_support_full_for_encoder) - 1:
                    encoder_input_ids.append(item_sep_token_id)
            
            if tokenized_support_full_for_encoder:
                encoder_input_ids.append(item_sep_token_id)  # Separator before query
            encoder_input_ids.extend(query_input_tokens) # query_input_tokens are already feature-based

            # Create multi-hot target vector for output features
            output_feature_strings = re.findall(r':\s*([\w-]+)', query_output_state_str)
            target_feature_vector = [0] * len(all_output_features_list)
            for feature_str in output_feature_strings:
                if feature_str in feature_to_idx_map:
                    target_feature_vector[feature_to_idx_map[feature_str]] = 1
            
            processed_data.append({
                "encoder_input_ids": encoder_input_ids,
                "target_feature_vector": target_feature_vector
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
                 task_type: str = "classification", # "seq2seq", "classification", "classification_multi_label"
                 vocab_word2idx: Dict[str, int] = None, 
                 vocab_idx2word: Dict[int, str] = None,
                 stone_state_to_id: Dict[str, int] = None, # For classification
                 filter_query_from_support: bool = False, # Filter query examples from support sets
                 num_workers: int = 4, # Number of workers for multiprocessing
                 chunk_size: int = 10000, # Number of episodes to process in each chunk (memory management)
                 val_split: Optional[float] = None, # Validation split ratio (e.g., 0.2 for 20%)
                 val_split_seed: int = 42): # Seed for reproducible splits
        
        self.task_type = task_type
        self.filter_query_from_support = filter_query_from_support # Only relevant if support hop length == query_hop_length.
        self.num_workers = max(1, num_workers)  # Ensure at least 1 worker
        self.chunk_size = max(1, chunk_size)  # Ensure at least 1 episode per chunk
        self.val_split = val_split
        self.val_split_seed = val_split_seed
        
        self.PAD_TOKEN_STR = "<pad>"
        self.SOS_TOKEN_STR = "<sos>"
        self.EOS_TOKEN_STR = "<eos>"
        self.IO_SEP_TOKEN_STR = "<io>"
        self.ITEM_SEP_TOKEN_STR = "<item_sep>"
        self.UNK_TOKEN_STR = "<unk>"

        self.special_tokens = [self.PAD_TOKEN_STR, self.SOS_TOKEN_STR, self.EOS_TOKEN_STR, 
                               self.IO_SEP_TOKEN_STR, self.ITEM_SEP_TOKEN_STR, self.UNK_TOKEN_STR]

        if vocab_word2idx and vocab_idx2word:
            self.word2idx = vocab_word2idx
            self.idx2word = vocab_idx2word
        else:
            if self.task_type == "classification":
                # For classification, build vocabulary with special tokens + potions + stone states
                self.word2idx, self.idx2word = self._build_classification_vocab(json_file_path)
            elif self.task_type == "seq2seq" or self.task_type == "classification_multi_label":
                # For seq2seq and classification_multi_label, build feature/potion vocabulary
                self.word2idx, self.idx2word = self._build_feature_potion_vocab(json_file_path)
            else:
                raise ValueError(f"Unknown task_type for vocab building: {self.task_type}")

        self.pad_token_id = self.word2idx[self.PAD_TOKEN_STR]
        self.sos_token_id = self.word2idx[self.SOS_TOKEN_STR]
        self.eos_token_id = self.word2idx[self.EOS_TOKEN_STR]
        self.io_sep_token_id = self.word2idx[self.IO_SEP_TOKEN_STR]
        self.item_sep_token_id = self.word2idx[self.ITEM_SEP_TOKEN_STR]
        self.unk_token_id = self.word2idx[self.UNK_TOKEN_STR]

        self.stone_state_to_id = None
        self.id_to_stone_state = None
        if self.task_type == "classification":
            if stone_state_to_id:
                self.stone_state_to_id = stone_state_to_id
                self.id_to_stone_state = {v: k for k, v in stone_state_to_id.items()}
            else:
                # For classification, stone states are already included in the unified vocabulary
                self.stone_state_to_id = {state: token_id for state, token_id in self.word2idx.items() 
                                         if state.startswith('{') and state.endswith('}')}
                self.id_to_stone_state = {v: k for k, v in self.stone_state_to_id.items()}
        
        # For classification_multi_label, we need a mapping for output features
        self.all_output_features_list = None
        self.feature_to_idx_map = None
        self.num_output_features = None
        if self.task_type == "classification_multi_label":
            all_features_in_dataset: Set[str] = set()
            with open(json_file_path, 'r') as f:
                raw_data_for_features = json.load(f)
            for episode_id, episode_content in raw_data_for_features["episodes"].items():
                if not episode_content: continue
                all_example_strings = episode_content.get("support", []) + episode_content.get("query", [])
                for example_str in all_example_strings:
                    # Use a simplified way to get output_state_str, then parse features
                    # This avoids calling the full tokenize_single_example_str_helper here if not needed
                    # or ensures all its dependencies are met if it were used.
                    try:
                        output_state_str = example_str.split(" -> ")[1]
                        features_output = re.findall(r':\s*([\w-]+)', output_state_str)
                        all_features_in_dataset.update(features_output)
                    except IndexError:
                        print(f"Warning: Malformed example string in episode {episode_id}: {example_str}")
            
            self.all_output_features_list = sorted(list(all_features_in_dataset))
            self.feature_to_idx_map = {feature: idx for idx, feature in enumerate(self.all_output_features_list)}
            self.num_output_features = len(self.all_output_features_list)
            print(f"Built output feature mapping for multi-label classification: {self.num_output_features} unique features.")

        self.data = self._load_and_preprocess_data(json_file_path, self.num_workers, self.chunk_size)
        
        # Create train/val splits if requested
        self.train_set = None
        self.val_set = None
        if self.val_split is not None:
            self._create_train_val_splits()

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
        all_output_features_list_arg = self.all_output_features_list if self.task_type == "classification_multi_label" else None
        feature_to_idx_map_arg = self.feature_to_idx_map if self.task_type == "classification_multi_label" else None

        episode_args = []
        for episode_id, episode_content in raw_data["episodes"].items():
            if episode_content:  # Skip empty episodes
                episode_args.append((
                    episode_id, episode_content, self.task_type, self.word2idx, 
                    self.stone_state_to_id, special_token_ids, self.filter_query_from_support,
                    all_output_features_list_arg, feature_to_idx_map_arg # Pass new args
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

def collate_fn(batch: List[Dict[str, torch.Tensor]], pad_token_id: int, task_type: str = "seq2seq") -> Dict[str, torch.Tensor]:
    encoder_inputs = [item["encoder_input_ids"] for item in batch]
    padded_encoder_inputs = torch.nn.utils.rnn.pad_sequence(encoder_inputs, batch_first=True, padding_value=pad_token_id)
    
    if task_type == "seq2seq":
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
        target_feature_vectors = torch.stack([item["target_feature_vector"] for item in batch])
        return {
            "encoder_input_ids": padded_encoder_inputs,
            "target_feature_vector": target_feature_vectors
        }
    else:
        raise ValueError(f"Unknown task_type in collate_fn: {task_type}")

if __name__ == '__main__':
    # FILE_PATH = "/home/rsaha/projects/dm_alchemy/src/data/chemistry_samples.json" 
    # Using the provided context file for testing
    FILE_PATH = "/home/rsaha/projects/dm_alchemy/src/data/generated_data/decompositional_chemistry_samples_val_shop_2_qhop_1.json"
    
    # Test Seq2Seq
    print("\n--- Testing Seq2Seq Task ---")
    try:
        dataset_s2s = AlchemyDataset(json_file_path=FILE_PATH, task_type="seq2seq")
        print(f"Dataset (Seq2Seq) initialized. Number of samples: {len(dataset_s2s)}")
        print(f"Feature/Potion Vocabulary size: {len(dataset_s2s.word2idx)}")
        
        custom_collate_s2s = partial(collate_fn, pad_token_id=dataset_s2s.pad_token_id, task_type="seq2seq")
        dataloader_s2s = DataLoader(dataset_s2s, batch_size=2, shuffle=False, collate_fn=custom_collate_s2s)
        print("DataLoader (Seq2Seq) created.")

        for i, batch_data in enumerate(dataloader_s2s):
            if i >= 1: break
            print(f"\n--- Seq2Seq Batch {i+1} --- ")
            print("Encoder Inputs Shape:", batch_data["encoder_input_ids"].shape)
            print("Decoder Inputs Shape:", batch_data["decoder_input_ids"].shape)
            print("Decoder Targets Shape:", batch_data["decoder_target_ids"].shape)
            if len(batch_data["encoder_input_ids"]) > 0:
                print("Encoder input (first sample):", batch_data["encoder_input_ids"][0])
                print("Decoder target (first sample):", batch_data["decoder_target_ids"][0])
        if len(dataset_s2s) == 0: print("WARNING: Seq2Seq dataset is empty.")
    except Exception as e:
        print(f"Error during Seq2Seq DataLoader testing: {e}")
        import traceback
        traceback.print_exc()

    # Test Classification
    print("\n\n--- Testing Classification Task ---")
    try:
        dataset_clf = AlchemyDataset(json_file_path=FILE_PATH, task_type="classification")
        print(f"Dataset (Classification) initialized. Number of samples: {len(dataset_clf)}")
        print(f"Feature/Potion Vocabulary size: {len(dataset_clf.word2idx)}")
        if dataset_clf.stone_state_to_id:
            print(f"Stone State Vocabulary (Num Classes): {len(dataset_clf.stone_state_to_id)}")
            # print("Sample stone_state_to_id:", list(dataset_clf.stone_state_to_id.items())[:5])

        custom_collate_clf = partial(collate_fn, pad_token_id=dataset_clf.pad_token_id, task_type="classification")
        dataloader_clf = DataLoader(dataset_clf, batch_size=2, shuffle=False, collate_fn=custom_collate_clf)
        print("DataLoader (Classification) created.")

        for i, batch_data in enumerate(dataloader_clf):
            if i >= 1: break
            print(f"\n--- Classification Batch {i+1} --- ")
            print("Encoder Inputs Shape:", batch_data["encoder_input_ids"].shape)
            print("Target Class ID Shape:", batch_data["target_class_id"].shape)
            if len(batch_data["encoder_input_ids"]) > 0:
                 print("Encoder input (first sample):", batch_data["encoder_input_ids"][0])
                 print("Target class ID (first sample):", batch_data["target_class_id"][0])

        if len(dataset_clf) == 0: print("WARNING: Classification dataset is empty.")

    except FileNotFoundError:
        print(f"ERROR: JSON data file not found at {FILE_PATH}. Please ensure the file exists.")
    except Exception as e:
        print(f"An error occurred during Classification DataLoader testing: {e}")
        import traceback
        traceback.print_exc()

    # Test Classification Multi-Label
    print("\n\n--- Testing Classification Multi-Label Task ---")
    try:
        dataset_ml = AlchemyDataset(json_file_path=FILE_PATH, task_type="classification_multi_label", num_workers=1) # Using 1 worker for easier debugging if needed
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

