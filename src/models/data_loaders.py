import torch
from torch.utils.data import Dataset, DataLoader
import json
import re
from typing import List, Dict, Tuple, Any, Set
from functools import partial
from tqdm import tqdm

class AlchemyDataset(Dataset):
    def __init__(self, json_file_path: str, 
                 task_type: str = "seq2seq", # "seq2seq" or "classification"
                 vocab_word2idx: Dict[str, int] = None, 
                 vocab_idx2word: Dict[int, str] = None,
                 stone_state_to_id: Dict[str, int] = None, # For classification
                 filter_query_from_support: bool = False): # Filter query examples from support sets
        
        self.task_type = task_type
        self.filter_query_from_support = filter_query_from_support # Only relevant if support hop length == query_hop_length.
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
            else:
                # For seq2seq, build feature/potion vocabulary (includes individual features)
                self.word2idx, self.idx2word = self._build_feature_potion_vocab(json_file_path)

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
        
        self.data = self._load_and_preprocess_data(json_file_path)

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
        word2idx = {word: i for i, word in enumerate(self.special_tokens + sorted_words)}
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
        
        all_tokens = self.special_tokens + sorted_potions + sorted_states
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

    def _load_and_preprocess_data(self, json_file_path: str) -> List[Dict[str, Any]]:
        processed_data: List[Dict[str, Any]] = []
        with open(json_file_path, 'r') as f:
            raw_data = json.load(f)
        
        pbar = tqdm(total=len(raw_data["episodes"]), desc=f"Processing episodes ({self.task_type})", unit="episode")

        for episode_id, episode_content in raw_data["episodes"].items():
            if not episode_content: # Handle potentially empty episode entries
                pbar.update(1)
                continue
            support_examples_str = episode_content.get("support", [])
            query_examples_str = episode_content.get("query", [])

            for query_ex_str in query_examples_str:
                
                # First, filter the support examples based on the query example.
                filtered_support_examples_str = self._filter_support_examples(support_examples_str, query_ex_str)
                if len(filtered_support_examples_str) < len(support_examples_str):
                    pbar.set_postfix_str(f"Original support examples: {len(support_examples_str)}, filtered support examples {len(filtered_support_examples_str)}.")

                # Second, tokenize the query example.
                _, query_input_tokens, query_output_feature_tokens, query_initial_state_str, query_output_state_str = self._tokenize_single_example_str(query_ex_str)
                
                if self.task_type == "seq2seq":
                    # For seq2seq, use the existing tokenization (features as individual tokens)
                    tokenized_support_full_for_encoder: List[List[int]] = []
                    for sup_ex_str in filtered_support_examples_str:
                        # For encoder, we use the full "input -> output" representation of support examples
                        full_sup_tokens, _, _, _ = self._tokenize_single_example_str(sup_ex_str)
                        tokenized_support_full_for_encoder.append(full_sup_tokens)
                    
                    encoder_input_ids: List[int] = []
                    for i, sup_tokens in enumerate(tokenized_support_full_for_encoder):
                        encoder_input_ids.extend(sup_tokens)
                        if i < len(tokenized_support_full_for_encoder) - 1:
                            encoder_input_ids.append(self.item_sep_token_id)
                    
                    if tokenized_support_full_for_encoder:
                        encoder_input_ids.append(self.item_sep_token_id) # Separator before query
                    encoder_input_ids.extend(query_input_tokens)

                    decoder_input_ids = [self.sos_token_id] + query_output_feature_tokens
                    decoder_target_ids = query_output_feature_tokens + [self.eos_token_id]
                    processed_data.append({
                        "encoder_input_ids": encoder_input_ids,
                        "decoder_input_ids": decoder_input_ids,
                        "decoder_target_ids": decoder_target_ids
                    })
                    
                elif self.task_type == "classification":
                    # For classification, convert stone states to single tokens
                    tokenized_support_for_encoder: List[List[int]] = []
                    for sup_ex_str in filtered_support_examples_str:
                        # Parse the support example to get input state, potions, and output state
                        parts = sup_ex_str.split(" -> ")
                        input_part_str = parts[0]
                        output_state_str = parts[1]
                        
                        # Extract input stone state and potions
                        first_brace_end = input_part_str.find("}")
                        input_stone_state = input_part_str[:first_brace_end+1]
                        potions_str = input_part_str[first_brace_end+1:].strip()
                        
                        # Convert input stone state to single token
                        input_state_token_id = self.word2idx.get(input_stone_state, self.unk_token_id)
                        
                        # Convert potions to tokens 
                        tokenized_potions: List[int] = []
                        if potions_str:
                            potions = potions_str.split(" ")
                            tokenized_potions = [self.word2idx.get(p, self.unk_token_id) for p in potions]
                        
                        # Convert output stone state to single token
                        output_state_token_id = self.word2idx.get(output_state_str, self.unk_token_id)
                        
                        # Build full support example: input_state + potions + separator + output_state
                        full_sup_tokens = [input_state_token_id] + tokenized_potions + [self.io_sep_token_id] + [output_state_token_id]
                        tokenized_support_for_encoder.append(full_sup_tokens)
                    
                    # Process query input similarly - convert input state to single token
                    query_parts = query_ex_str.split(" -> ")
                    query_input_part_str = query_parts[0]
                    query_first_brace_end = query_input_part_str.find("}")
                    query_potions_str = query_input_part_str[query_first_brace_end+1:].strip()
                    
                    # Convert query input stone state to single token
                    query_input_state_token_id = self.word2idx.get(query_initial_state_str, self.unk_token_id)
                    
                    # Convert query potions to tokens
                    query_tokenized_potions: List[int] = []
                    if query_potions_str:
                        query_potions = query_potions_str.split(" ")
                        query_tokenized_potions = [self.word2idx.get(p, self.unk_token_id) for p in query_potions]
                    
                    query_input_tokens_classification = [query_input_state_token_id] + query_tokenized_potions
                    
                    # Build encoder input
                    encoder_input_ids: List[int] = []
                    for i, sup_tokens in enumerate(tokenized_support_for_encoder):
                        encoder_input_ids.extend(sup_tokens)
                        if i < len(tokenized_support_for_encoder) - 1:
                            encoder_input_ids.append(self.item_sep_token_id)
                    
                    if tokenized_support_for_encoder:
                        encoder_input_ids.append(self.item_sep_token_id) # Separator before query
                    encoder_input_ids.extend(query_input_tokens_classification)

                    target_class_id = self.stone_state_to_id.get(query_output_state_str, -1) # -1 for unknown state if strict
                    if target_class_id == -1:
                        # This should not happen if stone_state_to_id is built from the same dataset
                        print(f"Warning: Unknown stone state encountered in classification: {query_output_state_str}")
                        continue 
                    processed_data.append({
                        "encoder_input_ids": encoder_input_ids,
                        "target_class_id": target_class_id
                    })
            pbar.update(1)
        pbar.close()
        return processed_data
    
    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx] # self.data has the data that is returned from _load_and_preprocess_data.
        if self.task_type == "seq2seq":
            return {
                "encoder_input_ids": torch.tensor(item["encoder_input_ids"], dtype=torch.long),
                "decoder_input_ids": torch.tensor(item["decoder_input_ids"], dtype=torch.long),
                "decoder_target_ids": torch.tensor(item["decoder_target_ids"], dtype=torch.long)
            }
        elif self.task_type == "classification":
            return {
                "encoder_input_ids": torch.tensor(item["encoder_input_ids"], dtype=torch.long),
                "target_class_id": torch.tensor(item["target_class_id"], dtype=torch.long) # Single ID
            }
        else:
            raise ValueError(f"Unknown task_type: {self.task_type}")

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
        target_class_ids = torch.stack([item["target_class_id"] for item in batch]) # Stack to form a batch of IDs
        return {
            "encoder_input_ids": padded_encoder_inputs,
            "target_class_id": target_class_ids
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

