#!/usr/bin/env python3
# filepath: preprocess_dataset.py
"""
Script to preprocess Alchemy datasets and save them to disk for faster loading.
This avoids memory issues and speeds up job startup times.
"""

import argparse
import json
import os
import pickle
import time
from typing import Dict, Any, List
import numpy as np
from tqdm import tqdm

# Import your existing classes
from src.models.data_loaders import AlchemyDataset


def preprocess_and_save_dataset(
    json_file_path: str,
    task_type: str,
    output_dir: str,
    vocab_word2idx: Dict[str, int] = None,
    vocab_idx2word: Dict[int, str] = None,
    stone_state_to_id: Dict[str, int] = None,
    filter_query_from_support: bool = False,
    num_workers: int = 4,
    chunk_size: int = 10000,
    input_format: str = None,
    output_format: str = None,
    num_query_samples: int = None
):
    """
    Preprocess a dataset and save it to disk.
    
    Args:
        json_file_path: Path to the input JSON file
        task_type: Type of task ("seq2seq", "classification", etc.)
        output_dir: Directory to save preprocessed files
        vocab_word2idx: Existing vocabulary (optional)
        vocab_idx2word: Existing vocabulary (optional)
        stone_state_to_id: Existing stone state mapping (optional)
        filter_query_from_support: Whether to filter query from support
        num_workers: Number of workers for multiprocessing
        chunk_size: Chunk size for processing
    """
    
    print(f"Starting preprocessing for {json_file_path}")
    print(f"Task type: {task_type}")
    print(f"Output directory: {output_dir}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    start_time = time.time()

    if output_format is None:
        # Default output format based on task type
        if task_type == 'seq2seq':
            output_format = 'features'
        elif task_type == 'classification':
            output_format = 'stone_states'
        elif task_type == 'classification_multi_label':
            output_format = 'features'
        elif task_type == 'seq2seq_stone_state':
            output_format = 'stone_states'
    
    # Create dataset (this will do all the preprocessing)
    dataset = AlchemyDataset(
        json_file_path=json_file_path,
        task_type=task_type,
        vocab_word2idx=vocab_word2idx,
        vocab_idx2word=vocab_idx2word,
        stone_state_to_id=stone_state_to_id,
        filter_query_from_support=filter_query_from_support,
        num_workers=num_workers,
        chunk_size=chunk_size,
        val_split=None, # Don't create splits during preprocessing
        use_preprocessed=False, # Initialize from scratch.
        input_format=input_format,
        output_format=output_format,
        num_query_samples=num_query_samples,
    )
    
    # Generate output filenames based on input file and parameters
    base_name = os.path.splitext(os.path.basename(json_file_path))[0]
    
    # Create a unique suffix based on parameters
    suffix_parts = [
        task_type,
        f"filter_{filter_query_from_support}",
    ]
    if input_format:
        suffix_parts.append(f"input_{input_format}")
    if output_format: 
        suffix_parts.append(f"output_{output_format}")
    else:
        # Default output format based on task type
        if task_type == 'seq2seq':
            suffix_parts.append("output_features")
        elif task_type == 'classification':
            suffix_parts.append("output_stone_states")
        elif task_type == 'classification_multi_label':
            suffix_parts.append("output_features")
        elif task_type == 'seq2seq_stone_state':
            suffix_parts.append("output_stone_states")
            
            
    suffix = "_".join(suffix_parts)
    
    # Save the preprocessed data
    data_file = os.path.join(output_dir, f"{base_name}_{suffix}_data.pkl")
    vocab_file = os.path.join(output_dir, f"{base_name}_{suffix}_vocab.pkl")
    metadata_file = os.path.join(output_dir, f"{base_name}_{suffix}_metadata.json")
    
    print(f"Saving preprocessed data to {data_file}")
    with open(data_file, 'wb') as f:
        pickle.dump(dataset.data, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    print(f"Saving vocabulary to {vocab_file}")
    vocab_data = {
        'word2idx': dataset.word2idx,
        'idx2word': dataset.idx2word,
        'pad_token_id': dataset.pad_token_id,
        'sos_token_id': dataset.sos_token_id,
        'eos_token_id': dataset.eos_token_id,
        'io_sep_token_id': dataset.io_sep_token_id,
        'item_sep_token_id': dataset.item_sep_token_id,
        'unk_token_id': dataset.unk_token_id,
        'stone_state_to_id': getattr(dataset, 'stone_state_to_id', None),
        'id_to_stone_state': getattr(dataset, 'id_to_stone_state', None),
        'all_output_features_list': getattr(dataset, 'all_output_features_list', None),
        'feature_to_idx_map_input': getattr(dataset, 'feature_to_idx_map_input', None),
        'feature_to_idx_map_output': getattr(dataset, 'feature_to_idx_map_output', None),
        'num_output_features': getattr(dataset, 'num_output_features', None),
        'input_word2idx': getattr(dataset, 'input_word2idx', None),
        'input_idx2word': getattr(dataset, 'input_idx2word', None),
        'output_word2idx': getattr(dataset, 'output_word2idx', None),
        'output_idx2word': getattr(dataset, 'output_idx2word', None),
    }
    
    with open(vocab_file, 'wb') as f:
        pickle.dump(vocab_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    # Save metadata for verification
    metadata = {
        'original_json_path': json_file_path,
        'task_type': task_type,
        'filter_query_from_support': filter_query_from_support,
        'input_format': input_format,
        'output_format': output_format,
        'num_samples': len(dataset.data),
        'vocab_size': len(dataset.word2idx),
        'num_classes': len(dataset.stone_state_to_id) if hasattr(dataset, 'stone_state_to_id') and dataset.stone_state_to_id else None,
        'num_output_features': getattr(dataset, 'num_output_features', None),
        'preprocessing_time': time.time() - start_time,
        'created_at': time.strftime('%Y-%m-%d %H:%M:%S'),
        'special_tokens': dataset.special_tokens,
    }
    
    print(f"Saving metadata to {metadata_file}")
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    end_time = time.time()
    print(f"Preprocessing completed in {end_time - start_time:.2f} seconds")
    print(f"Number of samples: {len(dataset.data)}")
    print(f"Vocabulary size: {len(dataset.word2idx)}")
    
    if hasattr(dataset, 'stone_state_to_id') and dataset.stone_state_to_id:
        print(f"Number of classes: {len(dataset.stone_state_to_id)}")
    
    if hasattr(dataset, 'num_output_features') and dataset.num_output_features:
        print(f"Number of output features: {dataset.num_output_features}")
    
    print(f"Files saved:")
    print(f"  Data: {data_file}")
    print(f"  Vocabulary: {vocab_file}")
    print(f"  Metadata: {metadata_file}")
    
    return {
        'data_file': data_file,
        'vocab_file': vocab_file,
        'metadata_file': metadata_file,
        'dataset': dataset
    }


def main():
    # python src/models/preprocess_dataset.py \
    # --train_json_file /home/rsaha/projects/dm_alchemy/src/data/generated_data/decompositional_chemistry_samples_167424_80_unique_stones_train_shop_5_qhop_1_seed_0.json \
    # --val_json_file /home/rsaha/projects/dm_alchemy/src/data/generated_data/decompositional_chemistry_samples_167424_80_unique_stones_val_shop_5_qhop_1_seed_0.json \
    # --task_type classification_multi_label \
    # --output_dir src/data/preprocessed_fixed_multi_label \
    # --filter_query_from_support \
    # --num_workers 11
    
    parser = argparse.ArgumentParser(description="Preprocess Alchemy datasets")
    parser.add_argument("--train_json_file", type=str, required=False,
                        # default="/home/rsaha/projects/dm_alchemy/src/data/complete_graph_generated_data_enhanced_qnodes_in_snodes/compositional_chemistry_samples_167424_80_unique_stones_train_shop_1_qhop_2_seed_.json",
                        # default="/home/rsaha/projects/dm_alchemy/src/data/subsampled_balanced_complete_graph_generated_data_enhanced_qnodes_in_snodes/compositional_chemistry_samples_167424_80_unique_stones_train_shop_1_qhop_5_seed_.json",
                        default="/home/rsaha/projects/dm_alchemy/src/data/shuffled_held_out_exps_generated_data_enhanced/compositional_chemistry_samples_167424_80_unique_stones_train_shop_1_qhop_1_single_held_out_color_4_edges_exp_seed_.json",
                        help="Path to the training JSON file")
    parser.add_argument("--val_json_file", type=str, required=False,
                        # default="/home/rsaha/projects/dm_alchemy/src/data/complete_graph_generated_data_enhanced_qnodes_in_snodes/compositional_chemistry_samples_167424_80_unique_stones_val_shop_1_qhop_2_seed_.json",
                        # default="/home/rsaha/projects/dm_alchemy/src/data/subsampled_balanced_complete_graph_generated_data_enhanced_qnodes_in_snodes/compositional_chemistry_samples_167424_80_unique_stones_val_shop_1_qhop_5_seed_.json",
                        default="/home/rsaha/projects/dm_alchemy/src/data/shuffled_held_out_exps_generated_data_enhanced/compositional_chemistry_samples_167424_80_unique_stones_val_shop_1_qhop_1_single_held_out_color_4_edges_exp_seed_.json",
                        help="Path to the validation JSON file")
    parser.add_argument("--task_type", type=str, required=False,
                        choices=["seq2seq", "classification", "classification_multi_label", "seq2seq_stone_state"],
                        default="classification",
                        help="Type of task")
    # parser.add_argument("--output_dir", type=str, default="src/data/subsampled_balanced_complete_graph_preprocessed_separate_enhanced_qnodes_in_snodes",
                        # help="Directory to save preprocessed files")
    parser.add_argument("--output_dir", type=str, default="src/data/shuffled_held_out_exps_preprocessed_separate_enhanced",
                        help="Directory to save preprocessed files")
    parser.add_argument("--filter_query_from_support", action="store_true", default=True,
                        help="Filter query examples from support sets")
    parser.add_argument("--num_workers", type=int, default=20,
                        help="Number of workers for multiprocessing")
    parser.add_argument("--chunk_size", type=int, default=60000,
                        help="Chunk size for processing")
    parser.add_argument("--input_format", type=str, default='features', choices=["stone_states", "features"],
                        help="Input format: 'stone_states' for complete states as tokens, 'features' for individual features as tokens. Default inferred from task_type.")
    parser.add_argument("--output_format", type=str, default='stone_states', choices=["stone_states", "features"],
                        help="Output format: 'stone_states' for classification targets, 'features' for multi-hot vectors. Default inferred from task_type.")
    parser.add_argument("--num_query_samples", type=int, default=None,
                        help="Number of query samples to use (for debugging). Default is None (use all).")
    
    args = parser.parse_args()
    
    print("="*60)
    print("PREPROCESSING TRAINING DATA")
    print("="*60)
    
    # Do some assertions here.
    if args.output_format == 'features':
        assert args.task_type in ['classification_multi_label', 'seq2seq']
    if args.output_format == 'stone_states':
        assert args.task_type in ['classification', 'seq2seq_stone_state']
    
    # for seed in [0, 1, 2]:
    for seed in [0,1,2,3,4]:
        train_json_file = args.train_json_file.replace(f"seed_", f"seed_{seed}")
        val_json_file = args.val_json_file.replace(f"seed_", f"seed_{seed}")
        # Preprocess training data first (this will create the vocabulary)
        train_result = preprocess_and_save_dataset(
            json_file_path=train_json_file,
            task_type=args.task_type,
            output_dir=args.output_dir,
            vocab_word2idx=None,  # Let training data create the vocabulary
            vocab_idx2word=None,
            stone_state_to_id=None,
            filter_query_from_support=args.filter_query_from_support,
            num_workers=args.num_workers,
            chunk_size=args.chunk_size,
            input_format=args.input_format,
            output_format=args.output_format,
            num_query_samples=args.num_query_samples
        )
        
        print("\n" + "="*60)
        print("PREPROCESSING VALIDATION DATA")
        print("="*60)
        
        # Load vocabulary from training data to ensure consistency
        print(f"Loading vocabulary from training data: {train_result['vocab_file']}")
        with open(train_result['vocab_file'], 'rb') as f:
            vocab_data = pickle.load(f)
            vocab_word2idx = vocab_data['word2idx']
            vocab_idx2word = vocab_data['idx2word']
            stone_state_to_id = vocab_data.get('stone_state_to_id')
        
        # Preprocess validation data using the training vocabulary
        val_result = preprocess_and_save_dataset(
            json_file_path=val_json_file,
            task_type=args.task_type,
            output_dir=args.output_dir,
            vocab_word2idx=vocab_word2idx,  # Use training vocabulary
            vocab_idx2word=vocab_idx2word,
            stone_state_to_id=stone_state_to_id,
            filter_query_from_support=args.filter_query_from_support,
            num_workers=args.num_workers,
            chunk_size=args.chunk_size,
            input_format=args.input_format,
            output_format=args.output_format,
            num_query_samples=args.num_query_samples
        )
        
        print("\n" + "="*60)
        print("PREPROCESSING COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"Training data processed: {len(train_result['dataset'].data)} samples")
        print(f"Validation data processed: {len(val_result['dataset'].data)} samples")
        print(f"Shared vocabulary size: {len(vocab_word2idx)}")
        
        if args.task_type == "classification":
            print(f"Number of classes: {len(stone_state_to_id) if stone_state_to_id else 'N/A'}")
        
        print(f"\nFiles created:")
        print(f"Training files:")
        print(f"  - Data: {train_result['data_file']}")
        print(f"  - Vocab: {train_result['vocab_file']}")
        print(f"  - Metadata: {train_result['metadata_file']}")
        print(f"Validation files:")
        print(f"  - Data: {val_result['data_file']}")
        print(f"  - Vocab: {val_result['vocab_file']}")
        print(f"  - Metadata: {val_result['metadata_file']}")
        print("="*60)


if __name__ == "__main__":
    main()