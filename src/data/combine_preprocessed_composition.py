#!/usr/bin/env python3
import os
import pickle
import json
import time

def combine_preprocessed_split(split, hops, input_dir, output_dir, seed=0):
    print(f"=== Combining {split} split data for seed {seed} ===")
    
    combined_data = []
    sample_breakdown = {}
    unified_vocab = None
    first_vocab_path = None
    
    # Filename components matching train/val files
    task_type = "classification"
    filter_val = "True"
    input_format = "features"
    output_format = "stone_states"
    
    for hop in hops:
        base_pattern = f"compositional_chemistry_samples_167424_80_unique_stones_{split}_shop_1_qhop_{hop}_seed_{seed}_{task_type}_filter_{filter_val}_input_{input_format}_output_{output_format}"
        
        data_path = os.path.join(input_dir, f"{base_pattern}_data.pkl")
        vocab_path = os.path.join(input_dir, f"{base_pattern}_vocab.pkl")
        metadata_path = os.path.join(input_dir, f"{base_pattern}_metadata.json")
        
        # Verify files exist
        if not all(os.path.exists(p) for p in [data_path, vocab_path, metadata_path]):
            print(f"Warning: Missing files for hop {hop}. Looked for:")
            print(f"  Data: {data_path}")
            print(f"  Vocab: {vocab_path}")
            print(f"  Metadata: {metadata_path}")
            continue
            
        # Load and append data
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
            print(f"Loaded hop {hop}: {len(data)} samples")
            combined_data.extend(data)
            sample_breakdown[f"qhop_{hop}"] = len(data)
            
        # Load and verify vocab
        with open(vocab_path, 'rb') as f:
            vocab = pickle.load(f)
            if unified_vocab is None:
                unified_vocab = vocab
                first_vocab_path = vocab_path
            else:
                # Assert vocabularies are equivalent
                assert vocab == unified_vocab, f"Vocabulary mismatch between {first_vocab_path} and {vocab_path}!"
                
    if not combined_data:
        print(f"No data found for {split} split!")
        return
        
    print(f"Total combined samples: {len(combined_data)}")
    
    # Build output filenames
    combined_base = f"compositional_chemistry_samples_167424_80_unique_stones_{split}_shop_1_qhop_2_3_4_5_seed_{seed}_{task_type}_filter_{filter_val}_input_{input_format}_output_{output_format}"
    
    out_data_path = os.path.join(output_dir, f"{combined_base}_data.pkl")
    out_vocab_path = os.path.join(output_dir, f"{combined_base}_vocab.pkl")
    out_metadata_path = os.path.join(output_dir, f"{combined_base}_metadata.json")
    
    # Save combined data
    print(f"Saving combined data to: {out_data_path}")
    with open(out_data_path, 'wb') as f:
        pickle.dump(combined_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        
    # Save unified vocabulary
    print(f"Saving vocabulary to: {out_vocab_path}")
    with open(out_vocab_path, 'wb') as f:
        pickle.dump(unified_vocab, f, protocol=pickle.HIGHEST_PROTOCOL)
        
    # Generate and save metadata
    combined_metadata = {
        'task_type': task_type,
        'filter_query_from_support': True,
        'input_format': input_format,
        'output_format': output_format,
        'num_samples': len(combined_data),
        'sample_breakdown': sample_breakdown,
        'vocab_size': len(unified_vocab['word2idx']),
        'num_classes': len(unified_vocab['stone_state_to_id']) if unified_vocab.get('stone_state_to_id') else None,
        'created_at': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    
    print(f"Saving metadata to: {out_metadata_path}")
    with open(out_metadata_path, 'w') as f:
        json.dump(combined_metadata, f, indent=2)
        
    print(f"=== Successfully completed {split} split combining ===\n")

def main():
    input_dir = "/home/rsaha/projects/def-afyshe-ab/rsaha/dm_alchemy/src/data/complete_graph_composition_fully_shuffled_balanced_grouped_by_unique_end_state_preprocessed"
    output_dir = "/home/rsaha/projects/def-afyshe-ab/rsaha/dm_alchemy/src/data/combined_composition_preprocessed"
    
    os.makedirs(output_dir, exist_ok=True)
    hops = [2, 3, 4, 5]
    seed = 0
    
    combine_preprocessed_split("train", hops, input_dir, output_dir, seed)
    combine_preprocessed_split("val", hops, input_dir, output_dir, seed)

if __name__ == "__main__":
    main()
