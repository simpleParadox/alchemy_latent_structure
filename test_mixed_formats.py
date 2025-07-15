#!/usr/bin/env python3

import sys
import os
sys.path.append('/home/rsaha/projects/dm_alchemy')

from src.models.data_loaders import AlchemyDataset

# Test file path
FILE_PATH = "/home/rsaha/projects/dm_alchemy/src/data/generated_data/compositional_chemistry_samples_167424_80_unique_stones_train_shop_1_qhop_2_seed_0.json"

def test_classification_with_features_input():
    """Test classification task with features input format (mixed vocab scenario)"""
    print("=" * 80)
    print("Testing classification task with features input format")
    print("=" * 80)
    
    try:
        dataset = AlchemyDataset(
            json_file_path=FILE_PATH, 
            task_type="classification",
            input_format="features",  # Use features input
            output_format="stone_states",  # Keep stone_states output (default for classification)
            num_workers=4,
            use_preprocessed=False,
            val_split=0.1,
            val_split_seed=42
        )
        
        print(f"Dataset created successfully!")
        print(f"Total examples: {len(dataset)}")
        print(f"Vocabulary size: {len(dataset.word2idx)}")
        print(f"Number of stone states: {len(dataset.stone_state_to_id)}")
        print(f"Input format: {dataset.input_format}")
        print(f"Output format: {dataset.output_format}")
        
        # Check a sample
        sample = dataset[0]
        print(f"\nSample structure:")
        for key, value in sample.items():
            if hasattr(value, 'shape'):
                print(f"  {key}: {value.shape} (tensor)")
            else:
                print(f"  {key}: {type(value)} - {value}")
        
        print(f"\nFirst few input tokens: {sample['encoder_input_ids'][:10]}")
        print(f"Target class: {sample['target']}")
        
        # Test that vocabulary contains both features and stone states
        print(f"\nVocabulary sample (first 10 items):")
        for i, (word, idx) in enumerate(list(dataset.word2idx.items())[:10]):
            print(f"  '{word}': {idx}")
            
        # Look for stone state tokens in vocabulary
        stone_state_tokens = [word for word in dataset.word2idx if word.startswith('stone_state_')]
        print(f"\nFound {len(stone_state_tokens)} stone state tokens in vocabulary")
        if stone_state_tokens:
            print(f"Example stone state tokens: {stone_state_tokens[:5]}")
            
        # Look for feature tokens in vocabulary  
        feature_tokens = [word for word in dataset.word2idx if not word.startswith('stone_state_') and word not in ['<pad>', '<sos>', '<eos>', '<cls>', '<sep>']]
        print(f"Found {len(feature_tokens)} feature tokens in vocabulary")
        if feature_tokens:
            print(f"Example feature tokens: {feature_tokens[:5]}")
        
        return True
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_classification_multi_label_with_stone_states_input():
    """Test classification_multi_label task with stone_states input format (mixed vocab scenario)"""
    print("=" * 80)
    print("Testing classification_multi_label task with stone_states input format")
    print("=" * 80)
    
    try:
        dataset = AlchemyDataset(
            json_file_path=FILE_PATH, 
            task_type="classification_multi_label",
            input_format="stone_states",  # Use stone_states input
            output_format="features",  # Keep features output (default for classification_multi_label)
            num_workers=4,
            use_preprocessed=False,
            val_split=0.1,
            val_split_seed=42
        )
        
        print(f"Dataset created successfully!")
        print(f"Total examples: {len(dataset)}")
        print(f"Vocabulary size: {len(dataset.word2idx)}")
        print(f"Number of output features: {dataset.num_output_features}")
        print(f"Input format: {dataset.input_format}")
        print(f"Output format: {dataset.output_format}")
        
        # Check a sample
        sample = dataset[0]
        print(f"\nSample structure:")
        for key, value in sample.items():
            if hasattr(value, 'shape'):
                print(f"  {key}: {value.shape} (tensor)")
            else:
                print(f"  {key}: {type(value)} - {value}")
        
        print(f"\nFirst few input tokens: {sample['encoder_input_ids'][:10]}")
        print(f"Target vector shape: {sample['target'].shape}")
        print(f"Target vector (first 10 elements): {sample['target'][:10]}")
        
        # Test that vocabulary contains both features and stone states
        print(f"\nVocabulary sample (first 10 items):")
        for i, (word, idx) in enumerate(list(dataset.word2idx.items())[:10]):
            print(f"  '{word}': {idx}")
            
        # Look for stone state tokens in vocabulary
        stone_state_tokens = [word for word in dataset.word2idx if word.startswith('stone_state_')]
        print(f"\nFound {len(stone_state_tokens)} stone state tokens in vocabulary")
        if stone_state_tokens:
            print(f"Example stone state tokens: {stone_state_tokens[:5]}")
            
        # Look for feature tokens in vocabulary  
        feature_tokens = [word for word in dataset.word2idx if not word.startswith('stone_state_') and word not in ['<pad>', '<sos>', '<eos>', '<cls>', '<sep>']]
        print(f"Found {len(feature_tokens)} feature tokens in vocabulary")
        if feature_tokens:
            print(f"Example feature tokens: {feature_tokens[:5]}")
        
        return True
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success1 = test_classification_with_features_input()
    print("\n" + "="*80 + "\n")
    success2 = test_classification_multi_label_with_stone_states_input()
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Classification with features input: {'✓ SUCCESS' if success1 else '✗ FAILED'}")
    print(f"Classification_multi_label with stone_states input: {'✓ SUCCESS' if success2 else '✗ FAILED'}")
    
    if success1 and success2:
        print("\n🎉 All mixed format tests passed!")
    else:
        print("\n❌ Some tests failed.")
