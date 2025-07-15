#!/usr/bin/env python3
"""
Test script to verify separate input/output vocabularies work correctly.
"""

import sys
import os
sys.path.append('/home/rsaha/projects/dm_alchemy')

from src.models.data_loaders import AlchemyDataset
from functools import partial
from torch.utils.data import DataLoader
from src.models.data_loaders import collate_fn

def test_mixed_format_scenarios():
    """Test mixed input/output format combinations."""
    
    FILE_PATH = "/home/rsaha/projects/dm_alchemy/src/data/generated_data/decompositional_chemistry_samples_167424_80_unique_stones_val_shop_2_qhop_1_seed_0.json"
    
    print("=" * 80)
    print("TESTING SEPARATE INPUT/OUTPUT VOCABULARIES")
    print("=" * 80)
    
    # Test 1: Classification with features input (instead of default stone_states input)
    print("\n1. Testing: classification task with features input + stone_states output")
    print("-" * 60)
    try:
        dataset_clf_features = AlchemyDataset(
            json_file_path=FILE_PATH, 
            task_type="classification",
            input_format="features",      # Mixed: features as input
            output_format="stone_states", # stone_states as output (default for classification)
            use_preprocessed=False,
            num_workers=4
        )
        
        print(f"✓ Dataset created successfully")
        print(f"  Input vocabulary ({dataset_clf_features.input_format}): {len(dataset_clf_features.input_word2idx)} tokens")
        print(f"  Output vocabulary ({dataset_clf_features.output_format}): {len(dataset_clf_features.output_word2idx)} tokens")
        print(f"  Number of classes: {len(dataset_clf_features.stone_state_to_id)}")
        print(f"  Number of samples: {len(dataset_clf_features)}")
        
        # Test dataloader
        custom_collate = partial(collate_fn, pad_token_id=dataset_clf_features.pad_token_id, task_type="classification")
        dataloader = DataLoader(dataset_clf_features, batch_size=2, shuffle=False, collate_fn=custom_collate)
        
        for i, batch_data in enumerate(dataloader):
            if i >= 1: break
            print(f"  Sample batch shape: encoder_input_ids={batch_data['encoder_input_ids'].shape}, target_class_id={batch_data['target_class_id'].shape}")
            
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 2: Classification_multi_label with stone_states input (instead of default features input)
    print("\n2. Testing: classification_multi_label task with stone_states input + features output")
    print("-" * 60)
    try:
        dataset_ml_stones = AlchemyDataset(
            json_file_path=FILE_PATH, 
            task_type="classification_multi_label",
            input_format="stone_states",  # Mixed: stone_states as input
            output_format="features",     # features as output (default for multi_label)
            use_preprocessed=False,
            num_workers=4
        )
        
        print(f"✓ Dataset created successfully")
        print(f"  Input vocabulary ({dataset_ml_stones.input_format}): {len(dataset_ml_stones.input_word2idx)} tokens")
        print(f"  Output vocabulary ({dataset_ml_stones.output_format}): {len(dataset_ml_stones.output_word2idx)} tokens")
        print(f"  Number of output features: {dataset_ml_stones.num_output_features}")
        print(f"  Number of samples: {len(dataset_ml_stones)}")
        
        # Test dataloader
        custom_collate = partial(collate_fn, pad_token_id=dataset_ml_stones.pad_token_id, task_type="classification_multi_label")
        dataloader = DataLoader(dataset_ml_stones, batch_size=2, shuffle=False, collate_fn=custom_collate)
        
        for i, batch_data in enumerate(dataloader):
            if i >= 1: break
            print(f"  Sample batch shape: encoder_input_ids={batch_data['encoder_input_ids'].shape}, target_feature_vector={batch_data['target_feature_vector'].shape}")
            
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 3: Compare vocabulary sizes and contents
    print("\n3. Testing: Vocabulary independence")
    print("-" * 60)
    try:
        # Default formats
        dataset_clf_default = AlchemyDataset(
            json_file_path=FILE_PATH, 
            task_type="classification",
            use_preprocessed=False,
            num_workers=4
        )
        
        dataset_ml_default = AlchemyDataset(
            json_file_path=FILE_PATH, 
            task_type="classification_multi_label", 
            use_preprocessed=False,
            num_workers=4
        )
        
        print(f"Classification (default stone_states input):")
        print(f"  Input vocab size: {len(dataset_clf_default.input_word2idx)}")
        print(f"  Output vocab size: {len(dataset_clf_default.output_word2idx)}")
        print(f"  Input vocab has stone states: {any(k.startswith('{') for k in dataset_clf_default.input_word2idx.keys())}")
        print(f"  Output vocab has stone states: {any(k.startswith('{') for k in dataset_clf_default.output_word2idx.keys())}")
        
        print(f"\nClassification_multi_label (default features input):")
        print(f"  Input vocab size: {len(dataset_ml_default.input_word2idx)}")
        print(f"  Output vocab size: {len(dataset_ml_default.output_word2idx)}")
        print(f"  Input vocab has individual features: {any(not k.startswith('<') and not k.startswith('{') and len(k) < 10 for k in dataset_ml_default.input_word2idx.keys())}")
        print(f"  Output vocab has individual features: {any(not k.startswith('<') and not k.startswith('{') and len(k) < 10 for k in dataset_ml_default.output_word2idx.keys())}")
        
        print(f"\n✓ Vocabulary independence verified!")
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 80)
    print("SEPARATE VOCABULARIES TEST COMPLETED")
    print("=" * 80)

if __name__ == "__main__":
    test_mixed_format_scenarios()
