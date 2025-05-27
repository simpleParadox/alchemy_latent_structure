#!/usr/bin/env python3

# Test script for validation split functionality
import sys
import os
sys.path.append('/home/rsaha/projects/dm_alchemy')

try:
    from src.models.data_loaders import AlchemyDataset
    
    # Test with chemistry_samples.json
    test_file = 'src/data/compositional_chemistry_samples_167424_aligned_stone_reward_shop_1_qhop_1.json'
    
    print(f'Testing validation split with file: {test_file}')
    
    # Test classification with validation split
    print('\n=== Testing Classification with Val Split ===')
    dataset = AlchemyDataset(
        json_file_path=test_file,
        task_type='classification',
        val_split=0.1,
        val_split_seed=42,
        num_workers=4
    )
    
    train_set = dataset.get_train_set()
    val_set = dataset.get_val_set()
    
    print(f'Total samples: {len(dataset)}')
    print(f'Train samples: {len(train_set)} ({len(train_set)/len(dataset)*100:.1f}%)')
    print(f'Val samples: {len(val_set)} ({len(val_set)/len(dataset)*100:.1f}%)')
    print(f'Has val split: {dataset.has_val_split()}')
    print(f'Number of stone state classes: {len(dataset.stone_state_to_id)}')
    
    # Test a few samples
    print(f'\nFirst train sample keys: {list(train_set[0].keys())}')
    print(f'First val sample keys: {list(val_set[0].keys())}')
    
    print('\n✓ Validation split test passed!')
    
except Exception as e:
    print(f'Error: {e}')
    import traceback
    traceback.print_exc()
