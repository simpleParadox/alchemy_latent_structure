#!/usr/bin/env python3
"""
Test script to demonstrate the chunking functionality for memory management.
"""

from src.models.data_loaders import AlchemyDataset

def test_chunking():
    """Test the chunking functionality with different chunk sizes."""
    
    FILE_PATH = "/home/rsaha/projects/dm_alchemy/src/data/generated_data/decompositional_chemistry_samples_167424_80_unique_stones_val_shop_2_qhop_1.json"
    
    print("=== Testing Chunking Functionality ===\n")
    
    # Test with small chunk size for memory efficiency
    print("1. Testing with small chunk size (50 episodes per chunk):")
    try:
        dataset_small_chunks = AlchemyDataset(
            json_file_path=FILE_PATH, 
            task_type="classification",
            num_workers=2,
            chunk_size=50  # Small chunks for memory efficiency
        )
        print(f"   ✓ Successfully loaded {len(dataset_small_chunks)} samples with small chunks\n")
    except Exception as e:
        print(f"   ✗ Error with small chunks: {e}\n")
    
    # Test with larger chunk size
    print("2. Testing with larger chunk size (200 episodes per chunk):")
    try:
        dataset_large_chunks = AlchemyDataset(
            json_file_path=FILE_PATH, 
            task_type="classification",
            num_workers=2,
            chunk_size=200  # Larger chunks
        )
        print(f"   ✓ Successfully loaded {len(dataset_large_chunks)} samples with large chunks\n")
    except Exception as e:
        print(f"   ✗ Error with large chunks: {e}\n")
    
    # Test with single worker (no multiprocessing)
    print("3. Testing with single worker (no chunking needed):")
    try:
        dataset_single_worker = AlchemyDataset(
            json_file_path=FILE_PATH, 
            task_type="classification",
            num_workers=1,
            chunk_size=100  # Chunk size ignored when num_workers=1
        )
        print(f"   ✓ Successfully loaded {len(dataset_single_worker)} samples with single worker\n")
    except Exception as e:
        print(f"   ✗ Error with single worker: {e}\n")

def memory_efficient_example():
    """Example of how to use chunking for memory-constrained environments."""
    
    FILE_PATH = "/home/rsaha/projects/dm_alchemy/src/data/generated_data/decompositional_chemistry_samples_val_shop_2_qhop_1.json"
    
    print("=== Memory-Efficient Configuration Example ===\n")
    
    # Configuration for memory-constrained environments
    memory_efficient_config = {
        'json_file_path': FILE_PATH,
        'task_type': 'classification',
        'num_workers': 2,         # Reduce from default 4
        'chunk_size': 25,         # Small chunks (default is 100)
        'val_split': 0.2          # Optional: create train/val splits
    }
    
    print("Configuration for memory-constrained environments:")
    for key, value in memory_efficient_config.items():
        print(f"  {key}: {value}")
    print()
    
    try:
        dataset = AlchemyDataset(**memory_efficient_config)
        print(f"✓ Successfully created dataset with {len(dataset)} samples")
        
        if dataset.has_val_split():
            print(f"  - Training samples: {len(dataset.get_train_set())}")
            print(f"  - Validation samples: {len(dataset.get_val_set())}")
        
    except Exception as e:
        print(f"✗ Error: {e}")

if __name__ == "__main__":
    test_chunking()
    memory_efficient_example()
