import json
import random
import torch
from torch.utils.data import DataLoader, Dataset
import sys; sys.path.append('src/models')
from data_loaders import AlchemyDataset

# 1. Compare the underlying JSON files
path_orig = 'src/data/shuffled_held_out_exps_generated_data_enhanced/compositional_chemistry_samples_167424_80_unique_stones_train_shop_1_qhop_1_single_held_out_color_4_edges_exp_seed_0.json'
path_pair0 = 'src/data/chemistry_pickles/original_reward_potion_remap_generated_data/compositional_chemistry_samples_167424_80_unique_stones_train_shop_1_qhop_1_single_held_out_color_4_edges_exp_seed_0_pairing_index_0.json'

with open(path_orig, 'r') as f:
    orig_json = json.load(f)
with open(path_pair0, 'r') as f:
    pair0_json = json.load(f)

orig_keys = list(orig_json['episodes'].keys())
pair0_keys = list(pair0_json['episodes'].keys())

print(f"Number of episodes in Orig JSON: {len(orig_keys)}")
print(f"Number of episodes in Pair0 JSON: {len(pair0_keys)}")
print(f"Are the sets of keys identical? {set(orig_keys) == set(pair0_keys)}")
print(f"Are the lists of keys perfectly ordered the same? {orig_keys == pair0_keys}")
print(f"First 3 keys in Orig JSON: {orig_keys[:3]}")
print(f"First 3 keys in Pair0 JSON: {pair0_keys[:3]}")
print("-" * 50)

# 2. Show how this affects PyTorch DataLoader
class DummyDataset(Dataset):
    def __init__(self, data_list):
        self.data = data_list
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]

# Imagine we have 4 items, but they are ordered differently based on JSON insertion
items_orig = ['Item_A', 'Item_B', 'Item_C', 'Item_D']
items_pair0 = ['Item_D', 'Item_A', 'Item_B', 'Item_C']

# If we pass them to DataLoader with shuffle=True and a FIXED seed (like in train_continual.py)
seed = 42

gen1 = torch.Generator().manual_seed(seed)
loader_orig = DataLoader(DummyDataset(items_orig), batch_size=2, shuffle=True, generator=gen1)
print(f"Batches seen by model using Original Ordering (Seed {seed}):")
for batch in loader_orig:
    print(batch)

gen2 = torch.Generator().manual_seed(seed)
loader_pair0 = DataLoader(DummyDataset(items_pair0), batch_size=2, shuffle=True, generator=gen2)
print(f"\nBatches seen by model using Pair0 Ordering (Seed {seed}):")
for batch in loader_pair0:
    print(batch)

