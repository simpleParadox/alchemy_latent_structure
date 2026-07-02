import json
import re

train_file = "src/data/baseline_generated_data_from_normalized_reward/compositional_baseline_compositional_chemistry_samples_train_shop_1_qhop_1_single_held_out_color_4_edges_exp_seed_0.json"
val_file = "src/data/baseline_generated_data_from_normalized_reward/compositional_baseline_compositional_chemistry_samples_val_shop_1_qhop_1_single_held_out_color_4_edges_exp_seed_0.json"

def get_stones(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    stones = set()
    for ep in data['episodes'].values():
        for ex in ep.get('support', []) + ep.get('query', []):
            parts = ex.split(' -> ')
            for p in parts:
                m = re.search(r'\{color.*?\}', p)
                if m:
                    stones.add(m.group(0))
    return stones

try:
    train_stones = get_stones(train_file)
    val_stones = get_stones(val_file)
    
    print(f"Unique stones in TRAIN: {len(train_stones)}")
    print(f"Unique stones in VAL: {len(val_stones)}")
    
    overlap = train_stones.intersection(val_stones)
    only_val = val_stones - train_stones
    
    print(f"Stones in BOTH: {len(overlap)}")
    print(f"Stones ONLY in VAL: {len(only_val)}")
    
    if len(only_val) > 0:
        print("\nMissing stones in training set! The model cannot predict these during validation.")
        for s in list(only_val)[:5]:
            print(f"  {s}")
except Exception as e:
    print(f"Error: {e}")
