import json

def get_reward_mapping(ep_data):
    mapping = {}
    for example in ep_data.get("support", []):
        parts = example.split(" -> ")
        if len(parts) != 2: continue
        # Extract features and reward from in/out states
        # {color: blue, size: large, roundness: pointy, reward: 1}
        # We'll use (color, size, roundness) as key
        for s in [parts[0].split("}")[0]+"} ", parts[1]]:
             import re
             res = re.search(r'color: (\w+), size: (\w+), roundness: (\w+), reward: (-?\d+)', s)
             if res:
                 key = (res.group(1), res.group(2), res.group(3))
                 mapping[key] = int(res.group(4))
    return mapping

orig_path = "src/data/shuffled_held_out_exps_generated_data_enhanced/compositional_chemistry_samples_167424_80_unique_stones_train_shop_1_qhop_1_single_held_out_color_4_edges_exp_seed_0.json"
rand_path = "src/data/held_out_randomized_reward_generated_data/compositional_chemistry_samples_167424_80_unique_stones_train_shop_1_qhop_1_single_held_out_color_4_edges_exp_seed_0.json"

with open(orig_path, 'r') as f:
    orig_data = json.load(f)["episodes"]

with open(rand_path, 'r') as f:
    rand_data = json.load(f)["episodes"]

ep_ids = list(orig_data.keys())[:5]

for ep_id in ep_ids:
    if ep_id not in rand_data: continue
    orig_map = get_reward_mapping(orig_data[ep_id])
    rand_map = get_reward_mapping(rand_data[ep_id])
    
    print(f"\nEpisode {ep_id}:")
    match = True
    for key in orig_map:
        if key in rand_map:
            if orig_map[key] != rand_map[key]:
                print(f"  MISMATCH for {key}: Orig={orig_map[key]}, Rand={rand_map[key]}")
                match = False
    if match:
        print("  All visible rewards match!")
    else:
        print("  RANDOMIZATION DETECTED!")
