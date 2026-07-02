import json
import re
from typing import Set

json_path = "/home/rsaha/projects/def-afyshe-ab/rsaha/dm_alchemy/src/data/held_out_randomized_reward_generated_data/compositional_chemistry_samples_167424_80_unique_stones_train_shop_1_qhop_1_single_held_out_color_4_edges_exp_seed_0.json"

unique_stone_states: Set[str] = set()

with open(json_path, 'r') as f:
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

        first_brace_end = input_part_str.find("}")
        initial_state_str = input_part_str[:first_brace_end+1]
        
        unique_stone_states.add(initial_state_str)
        unique_stone_states.add(output_state_str)

print(f"Total unique stone states in JSON: {len(unique_stone_states)}")
print("Sample states:")
for s in list(unique_stone_states)[:10]:
    print(f"  {s}")

