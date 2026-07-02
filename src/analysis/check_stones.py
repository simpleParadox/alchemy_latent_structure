import json
import pickle

json_path = "/home/rsaha/projects/def-afyshe-ab/rsaha/dm_alchemy/src/data/held_out_randomized_reward_generated_data/compositional_chemistry_samples_167424_80_unique_stones_train_shop_1_qhop_1_single_held_out_color_4_edges_exp_seed_0.json"
orig_vocab_path = "/home/rsaha/projects/def-afyshe-ab/rsaha/dm_alchemy/src/data/shuffled_held_out_exps_preprocessed_separate_enhanced/compositional_chemistry_samples_167424_80_unique_stones_val_shop_1_qhop_1_single_held_out_color_4_edges_exp_seed_0_classification_filter_True_input_features_output_stone_states_vocab.pkl"

with open(orig_vocab_path, "rb") as f:
    orig_vocab = pickle.load(f)

orig_stones = set(orig_vocab["stone_state_to_id"].keys())

with open(json_path, 'r') as f:
    raw_data = json.load(f)

json_stones = set()
for episode_id, episode_content in raw_data["episodes"].items():
    if not episode_content: continue
    all_example_strings = episode_content.get("support", []) + episode_content.get("query", [])
    
    for example_str in all_example_strings:
        parts = example_str.split(" -> ")
        if len(parts) != 2:
            continue
        first_brace_end = parts[0].find("}")
        json_stones.add(parts[0][:first_brace_end+1])
        json_stones.add(parts[1])

only_in_orig = orig_stones - json_stones
only_in_json = json_stones - orig_stones

print(f"Original vocab stones: {len(orig_stones)}")
print(f"JSON generated stones: {len(json_stones)}")
print(f"Stones ONLY in Original : {len(only_in_orig)}")
print(f"Stones ONLY in JSON     : {len(only_in_json)}")

if only_in_json:
    print("Sample stones only in JSON:")
    for s in list(only_in_json)[:5]: print("  ", s)
