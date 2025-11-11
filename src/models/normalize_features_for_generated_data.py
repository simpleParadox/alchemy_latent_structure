import json
import os
import re
from tqdm import tqdm


def normalize_reward_in_state(state_str: str, normalized_value: int = 1) -> str:
    """
    Normalize the reward value in a stone state string to a fixed normalized_value.
    
    Args:
        state_str (str): The stone state string, e.g., "stone(red,3,2)".
        normalized_value (int): The value to which the reward should be normalized.
        
    Returns:
        str: The stone state string with the normalized reward value.
    """
    pattern = r'stone\(([^,]+),([^,]+),([^)]+)\)'
    match = re.match(pattern, state_str)
    if match:
        color, shape, _ = match.groups()
        return f'stone({color},{shape},{normalized_value})'
    else:
        raise ValueError(f"Invalid stone state format: {state_str}")
    
    
def normalize_data_file(input_file: str, normalized_value: int = -3) -> dict:
    
    
    normalized_reward_value ='reward: ' + str(normalized_value)
    with open(input_file, 'r') as f:
        raw_data = json.load(f)
 
    normalized_data = {}
        
    normalized_data["metadata"] = raw_data["metadata"]
    # normalized_data['support_num_generated'] = raw_data.get('support_num_generated', 0)
    # normalized_data['query_num_generated'] = raw_data.get('query_num_generated', 0)
    # normalized_data['support_samples_info'] = raw_data.get('support_samples_info', {})
    # normalized_data['query_samples_info'] = raw_data.get('query_samples_info', {})
    normalized_data['episodes'] = {}
        
    for episode_id, episode_content in tqdm(raw_data["episodes"].items(), desc="Processing episodes"):
        if not episode_content: continue # Skip empty episodes
        
        # Process support examples
        support_examples = episode_content.get("support", [])
        normalized_support_examples = []
        for example_str in support_examples:
            # Replace all reward values with normalized value
            normalized_example = re.sub(r'reward:\s*(-3|-1|1|3)', normalized_reward_value, example_str)
            normalized_support_examples.append(normalized_example)
        
        # Process query examples
        query_examples = episode_content.get("query", [])
        normalized_query_examples = []
        for example_str in query_examples:
            # Replace all reward values with normalized value
            normalized_example = re.sub(r'reward:\s*(-3|-1|1|3)', normalized_reward_value, example_str)
            normalized_query_examples.append(normalized_example)
        
        # Store normalized examples
        normalized_data['episodes'][episode_id] = {
            'support': normalized_support_examples,
            'query': normalized_query_examples
        }
        # Copy other episode-level metadata if any.
        normalized_data['episodes'][episode_id].update({k: v for k, v in episode_content.items() if k not in ['support', 'query']})
    
    return normalized_data


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Normalize reward values in stone state strings.")
    parser.add_argument("--train_file", type=str, help="Path to the training data file.",
                        default='/home/rsaha/projects/dm_alchemy/src/data/shuffled_held_out_exps_generated_data_enhanced/compositional_chemistry_samples_167424_80_unique_stones_train_shop_1_qhop_1_single_held_out_color_4_edges_exp_seed_4.json')
    parser.add_argument("--val_file", type=str, help="Path to the validation data file.",
                        default='/home/rsaha/projects/dm_alchemy/src/data/shuffled_held_out_exps_generated_data_enhanced/compositional_chemistry_samples_167424_80_unique_stones_val_shop_1_qhop_1_single_held_out_color_4_edges_exp_seed_4.json')
    parser.add_argument("--output_dir", type=str, help="Directory to save the normalized data files.",
                        default='/home/rsaha/projects/dm_alchemy/src/data/same_reward_shuffled_held_out_exps_generated_data_enhanced/')
    
    args = parser.parse_args()
        
    normalized_train_data = normalize_data_file(args.train_file, normalized_value=-3)
    normalized_val_data = normalize_data_file(args.val_file, normalized_value=-3)
        
       
    train_file_name = os.path.basename(args.train_file)
    val_file_name = os.path.basename(args.val_file)
    normalized_train_file = os.path.join(args.output_dir, f"normalized_{train_file_name}")
    normalized_val_file = os.path.join(args.output_dir, f"normalized_{val_file_name}")
    
    # Create the directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    with open(normalized_train_file, 'w') as f:
        json.dump(normalized_train_data, f)
    with open(normalized_val_file, 'w') as f:
        json.dump(normalized_val_data, f)
    print(f"Normalized training data saved to {normalized_train_file}")
    print(f"Normalized validation data saved to {normalized_val_file}")


if __name__ == "__main__":
    main()    