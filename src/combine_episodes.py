import json


def combine_episodes(inputs_files: list, output_file: str):
    """_summary_

    Args:
        inputs_files (list): _description_
        output_file (str): _description_
        
    """
    
    '''
    "metadata": {
		"num_episodes": 16742,
		"samples_requested_per_episode": 10000,
		"support-steps": 2,
		"query-steps": 1,
		"seed": 0,
		"dataset_type": "val"
	},
    '''
    combined_metadata = {
        "num_episodes": 0,
        "samples_requested_per_episode": 0,
        "support-steps": 0,
        "query-steps": 0,
        "seed": 0,
        "dataset_type": "combined"
    }
    
    combined_episodes = {}
    for file in inputs_files:
        with open(file, 'r') as f:
            episodes = json.load(f)['episodes']
            
            # Episodes is a dict containing episode_id as key and episode data as another dict as the value.
            for episode_id, episode_data in episodes.items():
                if episode_id in combined_episodes:
                    print(f"Warning: Duplicate episode_id {episode_id} found. Overwriting.")
                combined_episodes[episode_id] = episode_data
    
    combined_metadata['num_episodes'] = len(combined_episodes)
    combined_metadata['samples_requested_per_episode'] = 10000
    combined_metadata['support-steps'] = json.load(open(inputs_files[0]))['metadata']['support-steps']
    combined_metadata['query-steps'] = json.load(open(inputs_files[0]))['metadata']['query-steps']
    combined_metadata['seed'] = None
    
    
    with open(output_file, 'w') as f:
        json.dump({
            "metadata": combined_metadata,
            "episodes": combined_episodes
        }, f, indent=4)
    
    print("Completed combining episodes.")
    # Load the output file and print the number of episodes
    with open(output_file, 'r') as f:
        data = json.load(f)
        print(f"Output file: {output_file}")
        print(f"Number of episodes: {data['metadata']['num_episodes']}")
        print(f"Samples requested per episode: {data['metadata']['samples_requested_per_episode']}")
        print(f"Support steps: {data['metadata']['support-steps']}")
        print(f"Query steps: {data['metadata']['query-steps']}")
        
        # Print the first episode.
        print(f"First episode: {list(data['episodes'].items())[0]}")
    
def main():
   
    # Make the output file wiwth the same directory as the first input file
    hops = [2,3,4,5]
    
    for hop in hops:
        input_files = []
        input_files.append(f"/home/rsaha/projects/dm_alchemy/src/data/complete_graph_generated_data_enhanced_qnodes_in_snodes/decompositional_chemistry_samples_167424_80_unique_stones_train_shop_{hop}_qhop_1_seed_0.json")
        input_files.append(f"/home/rsaha/projects/dm_alchemy/src/data/complete_graph_generated_data_enhanced_qnodes_in_snodes/decompositional_chemistry_samples_167424_80_unique_stones_val_shop_{hop}_qhop_1_seed_0.json")
    
        # Make the output file with the same directory and the same name as the first input file but with _combined.json suffix
        output_file = input_files[0].replace("seed_0.json", "combined.json")
        combine_episodes(input_files, output_file)
    
    
    
if __name__ == "__main__":
    main()