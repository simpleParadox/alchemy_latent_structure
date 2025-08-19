#!/usr/bin/env python3
"""
Generate samples from chemistry graph for in-context learning.
This script reads the chemistry_graph.json file and generates trajectories
of stone transformations when applying different potions.
"""

import json
import random
import argparse
import os
import gzip
from typing import Dict, List, Tuple, Set, Any, Union
from tqdm import tqdm
import pickle

def load_chemistry_graph(file_path: str) -> Dict[str, Dict]:
    """Load and parse the chemistry graph JSON file with multiple episodes.
    Supports both compressed (.gz) and uncompressed JSON files.
    """
    if file_path.endswith('.pkl') or file_path.endswith('.pickle'):
       return pickle.load(open(file_path, 'rb')) 
    if file_path.endswith('.gz'):
        with gzip.open(file_path, 'rt', encoding='utf-8') as f:
            return json.load(f)
    else:
        with open(file_path, 'r') as f:
            return json.load(f)


def get_stone_description(node_data: Dict) -> str:
    """Extract the stone description from a node in the graph."""
    return node_data["current_stone_description"]


def get_possible_transitions(node_data: Dict, visited_nodes: Set[str] = None) -> List[Dict]:
    """
    Get possible transitions from a node, excluding those that lead back to visited nodes.
    """
    if visited_nodes is None:
        return node_data["transitions"]
    
    return [
        transition for transition in node_data["transitions"]
        if transition["next_node_str"] not in visited_nodes
    ]


def generate_single_step_sample(graph: Dict, node_id: str, generate_all_edges: bool = False) -> Union[Tuple[str, Dict], List[Tuple[str, Dict]], None]:
    """Generate single-step sample(s) from the given node.
    If generate_all_edges is True, returns a list of (sample_str, sample_info).
    Otherwise, returns a single (sample_str, sample_info) for a random transition.
    Returns None if no transitions are possible for the random choice case and generate_all_edges is False.
    Returns an empty list if no transitions and generate_all_edges is True.
    """
    node_data = graph[node_id]
    stone_state_1 = get_stone_description(node_data)
    
    if not node_data.get("transitions"): # Check if transitions list exists and is not empty
        return [] if generate_all_edges else None

    if generate_all_edges:
        all_step_samples = []
        for transition in node_data["transitions"]:
            potion = transition["potion_color"]
            next_node_id = transition["next_node_str"]
            if next_node_id not in graph: continue # Skip if next node doesn't exist
            stone_state_2 = get_stone_description(graph[next_node_id])
            sample_str = f"{stone_state_1} {potion} -> {stone_state_2}"
            sample_info = {
                "start_node": node_id,
                "end_node": next_node_id,
                "potions": [potion],
                "path_nodes": [node_id, next_node_id]
            }
            all_step_samples.append((sample_str, sample_info))
        return all_step_samples
    else:
        # Choose a random transition
        transition = random.choice(node_data["transitions"])
        potion = transition["potion_color"]
        next_node_id = transition["next_node_str"]
        if next_node_id not in graph: return None # Should ideally not happen with valid graph
        stone_state_2 = get_stone_description(graph[next_node_id])
        sample_str = f"{stone_state_1} {potion} -> {stone_state_2}"
        sample_info = {
            "start_node": node_id,
            "end_node": next_node_id,
            "potions": [potion],
            "path_nodes": [node_id, next_node_id]
        }
        return sample_str, sample_info


def generate_multi_step_sample(
    graph: Dict, 
    start_node_id: str, 
    num_steps: int
) -> List[Tuple[str, Dict]]: # Return type changed to List
    """Generate all multi-step sample trajectories of a specific length from a start node."""
    all_trajectories = []
    if start_node_id not in graph:
        return [] # Start node not in graph
        
    initial_stone_state_text = get_stone_description(graph[start_node_id])

    # Stack for DFS: (current_node_id, list_of_nodes_in_current_path, list_of_potions_in_current_path, current_step_count)
    stack = [(start_node_id, [start_node_id], [], 0)]

    while stack:
        curr_id, path_nodes, potions, steps_taken = stack.pop()

        if steps_taken == num_steps:
            # Path of desired length found
            if curr_id not in graph: # Should be caught by transition check, but as a safeguard
                continue
            
            final_stone_state_text = get_stone_description(graph[curr_id])
            sample_str = f"{initial_stone_state_text} {' '.join(potions)} -> {final_stone_state_text}"
            sample_info = {
                "start_node": start_node_id, # The original start_node_id for this multi-step trajectory
                "end_node": curr_id,         # The end node of this specific path
                "potions": potions,
                "path_nodes": path_nodes     # Full list of nodes in this path
            }
            all_trajectories.append((sample_str, sample_info))
            continue # Found a complete path of num_steps, continue DFS for other paths

        # If steps_taken < num_steps, explore further
        if curr_id not in graph or not graph[curr_id].get("transitions"):
            continue # Node not in graph or no outgoing transitions

        node_data = graph[curr_id]
        
        for transition in node_data["transitions"]:
            next_node_id = transition["next_node_str"]
            
            # Explicitly prevent the path from immediately reversing to the previous node
            # path_nodes[-1] is the current node (curr_id), path_nodes[-2] is the node before it.
            if len(path_nodes) > 1 and next_node_id == path_nodes[-2]:
                continue

            # Prevent cycles within the current path being built (original check)
            # This original check would also catch the immediate reversal if the explicit check above wasn't present.
            if next_node_id in path_nodes:
                continue
            
            # Ensure the next node exists in the graph before proceeding
            if next_node_id not in graph:
                continue

            new_path_nodes = path_nodes + [next_node_id]
            new_potions = potions + [transition["potion_color"]]
            stack.append((next_node_id, new_path_nodes, new_potions, steps_taken + 1))
            
    return all_trajectories


def is_reverse_trajectory(sample1: Dict, sample2: Dict) -> bool:
    """
    Check if sample2 is a reverse of sample1.
    A reverse trajectory has:
    - sample1's end_node as sample2's start_node
    - sample1's start_node as sample2's end_node
    - reversed sequence of potions
    """
    # For single-step trajectories, we can do a simple check
    if len(sample1["potions"]) == 1 and len(sample2["potions"]) == 1:
        return (
            sample1["start_node"] == sample2["end_node"] and
            sample1["end_node"] == sample2["start_node"]
        )
    
    # For multi-step trajectories, it's more complex as we need to check
    # if there's any path from end_node to start_node that's a reverse
    # of the original path
    return (
        sample1["start_node"] == sample2["end_node"] and sample1["end_node"] == sample2["start_node"]
        and sample1["potions"] == sample2["potions"][::-1]  # Check if potions are reversed
        # NOTE: the reverse would actually never happen because the colors always paired together.
    )


def generate_held_out_color_pair_data(graph: Dict, num_held_out_edges, seed=0) -> Dict[str, Any]:
    """
    Generates a support and query set by holding out one pair of colors.
    The support set contains all transitions for other color pairs, plus 'num_held_out_edges'
    example transition from the held-out pair. The query set contains the
    remaining transitions for the held-out pair.
    """
    # Define the color pairs. This should match with the colors used in the deterministic chemistry graphs.
    color_pairs = [('RED', 'GREEN'), ('PINK', 'CYAN'), ('ORANGE', 'YELLOw')]
    
    # 1. Randomly select one color pair to hold out
    random.seed(seed)
    held_out_pair = random.choice(color_pairs)
    support_colors = [color for pair in color_pairs if pair != held_out_pair for color in pair]
    
    print(f"Holding out color pair: {held_out_pair}")

    # 2. Find all transitions for the held-out pair
    held_out_transitions = []
    support_transitions = []

    for node_id, node_data in graph.items():
        if not node_data.get("transitions"):
            continue
        for transition in node_data["transitions"]:
            potion = transition["potion_color"]
            next_node_id = transition["next_node_str"]
            if next_node_id not in graph:
                continue
            
            start_desc = get_stone_description(graph[node_id])
            end_desc = get_stone_description(graph[next_node_id])
            sample_str = f"{start_desc} {potion} -> {end_desc}"
            sample_info = {
                "start_node": node_id, "end_node": next_node_id, "potions": [potion]
            }

            if potion in held_out_pair:
                held_out_transitions.append((sample_str, sample_info)) # This stores the transitions FOR EACH color in the held-out pair
            else:
                support_transitions.append((sample_str, sample_info))

    # Group held-out transitions by the pair of stones they connect
    held_out_stone_pairs = {}
    for s_str, s_info in held_out_transitions:
        # Create a key that preserves the direction of the transition
        pair_key = tuple(sorted((s_info["start_node"], s_info["end_node"])))
        if pair_key not in held_out_stone_pairs:
            held_out_stone_pairs[pair_key] = []
        held_out_stone_pairs[pair_key].append((s_str, s_info))

    # 3. Randomly select one stone pair from the held-out set to add to support
    held_out_pair_keys = list(held_out_stone_pairs.keys())
    random.shuffle(held_out_pair_keys)
    
    if not held_out_pair_keys:
        print("Warning: No transitions found for the held-out color pair.")
        return {"support": [], "query": []}
    
    
    if len(held_out_pair_keys) < num_held_out_edges:
        print(f"Warning: Only {len(held_out_pair_keys)} transitions found for the held-out pair, which is less than the requested {num_held_out_edges}.")
        num_held_out_edges = len(held_out_pair_keys)
        print(f"Adjusting num_held_out_edges to {num_held_out_edges}.")
        print("This will result in unexpected behavior if the held-out pair has fewer transitions than requested.")
    # Get num_hled_out_edges transitions for the held-out pair
    # The following is a generic way to select the num_held_out_edges transitions for the held-out pair.
    support_example_keys = held_out_pair_keys[num_held_out_edges:] # Limit to the number of edges to hold out
    support_example_transitions = []
    for support_example_key in support_example_keys:
        support_example_transitions.extend(held_out_stone_pairs[support_example_key])
    
    support_transitions.extend(support_example_transitions)
    

    # 4. The remaining held-out transitions form the query set
    held_out_pair_keys = held_out_pair_keys[:num_held_out_edges] # Limit to the number of edges to hold out
    query_transitions = []
    for key in held_out_pair_keys:
        query_transitions.extend(held_out_stone_pairs[key])

    # Finalize support and query sets
    support_samples = [s[0] for s in support_transitions]
    query_samples = [q[0] for q in query_transitions]
    
    
    # Check for overlap between support and query samples
    support_set = set(support_samples)
    query_set = set(query_samples)
    overlap = support_set.intersection(query_set)
    if overlap or len(overlap) > 0:
        print(f"Warning: Found {len(overlap)} overlapping samples between support and query sets. This should not happen in a well-formed graph.")
        print(f"Overlapping samples: {overlap}")
    # To maintain consistency with the other function's return type
    return {
        "support": support_samples,
        "query": query_samples,
        "support_num_generated": len(support_samples),
        "query_num_generated": len(query_samples),
        "support_samples_info": [s[1] for s in support_transitions],
        "query_samples_info": [q[1] for q in query_transitions],
        "held_out_pair": held_out_pair
    }
   
    
def generate_support_and_query_examples(
    graph: Dict, 
    num_samples: int, 
    support_hop_length: int = 1, 
    query_hop_length: int = 1,
) -> Dict[str, Any]: # Corrected return type hint
    """Generate a set of random samples from a single episode's chemistry graph."""
    samples = []
    samples_info = []
    generated_samples = set()  # Tracks unique sample strings for support
    
    query_samples = []
    query_samples_info = []
    query_generated_samples = set()  # Tracks unique sample strings for query

    node_ids_available = list(graph.keys())
    random.shuffle(node_ids_available) # Shuffle to process nodes in a varied order: NOTE: We are shuffling because we want to ensure that the samples are not biased towards any particular node.

    processed_nodes_count = 0 # To ensure we don't loop indefinitely if nodes are exhausted

    # Loop until enough samples for both support and query are collected, or nodes/attempts exhausted
    while (len(samples) < num_samples or len(query_samples) < num_samples) and processed_nodes_count < len(node_ids_available):
        if not node_ids_available: # Should be caught by processed_nodes_count check
            break

        start_node_id = node_ids_available[processed_nodes_count] # Pick next node
        processed_nodes_count += 1

        if not graph.get(start_node_id) or not graph[start_node_id].get("transitions"):
            print(f"Skipping node {start_node_id} as it has no transitions or is invalid.")
            continue # Skip if node is invalid or has no transitions

        # --- Generate Support Samples from current start_node_id ---
        if len(samples) < num_samples:
            potential_support_trajectories = []
            if support_hop_length == 1:
                potential_support_trajectories = generate_single_step_sample(graph, start_node_id, generate_all_edges=True) # For this case, the reverse edge really does not matter because all the nodes are processed separately.
            else:
                potential_support_trajectories = generate_multi_step_sample(graph, start_node_id, support_hop_length)
            
            random.shuffle(potential_support_trajectories) # Process trajectories from this node in random order

            for s_str, s_info in potential_support_trajectories:
                if len(samples) >= num_samples:
                    break 
                
                is_duplicate = s_str in generated_samples
                # if not is_duplicate:
                #     for existing_info in samples_info: # Check against existing support samples
                #         # if is_reverse_trajectory(existing_info, s_info):
                #         is_duplicate = True
                #         break
                
                if not is_duplicate:
                    samples.append(s_str)
                    samples_info.append(s_info)
                    generated_samples.add(s_str)

        # --- Generate Query Samples from current start_node_id ---
        if len(query_samples) < num_samples:
            potential_query_trajectories = []
            if query_hop_length == 1:
                potential_query_trajectories = generate_single_step_sample(graph, start_node_id, generate_all_edges=True)
            else:
                potential_query_trajectories = generate_multi_step_sample(graph, start_node_id, query_hop_length)

            random.shuffle(potential_query_trajectories) # Process trajectories from this node in random order

            for q_str, q_info in potential_query_trajectories:
                if len(query_samples) >= num_samples:
                    break

                # Check uniqueness against both support and existing query samples
                if query_hop_length == 1:
                    is_dup_in_query = q_str in query_generated_samples
                else:
                    is_dup_in_support = q_str in generated_samples
                    is_dup_in_query = q_str in query_generated_samples
                # is_dup_in_support = q_str in generated_samples if query_hop_length == 1 else False
                # is_dup_in_query = q_str in query_generated_samples if query_hop_length == 1 else False
                is_duplicate = is_dup_in_support or is_dup_in_query if query_hop_length != 1 else is_dup_in_query 

                # if not is_duplicate:
                    # Check against all collected samples (support + query) for reverse
                    # for existing_info in samples_info + query_samples_info:
                    #     # if is_reverse_trajectory(existing_info, q_info):
                    #     is_duplicate = True
                    #     break
                
                if not is_duplicate:
                    query_samples.append(q_str)
                    query_samples_info.append(q_info)
                    query_generated_samples.add(q_str)
            
    all_samples_data = { # Corrected to match original return structure
        "support": samples,
        "query": query_samples,
        "support_num_generated": len(samples),
        "query_num_generated": len(query_samples),
        "support_samples_info": samples_info,
        "query_samples_info": query_samples_info,
    }
    return all_samples_data # Return the dictionary


def calculate_max_unique_samples(graph: Dict, max_steps: int) -> int:
    """
    Calculate the theoretical maximum number of unique samples possible with this graph.
    This is a simplified calculation and the actual number might be lower due to path constraints.
    """
    num_nodes = len(graph)
    max_transitions_per_node = max(len(node_data["transitions"]) for node_data in graph.values())
    
    # For single-step samples
    single_step_max = sum(len(node_data["transitions"]) for node_data in graph.values())
    
    # For multi-step samples (rough estimation)
    multi_step_estimate = 0
    for steps in range(max_steps, max_steps + 1):
        # This is a simplified upper bound estimation
        multi_step_estimate += num_nodes * (max_transitions_per_node ** steps)
    
    return single_step_max + multi_step_estimate


def main():
    parser = argparse.ArgumentParser(description="Generate samples from chemistry graph")
    # parser.add_argument("--input", default="/home/rsaha/projects/dm_alchemy/src/data/deterministic_chemistries_167424_80_unique_stones_aligned_stone.json.gz",
    #                     help="Path to the chemistry graph JSON file")
    parser.add_argument("--input", default="/home/rsaha/projects/dm_alchemy/src/data/enhanced_chemistries_with_transitions.pkl")
    parser.add_argument("--output", default="chemistry_samples_167424_80_unique_stones.json",
                        help="Output JSON file path for generated samples")
    parser.add_argument("--samples_per_episode", type=int, default=1000,
                        help="Number of samples to generate for each episode")
    parser.add_argument("--support_steps", type=int, default=2,
                        help="Minimum number of transformation steps in each sample")
    parser.add_argument("--query_steps", type=int, default=1,
                        help="Maximum number of transformation steps in each sample")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed for reproducibility")
    parser.add_argument("--create_val_from_train", action="store_true",
                        help="Create a validation set from the training set", default=True)
    parser.add_argument("--process_complete_graph_only", action="store_true",
                        help="Process the complete graphs only", default=False)
    parser.add_argument("--output_dir", default="generated_data_enhanced",
                        help="Directory to save the output files. Default is current directory.") # held_out_exps_generated_data_enhanced
    
    # Add a new argument for your experiment
    parser.add_argument("--held_out_color_exp", action="store_true",
                        help="Generate data for the held-out color pair experiment.", default=False)
    parser.add_argument("--num_held_out_edges", type=int, default=1,
                        help="Number of edges to hold out for the held-out color pair experiment. Default is 1.")

    args = parser.parse_args()
    
    output_file = args.output
    
    # Set random seed for reproducibility
    random.seed(args.seed)
    
    # Load chemistry graph with multiple episodes
    print(f"Loading chemistry graph from {args.input}...")
    chemistry_graphs = load_chemistry_graph(args.input)
    num_episodes = len(chemistry_graphs)
    print(f"Loaded data for {num_episodes} episodes")

    # Split episodes into training and validation sets if requested
    if args.create_val_from_train:
        # Determine the number of episodes for validation (10%)
        num_val_episodes = max(1, int(num_episodes * 0.1))
        num_train_episodes = num_episodes - num_val_episodes
        
        print(f"Creating {num_train_episodes} training episodes and {num_val_episodes} validation episodes")
        
        # Randomly select episodes for validation
        episode_ids = list(chemistry_graphs.keys())
        random.shuffle(episode_ids)
        val_episode_ids = set(episode_ids[:num_val_episodes])
        train_episode_ids = set(episode_ids[num_val_episodes:])
        
        ordered_train_episode_ids = sorted(list(train_episode_ids))
        ordered_val_episode_ids = sorted(list(val_episode_ids))
        
        # Create separate dictionaries for training and validation
        train_graphs = {ep_id: chemistry_graphs[ep_id] for ep_id in train_episode_ids}
        val_graphs = {ep_id: chemistry_graphs[ep_id] for ep_id in val_episode_ids}
        
        # Output file names
        support_hop = args.support_steps
        query_hop = args.query_steps
        train_output_file = os.path.splitext(output_file)[0] + "_train_" + f"shop_{support_hop}_qhop_{query_hop}.json"
        # Validation output file
        val_output_file = os.path.splitext(output_file)[0] + "_val_" + f"shop_{support_hop}_qhop_{query_hop}.json"
        
        # Change prefix based on support and query steps.
        prefix = "compositional_" if args.support_steps <= args.query_steps else "decompositional_"
        
        # Add seed to the filename
        base_train = os.path.splitext(train_output_file)[0]
        ext_train = os.path.splitext(train_output_file)[1]
        if args.held_out_color_exp:
            train_output_file = f"{base_train}_single_held_out_color_{args.num_held_out_edges}_edges_exp_seed_{args.seed}{ext_train}"
        else:
            train_output_file = f"{base_train}_seed_{args.seed}{ext_train}"
        
        
        base_val = os.path.splitext(val_output_file)[0]
        ext_val = os.path.splitext(val_output_file)[1]
        if args.held_out_color_exp:
            val_output_file = f"{base_val}_single_held_out_color_{args.num_held_out_edges}_edges_exp_seed_{args.seed}{ext_val}"
        else:
            val_output_file = f"{base_val}_seed_{args.seed}{ext_val}"
        
        
        train_output_file, val_output_file = [os.path.join(os.path.dirname(f), prefix + os.path.basename(f)) for f in [train_output_file, val_output_file]]
        
        print(f"Creating separate training ({num_train_episodes} episodes) and validation ({num_val_episodes} episodes) sets")
        print(f"Training data will be saved to: {train_output_file}")
        print(f"Validation data will be saved to: {output_file}")
    else:
        # Use all episodes for training
        train_graphs = chemistry_graphs
        val_graphs = {}
        train_output_file = output_file
        
        # Change prefix and add hop information for single dataset case as well
        support_hop = args.support_steps
        query_hop = args.query_steps
        prefix = "compositional_" if args.support_steps <= args.query_steps else "decompositional_"
        train_output_file = os.path.splitext(output_file)[0] + f"_shop_{support_hop}_qhop_{query_hop}.json"
        train_output_file = os.path.join(os.path.dirname(train_output_file), prefix + os.path.basename(train_output_file))
        
    
    # Process training episodes
    if train_graphs:
        # Initialize output structure for training
        train_output_data = {
            "metadata": {
                "num_episodes": len(train_graphs),
                "samples_requested_per_episode": args.samples_per_episode,
                "support-steps": args.support_steps,
                "query-steps": args.query_steps,
                "seed": args.seed,
                "dataset_type": "train"
            },
            "episodes": {}
        }
        
        # Process each training episode
        total_train_samples = 0
        print("\nProcessing training episodes...")
        
        for episode_id, episode_data in tqdm(train_graphs.items(), desc="Processing Training Episodes"):
            
            # Extract the graph for this episode
            
            # Continue only if the graph is complete. Each graph should have an 'is_complete' key.
            if args.process_complete_graph_only and not episode_data.get("is_complete", False):
                print(f"Skipping episode {episode_id} as it is not a complete graph.")
                continue
            print(f"Processing Training Episode {episode_id}...")
            # Extract the graph for this episode
            if not (episode_id == '_metadata'):
                graph = episode_data["graph"]
            
            # Estimate maximum unique samples for this episode
            max_unique_support = calculate_max_unique_samples(graph, args.support_steps)
            
            # Estimate maximum unique samples for this episode
            max_unique_query = calculate_max_unique_samples(graph, args.query_steps)
            
            if args.samples_per_episode > max_unique_support:
                print(f"  WARNING: Requested samples ({args.samples_per_episode}) may exceed maximum possible unique support samples (~{max_unique_support}) for episode {episode_id}.")
            
            if args.samples_per_episode > max_unique_query:
                print(f"  WARNING: Requested samples ({args.samples_per_episode}) may exceed maximum possible unique query samples (~{max_unique_query}) for episode {episode_id}.")
            
            if args.held_out_color_exp:
                support_and_query_samples = generate_held_out_color_pair_data(graph, args.num_held_out_edges, seed=args.seed)
            else:
                support_and_query_samples = generate_support_and_query_examples(
                    graph, 
                    args.samples_per_episode,
                    args.support_steps,
                    args.query_steps,
                )
            
            # Store the episode samples
            train_output_data["episodes"][episode_id] = {
                "support": support_and_query_samples["support"],
                "query": support_and_query_samples["query"],
                "support_num_generated": support_and_query_samples["support_num_generated"],
                "query_num_generated": support_and_query_samples["query_num_generated"],
                "support_samples_info": support_and_query_samples["support_samples_info"],
                "query_samples_info": support_and_query_samples["query_samples_info"],
                "is_complete": episode_data.get("is_complete", False)  # Store graph completeness status
            }
            
            # Update total samples
            support_num_generated = support_and_query_samples["support_num_generated"]
            query_num_generated = support_and_query_samples["query_num_generated"]
            total_train_samples += support_num_generated + query_num_generated
            
            # Check if the total number of samples = 24 if the graph is complete.
            # if episode_data.get("is_complete", False):
            #     if support_num_generated + query_num_generated != 24:
            #         print(f"WARNING: Total samples for complete episode {episode_id} is {support_num_generated + query_num_generated}, expected 24.")
                    
                    
        # Write training output to JSON file
        # Ensure the output directory exists
        # args.output_dir = os.getcwd() + '/' + args.output_dir if not args.output_dir == '.' else args.output_dir
        # os.makedirs(os.path.dirname(args.output_dir), exist_ok=True)
        train_output_file = os.path.join(args.output_dir, train_output_file)
        with open(train_output_file, 'w') as f:
            json.dump(train_output_data, f)
        
        print(f"\nGenerated a total of {total_train_samples} unique training samples across {len(train_graphs)} episodes")
        print(f"Training output saved to {train_output_file}")
    
    # Process validation episodes if requested
    if args.create_val_from_train and val_graphs:
        # Initialize output structure for validation
        val_output_data = {
            "metadata": {
                "num_episodes": len(val_graphs),
                "samples_requested_per_episode": args.samples_per_episode,
                "support-steps": args.support_steps,
                "query-steps": args.query_steps,
                "seed": args.seed,
                "dataset_type": "val"
            },
            "episodes": {}
        }
        
        # Process each validation episode
        total_val_samples = 0
        print("\nProcessing validation episodes...")
        for episode_id, episode_data in val_graphs.items():
            print(f"Processing Validation Episode {episode_id}...")
            
            # Continue only if the graph is complete. Each graph should have an 'is_complete' key.
            if args.process_complete_graph_only and not episode_data.get("is_complete", False):
                print(f"Skipping episode {episode_id} as it is not a complete graph.")
                continue
            # Continue only if the graph is complete. Each graph should have an 'is_complete' key.
            
            # Extract the graph for this episode
            graph = episode_data["graph"]
            
            # Estimate maximum unique samples for this episode
            max_unique_support = calculate_max_unique_samples(graph, args.support_steps)
            print(f"  Estimated maximum unique samples for episode {episode_id}: ~{max_unique_support}")
            
            if args.samples_per_episode > max_unique_support:
                print(f"  WARNING: Requested samples ({args.samples_per_episode}) may exceed maximum possible unique samples.")
            
            
            if args.held_out_color_exp:
                support_and_query_samples = generate_held_out_color_pair_data(graph, args.num_held_out_edges)
            else:
                support_and_query_samples = generate_support_and_query_examples(
                    graph, 
                    args.samples_per_episode,
                    args.support_steps,
                    args.query_steps,
                )
            
            # Store the episode samples
            val_output_data["episodes"][episode_id] = {
                "support": support_and_query_samples["support"],
                "query": support_and_query_samples["query"],
                "support_num_generated": support_and_query_samples["support_num_generated"],
                "query_num_generated": support_and_query_samples["query_num_generated"],
                "support_samples_info": support_and_query_samples["support_samples_info"],
                "query_samples_info": support_and_query_samples["query_samples_info"]
            }
            
            print(f"  Generated {support_and_query_samples['support_num_generated']} support samples and {support_and_query_samples['query_num_generated']} query samples")
            # Update total samples
            support_num_generated = support_and_query_samples["support_num_generated"]
            query_num_generated = support_and_query_samples["query_num_generated"]
            total_val_samples += support_num_generated + query_num_generated
        
        # Write validation output to JSON file
        val_output_file = os.path.join(args.output_dir, val_output_file)
        with open(val_output_file, 'w') as f:
            json.dump(val_output_data, f)
        
        print(f"\nGenerated a total of {total_val_samples} unique validation samples across {len(val_graphs)} episodes")
        print(f"Validation output saved to {val_output_file}")
    
    # Use the appropriate output data for printing examples
    output_data = train_output_data if args.create_val_from_train else train_output_data
    
    # Print a few example samples
    print("\nExample samples:")
    example_count = 0
    for episode_id, episode_data in output_data["episodes"].items():
        support_samples = episode_data['support']
        query_samples = episode_data['query']
        if support_samples:
            for i in range(min(2, len(support_samples))):
                print(f"  Episode {episode_id}, support sample {i+1}: {support_samples[i]}")
                example_count += 1
                if example_count >= 5:
                    break
        if query_samples:
            for i in range(min(2, len(query_samples))):
                print(f"  Episode {episode_id}, query sample {i+1}: {query_samples[i]}")
                example_count += 1
                if example_count >= 5:
                    break
        
        # Stop after printing 5 episodes
        if example_count >= 10:
            break

if __name__ == "__main__":
    main()