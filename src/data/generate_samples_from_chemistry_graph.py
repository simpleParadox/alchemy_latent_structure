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
from typing import Dict, List, Tuple, Set, Any


def load_chemistry_graph(file_path: str) -> Dict[str, Dict]:
    """Load and parse the chemistry graph JSON file with multiple episodes."""
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


def generate_single_step_sample(graph: Dict, node_id: str) -> Tuple[str, Dict]:
    """Generate a single-step sample from the given node."""
    node_data = graph[node_id]
    stone_state_1 = get_stone_description(node_data)
    
    # Choose a random transition
    transition = random.choice(node_data["transitions"])
    potion = transition["potion_color"]
    next_node_id = transition["next_node_str"]
    
    stone_state_2 = get_stone_description(graph[next_node_id])
    
    sample_str = f"{stone_state_1} {potion} -> {stone_state_2}"
    
    # Return sample and info to create a reverse if needed
    sample_info = {
        "start_node": node_id,
        "end_node": next_node_id,
        "potions": [potion]
    }
    
    return sample_str, sample_info


def generate_multi_step_sample(
    graph: Dict, 
    start_node_id: str, 
    num_steps: int
) -> Tuple[str, Dict]:
    """Generate a multi-step sample trajectory."""
    current_node_id = start_node_id
    visited_nodes = {current_node_id}
    potions_used = []
    path_nodes = [current_node_id]
    
    # Record initial stone state
    initial_stone_state = get_stone_description(graph[current_node_id])
    
    # Generate the path
    for _ in range(num_steps):
        node_data = graph[current_node_id]
        
        # Get transitions that don't lead back to visited nodes
        valid_transitions = get_possible_transitions(node_data, visited_nodes)
        
        # If no valid transitions, break out of loop
        if not valid_transitions:
            break
            
        # Choose a random transition
        transition = random.choice(valid_transitions)
        potions_used.append(transition["potion_color"])
        
        # Move to the next node
        current_node_id = transition["next_node_str"]
        path_nodes.append(current_node_id)
        visited_nodes.add(current_node_id)
    
    # Get final stone state
    final_stone_state = get_stone_description(graph[current_node_id])
    
    # Format the sample
    sample_str = f"{initial_stone_state} {' '.join(potions_used)} -> {final_stone_state}"
    
    # Return sample and info to create a reverse if needed
    sample_info = {
        "start_node": start_node_id,
        "end_node": current_node_id,
        "potions": potions_used,
        "path_nodes": path_nodes # Ground truth intermediate nodes.
    }
    
    return sample_str, sample_info


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
    )


def generate_samples_for_episode(
    graph: Dict, 
    num_samples: int, 
    min_steps: int = 1, 
    max_steps: int = 3,
    allow_reverse_trajectories: bool = False
) -> Tuple[List[str], int, List[Dict]]:
    """Generate a set of random samples from a single episode's chemistry graph."""
    samples = []
    samples_info = []  # Store info about each sample for reverse checking
    node_ids = list(graph.keys())
    generated_samples = set()  # Track already generated samples
    max_attempts = num_samples * 10  # Limit attempts to avoid infinite loop
    attempts = 0
    
    while len(samples) < num_samples and attempts < max_attempts:
        attempts += 1
        
        # Choose a random starting node
        start_node_id = random.choice(node_ids)
        
        # Decide on the number of steps for this sample
        num_steps = random.randint(min_steps, max_steps)
        
        # Generate a sample
        if num_steps == 1:
            sample, sample_info = generate_single_step_sample(graph, start_node_id)
        else:
            sample, sample_info = generate_multi_step_sample(graph, start_node_id, num_steps)
        
        # Check if this sample is unique
        is_duplicate = sample in generated_samples
        
        # If not allowing reverse trajectories, also check if sample is a reverse of an existing one
        if not is_duplicate and not allow_reverse_trajectories:
            for existing_info in samples_info:
                if is_reverse_trajectory(existing_info, sample_info):
                    is_duplicate = True
                    break
        
        if not is_duplicate:
            samples.append(sample)
            samples_info.append(sample_info)
            generated_samples.add(sample)
    
    # Return the samples, how many were actually generated, and their info
    return samples, len(samples), samples_info


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
    for steps in range(2, max_steps + 1):
        # This is a simplified upper bound estimation
        multi_step_estimate += num_nodes * (max_transitions_per_node ** steps)
    
    return single_step_max + multi_step_estimate


def main():
    parser = argparse.ArgumentParser(description="Generate samples from chemistry graph")
    parser.add_argument("--input", default="chemistry_graph.json",
                        help="Path to the chemistry graph JSON file")
    parser.add_argument("--output", default="chemistry_samples.json",
                        help="Output JSON file path for generated samples")
    parser.add_argument("--samples-per-episode", type=int, default=20,
                        help="Number of samples to generate for each episode")
    parser.add_argument("--min-steps", type=int, default=3,
                        help="Minimum number of transformation steps in each sample")
    parser.add_argument("--max-steps", type=int, default=3,
                        help="Maximum number of transformation steps in each sample")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--allow-reverse-trajectories", action="store_true",
                        help="Allow trajectories that are reverses of each other (e.g., A→B and B→A)")
    
    args = parser.parse_args()
    
    # Modify output filename if allowing reverse trajectories
    if args.allow_reverse_trajectories:
        base_name, ext = os.path.splitext(args.output)
        output_file = f"{base_name}_reverse_trajectories_included{ext}"
    else:
        output_file = args.output
    
    # Set random seed for reproducibility
    random.seed(args.seed)
    
    # Load chemistry graph with multiple episodes
    print(f"Loading chemistry graph from {args.input}...")
    chemistry_graphs = load_chemistry_graph(args.input)
    num_episodes = len(chemistry_graphs)
    print(f"Loaded data for {num_episodes} episodes")
    
    # Initialize output structure
    output_data = {
        "metadata": {
            "num_episodes": num_episodes,
            "samples_requested_per_episode": args.samples_per_episode,
            "min_steps": args.min_steps,
            "max_steps": args.max_steps,
            "seed": args.seed,
            "allow_reverse_trajectories": args.allow_reverse_trajectories
        },
        "episodes": {}
    }
    
    # Process each episode
    total_samples = 0
    for episode_id, episode_data in chemistry_graphs.items():
        print(f"\nProcessing Episode {episode_id}...")
        
        # Extract the graph for this episode
        graph = episode_data["graph"]
        
        # Estimate maximum unique samples for this episode
        max_unique = calculate_max_unique_samples(graph, args.max_steps)
        print(f"  Estimated maximum unique samples for episode {episode_id}: ~{max_unique}")
        
        if args.samples_per_episode > max_unique:
            print(f"  WARNING: Requested samples ({args.samples_per_episode}) may exceed maximum possible unique samples.")
        
        # Generate samples for this episode
        print(f"  Generating up to {args.samples_per_episode} unique samples (steps: {args.min_steps}-{args.max_steps})...")
        print(f"  {'Allowing' if args.allow_reverse_trajectories else 'Excluding'} reverse trajectories")
        
        samples, num_generated, _ = generate_samples_for_episode(
            graph, 
            args.samples_per_episode,
            args.min_steps,
            args.max_steps,
            args.allow_reverse_trajectories
        )
        
        # Store the episode samples
        output_data["episodes"][episode_id] = {
            "samples": samples,
            "num_generated": num_generated,
            "estimated_max_samples": max_unique
        }
        
        print(f"  Generated {num_generated} unique samples for episode {episode_id}")
        total_samples += num_generated
    
    # Write output to JSON file
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\nGenerated a total of {total_samples} unique samples across {num_episodes} episodes")
    print(f"Output saved to {output_file}")
    
    # Print a few example samples
    print("\nExample samples:")
    example_count = 0
    for episode_id, episode_data in output_data["episodes"].items():
        samples = episode_data["samples"]
        if samples:
            for i in range(min(2, len(samples))):
                print(f"  Episode {episode_id}, Sample {i+1}: {samples[i]}")
                example_count += 1
                if example_count >= 5:
                    break
            if example_count >= 5:
                break


if __name__ == "__main__":
    main()