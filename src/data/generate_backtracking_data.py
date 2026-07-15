#!/usr/bin/env python3
"""
Generate backtracking validation datasets for DM Alchemy.
This script extracts validation episodes from an existing composition val dataset,
finds all k-hop acyclic forward paths on the hypercube transition graph,
applies m reverse transitions (complementary potions) in reverse order,
and creates corresponding support/query JSON files.
"""

import json
import random
import argparse
import os
import pickle
from typing import Dict, List, Tuple, Set, Any
from tqdm import tqdm

def find_complementary_potion(graph: Dict, curr_node_id: str, next_node_id: str) -> str:
    """
    Given a transition from curr_node_id to next_node_id,
    find the potion at next_node_id that goes back to curr_node_id.
    """
    next_node_data = graph.get(next_node_id)
    if not next_node_data:
        return None
    for transition in next_node_data.get("transitions", []):
        if transition["next_node_str"] == curr_node_id:
            return transition["potion_color"]
    return None

def get_stone_description(node_data: Dict) -> str:
    return node_data["current_stone_description"]

def generate_single_step_samples(graph: Dict) -> Tuple[List[str], List[Dict]]:
    """
    Generate all single-step transitions (1-hop support set) for the chemistry graph.
    """
    samples = []
    samples_info = []
    seen = set()
    
    # Sort keys for deterministic output
    for node_id in sorted(graph.keys()):
        node_data = graph[node_id]
        stone_state_1 = get_stone_description(node_data)
        for transition in node_data.get("transitions", []):
            potion = transition["potion_color"]
            next_node_id = transition["next_node_str"]
            if next_node_id not in graph:
                continue
            stone_state_2 = get_stone_description(graph[next_node_id])
            sample_str = f"{stone_state_1} {potion} -> {stone_state_2}"
            if sample_str in seen:
                continue
            seen.add(sample_str)
            samples.append(sample_str)
            samples_info.append({
                "start_node": node_id,
                "end_node": next_node_id,
                "potions": [potion],
                "path_nodes": [node_id, next_node_id]
            })
            
    return samples, samples_info

def generate_forward_paths(graph: Dict, start_node_id: str, num_steps: int) -> List[Dict]:
    """
    Generate all multi-step acyclic forward paths of length num_steps from start_node_id.
    Replicates the generate_multi_step_sample DFS logic.
    """
    all_paths = []
    if start_node_id not in graph:
        return []
        
    # Stack: (current_node_id, path_nodes, potions, steps_taken)
    stack = [(start_node_id, [start_node_id], [], 0)]
    
    while stack:
        curr_id, path_nodes, potions, steps_taken = stack.pop()
        
        if steps_taken == num_steps:
            all_paths.append({
                "start_node": start_node_id,
                "end_node": curr_id,
                "potions": list(potions),
                "path_nodes": list(path_nodes)
            })
            continue
            
        node_data = graph.get(curr_id)
        if not node_data or not node_data.get("transitions"):
            continue
            
        for transition in node_data["transitions"]:
            next_node_id = transition["next_node_str"]
            # Prevent immediate reversal and cycles to enforce acyclic forward path
            if len(path_nodes) > 1 and next_node_id == path_nodes[-2]:
                continue
            if next_node_id in path_nodes:
                continue
            if next_node_id not in graph:
                continue
                
            new_path_nodes = path_nodes + [next_node_id]
            new_potions = potions + [transition["potion_color"]]
            stack.append((next_node_id, new_path_nodes, new_potions, steps_taken + 1))
            
    return all_paths

def generate_backtracking_queries(
    graph: Dict, 
    forward_hops: int, 
    num_backtracks: int,
    max_queries_per_start_node: int
) -> Tuple[List[str], List[Dict]]:
    """
    Generate backtracking queries by taking forward paths and retracing m steps.
    """
    query_samples = []
    query_samples_info = []
    
    # Sort keys for determinism
    for start_node_id in sorted(graph.keys()):
        # Generate all acyclic forward paths of length forward_hops
        forward_paths = generate_forward_paths(graph, start_node_id, forward_hops)
        # Shuffle/randomize to ensure query diversity
        random.shuffle(forward_paths)
        
        count = 0
        for path in forward_paths:
            if count >= max_queries_per_start_node:
                break
                
            path_nodes = path["path_nodes"]
            forward_potions = path["potions"]
            
            # Find backtracking potions to reverse the last `num_backtracks` steps
            backward_potions = []
            valid = True
            for i in range(1, num_backtracks + 1):
                curr = path_nodes[forward_hops - i + 1]
                prev = path_nodes[forward_hops - i]
                comp_potion = find_complementary_potion(graph, prev, curr) # The function returns the potion that goes from 'curr' to 'prev' node.
                if not comp_potion:
                    valid = False
                    break
                backward_potions.append(comp_potion)
                
            if not valid:
                continue
                
            # Construct the final backtracking potion sequence and target state
            full_potions = forward_potions + backward_potions
            start_stone_desc = get_stone_description(graph[start_node_id])
            
            # Target is the stone at index forward_hops - num_backtracks
            target_node_id = path_nodes[forward_hops - num_backtracks]
            target_stone_desc = get_stone_description(graph[target_node_id])
            
            query_str = f"{start_stone_desc} {' '.join(full_potions)} -> {target_stone_desc}"
            
            query_samples.append(query_str)
            query_samples_info.append({
                "start_node": start_node_id,
                "end_node": target_node_id,
                "potions": full_potions,
                "path_nodes": path_nodes + list(reversed(path_nodes[forward_hops - num_backtracks + 1 : forward_hops]))
            })
            count += 1
            
    return query_samples, query_samples_info

def main():
    parser = argparse.ArgumentParser(description="Generate backtracking validation data.")
    parser.add_argument("--input", type=str,
                        default="src/data/chemistry_pickles/original_reward_mapping_potion_remaps/potion_remapping_0_original_reward_remapping_with_transitions.pkl",
                        help="Path to the chemistry graph pickle file.")
    parser.add_argument("--val_episode_source", type=str, required=True,
                        help="Path to an existing validation JSON file to match episode/chemistry IDs.")
    parser.add_argument("--output_dir", type=str, default="src/data/backtracking_generated_data",
                        help="Output directory for generated JSON files.")
    parser.add_argument("--forward_hops", type=int, required=True,
                        help="Number of forward steps in query (k).")
    parser.add_argument("--num_backtracks", type=int, required=True,
                        help="Number of backward backtrack steps (m).")
    parser.add_argument("--max_queries_per_start_node", type=int, default=10000,
                        help="Maximum queries to generate per start node.")
    parser.add_argument("--seed", type=int, default=0,
                        help="Seed for randomization.")
    
    args = parser.parse_args()
    random.seed(args.seed)
    
    # 1. Load chemistry graphs
    print(f"Loading chemistry graphs from {args.input}...")
    with open(args.input, "rb") as f:
        chemistry_graphs = pickle.load(f)
        
    # 2. Load validation episode IDs to ensure no train leakage
    print(f"Loading validation episodes from {args.val_episode_source}...")
    with open(args.val_episode_source, "r") as f:
        val_source_data = json.load(f)
    
    val_episode_ids = set(val_source_data.get("episodes", {}).keys())
    print(f"Found {len(val_episode_ids)} validation episodes to generate data for.")
    
    # 3. Generate data
    output_data = {
        "metadata": {
            "num_episodes": len(val_episode_ids),
            "samples_requested_per_episode": 10000,
            "support-steps": 1,
            "query-steps": args.forward_hops + args.num_backtracks,
            "backtrack-steps": args.num_backtracks,
            "seed": args.seed,
            "dataset_type": "val",
            "val_episode_source": os.path.basename(args.val_episode_source)
        },
        "episodes": {}
    }
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    for episode_id in tqdm(sorted(val_episode_ids), desc="Generating episodes"):
        if episode_id not in chemistry_graphs:
            # Re-mapping or edge cases
            continue
            
        episode_data = chemistry_graphs[episode_id]
        graph = episode_data["graph"]
        
        # Support is standard 1-hop
        support_samples, support_info = generate_single_step_samples(graph)
        
        # Query is k-forward + m-backward
        query_samples, query_info = generate_backtracking_queries(
            graph, 
            args.forward_hops, 
            args.num_backtracks,
            args.max_queries_per_start_node
        )
        
        output_data["episodes"][episode_id] = {
            "support": support_samples,
            "query": query_samples,
            "support_num_generated": len(support_samples),
            "query_num_generated": len(query_samples),
            "support_samples_info": support_info,
            "query_samples_info": query_info,
            "is_complete": episode_data.get("is_complete", graph.get("_metadata", {}).get("is_complete", True))
        }
        
    # 4. Save output
    output_filename = f"backtracking_val_shop_1_qhop_{args.forward_hops}_backtrack_{args.num_backtracks}_seed_{args.seed}.json"
    output_path = os.path.join(args.output_dir, output_filename)
    
    print(f"Saving generated dataset to {output_path}...")
    with open(output_path, "w") as f:
        json.dump(output_data, f)
        
    print("Generation completed successfully!")

if __name__ == "__main__":
    main()
