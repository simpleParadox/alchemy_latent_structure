#!/usr/bin/env python3
"""
Deterministically generates all unique chemistry graphs by combining
all possible PotionMaps, StoneMaps, Graphs (from constraints), and Rotations.
"""
import itertools
import json
import hashlib
import argparse
import gzip
from tqdm import tqdm
import numpy as np
from typing import Dict, Any, Tuple, List, Set

# Imports from dm_alchemy
from dm_alchemy.types import stones_and_potions, utils as type_utils, graphs
from dm_alchemy.types.stones_and_potions import PotionMap, StoneMap, LatentStone, AlignedStone, PerceivedStone, unalign, Potion

# --- Constants ---
POTION_INDEX_TO_COLOR_VALUE = [-1.0, -2/3, -1/3, 0.0, 1/3, 2/3]
SEEN_CHEMISTRY_HASHES: Set[str] = set()

# --- Helper Functions (copied and adapted from generate_unique_chemisty_graphs.py) ---

def get_stone_feature_name(feature_index: int, value: float) -> str:
    """Maps numerical stone feature values (-1, 0, 1) to descriptive names.
    Indices: 0: color, 1: size, 2: roundness.
    """
    if feature_index == 0: # Color
        if np.isclose(value, -1.0): return "blue"
        if np.isclose(value, 0.0): return "purple"
        if np.isclose(value, 1.0): return "red"
    elif feature_index == 1: # Size
        if np.isclose(value, -1.0): return "small"
        if np.isclose(value, 0.0): return "medium"
        if np.isclose(value, 1.0): return "large"
    elif feature_index == 2: # Roundness
        if np.isclose(value, -1.0): return "pointy"
        if np.isclose(value, 0.0): return "medium_round"
        if np.isclose(value, 1.0): return "round"
    return "unknown"

def get_stone_feature_name_extended(feature_index: int, value: float) -> str:
    """Maps aligned coordinate values to descriptive names, including reward dimension.
    Extended version that handles aligned coordinates directly without rotation effects.
    Indices: 0: color, 1: size, 2: roundness, 3: reward.
    """
    if feature_index == 0: # Color
        if np.isclose(value, -1.0): return "blue"
        if np.isclose(value, 0.0): return "purple"
        if np.isclose(value, 1.0): return "red"
    elif feature_index == 1: # Size
        if np.isclose(value, -1.0): return "small"
        if np.isclose(value, 0.0): return "medium"
        if np.isclose(value, 1.0): return "large"
    elif feature_index == 2: # Roundness
        if np.isclose(value, -1.0): return "pointy"
        if np.isclose(value, 0.0): return "medium_round"
        if np.isclose(value, 1.0): return "round"
    elif feature_index == 3: # Reward
        if np.isclose(value, -1.0): return "no_reward"
        if np.isclose(value, 0.0): return "medium_reward"  
        if np.isclose(value, 1.0): return "high_reward"
        # Handle the additional reward value (3.0) that can occur in some configurations
        if np.isclose(value, 3.0): return "max_reward"
    return "unknown"

def get_potion_color_name(color_value: float) -> str:
    """Maps the numerical potion color value from the observation to its name."""
    if np.isclose(color_value, -1.0): return "RED"
    if np.isclose(color_value, -2/3): return "GREEN"
    if np.isclose(color_value, -1/3): return "ORANGE"
    if np.isclose(color_value, 0.0): return "YELLOW"
    if np.isclose(color_value, 1/3): return "PINK"
    if np.isclose(color_value, 2/3): return "CYAN"
    return "USED/UNKNOWN"

def potion_index_to_color_name(potion_idx: int) -> str:
    """Maps a potion index (0-5) to its color name."""
    if 0 <= potion_idx < len(POTION_INDEX_TO_COLOR_VALUE):
        color_value = POTION_INDEX_TO_COLOR_VALUE[potion_idx]
        return get_potion_color_name(color_value)
    return "INVALID_INDEX"

def is_graph_complete(graph) -> bool:
    """
    Checks if a chemistry graph is complete (has 8 nodes with 3 outgoing edges each).
    """
    if not graph or not hasattr(graph, 'node_list') or not hasattr(graph, 'edge_list'):
        return False
    
    nodes = graph.node_list.nodes
    edges = graph.edge_list.edges
    
    if len(nodes) != 8:
        return False
    
    for node in nodes:
        if node not in edges or len(edges[node]) != 3:
            return False
    return True

def generate_canonical_graph_string(chemistry) -> str:
    """
    Generates a canonical string representation of a chemistry graph.
    This is used to create a unique hash for chemistry comparison.
    """
    if not chemistry or not hasattr(chemistry, 'graph'):
        return ""
    
    parts = []
    nodes = sorted(chemistry.graph.node_list.nodes, key=str)
    
    for node in nodes:
        node_str = str(node)
        parts.append(f"NODE:{node_str}")
        
        try:
            latent_stone = LatentStone(node.coords)
            # NOTE: The aligned map coords for the AlignedStone is just the flipped coords of the LatentStone.
            aligned_stone_from_map = chemistry.stone_map.apply_inverse(latent_stone)
            aligned_stone_for_unalign = AlignedStone(
                aligned_stone_from_map.reward,
                aligned_stone_from_map.aligned_coords
            )
            
            # In the perceived stone, the coords actually change from having the values between -1 and 1 to being between -1 and 1, but including 0 as well. For example, (-1, 1, 1) may change to (0, 1, -1).
            # The aligned stone coords are always between -1 and 1, but 0 is also a valid value (unlike the latent stone where only -1 and 1 are valid values).
            perceived_stone = unalign(aligned_stone_for_unalign, chemistry.rotation)
            
            color = get_stone_feature_name(0, perceived_stone.perceived_coords[0])
            size = get_stone_feature_name(1, perceived_stone.perceived_coords[1])
            roundness = get_stone_feature_name(2, perceived_stone.perceived_coords[2])
            stone_desc_str = f"DESC:{{color:{color},size:{size},roundness:{roundness},reward:{aligned_stone_for_unalign.reward}}}"
            parts.append(stone_desc_str)

        except Exception as e:
            print(f"Warning: Could not generate full stone data for node {node_str}: {e}")
            parts.append(f"STONE:error")
            parts.append(f"DESC:error")
        
        if node in chemistry.graph.edge_list.edges:
            graph_edges = chemistry.graph.edge_list.edges[node]
            sorted_edges = sorted(graph_edges.items(), key=lambda x: str(x[0]))

            for next_node, edge_info in sorted_edges:
                if len(edge_info) >= 2 and isinstance(edge_info[1], Potion):
                    potion = edge_info[1]
                    potion_color_name = potion_index_to_color_name(potion.idx)
                    next_stone_desc_str = "NEXT_STONE_DESC:error"
                    try:
                        next_latent_stone = LatentStone(next_node.coords)
                        next_aligned_stone_from_map = chemistry.stone_map.apply_inverse(next_latent_stone)
                        next_aligned_stone_for_unalign = AlignedStone(
                            next_aligned_stone_from_map.reward,
                            next_aligned_stone_from_map.aligned_coords
                        )
                        next_perceived_stone = unalign(next_aligned_stone_for_unalign, chemistry.rotation)
                        next_color = get_stone_feature_name(0, next_perceived_stone.perceived_coords[0])
                        next_size = get_stone_feature_name(1, next_perceived_stone.perceived_coords[1])
                        next_roundness = get_stone_feature_name(2, next_perceived_stone.perceived_coords[2])
                        next_stone_desc_str = f"NEXT_STONE_DESC:{{color:{next_color},size:{next_size},roundness:{next_roundness},reward:{next_aligned_stone_for_unalign.reward}}}"
                    except Exception as e:
                        print(f"Warning: Could not generate description for next_node {str(next_node)}: {e}")
                    parts.append(f"EDGE:{str(next_node)}:{potion.idx}:{potion_color_name}:{potion.dimension}:{potion.direction}:{next_stone_desc_str}")

    parts.append(f"POTION_MAP:{':'.join(map(str, chemistry.potion_map.dim_map))}")
    parts.append(f"POTION_DIR:{':'.join(map(str, chemistry.potion_map.dir_map))}")
    parts.append(f"STONE_MAP:{':'.join(map(str, chemistry.stone_map.latent_pos_dir))}")
    rotation_flat = chemistry.rotation.flatten()
    parts.append(f"ROTATION:{':'.join(map(str, rotation_flat))}")
    
    return "||".join(parts)

def generate_canonical_graph_string_extended(chemistry) -> str:
    """
    Generates a canonical string representation using perceived visual coordinates
    and aligned reward. This allows for 3x3x3 visual features and 4 reward values,
    giving a capacity for 108 unique stone state combinations.
    """
    if not chemistry or not hasattr(chemistry, 'graph'):
        return ""
    
    parts = []
    nodes = sorted(chemistry.graph.node_list.nodes, key=str)
    
    for node in nodes:
        node_str = str(node)
        parts.append(f"NODE:{node_str}")
        
        try:
            latent_stone = LatentStone(node.coords)
            # Get aligned stone directly from stone map
            aligned_stone_from_map = chemistry.stone_map.apply_inverse(latent_stone)
            
            # Create AlignedStone suitable for unaligning
            aligned_stone_for_unalign = AlignedStone(
                aligned_stone_from_map.reward,
                aligned_stone_from_map.aligned_coords
            )
            # Get perceived stone by applying rotation
            perceived_stone = unalign(aligned_stone_for_unalign, chemistry.rotation)
            
            # Use perceived coordinates for visual features with get_stone_feature_name
            color = get_stone_feature_name(0, perceived_stone.perceived_coords[0])
            size = get_stone_feature_name(1, perceived_stone.perceived_coords[1])
            roundness = get_stone_feature_name(2, perceived_stone.perceived_coords[2])
            # Use aligned reward directly
            reward = aligned_stone_from_map.reward
            stone_desc_str = f"DESC:{{color:{color},size:{size},roundness:{roundness},reward:{reward}}}"
            parts.append(stone_desc_str)

        except Exception as e:
            print(f"Warning: Could not generate full stone data for node {node_str}: {e}")
            parts.append(f"STONE:error")
            parts.append(f"DESC:error")
        
        if node in chemistry.graph.edge_list.edges:
            graph_edges = chemistry.graph.edge_list.edges[node]
            sorted_edges = sorted(graph_edges.items(), key=lambda x: str(x[0]))

            for next_node, edge_info in sorted_edges:
                if len(edge_info) >= 2 and isinstance(edge_info[1], Potion):
                    potion = edge_info[1]
                    potion_color_name = potion_index_to_color_name(potion.idx)
                    next_stone_desc_str = "NEXT_STONE_DESC:error"
                    try:
                        next_latent_stone = LatentStone(next_node.coords)
                        next_aligned_stone_from_map = chemistry.stone_map.apply_inverse(next_latent_stone)
                        
                        # Create AlignedStone suitable for unaligning for the next stone
                        next_aligned_stone_for_unalign = AlignedStone(
                            next_aligned_stone_from_map.reward,
                            next_aligned_stone_from_map.aligned_coords
                        )
                        # Get perceived stone for the next stone
                        next_perceived_stone = unalign(next_aligned_stone_for_unalign, chemistry.rotation)

                        # Use perceived coordinates for next stone's visual features
                        next_color = get_stone_feature_name(0, next_perceived_stone.perceived_coords[0])
                        next_size = get_stone_feature_name(1, next_perceived_stone.perceived_coords[1])
                        next_roundness = get_stone_feature_name(2, next_perceived_stone.perceived_coords[2])
                        # Use aligned reward directly for next stone
                        next_reward = next_aligned_stone_from_map.reward
                        next_stone_desc_str = f"NEXT_STONE_DESC:{{color:{next_color},size:{next_size},roundness:{next_roundness},reward:{next_reward}}}"
                    except Exception as e:
                        print(f"Warning: Could not generate description for next_node {str(next_node)}: {e}")
                    parts.append(f"EDGE:{str(next_node)}:{potion.idx}:{potion_color_name}:{potion.dimension}:{potion.direction}:{next_stone_desc_str}")

    parts.append(f"POTION_MAP:{':'.join(map(str, chemistry.potion_map.dim_map))}")
    parts.append(f"POTION_DIR:{':'.join(map(str, chemistry.potion_map.dir_map))}")
    parts.append(f"STONE_MAP:{':'.join(map(str, chemistry.stone_map.latent_pos_dir))}")
    rotation_flat = chemistry.rotation.flatten()
    parts.append(f"ROTATION:{':'.join(map(str, rotation_flat))}")
    
    return "||".join(parts)

# --- Component Generation Functions ---

def generate_all_potion_maps() -> List[PotionMap]:
    potion_maps = []
    dim_permutations = list(itertools.permutations(range(3)))
    dir_combinations = list(itertools.product([-1, 1], repeat=3))
    for dim_map in dim_permutations:
        for dir_map_tuple in dir_combinations:
            # Ensure dir_map is a numpy array as expected by PotionMap
            dir_map_array = np.array(dir_map_tuple, dtype=np.int32)
            potion_maps.append(PotionMap(dim_map=np.array(dim_map), dir_map=dir_map_array))
    return potion_maps

def generate_all_stone_maps() -> List[StoneMap]:
    return stones_and_potions.possible_stone_maps()

def generate_all_graphs_with_constraints() -> List[Dict[str, Any]]:
    graphs_with_constraints = []
    possible_constraints = graphs.possible_constraints()
    for constraint in possible_constraints:
        graph = graphs.create_graph_from_constraint(constraint)
        graphs_with_constraints.append({"graph": graph, "constraint_str": str(constraint)})
    return graphs_with_constraints

def generate_all_rotations() -> List[np.ndarray]:
    return stones_and_potions.possible_rotations()


def save_json_data(data: Dict[str, Any], filename: str, compress: bool = True):
    """Save data as JSON, optionally compressed with gzip."""
    if compress:
        if not filename.endswith('.gz'):
            filename += '.gz'
        with gzip.open(filename, 'wt', encoding='utf-8') as f:
            json.dump(data, f)
    else:
        with open(filename, 'w') as f:
            json.dump(data, f)
    return filename

def main():
    parser = argparse.ArgumentParser(description="Deterministically generate unique chemistry graphs.")
    parser.add_argument(
        "--output_file",
        type=str,
        default="deterministic_chemistries_167424_80_unique_stones.json",
        help="Path to save the generated chemistries JSON file."
    )
    parser.add_argument(
        "--save_interval",
        type=int,
        default=1670000,
        help="How often to save intermediate results (number of combinations processed)."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit the number of combinations to process (0 for no limit)."
    )
    parser.add_argument(
        "--compress_json",
        type=str,
        default="true",
        help="Whether to compress the JSON output with gzip (true/false)."
    )
    args = parser.parse_args()

    # Convert string argument to boolean
    compress = args.compress_json.lower() == "true"

    print("Generating all PotionMap components...")
    all_potion_maps = generate_all_potion_maps()
    print(f"Generated {len(all_potion_maps)} PotionMaps.")

    print("Generating all StoneMap components...")
    all_stone_maps = generate_all_stone_maps()
    print(f"Generated {len(all_stone_maps)} StoneMaps.")

    print("Generating all Graph components (from constraints)...")
    all_graphs_with_constraints = generate_all_graphs_with_constraints()
    print(f"Generated {len(all_graphs_with_constraints)} Graphs.")

    print("Generating all Rotation components...")
    all_rotations = generate_all_rotations()
    print(f"Generated {len(all_rotations)} Rotations.")

    generated_chemistries_data: Dict[str, Any] = {}
    unique_chemistries_count = 0
    
    total_combinations = len(all_potion_maps) * len(all_stone_maps) * len(all_graphs_with_constraints) * len(all_rotations) 
    print(f"Total combinations to process: {total_combinations}")

    pbar = tqdm(total=total_combinations, desc="Generating and processing chemistries", unit="chemistry")

    combination_iterator = itertools.product(
        all_potion_maps, all_stone_maps, all_graphs_with_constraints, all_rotations
    )
    
    for i, (potion_map, stone_map, graph_info, rotation) in enumerate(combination_iterator):
        current_graph = graph_info["graph"]
        constraint_str = graph_info["constraint_str"] # For logging
        
        # # Force a duplicate on the second iteration to test the else condition
        # if i == 1:
        #     # Reuse the first combination to create a duplicate
        #     potion_map = all_potion_maps[0]
        #     stone_map = all_stone_maps[0] 
        #     current_graph = all_graphs_with_constraints[0]["graph"]
        #     constraint_str = all_graphs_with_constraints[0]["constraint_str"]
        #     rotation = all_rotations[0]

        current_chemistry = type_utils.Chemistry(
            potion_map=potion_map,
            stone_map=stone_map,
            graph=current_graph,
            rotation=rotation
        )

        is_complete_graph = is_graph_complete(current_chemistry.graph)
        canonical_string = generate_canonical_graph_string_extended(current_chemistry)
        chemistry_hash = hashlib.sha256(canonical_string.encode('utf-8')).hexdigest()
        
        is_unique_hash = chemistry_hash not in SEEN_CHEMISTRY_HASHES
        
        if is_unique_hash:
            SEEN_CHEMISTRY_HASHES.add(chemistry_hash)
            
            # Extract detailed graph data for JSON output
            current_chemistry_graph_output: Dict[str, Any] = {}
            try:
                nodes = current_chemistry.graph.node_list.nodes
                graph_edges = current_chemistry.graph.edge_list.edges

                for node_obj in nodes:
                    node_obj_str = str(node_obj)
                    try:
                        latent_stone = LatentStone(node_obj.coords)
                        aligned_stone = current_chemistry.stone_map.apply_inverse(latent_stone)
                        aligned_stone_with_reward = AlignedStone(aligned_stone.reward, aligned_stone.aligned_coords)
                        perceived_stone = unalign(aligned_stone_with_reward, current_chemistry.rotation)
                        
                        color = get_stone_feature_name(0, perceived_stone.perceived_coords[0])
                        size = get_stone_feature_name(1, perceived_stone.perceived_coords[1])
                        roundness = get_stone_feature_name(2, perceived_stone.perceived_coords[2])
                        current_stone_desc = f"{{color: {color}, size: {size}, roundness: {roundness},reward: {aligned_stone_with_reward.reward}}}"
                    except Exception as desc_e:
                        print(f"Warning: Could not generate description for node {node_obj_str}: {desc_e}")
                        current_stone_desc = "Error generating description"

                    current_chemistry_graph_output[node_obj_str] = {
                        "current_stone_description": current_stone_desc,
                        "transitions": []
                    }

                    if node_obj in graph_edges:
                        for next_node_obj, edge_info_tuple in graph_edges[node_obj].items():
                            if len(edge_info_tuple) >= 2 and isinstance(edge_info_tuple[1], Potion):
                                potion_obj = edge_info_tuple[1]
                                potion_idx = potion_obj.idx
                                potion_color = potion_index_to_color_name(potion_idx)
                                next_node_obj_str = str(next_node_obj)
                                
                                current_chemistry_graph_output[node_obj_str]["transitions"].append({
                                    "potion_index": potion_idx,
                                    "potion_color": potion_color,
                                    "next_node_str": next_node_obj_str,
                                    "potion_details": str(potion_obj)
                                })
                            else:
                                print(f"Warning: Unexpected edge info for edge from {node_obj_str} to {str(next_node_obj)}: {edge_info_tuple}")
                
                generated_chemistries_data[str(unique_chemistries_count)] = {
                    "graph": current_chemistry_graph_output,
                    "is_complete": is_complete_graph,
                    "is_unique_hash": True, # True because we added it to SEEN_CHEMISTRY_HASHES
                    "canonical_string": canonical_string,
                    "original_components": {
                        # "potion_map_dim": potion_map.dim_map,
                        # "potion_map_dir": potion_map.dir_map,
                        # "stone_map_latent_pos_dir": list(stone_map.latent_pos_dir),
                        "graph_constraint": constraint_str,
                        # "rotation": rotation.tolist() 
                    },
                    "canonical_hash": chemistry_hash # Store the hash for reference
                }
                unique_chemistries_count += 1

            except Exception as e:
                print(f"Error extracting graph data for chemistry {unique_chemistries_count}: {e}")
        else:
            print(f"Skipping duplicate chemistry with hash {chemistry_hash} (canonical string: {canonical_string})")
        
        pbar.update(1)
        if (i + 1) % args.save_interval == 0:
            print(f"\nSaving intermediate results ({unique_chemistries_count} unique chemistries found so far)...")
            final_filename = save_json_data(generated_chemistries_data, args.output_file, compress)
            print(f"Saved to {final_filename}")
        
        if args.limit > 0 and (i + 1) >= args.limit:
            print(f"\nReached limit of {args.limit} combinations. Stopping early.")
            break

    pbar.close()
    print(f"\nProcessed all {total_combinations} combinations.")
    print(f"Found {unique_chemistries_count} unique chemistries.")
    
    print(f"Saving final data to {args.output_file}...")
    final_filename = save_json_data(generated_chemistries_data, args.output_file, compress)
    print(f"Saved to {final_filename}")
    print("All done!")

if __name__ == "__main__":
    main()
