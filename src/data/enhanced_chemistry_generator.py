#!/usr/bin/env python3
"""
Enhanced chemistry generator that combines structural analysis with transition generation.
This script generates all unique chemistry configurations by considering both:
1. Chemistry structure (components, graph topology)
2. Actual transitions (what happens when you apply potions to stones)

This addresses the issue where structurally different chemistries might produce
identical transitions, providing more robust uniqueness detection.
"""
import argparse
import gzip
import hashlib
import itertools
import json
import multiprocessing as mp
import numpy as np
import pickle
from functools import partial
from tqdm import tqdm
from typing import Dict, Any, Tuple, List, Set

# Imports from dm_alchemy
from dm_alchemy.types import stones_and_potions, utils as type_utils, graphs
from dm_alchemy.types.stones_and_potions import (
    PotionMap, StoneMap, LatentStone, AlignedStone, PerceivedStone, 
    unalign, align, Potion, possible_latent_stones, possible_perceived_potions
)
from dm_alchemy.ideal_observer import precomputed_maps

# --- Constants ---
POTION_INDEX_TO_COLOR_VALUE = [-1.0, -2/3, -1/3, 0.0, 1/3, 2/3]

# --- Helper Functions ---

def get_stone_feature_name(feature_index: int, value: float) -> str:
    """Maps numerical stone feature values (-1, 0, 1) to descriptive names."""
    if feature_index == 0:  # Color
        if np.isclose(value, -1.0): return "blue"
        if np.isclose(value, 0.0): return "purple"
        if np.isclose(value, 1.0): return "red"
    elif feature_index == 1:  # Size
        if np.isclose(value, -1.0): return "small"
        if np.isclose(value, 0.0): return "medium"
        if np.isclose(value, 1.0): return "large"
    elif feature_index == 2:  # Roundness
        if np.isclose(value, -1.0): return "pointy"
        if np.isclose(value, 0.0): return "medium_round"
        if np.isclose(value, 1.0): return "round"
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
    """Checks if a chemistry graph is complete (has 8 nodes with 3 outgoing edges each)."""
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

# --- Transition Generation ---

class TempAlchemyEnv:
    """Temporary AlchemyEnv-like class for generating transitions."""
    
    def __init__(self, chemistry):
        self.stone_map = chemistry.stone_map
        self.potion_map = chemistry.potion_map
        self.graph = chemistry.graph
        self.rotation = chemistry.rotation
    
    def possible_perceived_stones(self):
        aligned_stones = [
            self.stone_map.apply_inverse(stone) for stone in possible_latent_stones()
        ]
        return [unalign(stone, self.rotation) for stone in aligned_stones]
    
    def possible_perceived_potions(self):
        return possible_perceived_potions()
    
    def apply_potion(self, perceived_stone, perceived_potion):
        """Apply potion to stone, same logic as in alchemy_datagen.py"""
        aligned_stone = align(perceived_stone, self.rotation)
        latent_stone = self.stone_map.apply(aligned_stone)
        latent_potion = self.potion_map.apply(perceived_potion)

        start_node = self.graph.node_list.get_node_by_coords(
            list(latent_stone.latent_coords)
        )

        end_node = None
        for node, v in self.graph.edge_list.edges[start_node].items():
            if v[1].latent_potion() == latent_potion: # latent_potion() is a method of Potion.
                end_node = node
                break

        if end_node is not None:
            end_latent_stone = LatentStone(np.array(end_node.coords))
            end_aligned_stone = self.stone_map.apply_inverse(end_latent_stone)
            end_stone = unalign(end_aligned_stone, self.rotation)
            return end_stone
        else:
            return None  # No valid transition

def generate_transitions_for_chemistry(chemistry):
    """
    Generate all transitions for a given chemistry, similar to alchemy_datagen.py
    Returns a sorted list of transition tuples for consistent canonical representation.
    """
    alchemy_env = TempAlchemyEnv(chemistry)
    
    # Generate all transitions
    transitions = []
    potions = alchemy_env.possible_perceived_potions()
    stones = alchemy_env.possible_perceived_stones()

    for p, s in itertools.product(potions, stones):
        s_ = alchemy_env.apply_potion(s, p)
        
        if s_ is not None:  # Only include valid transitions
            transition_tuple = (
                tuple(s.perceived_coords),
                s.reward,
                p.index(),
                tuple(s_.perceived_coords),
                s_.reward
            )
            transitions.append(transition_tuple)
    
    # Sort transitions for consistent canonical representation
    return sorted(transitions)

# --- Rolled Out Graph Generation ---

def generate_rolled_out_graph(chemistry) -> Dict[str, Any]:
    """
    Generate a rolled-out graph structure similar to deterministic_chemistry_generator.py
    This creates a detailed graph representation that can be used for data generation.
    """
    if not chemistry or not hasattr(chemistry, 'graph'):
        return {}
    
    rolled_out_graph: Dict[str, Any] = {}
    
    try:
        nodes = chemistry.graph.node_list.nodes
        graph_edges = chemistry.graph.edge_list.edges

        for node_obj in nodes:
            node_obj_str = str(node_obj)
            try:
                latent_stone = LatentStone(node_obj.coords)
                aligned_stone = chemistry.stone_map.apply_inverse(latent_stone)
                perceived_stone = unalign(aligned_stone=aligned_stone, rotation=chemistry.rotation)
                
                color = get_stone_feature_name(0, perceived_stone.perceived_coords[0])
                size = get_stone_feature_name(1, perceived_stone.perceived_coords[1])
                roundness = get_stone_feature_name(2, perceived_stone.perceived_coords[2])
                current_stone_desc = f"{{color: {color}, size: {size}, roundness: {roundness}, reward: {perceived_stone.reward}}}"
            except Exception as desc_e:
                print(f"Warning: Could not generate description for node {node_obj_str}: {desc_e}")
                current_stone_desc = "Error generating description"

            rolled_out_graph[node_obj_str] = {
                "current_stone_description": current_stone_desc,
                "transitions": []
            }

            if node_obj in graph_edges:
                for next_node_obj, edge_info_tuple in graph_edges[node_obj].items():
                    if len(edge_info_tuple) >= 2 and isinstance(edge_info_tuple[1], Potion):
                        potion_obj = edge_info_tuple[1]
                        
                        # Convert latent potion to perceived potion for consistency
                        latent_potion = potion_obj.latent_potion()
                        perceived_potion = chemistry.potion_map.apply_inverse(latent_potion)
                        perceived_potion_idx = perceived_potion.index()
                        potion_color = potion_index_to_color_name(perceived_potion_idx)
                        next_node_obj_str = str(next_node_obj)
                        
                        rolled_out_graph[node_obj_str]["transitions"].append({
                            "potion_index": perceived_potion_idx,  # Use perceived potion index
                            "potion_color": potion_color,
                            "next_node_str": next_node_obj_str,
                            "potion_details": str(potion_obj)
                        })
                    else:
                        print(f"Warning: Unexpected edge info for edge from {node_obj_str} to {str(next_node_obj)}: {edge_info_tuple}")
    
    except Exception as e:
        print(f"Error extracting rolled-out graph data: {e}")
        return {}
    
    return rolled_out_graph

# --- Canonical String Generation ---

def generate_structural_parts_from_transitions(transitions):
    """
    Generate structural parts derived from transitions (actual behavior).
    This captures what transitions actually exist, potentially different from intended structure.
    Returns string parts in the same format as chemistry structure for easy comparison.
    """
    structural_parts_from_transitions = []
    
    # Group transitions by starting stone
    transitions_by_start = {}
    for t in transitions:
        start_coords = t[0]  # (x, y, z)
        start_reward = t[1]
        potion_idx = t[2]
        end_coords = t[3]  # (x, y, z)
        end_reward = t[4]
        
        start_key = f"{start_coords},{start_reward}"
        if start_key not in transitions_by_start:
            transitions_by_start[start_key] = []
        transitions_by_start[start_key].append((potion_idx, end_coords, end_reward))
    
    # Sort for consistency
    sorted_starts = sorted(transitions_by_start.keys())
    
    for start_key in sorted_starts:
        transitions_from_start = transitions_by_start[start_key]
        
        # Parse start state
        start_parts = start_key.split(',')
        start_coords = (int(start_parts[0][1:]), int(start_parts[1]), int(start_parts[2][:-1]))  # Remove parentheses
        start_reward = int(start_parts[3])
        
        # Add start stone description
        start_color = get_stone_feature_name(0, start_coords[0])
        start_size = get_stone_feature_name(1, start_coords[1])
        start_roundness = get_stone_feature_name(2, start_coords[2])
        
        structural_parts_from_transitions.append(
            f"{start_coords}:DESC:{{color:{start_color},size:{start_size},roundness:{start_roundness},reward:{start_reward}}}"
        )
        
        # Sort transitions from this start state
        sorted_transitions = sorted(transitions_from_start, key=lambda x: (x[0], x[1], x[2]))
        
        for potion_idx, end_coords, end_reward in sorted_transitions:
            potion_color = potion_index_to_color_name(potion_idx)
            end_color = get_stone_feature_name(0, end_coords[0])
            end_size = get_stone_feature_name(1, end_coords[1])
            end_roundness = get_stone_feature_name(2, end_coords[2])
            
            structural_parts_from_transitions.append(
                f"{end_coords}:{potion_idx}:{potion_color}:END_DESC:{{color:{end_color},size:{end_size},roundness:{end_roundness},reward:{end_reward}}}"
            )
        structural_parts_from_transitions.append('|')  # Delimiter for transitions from this start state
    
    # Remove the last '|' added by the loop
    if structural_parts_from_transitions and structural_parts_from_transitions[-1] == '|':
        structural_parts_from_transitions.pop()
    
    return structural_parts_from_transitions

# def generate_canonical_graph_string_with_transitions(chemistry) -> str:
#     """
#     Enhanced canonical string that includes:
#     1. Chemistry structure (components, graph topology)
#     2. Structural parts derived from actual transitions (behavior-based)
#     3. Actual transitions (what happens when you apply potions to stones)
#     This ensures that chemistries with different transitions are treated as unique.
#     """
#     if not chemistry or not hasattr(chemistry, 'graph'):
#         return ""
    
#     # Generate transitions first (we need them for both structural analysis and final string)
#     transitions = generate_transitions_for_chemistry(chemistry)
    
#     # Get the structural canonical string (from chemistry configuration)
#     structural_parts = []
#     nodes = chemistry.graph.node_list.nodes
    
#     # Sort nodes for consistency with transition structure generation
#     sorted_nodes = sorted(nodes, key=lambda node: str(node.coords))
    
#     for node in sorted_nodes:
#         node_str = str(node)
        
#         try:
#             latent_stone = LatentStone(node.coords)
#             aligned_stone_from_map = chemistry.stone_map.apply_inverse(latent_stone)
#             perceived_stone = unalign(aligned_stone_from_map, chemistry.rotation)
            
#             color = get_stone_feature_name(0, perceived_stone.perceived_coords[0])
#             size = get_stone_feature_name(1, perceived_stone.perceived_coords[1])
#             roundness = get_stone_feature_name(2, perceived_stone.perceived_coords[2])
#             reward = aligned_stone_from_map.reward
            
#             # Remove STRUCT_START prefix for consistency
#             structural_parts.append(
#                 f"{tuple(perceived_stone.perceived_coords)}:DESC:{{color:{color},size:{size},roundness:{roundness},reward:{reward}}}"
#             )
#         except Exception as e:
#             structural_parts.append(f"error:DESC:error")
        
#         if node in chemistry.graph.edge_list.edges:
#             graph_edges = chemistry.graph.edge_list.edges[node]
#             for next_node, edge_info in graph_edges.items():
#                 if len(edge_info) >= 2 and isinstance(edge_info[1], Potion):
#                     potion = edge_info[1]
                    
#                     # Convert latent potion to perceived potion
#                     latent_potion = potion.latent_potion()
#                     perceived_potion = chemistry.potion_map.apply_inverse(latent_potion)
#                     perceived_potion_idx = perceived_potion.index()
#                     potion_color_name = potion_index_to_color_name(perceived_potion_idx)
                    
#                     try:
#                         # Get next stone description for consistency with TRANS_EDGE format
#                         next_latent_stone = LatentStone(next_node.coords)
#                         next_aligned_stone_from_map = chemistry.stone_map.apply_inverse(next_latent_stone)
#                         next_perceived_stone = unalign(next_aligned_stone_from_map, chemistry.rotation)
                        
#                         next_color = get_stone_feature_name(0, next_perceived_stone.perceived_coords[0])
#                         next_size = get_stone_feature_name(1, next_perceived_stone.perceived_coords[1])
#                         next_roundness = get_stone_feature_name(2, next_perceived_stone.perceived_coords[2])
#                         next_reward = next_aligned_stone_from_map.reward
                        
#                         # Use perceived potion index instead of latent potion index
#                         structural_parts.append(
#                             f"{tuple(next_perceived_stone.perceived_coords)}:{perceived_potion_idx}:{potion_color_name}:END_DESC:{{color:{next_color},size:{next_size},roundness:{next_roundness},reward:{next_reward}}}"
#                         )
#                     except Exception as e:
#                         structural_parts.append(f"error:{perceived_potion_idx}:{potion_color_name}:END_DESC:error")

#     structural_parts.append(f"POTION_MAP:{':'.join(map(str, chemistry.potion_map.dim_map))}")
#     structural_parts.append(f"POTION_DIR:{':'.join(map(str, chemistry.potion_map.dir_map))}")
#     structural_parts.append(f"STONE_MAP:{':'.join(map(str, chemistry.stone_map.latent_pos_dir))}")
#     rotation_flat = chemistry.rotation.flatten()
#     structural_parts.append(f"ROTATION:{':'.join(map(str, rotation_flat))}")
    
#     structural_string = "||".join(structural_parts)
    
#     # Get structural parts derived from actual transitions (behavior-based)
#     structural_parts_from_transitions = generate_structural_parts_from_transitions(transitions)
#     structural_from_transitions_string = "||".join(structural_parts_from_transitions)
    
#     # Generate transitions string
#     transitions_string = "TRANSITIONS:" + "|".join([
#         f"{t[0]},{t[1]},{t[2]},{t[3]},{t[4]}" for t in transitions
#     ])
    
#     return f"STRUCT:{structural_string}||STRUCT_FROM_TRANS:{structural_from_transitions_string}||{transitions_string}"

def generate_canonical_data_with_transitions(chemistry) -> Dict[str, Any]:
    """
    Enhanced canonical data that separates all components into their own keys:
    1. structural_string: Chemistry structure (without prefixes)
    2. transition_structural_string: Structural parts from transitions (without prefixes)  
    3. transitions_string: Raw transition data
    4. transitions: List of transition tuples
    5. canonical_string: Combined string for hashing/uniqueness
    This keeps the string formats identical between chemistry and transitions for easy comparison.
    """
    if not chemistry or not hasattr(chemistry, 'graph'):
        return {}
    
    # Generate transitions first
    transitions = generate_transitions_for_chemistry(chemistry)
    
    # Get the structural canonical string (from chemistry configuration)
    structural_parts = []
    nodes = chemistry.graph.node_list.nodes
    
    # Sort nodes for consistency with transition structure generation
    sorted_nodes = sorted(nodes, key=lambda node: str(node.coords))
    
    for node in sorted_nodes:
        try:
            latent_stone = LatentStone(node.coords)
            aligned_stone_from_map = chemistry.stone_map.apply_inverse(latent_stone)
            perceived_stone = unalign(aligned_stone_from_map, chemistry.rotation)
            
            color = get_stone_feature_name(0, perceived_stone.perceived_coords[0])
            size = get_stone_feature_name(1, perceived_stone.perceived_coords[1])
            roundness = get_stone_feature_name(2, perceived_stone.perceived_coords[2])
            reward = aligned_stone_from_map.reward
            
            structural_parts.append(
                f"{tuple(perceived_stone.perceived_coords)}:DESC:{{color:{color},size:{size},roundness:{roundness},reward:{reward}}}"
            )
        except Exception as e:
            structural_parts.append(f"error:DESC:error")
        
        if node in chemistry.graph.edge_list.edges:
            graph_edges = chemistry.graph.edge_list.edges[node]
            for next_node, edge_info in graph_edges.items():
                if len(edge_info) >= 2 and isinstance(edge_info[1], Potion):
                    potion = edge_info[1] # in the latent representation.
                    
                    # Convert latent potion to perceived potion
                    latent_potion = potion.latent_potion() # The edges on the graph are already latent potions.
                    # So we need to convert it to perceived potion.
                    
                    # This is similar to the alchemy_datagen.py logic.
                    perceived_potion = chemistry.potion_map.apply_inverse(latent_potion)
                    perceived_potion_idx = perceived_potion.index()
                    potion_color_name = potion_index_to_color_name(perceived_potion_idx)
                    
                    try:
                        next_latent_stone = LatentStone(next_node.coords)
                        next_aligned_stone_from_map = chemistry.stone_map.apply_inverse(next_latent_stone)
                        next_perceived_stone = unalign(next_aligned_stone_from_map, chemistry.rotation)
                        
                        next_color = get_stone_feature_name(0, next_perceived_stone.perceived_coords[0])
                        next_size = get_stone_feature_name(1, next_perceived_stone.perceived_coords[1])
                        next_roundness = get_stone_feature_name(2, next_perceived_stone.perceived_coords[2])
                        next_reward = next_aligned_stone_from_map.reward
                        
                        structural_parts.append(
                            f"{tuple(next_perceived_stone.perceived_coords)}:{perceived_potion_idx}:{potion_color_name}:END_DESC:{{color:{next_color},size:{next_size},roundness:{next_roundness},reward:{next_reward}}}"
                        )
                    except Exception as e:
                        structural_parts.append(f"error:{perceived_potion_idx}:{potion_color_name}:END_DESC:error")
    #     # For each node, append a ||
        structural_parts.append("|")  # Delimiter for nodes.
    
    # Before appending the other parts, # remove the last '|' added by the loop.
    if structural_parts and structural_parts[-1] == "|":
        structural_parts.pop()
    
    # # Now first concatenate the structural parts.
    # structural_parts_temp = ''.join(structural_parts)            

    # Add chemistry component mappings to structural parts
    structural_parts.append(f"POTION_MAP:{':'.join(map(str, chemistry.potion_map.dim_map))}")
    structural_parts.append(f"POTION_DIR:{':'.join(map(str, chemistry.potion_map.dir_map))}")
    structural_parts.append(f"STONE_MAP:{':'.join(map(str, chemistry.stone_map.latent_pos_dir))}")
    rotation_flat = chemistry.rotation.flatten()
    structural_parts.append(f"ROTATION:{':'.join(map(str, rotation_flat))}")
    
    
    # For the structural strings from both cases, use a single '|' delimiter.
    # Create separate strings
    # For all the parts except the last 4, use '|' as a separator, but for the rest, use '||'.
    
    temp_structural_parts = structural_parts[:-4]  # All but the last 4 parts
    final_structural_parts = structural_parts[-4:]  # The last 4 parts (mappings and rotation)
    
    structural_string = ''.join(temp_structural_parts)
    # The last 4 parts are the mappings and rotation, which we keep as a single string.
    final_structural_string = "||".join(final_structural_parts)
    # Combine the two parts
    structural_string += "||" + final_structural_string
    
    
    # Get structural parts derived from actual transitions (behavior-based)
    structural_parts_from_transitions = generate_structural_parts_from_transitions(transitions)
    transition_structural_string = ''.join(structural_parts_from_transitions)
    
    # Generate transitions string
    transitions_string = "TRANSITIONS:" + "|".join([
        f"{t[0]},{t[1]},{t[2]},{t[3]},{t[4]}" for t in transitions
    ])
    
    # Combined canonical string for hashing (add prefixes here for uniqueness)
    canonical_string = f"STRUCT:{structural_string}||STRUCT_FROM_TRANS:{transition_structural_string}||{transitions_string}"
    
    # Generate rolled-out graph structure for data generation
    rolled_out_graph = generate_rolled_out_graph(chemistry)
    
    # Create separated canonical data dictionary
    canonical_data = {
        "structural_string": structural_string,
        "transition_structural_string": transition_structural_string,
        "transitions_string": transitions_string,
        "transitions": transitions,
        "canonical_string": canonical_string,
        "summary": {
            "num_transitions": len(transitions),
            "is_complete": is_graph_complete(chemistry.graph)
        },
        "graph": rolled_out_graph,  # Store the rolled-out graph for data generation
    }
    
    return canonical_data

# --- Component Generation Functions ---

def generate_all_potion_maps() -> List[PotionMap]:
    """Generate all possible potion maps."""
    potion_maps = []
    dim_permutations = list(itertools.permutations(range(3)))
    dir_combinations = list(itertools.product([-1, 1], repeat=3))
    for dim_map in dim_permutations:
        for dir_map_tuple in dir_combinations:
            dir_map = list(dir_map_tuple)
            potion_map = PotionMap(list(dim_map), dir_map)
            potion_maps.append(potion_map)
    return potion_maps

def generate_all_stone_maps() -> List[StoneMap]:
    """Generate all possible stone maps."""
    return stones_and_potions.possible_stone_maps()

def generate_all_graphs_with_constraints() -> List[Dict[str, Any]]:
    """Generate all possible graphs with their constraints."""
    graphs_with_constraints = []
    
    # Use the same approach as alchemy_datagen.py
    constraints = graphs.possible_constraints()
    graphs_distr = graphs.graph_distr(constraints)
    graphs_distr_as_list = list(graphs_distr.items())
    graphs_distr_constraints = [
        graphs.constraint_from_graph(k) for k, _ in graphs_distr_as_list
    ]
    graphs_distr_num_constraints = graphs.get_num_constraints(
        graphs_distr_constraints
    )
    graphs_distr_sorted = sorted(
        zip(
            graphs_distr_as_list,
            graphs_distr_num_constraints,
            graphs_distr_constraints,
        ),
        key=lambda x: (x[2], str(x[1])),
    )
    
    for (graph, _), _, constraint in graphs_distr_sorted:
        graphs_with_constraints.append({
            "graph": graph, 
            "constraint_str": str(constraint)
        })
    
    return graphs_with_constraints

def generate_all_rotations() -> List[np.ndarray]:
    """Generate all possible rotations."""
    return stones_and_potions.possible_rotations()

# --- Multiprocessing Worker Function ---

def process_chemistry_chunk(chunk_data, all_potion_maps, all_stone_maps, all_graphs_with_constraints, all_rotations, save_transitions):
    """
    Process a chunk of (i, j, k, l) combinations and return unique chemistry data.
    Similar to data_gen_with_hash.py but for chemistry uniqueness detection.
    """
    local_seen_hashes = set()
    chunk_results = []
    
    for i, j, k, l in tqdm(chunk_data):
        potion_map = all_potion_maps[i]
        stone_map = all_stone_maps[j]
        graph_info = all_graphs_with_constraints[k]
        rotation = all_rotations[l]
        
        current_graph = graph_info["graph"]
        constraint_str = graph_info["constraint_str"]
        
        # Create chemistry object
        current_chemistry = type_utils.Chemistry(
            potion_map=potion_map,
            stone_map=stone_map,
            graph=current_graph,
            rotation=rotation
        )

        # Use the enhanced canonical data with separated components
        canonical_data = generate_canonical_data_with_transitions(current_chemistry)
        
        if not canonical_data:
            print("Warning: Empty canonical data for combination ")
            print(f"i={i}, j={j}, k={k}, l={l}")
            continue
            
        # Use the combined canonical string for hashing
        canonical_string = canonical_data["canonical_string"]
        chemistry_hash = hashlib.sha256(canonical_string.encode('utf-8')).hexdigest()
        
        # Check uniqueness within this chunk
        if chemistry_hash not in local_seen_hashes:
            local_seen_hashes.add(chemistry_hash)
            
            # Convert transitions to numpy array format similar to alchemy_datagen.py
            transitions = canonical_data["transitions"]
            transition_matrix = None
            if transitions and save_transitions:
                transition_array = np.array([
                    list(t[0]) + [t[1], t[2]] + list(t[3]) + [t[4]] 
                    for t in transitions
                ], dtype=int)
                transition_matrix = transition_array
            
            # Store comprehensive chemistry data with separated components
            chemistry_data = {
                "chemistry_indices": {
                    "potion_map_idx": i,
                    "stone_map_idx": j, 
                    "graph_idx": k,
                    "rotation_idx": l
                },
                "constraint_str": constraint_str,
                "canonical_hash": chemistry_hash,
                # Separated canonical string components
                "structural_string": canonical_data["structural_string"],
                "transition_structural_string": canonical_data["transition_structural_string"],
                "transitions_string": canonical_data["transitions_string"],
                "canonical_string": canonical_data["canonical_string"],
                # Summary data
                "num_transitions": len(transitions),
                "is_complete": canonical_data["summary"]["is_complete"],
                # Rolled-out graph for data generation
                "graph": canonical_data["graph"],
            }
            
            if save_transitions:
                chemistry_data.update({
                    "transitions": transitions,  # List of transition tuples
                    "transition_matrix": transition_matrix.tolist() if transition_matrix is not None else None,
                })
            
            chunk_results.append(chemistry_data)
    
    return chunk_results


def save_pickle_data(data: Dict[str, Any], filename: str):
    """Save data as pickle."""
    with open(filename, 'wb') as f:
        pickle.dump(data, f)
    return filename


def main():
    parser = argparse.ArgumentParser(description="Enhanced chemistry generator with transitions.")
    parser.add_argument(
        "--output_file",
        type=str,
        default="enhanced_chemistries_with_transitions.pkl",
        help="Path to save the generated chemistries pickle file."
    )
    parser.add_argument(
        "--save_interval",
        type=int,
        default=10000,
        help="How often to save intermediate results (number of combinations processed)."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit the number of combinations to process (0 for no limit)."
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=10,
        help="Number of worker processes for multiprocessing."
    )
    parser.add_argument(
        "--save_transitions",
        type=str,
        default="true",
        help="Whether to save transition matrices (true/false)."
    )
    args = parser.parse_args()

    # Convert string arguments to boolean
    save_transitions = args.save_transitions.lower() == "true"

    print("Generating all components...")
    all_potion_maps = generate_all_potion_maps()
    all_stone_maps = generate_all_stone_maps()
    all_graphs_with_constraints = generate_all_graphs_with_constraints()
    all_rotations = generate_all_rotations()
    
    print(f"Components: {len(all_potion_maps)} PotionMaps, {len(all_stone_maps)} StoneMaps, "
          f"{len(all_graphs_with_constraints)} Graphs, {len(all_rotations)} Rotations")

    # Generate all combinations of indices
    all_combinations = []
    for i in range(len(all_potion_maps)):
        for j in range(len(all_stone_maps)):
            for k in range(len(all_graphs_with_constraints)):
                for l in range(len(all_rotations)):
                    all_combinations.append((i, j, k, l))
    
    total_combinations = len(all_combinations)
    print(f"Total combinations to process: {total_combinations}")
    
    # Apply limit if specified
    if args.limit > 0:
        all_combinations = all_combinations[:args.limit]
        print(f"Limited to {len(all_combinations)} combinations for testing")

    # Split combinations into chunks for multiprocessing
    num_workers = args.num_workers
    chunk_size = len(all_combinations) // num_workers
    if len(all_combinations) % num_workers != 0:
        chunk_size += 1
    
    chunks = [all_combinations[i:i + chunk_size] for i in range(0, len(all_combinations), chunk_size)]
    print(f"Split into {len(chunks)} chunks with approximately {chunk_size} combinations each")
    
    # Create a partial function with the shared data
    worker_func = partial(process_chemistry_chunk, 
                         all_potion_maps=all_potion_maps,
                         all_stone_maps=all_stone_maps, 
                         all_graphs_with_constraints=all_graphs_with_constraints,
                         all_rotations=all_rotations,
                         save_transitions=save_transitions)
    
    # Process chunks in parallel
    print(f"Starting multiprocessing with {num_workers} workers...")
    with mp.Pool(num_workers) as pool:
        chunk_results = list(tqdm(pool.imap(worker_func, chunks), 
                                 total=len(chunks), 
                                 desc="Processing chunks"))
    
    # Flatten the results and remove global duplicates
    print("Consolidating results and removing global duplicates...")
    generated_chemistries_data: Dict[str, Any] = {}
    global_seen_hashes = set()
    unique_chemistries_count = 0
    total_processed = len(all_combinations)
    
    for chunk_result in tqdm(chunk_results, desc="Consolidating chunks"):
        for chemistry_data in chunk_result:
            chemistry_hash = chemistry_data["canonical_hash"]
            if chemistry_hash not in global_seen_hashes:
                global_seen_hashes.add(chemistry_hash)
                generated_chemistries_data[str(unique_chemistries_count)] = chemistry_data
                unique_chemistries_count += 1

    print(f"\nProcessed {total_processed} combinations.")
    print(f"Found {unique_chemistries_count} unique chemistries (including transitions).")
    print(f"Uniqueness ratio: {unique_chemistries_count/total_processed:.4f}")
    
    # Add summary statistics
    generated_chemistries_data["_metadata"] = {
        "total_combinations_processed": total_processed,
        "unique_chemistries_found": unique_chemistries_count,
        "uniqueness_ratio": unique_chemistries_count/total_processed,
        "components_counts": {
            "potion_maps": len(all_potion_maps),
            "stone_maps": len(all_stone_maps),
            "graphs": len(all_graphs_with_constraints),
            "rotations": len(all_rotations)
        },
        "includes_transitions": save_transitions,
        "includes_structural_from_transitions": True,
        "num_workers": num_workers
    }
    
    # Final save
    final_filename = save_pickle_data(generated_chemistries_data, args.output_file)
    print(f"Final data saved to {final_filename}")
    print("Enhanced chemistry generation completed!")

if __name__ == "__main__":
    # This guard is important for multiprocessing on some platforms
    mp.set_start_method('spawn', force=True)
    main()
