"""Generates a textual prompt for the Alchemy decision-making task
based on the symbolic observation vector, supporting a conversational flow.
"""
import numpy as np
import dm_env
from dm_alchemy import symbolic_alchemy
from dm_alchemy.types import stones_and_potions, utils as type_utils, graphs # Added graphs
from dm_alchemy.encode import chemistries_proto_conversion
from dm_alchemy import event_tracker
from typing import Optional, Dict, Any, Tuple, Set
import json # Added import
import hashlib # Added for chemistry hashing
import argparse
from tqdm import tqdm # Added for progress bar
# --- Constants based on symbolic_obs structure ---
# Set NUM_STONES here to synchronize with environment
NUM_STONES = 2  # Change this value to match num_stones_per_trial in env
FEATURES_PER_STONE = 5
NUM_POTIONS = 12
FEATURES_PER_POTION = 2
STONE_FEATURES_END = NUM_STONES * FEATURES_PER_STONE # Index 15
POTION_FEATURES_END = STONE_FEATURES_END + NUM_POTIONS * FEATURES_PER_POTION # Index 39

# Store seen chemistry hashes to track uniqueness
SEEN_CHEMISTRY_HASHES = set()

# Action indices mapping (based on notebook description)
# 0: no-op (assuming no end_trial action for simplicity here)
# 1: stone 0 -> cauldron
# 2: stone 0 -> potion 0
# ...
# 14: stone 1 -> cauldron
# 15: stone 1 -> potion 0
# ...
POTIONS_PLUS_CAULDRON = NUM_POTIONS + 1 # 13

# --- Mappings ---

# Added helper function for potion index to color mapping
POTION_INDEX_TO_COLOR_VALUE = [-1.0, -2/3, -1/3, 0.0, 1/3, 2/3]

def get_stone_feature_name(feature_index: int, value: float) -> str:
    """Maps numerical stone feature values (-1, 0, 1) to descriptive names.
    Indices: 0: color, 1: size, 2: roundness.
    """
    # Use np.isclose for robust float comparison
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

def get_reward_value(reward_indicator: float) -> int:
    """Maps the numerical reward indicator to the actual reward value."""
    if np.isclose(reward_indicator, -1.0): return -1
    if np.isclose(reward_indicator, -1/3): return -1/3
    if np.isclose(reward_indicator, 1/3): return 1/3
    if np.isclose(reward_indicator, 1.0): return 1.0
    return 0

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
    
    # A complete graph in this environment has 8 nodes with 3 outgoing edges each
    if len(nodes) != 8:
        return False
    
    # Check that each node has exactly 3 outgoing edges
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
    
    # Sort nodes to ensure consistent order
    nodes = sorted(chemistry.graph.node_list.nodes, key=str)
    
    for node in nodes:
        node_str = str(node)
        parts.append(f"NODE:{node_str}") # this appends the node's coords.
        
        # Get the node's perceived stone description
        try:
            latent_stone = stones_and_potions.LatentStone(node.coords)
            # Assuming chemistry.stone_map.apply_inverse returns an object
            # with .reward and .aligned_coords (e.g., an AlignedStone namedtuple)
            aligned_stone_from_map = chemistry.stone_map.apply_inverse(latent_stone)
            # The original code re-constructs AlignedStone, which is safe and explicit:
            aligned_stone_for_unalign = stones_and_potions.AlignedStone(
                aligned_stone_from_map.reward,
                aligned_stone_from_map.aligned_coords
            )
            perceived_stone = stones_and_potions.unalign(aligned_stone_for_unalign, chemistry.rotation)
            
            # Add perceived stone coords and reward (existing part)
            # coord_str = ":".join([str(c) for c in perceived_stone.perceived_coords])
            # parts.append(f"STONE:{perceived_stone.reward}:{coord_str}")

            # Add perceived stone descriptive string (new part, similar to line 276 logic)
            color = get_stone_feature_name(0, perceived_stone.perceived_coords[0])
            size = get_stone_feature_name(1, perceived_stone.perceived_coords[1])
            roundness = get_stone_feature_name(2, perceived_stone.perceived_coords[2])
            stone_desc_str = f"DESC:{{color:{color},size:{size},roundness:{roundness}}}"
            parts.append(stone_desc_str)

        except Exception as e:
            # Updated warning to reflect that full data generation might fail and log the exception
            print(f"Warning: Could not generate full stone data (coords, reward, description) for node {node_str}: {e}")
            parts.append(f"STONE:error")
            parts.append(f"DESC:error") # Add error placeholder for the description part as well
        
        # Sort transitions for consistent ordering
        if node in chemistry.graph.edge_list.edges:
            edges = chemistry.graph.edge_list.edges[node]
            sorted_edges = sorted(edges.items(), key=lambda x: str(x[0]))

            for next_node, edge_info in sorted_edges:
                if len(edge_info) >= 2 and isinstance(edge_info[1], stones_and_potions.Potion):
                    potion = edge_info[1]
                    potion_color_name = potion_index_to_color_name(potion.idx) # Get potion color name

                    # Get next stone's perceived state description
                    next_stone_desc_str = "DESC:error" # Default in case of error
                    try:
                        next_latent_stone = stones_and_potions.LatentStone(next_node.coords)
                        next_aligned_stone_from_map = chemistry.stone_map.apply_inverse(next_latent_stone)
                        next_aligned_stone_for_unalign = stones_and_potions.AlignedStone(
                            next_aligned_stone_from_map.reward,
                            next_aligned_stone_from_map.aligned_coords
                        )
                        next_perceived_stone = stones_and_potions.unalign(next_aligned_stone_for_unalign, chemistry.rotation)

                        next_color = get_stone_feature_name(0, next_perceived_stone.perceived_coords[0])
                        next_size = get_stone_feature_name(1, next_perceived_stone.perceived_coords[1])
                        next_roundness = get_stone_feature_name(2, next_perceived_stone.perceived_coords[2])
                        next_stone_desc_str = f"NEXT_STONE_DESC:{{color:{next_color},size:{next_size},roundness:{next_roundness}}}"
                    except Exception as e:
                        print(f"Warning: Could not generate description for next_node {str(next_node)}: {e}")

                    parts.append(f"EDGE:{str(next_node)}:{potion.idx}:{potion_color_name}:{potion.dimension}:{potion.direction}:{next_stone_desc_str}")

    # Add potion map information
    parts.append(f"POTION_MAP:{':'.join(map(str, chemistry.potion_map.dim_map))}")
    parts.append(f"POTION_DIR:{':'.join(map(str, chemistry.potion_map.dir_map))}")
    
    # Add stone map information
    parts.append(f"STONE_MAP:{':'.join(map(str, chemistry.stone_map.latent_pos_dir))}")
    
    # Add rotation information
    rotation_flat = chemistry.rotation.flatten()
    parts.append(f"ROTATION:{':'.join(map(str, rotation_flat))}")
    
    return "||".join(parts)

def check_chemistry_uniqueness_and_completeness(chemistry, episode_id) -> Tuple[bool, bool]:
    """
    Checks if a chemistry is both complete and unique (not seen before).
    
    Args:
        chemistry: The chemistry object to check
        episode_id: Current episode ID (for logging)
        
    Returns:
        Tuple of (is_complete, is_newly_seen_unique_and_complete)
    """
    # # First check if the graph is complete
    is_complete = is_graph_complete(chemistry.graph)
    
    # if not is_complete_graph:
    #     # print(f"Episode {episode_id}: Graph is not complete, skipping.")
    #     return (False, False)
    
    # Graph is complete, now check for uniqueness
    canonical_string = generate_canonical_graph_string(chemistry)
    chemistry_hash = hashlib.sha256(canonical_string.encode('utf-8')).hexdigest()
    
    is_unique = chemistry_hash not in SEEN_CHEMISTRY_HASHES
    
    if is_unique:
        # print(f"Episode {episode_id}: Found new unique chemistry.")
        # SEEN_CHEMISTRY_HASHES.add(chemistry_hash)
        # return (True, True)
        return (is_complete, True)
    return (is_complete, False) 
    # else:
    #     # print(f"Episode {episode_id}: Chemistry is a duplicate, skipping.")
    #     return (True, False)

# --- Example Usage --- #
if __name__ == "__main__":
    
    argument_parser = argparse.ArgumentParser(description="Alchemy Prompt Generator")
    argument_parser.add_argument("--max_train_episodes", type=int, help="Maximum number of episodes to run", default=5)
    argument_parser.add_argument("--max_trials_per_episode", type=int, help="Maximum number of trials per episode", default=5)
    argument_parser.add_argument("--max_steps_per_trial", type=int, help="Maximum number of steps per trial", default=20)
    argument_parser.add_argument("--eval_episodes", type=int, help="Number of evaluation episodes", default=10)
    
    args = argument_parser.parse_args()
    MAX_EPISODES = args.max_train_episodes
    MAX_TRIALS_PER_EPISODE = args.max_trials_per_episode
    MAX_STEPS_PER_TRIAL = args.max_steps_per_trial
    EVAL_EPISODES = args.eval_episodes

    print("Setting up symbolic environment...")
    env = symbolic_alchemy.get_symbolic_alchemy_level(
        level_name="alchemy/perceptual_mapping_randomized_with_random_bottleneck",
        num_trials=MAX_TRIALS_PER_EPISODE,
        max_steps_per_trial=MAX_STEPS_PER_TRIAL,
        num_stones_per_trial=NUM_STONES,  # Synchronize with prompt generator
        seed=42 # Use a fixed seed
    )
    print("Environment created.")

    episode_num = 0
    train_unique_chemistries_found = 0
    target_unique_chemistries = MAX_EPISODES  # We want to find 5 unique chemistries
    
    graph_data = {}
    pbar = tqdm(total=target_unique_chemistries, desc="Finding unique chemistries", unit="chemistry") 
    while train_unique_chemistries_found < target_unique_chemistries:
        # print(f"\n===== Starting Episode {episode_num} =====")
        time_step, chemistry = env.reset(return_chemistry=True)
        
        # Check if chemistry is complete and unique
        # is_complete_graph, is_newly_seen_unique_chemistry = check_chemistry_uniqueness_and_completeness(chemistry, episode_num)
        is_complete, is_unique_chemistry = check_chemistry_uniqueness_and_completeness(chemistry, episode_num)
        
        if not is_unique_chemistry:
            # print(f"Skipping duplicate chemistry in episode {episode_num}")
            episode_num += 1
            continue
        
        # if not is_complete_graph:
        #     # print(f"Skipping incomplete graph in episode {episode_num}")
        #     episode_num += 1
        #     # continue
            
        # if not is_newly_seen_unique_chemistry:
        #     # print(f"Skipping duplicate chemistry in episode {episode_num}")
        #     episode_num += 1
        #     continue
            
        # We have a complete and unique chemistry, increment our counter
        train_unique_chemistries_found += 1
        # print(f"Found {train_unique_chemistries_found} unique chemistries so far")

        # Extract graph data for this unique chemistry
        current_chemistry_graph = {}
        try:
            nodes = chemistry.graph.node_list.nodes
            edges = chemistry.graph.edge_list.edges
            # Import necessary types locally if not already imported at top level
            from dm_alchemy.types.stones_and_potions import LatentStone, AlignedStone, PerceivedStone, unalign

            for current_node in nodes:
                current_node_str = str(current_node)

                # --- Added: Get current stone description ---
                try:
                    # 1. Create LatentStone from node coords
                    # Ensure coords are in the expected format if necessary
                    latent_stone = LatentStone(current_node.coords)

                    # 2. Convert LatentStone -> AlignedStone using stone_map
                    aligned_stone = chemistry.stone_map.apply_inverse(latent_stone)
                    # We need the reward for the PerceivedStone, use the node's reward
                    aligned_stone_with_reward = AlignedStone(aligned_stone.reward, aligned_stone.aligned_coords)

                    # 3. Convert AlignedStone -> PerceivedStone using rotation
                    perceived_stone = unalign(aligned_stone_with_reward, chemistry.rotation)

                    # 4. Generate description string
                    color = get_stone_feature_name(0, perceived_stone.perceived_coords[0])
                    size = get_stone_feature_name(1, perceived_stone.perceived_coords[1])
                    roundness = get_stone_feature_name(2, perceived_stone.perceived_coords[2])
                    current_stone_desc = f"{{color: {color}, size: {size}, roundness: {roundness}}}"
                except Exception as desc_e:
                    print(f"Warning: Could not generate description for node {current_node_str}: {desc_e}")
                    current_stone_desc = "Error generating description"
                # --- End Added: Get current stone description ---

                # Initialize entry for the current node with description and transitions list
                current_chemistry_graph[current_node_str] = {
                    "current_stone_description": current_stone_desc,
                    "transitions": []
                }

                if current_node in edges:
                    for next_node, edge_info in edges[current_node].items():
                        # Assuming edge_info is a tuple like (1, Potion(...))
                        if len(edge_info) >= 2 and isinstance(edge_info[1], stones_and_potions.Potion):
                            potion = edge_info[1]
                            potion_idx = potion.idx
                            potion_color = potion_index_to_color_name(potion_idx)
                            next_node_str = str(next_node)
                            # Append transition to the 'transitions' list
                            current_chemistry_graph[current_node_str]["transitions"].append({
                                "potion_index": potion_idx,
                                "potion_color": potion_color,
                                "next_node_str": next_node_str,
                                "potion_details": str(potion) # Optional: include full potion details
                            })
                        else:
                            print(f"Warning: Unexpected edge info format for edge from {current_node_str} to {str(next_node)}: {edge_info}")
                
            # Store the current episode's graph data with metadata about uniqueness
            graph_data[str(episode_num)] = {
                "graph": current_chemistry_graph,
                "is_complete": is_complete,
                # "is_unique": is_newly_seen_unique_chemistry
            }

        except Exception as e:
            print(f"Error extracting or saving chemistry graph: {e}")
        # Save the updated graph data to the JSON file
        output_filename = "train_chemistry_graph.json"
        with open(output_filename, 'w') as f:
            json.dump(graph_data, f, indent=4)
        # print(f"Chemistry graph data saved to {output_filename}")

        # print(f"===== Episode {episode_num} Ended ====")
        episode_num += 1
        pbar.update(1)  # Update progress bar

    print(f"\nFound {train_unique_chemistries_found} unique chemistries out of {episode_num} episodes processed")
    env.close()
    print("\Train Environment closed.")
    
    
    # # Create evaluation episodes.
    # eval_chems = chemistries_proto_conversion.load_chemistries_and_items(
    #     'chemistries/perceptual_mapping_randomized_with_random_bottleneck/chemistries'
    #     )
    
    # eval_graph_data = {}
    # val_unique_chemistries_found = 0
    # print(f"Loaded {len(eval_chems)} chemistries for evaluation.")
    # val_pbar = tqdm(total=len(eval_chems), desc="Evaluation chemistries", unit="chemistry")
    # for chem, items in eval_chems:
    #     env, chemistry = symbolic_alchemy.get_symbolic_alchemy_fixed(chemistry=chem, episode_items=items, return_chemistry=True)
        
    #     # Follow the same process as above to check for unique chemistries
    #     # and then create prompts and then store them in eval_chemistries_graph.json.
    #     # This part is similar to the training loop above, but for evaluation.
        
    #     is_complete_graph, is_newly_seen_unique_chemistry = check_chemistry_uniqueness_and_completeness(chemistry, episode_num)
    #     if not is_complete_graph:
    #         # print(f"Skipping incomplete graph in episode {episode_num}")
    #         episode_num += 1
    #         # continue
    #     if not is_newly_seen_unique_chemistry:
    #         # print(f"Skipping duplicate chemistry in episode {episode_num}")
    #         episode_num += 1
    #         continue
    #     val_unique_chemistries_found += 1
    #     # print(f"Found {unique_chemistries_found} unique chemistries so far")
    #     # Extract graph data for this unique chemistry
    #     current_chemistry_graph = {}
    #     try:
    #         nodes = chemistry.graph.node_list.nodes
    #         edges = chemistry.graph.edge_list.edges
    #         # Import necessary types locally if not already imported at top level
    #         from dm_alchemy.types.stones_and_potions import LatentStone, AlignedStone, PerceivedStone, unalign

    #         for current_node in nodes:
    #             current_node_str = str(current_node)

    #             # --- Added: Get current stone description ---
    #             try:
    #                 # 1. Create LatentStone from node coords
    #                 # Ensure coords are in the expected format if necessary
    #                 latent_stone = LatentStone(current_node.coords)

    #                 # 2. Convert LatentStone -> AlignedStone using stone_map
    #                 aligned_stone = chemistry.stone_map.apply_inverse(latent_stone)
    #                 # We need the reward for the PerceivedStone, use the node's reward
    #                 aligned_stone_with_reward = AlignedStone(aligned_stone.reward, aligned_stone.aligned_coords)

    #                 # 3. Convert AlignedStone -> PerceivedStone using rotation
    #                 perceived_stone = unalign(aligned_stone_with_reward, chemistry.rotation)

    #                 # 4. Generate description string
    #                 color = get_stone_feature_name(0, perceived_stone.perceived_coords[0])
    #                 size = get_stone_feature_name(1, perceived_stone.perceived_coords[1])
    #                 roundness = get_stone_feature_name(2, perceived_stone.perceived_coords[2])
    #                 current_stone_desc = f"{{color: {color}, size: {size}, roundness: {roundness}}}"
    #             except Exception as desc_e:
    #                 print(f"Warning: Could not generate description for node {current_node_str}: {desc_e}")
    #                 current_stone_desc = "Error generating description"
    #             # --- End Added: Get current stone description ---

    #             # Initialize entry for the current node with description and transitions list
    #             current_chemistry_graph[current_node_str] = {
    #                 "current_stone_description": current_stone_desc,
    #                 "transitions": []
    #             }

    #             if current_node in h:
    #                 for next_node, edge_info in edges[current_node].items():
    #                     # Assuming edge_info is a tuple like (1, Potion(...))
    #                     if len(edge_info) >= 2 and isinstance(edge_info[1], stones_and_potions.Potion):
    #                         potion = edge_info[1]
    #                         potion_idx = potion.idx
    #                         potion_color = potion_index_to_color_name(potion_idx)
    #                         next_node_str = str(next_node)
    #                         # Append transition to the 'transitions' list
    #                         current_chemistry_graph[current_node_str]["transitions"].append({
    #                             "potion_index": potion_idx,
    #                             "potion_color": potion_color,
    #                             "next_node_str": next_node_str,
    #                             "potion_details": str(potion) # Optional: include full potion details
    #                         })
    #                     else:
    #                         print(f"Warning: Unexpected edge info format for edge from {current_node_str} to {str(next_node)}: {edge_info}")
    #         # Store the current episode's graph data with metadata about uniqueness
    #         eval_graph_data[str(episode_num)] = {
    #             "graph": current_chemistry_graph,
    #             "is_complete": is_complete_graph,
    #             "is_unique": is_newly_seen_unique_chemistry
    #         }
    #     except Exception as e:
    #         print(f"Error extracting or saving chemistry graph: {e}")
        
    #     episode_num += 1
    #     val_pbar.update(1)
    # # Save the updated graph data to the JSON file
    # output_filename = "eval_chemistry_graph.json"
    # with open(output_filename, 'w') as f:
    #     json.dump(eval_graph_data, f, indent=4)
    # print(f"Chemistry graph data saved to {output_filename}")

    # print(f"\nFound {val_unique_chemistries_found} unique and complete chemistries out of {episode_num} validation episodes processed")
    # env.close()
    # print("\nEvaluation Environment closed.")
    print("All done!")