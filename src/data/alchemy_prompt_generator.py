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

# --- State Extraction --- #

def extract_state_info(symbolic_obs_vector: np.ndarray) -> Dict[str, Any]:
    """Extracts structured information from the symbolic observation vector."""
    if not isinstance(symbolic_obs_vector, np.ndarray) or symbolic_obs_vector.shape != (POTION_FEATURES_END,):
        raise ValueError(f"Invalid symbolic_obs_vector shape. Expected ({POTION_FEATURES_END},), got {symbolic_obs_vector.shape}")

    state_info = {
        "stones": [],
        "available_potions": [],
        "available_potion_indices": [],
        "target_stone_index": -1, # Index of the first available stone
        "target_stone_desc": "No available stones.",
        "target_stone_value": "N/A"
    }

    # Extract Stone Info
    stone_features_flat = symbolic_obs_vector[:STONE_FEATURES_END]
    for i in range(NUM_STONES):
        start_idx = i * FEATURES_PER_STONE
        end_idx = start_idx + FEATURES_PER_STONE
        stone_vec = stone_features_flat[start_idx:end_idx]
        is_used_in_cauldron = np.isclose(stone_vec[4], 1.0)
        color = get_stone_feature_name(0, stone_vec[0])
        size = get_stone_feature_name(1, stone_vec[1])
        roundness = get_stone_feature_name(2, stone_vec[2])
        value = get_reward_value(stone_vec[3])
        stone_desc = f"{{color: {color}, size: {size}, roundness: {roundness}}}"
        state_info["stones"].append({
            "index": i,
            "description": stone_desc,
            "value": value,
            "is_used": is_used_in_cauldron
        })
        if not is_used_in_cauldron and state_info["target_stone_index"] == -1:
            state_info["target_stone_index"] = i
            state_info["target_stone_desc"] = f"Stone {i} state: {stone_desc}"
            state_info["target_stone_value"] = value

    # Extract Potion Info
    potions_features_flat = symbolic_obs_vector[STONE_FEATURES_END:POTION_FEATURES_END]
    for i in range(NUM_POTIONS):
        start_idx = i * FEATURES_PER_POTION
        end_idx = start_idx + FEATURES_PER_POTION
        potion_vec = potions_features_flat[start_idx:end_idx]
        is_used = np.isclose(potion_vec[1], 1.0)
        if not is_used:
            color_name = get_potion_color_name(potion_vec[0])
            if color_name != "USED/UNKNOWN":
                state_info["available_potions"].append(color_name)
                state_info["available_potion_indices"].append(i)

    state_info["available_potions_str"] = f"[{', '.join(state_info['available_potions'])}]" if state_info["available_potions"] else "[]"

    # Define Available Actions
    actions = []
    if state_info["target_stone_index"] != -1:
        stone_idx = state_info["target_stone_index"]
        if state_info["available_potions"]:
            # List specific potion actions
            for p_idx, p_color in zip(state_info["available_potion_indices"], state_info["available_potions"]):
                 actions.append(f"dip stone {stone_idx} in potion {p_idx} ({p_color})")
        actions.append(f"dip stone {stone_idx} in cauldron")
    if not actions:
        actions.append("end trial (no valid actions)")

    state_info["available_actions_str"] = ", ".join(actions)

    return state_info

# --- Action/Outcome Description --- #

def int_action_to_description(action: int, prev_obs_vector: np.ndarray) -> Tuple[str, Optional[int], Optional[int]]:
    """Converts an integer action to a human-readable description based on the state *before* the action."""
    if action == 0: # No-op
        return "You did nothing.", None, None

    prev_state = extract_state_info(prev_obs_vector)
    target_stone_idx = prev_state["target_stone_index"]

    if target_stone_idx == -1:
        return "(Invalid action: No available stones)", None, None

    # Calculate stone index and potion/cauldron index from action
    # Action 1 = stone 0 cauldron, Action 2 = stone 0 potion 0, ..., Action 14 = stone 1 cauldron, etc.
    action_offset = action - 1
    stone_idx = action_offset // POTIONS_PLUS_CAULDRON
    potion_or_cauldron_idx = action_offset % POTIONS_PLUS_CAULDRON

    if stone_idx != target_stone_idx:
        return f"(Invalid action: Tried to use stone {stone_idx}, but target stone is {target_stone_idx})", None, None

    stone_desc = prev_state["stones"][stone_idx]["description"]

    if potion_or_cauldron_idx == 0: # Cauldron
        return f"You dipped stone {stone_idx} {stone_desc} in the cauldron.", stone_idx, -1 # -1 for cauldron
    else: # Potion
        potion_slot_idx = potion_or_cauldron_idx - 1
        if potion_slot_idx not in prev_state["available_potion_indices"]:
             return f"(Invalid action: Tried to use potion slot {potion_slot_idx}, which is unavailable or used)", None, None

        potion_color = get_potion_color_name(prev_obs_vector[STONE_FEATURES_END + potion_slot_idx * FEATURES_PER_POTION])
        return f"You dipped stone {stone_idx} {stone_desc} in potion {potion_slot_idx} ({potion_color}).", stone_idx, potion_slot_idx

def get_outcome_description(prev_obs_vector: np.ndarray, current_obs_vector: np.ndarray,
                          reward: Optional[float], action_stone_idx: Optional[int], action_potion_idx: Optional[int]) -> str:
    """Describes the outcome of the last action."""
    if action_stone_idx is None: # Invalid action or no-op
        return "No change in state."

    prev_state = extract_state_info(prev_obs_vector)
    current_state = extract_state_info(current_obs_vector)

    if action_potion_idx == -1: # Cauldron action
        if reward is not None and reward > 0:
            return f"You received a reward of {reward:.0f} from the cauldron."
        elif reward is not None:
             return f"You received a reward of {reward:.0f} from the cauldron. The stone is now removed."
        else:
             return "The stone was placed in the cauldron and is now removed."
    else: # Potion action
        prev_stone = prev_state["stones"][action_stone_idx]
        current_stone = current_state["stones"][action_stone_idx]

        if current_stone["is_used"]:
             # This case shouldn't happen if action validation is correct, but good to handle.
             return "(Error: Stone seems to have been used in cauldron unexpectedly)"

        if prev_stone["description"] == current_stone["description"]:
            return "The potion had no effect on the stone."
        else:
            return f"The stone is now {current_stone['description']}." 
        # It has a value of {current_obs_vector[action_stone_idx, 3]}."

# --- Prompt Templates --- #




# *   Potion effects are *consistent within an episode* but *change between episodes*.


INITIAL_PROMPT_TEMPLATE = """
**Decision Making Task:**

**Goal:** Maximize the value of stones by dipping them in potions before placing them in the cauldron.

**Game Rules:**
*   You start with stones having features: Color (blue, purple, red), Size (small, medium, large), Roundness (pointy, medium_round, round).
*   Each stone has an inherent value (-3, -1, 1, 15).
*   There are 12 potions with colors: RED, PINK, GREEN, ORANGE, YELLOW, CYAN.
*   Dipping a stone consumes the potion and might change *one* feature of the stone.
*   Placing a stone in the cauldron gives you its current value and removes the stone.
*   A trial ends when you run out if you exhaust 20 steps, or choose to end it.

**Your Task:** Choose actions strategically to get the highest total reward in each trial.

--- Beginning Episode {episode_num} ---

**Trial {trial_num}, Step {step_num}:**

{stone_info}
Available potions: {available_potions_str}
Available actions: {available_actions_str}
Stone value: {stone_value}

Your action is:
"""

TRIAL_START_PROMPT_TEMPLATE = """
--- Beginning Trial {trial_num}, Step {step_num} ---

{stone_info}
Stone value: {stone_value}
Available potions: {available_potions_str}
Available actions: {available_actions_str}

Your action is:
"""

STEP_FEEDBACK_PROMPT_TEMPLATE = """
**Outcome:** {outcome_desc}
Stone value: {stone_value}
--- Trial {trial_num}, Step {step_num} ---

{stone_info}
Available potions: {available_potions_str}
Available actions: {available_actions_str}

Your action is:
"""

# --- Prompt Generation Function --- #

def generate_prompt_or_feedback(
    current_obs_vector: np.ndarray,
    episode_num: int,
    trial_num: int,
    step_num: int,
    is_initial_episode_step: bool = False,
    is_initial_trial_step: bool = False,
    previous_action_desc: Optional[str] = None,
    previous_reward: Optional[float] = None,
    previous_obs_vector: Optional[np.ndarray] = None,
    action_stone_idx: Optional[int] = None, # Stone index used in the *previous* action
    action_potion_idx: Optional[int] = None # Potion index (-1 for cauldron) used in the *previous* action
) -> str:
    """Generates the appropriate textual prompt or feedback for the current state."""

    current_state_info = extract_state_info(current_obs_vector)

    if is_initial_episode_step:
        template = INITIAL_PROMPT_TEMPLATE
        format_args = {
            "episode_num": episode_num,
            "trial_num": trial_num,
            "step_num": step_num,
            "stone_info": current_state_info["target_stone_desc"],
            "available_potions_str": current_state_info["available_potions_str"],
            "available_actions_str": current_state_info["available_actions_str"],
            "stone_value": current_state_info["target_stone_value"]
        }
    elif is_initial_trial_step:
        template = TRIAL_START_PROMPT_TEMPLATE
        format_args = {
            "trial_num": trial_num,
            "step_num": step_num,
            "stone_info": current_state_info["target_stone_desc"],
            "available_potions_str": current_state_info["available_potions_str"],
            "available_actions_str": current_state_info["available_actions_str"],
            "stone_value": current_state_info["target_stone_value"]
        }
    else: # Mid-trial step, provide feedback
        template = STEP_FEEDBACK_PROMPT_TEMPLATE
        if previous_obs_vector is None or previous_action_desc is None:
             raise ValueError("Missing previous state info for feedback prompt.")

        outcome_desc = get_outcome_description(
            previous_obs_vector, current_obs_vector, previous_reward, action_stone_idx, action_potion_idx)

        format_args = {
            "previous_action_desc": previous_action_desc,
            "outcome_desc": outcome_desc,
            "trial_num": trial_num,
            "step_num": step_num,
            "stone_info": current_state_info["target_stone_desc"],
            "available_potions_str": current_state_info["available_potions_str"],
            "available_actions_str": current_state_info["available_actions_str"],
            "stone_value": current_state_info["target_stone_value"]
        }

    return template.format(**format_args)

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
        parts.append(f"NODE:{node_str}")
        
        # Get the node's perceived stone description
        try:
            latent_stone = stones_and_potions.LatentStone(node.coords)
            aligned_stone = chemistry.stone_map.apply_inverse(latent_stone)
            aligned_stone_with_reward = stones_and_potions.AlignedStone(aligned_stone.reward, aligned_stone.aligned_coords)
            perceived_stone = stones_and_potions.unalign(aligned_stone_with_reward, chemistry.rotation)
            
            # Add perceived stone coords and reward
            coord_str = ":".join([str(c) for c in perceived_stone.perceived_coords])
            parts.append(f"STONE:{perceived_stone.reward}:{coord_str}")
        except Exception:
            # If we can't get the stone description, still include something unique for this node
            print(f"Warning: Could not generate description for node {node_str}")
            parts.append(f"STONE:error")
        
        # Sort transitions for consistent ordering
        if node in chemistry.graph.edge_list.edges:
            edges = chemistry.graph.edge_list.edges[node]
            sorted_edges = sorted(edges.items(), key=lambda x: str(x[0]))
            
            for next_node, edge_info in sorted_edges:
                if len(edge_info) >= 2 and isinstance(edge_info[1], stones_and_potions.Potion):
                    potion = edge_info[1]
                    parts.append(f"EDGE:{str(next_node)}:{potion.idx}:{potion.dimension}:{potion.direction}")
    
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
    # First check if the graph is complete
    is_complete_graph = is_graph_complete(chemistry.graph)
    
    if not is_complete_graph:
        # print(f"Episode {episode_id}: Graph is not complete, skipping.")
        return (False, False)
    
    # Graph is complete, now check for uniqueness
    canonical_string = generate_canonical_graph_string(chemistry)
    chemistry_hash = hashlib.sha256(canonical_string.encode('utf-8')).hexdigest()
    
    is_unique = chemistry_hash not in SEEN_CHEMISTRY_HASHES
    
    if is_unique:
        # print(f"Episode {episode_id}: Found new unique chemistry.")
        SEEN_CHEMISTRY_HASHES.add(chemistry_hash)
        return (True, True)
    else:
        # print(f"Episode {episode_id}: Chemistry is a duplicate, skipping.")
        return (True, False)

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
        is_complete_graph, is_newly_seen_unique_chemistry = check_chemistry_uniqueness_and_completeness(chemistry, episode_num)
        
        if not is_complete_graph:
            # print(f"Skipping incomplete graph in episode {episode_num}")
            episode_num += 1
            # continue
            
        if not is_newly_seen_unique_chemistry:
            # print(f"Skipping duplicate chemistry in episode {episode_num}")
            episode_num += 1
            continue
            
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
                "is_complete": is_complete_graph,
                "is_unique": is_newly_seen_unique_chemistry
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
    
    
    # Create evaluation episodes.
    eval_chems = chemistries_proto_conversion.load_chemistries_and_items(
        'chemistries/perceptual_mapping_randomized_with_random_bottleneck/chemistries'
        )
    
    eval_graph_data = {}
    val_unique_chemistries_found = 0
    print(f"Loaded {len(eval_chems)} chemistries for evaluation.")
    val_pbar = tqdm(total=len(eval_chems), desc="Evaluation chemistries", unit="chemistry")
    for chem, items in eval_chems:
        env, chemistry = symbolic_alchemy.get_symbolic_alchemy_fixed(chemistry=chem, episode_items=items, return_chemistry=True)
        
        # Follow the same process as above to check for unique chemistries
        # and then create prompts and then store them in eval_chemistries_graph.json.
        # This part is similar to the training loop above, but for evaluation.
        
        is_complete_graph, is_newly_seen_unique_chemistry = check_chemistry_uniqueness_and_completeness(chemistry, episode_num)
        if not is_complete_graph:
            # print(f"Skipping incomplete graph in episode {episode_num}")
            episode_num += 1
            # continue
        if not is_newly_seen_unique_chemistry:
            # print(f"Skipping duplicate chemistry in episode {episode_num}")
            episode_num += 1
            continue
        val_unique_chemistries_found += 1
        # print(f"Found {unique_chemistries_found} unique chemistries so far")
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

                if current_node in h:
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
            eval_graph_data[str(episode_num)] = {
                "graph": current_chemistry_graph,
                "is_complete": is_complete_graph,
                "is_unique": is_newly_seen_unique_chemistry
            }
        except Exception as e:
            print(f"Error extracting or saving chemistry graph: {e}")
        
        episode_num += 1
        val_pbar.update(1)
    # Save the updated graph data to the JSON file
    output_filename = "eval_chemistry_graph.json"
    with open(output_filename, 'w') as f:
        json.dump(eval_graph_data, f, indent=4)
    print(f"Chemistry graph data saved to {output_filename}")

    print(f"\nFound {val_unique_chemistries_found} unique and complete chemistries out of {episode_num} validation episodes processed")
    env.close()
    print("\nEvaluation Environment closed.")
    print("All done!")