"""Minimal pytest example for the symbolic Alchemy environment.

This test resets the environment, inspects the flat symbolic observation vector,
and performs a simple step.
"""
# import pytest
import numpy as np
import dm_env
from dm_alchemy.symbolic_alchemy import get_symbolic_alchemy_fixed
from dm_alchemy import symbolic_alchemy
from dm_alchemy.types import stones_and_potions, utils as type_utils
from dm_alchemy import event_tracker

# Constants based on the described feature structure
NUM_STONES = 3
FEATURES_PER_STONE = 5
NUM_POTIONS = 12 # symbolic_alchemy.MAX_POTIONS
FEATURES_PER_POTION = 2
STONE_FEATURES_END = NUM_STONES * FEATURES_PER_STONE # Index 15
POTION_FEATURES_END = STONE_FEATURES_END + NUM_POTIONS * FEATURES_PER_POTION # Index 39

def get_potion_color_name(color_value: float) -> str:
    """Maps the numerical color value from the observation to its name."""
    # Use np.isclose for robust floating-point comparison
    if np.isclose(color_value, -1.0):
        return "red"
    elif np.isclose(color_value, -2/3):
        return "green"
    elif np.isclose(color_value, -1/3):
        return "orange"
    elif np.isclose(color_value, 0.0):
        return "yellow"
    elif np.isclose(color_value, 1/3):
        return "pink"
    elif np.isclose(color_value, 2/3):
        return "turquoise"
    else:
        # This might indicate a used potion or an unexpected value
        return "unknown/used"

def test_symbolic_alchemy_minimal():
    # Build the environment
    env = symbolic_alchemy.get_symbolic_alchemy_level(level_name="alchemy/perceptual_mapping_randomized_with_random_bottleneck")

    # 1) RESET → FIRST TimeStep
    ts = env.reset()
    assert isinstance(ts, dm_env.TimeStep)
    assert ts.first(), "Expected FIRST timestep on reset"

    # Unpack the observation dict
    obs_dict = ts.observation
    assert 'symbolic_obs' in obs_dict, "Expected 'symbolic_obs' key in observation"
    symbolic_obs_vector = obs_dict['symbolic_obs']
    assert isinstance(symbolic_obs_vector, np.ndarray), "symbolic_obs should be a numpy array"
    assert symbolic_obs_vector.ndim == 1, "symbolic_obs should be a 1D array"
    assert len(symbolic_obs_vector) == POTION_FEATURES_END, f"Expected {POTION_FEATURES_END} features in symbolic_obs"

    # Extract stone and potion features by slicing
    stone_features_flat = symbolic_obs_vector[:STONE_FEATURES_END]
    potions_features_flat = symbolic_obs_vector[STONE_FEATURES_END:POTION_FEATURES_END]

    # Optional: Print stone features (reshape for clarity)
    print("\n--- Stone Features (reshaped) ---")
    print(stone_features_flat.reshape((NUM_STONES, FEATURES_PER_STONE)))
    print("---------------------------------\n")


    # Print potion inventory details from the sliced array
    print("\n--- Potion Inventory Details ---")
    assert len(potions_features_flat) == NUM_POTIONS * FEATURES_PER_POTION
    for i in range(NUM_POTIONS):
        # Calculate indices within the potions_features_flat slice
        start_idx = i * FEATURES_PER_POTION
        end_idx = start_idx + FEATURES_PER_POTION
        potion_features = potions_features_flat[start_idx:end_idx]

        color_value = potion_features[0]
        used_value = potion_features[1]

        color_name = get_potion_color_name(color_value)
        is_used = np.isclose(used_value, 1.0)

        print(f"Potion Slot {i}: Raw Value={color_value:.3f}, Color='{color_name}', Used={is_used}")
    print("------------------------------\n")

    # 2) STEP → choose a valid integer action
    spec = env.action_spec()
    # Example: Use first stone (index 0) and first potion slot (index 0)
    example_slot_action = type_utils.SlotBasedAction(stone_ind=0, potion_ind=0)
    action_int = env._slot_based_action_to_int(example_slot_action)

    print(f"Taking integer action: {action_int}")
    ts2 = env.step(action_int)

    # 3) RETRIEVE COLOR MAPPING from the tracker (remains the same)
    outcome_tracker: event_tracker.LatestOutcomeTracker = env.game_state.trackers['latest_outcome']
    tracked_action: type_utils.TypeBasedAction = outcome_tracker.type_based_action

    if tracked_action and tracked_action.using_potion:
        perceived_potion: stones_and_potions.PerceivedPotion = tracked_action.perceived_potion
        potion_color_index: stones_and_potions.PerceivedPotionIndex = perceived_potion.index()
        potion_color_name_from_enum: str = perceived_potion.name
        print(f"Step Result: Potion used had perceived color index: {potion_color_index} ({potion_color_name_from_enum})")
    elif tracked_action and tracked_action.cauldron:
         print("Step Result: Action used the cauldron.")
    elif tracked_action and tracked_action.no_op:
         print("Step Result: Action was a no-op.")
    elif tracked_action and tracked_action.end_trial:
         print("Step Result: Action was end_trial.")
    else:
        # This might happen if the action was invalid (e.g., using non-existent stone/potion)
        print("Step Result: No potion/cauldron used, or action was invalid.")

    # Validate the returned timestep
    assert isinstance(ts2, dm_env.TimeStep)
    assert isinstance(ts2.reward, (float, int)), "Reward should be numeric"
    assert 0.0 <= ts2.discount <= 1.0, "Discount must be in [0,1]"

# Call the test function if this script is run directly
if __name__ == "__main__":
    test_symbolic_alchemy_minimal()
    print("\nTest passed!")
