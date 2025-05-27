import unittest
import itertools

# Assuming the script is run from the root of the dm_alchemy project,
# or that PYTHONPATH is set up to find modules in src and dm_alchemy.
from src.data.deterministic_chemistry_generator import get_stone_feature_name
from dm_alchemy.types import stones_and_potions

class TestStoneStateRepresentation(unittest.TestCase):

    def test_unique_stone_state_string_capacity(self):
        """
        Tests that the stone description string format has the capacity
        to represent 108 unique stone states.

        The format is: "DESC:{{color:{c},size:{s},roundness:{r},reward:{rew}}}"
        - Color, size, roundness each have 3 possible string values derived
          from perceived coordinates (-1, 0, 1).
        - Reward has 4 possible numerical values.
        Total expected unique states = 3 * 3 * 3 * 4 = 108.
        """
        unique_stone_strings = set()

        # 1. Define possible string values for visual features
        # These are the outputs of get_stone_feature_name for values -1.0, 0.0, 1.0
        possible_perceived_coord_values = [-1.0, 0.0, 1.0]
        
        colors = sorted(list(set([get_stone_feature_name(0, v) for v in possible_perceived_coord_values])))
        sizes = sorted(list(set([get_stone_feature_name(1, v) for v in possible_perceived_coord_values])))
        roundnesses = sorted(list(set([get_stone_feature_name(2, v) for v in possible_perceived_coord_values])))

        # Ensure we actually got 3 unique values for each visual feature
        self.assertEqual(len(colors), 3, "Should have 3 unique color names")
        self.assertEqual(len(sizes), 3, "Should have 3 unique size names")
        self.assertEqual(len(roundnesses), 3, "Should have 3 unique roundness names")

        # 2. Define possible reward values
        # These are the raw reward values from POSS_REWARDS
        possible_reward_values = list(stones_and_potions.POSS_REWARDS)
        self.assertEqual(len(possible_reward_values), 4, "Should have 4 unique reward values")

        # 3. Generate all combinations and form the description strings
        for color_str in colors:
            for size_str in sizes:
                for roundness_str in roundnesses:
                    for reward_val in possible_reward_values:
                        stone_desc_str = f"DESC:{{color:{color_str},size:{size_str},roundness:{roundness_str},reward:{reward_val}}}"
                        unique_stone_strings.add(stone_desc_str)
        
        # 4. Assert the number of unique strings
        self.assertEqual(len(unique_stone_strings), 108, 
                         f"Expected 108 unique stone state strings, but got {len(unique_stone_strings)}")

if __name__ == '__main__':
    unittest.main()
