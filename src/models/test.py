# Testing if there's overlap between the stone states for the two vocabs after the reward remapping.

import pickle
from tqdm import tqdm

v1 = pickle.load(open(
        '/home/rsaha/projects/def-afyshe-ab/rsaha/dm_alchemy/src/data/shuffled_held_out_exps_preprocessed_separate_enhanced/compositional_chemistry_samples_167424_80_unique_stones_train_shop_1_qhop_1_single_held_out_color_4_edges_exp_seed_0_classification_filter_True_input_features_output_stone_states_vocab.pkl', 'rb'
        ))

v2 = pickle.load(open(
    '/home/rsaha/projects/def-afyshe-ab/rsaha/dm_alchemy/src/data/continual_data/held_out_endpoint_reward_swap_preprocessed/compositional_chemistry_samples_167424_80_unique_stones_train_shop_1_qhop_1_single_held_out_color_4_edges_exp_seed_0_classification_filter_True_input_features_output_stone_states_vocab.pkl', 'rb'
    ))


for stone_id in tqdm(v2['stone_state_to_id'].keys()):
    # print(stone_id)
    if stone_id not in v1['stone_state_to_id']:
        print(f"Stone ID: {stone_id} is not in the keys of v1")
    try:
        target_v1 = v1['stone_state_to_id'][stone_id]
        target_v2 = v2['stone_state_to_id'][stone_id]
        if target_v1 != target_v2:
            print(f"Target IDs are not equal: {target_v1} != {target_v2}")
    except:
        print(stone_id.tolist())
    