import pickle
import torch
import hashlib
import os

orig_pkl_path = '/home/rsaha/projects/def-afyshe-ab/rsaha/dm_alchemy/src/data/shuffled_held_out_exps_preprocessed_separate_enhanced/compositional_chemistry_samples_167424_80_unique_stones_train_shop_1_qhop_1_single_held_out_color_4_edges_exp_seed_0_classification_filter_True_input_features_output_stone_states_data.pkl'
with open(orig_pkl_path, 'rb') as f:
    orig_data = pickle.load(f)

new_pkl_path = '/tmp/preprocessed/compositional_chemistry_samples_167424_80_unique_stones_train_shop_1_qhop_1_single_held_out_color_4_edges_exp_seed_0_pairing_index_0_classification_filter_True_input_features_output_stone_states_data.pkl'
with open(new_pkl_path, 'rb') as f:
    new_data = pickle.load(f)

def hash_sample(s):
    t_id = s['target_class_id']
    enc = s['encoder_input_ids']
    if isinstance(enc, torch.Tensor):
        enc = tuple(enc.tolist())
    else:
        enc = tuple(enc)
    return hash((t_id, enc))

orig_hashes = {}
for s in orig_data:
    h = hash_sample(s)
    orig_hashes[h] = orig_hashes.get(h, 0) + 1

new_hashes = {}
for s in new_data:
    h = hash_sample(s)
    new_hashes[h] = new_hashes.get(h, 0) + 1

if orig_hashes == new_hashes:
    print('The sets of samples are EXACTLY IDENTICAL in content, just permuted!')
else:
    print('The sets of samples are DIFFERENT!')
    
    # find diffs
    orig_only = sum(v for k, v in orig_hashes.items() if k not in new_hashes)
    new_only = sum(v for k, v in new_hashes.items() if k not in orig_hashes)
    print(f'orig only: {orig_only}, new only: {new_only}')

