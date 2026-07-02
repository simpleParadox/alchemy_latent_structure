import json

orig_json_path = '/home/rsaha/projects/def-afyshe-ab/rsaha/dm_alchemy/src/data/shuffled_held_out_exps_generated_data_enhanced/compositional_chemistry_samples_167424_80_unique_stones_train_shop_1_qhop_1_single_held_out_color_4_edges_exp_seed_0.json'
d_orig = json.load(open(orig_json_path))

new_json_path = '/home/rsaha/projects/def-afyshe-ab/rsaha/dm_alchemy/src/data/chemistry_pickles/original_reward_potion_remap_generated_data/compositional_chemistry_samples_167424_80_unique_stones_train_shop_1_qhop_1_single_held_out_color_4_edges_exp_seed_0_pairing_index_0.json'
d_new = json.load(open(new_json_path))

ep_orig = d_orig['episodes']
ep_new = d_new['episodes']

# Remove metadata
if '_metadata' in ep_orig: del ep_orig['_metadata']
if '_metadata' in ep_new: del ep_new['_metadata']

if ep_orig == ep_new:
    print('The JSON dictionaries are EXACTLY IDENTICAL in content!')
else:
    print('The JSON dictionaries are DIFFERENT!')
    # Let's find one difference
    for k in ep_orig.keys():
        if k not in ep_new:
            print(f'Key {k} in orig but not new')
            break
        if ep_orig[k] != ep_new[k]:
            print(f'Mismatch at key {k}:')
            print('Orig:', ep_orig[k])
            print('New:', ep_new[k])
            break
    for k in ep_new.keys():
        if k not in ep_orig:
            print(f'Key {k} in new but not orig')
            break

