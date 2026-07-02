import pickle
import json
import os
import sys

sys.path.append(os.path.abspath('/home/rsaha/projects/def-afyshe-ab/rsaha/dm_alchemy'))

from src.models.data_loaders import AlchemyDataset

orig_pkl_path = "/home/rsaha/projects/def-afyshe-ab/rsaha/dm_alchemy/src/data/shuffled_held_out_exps_preprocessed_separate_enhanced/compositional_chemistry_samples_167424_80_unique_stones_train_shop_1_qhop_1_single_held_out_color_4_edges_exp_seed_0_classification_filter_True_input_features_output_stone_states_data.pkl"
orig_vocab_path = orig_pkl_path.replace("_data.pkl", "_vocab.pkl")

with open(orig_vocab_path, "rb") as f:
    orig_vocab = pickle.load(f)

new_json = "/home/rsaha/projects/def-afyshe-ab/rsaha/dm_alchemy/src/data/chemistry_pickles/original_reward_potion_remap_generated_data/compositional_chemistry_samples_167424_80_unique_stones_train_shop_1_qhop_1_single_held_out_color_4_edges_exp_seed_0_pairing_index_0.json"

dataset = AlchemyDataset(
    json_file_path=new_json,
    task_type="classification",
    filter_query_from_support=True,
    use_preprocessed=False,
    input_format="features",
    output_format="stone_states"
)

dyn_vocab_in = dataset.input_word2idx
orig_vocab_in = orig_vocab['input_word2idx']

if dyn_vocab_in == orig_vocab_in:
    print("Input vocabs match perfectly!")
else:
    print("Input vocabs differ!")
    for k in dyn_vocab_in:
        if dyn_vocab_in[k] != orig_vocab_in.get(k):
            print(f"Key {k}: dynamic={dyn_vocab_in[k]}, orig={orig_vocab_in.get(k)}")

dyn_vocab_out = dataset.stone_state_to_id
orig_vocab_out = orig_vocab.get('stone_state_to_id', {})

if dyn_vocab_out == orig_vocab_out:
    print("Output stone states match perfectly!")
else:
    print("Output stone states differ!")
    for k in dyn_vocab_out:
        if dyn_vocab_out[k] != orig_vocab_out.get(k):
            print(f"Key {k}: dynamic={dyn_vocab_out[k]}, orig={orig_vocab_out.get(k)}")

