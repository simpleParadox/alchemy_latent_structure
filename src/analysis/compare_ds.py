import json
import pickle
import os
import sys

sys.path.append(os.path.abspath('/home/rsaha/projects/def-afyshe-ab/rsaha/dm_alchemy'))

from src.models.preprocess_dataset import preprocess_and_save_dataset

orig_pkl_path = "/home/rsaha/projects/def-afyshe-ab/rsaha/dm_alchemy/src/data/shuffled_held_out_exps_preprocessed_separate_enhanced/compositional_chemistry_samples_167424_80_unique_stones_train_shop_1_qhop_1_single_held_out_color_4_edges_exp_seed_0_classification_filter_True_input_features_output_stone_states_data.pkl"
with open(orig_pkl_path, "rb") as f:
    orig_data = pickle.load(f)
    
orig_vocab_path = orig_pkl_path.replace("_data.pkl", "_vocab.pkl")
with open(orig_vocab_path, "rb") as f:
    orig_vocab = pickle.load(f)

new_json = "/home/rsaha/projects/def-afyshe-ab/rsaha/dm_alchemy/src/data/chemistry_pickles/original_reward_potion_remap_generated_data/compositional_chemistry_samples_167424_80_unique_stones_train_shop_1_qhop_1_single_held_out_color_4_edges_exp_seed_0_pairing_index_0.json"
output_dir = "/tmp/preprocessed"
os.makedirs(output_dir, exist_ok=True)

# Generate new dataset using the exact same vocab
preprocess_and_save_dataset(
    json_file_path=new_json,
    task_type="classification",
    output_dir=output_dir,
    vocab_word2idx=orig_vocab['input_word2idx'],
    vocab_idx2word=orig_vocab['input_idx2word'],
    filter_query_from_support=True,
    input_format="features",
    output_format="stone_states"
)

import glob
new_data_file = glob.glob(output_dir + "/*_data.pkl")[0]
with open(new_data_file, "rb") as f:
    new_data = pickle.load(f)

print("Orig data length:", len(orig_data))
print("New data length:", len(new_data))

orig_sample = orig_data[0]
new_sample = new_data[0]

print("Orig keys:", orig_sample.keys())
print("New keys:", new_sample.keys())

for k in orig_sample.keys():
    if k in new_sample:
        import torch
        orig_t = orig_sample[k]
        new_t = new_sample[k]
        if isinstance(orig_t, torch.Tensor) and isinstance(new_t, torch.Tensor):
            if not torch.equal(orig_t, new_t):
                print(f"Mismatch in {k}!")
            else:
                print(f"Match for {k}!")
        elif orig_t != new_t:
            print(f"Mismatch in {k}!")
            print(orig_t, new_t)
        else:
            print(f"Match for {k}!")

# Compare entirely
exact_match = True
for i in range(min(len(orig_data), len(new_data))):
    for k in orig_data[i].keys():
        if k in new_data[i]:
            orig_t = orig_data[i][k]
            new_t = new_data[i][k]
            if isinstance(orig_t, torch.Tensor):
                if not torch.equal(orig_t, new_t):
                    exact_match = False
                    print(f"Diff at idx {i}, key {k}")
                    break
            elif orig_t != new_t:
                exact_match = False
                print(f"Diff at idx {i}, key {k}: {orig_t} != {new_t}")
                break
    if not exact_match:
        break

if exact_match:
    print("ALL MATCH PERFECTLY!")

