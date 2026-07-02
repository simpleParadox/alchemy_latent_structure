import pickle

orig_vocab_path = "/home/rsaha/projects/def-afyshe-ab/rsaha/dm_alchemy/src/data/shuffled_held_out_exps_preprocessed_separate_enhanced/compositional_chemistry_samples_167424_80_unique_stones_val_shop_1_qhop_1_single_held_out_color_4_edges_exp_seed_0_classification_filter_True_input_features_output_stone_states_vocab.pkl"
rand_vocab_path = "/home/rsaha/projects/def-afyshe-ab/rsaha/dm_alchemy/src/data/held_out_randomized_reward_preprocessed_separate_enhanced/compositional_chemistry_samples_167424_80_unique_stones_val_shop_1_qhop_1_single_held_out_color_4_edges_exp_seed_0_classification_filter_True_input_features_output_stone_states_vocab.pkl"

with open(orig_vocab_path, "rb") as f:
    orig_vocab = pickle.load(f)
with open(rand_vocab_path, "rb") as f:
    rand_vocab = pickle.load(f)

orig_in = orig_vocab.get("input_word2idx", {})
rand_in = rand_vocab.get("input_word2idx", {})

print(f"Original input_word2idx size: {len(orig_in)}")
print(f"Randomized input_word2idx size: {len(rand_in)}")

intersection = set(orig_in.keys()).intersection(set(rand_in.keys()))
mismatched = 0
for k in intersection:
    if orig_in[k] != rand_in[k]:
        mismatched += 1

print(f"Mismatched input_word2idx: {mismatched} out of {len(intersection)}")
