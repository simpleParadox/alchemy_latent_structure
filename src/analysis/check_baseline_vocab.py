import pickle

baseline_vocab = "/home/rsaha/projects/def-afyshe-ab/rsaha/dm_alchemy/src/data/baseline_preprocessed_from_normalized_reward/compositional_baseline_compositional_chemistry_samples_train_shop_1_qhop_1_single_held_out_color_4_edges_exp_seed_0_classification_filter_True_input_features_output_stone_states_vocab.pkl"

with open(baseline_vocab, 'rb') as f:
    vocab = pickle.load(f)

s2id = vocab['stone_state_to_id']
print(f"Baseline train vocab size: {len(s2id)}")
print(f"First 5 stones:")
for k, v in list(s2id.items())[:5]:
    print(f"  {v}: {k}")

# Also check val
baseline_val_vocab = "/home/rsaha/projects/def-afyshe-ab/rsaha/dm_alchemy/src/data/baseline_preprocessed_from_normalized_reward/compositional_baseline_compositional_chemistry_samples_val_shop_1_qhop_1_single_held_out_color_4_edges_exp_seed_0_classification_filter_True_input_features_output_stone_states_vocab.pkl"
with open(baseline_val_vocab, 'rb') as f:
    val_vocab = pickle.load(f)

val_s2id = val_vocab['stone_state_to_id']
print(f"\nBaseline val vocab size: {len(val_s2id)}")

# Does it match the original?
orig_vocab = "/home/rsaha/projects/def-afyshe-ab/rsaha/dm_alchemy/src/data/shuffled_held_out_exps_preprocessed_separate_enhanced/compositional_chemistry_samples_167424_80_unique_stones_train_shop_1_qhop_1_single_held_out_color_4_edges_exp_seed_0_classification_filter_True_input_features_output_stone_states_vocab.pkl"
with open(orig_vocab, 'rb') as f:
    o_vocab = pickle.load(f)

o_s2id = o_vocab['stone_state_to_id']
print(f"\nOriginal vocab size: {len(o_s2id)}")

# Check mapping mismatch
mismatches = 0
for k, v in s2id.items():
    if k in o_s2id and o_s2id[k] != v:
        mismatches += 1

print(f"\nMismatched IDs between original and baseline vocabs: {mismatches}")
