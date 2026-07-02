import pickle

train_vocab_file = "/home/rsaha/projects/def-afyshe-ab/rsaha/dm_alchemy/src/data/held_out_randomized_reward_preprocessed_separate_enhanced/compositional_chemistry_samples_167424_80_unique_stones_train_shop_1_qhop_1_single_held_out_color_4_edges_exp_seed_0_classification_filter_True_input_features_output_stone_states_vocab.pkl"
val_vocab_file = "/home/rsaha/projects/def-afyshe-ab/rsaha/dm_alchemy/src/data/held_out_randomized_reward_preprocessed_separate_enhanced/compositional_chemistry_samples_167424_80_unique_stones_val_shop_1_qhop_1_single_held_out_color_4_edges_exp_seed_0_classification_filter_True_input_features_output_stone_states_vocab.pkl"

with open(train_vocab_file, 'rb') as f:
    train_vocab = pickle.load(f)

with open(val_vocab_file, 'rb') as f:
    val_vocab = pickle.load(f)

train_s2id = train_vocab['stone_state_to_id']
val_s2id = val_vocab['stone_state_to_id']

print(f"Train stone_state_to_id: {len(train_s2id)} entries")
print(f"Val   stone_state_to_id: {len(val_s2id)} entries")

# Check if they are identical
mismatches = 0
for stone_str, train_id in train_s2id.items():
    val_id = val_s2id.get(stone_str)
    if val_id is None:
        print(f"  MISSING in val: {stone_str} (train ID={train_id})")
        mismatches += 1
    elif val_id != train_id:
        print(f"  ID MISMATCH: {stone_str} -> train_id={train_id}, val_id={val_id}")
        mismatches += 1

for stone_str in val_s2id:
    if stone_str not in train_s2id:
        print(f"  MISSING in train: {stone_str} (val ID={val_s2id[stone_str]})")
        mismatches += 1

if mismatches == 0:
    print("Train and Val stone_state_to_id are IDENTICAL.")
else:
    print(f"\n*** FOUND {mismatches} MISMATCHES! ***")
    print("This could explain the P(A) failure!")

# Also compare input vocabularies
train_input = train_vocab['input_word2idx']
val_input = val_vocab['input_word2idx']
input_mismatches = 0
for word, train_id in train_input.items():
    val_id = val_input.get(word)
    if val_id != train_id:
        input_mismatches += 1
        print(f"  INPUT VOCAB MISMATCH: '{word}' -> train={train_id}, val={val_id}")

print(f"\nInput vocab mismatches: {input_mismatches}")
