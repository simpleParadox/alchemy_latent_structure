import torch
import pickle
from src.models.models import StoneStateDecoderClassifier

checkpoint = torch.load('src/saved_models/shuffled_held_out_exps_preprocessed_separate_enhanced_new/held_out_color_exp/held_out_edges_4/all_graphs/scheduler_cosine/wd_0.01_lr_0.0001/eta_min_9.5e-05/xsmall/decoder/classification/input_features/output_stone_states/shop_1_qhop_1/seed_0/init_seed_42/best_model_epoch_995_classification_xsmall.pt', map_location='cpu', weights_only=False)
state_dict = checkpoint.get('model_state_dict', checkpoint)

vocab0 = pickle.load(open('src/data/chemistry_pickles/original_reward_potion_remap_preprocessed_data/compositional_chemistry_samples_167424_80_unique_stones_val_shop_1_qhop_1_single_held_out_color_4_edges_exp_seed_0_pairing_index_0_classification_filter_True_input_features_output_stone_states_vocab.pkl', 'rb'))
input_word2idx = vocab0['input_word2idx']
idx2word = vocab0['input_idx2word']
id_to_state = vocab0['id_to_stone_state']

# Create unpermuted model
model_unperm = StoneStateDecoderClassifier(
    num_decoder_layers=4, emb_size=256, nhead=4, dim_feedforward=512, dropout=0.1,
    src_vocab_size=len(input_word2idx), num_classes=108, use_flash_attention=False,
    max_len=state_dict['positional_encoding.pe'].shape[0], vocab=input_word2idx
)
model_unperm.load_state_dict(state_dict)
model_unperm.eval()

# Create permuted model
model_perm = StoneStateDecoderClassifier(
    num_decoder_layers=4, emb_size=256, nhead=4, dim_feedforward=512, dropout=0.1,
    src_vocab_size=len(input_word2idx), num_classes=108, use_flash_attention=False,
    max_len=state_dict['positional_encoding.pe'].shape[0], vocab=input_word2idx
)
model_perm.load_state_dict(state_dict)
model_perm.eval()

# Permute embeddings
from src.mech_interp.oracle_permutation_eval import PAIRINGS
L0 = PAIRINGS[0]
Lk = PAIRINGS[6]
E = model_perm.src_tok_emb.weight.data
E_new = E.clone()
for i, c in enumerate(L0):
    src_idx = input_word2idx[c]
    tgt_idx = input_word2idx[Lk[i]]
    E_new[tgt_idx] = E[src_idx]
E.copy_(E_new)

d6 = pickle.load(open('src/data/chemistry_pickles/original_reward_potion_remap_preprocessed_data/compositional_chemistry_samples_167424_80_unique_stones_val_shop_1_qhop_1_single_held_out_color_4_edges_exp_seed_0_pairing_index_6_classification_filter_True_input_features_output_stone_states_data.pkl', 'rb'))

unperm_correct = 0
perm_correct = 0
total_interior = 0

for idx, s in enumerate(d6):
    inp = s['encoder_input_ids']
    query_potion = idx2word[inp[-1]]
    
    if query_potion in ['RED', 'YELLOW', 'GREEN', 'ORANGE']:
        color = idx2word[inp[-5]]
        size = idx2word[inp[-4]]
        
        is_interior = False
        if query_potion in ['RED', 'YELLOW'] and color == 'purple':
            is_interior = True
        elif query_potion in ['GREEN', 'ORANGE'] and size == 'medium':
            is_interior = True
            
        if is_interior:
            inp_tensor = torch.tensor([inp], dtype=torch.long)
            target = s['target_class_id']
            with torch.no_grad():
                pred_unperm = model_unperm(inp_tensor).argmax(dim=-1).item()
                pred_perm = model_perm(inp_tensor).argmax(dim=-1).item()
                
            if pred_unperm == target:
                unperm_correct += 1
            if pred_perm == target:
                perm_correct += 1
            total_interior += 1

print(f'Total interior-state queries: {total_interior}')
if total_interior > 0:
    print(f'Unpermuted Model Accuracy: {unperm_correct/total_interior*100:.4f}% ({unperm_correct}/{total_interior})')
    print(f'Permuted Model Accuracy: {perm_correct/total_interior*100:.4f}% ({perm_correct}/{total_interior})')
else:
    print("No interior-state queries found.")
