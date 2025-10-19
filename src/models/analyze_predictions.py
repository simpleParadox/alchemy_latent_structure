import json
import pickle
import torch
import numpy as np
from collections import defaultdict, Counter
from tqdm import tqdm
import matplotlib.pyplot as plt
from cluster_profile import cluster

# Load the metadata, data, and vocab.
if cluster == 'vulcan':
    infix = 'def-afyshe-ab/rsaha/'
else:
    # Usually for cirrus but might change later.
    infix = ''

def create_reverse_stone_mapping(stone_state_to_id):
    """Create reverse mapping from class ID to stone state string."""
    return {v: k for k, v in stone_state_to_id.items()}

def parse_stone_states_from_input(encoder_input_ids, input_vocab, stone_state_to_id):
    """
    Parse stone states from encoder_input_ids based on the pattern:
    [4 features] + [UPPERCASE color] + [<io_sep>] + [4 features] + [<item_sep>] + ...
    Final: [<item_sep>] + [4 features] + [UPPERCASE color]
    
    Returns list of (stone_state_string, class_id) tuples found in input.
    """
    idx2word = {v: k for k, v in input_vocab.items()}
    io_sep_id = input_vocab['<io>']
    item_sep_id = input_vocab['<item_sep>']
    
    # Convert to tokens
    tokens = [idx2word[token_id] for token_id in encoder_input_ids]
    
    stone_states = []
    i = 0
    
    while i < len(tokens):
        # Look for pattern: 4 features + UPPERCASE color
        if i + 4 < len(tokens):
            # Check if we have 4 feature tokens followed by an uppercase color
            features = tokens[i:i+4]
            potential_color = tokens[i+4]
            
            # Check if it's an uppercase color (these are the reward colors)
            uppercase_colors = ['CYAN', 'GREEN', 'ORANGE', 'PINK', 'RED', 'YELLOW']
            if potential_color in uppercase_colors:
                # Construct stone state string
                # Assuming order is [color, size, roundness, reward]
                color, size, roundness, reward = features
                stone_state_str = f"{{color: {color}, size: {size}, roundness: {roundness}, reward: {reward}}}"
                
                # Get class ID if it exists
                if stone_state_str in stone_state_to_id:
                    class_id = stone_state_to_id[stone_state_str]
                    stone_states.append((stone_state_str, class_id))
                else:
                    stone_states.append((stone_state_str, None))  # Unknown stone state
                
                # Move past this stone state
                i += 5
                
                # Skip io_sep or item_sep if present
                if i < len(tokens) and tokens[i] in ['<io>', '<item_sep>']:
                    i += 1
            else:
                i += 1
        else:
            i += 1
    
    return stone_states



def analyze_half_chemistry_behaviour(data, vocab, stone_state_to_id, predictions_by_epoch, exp_typ='held_out', hop=2):
    """
    Analyze model behavior on half-chemistry tasks where only one stone is present.
    
    Args:
        data: List of validation samples
        vocab: Vocabulary dictionaries  
        stone_state_to_id: Mapping from stone state strings to class IDs
    """
    # I can first get all the samples that have the same support sentence - in the exact same order.
    # I should find 8 such samples because there are 8 such queries.
    
    # First for all the samples, create a key based on the support sentence which is everything except the last 5 tokens of the sample.
    support_to_query_mappings = {}
    for i, sample in enumerate(data):
        encoder_input_ids = sample['encoder_input_ids']
        target_class_id = sample['target_class_id']
        support = encoder_input_ids[:-5]  # Everything except last 5 tokens
        # Create a hashable string key
        support_key = tuple(support)
        
        if support_key not in support_to_query_mappings: # This is for the input to the model. This will be the same for all epochs.
            support_to_query_mappings[support_key] = {}
            
            
    input_vocab = vocab['input_word2idx']
    feature_to_id_vocab = {v: k for k, v in input_vocab.items()}
            
    for i, sample in enumerate(data):
        encoder_input_ids = sample['encoder_input_ids']
        target_class_id = sample['target_class_id']
        support = encoder_input_ids[:-5]  # Everything except last 5 tokens. Works for decomposition too.
        # Create a hashable string key
        support_key = tuple(support)
        
        if exp_typ == 'composition':
            # Based on the number of hops, we need to adjust the query parsing. The hops denote the number of potions in the query.
            query = encoder_input_ids[-(hop + 4):] # 4 featuresd + hop potions.
            query_potion = query[-hop:]  # Last hop tokens
            query_stones = query[:-hop]
            
            # Create a string representation of the query potion sequence from the feature_to_id_vocab and join them.
            query_potion_str = ' | '.join([feature_to_id_vocab[token_id] for token_id in query_potion])
            query_potion = query_potion_str
            
        else:
            query = encoder_input_ids[-5:]    # Last 5 tokens
            query_potion = query[-1]
            query_stones = query[:-1]
        
        # First check if the query_potion key is already a list. If not, create an empty list.
        if feature_to_id_vocab[query_potion] not in support_to_query_mappings[support_key]:
            support_to_query_mappings[support_key][feature_to_id_vocab[query_potion]] = [target_class_id]

        else:
            support_to_query_mappings[support_key][feature_to_id_vocab[query_potion]].append(target_class_id)

    # Store the predictions for each support key and query potion.

    support_to_query_per_epoch_predictions = {}

    for epoch in predictions_by_epoch.keys():
        support_to_query_per_epoch_predictions[epoch] = {}
        for support_key in support_to_query_mappings.keys():
            support_to_query_per_epoch_predictions[epoch][support_key] = {}
            for potion in support_to_query_mappings[support_key].keys(): # For the composition experiments, this will be multiple potions.
                support_to_query_per_epoch_predictions[epoch][support_key][potion] = []


    # Now for each sample, we can store the prediction in the corresponding support key and query potion.
    for epoch, predictions in tqdm(predictions_by_epoch.items(), desc="Organizing predictions by support and query"):
        for i, sample in enumerate(data):
            encoder_input_ids = sample['encoder_input_ids']
            target_class_id = sample['target_class_id']
            predicted_class_id = predictions[i]
            support = encoder_input_ids[:-5]  # Everything except last 5 tokens
            support_key = tuple(support)
            
            if exp_typ == 'composition':
                # Based on the number of hops, we need to adjust the query parsing. The hops denote the number of potions in the query.
                query = encoder_input_ids[-(hop + 4):] # 4 featuresd + hop potions.
                query_potion = query[-hop:]  # Last hop tokens
                
                # Create a string representation of the query potion sequence from the feature_to_id_vocab and join them.
                query_potion_str = ' | '.join([feature_to_id_vocab[token_id] for token_id in query_potion])
                query_potion = query_potion_str
                
            else:  
                query = encoder_input_ids[-5:]    # Last 5 tokens
                query_potion = query[-1]

            support_to_query_per_epoch_predictions[epoch][support_key][feature_to_id_vocab[query_potion]].append(predicted_class_id)

            
    # Now for each of the predictions, we can check which half-chemistry for that support set does the true target belong to.
    
    # Initialize accumulators for averaging
    predicted_in_context_accuracies = []
    predicted_in_context_correct_half_accuracies = []
    predicted_in_context_other_half_accuracies = []
    predicted_in_context_correct_half_exact_accuracies = []
    predicted_correct_within_context = []
    
    for epoch, predictions in tqdm(predictions_by_epoch.items(), desc="Analyzing epochs"):
        # print(f"\n--- Epoch {epoch} Half-Chemistry Analysis ---")
        correct = 0
        other_half_correct = 0
        total = 0
        
        
        within_class_correct = 0
        within_class_total = 0

        predicted_in_context_count = 0
        correct_half_chemistry_count = 0

        predicted_correct_within_context_count = 0
        
        for i, sample in enumerate(data):
            encoder_input_ids = sample['encoder_input_ids']
            target_class_id = sample['target_class_id']
            predicted_class_id = predictions[i]
            support = encoder_input_ids[:-5]  # Everything except last 5 tokens
            support_key = tuple(support)
            
            if exp_typ == 'composition':
                # Based on the number of hops, we need to adjust the query parsing. The hops denote the number of potions in the query.
                query = encoder_input_ids[-(hop + 4):] # 4 featuresd + hop potions.
                query_potion = query[-hop:]  # Last hop tokens
                
                # Create a string representation of the query potion sequence from the feature_to_id_vocab and join them.
                query_potion_str = ' | '.join([feature_to_id_vocab[token_id] for token_id in query_potion])
                query_potion = query_potion_str
            else:
                query = encoder_input_ids[-5:]    # Last 5 tokens
                query_potion = query[-1]

            # First we check if the predicted class ID is in any of the two half-chemistry sets for this support key.

            potions_for_support = list(support_to_query_mappings[support_key].keys())
            # TODO: This will fail for composition experiments because the keys will be a combination of potions. Thus we need to adjust this.
            
            if exp_typ == 'composition':
                raise NotImplementedError("Half-chemistry analysis for composition experiments is not implemented yet.")
            
            assert len(potions_for_support) in [2,6], f"Expected 2 or 6 potions."
            correct_half_chemistry = support_to_query_mappings[support_key][feature_to_id_vocab[query_potion]]
            other_half_chemistry = support_to_query_mappings[support_key][potions_for_support[0]] if potions_for_support[1] == feature_to_id_vocab[query_potion] else support_to_query_mappings[support_key][potions_for_support[1]]

            if exp_typ == 'decomposition' or exp_typ == 'composition':
                # Create a set of all stones in the support set.
                all_stones_in_support = set()
                for potion in potions_for_support:
                    all_stones_in_support.update(support_to_query_mappings[support_key][potion])
                combined_set = all_stones_in_support
                assert len(combined_set) == 8, f"Expected 8 unique stones for support {support_key}, got {len(combined_set)}"
            else:
                combined_set = set(correct_half_chemistry + other_half_chemistry)
                assert len(combined_set) < 8, f"Expected 8 unique stones for support {support_key}, got {len(combined_set)}"

            import pdb; pdb.set_trace()

            # First do the classification for 8 vs 108.
            if predicted_class_id in combined_set:
                predicted_in_context_count += 1

                if predicted_class_id == target_class_id:
                    predicted_correct_within_context_count += 1
                

                # Now if the predicted class ID is in the combined set, we can check if it is in the correct half-chemistry set.
                # This is conditional probability p(y in correct_half | x) * p(predicted_y in context_ids).
                if predicted_class_id in correct_half_chemistry:
                    correct_half_chemistry_count += 1

                    if predicted_class_id == target_class_id:
                        # This is another conditional probability p(y = target | y in correct_half, x) * p(y in correct_half | x) * p(predicted_y in context_ids).
                        within_class_correct += 1

                elif predicted_class_id in other_half_chemistry:
                    # This is also conditional probability p(y in other_half | x) * p(predicted_y in context_ids).\
                    other_half_correct += 1

            else:
                # This means the predicted class ID is not in either half-chemistry set and is technically and incorrect prediction.
                pass
            total += 1

        # Now calculate accuracies and store them.

        predicted_in_context_accuracy = predicted_in_context_count / total if total > 0 else 0 # The chance is 8/108 here.
        predicted_in_context_accuracies.append(predicted_in_context_accuracy)

        # Now calculate the correct half-chemistry accuracy if predicted in context.
        predicted_in_context_correct_half_accuracy = correct_half_chemistry_count / predicted_in_context_count if predicted_in_context_count > 0 else 0 # The chance is 4/8 here.
        predicted_in_context_correct_half_accuracies.append(predicted_in_context_correct_half_accuracy)

        # Now calculate the other half-chemistry accuracy if predicted in context.
        predicted_in_context_other_half_accuracy = other_half_correct / predicted_in_context_count if predicted_in_context_count > 0 else 0 # The chance is 4/8 here.
        predicted_in_context_other_half_accuracies.append(predicted_in_context_other_half_accuracy)

        # Now calculate the within-class accuracy if predicted in context and in correct half-chemistry.
        predicted_in_context_correct_half_exact_accuracy = within_class_correct / correct_half_chemistry_count if correct_half_chemistry_count > 0 else 0 # The chance is 1/4 here.
        predicted_in_context_correct_half_exact_accuracies.append(predicted_in_context_correct_half_exact_accuracy)

        # Now calculate the correct within context accuracy.
        predicted_correct_within_context_accuracy = predicted_correct_within_context_count / predicted_in_context_count if predicted_in_context_count > 0 else 0 # The chance is 1/8 here.
        predicted_correct_within_context.append(predicted_correct_within_context_accuracy)



    return predicted_in_context_accuracies, predicted_in_context_correct_half_accuracies, predicted_in_context_other_half_accuracies, predicted_in_context_correct_half_exact_accuracies, predicted_correct_within_context


def load_epoch_data(exp_typ: str = 'held_out', hop = 2, epoch_range = (0, 500), seeds = [2], scheduler_prefix=''):
    """
    Load predictions and inputs/targets/predictions for specified experiment type and hop (if applicable).
    
    Args:
        exp_typ: 'latent' or 'classification'
        hops: List of hop counts to load data for
    """ 

    
    epoch_start, epoch_end = epoch_range
    
    predictions_by_epoch_by_seed = {}
    
    inputs_by_seed = {}
    targets_by_seed = {}

    print("Seeds to load: ", seeds)

    for seed in tqdm(seeds):
        predictions_by_epoch = {}
        
        for epoch in range(epoch_start, epoch_end + 1):
            # Reformat the epoch_number because the files are saved with epoch numbers like 001, 002, ..., 1000
            epoch_number = str(epoch).zfill(3)
            
            # for hop in hops:
            base_file_path = ''
            
            if exp_typ == 'held_out':
                base_file_path = f'/home/rsaha/projects/{infix}dm_alchemy/src/saved_models/held_out_color_exp/held_out_edges_{hop}/all_graphs/xsmall/decoder/classification/{scheduler_prefix}input_features/output_stone_states/shop_1_qhop_1/seed_{seed}/predictions'
                    
            elif exp_typ == 'decomposition':
                base_file_path = f"/home/rsaha/projects/{infix}dm_alchemy/src/saved_models/complete_graph/xsmall/decoder/classification/{scheduler_prefix}input_features/output_stone_states/shop_{hop}_qhop_1/seed_{seed}/predictions" 
            elif exp_typ == 'composition':
                base_file_path = f"/home/rsaha/projects/{infix}dm_alchemy/src/saved_models/complete_graph/fully_shuffled/xsmall/decoder/classification/{scheduler_prefix}input_features/output_stone_states/shop_1_qhop_{hop}/seed_{seed}/predictions"
                
                    
            predictions_raw_file_path = f'{base_file_path}/predictions_classification_epoch_{epoch_number}.npz'
            
            try:
                # if exp_typ == 'decomposition':
                #     if epoch_number == '035':
                #         continue

                print(f"epoch number, ", epoch_number)
                if epoch_number == '828' and exp_typ == 'held_out':
                    continue

                predictions_raw = np.load(predictions_raw_file_path, allow_pickle=True)['predictions']
                # Store predictions for this epoch
                predictions_by_epoch[epoch_number] = predictions_raw.tolist()
                
                # print(f"Loaded epoch {epoch_number}: {len(predictions_raw)} predictions")
                
            except FileNotFoundError:
                print(f"Warning: Files for epoch {epoch_number} not found, skipping...")
                continue

        try:

            predictions_by_epoch_by_seed[seed] = predictions_by_epoch 
            inputs_raw_file_path = f'{base_file_path}/inputs_classification_epoch_001.npz' # Use the last epoch number loaded. Doesn't matter because inputs are same for all epochs.
            targets_raw_file_path = f'{base_file_path}/targets_classification_epoch_001.npz' # Use the last epoch number loaded.
            inputs_raw = np.load(inputs_raw_file_path, allow_pickle=True)['inputs']
            targets_raw = np.load(targets_raw_file_path, allow_pickle=True)['targets']
            
            stacked_inputs = np.vstack(inputs_raw) # Flatten inputs from (39, 32, 181) to (1240, 181) 
            data_with_targets = [{'encoder_input_ids': stacked_inputs[i].tolist(), 'target_class_id': int(targets_raw[i])} for i in range(len(targets_raw))]
            
            inputs_by_seed[seed] = data_with_targets
        except FileNotFoundError:
            print(f"Warning: Input/target files for seed {seed} not found, skipping...")
            continue
        
    return predictions_by_epoch_by_seed, inputs_by_seed
        
   
import argparse

# Parse command line arguments for experiment type and hop count.
parser = argparse.ArgumentParser(description="Analyze model predictions for different experiment types and hops.")
parser.add_argument('--exp_typ', type=str, choices=['held_out', 'decomposition', 'composition'], default='composition',
                    help="Type of experiment: 'held_out' or 'decomposition'")
parser.add_argument('--hop', type=int, choices=[2, 3, 4, 5], default=2,
                    help="Hop count for decomposition and composition experiments (ignored for held_out)")
args = parser.parse_args()
# exp_typ = 'decomposition'  # 'held_out' or 'decomposition'
exp_typ = args.exp_typ
hop = args.hop  # Only relevant for composition and decomposition experiments.
two_hop_epoch_values_text = [0, 200, 400, 600, 800, 999]
three_hop_epoch_values_text = [0, 200, 600, 800, 999]
four_hop_epoch_values_text = [0, 200, 400, 600, 800, 999]
five_hop_epoch_values_text = [0, 200, 600, 800, 999]

# Create a dictionary mapping hop counts to their hop-specific epoch values
hop_to_epoch_values = {
    2: two_hop_epoch_values_text,
    3: three_hop_epoch_values_text,
    4: four_hop_epoch_values_text,
    5: five_hop_epoch_values_text
}


scheduler_prefix = 'cosine/' if hop in [2,3,4] else ''
# scheduler_prefix = '' 
# seed_values = [2,3,4]
if exp_typ == 'decomposition':
    
    # NOTE: Do not change this.
    seed_values_2_hop = [2,3,4]
    seed_values_3_hop = [0,1,2,3,4]
    seed_values_4_hop = [2,3,4] # For the 4
    seed_values_5_hop = [1,2,3]
elif exp_typ == 'held_out':
    seed_values_2_hop = [2,3,4]
    seed_values_3_hop = [2,3,4]
    seed_values_4_hop = [2,3,4]
    seed_values_5_hop = [2,3,4]

seed_values_hop_dict = {
    2: seed_values_2_hop,
    3: seed_values_3_hop,
    4: seed_values_4_hop,
    5: seed_values_5_hop
}
elif exp_typ == 'composition':
    hop_to_epoch_values = {
        2: [0, 200, 400, 499],
        3: [0, 200, 400, 499],
        4: [0, 200, 400, 499],
        5: [0, 200, 400, 499]
    }
    
    seed_values_hop_dict = {
        2: [0, 16, 29],
        3: [0, 16, 29],
        4: [0, 16, 29],
        5: [0, 16, 29]
    }
    

"""
for 2 hop, use seeds
for 3 hop, use seeds 4, and 0.
for 4 hop, can use seeds 2,3,4
for 5 hop, use seed 1,2,3
"""
# Load the for all the seeds.
predictions_by_epoch_by_seed, inputs_by_seed  = load_epoch_data(
    exp_typ = exp_typ,
    hop = hop,
    epoch_range = (hop_to_epoch_values[hop][0], hop_to_epoch_values[hop][-1]),
    seeds = seed_values_hop_dict[hop],
    scheduler_prefix = scheduler_prefix
)
# import pdb; pdb.set_trace()
seed_data_files = {}
for seed in predictions_by_epoch_by_seed.keys():
    if exp_typ == 'held_out':
        data_files = {
            "data": f"/home/rsaha/projects/{infix}dm_alchemy/src/data/shuffled_held_out_exps_preprocessed_separate_enhanced/compositional_chemistry_samples_167424_80_unique_stones_val_shop_1_qhop_1_single_held_out_color_4_edges_exp_seed_{seed}_classification_filter_True_input_features_output_stone_states_data.pkl",
            "vocab": f"/home/rsaha/projects/{infix}dm_alchemy/src/data/shuffled_held_out_exps_preprocessed_separate_enhanced/compositional_chemistry_samples_167424_80_unique_stones_val_shop_1_qhop_1_single_held_out_color_4_edges_exp_seed_{seed}_classification_filter_True_input_features_output_stone_states_vocab.pkl",
            "metadata": f"/home/rsaha/projects/{infix}dm_alchemy/src/data/shuffled_held_out_exps_preprocessed_separate_enhanced/compositional_chemistry_samples_167424_80_unique_stones_val_shop_1_qhop_1_single_held_out_color_4_edges_exp_seed_{seed}_classification_filter_True_input_features_output_stone_states_metadata.json"
        }

        vocab = pickle.load(open(data_files["vocab"], "rb"))
        with open(data_files["metadata"], "r") as f:
            metadata = json.load(f)

        seed_data_files[seed] = {'vocab': vocab, 'metadata': metadata}


    elif exp_typ == 'decomposition':
        data_files = {
            "metadata": f"/home/rsaha/projects/def-afyshe-ab/rsaha/dm_alchemy/src/data/complete_graph_preprocessed_separate_enhanced_qnodes_in_snodes/decompositional_chemistry_samples_167424_80_unique_stones_val_shop_{hop}_qhop_1_seed_{seed}_classification_filter_True_input_features_output_stone_states_metadata.json",
            "vocab": f"/home/rsaha/projects/def-afyshe-ab/rsaha/dm_alchemy/src/data/complete_graph_preprocessed_separate_enhanced_qnodes_in_snodes/decompositional_chemistry_samples_167424_80_unique_stones_val_shop_{hop}_qhop_1_seed_{seed}_classification_filter_True_input_features_output_stone_states_vocab.pkl",
        }
        
        with open(data_files["metadata"], "r") as f:
            metadata = json.load(f)
        vocab = pickle.load(open(data_files["vocab"], "rb"))

        seed_data_files[seed] = {'vocab': vocab, 'metadata': metadata}
        
    elif exp_typ == 'composition':
        data_files = {
            "metadata": f"/home/rsaha/projects/{infix}dm_alchemy/src/data/complete_graph_composition_fully_shuffled_balanced_grouped_by_unique_end_state_preprocessed/compositional_chemistry_samples_167424_80_unique_stones_val_shop_1_qhop_2_seed_0_classification_filter_True_input_features_output_stone_states_metadata.json",
            "vocab": f"/home/rsaha/projects/{infix}dm_alchemy/src/data/complete_graph_composition_fully_shuffled_balanced_grouped_by_unique_end_state_preprocessed/compositional_chemistry_samples_167424_80_unique_stones_val_shop_1_qhop_2_seed_0_classification_filter_True_input_features_output_stone_states_vocab.pkl",
        }

        with open(data_files["metadata"], "r") as f:
            metadata = json.load(f)
        vocab = pickle.load(open(data_files["vocab"], "rb"))

        seed_data_files[seed] = {'vocab': vocab, 'metadata': metadata}


# Whether it predicts the correct half, if within the correct half, whether it predicts the correct stone, and whether the model predicts that
# the stone is in the other incorrect half.
seed_results = {}
for seed in predictions_by_epoch_by_seed.keys():
    print(f"\n\nAnalyzing seed {seed}...")
    predictions_by_epoch = predictions_by_epoch_by_seed[seed]
    data_with_predictions = inputs_by_seed[seed]
    
    # Load the correct vocab for this seed
    vocab = seed_data_files[seed]['vocab']
    
    # Run the analysis
    print("\n" + "="*60)
    print("Running half-chemistry behavior analysis...")
    print("="*60)
    # model_selection_results = analyze_model_selection_behavior(
    #     data_with_predictions, 
    #     vocab, 
    #     vocab['stone_state_to_id'], 
    #     predictions_by_epoch
    # )

    # Get half_chemistry_analysis results.
    half_chemistry_results = analyze_half_chemistry_behaviour(
        data_with_predictions, vocab, vocab['stone_state_to_id'], predictions_by_epoch, exp_typ=exp_typ, hop=hop
    )
    predicted_in_context_accuracies, \
        predicted_in_context_correct_half_accuracies, \
            predicted_in_context_other_half_accuracies, \
                predicted_in_context_correct_half_exact_accuracies, \
                    predicted_correct_within_context = half_chemistry_results

    # Store results for this seed
    seed_results[seed] = {
        'predicted_in_context_accuracies': predicted_in_context_accuracies,
        'predicted_in_context_correct_half_accuracies': predicted_in_context_correct_half_accuracies,
        'predicted_in_context_other_half_accuracies': predicted_in_context_other_half_accuracies,
        'predicted_in_context_correct_half_exact_accuracies': predicted_in_context_correct_half_exact_accuracies,
        'predicted_correct_within_context': predicted_correct_within_context
    }

# import pdb; pdb.set_trace()
# Now the plotting begins. First we need to average the result for each metric across seeds.
# Average results across seeds
averaged_results = {}
std_errors = {}
for metric in ['predicted_in_context_accuracies', 'predicted_in_context_correct_half_accuracies', 'predicted_in_context_other_half_accuracies', 'predicted_in_context_correct_half_exact_accuracies', 'predicted_correct_within_context']:
    all_seed_values = [seed_results[seed][metric] for seed in seed_results.keys()]
    
    # Find the maximum length across all seeds
    max_length = max(len(values) for values in all_seed_values)
    
    # Pad shorter sequences with imputed values
    padded_seed_values = []
    for values in all_seed_values:
        if len(values) < max_length:
            # For missing epochs, compute mean from seeds that have data at those epochs
            padded_values = list(values)
            for epoch_idx in range(len(values), max_length):
                # Get values from all seeds that have this epoch
                available_values = [seed_vals[epoch_idx] for seed_vals in all_seed_values if len(seed_vals) > epoch_idx]
                if available_values:
                    imputed_value = np.mean(available_values)
                else:
                    # Fallback: use the last available value from this seed
                    imputed_value = values[-1]
                padded_values.append(imputed_value)
            padded_seed_values.append(padded_values)
        else:
            padded_seed_values.append(values)
    
    # Now all sequences have the same length, can safely stack
    all_seed_values = np.array(padded_seed_values)
    averaged_results[metric] = np.mean(all_seed_values, axis=0)
    std_errors[metric] = np.std(all_seed_values, axis=0) / np.sqrt(len(all_seed_values))
    print(f"\nAveraged {metric} over seeds:")

# Plot the averaged results with error bars using fill_between
plt.figure(figsize=(14, 10))
epochs = range(len(averaged_results['predicted_in_context_accuracies']))

metrics = [
    ('predicted_in_context_accuracies', 'Prediction in support (8 out of 108)'),
    ('predicted_in_context_correct_half_accuracies', 'Correct Half accuracy (4 out of 8)'),
    ('predicted_in_context_other_half_accuracies', 'Incorrect Half accuracy (4 out of 8)'),
    ('predicted_in_context_correct_half_exact_accuracies', 'Within correct half Accuracy (1 out of 4)'),
    # ('predicted_correct_within_context', 'Exact Accuracy (1 out of 8)'),
]
if exp_typ == 'decomposition':
    # Only print the predicted_in_context_accuracies and the predicted_correct_within_context.
    metrics = [
        ('predicted_in_context_accuracies', 'In-support gating (8 out of 108)'),
        ('predicted_correct_within_context', 'In-support gated exact match (1 out of 8)'),
    ]
linestyles = {'predicted_in_context_accuracies': 'solid', 'predicted_correct_within_context': 'solid'}
colors = {
    2: {'predicted_in_context_accuracies': 'orange', 'predicted_correct_within_context': 'tab:blue'},
    3: {'predicted_in_context_accuracies': 'orange', 'predicted_correct_within_context': 'tab:green'},
    4: {'predicted_in_context_accuracies': 'orange', 'predicted_correct_within_context': 'tab:gray'},
    5: {'predicted_in_context_accuracies': 'orange', 'predicted_correct_within_context': 'tab:red'},
}

held_out_colors = {
    'predicted_in_context_accuracies': 'tab:blue',
    'predicted_in_context_correct_half_accuracies': 'tab:orange',
    'predicted_in_context_other_half_accuracies': 'tab:green',
    'predicted_in_context_correct_half_exact_accuracies': 'tab:red',
    # 'predicted_correct_within_context': 'tab:red',
}
for metric, label in metrics:
    mean = averaged_results[metric]
    sem = std_errors[metric]
    colors = held_out_colors if exp_typ == 'held_out' else colors
    if exp_typ == 'held_out':
        plt.plot(epochs, mean, label=label, linewidth=2, linestyle=linestyles.get(metric, 'solid'), color=colors.get(metric, 'black'))
    else:
        plt.plot(epochs, mean, label=label, linewidth=2, linestyle=linestyles.get(metric, 'solid'), color=colors.get(hop, {}).get(metric, 'black'))
    plt.fill_between(epochs, mean - sem, mean + sem, alpha=0.2, color=colors.get(hop, {}).get(metric, 'black'))
    
    # Add text annotations at specific epochs
    annotate_epochs = hop_to_epoch_values[hop]
    for anno_epoch in annotate_epochs:
        if anno_epoch < len(mean):
            try:
                plt.text(anno_epoch, mean[anno_epoch], f'{mean[anno_epoch]:.2f}', 
                        fontsize=20, ha='center', va='bottom',
                        color='black')
            except:
                print(f"Could not annotate epoch {anno_epoch} for metric {metric}")

plt.xlabel('Epoch', fontsize=26)
plt.ylabel('Accuracy', fontsize=26)
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)
# if exp_typ == 'decomposition':
#     plt.title(f'Phasic learning of intermediate stone inference ({hop}-hop)', fontsize=24, pad=60)
# else:
#     plt.title(f'Phasic learning of latent structure learning', fontsize=24, pad=60)
plt.legend(fontsize=18, loc='upper center', bbox_to_anchor=(0.5, 1.10), ncol=2, frameon=True)
plt.grid(True, alpha=0.3)
plt.ylim(0, 1)
# plt.tight_layout()
plt.savefig(f'{exp_typ}_{hop}_phasic_learning_of_latent_structure.png')
plt.savefig(f'{exp_typ}_{hop}_phasic_learning_of_latent_structure.pdf', bbox_inches='tight')




































# epoch_accuracies = [results[epoch]['predicted_in_context_accuracy'] for epoch in sorted(results.keys())]
# epoch_within_class_accuracies = [results[epoch]['recall'] for epoch in sorted(results.keys())]
# other_half_accuracies = [results[epoch]['accuracy'] for epoch in sorted(results.keys())]


# # # Plot half-chemistry accuracies over epochs
# plt.figure(figsize=(10, 5))
# plt.plot(range(len(epoch_accuracies)), half_chemistry_epoch_accuracies, label='Half-Chemistry Accuracy')
# plt.plot(range(len(epoch_within_class_accuracies)), epoch_within_class_accuracies , label='Within-Class Accuracy')
# plt.plot(range(len(other_half_accuracies)), other_half_accuracies , label='Other Half-Chemistry Accuracy')
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.title('Predicted Stone in Context Accuracy Over Epochs')
# plt.legend()
# plt.grid(True, alpha=0.3)
# plt.savefig('predicted_in_context.png')
# # Print results
# for epoch, result in results.items():
#     print(f"\n--- Epoch {epoch} Results ---")
#     print(f"Accuracy: {result['accuracy']:.4f} ({result['target_selected']}/{result['total_samples']})")
#     print(f"Avg context stones: {np.mean(result['num_context_stones']):.2f}")







            # This if condition means that the first we retrieve the potion and stone dictionary for this support key.
            # Then we check if the potion is in the dictionary - it should. If it doesn't then something is wrong.
            # if feature_to_id_vocab[query_potion] in support_to_query_mappings[support_key]:
            #     possible_targets_for_the_correct_chemistry_half = support_to_query_mappings[support_key][feature_to_id_vocab[query_potion]]
            #     if predicted_class_id in possible_targets_for_the_correct_chemistry_half: # 4 possible targets.
            #         # This checks whether the predicted class ID is in the correct half-chemistry set.
            #         correct += 1 # Chance is 25% here. p(y in correct_half | x) * p(predicted_y in context_ids).
                    
            #         # Now if the predicted class ID is in the correct half-chemistry set,
            #         # we can then track how many times it predicted the correct stone state.
            #         if predicted_class_id == target_class_id:
            #             within_class_correct += 1
            #         within_class_total += 1 # Should be equal to correct.
                
            #     # Now check if the predicted class ID is in the other half-chemistry set.
            #     else:
            #         other_half_chemistries = [k for k in support_to_query_mappings[support_key].keys() if k != feature_to_id_vocab[query_potion]]
            #         other_half_set = support_to_query_mappings[support_key][other_half_chemistries[0]] if other_half_chemistries else None
            #         assert other_half_set is not None, f"No other half-chemistry found for support {support_key}. This should not happen."
            #         if predicted_class_id in other_half_set:
            #             other_half_correct += 1
                    
            #     total += 1
            # else:
            #     print(f"Warning: Potion {feature_to_id_vocab[query_potion]} not found for support {support_key}")
            #     print("This should not happen - something is wrong.")
            
        # accuracy = correct / total if total > 0 else 0 # The chance is 8/108 here.
        # within_class_accuracy = within_class_correct / within_class_total if within_class_total > 0 else 0
        # other_half_accuracy = other_half_correct / total if total > 0 else 0
        
        # epoch_accuracies.append(accuracy)
        # other_half_accuracies.append(other_half_accuracy)
        # epoch_within_class_accuracies.append(within_class_accuracy)
        
        # print(f"Accuracy on half-chemistry tasks: {accuracy:.4f} ({correct}/{total})")
        # print(f"Within-class accuracy (if the model correctly predicts the half-chemistry it belongs to): {within_class_accuracy:.4f} ({within_class_correct}/{within_class_total})")
    
    # Print averaged results across all epochs
    # if epoch_accuracies:
    #     print(f"\n{'='*60}")
    #     print(f"AVERAGED RESULTS ACROSS {len(predictions_by_epoch)} EPOCH(S)")
    #     print(f"{'='*60}")
    #     avg_accuracy = np.mean(epoch_accuracies)
    #     avg_within_class = np.mean(epoch_within_class_accuracies)
    #     print(f"Average half-chemistry accuracy: {avg_accuracy:.4f} (±{np.std(epoch_accuracies):.4f})")
    #     print(f"Average within-class accuracy: {avg_within_class:.4f} (±{np.std(epoch_within_class_accuracies):.4f})")
           
    # # assert len(support_to_query_mappings[support_key]) <= 8, f"More than 8 queries found for support {support_key}"
    # print(f"\nTotal unique support sentences found: {len(support_to_query_mappings)}")


    # # Return the list of epoch_accuracies and epoch_within_class_accuracies for later plotting.
    # return epoch_accuracies, epoch_within_class_accuracies, other_half_accuracies

# def analyze_model_selection_behavior(data, vocab, stone_state_to_id, predictions_by_epoch):
#     """
#     Analyze which stone state from the input context the model selects at different epochs.
    
#     Args:
#         data: List of validation samples
#         vocab: Vocabulary dictionaries  
#         stone_state_to_id: Mapping from stone state strings to class IDs
#         predictions_by_epoch: Dict {epoch: list_of_predictions}
    
#     Returns:
#         Dictionary with analysis results per epoch
#     """
#     id_to_stone_state = create_reverse_stone_mapping(stone_state_to_id)
#     input_vocab = vocab['input_word2idx']
    
#     results_by_epoch = {}
    
#     for epoch, predictions in tqdm(predictions_by_epoch.items(), desc="Analyzing epochs"):
#         epoch_results = {
#             'total_samples': len(data),
#             'target_selected': [],
#             'selection_positions': [],  # Which position in context was selected (0=first, 1=second, etc.)
#             'num_context_stones': [],   # How many stone states in each context
#             'target_positions': [],     # Position of target in context
#             'in_context_ids': [],       # Class IDs of stones in context,
#             'num_unique_context_stones': [],  # Number of unique stones in context
#             'unique_context_ids': [],  # Unique class IDs in context
#             'unique_context_stones': [],  # Unique stone states in context
#             'predicted_in_context': []  # Whether predicted stone was in context
#         }
        
#         for i, sample in enumerate(data):
#             encoder_input_ids = sample['encoder_input_ids']
#             target_class_id = sample['target_class_id']
#             predicted_class_id = predictions[i]
            
#             # Parse stone states from input context
#             context_stones = parse_stone_states_from_input(
#                 encoder_input_ids, input_vocab, stone_state_to_id
#             )
            
#             context_class_ids = [class_id for _, class_id in context_stones]
            
#             # Find position of prediction in context. Not really necessary to calculate this.
#             if predicted_class_id in context_class_ids:
#                 selection_pos = context_class_ids.index(predicted_class_id)
#                 epoch_results['selection_positions'].append(selection_pos)
            
#             # # Find position of target in context
#             # if target_class_id in context_class_ids:
#             #     target_pos = context_class_ids.index(target_class_id)
#             #     epoch_results['target_positions'].append(target_pos)
#             # else:
#             #     epoch_results['target_positions'].append(-1)  # Target not in context
            
#             # Theh following condition is 8 vs 108.
#             if predicted_class_id in context_class_ids:
#                 epoch_results['predicted_in_context'].append(1)
#             else:
#                 epoch_results['predicted_in_context'].append(0)
            
#             # Check if target was selected. This is the global accuracy (1 vs 108).
#             if predicted_class_id == target_class_id:
#                 epoch_results['target_selected'].append(1)
#             else:
#                 epoch_results['target_selected'].append(0)
            
#             epoch_results['num_context_stones'].append(len(context_stones))
#             epoch_results['num_unique_context_stones'].append(len(set(context_class_ids)))
#             epoch_results['unique_context_ids'].append(list(set(context_class_ids)))
#             epoch_results['unique_context_stones'].append([list(set([stone for stone, _ in context_stones]))])
            
        
#         # Calculate recall (only for samples where target was in context)
#         samples_with_target_in_context = epoch_results['total_samples'] - epoch_results['target_positions'].count(-1)
#         recalled_count = sum(epoch_results['target_selected']) / samples_with_target_in_context if samples_with_target_in_context > 0 else 0
#         # Store the predicted_in_context accuracy
#         epoch_results['predicted_in_context_accuracy'] = sum(epoch_results['predicted_in_context']) / epoch_results['total_samples']

        
#         # Calculate accuracy
#         epoch_results['accuracy'] = sum(epoch_results['target_selected']) / epoch_results['total_samples']
#         epoch_results['recall'] = recalled_count
        
#         results_by_epoch[epoch] = epoch_results

#     return results_by_epoch

# def plot_selection_analysis(results_by_epoch):
    # """Plot analysis of model selection behavior over epochs."""
    # epochs = sorted(results_by_epoch.keys())
    # accuracies = [results_by_epoch[epoch]['accuracy'] for epoch in epochs]
    
    # fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # # Plot 1: Accuracy over epochs
    # axes[0, 0].plot(epochs, accuracies, 'b-', linewidth=2)
    # axes[0, 0].set_xlabel('Epoch')
    # axes[0, 0].set_ylabel('Accuracy')
    # axes[0, 0].set_title('Validation Accuracy Over Training')
    # axes[0, 0].grid(True, alpha=0.3)
    
    # # Plot 2: Selection position distribution for early vs late epochs
    # early_epoch = epochs[len(epochs)//4]  # 25% through training
    # late_epoch = epochs[-1]  # Final epoch
    
    # early_positions = results_by_epoch[early_epoch]['selection_positions']
    # late_positions = results_by_epoch[late_epoch]['selection_positions']
    
    # max_pos = max(max(early_positions, default=0), max(late_positions, default=0))
    # bins = range(max_pos + 2)
    
    # axes[0, 1].hist(early_positions, bins=bins, alpha=0.7, label=f'Epoch {early_epoch}', density=True)
    # axes[0, 1].hist(late_positions, bins=bins, alpha=0.7, label=f'Epoch {late_epoch}', density=True)
    # axes[0, 1].set_xlabel('Position in Context (0=first stone)')
    # axes[0, 1].set_ylabel('Density')
    # axes[0, 1].set_title('Selection Position Distribution')
    # axes[0, 1].legend()
    # axes[0, 1].grid(True, alpha=0.3)
    
    # # Plot 3: Target position vs selection position correlation
    # target_pos = results_by_epoch[late_epoch]['target_positions']
    # select_pos = results_by_epoch[late_epoch]['selection_positions']
    
    # axes[1, 0].scatter(target_pos, select_pos, alpha=0.6)
    # axes[1, 0].plot([0, max_pos], [0, max_pos], 'r--', label='Perfect correlation')
    # axes[1, 0].set_xlabel('Target Position in Context')
    # axes[1, 0].set_ylabel('Selected Position in Context')
    # axes[1, 0].set_title(f'Position Correlation (Epoch {late_epoch})')
    # axes[1, 0].legend()
    # axes[1, 0].grid(True, alpha=0.3)
    
    # # Plot 4: Evolution of selection bias over epochs
    # position_evolution = defaultdict(list)
    # for epoch in epochs[::10]:  # Sample every 10 epochs
    #     positions = results_by_epoch[epoch]['selection_positions']
    #     pos_counts = defaultdict(int)
    #     for pos in positions:
    #         pos_counts[pos] += 1
    #     total = len(positions)
        
    #     for pos in range(max_pos + 1):
    #         proportion = pos_counts[pos] / total if total > 0 else 0
    #         position_evolution[pos].append(proportion)
    
    # sampled_epochs = epochs[::10]
    # for pos in range(min(4, max_pos + 1)):  # Show first 4 positions
    #     axes[1, 1].plot(sampled_epochs, position_evolution[pos], 
    #                    label=f'Position {pos}', marker='o', markersize=3)
    
    # axes[1, 1].set_xlabel('Epoch')
    # axes[1, 1].set_ylabel('Selection Proportion')
    # axes[1, 1].set_title('Evolution of Position Selection Bias')
    # axes[1, 1].legend()
    # axes[1, 1].grid(True, alpha=0.3)
    
    # plt.tight_layout()
    # plt.show()
    
    # return fig























# ----------------------------------------------------------------
# # Load predictions for multiple epochs
# epoch_start = 0  # Define start epoch
# epoch_end = 500 # Define end epoch NOTE: it is possible that the files for the later epochs (were for a bad hyperparameter combination so the plot might appear weird).
# epoch_step = 1   # Define step size (load every Nth epoch)

# predictions_by_epoch = {}

# for epoch in range(epoch_start, epoch_end + 1, epoch_step):
#     # Reformat the epoch_number because the files are saved with epoch numbers like 001, 002, ..., 1000
#     epoch_number = str(epoch).zfill(3)

#     try:
#         inputs_raw = np.load(f'/home/rsaha/projects/{infix}dm_alchemy/src/saved_models/held_out_color_exp/held_out_edges_4/all_graphs/xsmall/decoder/classification/input_features/output_stone_states/shop_1_qhop_1/seed_2/predictions/inputs_classification_epoch_{epoch_number}.npz', allow_pickle=True)['inputs']
#         if epoch_number == '828':
#             continue # This file is corrupted.
#         predictions_raw = np.load(f'/home/rsaha/projects/{infix}dm_alchemy/src/saved_models/held_out_color_exp/held_out_edges_4/all_graphs/xsmall/decoder/classification/input_features/output_stone_states/shop_1_qhop_1/seed_2/predictions/predictions_classification_epoch_{epoch_number}.npz', allow_pickle=True)['predictions']
#         targets_raw = np.load(f'/home/rsaha/projects/{infix}dm_alchemy/src/saved_models/held_out_color_exp/held_out_edges_4/all_graphs/xsmall/decoder/classification/input_features/output_stone_states/shop_1_qhop_1/seed_2/predictions/targets_classification_epoch_{epoch_number}.npz', allow_pickle=True)['targets']
        
#         # Store predictions for this epoch
#         predictions_by_epoch[epoch_number] = predictions_raw.tolist()
        
#         print(f"Loaded epoch {epoch_number}: {len(predictions_raw)} predictions")
        
#     except FileNotFoundError:
#         print(f"Warning: Files for epoch {epoch_number} not found, skipping...")
#         continue

# # Use the first successfully loaded epoch to create data_with_predictions
# first_epoch = min(predictions_by_epoch.keys())
# print(f"\nUsing epoch {first_epoch} data for input/target structure.")
# inputs_raw = np.load(f'/home/rsaha/projects/{infix}dm_alchemy/src/saved_models/held_out_color_exp/held_out_edges_4/all_graphs/xsmall/decoder/classification/input_features/output_stone_states/shop_1_qhop_1/seed_2/predictions/inputs_classification_epoch_{first_epoch}.npz', allow_pickle=True)['inputs']
# targets_raw = np.load(f'/home/rsaha/projects/{infix}dm_alchemy/src/saved_models/held_out_color_exp/held_out_edges_4/all_graphs/xsmall/decoder/classification/input_features/output_stone_states/shop_1_qhop_1/seed_2/predictions/targets_classification_epoch_{first_epoch}.npz', allow_pickle=True)['targets']

# # Flatten inputs from (39, 32, 181) to (1240, 181)
# inputs_flattened = np.vstack(inputs_raw)
# print(f"\nInputs shape: {inputs_flattened.shape}")
# print(f"Targets shape: {targets_raw.shape}")

# # Create data structure matching what analyze_model_selection_behavior expects
# data_with_predictions = []
# for i in range(len(targets_raw)):
#     sample = {
#         'encoder_input_ids': inputs_flattened[i].tolist(),
#         'target_class_id': int(targets_raw[i])
#     }
#     data_with_predictions.append(sample)

# print(f"\nTotal validation samples: {len(data_with_predictions)}")
# print(f"Loaded {len(predictions_by_epoch)} epochs: {sorted(predictions_by_epoch.keys())}")

# print("\nExample sample:")
# print(f"  Input shape: {len(data_with_predictions[0]['encoder_input_ids'])}")
# print(f"  Target class: {data_with_predictions[0]['target_class_id']}")
# # print(f"  Predicted class: {predictions_by_epoch['075'][0]}")


# epoch_accuracies, epoch_within_class_accuracies, other_half_accuracies = analyze_half_chemistry_behaviour(
#     data_with_predictions, vocab, vocab['stone_state_to_id'], predictions_by_epoch
# )








# # # Run the analysis
# # print("\n" + "="*60)
# # print("Running selection behavior analysis...")
# # print("="*60)
# results = analyze_model_selection_behavior(
#     data_with_predictions, 
#     vocab, 
#     vocab['stone_state_to_id'], 
#     predictions_by_epoch
# )

# # Get the 'predicted_in_context_accuracy' for each epoch for plotting
# epoch_accuracies = [results[epoch]['predicted_in_context_accuracy'] for epoch in sorted(results.keys())]
# # epoch_within_class_accuracies = [results[epoch]['recall'] for epoch in sorted(results.keys())]
# # other_half_accuracies = [results[epoch]['accuracy'] for epoch


# # # Plot half-chemistry accuracies over epochs
# plt.figure(figsize=(10, 5))
# plt.plot(range(len(epoch_accuracies)), epoch_accuracies , label='Half-Chemistry Accuracy')
# # plt.plot(range(len(epoch_within_class_accuracies)), epoch_within_class_accuracies , label='Within-Class Accuracy')
# # plt.plot(range(len(other_half_accuracies)), other_half_accuracies , label='Other Half-Chemistry Accuracy')
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.title('Predicted Stone in Context Accuracy Over Epochs')
# plt.legend()
# plt.grid(True, alpha=0.3)
# plt.savefig('predicted_in_context.png')
# # Print results
# for epoch, result in results.items():
#     print(f"\n--- Epoch {epoch} Results ---")
#     print(f"Accuracy: {result['accuracy']:.4f} ({result['target_selected']}/{result['total_samples']})")
#     print(f"Avg context stones: {np.mean(result['num_context_stones']):.2f}")
    
#     # Selection position distribution
#     if result['selection_positions']:
#         from collections import Counter
#         pos_counts = Counter(result['selection_positions'])
#         print(f"\nSelection position distribution:")
#         for pos in sorted(pos_counts.keys()):
#             count = pos_counts[pos]
#             pct = 100 * count / len(result['selection_positions'])
#             print(f"  Position {pos}: {count:4d} ({pct:5.2f}%)")
    
#     # Target position distribution
#     if result['target_positions']:
#         target_pos_counts = Counter(result['target_positions'])
#         print(f"\nTarget position distribution:")
#         for pos in sorted(target_pos_counts.keys()):
#             count = target_pos_counts[pos]
#             pct = 100 * count / len(result['target_positions'])
#             print(f"  Position {pos}: {count:4d} ({pct:5.2f}%)")

# # Plot results
# print("\nGenerating visualization...")
# plot_selection_analysis(results)




