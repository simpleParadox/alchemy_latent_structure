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



def analyze_half_chemistry_behaviour(data, vocab, stone_state_to_id, predictions_by_epoch, exp_typ='held_out', hop=2, overlap_potion_count=1):
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
    if exp_typ in ['decomposition', 'held_out']:
        hop = 1

    support_to_query_mappings = {}
    for i, sample in enumerate(data):
        encoder_input_ids = sample['encoder_input_ids']
        target_class_id = sample['target_class_id']
        support = encoder_input_ids[:-(hop + 4)]  # Everything except last 5 tokens. Works for decomposition too.
        # Create a hashable string key
        support_key = tuple(support)
        
        if support_key not in support_to_query_mappings: # This is for the input to the model. This will be the same for all epochs.
            support_to_query_mappings[support_key] = {}
            
            
    input_vocab = vocab['input_word2idx']
    feature_to_id_vocab = {v: k for k, v in input_vocab.items()}
            
    for i, sample in enumerate(data):
        encoder_input_ids = sample['encoder_input_ids']
        target_class_id = sample['target_class_id']
        support = encoder_input_ids[:-(hop + 4)]  # Everything except last 5 tokens. Works for decomposition too.
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
            # For decomposition and held_out experiments.
            query = encoder_input_ids[-5:]    # Last 5 tokens
            query_potion = query[-1]
            query_stones = query[:-1]

        # First check if the query_potion key is already a list. If not, create an empty list.
        if exp_typ == 'composition':
            if query_potion not in support_to_query_mappings[support_key]:
                support_to_query_mappings[support_key][query_potion] = [target_class_id]
            else:
                support_to_query_mappings[support_key][query_potion].append(target_class_id)

        else:

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


    # Create another dict called composition_per_query_support_to_query_mappings that will store for each support_key, and each query_start_stone, the list of the target class ids for each potion combination.

    """
    The following code block stores the mapping from support sets to query stones and potions for composition experiments. For each support key, it organizes the target class IDs based on the query stones and potions for that support key.
    """
    if exp_typ == 'composition':
        composition_per_query_support_to_query_mappings = {}
        for i, sample in enumerate(data):
            encoder_input_ids = sample['encoder_input_ids']
            target_class_id = sample['target_class_id']
            support = encoder_input_ids[:-(hop + 4)]  # Everything except last 5 tokens
            support_key = tuple(support)
            
            # Based on the number of hops, we need to adjust the query parsing. The hops denote the number of potions in the query.
            query = encoder_input_ids[-(hop + 4):] # 4 featuresd + hop potions.
            query_potion = query[-hop:]  # Last hop tokens
            query_stones = query[:-hop]
            
            # Create a string representation of the query potion sequence from the feature_to_id_vocab and join them.
            query_potion_str = ' | '.join([feature_to_id_vocab[token_id] for token_id in query_potion])
            query_potion = query_potion_str
            
            query_stone_str = ' | '.join([feature_to_id_vocab[token_id] for token_id in query_stones])
            
            if support_key not in composition_per_query_support_to_query_mappings:
                composition_per_query_support_to_query_mappings[support_key] = {}
            if query_stone_str not in composition_per_query_support_to_query_mappings[support_key]:
                composition_per_query_support_to_query_mappings[support_key][query_stone_str] = {}
            if query_potion not in composition_per_query_support_to_query_mappings[support_key][query_stone_str]:
                composition_per_query_support_to_query_mappings[support_key][query_stone_str][query_potion] = [target_class_id]
            else:
                composition_per_query_support_to_query_mappings[support_key][query_stone_str][query_potion].append(target_class_id)



        
        # Now, for each query_stone for that support key, we can also store the predictions for each potion combination in a new dict.
        predictions_composition_per_query_support_to_query_mappings = {}
        for epoch in predictions_by_epoch.keys():
            predictions_composition_per_query_support_to_query_mappings[epoch] = {}
            for support_key in composition_per_query_support_to_query_mappings.keys():
                predictions_composition_per_query_support_to_query_mappings[epoch][support_key] = {}
                for query_stone_str in composition_per_query_support_to_query_mappings[support_key].keys():
                    predictions_composition_per_query_support_to_query_mappings[epoch][support_key][query_stone_str] = {}
                    for query_potion in composition_per_query_support_to_query_mappings[support_key][query_stone_str].keys():
                        predictions_composition_per_query_support_to_query_mappings[epoch][support_key][query_stone_str][query_potion] = []


        for epoch, predictions in tqdm(predictions_by_epoch.items(), desc="Organizing composition predictions by support, query stones, and potions"):
            for i, sample in enumerate(data):
                encoder_input_ids = sample['encoder_input_ids']
                target_class_id = sample['target_class_id']
                predicted_class_id = predictions[i]
                support = encoder_input_ids[:-(hop + 4)]  # Everything except last 5 tokens
                support_key = tuple(support)

                # Based on the number of hops, we need to adjust the query parsing. The hops denote the number of potions in the query.
                query = encoder_input_ids[-(hop + 4):] # 4 featuresd + hop potions.
                query_potion = query[-hop:]  # Last hop tokens
                query_stones = query[:-hop]
                # Create a string representation of the query potion sequence from the feature_to_id_vocab and join them.
                query_potion_str = ' | '.join([feature_to_id_vocab[token_id] for token_id in query_potion])
                query_potion = query_potion_str
                query_stone_str = ' | '.join([feature_to_id_vocab[token_id] for token_id in query_stones])

                # Now directly store the predicted class id in the corresponding dict.
                predictions_composition_per_query_support_to_query_mappings[epoch][support_key][query_stone_str][query_potion].append(predicted_class_id) # The dictionary is already created above.

        

        # Now we calculate metrics based on the above predictions and mappings.
        # 1. For each query stone and potion combination, calculate if the prediction was in the support set. The support set is union of all the target class ids for that support key.
        # 2. For each query stone and potion combination, calculate if the prediction was in the possible target class ids for that query stone (across all potion combinations) if the query output was in the support set.

        predicted_in_context_accuracies = []
        predicted_in_context_correct_candidate_accuracies = []
        correct_within_candidates = []
        overlap_metrics_by_epoch = {}  # Initialize here, populate in the main loop

        for pred_epoch in tqdm(predictions_composition_per_query_support_to_query_mappings.keys(), desc="Analyzing composition epochs"):
            correct = 0
            correct_candidate = 0
            total = 0
            predicted_in_context_count = 0

            epoch_preds = predictions_composition_per_query_support_to_query_mappings[pred_epoch]
            
            # Initialize overlap analysis for this epoch
            epoch_overlap_analysis = {}

            for support_key in composition_per_query_support_to_query_mappings.keys():
                # Initialize for this support_key
                if support_key not in epoch_overlap_analysis:
                    epoch_overlap_analysis[support_key] = {}
                    for overlap_count in range(1, hop):
                        epoch_overlap_analysis[support_key][overlap_count] = {}
                
                res = list(composition_per_query_support_to_query_mappings[support_key].values())
                all_target_class_ids = set([v[0] for sublist in res for k, v in sublist.items()])
                assert len(all_target_class_ids) == 8, f"Expected 8 unique target class ids for support {support_key}, got {len(all_target_class_ids)}"

                # For each query stone and potion combination, check the predictions
                for query_stone_str in composition_per_query_support_to_query_mappings[support_key].keys():
                    for query_potion in composition_per_query_support_to_query_mappings[support_key][query_stone_str].keys():
                        
                        pred = epoch_preds[support_key][query_stone_str][query_potion][0]
                        total += 1

                        # Get the set of target_class_ids for this query_stone and ALL potion combinations.
                        possible_target_class_ids_for_this_query_stone = set([v for sublist in composition_per_query_support_to_query_mappings[support_key][query_stone_str].values() for v in sublist])

                        # Calculate standard composition metrics
                        if pred in all_target_class_ids:
                            predicted_in_context_count += 1

                            if pred in possible_target_class_ids_for_this_query_stone:
                                correct_candidate += 1

                                if pred in composition_per_query_support_to_query_mappings[support_key][query_stone_str][query_potion]:
                                    correct += 1

                        # Populate overlap analysis - now organized by support_key
                        potion_tokens = query_potion.split(' | ')
                        
                        for overlap_count in range(1, hop):
                            potion_prefix = ' | '.join(potion_tokens[:overlap_count])

                            # initialize query_stone_str dict if not exists for this support_key and overlap_count
                            if query_stone_str not in epoch_overlap_analysis[support_key][overlap_count]:
                                epoch_overlap_analysis[support_key][overlap_count][query_stone_str] = {}
                            
                            # initialize potion_prefix dict if not exists
                            if potion_prefix not in epoch_overlap_analysis[support_key][overlap_count][query_stone_str]:
                                epoch_overlap_analysis[support_key][overlap_count][query_stone_str][potion_prefix] = {
                                    'reachable_stones': set(),
                                    'full_sequences': [],
                                    'predictions': []
                                }
                            
                            # add reachable stones for this full sequence
                            target_stones = composition_per_query_support_to_query_mappings[support_key][query_stone_str][query_potion]
                            epoch_overlap_analysis[support_key][overlap_count][query_stone_str][potion_prefix]['reachable_stones'].update(target_stones)
                            # add full sequence
                            epoch_overlap_analysis[support_key][overlap_count][query_stone_str][potion_prefix]['full_sequences'].append(query_potion)
                            
                            # add prediction
                            epoch_overlap_analysis[support_key][overlap_count][query_stone_str][potion_prefix]['predictions'].append(pred)


            # calculate overlap metrics for this epoch
            # now calculate per-support_key metrics and aggregated metrics
            epoch_overlap_metrics = {}

            for overlap_count in range(1, hop):
                # aggregated metrics across all support keys
                predicted_reachable_total = 0
                total_overlap_total = 0
                all_predictions_total = []
                all_reachable_total = set()
                
                # NEW: Track per-query-stone accuracies for unweighted averaging
                all_query_stone_accuracies = []
                
                # per-support metrics
                per_support_metrics = {}
                
                # iterate through each support key
                for support_key in epoch_overlap_analysis.keys():
                    predicted_reachable_support = 0
                    total_overlap_support = 0
                    all_predictions_support = []
                    all_reachable_support = set()
                    
                    # NEW: Track per-query-stone accuracies within this support
                    per_query_stone_accuracies = []
                    per_query_stone_metrics = {}
                    
                    # iterate through query stones for this support key
                    for query_stone_str in epoch_overlap_analysis[support_key][overlap_count].keys():
                        # Initialize counters for this specific query stone
                        predicted_reachable_query_stone = 0
                        total_overlap_query_stone = 0
                        all_predictions_query_stone = []
                        all_reachable_query_stone = set()
                        
                        # Now calculate metrics for each prefix under this query stone
                        for potion_prefix, data in epoch_overlap_analysis[support_key][overlap_count][query_stone_str].items():
                            reachable_stones = data['reachable_stones'] # set of reachable stones for this prefix and for this query stone
                            predictions = data['predictions'] # list of predictions for this prefix and for this query stone
                            
                            all_reachable_support.update(reachable_stones) # For this support key, what are all the reachable stones across query stones and prefixes.
                            # This is me writing for clarity: the union of all reachable stones for this support key across all query stones and prefixes. The size should be eight.


                            all_reachable_query_stone.update(reachable_stones) # For this specific query stone, what are all the reachable stones across prefixes.
                            
                            for pred in predictions:
                                total_overlap_query_stone += 1
                                all_predictions_query_stone.append(pred)
                                
                                total_overlap_support += 1
                                all_predictions_support.append(pred)
                                
                                total_overlap_total += 1
                                all_predictions_total.append(pred)
                                
                                if pred in reachable_stones:
                                    predicted_reachable_query_stone += 1
                                    predicted_reachable_support += 1
                                    predicted_reachable_total += 1
                        
                        #  Calculate and store accuracy for THIS query stone
                        query_stone_accuracy = predicted_reachable_query_stone / total_overlap_query_stone if total_overlap_query_stone > 0 else 0.0
                        per_query_stone_accuracies.append(query_stone_accuracy)
                        all_query_stone_accuracies.append(query_stone_accuracy)
                        
                        # Store per-query-stone metrics
                        per_query_stone_metrics[query_stone_str] = {
                            'reachable_accuracy': query_stone_accuracy,
                            'total': total_overlap_query_stone,
                            'predicted_reachable': predicted_reachable_query_stone,
                            'prediction_distribution': Counter(all_predictions_query_stone),
                            'reachable_stones': all_reachable_query_stone,
                            'per_prefix_data': epoch_overlap_analysis[support_key][overlap_count][query_stone_str]
                        }
                    
                    # store per-support metrics
                    per_support_metrics[support_key] = {
                        # OLD: Weighted aggregation (what you're currently plotting)
                        'reachable_accuracy_weighted': predicted_reachable_support / total_overlap_support if total_overlap_support > 0 else 0.0,
                        
                        # NEW: Unweighted per-query-stone average (what you want to plot)
                        'reachable_accuracy_per_query_stone': np.mean(per_query_stone_accuracies) if per_query_stone_accuracies else 0.0,
                        
                        'total': total_overlap_support,
                        'predicted_reachable': predicted_reachable_support,
                        'prediction_distribution': Counter(all_predictions_support),
                        'reachable_stones': all_reachable_support,
                        'per_query_stone_data': epoch_overlap_analysis[support_key][overlap_count],
                        'per_query_stone_metrics': per_query_stone_metrics,
                        
                        # NEW: Store individual query stone accuracies for variance analysis
                        'per_query_stone_accuracies': per_query_stone_accuracies
                    }
                
                # store both aggregated and per-support metrics
                epoch_overlap_metrics[overlap_count] = {
                    # OLD: Weighted aggregation
                    'reachable_accuracy_weighted': predicted_reachable_total / total_overlap_total if total_overlap_total > 0 else 0.0,
                    
                    # NEW: Unweighted per-query-stone average
                    'reachable_accuracy_per_query_stone': np.mean(all_query_stone_accuracies) if all_query_stone_accuracies else 0.0,
                    
                    'total': total_overlap_total,
                    'predicted_reachable': predicted_reachable_total,
                    'prediction_distribution': Counter(all_predictions_total),
                    'reachable_stones': all_reachable_total,
                    'per_support_metrics': per_support_metrics
                }


            # store overlap metrics for this epoch
            overlap_metrics_by_epoch[pred_epoch] = epoch_overlap_metrics

            # Calculate and store standard metrics
            predicted_in_context_accuracy = predicted_in_context_count / total if total > 0 else 0
            predicted_in_context_accuracies.append(predicted_in_context_accuracy)
            
            predicted_in_context_correct_candidate_accuracy = correct_candidate / predicted_in_context_count if predicted_in_context_count > 0 else 0
            predicted_in_context_correct_candidate_accuracies.append(predicted_in_context_correct_candidate_accuracy)

            correct_within_candidate = correct / correct_candidate if correct_candidate > 0 else 0
            correct_within_candidates.append(correct_within_candidate)

        # Return overlap metrics along with existing metrics.
        # import pdb; pdb.set_trace()
        return (predicted_in_context_accuracies, predicted_in_context_correct_candidate_accuracies, 
                correct_within_candidates, overlap_metrics_by_epoch)


    # The following are the for the decomposition experiments.

    # Now for each sample, we can store the prediction in the corresponding support key and query potion.
    for epoch, predictions in tqdm(predictions_by_epoch.items(), desc="Organizing predictions by support and query"):
        for i, sample in enumerate(data):
            encoder_input_ids = sample['encoder_input_ids']
            target_class_id = sample['target_class_id']
            predicted_class_id = predictions[i]
            support = encoder_input_ids[:-(hop + 4)]  # Everything except last 5 tokens
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
            if exp_typ == 'composition':
                support_to_query_per_epoch_predictions[epoch][support_key][query_potion].append(predicted_class_id)

            else:
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
            support = encoder_input_ids[:-(hop + 4)]  # Everything except last 5 tokens
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
            import pdb; pdb.set_trace()

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
                base_file_path = f"/home/rsaha/projects/{infix}dm_alchemy/src/saved_models/complete_graph/fully_shuffled/{scheduler_prefix}xsmall/decoder/classification/input_features/output_stone_states/shop_1_qhop_{hop}/seed_{seed}/predictions"
            
                
                    
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
parser.add_argument('--hop', type=int, choices=[2, 3, 4, 5], default=3,
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


scheduler_prefix = 'scheduler_cosine/' 
# scheduler_prefix = '' 
# seed_values = [2,3,4]
if exp_typ == 'decomposition':
    # NOTE: Do not change this.
    seed_values_2_hop = [2,3,4]
    seed_values_3_hop = [0,1,2,3,4]
    seed_values_4_hop = [2,3,4] # For the 4
    seed_values_5_hop = [1,2,3]
elif exp_typ == 'held_out':
    # NOTE: Do not change this.
    seed_values_2_hop = [2,3,4]
    seed_values_3_hop = [2,3,4]
    seed_values_4_hop = [2,3,4]
    seed_values_5_hop = [2,3,4]

if exp_typ == 'decomposition' or exp_typ == 'held_out':
    seed_values_hop_dict = {
        2: seed_values_2_hop,
        3: seed_values_3_hop,
        4: seed_values_4_hop,
        5: seed_values_5_hop
    }
if exp_typ == 'composition':
    hop_to_epoch_values = {
        2: [0, 200, 400, 499],
        3: [0, 200, 400, 499],
        4: [0, 200, 400, 499],
        5: [0, 200, 400, 499]
    }
    
    seed_values_hop_dict = {
        2: [16, 29],
        3: [0, 16],
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
            "metadata": f"/home/rsaha/projects/{infix}dm_alchemy/src/data/complete_graph_composition_fully_shuffled_balanced_grouped_by_unique_end_state_preprocessed/compositional_chemistry_samples_167424_80_unique_stones_val_shop_1_qhop_{hop}_seed_{seed}_classification_filter_True_input_features_output_stone_states_metadata.json",
            "vocab": f"/home/rsaha/projects/{infix}dm_alchemy/src/data/complete_graph_composition_fully_shuffled_balanced_grouped_by_unique_end_state_preprocessed/compositional_chemistry_samples_167424_80_unique_stones_val_shop_1_qhop_{hop}_seed_{seed}_classification_filter_True_input_features_output_stone_states_vocab.pkl",
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
    print("Running half-chemistry behavior analysis...")
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
    if exp_typ == 'composition':
        predicted_in_context_accuracies, \
        predicted_in_context_correct_candidate_accuracies, \
            correct_within_candidates, overlap_metrics_by_epoch = half_chemistry_results
        
        # Store results for this seed
        seed_results[seed] = {
            'predicted_in_context_accuracies': predicted_in_context_accuracies,
            'predicted_in_context_correct_candidate_accuracies': predicted_in_context_correct_candidate_accuracies,
            'correct_within_candidates': correct_within_candidates,
            'overlap_metrics_by_epoch': overlap_metrics_by_epoch
        }
    else:
        predicted_in_context_accuracies, predicted_in_context_correct_half_accuracies, predicted_in_context_other_half_accuracies, predicted_in_context_correct_half_exact_accuracies, predicted_correct_within_context = half_chemistry_results
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
metrics = ['predicted_in_context_accuracies', 'predicted_in_context_correct_half_accuracies', 'predicted_in_context_other_half_accuracies', 'predicted_in_context_correct_half_exact_accuracies', 'predicted_correct_within_context'] if exp_typ != 'composition' else ['predicted_in_context_accuracies', 'predicted_in_context_correct_candidate_accuracies', 'correct_within_candidates']
for metric in metrics:
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

# Average overlap metrics across seeds
max_epochs = max(len(seed_results[seed]['predicted_in_context_accuracies']) for seed in seed_results.keys())    
averaged_overlap_metrics = {}
for overlap_count in range(1, hop):
    averaged_overlap_metrics[overlap_count] = {
        'reachable_accuracy': [],
        'std_error': []
    }
    
    for epoch_idx in range(max_epochs):
        epoch_accuracies = []
        
        for seed in seed_results.keys():
            overlap_data = seed_results[seed]['overlap_metrics_by_epoch']
            epoch_keys = sorted(overlap_data.keys())
 
            if epoch_idx < len(epoch_keys):
                epoch_key = epoch_keys[epoch_idx]
                if overlap_count in overlap_data[epoch_key]:
                    # NEW: Use per-query-stone-averaged accuracy
                    accuracy = overlap_data[epoch_key][overlap_count]['reachable_accuracy_per_query_stone']
                    epoch_accuracies.append(accuracy)
        
        if epoch_accuracies:
            averaged_overlap_metrics[overlap_count]['reachable_accuracy'].append(np.mean(epoch_accuracies))
            averaged_overlap_metrics[overlap_count]['std_error'].append(
                np.std(epoch_accuracies) / np.sqrt(len(epoch_accuracies))
            )

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
elif exp_typ == 'composition':
    metrics = [
        ('predicted_in_context_accuracies', 'Prediction in support (8 out of 108)'),
        ('predicted_in_context_correct_candidate_accuracies', 'Correct Candidate accuracy'),
        ('correct_within_candidates', 'Within correct candidate Accuracy'),
    ]
linestyles = {'predicted_in_context_accuracies': 'solid', 'predicted_correct_within_context': 'solid'}
colors = {
    2: {'predicted_in_context_accuracies': 'orange', 'predicted_correct_within_context': 'tab:blue'},
    3: {'predicted_in_context_accuracies': 'orange', 'predicted_correct_within_context': 'tab:green'},
    4: {'predicted_in_context_accuracies': 'orange', 'predicted_correct_within_context': 'tab:gray'},
    5: {'predicted_in_context_accuracies': 'orange', 'predicted_correct_within_context': 'tab:red'},
}
colors_composition = {
    2: {'predicted_in_context_accuracies': 'orange', 'predicted_in_context_correct_candidate_accuracies': 'purple', 'correct_within_candidates': 'tab:blue'},
    3: {'predicted_in_context_accuracies': 'orange', 'predicted_in_context_correct_candidate_accuracies': 'purple', 'correct_within_candidates': 'tab:green'},
    4: {'predicted_in_context_accuracies': 'orange', 'predicted_in_context_correct_candidate_accuracies': 'purple', 'correct_within_candidates': 'tab:gray'},
    5: {'predicted_in_context_accuracies': 'orange', 'predicted_in_context_correct_candidate_accuracies': 'purple', 'correct_within_candidates': 'tab:red'}
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

# Create multi-panel figure for 3-hop case
if hop >= 3:
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # Panel A: Existing composition metrics (spans both columns, top row)
    ax1 = fig.add_subplot(gs[0, :])
    epochs = range(len(averaged_results['predicted_in_context_accuracies']))
    
    composition_metrics = [
        ('predicted_in_context_accuracies', 'In-support gating (8/108)', 'tab:blue'),
        ('predicted_in_context_correct_candidate_accuracies', 'Correct candidate (3-4/8)', 'tab:orange'),
        ('correct_within_candidates', 'Exact match (1/3-4)', 'tab:green'),
    ]
    
    for metric, label, color in composition_metrics:
        mean = averaged_results[metric]
        sem = std_errors[metric]
        ax1.plot(epochs, mean, label=label, linewidth=2, color=color)
        ax1.fill_between(epochs, mean - sem, mean + sem, alpha=0.2, color=color)
    
    ax1.set_xlabel('Epoch', fontsize=18)
    ax1.set_ylabel('Accuracy', fontsize=18)
    ax1.set_title(f'A. Standard Composition Metrics ({hop}-hop)', fontsize=20, fontweight='bold', pad=20)
    ax1.legend(fontsize=14, loc='lower right')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1)
    ax1.tick_params(labelsize=14)
    
    # Panel B: Overlap analysis line plot.
    ax2 = fig.add_subplot(gs[1, :])
    
    overlap_colors = {1: 'tab:purple', 2: 'tab:cyan', 3: 'tab:pink', 4: 'tab:brown'}
    
    for overlap_count in range(1, hop):
        if overlap_count in averaged_overlap_metrics:
            mean = averaged_overlap_metrics[overlap_count]['reachable_accuracy']
            sem = averaged_overlap_metrics[overlap_count]['std_error']
            
            label = f'{overlap_count}-potion overlap ({overlap_count}/{hop} potions)'
            color = overlap_colors.get(overlap_count, 'black')
            
            ax2.plot(epochs[:len(mean)], mean, label=label, linewidth=2, color=color, marker='o', markersize=4)
            ax2.fill_between(epochs[:len(mean)], 
                            np.array(mean) - np.array(sem), 
                            np.array(mean) + np.array(sem), 
                            alpha=0.2, color=color)
    
    ax2.set_xlabel('Epoch', fontsize=18)
    ax2.set_ylabel('Reachable Stone Accuracy', fontsize=18)
    ax2.set_title(f'B. Compositional Reasoning: Accuracy vs Potion Overlap', fontsize=20, fontweight='bold', pad=20)
    ax2.legend(fontsize=14, loc='lower right')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)
    ax2.tick_params(labelsize=14)
    
    # Add annotation explaining the metric
    ax2.text(0.02, 0.98, 
             'Higher overlap means more constrained possible outcomes\n'
             'Model should predict only stones reachable via prefix path',
             transform=ax2.transAxes, fontsize=11, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    # Panel C: Detailed breakdown for specific epochs (bottom row)
    # Show early, middle, and late training
    analysis_epochs = [0, len(epochs)//2, len(epochs)-1]
    epoch_labels = ['Early (Epoch 0)', f'Mid (Epoch {analysis_epochs[1]})', f'Late (Epoch {len(epochs)-1})']
    
    for idx, (epoch_idx, epoch_label) in enumerate(zip(analysis_epochs, epoch_labels)):
        ax = fig.add_subplot(gs[2, idx if idx < 2 else 1])
        
        # Get data for this epoch from one seed (for visualization)
        seed = list(seed_results.keys())[0]
        overlap_data = seed_results[seed]['overlap_metrics_by_epoch']
        epoch_keys = sorted(overlap_data.keys())
        
        if epoch_idx < len(epoch_keys):
            epoch_key = epoch_keys[epoch_idx]
            
            # Create grouped bar chart
            overlap_counts = list(range(1, hop))
            reachable_accs = [overlap_data[epoch_key].get(oc, {}).get('reachable_accuracy', 0) 
                             for oc in overlap_counts]
            
            bars = ax.bar(overlap_counts, reachable_accs, color=[overlap_colors.get(oc) for oc in overlap_counts])
            
            # Add value labels on bars
            for bar, acc in zip(bars, reachable_accs):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{acc:.2f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
            
            ax.set_xlabel('Potion Overlap Count', fontsize=14)
            ax.set_ylabel('Reachable Accuracy', fontsize=14)
            ax.set_title(f'C{idx+1}. {epoch_label}', fontsize=16, fontweight='bold')
            ax.set_ylim(0, 1)
            ax.set_xticks(overlap_counts)
            ax.grid(True, alpha=0.3, axis='y')
            ax.tick_params(labelsize=12)
    
    plt.savefig(f'composition_{hop}_hop_comprehensive_overlap_analysis.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'composition_{hop}_hop_comprehensive_overlap_analysis.pdf', bbox_inches='tight')
    print(f"\nSaved comprehensive multi-panel figure for {hop}-hop composition analysis")




