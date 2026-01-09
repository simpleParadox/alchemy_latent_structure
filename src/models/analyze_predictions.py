import json
import pickle
import torch
import numpy as np
from collections import defaultdict, Counter
from tqdm import tqdm
import matplotlib.pyplot as plt
from cluster_profile import cluster
import os
import re

# Load the metadata, data, and vocab.
if cluster == 'vulcan':
    infix = 'def-afyshe-ab/rsaha/'
else:
    # Usually for cirrus but might change later.
    infix = ''

def create_reverse_stone_mapping(stone_state_to_id):
    """Create reverse mapping from class ID to stone state string."""
    return {v: k for k, v in stone_state_to_id.items()}


def extract_support_transitions(support_ids, input_vocab, stone_state_to_id):
    """
    Parse transitions from the support segment.

    Pattern (in tokens):
        input_4_feature_tokens  potion  <io>  output_4_feature_tokens  <item_sep> ...

    Returns:
        transitions: list of (input_class_id, output_class_id)
        stone_ids_in_support: set of all stone class_ids that appear as input/output
    """
    idx2word = {v: k for k, v in input_vocab.items()}
    io_tok = '<io>'
    item_sep_tok = '<item_sep>'
    io_id = input_vocab[io_tok]
    item_sep_id = input_vocab[item_sep_tok]

    tokens = [idx2word[t] for t in support_ids]
    transitions = []
    stone_ids_in_support = set()

    i = 0
    while i + 4 <= len(tokens):
        # Need: input_4_feat, potion, <io>, output_4_feat
        if i + 4 + 1 + 1 + 4 > len(tokens):
            break

        in_feat = tokens[i:i+4]
        potion = tokens[i+4]
        sep = tokens[i+5]

        if sep != io_tok:
            # Not matching the pattern, shift by 1 and continue
            i += 1
            raise ValueError(f"Expected <io> token but got {sep} at position {i+5}")

        out_feat = tokens[i+6:i+10]

        # Build stone state strings
        in_color, in_size, in_round, in_reward = in_feat
        out_color, out_size, out_round, out_reward = out_feat

        in_state_str = f"{{color: {in_color}, size: {in_size}, roundness: {in_round}, reward: {in_reward}}}"
        out_state_str = f"{{color: {out_color}, size: {out_size}, roundness: {out_round}, reward: {out_reward}}}"

        in_id = stone_state_to_id.get(in_state_str)
        out_id = stone_state_to_id.get(out_state_str)

        if in_id is not None:
            stone_ids_in_support.add(in_id)
        if out_id is not None:
            stone_ids_in_support.add(out_id)

        if in_id is not None and out_id is not None:
            transitions.append((in_id, out_id))

        # Move i past this transition: 4 + 1 + 1 + 4 = 10 tokens
        i += 10
        # If there is an <item_sep>, skip it
        if i < len(tokens) and tokens[i] == item_sep_tok:
            i += 1

    return transitions, stone_ids_in_support



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


def analyze_non_support_transition_behavior(data, vocab, stone_state_to_id, predictions_by_epoch, exp_typ='held_out', hop=4):
    """
    Analyze non-support transition prediction behavior for the 4-edge held-out experiment.
    
    For each chemistry (support_key) and each query stone:
      1. Find stones in the support for which NO transition from that query exists in the support set.
      2. Measure model behavior when predictions fall among these non-support stones.
    
    We additionally group metrics by query reward ('3', '-3', '1', '-1').
    """
    assert exp_typ == 'held_out', "non-support transition analysis is designed for held_out exp."

    reverse_stone_mapping = create_reverse_stone_mapping(stone_state_to_id)
    input_vocab = vocab['input_word2idx']
    feature_to_id_vocab = {v: k for k, v in input_vocab.items()}

    # ----------------------------------------------------------
    # 1) Precompute transitions and neighbor sets per chemistry
    # ----------------------------------------------------------
    # neighbors_per_chemistry[support_key][input_id] = set(output_ids)
    neighbors_per_chemistry = {}
    # support_stones_per_chemistry[support_key] = set(stone_ids in this support)
    support_stones_per_chemistry = {}

    for sample in data:
        encoder_input_ids = sample['encoder_input_ids']
        support = encoder_input_ids[:-(hop + 4)]  # everything except last 5 tokens
        support_key = tuple(support)

        if support_key in neighbors_per_chemistry:
            continue  # already processed this chemistry

        transitions, stone_ids_in_support = extract_support_transitions(support, input_vocab, stone_state_to_id)

        support_stones_per_chemistry[support_key] = stone_ids_in_support

        neighbors = defaultdict(set)
        for in_id, out_id in transitions:
            neighbors[in_id].add(out_id)
        neighbors_per_chemistry[support_key] = neighbors

    # ----------------------------------------------------------
    # 2) Initialize metrics structures
    # ----------------------------------------------------------
    non_support_metrics = {
        epoch: {
            reward: {
                'total_non_support_preds': 0,
                'correct_within_non_support': 0,
                'total_queries': 0,  # denominator for P(pred in non-support | query reward)
                'total_in_support_preds': 0  # denominator for P(pred in non-support | pred in support, query reward)
            }
            for reward in ['3', '-3', '1', '-1']
        }
        for epoch in predictions_by_epoch.keys()
    }

    # ----------------------------------------------------------
    # 3) Iterate over epochs and samples
    # ----------------------------------------------------------
    for epoch, predictions in tqdm(predictions_by_epoch.items(), desc="Analyzing non-support transitions"):
        for i, sample in enumerate(data):
            encoder_input_ids = sample['encoder_input_ids']
            target_class_id = sample['target_class_id']
            predicted_class_id = predictions[i]

            # Support / chemistry
            support = encoder_input_ids[:-(hop + 4)]
            support_key = tuple(support)

            # Decode query stone (last 5 tokens = 4 features + potion)
            query = encoder_input_ids[-5:]
            query_feat_ids = query[:-1]
            query_features = [feature_to_id_vocab[tok_id] for tok_id in query_feat_ids]
            q_color, q_size, q_round, q_reward = query_features
            query_state_str = f"{{color: {q_color}, size: {q_size}, roundness: {q_round}, reward: {q_reward}}}"
            query_stone_id = stone_state_to_id[query_state_str]
            query_reward = q_reward  # e.g., '3', '-1', etc.

            # Update query count for this reward
            if query_reward in non_support_metrics[epoch]:
                non_support_metrics[epoch][query_reward]['total_queries'] += 1

            support_stones = support_stones_per_chemistry[support_key]
            neighbors = neighbors_per_chemistry[support_key].get(query_stone_id, set())

            # Non-support set for this query: stones in support with no q→* transition, excluding q itself
            non_support_targets = support_stones - neighbors - {query_stone_id}


            # Only consider queries for which we are tracking this reward
            if query_reward in non_support_metrics[epoch]:
               # First: did the model predict within the support set at all?
               if predicted_class_id in support_stones:
                   non_support_metrics[epoch][query_reward]['total_in_support_preds'] += 1

                   # Within those, check if the prediction is in the non-support subset
                   if predicted_class_id in non_support_targets:
                       non_support_metrics[epoch][query_reward]['total_non_support_preds'] += 1
                       if predicted_class_id == target_class_id:
                           non_support_metrics[epoch][query_reward]['correct_within_non_support'] += 1

    # ----------------------------------------------------------
    # 4) Convert counts to per-epoch accuracies
    # ----------------------------------------------------------
    non_support_accuracies = {
        reward: {
            'p_pred_in_non_support': [],
            'p_correct_given_non_support': []
        }
        for reward in ['3', '-3', '1', '-1']
    }

    for epoch in sorted(predictions_by_epoch.keys()):
        for reward in ['3', '-3', '1', '-1']:
            stats = non_support_metrics[epoch][reward]
            total_q = stats['total_queries']
            total_non = stats['total_non_support_preds']
            correct_non = stats['correct_within_non_support']
            total_in_support = stats['total_in_support_preds']

            # P(pred in non-support | pred in support, query reward)
            p_in_non = total_non / total_in_support if total_in_support > 0 else 0.0
            # P(correct | pred in non-support)
            p_correct = correct_non / total_non if total_non > 0 else 0.0

            non_support_accuracies[reward]['p_pred_in_non_support'].append(p_in_non)
            non_support_accuracies[reward]['p_correct_given_non_support'].append(p_correct)

    return non_support_accuracies, non_support_metrics


# def analyze_adjacency_behavior(data, vocab, stone_state_to_id, predictions_by_epoch, exp_typ='held_out', hop=4):
#     """
#     Analyze adjacency-based prediction behavior for the 4-edge held-out experiment.
    
#     For each query stone (grouped by reward feature), calculate:
#     1. Within adjacent (reachable by reward) accuracy: P(pred in reachable_rewards | pred in support)
#     2. Correct target within adjacent: P(exact target | pred in reachable_rewards)
#     3. [NEW] Within connected neighbors accuracy: P(pred in connected_neighbors | pred in support)
    
#     Reachable stones definition:
#     - Reachable by reward: Stones having the reward values expected for neighbors.
#     - Connected neighbors: Stones actually linked to the query stone in the support graph.
#     """
    
#     reverse_stone_mapping = create_reverse_stone_mapping(stone_state_to_id)
#     input_vocab = vocab['input_word2idx']
#     feature_to_id_vocab = {v: k for k, v in input_vocab.items()}
    
#     # Define reachable rewards for each query stone reward
#     # This maps query reward -> list of reachable stone rewards
#     reachable_reward_mapping = {
#         '3': ['1', '1', '1'],      # 3 stones with +1
#         '-3': ['-1', '-1', '-1'],      # 3 stones with -1
#         '1': ['3', '-1', '-1', '-1'], # 1 with +3, 3 with -1
#         '-1': ['-3', '1', '1', '1']  # 1 with -3, 3 with +1
#     }
    
#     # Initialize tracking for each epoch and query reward
#     adjacency_metrics = {
#         epoch: {
#             reward: {
#                 'total_in_support': 0,
#                 'in_support_and_reachable': 0, # Reachable by reward value
#                 'total_reachable': 0,
#                 'reachable_and_correct': 0,
#                 'in_support_and_connected': 0, # Actually connected in graph
#                 'in_support_and_true_adjacent': 0, # [NEW] Connected in support OR target
#             } for reward in reachable_reward_mapping.keys()
#         } for epoch in predictions_by_epoch.keys()
#     }
    
#     # Pre-compute graph structure for each support set to find connected neighbors
#     # We can reuse logic similar to analyze_non_support_transition_behavior
#     neighbors_per_chemistry = defaultdict(lambda: defaultdict(set))
    
#     # We need to iterate data once to build graphs if we want to be efficient, 
#     # or we can do it inside the loop if dataset is small enough. 
#     # Given the structure, let's do it on the fly or pre-compute. 
#     # Let's pre-compute for safety and clarity.
#     print("Pre-computing chemistry graphs...")
#     for sample in data:
#         encoder_input_ids = sample['encoder_input_ids']
#         support = encoder_input_ids[:-(hop + 4)]
#         support_key = tuple(support)
        
#         if support_key not in neighbors_per_chemistry:
#             # Extract transitions: (start_stone_id, end_stone_id)
#             # extract_support_transitions returns (transitions_list, stone_ids_set)
#             transitions_list, _ = extract_support_transitions(list(support), input_vocab, stone_state_to_id)
#             for start_id, end_id in transitions_list:
#                 if start_id is not None and end_id is not None:
#                     neighbors_per_chemistry[support_key][start_id].add(end_id)


#     for epoch, predictions in tqdm(predictions_by_epoch.items(), desc="Analyzing adjacency behavior"):
#         for i, sample in enumerate(data):
#             encoder_input_ids = sample['encoder_input_ids']
#             target_class_id = sample['target_class_id']
#             predicted_class_id = predictions[i]
            
#             # Extract query stone reward (second-to-last token in query)
#             query = encoder_input_ids[-5:]  # Last 5 tokens
#             query_stone_reward = feature_to_id_vocab[query[-2]]
            
#             # Decode query stone ID to find its specific neighbors
#             query_feat_ids = query[:-1]
#             query_features = [feature_to_id_vocab[tok_id] for tok_id in query_feat_ids]
#             q_color, q_size, q_round, q_reward = query_features
#             query_state_str = f"{{color: {q_color}, size: {q_size}, roundness: {q_round}, reward: {q_reward}}}"
#             query_stone_id = stone_state_to_id.get(query_state_str)

#             # Get support set (all 8 stones in the chemistry)
#             support = encoder_input_ids[:-(hop + 4)]
#             support_key = tuple(support)
            
#             stone_states_in_input = parse_stone_states_from_input(
#                 encoder_input_ids, input_vocab, stone_state_to_id
#             )
#             support_stone_ids = set([
#                 stone_id for _, stone_id in stone_states_in_input 
#                 if stone_id is not None
#             ])

#             assert len(support_stone_ids) == 8, f"Expected 8 stones in support, got {len(support_stone_ids)}"
            
#             # Get predicted stone info
#             if predicted_class_id not in reverse_stone_mapping:
#                 continue
                
#             predicted_stone_state_str = reverse_stone_mapping[predicted_class_id]
#             predicted_reward = re.search(r'reward: (\+?-?\d+)', predicted_stone_state_str).group(1)
            
#             # Determine reachable rewards for this query stone
#             reachable_rewards = reachable_reward_mapping[query_stone_reward]
            
#             # Determine actual connected neighbors for this specific query stone
#             connected_neighbors = neighbors_per_chemistry[support_key].get(query_stone_id, set())

#             true_adjacent_set = connected_neighbors.union({target_class_id})
#             import pdb; pdb.set_trace()

#             # Metric 1: Check if prediction is in support
#             if predicted_class_id in support_stone_ids:
#                 adjacency_metrics[epoch][query_stone_reward]['total_in_support'] += 1
                
#                 # Metric 2: Check if prediction is reachable (adjacent or same reward in other half)
#                 if predicted_reward in reachable_rewards:
#                     adjacency_metrics[epoch][query_stone_reward]['in_support_and_reachable'] += 1
#                     adjacency_metrics[epoch][query_stone_reward]['total_reachable'] += 1
                    
#                     # Metric 3: Check if it's the exact correct target
#                     if predicted_class_id == target_class_id:
#                         adjacency_metrics[epoch][query_stone_reward]['reachable_and_correct'] += 1
                
#                 # Metric 4: Check if prediction is an actual connected neighbor (support only)
#                 if predicted_class_id in connected_neighbors:
#                     adjacency_metrics[epoch][query_stone_reward]['in_support_and_connected'] += 1

#                 # Metric 5: Check if prediction is in true_adjacent_set
#                 if predicted_class_id in true_adjacent_set:
#                     adjacency_metrics[epoch][query_stone_reward]['in_support_and_true_adjacent'] += 1
    
#     # Calculate accuracies over epochs
#     adjacency_accuracies = {
#         reward: {
#             'within_reachable_acc': [],
#             'correct_within_reachable_acc': [],
#             'within_connected_acc': [],
#             'within_true_adjacent_acc': [] # [NEW]
#         } for reward in reachable_reward_mapping.keys()
#     }
    
#     for epoch in sorted(adjacency_metrics.keys()):
#         for reward in reachable_reward_mapping.keys():
#             metrics = adjacency_metrics[epoch][reward]
            
#             # Accuracy 1: P(reachable | in-support)
#             if metrics['total_in_support'] > 0:
#                 reachable_acc = metrics['in_support_and_reachable'] / metrics['total_in_support']
#                 connected_acc = metrics['in_support_and_connected'] / metrics['total_in_support']
#                 true_adjacent_acc = metrics['in_support_and_true_adjacent'] / metrics['total_in_support'] # [NEW]
#             else:
#                 reachable_acc = 0
#                 connected_acc = 0
#                 true_adjacent_acc = 0
#             adjacency_accuracies[reward]['within_reachable_acc'].append(reachable_acc)
#             adjacency_accuracies[reward]['within_connected_acc'].append(connected_acc)
#             adjacency_accuracies[reward]['within_true_adjacent_acc'].append(true_adjacent_acc) # [NEW]
            
            
#             # Accuracy 2: P(exact target | reachable)
#             if metrics['total_reachable'] > 0:
#                 correct_acc = metrics['reachable_and_correct'] / metrics['total_reachable']
#             else:
#                 correct_acc = 0
#             adjacency_accuracies[reward]['correct_within_reachable_acc'].append(correct_acc)
    
#     return adjacency_accuracies

# def analyze_adjacency_behavior(data, vocab, stone_state_to_id, predictions_by_epoch):
      ##  NOTE: This is a variant where the metrics are being calculated at the end.
#     """
#     Analyze adjacency-based prediction behavior for the 4-edge held-out experiment.
#     """
#     hop = 1
    
#     reverse_stone_mapping = create_reverse_stone_mapping(stone_state_to_id)
#     input_vocab = vocab['input_word2idx']
#     feature_to_id_vocab = {v: k for k, v in input_vocab.items()}
    
#     # Define reachable rewards for each query stone reward
#     # This maps query reward -> set of reachable stone rewards
#     # Using sets for O(1) lookup
#     reachable_reward_mapping = {
#         '3': {'1'},          # +15 connects to +1
#         '-3': {'-1'},        # -3 connects to -1
#         '1': {'3', '-1'},    # +1 connects to +15 and -1
#         '-1': {'-3', '1'}    # -1 connects to -3 and +1
#     }
    
#     # Initialize tracking for each epoch and query reward
#     adjacency_metrics = {
#         epoch: {
#             reward: {
#                 'total_in_support': 0,
#                 'in_support_and_reachable': 0, # Reachable by reward value (Stone-based)
#                 'total_reachable': 0,
#                 'reachable_and_correct': 0,
#                 'in_support_and_connected': 0, # Actually connected in graph
#                 'in_support_and_true_adjacent': 0, # Connected in support OR target
#             } for reward in reachable_reward_mapping.keys()
#         } for epoch in predictions_by_epoch.keys()
#     }
    
#     # Pre-compute graph structure
#     neighbors_per_chemistry = defaultdict(lambda: defaultdict(set))
    
#     print("Pre-computing chemistry graphs...")
#     for sample in data:
#         encoder_input_ids = sample['encoder_input_ids']
#         support = encoder_input_ids[:-5]
#         support_key = tuple(support)
        
#         if support_key not in neighbors_per_chemistry:
#             transitions_list, _ = extract_support_transitions(list(support), input_vocab, stone_state_to_id)
#             for start_id, end_id in transitions_list:
#                 if start_id is not None and end_id is not None:
#                     neighbors_per_chemistry[support_key][start_id].add(end_id)
#     # import pdb; pdb.set_trace()

#     for epoch, predictions in tqdm(predictions_by_epoch.items(), desc="Analyzing adjacency behavior"):
#         for i, sample in enumerate(data):
#             encoder_input_ids = sample['encoder_input_ids']
#             target_class_id = sample['target_class_id']
#             predicted_class_id = predictions[i]
            
#             # Extract query stone reward
#             query = encoder_input_ids[-5:]
#             query_stone_reward = feature_to_id_vocab[query[-2]]
            
#             # Decode query stone ID
#             query_feat_ids = query[:-1]
#             query_features = [feature_to_id_vocab[tok_id] for tok_id in query_feat_ids]
#             q_color, q_size, q_round, q_reward = query_features
#             query_state_str = f"{{color: {q_color}, size: {q_size}, roundness: {q_round}, reward: {q_reward}}}"
#             query_stone_id = stone_state_to_id.get(query_state_str)

#             # Get support set
#             support = encoder_input_ids[:-(hop + 4)]
#             support_key = tuple(support)
            
#             # Parse all stones in the support to map IDs to Rewards
#             stone_states_in_input = parse_stone_states_from_input(
#                 encoder_input_ids, input_vocab, stone_state_to_id
#             )
            
#             support_stone_ids = set()
#             stone_id_to_reward = {}
            
#             for s_str, s_id in stone_states_in_input:
#                 if s_id is not None:
#                     support_stone_ids.add(s_id)
#                     # Extract reward from the state string to identify "Reachable" candidates
#                     # s_str format example: "{color: CYAN, ..., reward: 3}"
#                     r_match = re.search(r'reward: (\+?-?\d+)', s_str)
#                     if r_match:
#                         # Normalize to integer, then back to string ("+1" and "1" both -> "1")
#                         stone_id_to_reward[s_id] = str(int(r_match.group(1)))
#             assert len(support_stone_ids) == 8, f"Expected 8 stones in support, got {len(support_stone_ids)}"

#             # Determine reachable stones (Set of IDs)
#             # A stone is reachable if it is in the support AND its reward is in the allowed set
#             allowed_rewards = reachable_reward_mapping.get(query_stone_reward)
#             # Normalize allowed rewards as well
#             normalized_allowed_rewards = {str(int(r)) for r in allowed_rewards}

#             reachable_stones_set = {
#                 sid for sid in support_stone_ids 
#                 if stone_id_to_reward.get(sid) in normalized_allowed_rewards
#             }
#             # Cheack if the reachable_stones_set is always a subset of support_stone_ids
#             assert reachable_stones_set.issubset(support_stone_ids), "Reachable stones must be subset of support stones."

#             # Determine actual connected neighbors
#             connected_neighbors = neighbors_per_chemistry[support_key].get(query_stone_id, set())
#             assert len(connected_neighbors) == 2, f"Expected 2 connected neighbors, got {len(connected_neighbors)}"
#             true_adjacent_set = connected_neighbors.union({target_class_id})

#             # Optional sanity checks
#             if query_stone_reward in ['3', '-3']:
#                 assert len(reachable_stones_set) == 3, (
#                     f"Expected 3 reachable stones for reward {query_stone_reward}, "
#                     f"got {len(reachable_stones_set)}"
#                 )
#                 assert len(true_adjacent_set) == 3, (
#                     f"Expected 3 true-adjacent stones for reward {query_stone_reward}, "
#                     f"got {len(true_adjacent_set)}"
#                 )
#             if query_stone_reward in ['1', '-1']:
#                 assert len(reachable_stones_set) == 4, (
#                     f"Expected 4 reachable stones for reward {query_stone_reward}, "
#                     f"got {len(reachable_stones_set)}"
#                 )
#                 assert len(true_adjacent_set) == 3, (
#                     f"Expected 4 true-adjacent stones for reward {query_stone_reward}, "
#                     f"got {len(true_adjacent_set)}"
#                 )
#             # Do assertions for connected neighbors too.
#             assert len(connected_neighbors) == len(true_adjacent_set) - 1, f"Connected neighbors should be one less than true adjacent set."


                        
#             # if epoch == '190':
#             #     import pdb; pdb.set_trace()

#             # Metric 1: Check if prediction is in support
#             if predicted_class_id in support_stone_ids:
#                 adjacency_metrics[epoch][query_stone_reward]['total_in_support'] += 1
                
#                 # Metric 2: Check if prediction is reachable (Stone-based check)
#                 if predicted_class_id in reachable_stones_set:
#                     adjacency_metrics[epoch][query_stone_reward]['in_support_and_reachable'] += 1
#                     adjacency_metrics[epoch][query_stone_reward]['total_reachable'] += 1
                    
#                     # Metric 3: Check if it's the exact correct target
#                     if predicted_class_id == target_class_id:
#                         adjacency_metrics[epoch][query_stone_reward]['reachable_and_correct'] += 1
                
#                 # Metric 4: Check if prediction is an actual connected neighbor
#                 if predicted_class_id in connected_neighbors:
#                     assert len(connected_neighbors) == 2, f"Expected 2 connected neighbors, got {len(connected_neighbors)}"
#                     adjacency_metrics[epoch][query_stone_reward]['in_support_and_connected'] += 1

#                 # Metric 5: Check if prediction is in true_adjacent_set
#                 if predicted_class_id in true_adjacent_set:
#                     adjacency_metrics[epoch][query_stone_reward]['in_support_and_true_adjacent'] += 1
    
#     # Calculate accuracies over epochs
#     adjacency_accuracies = {
#         reward: {
#             'within_reachable_acc': [],
#             'correct_within_reachable_acc': [],
#             'within_connected_acc': [],
#             'within_true_adjacent_acc': [],
#             'within_connected_in_reachable_acc': []
#         } for reward in reachable_reward_mapping.keys()
#     }

#     # import pdb; pdb.set_trace()

#     for epoch in sorted(adjacency_metrics.keys()):
#         for reward in reachable_reward_mapping.keys():
#             metrics = adjacency_metrics[epoch][reward]
            
#             if metrics['total_in_support'] > 0:
#                 reachable_acc = metrics['in_support_and_reachable'] / metrics['total_in_support']
#                 connected_acc = metrics['in_support_and_connected'] / metrics['total_in_support']
#                 true_adjacent_acc = metrics['in_support_and_true_adjacent'] / metrics['total_in_support']
#                 within_connected_in_reachable = metrics['in_support_and_connected'] / metrics['in_support_and_reachable'] if metrics['in_support_and_reachable'] > 0 else 0.0
#             else:
#                 reachable_acc = 0
#                 connected_acc = 0
#                 true_adjacent_acc = 0
            
#             adjacency_accuracies[reward]['within_reachable_acc'].append(reachable_acc)
#             adjacency_accuracies[reward]['within_connected_acc'].append(connected_acc)
#             adjacency_accuracies[reward]['within_true_adjacent_acc'].append(true_adjacent_acc)
#             adjacency_accuracies[reward]['within_connected_in_reachable_acc'].append(within_connected_in_reachable)
            
#             if metrics['total_reachable'] > 0:
#                 correct_acc = metrics['reachable_and_correct'] / metrics['total_reachable']
#             else:
#                 correct_acc = 0
#             adjacency_accuracies[reward]['correct_within_reachable_acc'].append(correct_acc)
    
#     return adjacency_accuracies

def analyze_adjacency_behavior(data, vocab, stone_state_to_id, predictions_by_epoch):
    """
    Analyze adjacency-based prediction behavior for the 4-edge held-out experiment.
    """
    hop = 1
    
    reverse_stone_mapping = create_reverse_stone_mapping(stone_state_to_id)
    input_vocab = vocab['input_word2idx']
    feature_to_id_vocab = {v: k for k, v in input_vocab.items()}
    
    # Define reachable rewards for each query stone reward
    # This maps query reward -> set of reachable stone rewards
    # Using sets for O(1) lookup
    reachable_reward_mapping = {
        '3': {'1'},          # +15 connects to +1
        '-3': {'-1'},        # -3 connects to -1
        '1': {'3', '-1'},    # +1 connects to +15 and -1
        '-1': {'-3', '1'}    # -1 connects to -3 and +1
    }
    
    # Initialize result containers (lists)
    adjacency_accuracies = {
        reward: {
            'within_reachable_acc': [],
            'correct_within_reachable_acc': [],
            'within_connected_acc': [],
            'within_true_adjacent_acc': [],
            'within_connected_in_reachable_acc': []
        } for reward in reachable_reward_mapping.keys()
    }
    
    # Pre-compute graph structure
    neighbors_per_chemistry = defaultdict(lambda: defaultdict(set))
    
    print("Pre-computing chemistry graphs...")
    for sample in data:
        encoder_input_ids = sample['encoder_input_ids']
        support = encoder_input_ids[:-5]
        support_key = tuple(support)
        
        if support_key not in neighbors_per_chemistry:
            transitions_list, _ = extract_support_transitions(list(support), input_vocab, stone_state_to_id)
            for start_id, end_id in transitions_list:
                if start_id is not None and end_id is not None:
                    neighbors_per_chemistry[support_key][start_id].add(end_id)
    
    # Sort epochs to ensure metrics are appended in correct order
    sorted_epochs = sorted(predictions_by_epoch.keys())

    for epoch in tqdm(sorted_epochs, desc="Analyzing adjacency behavior"):
        predictions = predictions_by_epoch[epoch]
        
        # Initialize counters for this specific epoch
        epoch_metrics = {
            reward: {
                'total_count': 0,
                'total_in_support': 0,
                'in_support_and_reachable': 0, # Reachable by reward value (Stone-based)
                'total_reachable': 0,
                'reachable_and_correct': 0,
                'in_support_and_connected': 0, # Actually connected in graph
                'in_support_and_true_adjacent': 0, # Connected in support OR target
            } for reward in reachable_reward_mapping.keys()
        }

        for i, sample in enumerate(data):
            encoder_input_ids = sample['encoder_input_ids']
            target_class_id = sample['target_class_id']
            predicted_class_id = predictions[i]
            
            # Extract query stone reward
            query = encoder_input_ids[-5:]
            query_stone_reward = feature_to_id_vocab[query[-2]]
            
            # Decode query stone ID
            query_feat_ids = query[:-1]
            query_features = [feature_to_id_vocab[tok_id] for tok_id in query_feat_ids]
            q_color, q_size, q_round, q_reward = query_features
            query_state_str = f"{{color: {q_color}, size: {q_size}, roundness: {q_round}, reward: {q_reward}}}"
            query_stone_id = stone_state_to_id.get(query_state_str)

            # Get support set
            support = encoder_input_ids[:-(hop + 4)]
            support_key = tuple(support)
            
            # Parse all stones in the support to map IDs to Rewards
            stone_states_in_input = parse_stone_states_from_input(
                encoder_input_ids, input_vocab, stone_state_to_id
            )
            
            support_stone_ids = set()
            stone_id_to_reward = {}
            
            for s_str, s_id in stone_states_in_input:
                if s_id is not None:
                    support_stone_ids.add(s_id)
                    # Extract reward from the state string to identify "Reachable" candidates
                    # s_str format example: "{color: CYAN, ..., reward: 3}"
                    r_match = re.search(r'reward: (\+?-?\d+)', s_str)
                    if r_match:
                        # Normalize to integer, then back to string ("+1" and "1" both -> "1")
                        stone_id_to_reward[s_id] = str(int(r_match.group(1)))
            assert len(support_stone_ids) == 8, f"Expected 8 stones in support, got {len(support_stone_ids)}"

            # Determine reachable stones (Set of IDs)
            # A stone is reachable if it is in the support AND its reward is in the allowed set
            allowed_rewards = reachable_reward_mapping.get(query_stone_reward)
            # Normalize allowed rewards as well
            normalized_allowed_rewards = {str(int(r)) for r in allowed_rewards}

            reachable_stones_set = {
                sid for sid in support_stone_ids 
                if stone_id_to_reward.get(sid) in normalized_allowed_rewards
            }
            # Cheack if the reachable_stones_set is always a subset of support_stone_ids
            assert reachable_stones_set.issubset(support_stone_ids), "Reachable stones must be subset of support stones."

            # Determine actual connected neighbors
            connected_neighbors = neighbors_per_chemistry[support_key].get(query_stone_id, set())
            assert len(connected_neighbors) == 2, f"Expected 2 connected neighbors, got {len(connected_neighbors)}"
            true_adjacent_set = connected_neighbors.union({target_class_id})

            # Optional sanity checks
            if query_stone_reward in ['3', '-3']:
                assert len(reachable_stones_set) == 3, (
                    f"Expected 3 reachable stones for reward {query_stone_reward}, "
                    f"got {len(reachable_stones_set)}"
                )
                assert len(true_adjacent_set) == 3, (
                    f"Expected 3 true-adjacent stones for reward {query_stone_reward}, "
                    f"got {len(true_adjacent_set)}"
                )
            if query_stone_reward in ['1', '-1']:
                assert len(reachable_stones_set) == 4, (
                    f"Expected 4 reachable stones for reward {query_stone_reward}, "
                    f"got {len(reachable_stones_set)}"
                )
                assert len(true_adjacent_set) == 3, (
                    f"Expected 4 true-adjacent stones for reward {query_stone_reward}, "
                    f"got {len(true_adjacent_set)}"
                )
            # Do assertions for connected neighbors too.
            assert len(connected_neighbors) == len(true_adjacent_set) - 1, f"Connected neighbors should be one less than true adjacent set."

            epoch_metrics[query_stone_reward]['total_count'] += 1

            # Metric 1: Check if prediction is in support
            if predicted_class_id in support_stone_ids:
                epoch_metrics[query_stone_reward]['total_in_support'] += 1
                
                # Metric 2: Check if prediction is reachable (Stone-based check)
                if predicted_class_id in reachable_stones_set:
                    epoch_metrics[query_stone_reward]['in_support_and_reachable'] += 1
                    epoch_metrics[query_stone_reward]['total_reachable'] += 1
                    
                    # Metric 3: Check if it's the exact correct target
                    if predicted_class_id == target_class_id:
                        epoch_metrics[query_stone_reward]['reachable_and_correct'] += 1
                
                # Metric 4: Check if prediction is an actual connected neighbor
                if predicted_class_id in connected_neighbors:
                    assert len(connected_neighbors) == 2, f"Expected 2 connected neighbors, got {len(connected_neighbors)}"
                    epoch_metrics[query_stone_reward]['in_support_and_connected'] += 1

                # Metric 5: Check if prediction is in true_adjacent_set
                if predicted_class_id in true_adjacent_set:
                    epoch_metrics[query_stone_reward]['in_support_and_true_adjacent'] += 1
        
        # Calculate accuracies for this epoch immediately
        for reward in reachable_reward_mapping.keys():
            metrics = epoch_metrics[reward]
            # import pdb; pdb.set_trace()
            
            if metrics['total_in_support'] > 0:
                reachable_acc = metrics['in_support_and_reachable'] / metrics['total_in_support']
                connected_acc = metrics['in_support_and_connected'] / metrics['total_in_support']
                true_adjacent_acc = metrics['in_support_and_true_adjacent'] / metrics['total_in_support']
                within_connected_in_reachable = metrics['in_support_and_connected'] / metrics['in_support_and_reachable'] if metrics['in_support_and_reachable'] > 0 else 0.0
            else:
                reachable_acc = 0
                connected_acc = 0
                true_adjacent_acc = 0
                within_connected_in_reachable = 0
            
            adjacency_accuracies[reward]['within_reachable_acc'].append(reachable_acc)
            adjacency_accuracies[reward]['within_connected_acc'].append(connected_acc)
            adjacency_accuracies[reward]['within_true_adjacent_acc'].append(true_adjacent_acc)
            adjacency_accuracies[reward]['within_connected_in_reachable_acc'].append(within_connected_in_reachable)
            
            if metrics['total_reachable'] > 0:
                correct_acc = metrics['reachable_and_correct'] / metrics['total_reachable']
            else:
                correct_acc = 0
            adjacency_accuracies[reward]['correct_within_reachable_acc'].append(correct_acc)
    
    return adjacency_accuracies


def analyze_half_chemistry_behaviour(data, vocab, stone_state_to_id, predictions_by_epoch, exp_typ='held_out', hop=2,
                                    composition_full_target_data=None, factorize_within_half_predictions=False):
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
        support = encoder_input_ids[:-(hop + 4)]  # everything except last 5 tokens. works for decomposition too.
        # create a hashable string key
        support_key = tuple(support)
        
        if support_key not in support_to_query_mappings: # this is for the input to the model. this will be the same for all epochs.
            support_to_query_mappings[support_key] = {}
            
            
    input_vocab = vocab['input_word2idx']
    feature_to_id_vocab = {v: k for k, v in input_vocab.items()}
            
    for i, sample in enumerate(data):
        encoder_input_ids = sample['encoder_input_ids']
        target_class_id = sample['target_class_id']
        support = encoder_input_ids[:-(hop + 4)]  # everything except last 5 tokens. works for decomposition too.
        # create a hashable string key
        support_key = tuple(support)
        
        if exp_typ == 'composition':
            # based on the number of hops, we need to adjust the query parsing. the hops denote the number of potions in the query.
            query = encoder_input_ids[-(hop + 4):] # 4 featuresd + hop potions.
            query_potion = query[-hop:]  # last hop tokens
            query_stones = query[:-hop]
            
            # create a string representation of the query potion sequence from the feature_to_id_vocab and join them.
            query_potion_str = ' | '.join([feature_to_id_vocab[token_id] for token_id in query_potion])
            query_potion = query_potion_str
            
        else:
            # for decomposition and held_out experiments.
            query = encoder_input_ids[-5:]    # last 5 tokens
            query_potion = query[-1]
            query_stones = query[:-1]

        # first check if the query_potion key is already a list. if not, create an empty list.
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

    # store the predictions for each support key and query potion.

    support_to_query_per_epoch_predictions = {}

    for epoch in predictions_by_epoch.keys():
        support_to_query_per_epoch_predictions[epoch] = {}
        for support_key in support_to_query_mappings.keys():
            support_to_query_per_epoch_predictions[epoch][support_key] = {}
            for potion in support_to_query_mappings[support_key].keys(): # for the composition experiments, this will be multiple potions.
                support_to_query_per_epoch_predictions[epoch][support_key][potion] = []


    # create another dict called composition_per_query_support_to_query_mappings that will store for each support_key, and each query_start_stone, the list of the target class ids for each potion combination.

    """
    the following code block stores the mapping from support sets to query stones and potions for composition experiments. for each support key, it organizes the target class ids based on the query stones and potions for that support key.
    """
    if exp_typ == 'composition':
        composition_per_query_support_to_query_mappings = {} # This is only for the subsampled data that was shown to the model.
        composition_full_target_per_query_support_to_query_mappings = {} # This is for the full data, to get the full target class ids for each support and query stone combination.
        for i, sample in enumerate(data):
            encoder_input_ids = sample['encoder_input_ids']
            target_class_id = sample['target_class_id']
            support = encoder_input_ids[:-(hop + 4)]  # everything except last 5 tokens
            support_key = tuple(support)

            # Split the support key on '23' and sort the stones to create a normalized support key.
            support_key_string = ' '.join(str(token_id) for token_id in support_key)
            support_key = sorted([chunk.strip() for chunk in support_key_string.split('23')])

            support_key = tuple(support_key)

            
            # based on the number of hops, we need to adjust the query parsing. the hops denote the number of potions in the query.
            query = encoder_input_ids[-(hop + 4):] # 4 featuresd + hop potions.
            query_potion = query[-hop:]  # last hop tokens
            query_stones = query[:-hop]
            
            # create a string representation of the query potion sequence from the feature_to_id_vocab and join them.
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

        # now do the same for the full target data.
        for i, sample in enumerate(composition_full_target_data):
            encoder_input_ids = sample['encoder_input_ids']
            target_class_id = sample['target_class_id']
            support = encoder_input_ids[:-(hop + 4)]  # everything except last 5 tokens
            support_key = tuple(support)

            # Split the support key on '23' and sort the stones to create a normalized support key.
            support_key_string = ' '.join(str(token_id) for token_id in support_key)
            support_key = sorted([chunk.strip() for chunk in support_key_string.split('23')])

            support_key = tuple(support_key)
            
            # based on the number of hops, we need to adjust the query parsing. the hops denote the number of potions in the query.
            query = encoder_input_ids[-(hop + 4):] # 4 featuresd + hop potions.
            query_potion = query[-hop:]  # last hop tokens
            query_stones = query[:-hop]
            
            # create a string representation of the query potion sequence from the feature_to_id_vocab and join them.
            query_potion_str = ' | '.join([feature_to_id_vocab[token_id] for token_id in query_potion])
            query_potion = query_potion_str
            
            query_stone_str = ' | '.join([feature_to_id_vocab[token_id] for token_id in query_stones])
            
            if support_key not in composition_full_target_per_query_support_to_query_mappings:
                composition_full_target_per_query_support_to_query_mappings[support_key] = {}
            if query_stone_str not in composition_full_target_per_query_support_to_query_mappings[support_key]:
                composition_full_target_per_query_support_to_query_mappings[support_key][query_stone_str] = {}
            if query_potion not in composition_full_target_per_query_support_to_query_mappings[support_key][query_stone_str]:
                composition_full_target_per_query_support_to_query_mappings[support_key][query_stone_str][query_potion] = [target_class_id]
            else:
                composition_full_target_per_query_support_to_query_mappings[support_key][query_stone_str][query_potion].append(target_class_id)

        target_support_keys = list(composition_full_target_per_query_support_to_query_mappings.keys())
        original_support_keys = list(composition_per_query_support_to_query_mappings.keys())
        test = [1 for key in target_support_keys if key not in original_support_keys]
        assert len(test) == 0, "Support keys in full target data do not match those in subsampled data."



        # Create the mapping for all possible outcomes for each query stone, organized by support_key
        per_query_reachable_stone_mapping = {}
        for support_key, query_stones_map in composition_full_target_per_query_support_to_query_mappings.items():
            per_query_reachable_stone_mapping[support_key] = defaultdict(set)
            for query_stone_str, potions_map in query_stones_map.items():
                for full_potion_sequence, target_stones in potions_map.items():
                    per_query_reachable_stone_mapping[support_key][query_stone_str].update(target_stones)


        # BUG FIX: Pre-calculate the correct reachable stones for each prefix
        prefix_to_reachable_stones_mapping = {}
        for support_key, query_stones_map in composition_full_target_per_query_support_to_query_mappings.items():
            prefix_to_reachable_stones_mapping[support_key] = {}
            for query_stone_str, potions_map in query_stones_map.items():
                prefix_to_reachable_stones_mapping[support_key][query_stone_str] = {}
                for overlap_count in range(1, hop):
                    prefix_to_reachable_stones_mapping[support_key][query_stone_str][overlap_count] = defaultdict(set)

                for full_potion_sequence, target_stones in potions_map.items():
                    potion_tokens = full_potion_sequence.split(' | ')
                    for overlap_count in range(1, hop):
                        potion_prefix = ' | '.join(potion_tokens[:overlap_count])
                        prefix_to_reachable_stones_mapping[support_key][query_stone_str][overlap_count][potion_prefix].update(target_stones)

        
        # now, for each query_stone for that support key, we can also store the 'predictions' for each potion combination in a new dict.
        predictions_composition_per_query_support_to_query_mappings = {}
        for epoch in predictions_by_epoch.keys():
            predictions_composition_per_query_support_to_query_mappings[epoch] = {}
            for support_key in composition_per_query_support_to_query_mappings.keys():
                predictions_composition_per_query_support_to_query_mappings[epoch][support_key] = {}
                for query_stone_str in composition_per_query_support_to_query_mappings[support_key].keys():
                    predictions_composition_per_query_support_to_query_mappings[epoch][support_key][query_stone_str] = {}
                    for query_potion in composition_per_query_support_to_query_mappings[support_key][query_stone_str].keys():
                        predictions_composition_per_query_support_to_query_mappings[epoch][support_key][query_stone_str][query_potion] = []


        for epoch, predictions in tqdm(predictions_by_epoch.items(), desc="organizing composition predictions by support, query stones, and potions"):
            for i, sample in enumerate(data):
                encoder_input_ids = sample['encoder_input_ids']
                target_class_id = sample['target_class_id']
                predicted_class_id = predictions[i]
                support = encoder_input_ids[:-(hop + 4)]  # everything except last 5 tokens
                support_key = tuple(support)

                # Split the support key on '23' and sort the stones to create a normalized support key.
                support_key_string = ' '.join(str(token_id) for token_id in support_key)
                support_key = sorted([chunk.strip() for chunk in support_key_string.split('23')])
                support_key = tuple(support_key)

                # based on the number of hops, we need to adjust the query parsing. the hops denote the number of potions in the query.
                query = encoder_input_ids[-(hop + 4):] # 4 featuresd + hop potions.
                query_potion = query[-hop:]  # last hop tokens
                query_stones = query[:-hop]
                # create a string representation of the query potion sequence from the feature_to_id_vocab and join them.
                query_potion_str = ' | '.join([feature_to_id_vocab[token_id] for token_id in query_potion])
                query_potion = query_potion_str
                query_stone_str = ' | '.join([feature_to_id_vocab[token_id] for token_id in query_stones])

                # now directly store the predicted class id in the corresponding dict.
                predictions_composition_per_query_support_to_query_mappings[epoch][support_key][query_stone_str][query_potion].append(predicted_class_id) # the dictionary is already created above.

        

        # now we calculate metrics based on the above predictions and mappings.
        # 1. for each query stone and potion combination, calculate if the prediction was in the support set. the support set is union of all the target class ids for that support key.
        # 2. for each query stone and potion combination, calculate if the prediction was in the possible target class ids for that query stone (across all potion combinations) if the query output was in the support set.
        # import pdb; pdb.set_trace()

        predicted_in_context_accuracies = []
        predicted_in_context_correct_candidate_accuracies = []
        correct_within_candidates = []
        overlap_metrics_by_epoch = {}  # initialize here, populate in the main loop

        for pred_epoch in tqdm(predictions_composition_per_query_support_to_query_mappings.keys(), desc="analyzing composition epochs"):
            correct = 0
            correct_candidate = 0
            total = 0
            predicted_in_context_count = 0

            epoch_preds = predictions_composition_per_query_support_to_query_mappings[pred_epoch]
            
            # initialize overlap analysis for this epoch
            epoch_overlap_analysis = {}

            for support_key in composition_per_query_support_to_query_mappings.keys():
                # initialize for this support_key
                if support_key not in epoch_overlap_analysis:
                    epoch_overlap_analysis[support_key] = {}
                    for overlap_count in range(1, hop):
                        epoch_overlap_analysis[support_key][overlap_count] = {}
                
                res = list(composition_per_query_support_to_query_mappings[support_key].values())
                all_target_class_ids = set([v[0] for sublist in res for k, v in sublist.items()])
                assert len(all_target_class_ids) == 8, f"expected 8 unique target class ids for support {support_key}, got {len(all_target_class_ids)}"

                # for each query stone and potion combination, check the predictions
                for query_stone_str in composition_per_query_support_to_query_mappings[support_key].keys():
                    for query_potion in composition_per_query_support_to_query_mappings[support_key][query_stone_str].keys():
                        
                        pred = epoch_preds[support_key][query_stone_str][query_potion][0]
                        total += 1

                        # get the set of target_class_ids for this query_stone and all potion combinations.
                        possible_target_class_ids_for_this_query_stone = set([v for sublist in composition_per_query_support_to_query_mappings[support_key][query_stone_str].values() for v in sublist])

                        # calculate standard composition metrics
                        if pred in all_target_class_ids:
                            predicted_in_context_count += 1

                            if pred in possible_target_class_ids_for_this_query_stone:
                                correct_candidate += 1

                                if pred in composition_per_query_support_to_query_mappings[support_key][query_stone_str][query_potion]:
                                    correct += 1

                        # populate overlap analysis - now organized by support_key
                        potion_tokens = query_potion.split(' | ')
                        
                        for overlap_count in range(1, hop):
                            potion_prefix = ' | '.join(potion_tokens[:overlap_count])

                            # initialize query_stone_str dict if not exists for this support_key and overlap_count
                            if query_stone_str not in epoch_overlap_analysis[support_key][overlap_count]:
                                epoch_overlap_analysis[support_key][overlap_count][query_stone_str] = {}
                            
                            # initialize potion_prefix dict if not exists
                            if potion_prefix not in epoch_overlap_analysis[support_key][overlap_count][query_stone_str]:
                                # Use the pre-calculated reachable stones
                                reachable_stones_for_prefix = prefix_to_reachable_stones_mapping[support_key][query_stone_str][overlap_count][potion_prefix]
                                epoch_overlap_analysis[support_key][overlap_count][query_stone_str][potion_prefix] = {
                                    'reachable_stones': reachable_stones_for_prefix,
                                    'full_sequences': [],
                                    'predictions': []
                                }
                            
                            # add full sequence
                            epoch_overlap_analysis[support_key][overlap_count][query_stone_str][potion_prefix]['full_sequences'].append(query_potion)
                            
                            # add prediction
                            epoch_overlap_analysis[support_key][overlap_count][query_stone_str][potion_prefix]['predictions'].append(pred)
                            # import pdb; pdb.set_trace()
                    


            
            # calculate and store standard metrics
            predicted_in_context_accuracy = predicted_in_context_count / total if total > 0 else 0
            # predicted_in_context_accuracies.append(predicted_in_context_accuracy)
            
            predicted_in_context_correct_candidate_accuracy = correct_candidate / predicted_in_context_count if predicted_in_context_count > 0 else 0
            # predicted_in_context_correct_candidate_accuracies.append(predicted_in_context_correct_candidate_accuracy)

            correct_within_candidate = correct / correct_candidate if correct_candidate > 0 else 0
            # correct_within_candidates.append(correct_within_candidate)


            # calculate overlap metrics for this epoch
            # now calculate per-support_key metrics and aggregated metrics
            epoch_overlap_metrics = {}

            for overlap_count in range(1, hop):
                per_support_metrics = {}
                
                for support_key in epoch_overlap_analysis.keys():
                    
                    # This dict will hold the accuracy for each query stone for the current overlap_count
                    per_query_accuracies = {}

                    for query_stone_str in epoch_overlap_analysis[support_key][overlap_count].keys():
                        
                        # This list will hold the accuracy for each prefix of the current length
                        prefix_accuracies = []

                        for potion_prefix, data in epoch_overlap_analysis[support_key][overlap_count][query_stone_str].items():
                            predictions = data['predictions']
                            reachable_stones_k = data['reachable_stones'] # Reachable set for prefix of length k


                            # Determine the denominator set based on overlap_count
                            if overlap_count == 1:
                                # For 1-potion overlap, the denominator is the set of all possible candidates for this query stone.
                                denominator_set = per_query_reachable_stone_mapping[support_key][query_stone_str]
                            else:
                                # For k > 1, the denominator is the reachable set from the (k-1) prefix.
                                parent_prefix = ' | '.join(potion_prefix.split(' | ')[:-1])
                                denominator_set = epoch_overlap_analysis[support_key][overlap_count - 1][query_stone_str][parent_prefix]['reachable_stones']

                            # Denominator: How many predictions fell into the denominator_set?
                            preds_in_denominator = [p for p in predictions if p in denominator_set]
                            denominator_count = len(preds_in_denominator)

                            if denominator_count > 0:
                                # Numerator: Of those, how many also fell into the more constrained set for the current prefix?
                                numerator_count = sum(1 for p in preds_in_denominator if p in reachable_stones_k)
                                
                                # Accuracy for this specific prefix
                                prefix_accuracy = numerator_count / denominator_count
                                prefix_accuracies.append(prefix_accuracy)
                            # if pred_epoch == '100':
                            #     import pdb; pdb.set_trace()

                        # Average the accuracies across all prefixes for this query stone
                        if prefix_accuracies:
                            per_query_accuracies[query_stone_str] = np.mean(prefix_accuracies)

                    # Calculate per-support metrics by averaging over query stones
                    per_support_metrics[support_key] = {
                        'incremental_learning_accuracy': np.mean(list(per_query_accuracies.values())) if per_query_accuracies else 0.0
                    }

                epoch_overlap_metrics[overlap_count] = {
                    'per_support_metrics': per_support_metrics,
                }
                
            overlap_metrics_by_epoch[pred_epoch] = {
                'epoch_overlap_metrics': epoch_overlap_metrics,
                'predicted_in_context_accuracy': predicted_in_context_accuracy,
                'predicted_in_context_correct_candidate_accuracy': predicted_in_context_correct_candidate_accuracy,
                'correct_within_candidate': correct_within_candidate
            }

         # Extract standard metrics as lists for compatibility with plotting code
        predicted_in_context_accuracies = []
        predicted_in_context_correct_candidate_accuracies = []
        correct_within_candidates = []
        
        for epoch_key in sorted(overlap_metrics_by_epoch.keys()):
            predicted_in_context_accuracies.append(overlap_metrics_by_epoch[epoch_key]['predicted_in_context_accuracy'])
            predicted_in_context_correct_candidate_accuracies.append(overlap_metrics_by_epoch[epoch_key]['predicted_in_context_correct_candidate_accuracy'])
            correct_within_candidates.append(overlap_metrics_by_epoch[epoch_key]['correct_within_candidate'])

        return (predicted_in_context_accuracies, predicted_in_context_correct_candidate_accuracies,
                correct_within_candidates, overlap_metrics_by_epoch)


    # The following are the for the decomposition experiments and held-out experiments.

    # Now for each sample, we can store the prediction in the corresponding support key and query potion.
    print("Decomposition / Held-out experiment.")
    if exp_typ in ['decomposition', 'held_out']:
        hop = 1
        # For decomposition, precompute per-query adjacency from targets:
        # per_query_adjacent_mapping[support_key][query_stone_id][potion_str] = set(target_class_ids)
        per_query_adjacent_mapping = {}
        if exp_typ == 'decomposition':
            input_vocab = vocab['input_word2idx']
            feature_to_id_vocab = {v: k for k, v in input_vocab.items()}

            for sample in data:
                encoder_input_ids = sample['encoder_input_ids']
                target_class_id = sample['target_class_id']

                # support and support_key
                support = encoder_input_ids[:-(hop + 4)]  # hop forced to 1 for decomposition/held_out above
                support_key = tuple(support)

                # decode query (last 5 tokens = 4 features + potion)
                query = encoder_input_ids[-5:]
                query_feat_ids = query[:-1]
                query_potion_id = query[-1]

                query_features = [feature_to_id_vocab[tok_id] for tok_id in query_feat_ids]
                q_color, q_size, q_round, q_reward = query_features
                query_state_str = f"{{color: {q_color}, size: {q_size}, roundness: {q_round}, reward: {q_reward}}}"
                query_stone_id = stone_state_to_id[query_state_str]

                query_potion_str = feature_to_id_vocab[query_potion_id]

                if support_key not in per_query_adjacent_mapping:
                    per_query_adjacent_mapping[support_key] = {}
                if query_stone_id not in per_query_adjacent_mapping[support_key]:
                    per_query_adjacent_mapping[support_key][query_stone_id] = {}
                if query_potion_str not in per_query_adjacent_mapping[support_key][query_stone_id]:
                    per_query_adjacent_mapping[support_key][query_stone_id][query_potion_str] = set()

                per_query_adjacent_mapping[support_key][query_stone_id][query_potion_str].add(target_class_id)
            
            # Now for each support_key and query_stone_id, compute the union of all target_class_ids across potions and store it as the 'per_support_per_query_adjacent_all_potions_mapping'
            per_support_per_query_adjacent_all_potions_mapping = {}
            for support_key, query_stone_map in per_query_adjacent_mapping.items():
                per_support_per_query_adjacent_all_potions_mapping[support_key] = {}
                for query_stone_id, potion_map in query_stone_map.items():
                    all_target_ids = set()
                    for potion_str, target_ids in potion_map.items():
                        all_target_ids.update(target_ids)
                    per_support_per_query_adjacent_all_potions_mapping[support_key][query_stone_id] = all_target_ids


    # import pdb; pdb.set_trace()
    
    for epoch, predictions in tqdm(predictions_by_epoch.items(), desc="Organizing predictions by support and query"):
        for i, sample in enumerate(data):
            encoder_input_ids = sample['encoder_input_ids']
            target_class_id = sample['target_class_id']
            predicted_class_id = predictions[i]
            support = encoder_input_ids[:-(hop + 4)]  # Everything except last 5 tokens for decomposition and held_out.
            support_key = tuple(support)
            
            # if exp_typ == 'composition':
            #     # Based on the number of hops, we need to adjust the query parsing. The hops denote the number of potions in the query.
            #     query = encoder_input_ids[-(hop + 4):] # 4 featuresd + hop potions.
            #     query_potion = query[-hop:]  # Last hop tokens
                
            #     # Create a string representation of the query potion sequence from the feature_to_id_vocab and join them.
            #     query_potion_str = ' | '.join([feature_to_id_vocab[token_id] for token_id in query_potion])
            #     query_potion = query_potion_str
                
            # else:  
            query = encoder_input_ids[-(hop + 4):]    # Last 5 tokens for decomposition and held_out.
            query_potion = query[-1]
            # if exp_typ == 'composition':
            #     support_to_query_per_epoch_predictions[epoch][support_key][query_potion].append(predicted_class_id)

            # else:
            support_to_query_per_epoch_predictions[epoch][support_key][feature_to_id_vocab[query_potion]].append(predicted_class_id)
    

    # Now for each of the predictions, we can check which half-chemistry for that support set does the true target belong to.
    
    # Initialize accumulators for averaging
    predicted_in_context_accuracies = []
    predicted_in_context_correct_half_accuracies = []
    predicted_in_context_other_half_accuracies = []
    predicted_in_context_correct_half_exact_accuracies = []
    predicted_correct_within_context = []



    predicted_exact_out_of_all_108 = []

    predicted_in_adjacent_and_correct_half_accuracies = []
    predicted_correct_half_within_adjacent_and_correct_half_accuracies = []



    complete_query_stone_state_per_reward_binned_accuracy = {'-3': [], '-1': [], '1': [], '3': []}
    within_support_query_stone_state_per_reward_binned_accuracy = {'-3': [], '-1': [], '1': [], '3': []}
    within_support_within_half_query_stone_state_per_reward_binned_accuracy = {'-3': [], '-1': [], '1': [], '3': []}
    
    for epoch, predictions in tqdm(predictions_by_epoch.items(), desc="Analyzing epochs"):
        correct = 0
        other_half_correct = 0
        total = 0
        
        
        within_class_correct = 0
        within_class_total = 0

        predicted_in_context_count = 0
        correct_half_chemistry_count = 0

        predicted_correct_within_context_count = 0

        predicted_exact_out_of_all_108_count = 0

        predicted_in_adjacent_and_correct_half_count = 0
        predicted_correct_half_within_adjacent_and_correct_half_count = 0

        per_epoch_complete_query_stone_state_per_reward_binned_counts = {'-3': 0, '-1': 0, '1': 0, '3': 0}
        per_epoch_within_support_query_stone_state_per_reward_binned_counts = {'-3': 0, '-1': 0, '1': 0, '3': 0}
        per_epoch_within_support_within_half_query_stone_state_per_reward_binned_counts = {'-3': 0, '-1': 0, '1': 0, '3': 0}

        per_epoch_total_samples_per_reward_bin = {'-3': 0, '-1': 0, '1': 0, '3': 0}
        per_epoch_in_support_samples_per_reward_bin = {'-3': 0, '-1': 0, '1': 0, '3': 0}
        per_epoch_in_support_correct_half_samples_per_reward_bin = {'-3': 0, '-1': 0, '1': 0, '3': 0}


        
        for i, sample in enumerate(data):
            encoder_input_ids = sample['encoder_input_ids']
            target_class_id = sample['target_class_id']
            predicted_class_id = predictions[i]
            support = encoder_input_ids[:-(hop + 4)]
            support_key = tuple(support)
            
            query = encoder_input_ids[-5:]
            query_potion = query[-1]

            # Get the reward value for binning
            query_start_stone_reward = feature_to_id_vocab[query[-2]]  # second last token in the query is the reward of the stone.
            # import pdb; pdb.set_trace()

            # First we check if the predicted class ID is in any of the two half-chemistry sets for this support key.
            # import pdb; pdb.set_trace()

            potions_for_support = list(support_to_query_mappings[support_key].keys())
            # TODO: This will fail for composition experiments because the keys will be a combination of potions. Thus we need to adjust this.
            
            # if exp_typ == 'composition':
            #     raise NotImplementedError("Half-chemistry analysis for composition experiments is not implemented yet.")
            
            assert len(potions_for_support) in [2,6], f"Expected 2 or 6 potions."
            correct_half_chemistry = support_to_query_mappings[support_key][feature_to_id_vocab[query_potion]]
            other_half_chemistry = support_to_query_mappings[support_key][potions_for_support[0]] if potions_for_support[1] == feature_to_id_vocab[query_potion] else support_to_query_mappings[support_key][potions_for_support[1]]

            if exp_typ == 'decomposition': 
                # Create a set of all stones in the support set.
                all_stones_in_support = set()
                for potion in potions_for_support:
                    all_stones_in_support.update(support_to_query_mappings[support_key][potion])
                combined_set = all_stones_in_support
                assert len(combined_set) == 8, f"Expected 8 unique stones for support {support_key}, got {len(combined_set)}"
            else:
                combined_set = set(correct_half_chemistry + other_half_chemistry)
                assert len(combined_set) <= 8, f"Expected 8 unique stones for support {support_key}, got {len(combined_set)}"
            
            if exp_typ == 'decomposition':
                # Decode query start stone id (same logic as in non-support analysis)
                query_feat_ids = query[:-1]  # 4 feature token IDs
                query_features = [feature_to_id_vocab[tok_id] for tok_id in query_feat_ids]
                q_color, q_size, q_round, q_reward = query_features
                query_state_str = f"{{color: {q_color}, size: {q_size}, roundness: {q_round}, reward: {q_reward}}}"
                query_stone_id = stone_state_to_id[query_state_str]

                query_potion_str = feature_to_id_vocab[query_potion]

                # Get adjacent stones for this (support_key, query_stone_id, potion)
                adjacent_stones = per_support_per_query_adjacent_all_potions_mapping[support_key].get(query_stone_id, set())
                adjacent_stones = set(adjacent_stones)

                # Combine with correct half for this potion
                adjacent_and_correct_half = set(correct_half_chemistry).union(adjacent_stones)

                # Sanity: adjacency is fully connected; expect 6 unique stones 
                # (4 in correct half, 3 adjacent, with 1 overlap)
                if len(adjacent_and_correct_half) != 6:
                    # You can relax this assert if you're worried about data edge cases
                    raise AssertionError(
                        f"Expected 6 stones in adjacent_and_correct_half, "
                        f"got {len(adjacent_and_correct_half)} for support {support_key}"
                    )
            else:
                adjacent_stones = set()
                adjacent_and_correct_half = set()


            # First do the classification for 8 vs 108.

            per_epoch_total_samples_per_reward_bin[query_start_stone_reward] += 1

            # First classification: exact match out of 108
            if predicted_class_id == target_class_id:
                predicted_exact_out_of_all_108_count += 1
                per_epoch_complete_query_stone_state_per_reward_binned_counts[query_start_stone_reward] += 1
            
            # Second classification: in-support
            if predicted_class_id in combined_set:
                predicted_in_context_count += 1
                
                # NEW: Increment in-support total for this reward bin
                per_epoch_in_support_samples_per_reward_bin[query_start_stone_reward] += 1

                if predicted_class_id == target_class_id:
                    predicted_correct_within_context_count += 1
                    per_epoch_within_support_query_stone_state_per_reward_binned_counts[query_start_stone_reward] += 1
                
                # Third classification: correct half
                if predicted_class_id in correct_half_chemistry:
                    correct_half_chemistry_count += 1
                    
                    # NEW: Increment correct-half total for this reward bin
                    per_epoch_in_support_correct_half_samples_per_reward_bin[query_start_stone_reward] += 1

                    if predicted_class_id == target_class_id:
                        within_class_correct += 1
                        per_epoch_within_support_within_half_query_stone_state_per_reward_binned_counts[query_start_stone_reward] += 1

                
                elif predicted_class_id in other_half_chemistry:
                    other_half_correct += 1
                
                if exp_typ == 'decomposition' and predicted_class_id in adjacent_and_correct_half:
                    predicted_in_adjacent_and_correct_half_count += 1
                    if predicted_class_id in correct_half_chemistry:
                        predicted_correct_half_within_adjacent_and_correct_half_count += 1

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

        # Now calculate the exact out of all 108 accuracy.
        predicted_exact_out_of_all_108_accuracy = predicted_exact_out_of_all_108_count / total if total > 0 else 0 # The chance is 1/108 here.
        predicted_exact_out_of_all_108.append(predicted_exact_out_of_all_108_accuracy)

        if exp_typ == 'decomposition':
            p_in_adjacent_and_correct_half = (
                predicted_in_adjacent_and_correct_half_count / predicted_in_context_count
                if predicted_in_context_count > 0 else 0.0
            )
            p_correct_half_given_adjacent_and_correct_half = (
                predicted_correct_half_within_adjacent_and_correct_half_count / predicted_in_adjacent_and_correct_half_count
                if predicted_in_adjacent_and_correct_half_count > 0 else 0.0
            )
        else:
            p_in_adjacent_and_correct_half = 0.0
            p_correct_half_given_adjacent_and_correct_half = 0.0


        predicted_in_adjacent_and_correct_half_accuracies.append(p_in_adjacent_and_correct_half)
        predicted_correct_half_within_adjacent_and_correct_half_accuracies.append(p_correct_half_given_adjacent_and_correct_half)


        # Assert that correct_half_chemistry_count is equal to the sum of per_epoch_within_support_within_half_query_stone_state_per_reward_binned_counts
        assert correct_half_chemistry_count == sum(per_epoch_in_support_correct_half_samples_per_reward_bin.values()), "Mismatch in correct half chemistry count."

        # import pdb; pdb.set_trace()



        # NOTE: The following is for the held_out experiments reward-binned analysis.
        # Add the per-epoch reward binned accuracies to the overall accumulators.
        # Add the per-epoch reward binned accuracies to the overall accumulators.
        for reward_bin in complete_query_stone_state_per_reward_binned_accuracy.keys():
            # Accuracy out of all 108
            total_for_bin = per_epoch_total_samples_per_reward_bin[reward_bin]
            if total_for_bin > 0:
                accuracy1 = per_epoch_complete_query_stone_state_per_reward_binned_counts[reward_bin] / total_for_bin
            else:
                accuracy1 = 0
            complete_query_stone_state_per_reward_binned_accuracy[reward_bin].append(accuracy1)

            in_support_for_bin = per_epoch_in_support_samples_per_reward_bin[reward_bin]
            if in_support_for_bin > 0:
                # accuracy2 = in_support_for_bin / total_for_bin

                # This is exactly the within-support accuracy.

                accuracy2 = per_epoch_within_support_query_stone_state_per_reward_binned_counts[reward_bin] / in_support_for_bin
            else:
                accuracy2 = 0
            within_support_query_stone_state_per_reward_binned_accuracy[reward_bin].append(accuracy2)

            # Accuracy within support and correct half (per reward bin)
            in_support_correct_half_for_bin = per_epoch_in_support_correct_half_samples_per_reward_bin[reward_bin]
            if in_support_correct_half_for_bin > 0:
                accuracy3 = per_epoch_within_support_within_half_query_stone_state_per_reward_binned_counts[reward_bin] / in_support_correct_half_for_bin
            else:
                accuracy3 = 0
            within_support_within_half_query_stone_state_per_reward_binned_accuracy[reward_bin].append(accuracy3)

        # Compute weighted average across all bins (for validation only)
        # total_correct_in_bins = sum(per_epoch_within_support_within_half_query_stone_state_per_reward_binned_counts.values())
        # total_samples_in_bins = sum(per_epoch_in_support_correct_half_samples_per_reward_bin.values())

        # if total_samples_in_bins > 0:
        #     avg_within_support_within_half_accuracy = total_correct_in_bins / total_samples_in_bins
        # else:
        #     avg_within_support_within_half_accuracy = 0

        # assert abs(avg_within_support_within_half_accuracy - predicted_in_context_correct_half_exact_accuracy) < 1e-8, "Mismatch in weighted average accuracy calculation."
            


    return (
        predicted_in_context_accuracies,
        predicted_in_context_correct_half_accuracies,
        predicted_in_context_other_half_accuracies,
        predicted_in_context_correct_half_exact_accuracies,
        predicted_correct_within_context,
        predicted_exact_out_of_all_108,
        predicted_in_adjacent_and_correct_half_accuracies,
        predicted_correct_half_within_adjacent_and_correct_half_accuracies,
        (
            complete_query_stone_state_per_reward_binned_accuracy,
            within_support_query_stone_state_per_reward_binned_accuracy,
            within_support_within_half_query_stone_state_per_reward_binned_accuracy,
        ),
    )


def load_epoch_data(exp_typ: str = 'held_out', hop = 2, epoch_range = (0, 500), seeds = [2], scheduler_prefix='', file_paths = None, file_paths_non_subsampled = None):
    """
    Load predictions and inputs/targets/predictions for specified experiment type and hop (if applicable).
    
    Args:
        exp_typ: 'held_out', 'decomposition', or 'composition'
        hops: List of hop counts to load data for
    """ 

    
    epoch_start, epoch_end = epoch_range
    
    predictions_by_epoch_by_seed = {}
    
    inputs_by_seed = {}
    non_subsampled_targets_by_seed = {}

    # file_paths is a list 
    if file_paths is not None:
        seeds = []
        for path in file_paths:
            # Extract seed from the path assuming the path contains 'seed_{seed_number}'
            import re
            match = re.search(r'seed_(\d+)', path)
            if match:
                seed_number = int(match.group(1))
                seeds.append(seed_number)
            else:
                raise ValueError(f"Seed not found in the provided file path: {path}")

        # Load the data for each provided file path
        print("Loading data from provided file paths for seeds: ", seeds)
        for i, path in enumerate(file_paths):
            print("Loading data for seed ", seeds[i], " from path ", path)
            seed = seeds[i]
            predictions_by_epoch = {}

            if exp_typ == 'composition' or exp_typ == 'decomposition':
                if file_paths_non_subsampled is not None:
                    non_subsampled_path = file_paths_non_subsampled[hop][i]
                    # Check if the seed in the non_subsampled_path matches the current seed
                    match_non_subsampled = re.search(r'seed_(\d+)', non_subsampled_path)
                    if match_non_subsampled:
                        seed_non_subsampled = int(match_non_subsampled.group(1))
                        # import pdb; pdb.set_trace()
                        if seed_non_subsampled != seed:
                            # print(f"Warning: Seed mismatch between subsampled and non-subsampled paths: {seed} vs {seed_non_subsampled}. Proceeding anyway.")
                            raise ValueError(f"Seed mismatch between subsampled and non-subsampled paths: {seed} vs {seed_non_subsampled}")
                        else:
                            non_subsampled_path_val_data = pickle.load(open(non_subsampled_path, 'rb'))
                            # Iterate through the non-subsampled data to extract inputs and targets
                            inputs_raw = []
                            targets_raw = []
                            for sample in non_subsampled_path_val_data:
                                inputs_raw.append(sample['encoder_input_ids'])
                                targets_raw.append(sample['target_class_id'])
                            
                            # Create a data_with_targets list
                            data_with_targets_non_subsampled = [{'encoder_input_ids': inputs_raw[i], 'target_class_id': targets_raw[i]} for i in range(len(targets_raw))]
                            print("Created non-subsampled data for seed ", seed, " with ", len(data_with_targets_non_subsampled), " samples.")
                            non_subsampled_targets_by_seed[seed] = data_with_targets_non_subsampled
                    else:
                        raise ValueError(f"Seed not found in the provided non-subsampled file path: {non_subsampled_path}")
                        
            
            for epoch in range(epoch_start, epoch_end + 1):
                # Reformat the epoch_number because the files are saved with epoch numbers like 001, 002, ..., 1000
                epoch_number = str(epoch).zfill(3)
                
                predictions_raw_file_path = f'{path}/predictions_classification_epoch_{epoch_number}.npz'
                
                try:
                    predictions_raw = np.load(predictions_raw_file_path, allow_pickle=True)['predictions']
                    # Store predictions for this epoch
                    predictions_by_epoch[epoch_number] = predictions_raw.tolist()
                    
                except FileNotFoundError:
                    print(f"Warning: Files for epoch {epoch_number} not found in path {path}, skipping...")
                    continue

            predictions_by_epoch_by_seed[seed] = predictions_by_epoch 
            inputs_raw_file_path = f'{path}/inputs_classification_epoch_001.npz' # Use the last epoch number loaded. Doesn't matter because inputs are same for all epochs.
            targets_raw_file_path = f'{path}/targets_classification_epoch_001.npz' # Use the last epoch number loaded.

            inputs_raw = np.load(inputs_raw_file_path, allow_pickle=True)['inputs']
            targets_raw = np.load(targets_raw_file_path, allow_pickle=True)['targets']
            
            stacked_inputs = np.vstack(inputs_raw) # Flatten inputs from (39, 32, 181) to (1240, 181) 
            data_with_targets = [{'encoder_input_ids': stacked_inputs[i].tolist(), 'target_class_id': int(targets_raw[i])} for i in range(len(targets_raw))]
            # if file_paths_non_subsampled is not None and exp_typ == 'composition' and hop > 2:
            #     data_with_targets = [{
            #         'encoder_input_ids': stacked_inputs[i].tolist(),
            #         'target_class_id': int(targets_raw[i])} 
            #     } for i in range(len(targets_raw)) if {
            #         'encoder_input_ids': stacked_inputs[i].tolist(),
            #     ]
            
            inputs_by_seed[seed] = data_with_targets
        
        if file_paths_non_subsampled is not None:
            return predictions_by_epoch_by_seed, inputs_by_seed, non_subsampled_targets_by_seed

        return predictions_by_epoch_by_seed, inputs_by_seed, None





    # print("Seeds to load: ", seeds)

    # for seed in tqdm(seeds):
    #     predictions_by_epoch = {}
        
    #     for epoch in range(epoch_start, epoch_end + 1):
    #         # Reformat the epoch_number because the files are saved with epoch numbers like 001, 002, ..., 1000
    #         epoch_number = str(epoch).zfill(3)
            
    #         # for hop in hops:
    #         base_file_path = ''
            
    #         if exp_typ == 'held_out':
    #             base_file_path = f'/home/rsaha/projects/{infix}dm_alchemy/src/saved_models/held_out_color_exp/held_out_edges_{hop}/all_graphs/xsmall/decoder/classification/{scheduler_prefix}input_features/output_stone_states/shop_1_qhop_1/seed_{seed}/predictions'
    #             # import pdb; pdb.set_trace()
                    
    #         elif exp_typ == 'decomposition':
    #             base_file_path = f"/home/rsaha/projects/{infix}dm_alchemy/src/saved_models/complete_graph/xsmall/decoder/classification/{scheduler_prefix}input_features/output_stone_states/shop_{hop}_qhop_1/seed_{seed}/predictions" 
    #         elif exp_typ == 'composition':
    #             base_file_path = f"/home/rsaha/projects/{infix}dm_alchemy/src/saved_models/complete_graph/fully_shuffled/{scheduler_prefix}xsmall/decoder/classification/input_features/output_stone_states/shop_1_qhop_{hop}/seed_{seed}/predictions"
            
                
                    
    #         predictions_raw_file_path = f'{base_file_path}/predictions_classification_epoch_{epoch_number}.npz'
            
    #         try:
    #             # if exp_typ == 'decomposition':
    #             #     if epoch_number == '035':
    #             #         continue

    #             print(f"epoch number, ", epoch_number)
    #             if epoch_number == '828' and exp_typ == 'held_out':
    #                 continue

    #             predictions_raw = np.load(predictions_raw_file_path, allow_pickle=True)['predictions']
    #             # Store predictions for this epoch
    #             predictions_by_epoch[epoch_number] = predictions_raw.tolist()
                
    #             # print(f"Loaded epoch {epoch_number}: {len(predictions_raw)} predictions")
                
    #         except FileNotFoundError:
    #             print(f"Warning: Files for epoch {epoch_number} not found, skipping...")
    #             continue

    #     try:

    #         predictions_by_epoch_by_seed[seed] = predictions_by_epoch 
    #         inputs_raw_file_path = f'{base_file_path}/inputs_classification_epoch_001.npz' # Use the last epoch number loaded. Doesn't matter because inputs are same for all epochs.
    #         targets_raw_file_path = f'{base_file_path}/targets_classification_epoch_001.npz' # Use the last epoch number loaded.
    #         inputs_raw = np.load(inputs_raw_file_path, allow_pickle=True)['inputs']
    #         targets_raw = np.load(targets_raw_file_path, allow_pickle=True)['targets']
            
    #         stacked_inputs = np.vstack(inputs_raw) # Flatten inputs from (39, 32, 181) to (1240, 181) 
    #         data_with_targets = [{'encoder_input_ids': stacked_inputs[i].tolist(), 'target_class_id': int(targets_raw[i])} for i in range(len(targets_raw))]
            
    #         inputs_by_seed[seed] = data_with_targets
    #     except FileNotFoundError:
    #         print(f"Warning: Input/target files for seed {seed} not found, skipping...")
    #         continue
        
    # return predictions_by_epoch_by_seed, inputs_by_seed, None
        
   
import argparse

# Parse command line arguments for experiment type and hop count.
parser = argparse.ArgumentParser(description="Analyze model predictions for different experiment types and hops.")
parser.add_argument('--exp_typ', type=str, choices=['held_out', 'decomposition', 'composition'], default='composition',
                    help="Type of experiment: 'held_out' or 'decomposition'")
parser.add_argument('--hop', type=int, choices=[1, 2, 3, 4, 5], default=4,
                    help="Hop count for decomposition and composition experiments (ignored for held_out)")
parser.add_argument('--custom_output_file', type=str, default=None,
                    help="Custom output file name for saving results")
parser.add_argument('--get_output_file_from_input_path', action='store_true',
                    help="Flag to generate output file name based on input file paths", default=False)

parser.add_argument('--plot_individual_seeds', action='store_true',
                    help="Flag to plot individual seed results", default=False)

parser.add_argument('--reward_binning_analysis_only', action='store_true',
                    help="Flag to only perform reward binning analysis for the 4 edge held out experiment.", default=False)


parser.add_argument('--normalized_reward', action='store_true',
                    help="Flag to indicate if normalized reward should be used for the held_out exp type.", default=False)

parser.add_argument('--annotated_epochs', action='store_true',
                    help="Flag to indicate if annotated epochs should be used for plotting.", default=False)


parser.add_argument('--get_adjacency_analysis_only', action='store_true',
                    help="Flag to indicate if only the adjacency analysis should be performed.", default=False)

parser.add_argument('--get_non_support_analysis_only', action='store_true',
                    help="Only perform non-support transition analysis (held_out).", default=False)

parser.add_argument('--custom_linestyle', type=str, default=None, 
                    help="Custom linestyle file for plotting.")



args = parser.parse_args()
# exp_typ = 'decomposition'  # 'held_out' or 'decomposition'
exp_typ = args.exp_typ
hop = args.hop  # Only relevant for composition and decomposition experiments.
two_hop_epoch_values_text = [0, 200, 400, 600, 800, 999]
three_hop_epoch_values_text = [0, 200, 600, 800, 999]
four_hop_epoch_values_text = [0, 200, 400, 600, 800, 999]
five_hop_epoch_values_text = [0, 200, 600, 800, 999]

four_edge_held_out_epoch_values_text = [0, 200, 300, 400, 500, 1000]

# Create a dictionary mapping hop counts to their hop-specific epoch values
hop_to_epoch_values = {
    2: two_hop_epoch_values_text,
    3: three_hop_epoch_values_text,
    4: four_hop_epoch_values_text,
    5: five_hop_epoch_values_text
}


scheduler_prefix = 'scheduler_cosine/' 
# NOTE: Do not change the following prefix values.
if exp_typ == 'decomposition':
    # These are the orginal settings used for the decomposition experiments - they might change based on the new results of hyperparemeter tuning. NOTE: the seeds might be different too.
    if hop in [2, 3, 4]:
        scheduler_prefix = 'cosine_restarts/'
    elif hop == 5:
        scheduler_prefix = ''
elif exp_typ == 'held_out':
    scheduler_prefix = ''

# scheduler_prefix = '' 
# seed_values = [2,3,4]
if exp_typ == 'decomposition':
    # NOTE: Do not change this.
    seed_values_2_hop = [0, 16, 29]
    # seed_values_3_hop = [0,1,2,3,4]
    # seed_values_3_hop = [3] # For the 3 Testing with only seed 3 for 3-hop decomposition for now.
    seed_values_3_hop = [0,16,29]
    # seed_values_4_hop = [2,3,4] # For the 4
    seed_values_4_hop = [0,16,29]
    # seed_values_5_hop = [1,2,3]

    # TEMP:
    seed_values_5_hop = [0,2,16] 
elif exp_typ == 'held_out':
    # NOTE: Do not change this.
    seed_values_4_hop = [2,3,4]

    hop_to_epoch_values = {
        4: four_edge_held_out_epoch_values_text
    }

if exp_typ == 'decomposition':
    seed_values_hop_dict = {
        2: seed_values_2_hop,
        3: seed_values_3_hop,
        4: seed_values_4_hop,
        5: seed_values_5_hop
    }
elif exp_typ == 'composition':
    hop_to_epoch_values = {
        # 2: [0, 200, 400, 600, 800, 999],
        # 3: [0, 200, 400, 600, 800, 999],
        # 4: [0, 200, 400, 600, 800, 999],
        # 5: [0, 200, 400, 600, 800, 999]
        # Till 500 only.
        2: [0, 200, 400, 500],
        3: [0, 200, 400, 500],
        4: [0, 200, 400, 500],
        5: [0, 200, 400, 500]
    }
    
    seed_values_hop_dict = {
        2: [0, 16, 29],
        3: [0, 16, 29],
        4: [0, 16, 29],
        5: [0, 16, 29]
    }
else:
    seed_values_hop_dict = {
        4: seed_values_4_hop,
    }


"""
for 2 hop, use seeds
for 3 hop, use seeds 4, and 0.
for 4 hop, can use seeds 2,3,4
for 5 hop, use seed 1,2,3
"""
# Load the for all the seeds.

# Specific run names for each hop count and seed.
"""
2: 29: classification_xsmall_20251027-023153, 16: classification_xsmall_20251026-224313, 0: classification_xsmall_20251026-224309
3: 29: classification_xsmall_20251027-071259, 16: classification_xsmall_20251027-034414, 0: classification_xsmall_20251026-231232
4:
5:
"""


decomposition_file_paths_non_subsampled = {
    2: [
        '/home/rsaha/projects/def-afyshe-ab/rsaha/dm_alchemy/src/data/complete_graph_preprocessed_separate_enhanced_qnodes_in_snodes/decompositional_chemistry_samples_167424_80_unique_stones_train_shop_2_qhop_1_seed_0_classification_filter_True_input_features_output_stone_states_data.pkl',
        '/home/rsaha/projects/def-afyshe-ab/rsaha/dm_alchemy/src/data/complete_graph_preprocessed_separate_enhanced_qnodes_in_snodes/decompositional_chemistry_samples_167424_80_unique_stones_train_shop_2_qhop_1_seed_16_classification_filter_True_input_features_output_stone_states_data.pkl',
        '/home/rsaha/projects/def-afyshe-ab/rsaha/dm_alchemy/src/data/complete_graph_preprocessed_separate_enhanced_qnodes_in_snodes/decompositional_chemistry_samples_167424_80_unique_stones_train_shop_2_qhop_1_seed_29_classification_filter_True_input_features_output_stone_states_data.pkl'

        # For anomaly runs of seed 29:
        # '/home/rsaha/projects/def-afyshe-ab/rsaha/dm_alchemy/src/data/complete_graph_preprocessed_separate_enhanced_qnodes_in_snodes/decompositional_chemistry_samples_167424_80_unique_stones_train_shop_2_qhop_1_seed_29_classification_filter_True_input_features_output_stone_states_data.pkl',
        # '/home/rsaha/projects/def-afyshe-ab/rsaha/dm_alchemy/src/data/complete_graph_preprocessed_separate_enhanced_qnodes_in_snodes/decompositional_chemistry_samples_167424_80_unique_stones_train_shop_2_qhop_1_seed_29_classification_filter_True_input_features_output_stone_states_data.pkl',
        # '/home/rsaha/projects/def-afyshe-ab/rsaha/dm_alchemy/src/data/complete_graph_preprocessed_separate_enhanced_qnodes_in_snodes/decompositional_chemistry_samples_167424_80_unique_stones_train_shop_2_qhop_1_seed_29_classification_filter_True_input_features_output_stone_states_data.pkl'
        ],

    3: [
        # '/home/rsaha/projects/def-afyshe-ab/rsaha/dm_alchemy/src/data/complete_graph_preprocessed_separate_enhanced_qnodes_in_snodes/decompositional_chemistry_samples_167424_80_unique_stones_train_shop_3_qhop_1_seed_0_classification_filter_True_input_features_output_stone_states_data.pkl',
        # '/home/rsaha/projects/def-afyshe-ab/rsaha/dm_alchemy/src/data/complete_graph_preprocessed_separate_enhanced_qnodes_in_snodes/decompositional_chemistry_samples_167424_80_unique_stones_train_shop_3_qhop_1_seed_3_classification_filter_True_input_features_output_stone_states_data.pkl', 
        # '/home/rsaha/projects/def-afyshe-ab/rsaha/dm_alchemy/src/data/complete_graph_preprocessed_separate_enhanced_qnodes_in_snodes/decompositional_chemistry_samples_167424_80_unique_stones_train_shop_3_qhop_1_seed_29_classification_filter_True_input_features_output_stone_states_data.pkl',
        # 'home/rsaha/projects/def-afyshe-ab/rsaha/dm_alchemy/src/data/complete_graph_preprocessed_separate_enhanced_qnodes_in_snodes/decompositional_chemistry_samples_167424_80_unique_stones_train_shop_3_qhop_1_seed_4_classification_filter_True_input_features_output_stone_states_data.pkl'

        # Good runs from wandb.
        '/home/rsaha/projects/def-afyshe-ab/rsaha/dm_alchemy/src/data/complete_graph_preprocessed_separate_enhanced_qnodes_in_snodes/decompositional_chemistry_samples_167424_80_unique_stones_train_shop_3_qhop_1_seed_0_classification_filter_True_input_features_output_stone_states_data.pkl',
        '/home/rsaha/projects/def-afyshe-ab/rsaha/dm_alchemy/src/data/complete_graph_preprocessed_separate_enhanced_qnodes_in_snodes/decompositional_chemistry_samples_167424_80_unique_stones_train_shop_3_qhop_1_seed_16_classification_filter_True_input_features_output_stone_states_data.pkl',
        '/home/rsaha/projects/def-afyshe-ab/rsaha/dm_alchemy/src/data/complete_graph_preprocessed_separate_enhanced_qnodes_in_snodes/decompositional_chemistry_samples_167424_80_unique_stones_train_shop_3_qhop_1_seed_29_classification_filter_True_input_features_output_stone_states_data.pkl'

        # Anomalous runs to show the effect of bad hyperparameters.

        # '/home/rsaha/projects/def-afyshe-ab/rsaha/dm_alchemy/src/data/complete_graph_preprocessed_separate_enhanced_qnodes_in_snodes/decompositional_chemistry_samples_167424_80_unique_stones_train_shop_3_qhop_1_seed_29_classification_filter_True_input_features_output_stone_states_data.pkl',
        # '/home/rsaha/projects/def-afyshe-ab/rsaha/dm_alchemy/src/data/complete_graph_preprocessed_separate_enhanced_qnodes_in_snodes/decompositional_chemistry_samples_167424_80_unique_stones_train_shop_3_qhop_1_seed_29_classification_filter_True_input_features_output_stone_states_data.pkl',
        # '/home/rsaha/projects/def-afyshe-ab/rsaha/dm_alchemy/src/data/complete_graph_preprocessed_separate_enhanced_qnodes_in_snodes/decompositional_chemistry_samples_167424_80_unique_stones_train_shop_3_qhop_1_seed_29_classification_filter_True_input_features_output_stone_states_data.pkl'
    ],
    4: [
        '/home/rsaha/projects/def-afyshe-ab/rsaha/dm_alchemy/src/data/complete_graph_preprocessed_separate_enhanced_qnodes_in_snodes/decompositional_chemistry_samples_167424_80_unique_stones_train_shop_4_qhop_1_seed_0_classification_filter_True_input_features_output_stone_states_data.pkl', 
        '/home/rsaha/projects/def-afyshe-ab/rsaha/dm_alchemy/src/data/complete_graph_preprocessed_separate_enhanced_qnodes_in_snodes/decompositional_chemistry_samples_167424_80_unique_stones_train_shop_4_qhop_1_seed_16_classification_filter_True_input_features_output_stone_states_data.pkl',
        '/home/rsaha/projects/def-afyshe-ab/rsaha/dm_alchemy/src/data/complete_graph_preprocessed_separate_enhanced_qnodes_in_snodes/decompositional_chemistry_samples_167424_80_unique_stones_train_shop_4_qhop_1_seed_29_classification_filter_True_input_features_output_stone_states_data.pkl'
    ],


    5: [
        '/home/rsaha/projects/def-afyshe-ab/rsaha/dm_alchemy/src/data/complete_graph_preprocessed_separate_enhanced_qnodes_in_snodes/decompositional_chemistry_samples_167424_80_unique_stones_train_shop_5_qhop_1_seed_0_classification_filter_True_input_features_output_stone_states_data.pkl',
        '/home/rsaha/projects/def-afyshe-ab/rsaha/dm_alchemy/src/data/complete_graph_preprocessed_separate_enhanced_qnodes_in_snodes/decompositional_chemistry_samples_167424_80_unique_stones_train_shop_5_qhop_1_seed_16_classification_filter_True_input_features_output_stone_states_data.pkl',
        '/home/rsaha/projects/def-afyshe-ab/rsaha/dm_alchemy/src/data/complete_graph_preprocessed_separate_enhanced_qnodes_in_snodes/decompositional_chemistry_samples_167424_80_unique_stones_train_shop_5_qhop_1_seed_29_classification_filter_True_input_features_output_stone_states_data.pkl'
        ]
}



decomposition_file_paths = {
    2: [
        '/home/rsaha/projects/def-afyshe-ab/rsaha/dm_alchemy/src/saved_models/complete_graph/scheduler_cosine/wd_0.01_lr_0.0001/eta_min_8e-05/xsmall/decoder/classification/input_features/output_stone_states/shop_2_qhop_1/seed_0/predictions/',
        '/home/rsaha/projects/def-afyshe-ab/rsaha/dm_alchemy/src/saved_models/complete_graph/scheduler_cosine/wd_0.01_lr_0.0001/eta_min_8e-05/xsmall/decoder/classification/input_features/output_stone_states/shop_2_qhop_1/seed_16/predictions/',
        '/home/rsaha/projects/def-afyshe-ab/rsaha/dm_alchemy/src/saved_models/complete_graph/scheduler_cosine/wd_0.1_lr_0.0001/eta_min_8e-05/xsmall/decoder/classification/input_features/output_stone_states/shop_2_qhop_1/seed_29/predictions/',


        # Anomalous runs to show the effect of bad hyperparameters.
        # '/home/rsaha/projects/def-afyshe-ab/rsaha/dm_alchemy/src/saved_models/complete_graph/scheduler_cosine/wd_0.01_lr_0.0001/eta_min_8e-05/xsmall/decoder/classification/input_features/output_stone_states/shop_2_qhop_1/seed_29/predictions/',
        # '/home/rsaha/projects/def-afyshe-ab/rsaha/dm_alchemy/src/saved_models/complete_graph/scheduler_cosine/wd_0.001_lr_0.0001/eta_min_8e-05/xsmall/decoder/classification/input_features/output_stone_states/shop_2_qhop_1/seed_29/predictions/',
        # '/home/rsaha/projects/def-afyshe-ab/rsaha/dm_alchemy/src/saved_models/complete_graph/scheduler_cosine/wd_0.1_lr_0.0001/eta_min_8e-05/xsmall/decoder/classification/input_features/output_stone_states/shop_2_qhop_1/seed_29/predictions/'
        ],

        # NOTE: Why are there so many files for 3-hop decomposition? It's because we wanted to see if different hyperparameter settings made a difference in when the final phase was being learned and if there was overlap with other stages.

    3: [
        # '/home/rsaha/projects/def-afyshe-ab/rsaha/dm_alchemy/src/saved_models/complete_graph/scheduler_cosine/wd_0.1_lr_0.0001/eta_min_8.5e-05/xsmall/decoder/classification/input_features/output_stone_states/shop_3_qhop_1/seed_0/predictions', 
        # '/home/rsaha/projects/def-afyshe-ab/rsaha/dm_alchemy/src/saved_models/complete_graph/scheduler_cosine/wd_0.1_lr_0.0001/eta_min_8e-05/xsmall/decoder/classification/input_features/output_stone_states/shop_3_qhop_1/seed_3/predictions',
        # '/home/rsaha/projects/def-afyshe-ab/rsaha/dm_alchemy/src/saved_models/complete_graph/scheduler_cosine/wd_0.01_lr_0.0001/eta_min_8e-05/xsmall/decoder/classification/input_features/output_stone_states/shop_3_qhop_1/seed_3/predictions/',
        # '/home/rsaha/projects/def-afyshe-ab/rsaha/dm_alchemy/src/saved_models/complete_graph/scheduler_cosine_restarts/wd_0.1_lr_0.0001/eta_min_1e-05/xsmall/decoder/classification/input_features/output_stone_states/shop_3_qhop_1/seed_0/predictions/',
        # '/home/rsaha/projects/def-afyshe-ab/rsaha/dm_alchemy/src/saved_models/complete_graph/scheduler_cosine/wd_0.01_lr_0.0001/eta_min_8e-05/xsmall/decoder/classification/input_features/output_stone_states/shop_3_qhop_1/seed_3/predictions/',
        # '/home/rsaha/projects/def-afyshe-ab/rsaha/dm_alchemy/src/saved_models/complete_graph/scheduler_cosine/wd_0.001_lr_0.0001/eta_min_7e-05/xsmall/decoder/classification/input_features/output_stone_states/shop_3_qhop_1/seed_0/predictions/',
        # '/home/rsaha/projects/def-afyshe-ab/rsaha/dm_alchemy/src/saved_models/complete_graph/scheduler_cosine_restarts/wd_0.1_lr_0.0001/eta_min_1e-05/xsmall/decoder/classification/input_features/output_stone_states/shop_3_qhop_1/seed_29/predictions/',
        # '/home/rsaha/projects/def-afyshe-ab/rsaha/dm_alchemy/src/saved_models/complete_graph/scheduler_cosine_restarts/wd_0.1_lr_0.0001/eta_min_1e-05/xsmall/decoder/classification/input_features/output_stone_states/shop_3_qhop_1/seed_4/predictions'

        # 3-hop anomalous runs to show the effect of bad hyperparameters.
        # '/home/rsaha/projects/def-afyshe-ab/rsaha/dm_alchemy/src/saved_models/complete_graph/scheduler_cosine/wd_0.1_lr_0.0001/eta_min_8e-05/xsmall/decoder/classification/input_features/output_stone_states/shop_3_qhop_1/seed_29/predictions/',
        # '/home/rsaha/projects/def-afyshe-ab/rsaha/dm_alchemy/src/saved_models/complete_graph/scheduler_cosine/wd_0.01_lr_0.0001/eta_min_8e-05/xsmall/decoder/classification/input_features/output_stone_states/shop_3_qhop_1/seed_29/predictions/',
        # '/home/rsaha/projects/def-afyshe-ab/rsaha/dm_alchemy/src/saved_models/complete_graph/scheduler_cosine/wd_0.001_lr_0.0001/eta_min_8e-05/xsmall/decoder/classification/input_features/output_stone_states/shop_3_qhop_1/seed_29/predictions/'


        # Control runs:
        # '/home/rsaha/projects/def-afyshe-ab/rsaha/dm_alchemy/src/saved_models/complete_graph/scheduler_cosine/wd_0.01_lr_0.0001/eta_min_9e-05/xsmall/decoder/classification/input_features/output_stone_states/shop_3_qhop_1/seed_0/predictions/'

        # Good runs from wandb.
        '/home/rsaha/projects/def-afyshe-ab/rsaha/dm_alchemy/src/saved_models/complete_graph/scheduler_cosine/wd_0.1_lr_0.0001/eta_min_7e-05/xsmall/decoder/classification/input_features/output_stone_states/shop_3_qhop_1/seed_0/predictions/',
        '/home/rsaha/projects/def-afyshe-ab/rsaha/dm_alchemy/src/saved_models/complete_graph/scheduler_cosine/wd_0.1_lr_0.0001/eta_min_8e-05/xsmall/decoder/classification/input_features/output_stone_states/shop_3_qhop_1/seed_16/predictions/',
        '/home/rsaha/projects/def-afyshe-ab/rsaha/dm_alchemy/src/saved_models/complete_graph/scheduler_cosine_restarts/wd_0.1_lr_0.0001/eta_min_8e-05/xsmall/decoder/classification/input_features/output_stone_states/shop_3_qhop_1/seed_29/predictions/',

        ],

    4: [
        '/home/rsaha/projects/def-afyshe-ab/rsaha/dm_alchemy/src/saved_models/complete_graph/scheduler_cosine_restarts/wd_0.1_lr_0.0001/eta_min_8e-05/xsmall/decoder/classification/input_features/output_stone_states/shop_4_qhop_1/seed_0/predictions/',
        '/home/rsaha/projects/def-afyshe-ab/rsaha/dm_alchemy/src/saved_models/complete_graph/scheduler_cosine/wd_0.1_lr_0.0001/eta_min_8e-05/xsmall/decoder/classification/input_features/output_stone_states/shop_4_qhop_1/seed_16/predictions/',
        '/home/rsaha/projects/def-afyshe-ab/rsaha/dm_alchemy/src/saved_models/complete_graph/scheduler_cosine/wd_0.1_lr_0.0001/eta_min_8e-05/xsmall/decoder/classification/input_features/output_stone_states/shop_4_qhop_1/seed_29/predictions/'],

    5: [
        '/home/rsaha/projects/def-afyshe-ab/rsaha/dm_alchemy/src/saved_models/complete_graph/scheduler_cosine_restarts/wd_0.1_lr_0.0001/eta_min_8e-05/xsmall/decoder/classification/input_features/output_stone_states/shop_5_qhop_1/seed_0/predictions/',
        # '/home/rsaha/projects/def-afyshe-ab/rsaha/dm_alchemy/src/saved_models/complete_graph/scheduler_cosine/wd_0.1_lr_0.0001/eta_min_8e-05/xsmall/decoder/classification/input_features/output_stone_states/shop_5_qhop_1/seed_2/predictions/',
        # '/home/rsaha/projects/def-afyshe-ab/rsaha/dm_alchemy/src/saved_models/complete_graph/scheduler_cosine_restarts/wd_0.1_lr_0.0001/eta_min_8e-05/xsmall/decoder/classification/input_features/output_stone_states/shop_5_qhop_1/seed_16/predictions',
        '/home/rsaha/projects/def-afyshe-ab/rsaha/dm_alchemy/src/saved_models/complete_graph/scheduler_cosine/wd_0.1_lr_0.0001/eta_min_8e-05/xsmall/decoder/classification/input_features/output_stone_states/shop_5_qhop_1/seed_16/predictions/',
        '/home/rsaha/projects/def-afyshe-ab/rsaha/dm_alchemy/src/saved_models/complete_graph/scheduler_cosine/wd_0.1_lr_0.0001/eta_min_8e-05/xsmall/decoder/classification/input_features/output_stone_states/shop_5_qhop_1/seed_29/predictions/'
        
        
        
        # Don't use
        # '/home/rsaha/projects/def-afyshe-ab/rsaha/dm_alchemy/src/saved_models/complete_graph/scheduler_cosine_restarts/wd_0.1_lr_0.0001/eta_min_8e-05/xsmall/decoder/classification/input_features/output_stone_states/shop_5_qhop_1/seed_16/predictions/',
        ]
}


composition_file_paths_non_subsampled = {
    2: [
        '/home/rsaha/projects/def-afyshe-ab/rsaha/dm_alchemy/src/data/complete_graphs_composition_non_subsampled_grouped_by_unique_end_state_preprocessed/compositional_chemistry_samples_167424_80_unique_stones_val_shop_1_qhop_2_seed_0_classification_filter_True_input_features_output_stone_states_data.pkl',
        '/home/rsaha/projects/def-afyshe-ab/rsaha/dm_alchemy/src/data/complete_graphs_composition_non_subsampled_grouped_by_unique_end_state_preprocessed/compositional_chemistry_samples_167424_80_unique_stones_val_shop_1_qhop_2_seed_16_classification_filter_True_input_features_output_stone_states_data.pkl',
        '/home/rsaha/projects/def-afyshe-ab/rsaha/dm_alchemy/src/data/complete_graphs_composition_non_subsampled_grouped_by_unique_end_state_preprocessed/compositional_chemistry_samples_167424_80_unique_stones_val_shop_1_qhop_2_seed_29_classification_filter_True_input_features_output_stone_states_data.pkl'
        ],
    # 2: [
    #     '',
    #     ''
    # ],
    
    3: [
        '/home/rsaha/projects/def-afyshe-ab/rsaha/dm_alchemy/src/data/complete_graphs_composition_non_subsampled_grouped_by_unique_end_state_preprocessed/compositional_chemistry_samples_167424_80_unique_stones_val_shop_1_qhop_3_seed_0_classification_filter_True_input_features_output_stone_states_data.pkl',
        '/home/rsaha/projects/def-afyshe-ab/rsaha/dm_alchemy/src/data/complete_graphs_composition_non_subsampled_grouped_by_unique_end_state_preprocessed/compositional_chemistry_samples_167424_80_unique_stones_val_shop_1_qhop_3_seed_16_classification_filter_True_input_features_output_stone_states_data.pkl',
        '/home/rsaha/projects/def-afyshe-ab/rsaha/dm_alchemy/src/data/complete_graphs_composition_non_subsampled_grouped_by_unique_end_state_preprocessed/compositional_chemistry_samples_167424_80_unique_stones_val_shop_1_qhop_3_seed_29_classification_filter_True_input_features_output_stone_states_data.pkl'
        ],
    4: ['/home/rsaha/projects/def-afyshe-ab/rsaha/dm_alchemy/src/data/complete_graphs_composition_non_subsampled_grouped_by_unique_end_state_preprocessed/compositional_chemistry_samples_167424_80_unique_stones_val_shop_1_qhop_4_seed_0_classification_filter_True_input_features_output_stone_states_data.pkl',
        '/home/rsaha/projects/def-afyshe-ab/rsaha/dm_alchemy/src/data/complete_graphs_composition_non_subsampled_grouped_by_unique_end_state_preprocessed/compositional_chemistry_samples_167424_80_unique_stones_val_shop_1_qhop_4_seed_16_classification_filter_True_input_features_output_stone_states_data.pkl',
        '/home/rsaha/projects/def-afyshe-ab/rsaha/dm_alchemy/src/data/complete_graphs_composition_non_subsampled_grouped_by_unique_end_state_preprocessed/compositional_chemistry_samples_167424_80_unique_stones_val_shop_1_qhop_4_seed_29_classification_filter_True_input_features_output_stone_states_data.pkl'
        ],
    5: ['/home/rsaha/projects/def-afyshe-ab/rsaha/dm_alchemy/src/data/complete_graphs_composition_non_subsampled_grouped_by_unique_end_state_preprocessed/compositional_chemistry_samples_167424_80_unique_stones_val_shop_1_qhop_5_seed_0_classification_filter_True_input_features_output_stone_states_data.pkl',
        '/home/rsaha/projects/def-afyshe-ab/rsaha/dm_alchemy/src/data/complete_graphs_composition_non_subsampled_grouped_by_unique_end_state_preprocessed/compositional_chemistry_samples_167424_80_unique_stones_val_shop_1_qhop_5_seed_16_classification_filter_True_input_features_output_stone_states_data.pkl',
        '/home/rsaha/projects/def-afyshe-ab/rsaha/dm_alchemy/src/data/complete_graphs_composition_non_subsampled_grouped_by_unique_end_state_preprocessed/compositional_chemistry_samples_167424_80_unique_stones_val_shop_1_qhop_5_seed_29_classification_filter_True_input_features_output_stone_states_data.pkl'
        ]
}

composition_file_paths = {
    # 2: ['/home/rsaha/projects/def-afyshe-ab/rsaha/dm_alchemy/src/saved_models/complete_graph/fully_shuffled/scheduler_step_lr/wd_0.001_lr_0.0001/step_size_165_gamma_0.2/xsmall/decoder/classification/input_features/output_stone_states/shop_1_qhop_2/seed_0/predictions',
    #     '/home/rsaha/projects/def-afyshe-ab/rsaha/dm_alchemy/src/saved_models/complete_graph/fully_shuffled/no_scheduler/wd_0.01_lr_0.0001/xsmall/decoder/classification/input_features/output_stone_states/shop_1_qhop_2/seed_16/predictions', 
    #     '/home/rsaha/projects/def-afyshe-ab/rsaha/dm_alchemy/src/saved_models/complete_graph/fully_shuffled/scheduler_step_lr/wd_0.001_lr_0.0001/step_size_250_gamma_0.4/xsmall/decoder/classification/input_features/output_stone_states/shop_1_qhop_2/seed_29/predictions'
    #     ],

    2: [
        # '/home/rsaha/projects/aip-afyshe/rsaha/dm_alchemy/src/saved_models/complete_graph/fully_shuffled/scheduler_cosine/wd_0.001_lr_0.0001/eta_min_7e-05/xsmall/decoder/classification/input_features/output_stone_states/shop_1_qhop_2/seed_0/predictions/',
        '/home/rsaha/projects/aip-afyshe/rsaha/dm_alchemy/src/saved_models/complete_graph/fully_shuffled/scheduler_cosine/wd_0.001_lr_0.0001/eta_min_7e-05/xsmall/decoder/classification/input_features/output_stone_states/shop_1_qhop_2/seed_0/flatten_linear_input/predictions/',
        '/home/rsaha/projects/aip-afyshe/rsaha/dm_alchemy/src/saved_models/complete_graph/fully_shuffled/no_scheduler/wd_0.01_lr_0.0001/eta_min_1e-05/xsmall/decoder/classification/input_features/output_stone_states/shop_1_qhop_2/seed_16/flatten_linear_input/predictions/',
        # '/home/rsaha/projects/aip-afyshe/rsaha/dm_alchemy/src/saved_models/complete_graph/fully_shuffled/scheduler_cosine/wd_0.001_lr_0.0001/eta_min_9.5e-05/xsmall/decoder/classification/input_features/output_stone_states/shop_1_qhop_2/seed_29/flatten_linear_input/predictions/',
        '/home/rsaha/projects/aip-afyshe/rsaha/dm_alchemy/src/saved_models/complete_graph/fully_shuffled/scheduler_cosine/wd_0.001_lr_0.0001/eta_min_9.5e-05/xsmall/decoder/classification/input_features/output_stone_states/shop_1_qhop_2/seed_29/flatten_linear_input/predictions/'
    ],
    3: [
        '/home/rsaha/projects/def-afyshe-ab/rsaha/dm_alchemy/src/saved_models/complete_graph/fully_shuffled/scheduler_cosine/xsmall/decoder/classification/input_features/output_stone_states/shop_1_qhop_3/seed_0/predictions', 
        '/home/rsaha/projects/def-afyshe-ab/rsaha/dm_alchemy/src/saved_models/complete_graph/fully_shuffled/scheduler_cosine/xsmall/decoder/classification/input_features/output_stone_states/shop_1_qhop_3/seed_16/predictions', 
        '/home/rsaha/projects/def-afyshe-ab/rsaha/dm_alchemy/src/saved_models/complete_graph/fully_shuffled/no_scheduler/wd_0.1_lr_0.0001/xsmall/decoder/classification/input_features/output_stone_states/shop_1_qhop_3/seed_29/predictions'],

    4: ['/home/rsaha/projects/def-afyshe-ab/rsaha/dm_alchemy/src/saved_models/complete_graph/fully_shuffled/no_scheduler/wd_0.01_lr_0.0001/xsmall/decoder/classification/input_features/output_stone_states/shop_1_qhop_4/seed_0/predictions', 
        '/home/rsaha/projects/def-afyshe-ab/rsaha/dm_alchemy/src/saved_models/complete_graph/fully_shuffled/no_scheduler/wd_0.01_lr_0.0001/xsmall/decoder/classification/input_features/output_stone_states/shop_1_qhop_4/seed_16/predictions', 
        '/home/rsaha/projects/def-afyshe-ab/rsaha/dm_alchemy/src/saved_models/complete_graph/fully_shuffled/scheduler_cosine/wd_0.01_lr_0.0001/xsmall/decoder/classification/input_features/output_stone_states/shop_1_qhop_4/seed_29/predictions'],

    5: ['/home/rsaha/projects/def-afyshe-ab/rsaha/dm_alchemy/src/saved_models/complete_graph/fully_shuffled/scheduler_cosine_restarts/wd_0.001_lr_0.0001/eta_min_8e-05/xsmall/decoder/classification/input_features/output_stone_states/shop_1_qhop_5/seed_0/predictions',
        '/home/rsaha/projects/def-afyshe-ab/rsaha/dm_alchemy/src/saved_models/complete_graph/fully_shuffled/no_scheduler/wd_0.1_lr_0.0001/xsmall/decoder/classification/input_features/output_stone_states/shop_1_qhop_5/seed_16/predictions',
        '/home/rsaha/projects/def-afyshe-ab/rsaha/dm_alchemy/src/saved_models/complete_graph/fully_shuffled/no_scheduler/wd_0.01_lr_0.0001/xsmall/decoder/classification/input_features/output_stone_states/shop_1_qhop_5/seed_29/predictions']
}

held_out_file_paths = {
    # Normalized reward paths.
    # 4: [
    #     "/home/rsaha/projects/def-afyshe-ab/rsaha/dm_alchemy/src/saved_models/held_out_color_exp/same_reward_held_out_color_4/all_graphs/scheduler_cosine/wd_0.1_lr_0.0001/eta_min_9e-05/xsmall/decoder/classification/input_features/output_stone_states/shop_1_qhop_1/seed_0/predictions/",
    #     "/home/rsaha/projects/def-afyshe-ab/rsaha/dm_alchemy/src/saved_models/held_out_color_exp/same_reward_held_out_color_4/all_graphs/scheduler_cosine/wd_0.1_lr_0.0001/eta_min_7e-05/xsmall/decoder/classification/input_features/output_stone_states/shop_1_qhop_1/seed_3/predictions/",
    #     # ""
    # ]
    4: [
        "/home/rsaha/projects/def-afyshe-ab/rsaha/dm_alchemy/src/saved_models/held_out_color_exp/held_out_edges_4/all_graphs/scheduler_cosine/wd_0.01_lr_0.0001/eta_min_9.5e-05/xsmall/decoder/classification/input_features/output_stone_states/shop_1_qhop_1/seed_0/predictions/",
        "/home/rsaha/projects/def-afyshe-ab/rsaha/dm_alchemy/src/saved_models/held_out_color_exp/held_out_edges_4/all_graphs/scheduler_cosine/wd_0.001_lr_0.0001/eta_min_7e-05/xsmall/decoder/classification/input_features/output_stone_states/shop_1_qhop_1/seed_2/predictions/",
        "/home/rsaha/projects/def-afyshe-ab/rsaha/dm_alchemy/src/saved_models/held_out_color_exp/held_out_edges_4/all_graphs/no_scheduler/wd_0.01_lr_0.0001/eta_min_1e-05/xsmall/decoder/classification/input_features/output_stone_states/shop_1_qhop_1/seed_3/predictions/"
    ]
}



predictions_by_epoch_by_seed, inputs_by_seed, non_subsampled_composition_data  = load_epoch_data(
    exp_typ = exp_typ,
    hop = hop,
    epoch_range = (hop_to_epoch_values[hop][0], hop_to_epoch_values[hop][-1]),
    seeds = seed_values_hop_dict[hop],
    scheduler_prefix = scheduler_prefix,
    file_paths = composition_file_paths[hop] if exp_typ == 'composition' else decomposition_file_paths[hop] if exp_typ == 'decomposition' else held_out_file_paths[hop] if exp_typ == 'held_out' else None,
    file_paths_non_subsampled = composition_file_paths_non_subsampled if exp_typ == 'composition' else decomposition_file_paths_non_subsampled if exp_typ == 'decomposition' else None,
)
# import pdb; pdb.set_trace()
seed_data_files = {}
for seed in predictions_by_epoch_by_seed.keys():
    if exp_typ == 'held_out':
        if not args.normalized_reward:
            data_files = {
                "vocab": f"/home/rsaha/projects/{infix}dm_alchemy/src/data/shuffled_held_out_exps_preprocessed_separate_enhanced/compositional_chemistry_samples_167424_80_unique_stones_val_shop_1_qhop_1_single_held_out_color_4_edges_exp_seed_{seed}_classification_filter_True_input_features_output_stone_states_vocab.pkl",
                "metadata": f"/home/rsaha/projects/{infix}dm_alchemy/src/data/shuffled_held_out_exps_preprocessed_separate_enhanced/compositional_chemistry_samples_167424_80_unique_stones_val_shop_1_qhop_1_single_held_out_color_4_edges_exp_seed_{seed}_classification_filter_True_input_features_output_stone_states_metadata.json"
            }
        else:
            data_files = {
                "vocab": f"/home/rsaha/projects/{infix}dm_alchemy/src/data/same_reward_shuffled_held_out_exps_preprocessed_separate_enhanced/normalized_compositional_chemistry_samples_167424_80_unique_stones_val_shop_1_qhop_1_single_held_out_color_4_edges_exp_seed_{seed}_classification_filter_True_input_features_output_stone_states_vocab.pkl",
                "metadata": f"/home/rsaha/projects/{infix}dm_alchemy/src/data/same_reward_shuffled_held_out_exps_preprocessed_separate_enhanced/normalized_compositional_chemistry_samples_167424_80_unique_stones_val_shop_1_qhop_1_single_held_out_color_4_edges_exp_seed_{seed}_classification_filter_True_input_features_output_stone_states_metadata.json"
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



if args.get_adjacency_analysis_only:
    # Run only adjacency analysis and then append results to a list for each seed, and then average across seeds.
    adjacency_results_by_seed = {}
    for seed in predictions_by_epoch_by_seed.keys():
        print(f"\n\nAnalyzing seed {seed} for adjacency analysis...")
        predictions_by_epoch = predictions_by_epoch_by_seed[seed]
        data_with_predictions = inputs_by_seed[seed]
        
        # Load the correct vocab for this seed
        vocab = seed_data_files[seed]['vocab']
        
        # Run the analysis
        assert exp_typ == 'held_out', "Adjacency analysis is only implemented for held-out experiments."
        print("Running adjacency behavior analysis")
        adjacency_results = analyze_adjacency_behavior(
            data_with_predictions, 
            vocab, 
            vocab['stone_state_to_id'], 
            predictions_by_epoch,
        )

        adjacency_results_by_seed[seed] = adjacency_results

    # Now average the adjacency results for each metric across seeds.
    averaged_adjacency_results = {}
    std_errors_adjacency = {}
    adjacency_metrics = ['within_reachable_acc', 'correct_within_reachable_acc', 'within_true_adjacent_acc', 'within_connected_acc', 'within_connected_in_reachable_acc']

    """
    for epoch in sorted(adjacency_metrics.keys()):
        for reward in reachable_reward_mapping.keys():
            metrics = adjacency_metrics[epoch][reward]
            
            # Accuracy 1: P(reachable | in-support)
            if metrics['total_in_support'] > 0:
                reachable_acc = metrics['in_support_and_reachable'] / metrics['total_in_support']
            else:
                reachable_acc = 0
            adjacency_accuracies[reward]['within_reachable_acc'].append(reachable_acc)
            
            # Accuracy 2: P(exact target | reachable)
            if metrics['total_reachable'] > 0:
                correct_acc = metrics['reachable_and_correct'] / metrics['total_reachable']
            else:
                correct_acc = 0
            adjacency_accuracies[reward]['correct_within_reachable_acc'].append(correct_acc)
    
    return adjacency_accuracies
    
    """

    # for each reward in the adjacency metrics, average across seeds for that reward across epochs and all seeds.
    for metric in adjacency_metrics:
        averaged_adjacency_results[metric] = {}
        std_errors_adjacency[metric] = {}
        # import pdb; pdb.set_trace()
        for reward in adjacency_results_by_seed[list(adjacency_results_by_seed.keys())[0]].keys():
            # Collect all accuracies for this reward across seeds
            all_accuracies = []
            for seed in adjacency_results_by_seed.keys():
                all_accuracies.append(adjacency_results_by_seed[seed][reward][metric])
            # Compute average and std error
            averaged_adjacency_results[metric][reward] = np.mean(all_accuracies, axis=0)
            std_errors_adjacency[metric][reward] = np.std(all_accuracies, axis=0) / np.sqrt(len(all_accuracies))
    
    # Now plot for each reward_bin, the respective accuracies with error bars.
    # there will be four subplots, one for each reward bin and each subplot will have two lines, one for within_reachable_acc and one for correct_within_reachable_acc.

    import matplotlib.pyplot as plt
    rewards = ['-3', '-1', '1', '3']
    reward_title_mapping = {
        '-3': '-3',
        '-1': '-1',
        '1': '1',
        '3': '+15',
    }

    fig, axs = plt.subplots(1, len(rewards), figsize=(20, 5))
    for i, reward in enumerate(rewards):
        mean_within = averaged_adjacency_results['within_reachable_acc'][reward]
        std_within = std_errors_adjacency['within_reachable_acc'][reward]
        epochs_range = range(len(mean_within))
        
        axs[i].plot(epochs_range, mean_within, label='Within Reward Adjacent', color='blue')
        axs[i].fill_between(epochs_range, 
                   np.array(mean_within) - np.array(std_within),
                   np.array(mean_within) + np.array(std_within),
                   alpha=0.2, color='blue')
        
        # [NEW] Plot True Adjacent Accuracy
        mean_true = averaged_adjacency_results['within_true_adjacent_acc'][reward]
        std_true = std_errors_adjacency['within_true_adjacent_acc'][reward]
        
        axs[i].plot(epochs_range, mean_true, label='Within True Adjacent (Graph)', color='orange')
        axs[i].fill_between(epochs_range,
                   np.array(mean_true) - np.array(std_true),
                   np.array(mean_true) + np.array(std_true),
                   alpha=0.2, color='orange')

        mean_correct = averaged_adjacency_results['correct_within_reachable_acc'][reward]
        std_correct = std_errors_adjacency['correct_within_reachable_acc'][reward]
        
        axs[i].plot(epochs_range, mean_correct, label='Correct in Reward Adjacent', linestyle='--', color='skyblue')
        axs[i].fill_between(epochs_range,
                   np.array(mean_correct) - np.array(std_correct),
                   np.array(mean_correct) + np.array(std_correct),
                   alpha=0.2, color='skyblue')


        # Plot the connected accuracy
        mean_connected = averaged_adjacency_results['within_connected_acc'][reward]
        std_connected = std_errors_adjacency['within_connected_acc'][reward]
        axs[i].plot(epochs_range, mean_connected, label='Within Adjacent in-support', color='red') 
        axs[i].fill_between(epochs_range,
                   np.array(mean_connected) - np.array(std_connected),
                   np.array(mean_connected) + np.array(std_connected),
                   alpha=0.2, color='red')

        # Plot the connected in reachable accuracy
        # mean_connected_in_reachable = averaged_adjacency_results['within_connected_in_reachable_acc'][reward]
        # std_connected_in_reachable = std_errors_adjacency['within_connected_in_reachable_acc'][reward]
        # axs[i].plot(epochs_range, mean_connected_in_reachable, label='Within Connected in Reachable', linestyle='--', color='black')
        # axs[i].fill_between(epochs_range,
        #            np.array(mean_connected_in_reachable) - np.array(std_connected_in_reachable),
        #            np.array(mean_connected_in_reachable) + np.array(std_connected_in_reachable),
        #            alpha=0.2, color='black')



        axs[i].set_title(f'Reward: {reward_title_mapping[reward]}')
        axs[i].set_xlabel('Epochs')
        axs[i].set_ylabel('Accuracy')
        axs[i].legend(fontsize=9, loc='center right')
        axs[i].set_ylim(0, 1)
        # Add gridlines.
        axs[i].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'nov_21_adjacency_analysis_hop_{hop}_exp_{exp_typ}_adjacent_true_adjacent_connected_within.png')
    plt.savefig(f'nov_21_adjacency_analysis_hop_{hop}_exp_{exp_typ}_adjacent_true_adjacent_connected_within.pdf', bbox_inches='tight')
    plt.close()


    exit(0)





if args.get_non_support_analysis_only:
    non_support_accuracies_by_seed = {}
    non_support_metrics_by_seed = {}

    for seed in predictions_by_epoch_by_seed.keys():
        print(f"\n\nAnalyzing seed {seed} for non-support transitions...")
        predictions_by_epoch = predictions_by_epoch_by_seed[seed]
        data_with_predictions = inputs_by_seed[seed]
        vocab = seed_data_files[seed]['vocab']

        non_support_accuracies, non_support_metrics = analyze_non_support_transition_behavior(
            data_with_predictions,
            vocab,
            vocab['stone_state_to_id'],
            predictions_by_epoch,
            exp_typ=exp_typ,
            hop=hop
        )
        non_support_accuracies_by_seed[seed] = non_support_accuracies
        non_support_metrics_by_seed[seed] = non_support_metrics  # kept in case you still need counts

    # ------------------------------------------------------
    # Average accuracies across seeds for each epoch & reward
    # ------------------------------------------------------
    rewards = ['-3', '-1', '1', '3']

    # infer number of epochs from one seed
    example_seed = next(iter(non_support_accuracies_by_seed.keys()))
    num_epochs = len(next(iter(non_support_accuracies_by_seed[example_seed].values()))['p_pred_in_non_support'])
    epochs_range = range(num_epochs)

    # structure: averaged_non_support[reward]['p_pred_in_non_support'] -> [mean per epoch]
    averaged_non_support = {
        reward: {
            'p_pred_in_non_support': [],
            'p_pred_in_non_support_std': [],
            'p_correct_given_non_support': [],
            'p_correct_given_non_support_std': [],
        }
        for reward in rewards
    }

    

    for reward in rewards:
        # collect per-seed arrays for this reward
        pred_in_non_support_seed = []
        correct_given_non_support_seed = []

        for seed, accs_per_reward in non_support_accuracies_by_seed.items():
            pred_in_non_support_seed.append(
                np.array(accs_per_reward[reward]['p_pred_in_non_support'], dtype=float)
            )
            correct_given_non_support_seed.append(
                np.array(accs_per_reward[reward]['p_correct_given_non_support'], dtype=float)
            )

        pred_in_non_support_seed = np.stack(pred_in_non_support_seed, axis=0)      # (n_seeds, n_epochs)
        correct_given_non_support_seed = np.stack(correct_given_non_support_seed, axis=0)

        mean_pred_in_non = pred_in_non_support_seed.mean(axis=0)
        std_pred_in_non = pred_in_non_support_seed.std(axis=0) / np.sqrt(pred_in_non_support_seed.shape[0])

        mean_correct_non = correct_given_non_support_seed.mean(axis=0)
        std_correct_non = correct_given_non_support_seed.std(axis=0) / np.sqrt(correct_given_non_support_seed.shape[0])




        averaged_non_support[reward]['p_pred_in_non_support'] = mean_pred_in_non
        averaged_non_support[reward]['p_pred_in_non_support_std'] = std_pred_in_non
        averaged_non_support[reward]['p_correct_given_non_support'] = mean_correct_non
        averaged_non_support[reward]['p_correct_given_non_support_std'] = std_correct_non

    # ------------------------------------------------------
    # Plot: one subplot per reward, two accuracy curves per subplot
    # ------------------------------------------------------
    fig, axs = plt.subplots(1, len(rewards), figsize=(20, 5))
    if len(rewards) == 1:
        axs = [axs]

    reward_title_mapping = {
        '-3': '-3',
        '-1': '-1',
        '1': '1',
        '3': '+15',
    }

    for i, reward in enumerate(rewards):
        ax = axs[i]
        mean_p_non = averaged_non_support[reward]['p_pred_in_non_support']
        std_p_non = averaged_non_support[reward]['p_pred_in_non_support_std']

        mean_p_correct = averaged_non_support[reward]['p_correct_given_non_support']
        std_p_correct = averaged_non_support[reward]['p_correct_given_non_support_std']

        ax.plot(epochs_range, mean_p_non, label='P(pred in non-support transitions)', color='tab:blue')
        ax.fill_between(
            epochs_range,
            mean_p_non - std_p_non,
            mean_p_non + std_p_non,
            color='tab:blue',
            alpha=0.2
        )

        ax.plot(epochs_range, mean_p_correct, label='P(correct | non-support transitions)', color='tab:orange')
        ax.fill_between(
            epochs_range,
            mean_p_correct - std_p_correct,
            mean_p_correct + std_p_correct,
            color='tab:orange',
            alpha=0.2
        )

        ax.set_title(f'Reward: {reward_title_mapping[reward]}')
        ax.set_xlabel('Epoch')
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.set_ylabel('Accuracy')

        ax.legend(loc='lower right')
        # Set y-axis limits to [0, 1]
        ax.set_ylim(0, 1)


    plt.tight_layout()
    plt.savefig(f'non_support_accuracy_hop_{hop}_exp_{exp_typ}.png')
    plt.savefig(f'non_support_accuracy_hop_{hop}_exp_{exp_typ}.pdf', bbox_inches='tight')
    plt.close()

    exit(0)








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
    print("Running half-chemistry behavior analysis")
    # model_selection_results = analyze_model_selection_behavior(
    #     data_with_predictions, 
    #     vocab, 
    #     vocab['stone_state_to_id'], 
    #     predictions_by_epoch
    # )

    # Get half_chemistry_analysis results.
    half_chemistry_results = analyze_half_chemistry_behaviour(
        data_with_predictions, vocab, vocab['stone_state_to_id'], predictions_by_epoch, exp_typ=exp_typ, hop=hop,
        composition_full_target_data = non_subsampled_composition_data[seed] if non_subsampled_composition_data is not None else None
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
        (
            predicted_in_context_accuracies,
            predicted_in_context_correct_half_accuracies,
            predicted_in_context_other_half_accuracies,
            predicted_in_context_correct_half_exact_accuracies,
            predicted_correct_within_context,
            predicted_exact_out_of_all_108,
            predicted_in_adjacent_and_correct_half_accuracies,
            predicted_correct_half_within_adjacent_and_correct_half_accuracies,
            query_start_stone_reward_binning_analysis,
        ) = half_chemistry_results

        complete_query_stone_state_per_reward_binned_accuracy, within_support_query_stone_state_per_reward_binned_accuracy, within_support_within_half_query_stone_state_per_reward_binned_accuracy = query_start_stone_reward_binning_analysis

        # Store results for this seed
        seed_results[seed] = {
            'predicted_in_context_accuracies': predicted_in_context_accuracies,
            'predicted_in_context_correct_half_accuracies': predicted_in_context_correct_half_accuracies,
            'predicted_in_context_other_half_accuracies': predicted_in_context_other_half_accuracies,
            'predicted_in_context_correct_half_exact_accuracies': predicted_in_context_correct_half_exact_accuracies,
            'predicted_correct_within_context': predicted_correct_within_context,
            'predicted_exact_out_of_all_108': predicted_exact_out_of_all_108,
            'predicted_in_adjacent_and_correct_half_accuracies': predicted_in_adjacent_and_correct_half_accuracies,
            'predicted_correct_half_within_adjacent_and_correct_half_accuracies': predicted_correct_half_within_adjacent_and_correct_half_accuracies,
            'complete_query_stone_state_per_reward_binned_accuracy': complete_query_stone_state_per_reward_binned_accuracy,
            'within_support_query_stone_state_per_reward_binned_accuracy': within_support_query_stone_state_per_reward_binned_accuracy,
            'within_support_within_half_query_stone_state_per_reward_binned_accuracy': within_support_within_half_query_stone_state_per_reward_binned_accuracy
        }






# Now the plotting begins. First we need to average the result for each metric across seeds.
# Average results across seeds
averaged_results = {}
std_errors = {}
individual_seed_results = {}
if exp_typ == 'decomposition':
    metrics = ['predicted_in_context_accuracies', 'predicted_in_context_correct_half_accuracies', 'predicted_in_context_other_half_accuracies', 'predicted_in_context_correct_half_exact_accuracies', 'predicted_correct_within_context', 'predicted_exact_out_of_all_108',
    'predicted_in_adjacent_and_correct_half_accuracies', 'predicted_correct_half_within_adjacent_and_correct_half_accuracies']
elif exp_typ == 'held_out':
    metrics = ['predicted_in_context_accuracies', 'predicted_in_context_correct_half_accuracies', 'predicted_in_context_other_half_accuracies', 'predicted_in_context_correct_half_exact_accuracies', 'predicted_correct_within_context', 'predicted_exact_out_of_all_108']
    if args.reward_binning_analysis_only:
        metrics = ['complete_query_stone_state_per_reward_binned_accuracy', 'within_support_query_stone_state_per_reward_binned_accuracy', 'within_support_within_half_query_stone_state_per_reward_binned_accuracy']
else:
    metrics = ['predicted_in_context_accuracies', 'predicted_in_context_correct_candidate_accuracies', 'correct_within_candidates']

if args.reward_binning_analysis_only and exp_typ == 'held_out':
    # Doing per reward averaging only for held-out experiments.
    complete_query_stone_state_per_reward_binned_accuracy_all_seeds = {'-3': [], '-1': [], '1': [], '3': []}
    within_support_query_stone_state_per_reward_binned_accuracy_all_seeds = {'-3': [], '-1': [], '1': [], '3': []}
    within_support_within_half_query_stone_state_per_reward_binned_accuracy_all_seeds = {'-3': [], '-1': [], '1': [], '3': []}

    for metric in metrics:
        for seed in seed_results.keys():
            if metric == 'complete_query_stone_state_per_reward_binned_accuracy':
                for reward_bin in complete_query_stone_state_per_reward_binned_accuracy_all_seeds.keys():
                    complete_query_stone_state_per_reward_binned_accuracy_all_seeds[reward_bin].append(
                        seed_results[seed][metric][reward_bin]
                    )
            elif metric == 'within_support_query_stone_state_per_reward_binned_accuracy':
                for reward_bin in within_support_query_stone_state_per_reward_binned_accuracy_all_seeds.keys():
                    within_support_query_stone_state_per_reward_binned_accuracy_all_seeds[reward_bin].append(
                        seed_results[seed][metric][reward_bin]
                    )
            elif metric == 'within_support_within_half_query_stone_state_per_reward_binned_accuracy':
                for reward_bin in within_support_within_half_query_stone_state_per_reward_binned_accuracy_all_seeds.keys():
                    within_support_within_half_query_stone_state_per_reward_binned_accuracy_all_seeds[reward_bin].append(
                        seed_results[seed][metric][reward_bin]
                    )

    # STEP 1: Find the maximum epoch length across all seeds and reward bins
    max_epoch_length = 0
    for reward_bin in complete_query_stone_state_per_reward_binned_accuracy_all_seeds.keys():
        for seed_values in complete_query_stone_state_per_reward_binned_accuracy_all_seeds[reward_bin]:
            if len(seed_values) > max_epoch_length:
                max_epoch_length = len(seed_values)
        for seed_values in within_support_query_stone_state_per_reward_binned_accuracy_all_seeds[reward_bin]:
            if len(seed_values) > max_epoch_length:
                max_epoch_length = len(seed_values)
        for seed_values in within_support_within_half_query_stone_state_per_reward_binned_accuracy_all_seeds[reward_bin]:
            if len(seed_values) > max_epoch_length:
                max_epoch_length = len(seed_values)

    print(f"Maximum epoch length across all seeds and reward bins: {max_epoch_length}")
    
    # STEP 2: Pad all sequences to the maximum length
    for reward_bin in complete_query_stone_state_per_reward_binned_accuracy_all_seeds.keys():
        # Pad complete_query_stone_state_per_reward_binned_accuracy
        padded_seed_values = []
        for values in complete_query_stone_state_per_reward_binned_accuracy_all_seeds[reward_bin]:
            if len(values) < max_epoch_length:
                padded_values = list(values)
                for epoch_idx in range(len(values), max_epoch_length):
                    # Get values from all seeds that have this epoch
                    available_values = [seed_vals[epoch_idx] for seed_vals in complete_query_stone_state_per_reward_binned_accuracy_all_seeds[reward_bin] if len(seed_vals) > epoch_idx]
                    if available_values:
                        imputed_value = np.mean(available_values)
                    else:
                        imputed_value = values[-1]
                    padded_values.append(imputed_value)
                padded_seed_values.append(padded_values)
            else:
                padded_seed_values.append(values)
        complete_query_stone_state_per_reward_binned_accuracy_all_seeds[reward_bin] = padded_seed_values

        # Pad within_support_query_stone_state_per_reward_binned_accuracy
        padded_seed_values = []
        for values in within_support_query_stone_state_per_reward_binned_accuracy_all_seeds[reward_bin]:
            if len(values) < max_epoch_length:
                padded_values = list(values)
                for epoch_idx in range(len(values), max_epoch_length):
                    available_values = [seed_vals[epoch_idx] for seed_vals in within_support_query_stone_state_per_reward_binned_accuracy_all_seeds[reward_bin] if len(seed_vals) > epoch_idx]
                    if available_values:
                        imputed_value = np.mean(available_values)
                    else:
                        imputed_value = values[-1]
                    padded_values.append(imputed_value)
                padded_seed_values.append(padded_values)
            else:
                padded_seed_values.append(values)
        within_support_query_stone_state_per_reward_binned_accuracy_all_seeds[reward_bin] = padded_seed_values

        # Pad within_support_within_half_query_stone_state_per_reward_binned_accuracy
        padded_seed_values = []
        for values in within_support_within_half_query_stone_state_per_reward_binned_accuracy_all_seeds[reward_bin]:
            if len(values) < max_epoch_length:
                padded_values = list(values)
                for epoch_idx in range(len(values), max_epoch_length):
                    available_values = [seed_vals[epoch_idx] for seed_vals in within_support_within_half_query_stone_state_per_reward_binned_accuracy_all_seeds[reward_bin] if len(seed_vals) > epoch_idx]
                    if available_values:
                        imputed_value = np.mean(available_values)
                    else:
                        imputed_value = values[-1]
                    padded_values.append(imputed_value)
                padded_seed_values.append(padded_values)
            else:
                padded_seed_values.append(values)
        within_support_within_half_query_stone_state_per_reward_binned_accuracy_all_seeds[reward_bin] = padded_seed_values

    # STEP 3: NOW compute averages and std errors from the PADDED data
    averaged_results = {}
    std_errors = {}
    for reward_bin in complete_query_stone_state_per_reward_binned_accuracy_all_seeds.keys():
        all_seed_values = np.array(complete_query_stone_state_per_reward_binned_accuracy_all_seeds[reward_bin])
        averaged_results[f'complete_query_stone_state_per_reward_binned_accuracy_{reward_bin}'] = np.mean(all_seed_values, axis=0)
        std_errors[f'complete_query_stone_state_per_reward_binned_accuracy_{reward_bin}'] = np.std(all_seed_values, axis=0) / np.sqrt(len(all_seed_values))
        
        all_seed_values = np.array(within_support_query_stone_state_per_reward_binned_accuracy_all_seeds[reward_bin])
        averaged_results[f'within_support_query_stone_state_per_reward_binned_accuracy_{reward_bin}'] = np.mean(all_seed_values, axis=0)
        std_errors[f'within_support_query_stone_state_per_reward_binned_accuracy_{reward_bin}'] = np.std(all_seed_values, axis=0) / np.sqrt(len(all_seed_values))
        
        all_seed_values = np.array(within_support_within_half_query_stone_state_per_reward_binned_accuracy_all_seeds[reward_bin])
        averaged_results[f'within_support_within_half_query_stone_state_per_reward_binned_accuracy_{reward_bin}'] = np.mean(all_seed_values, axis=0)
        std_errors[f'within_support_within_half_query_stone_state_per_reward_binned_accuracy_{reward_bin}'] = np.std(all_seed_values, axis=0) / np.sqrt(len(all_seed_values))
        
        print(f"Averaged {metric} over seeds for reward bin {reward_bin} (after padding)")




    # Now, we will plot the within_support_query_stone_state_per_reward_binned_accuracy, and within_support_within_half_query_stone_state_per_reward_binned_accuracy only.
    # For the within_support_query_stone_state_per_reward_binned_accuracy, there will be 4 lines (one for each reward bin) and the style will be solid.
    # For the within_support_within_half_query_stone_state_per_reward_binned_accuracy, there will be 4 lines (one for each reward bin) and the style will be dashed.
    # Make sure the colors for the solid and dashed lines for the same reward bin are the same.
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(1, 1, 1)
    ax.grid(True, alpha=0.3)
    epochs = range(len(averaged_results['within_support_query_stone_state_per_reward_binned_accuracy_-3']))
    reward_bin_colors = {'-3': 'tab:olive', '-1': 'tab:cyan', '1': 'tab:pink', '3': 'tab:brown'}

    reward_bin_mapping = {
        '-3': '-3',
        '-1': '-1',
        '1': '1',
        '3': '+15',
    }

    for reward_bin in ['-3', '-1', '1', '3']:
        # Plot within_support_query_stone_state_per_reward_binned_accuracy
        mean_values = averaged_results[f'within_support_query_stone_state_per_reward_binned_accuracy_{reward_bin}']
        std_error_values = std_errors[f'within_support_query_stone_state_per_reward_binned_accuracy_{reward_bin}']
        ax.plot(epochs, mean_values, label=f'Query with reward feature {reward_bin_mapping[reward_bin]}', color=reward_bin_colors[reward_bin], linestyle='dashed')
        ax.fill_between(epochs, mean_values - std_error_values, mean_values + std_error_values, color=reward_bin_colors[reward_bin], alpha=0.2)

        # # Plot within_support_within_half_query_stone_state_per_reward_binned_accuracy
        # mean_values = averaged_results[f'within_support_within_half_query_stone_state_per_reward_binned_accuracy_{reward_bin}']
        # std_error_values = std_errors[f'within_support_within_half_query_stone_state_per_reward_binned_accuracy_{reward_bin}']
        # ax.plot(epochs, mean_values, label=f'Query with reward feature = {reward_bin_mapping[reward_bin]}', color=reward_bin_colors[reward_bin], linestyle='solid')
        # ax.fill_between(epochs, mean_values - std_error_values, mean_values + std_error_values, color=reward_bin_colors[reward_bin], alpha=0.2)

    ax.set_xlabel('Epochs', fontsize=26)
    ax.set_ylabel('Accuracy', fontsize=26)
    
    ax.legend(fontsize=18, loc='lower right', ncol=1)

    # Set xticks size and yticks size
    ax.tick_params(axis='x', labelsize=24)
    ax.tick_params(axis='y', labelsize=24)
    plt.ylim(0, 1.0)
    
    # plt.savefig(f'reward_binned_accuracy_analysis_in_support_gating_hop_{hop}_{exp_typ}.png')
    # plt.savefig(f'reward_binned_accuracy_analysis_in_support_gating_hop_{hop}_{exp_typ}.pdf', bbox_inches='tight')


    # plt.savefig(f'reward_binned_accuracy_exact_match_within_correct_half_hop_{hop}_{exp_typ}.png')
    # plt.savefig(f'reward_binned_accuracy_exact_match_within_correct_half_hop_{hop}_{exp_typ}.pdf', bbox_inches='tight')

    plt.savefig(f'reward_binned_accuracy_analysis_hop_{hop}_{exp_typ}_within_support_denom.png')
    plt.savefig(f'reward_binned_accuracy_analysis_hop_{hop}_{exp_typ}_within_support_denom.pdf', bbox_inches='tight')

    exit(0)


    # -----------------------------------------------------------------------------------------------



# If not doing reward binning analysis only, do the normal averaging.
for metric in metrics:
    all_seed_values = [seed_results[seed][metric] for seed in seed_results.keys()]

    # Find the maximum length across all seeds
    max_length = max(len(values) for values in all_seed_values)
    # import pdb; pdb.set_trace()
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

    individual_seed_results[metric] = all_seed_values

    std_errors[metric] = np.std(all_seed_values, axis=0) / np.sqrt(len(all_seed_values))
    print(f"\nAveraged {metric} over seeds:")

# Plot the averaged results with error bars using fill_between
epochs = range(len(averaged_results['predicted_in_context_accuracies']))



if exp_typ == 'decomposition':
    # Only print the predicted_in_context_accuracies and the predicted_correct_within_context.
    metrics = [
        ('predicted_in_context_accuracies', 'P(A)'),
        ('predicted_in_context_correct_half_accuracies', 'P(B | A)'),
        ('predicted_in_context_correct_half_exact_accuracies', 'P(C | A ∩ B)'),
        # ('predicted_in_adjacent_and_correct_half_accuracies', 'P(EN | A)'),
        # ('predicted_correct_half_within_adjacent_and_correct_half_accuracies', 'P(NR | EN)'), 
        # ('predicted_correct_within_context', 'Exact Accuracy (1 out of 8)'),
        # ('predicted_exact_out_of_all_108', 'P(C) = P(A) . P(B | A) . P(C | A ∩ B) (1 out of 108)'),
    ]
elif exp_typ == 'composition':
    metrics = [
        ('predicted_in_context_accuracies', 'P(A)'),
        ('predicted_in_context_correct_candidate_accuracies', 'P(B | A)'),
        ('correct_within_candidates', 'P(C | A ∩ B)'),
    ]
elif exp_typ == 'held_out':
    metrics = [
        ('predicted_in_context_accuracies', 'P(A) (8 out of 108)'),
        ('predicted_in_context_correct_half_accuracies', 'P(B | A) (4 out of 8)'),
        ('predicted_in_context_other_half_accuracies', '1 - P(B|A) (4 out of 8)'),
        ('predicted_in_context_correct_half_exact_accuracies', 'P(C|A ∩ B) (1 out of 4)'),
    ]

linestyles = {'predicted_in_context_accuracies': 'solid', 'predicted_correct_within_context': 'solid'}
exact_match_cycle_colors = ['tab:blue', 'tab:green', 'tab:gray', 'tab:red']
exact_match_out_of_108_color = exact_match_cycle_colors[hop - 2]


colors = {
    2: {'predicted_in_context_accuracies': 'orange', 'predicted_in_context_correct_half_accuracies': 'tab:purple', 'predicted_in_context_correct_half_exact_accuracies': 'tab:blue', 'predicted_exact_out_of_all_108': exact_match_out_of_108_color, 'predicted_correct_half_within_adjacent_and_correct_half_accuracies': 'tab:pink', 'predicted_in_adjacent_and_correct_half_accuracies': 'tab:cyan'},
    3: {'predicted_in_context_accuracies': 'orange', 'predicted_in_context_correct_half_accuracies': 'tab:purple', 'predicted_in_context_correct_half_exact_accuracies': 'tab:green', 'predicted_exact_out_of_all_108': exact_match_out_of_108_color, 'predicted_correct_half_within_adjacent_and_correct_half_accuracies': 'tab:pink', 'predicted_in_adjacent_and_correct_half_accuracies': 'tab:cyan'},
    4: {'predicted_in_context_accuracies': 'orange', 'predicted_in_context_correct_half_accuracies': 'tab:purple', 'predicted_in_context_correct_half_exact_accuracies': 'tab:gray', 'predicted_exact_out_of_all_108': exact_match_out_of_108_color, 'predicted_correct_half_within_adjacent_and_correct_half_accuracies': 'tab:pink', 'predicted_in_adjacent_and_correct_half_accuracies': 'tab:cyan'},
    5: {'predicted_in_context_accuracies': 'orange', 'predicted_in_context_correct_half_accuracies': 'tab:purple', 'predicted_in_context_correct_half_exact_accuracies': 'tab:red', 'predicted_exact_out_of_all_108': exact_match_out_of_108_color, 'predicted_correct_half_within_adjacent_and_correct_half_accuracies': 'tab:pink', 'predicted_in_adjacent_and_correct_half_accuracies': 'tab:cyan'},
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
    'predicted_in_context_other_half_accuracies': 'tab:red',
    'predicted_in_context_correct_half_exact_accuracies': 'tab:green',
    # 'predicted_correct_within_context': 'tab:red',
}
if args.plot_individual_seeds:
    # Plot the metrics for each seed in a separate panel using the individual_seed_results
    fig = plt.figure(figsize=(30, 10))
    gs = fig.add_gridspec(1, len(seed_results), hspace=0.4)
    for i, seed in enumerate(seed_results.keys()):
        ax = fig.add_subplot(gs[0, i])
        epochs = range(len(individual_seed_results['predicted_in_context_accuracies'][i]))
        for metric, label in metrics:
            mean = individual_seed_results[metric][i]
            # Do not plot the exact out of 108 for individual seeds
            if metric == 'predicted_exact_out_of_all_108':
                continue
            ax.plot(epochs, mean, label=label, linewidth=2, linestyle=linestyles.get(metric, 'solid'), color=colors.get(hop, {}).get(metric, 'gold'))
        ax.set_title(f'Seed {seed}', fontsize=16)
        ax.set_xlabel('Epoch', fontsize=14)
        ax.set_ylabel('Accuracy', fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
    # plt.suptitle(f'Phasic learning of latent structure learning ({hop}-hop) - Individual Seeds', fontsize=20, y=1.02)
    plt.tight_layout()
    if args.normalized_reward:
        plt.savefig(f'{exp_typ}_{hop}_staged_learning_of_individual_seeds_normalized_reward.png')
        plt.savefig(f'{exp_typ}_{hop}_staged_learning_of_individual_seeds_normalized_reward.pdf', bbox_inches='tight')
    else:
        plt.savefig(f'{exp_typ}_{hop}_staged_learning_of_individual_seeds.png')
        plt.savefig(f'{exp_typ}_{hop}_staged_learning_of_individual_seeds.pdf', bbox_inches='tight')

    # Also plot the final exact 1 out of 108 accuracy across seeds
    # Now for each of the seeds, plot the exact out of 108 accuracy in the same plot.
    fig = plt.figure(figsize=(8, 6))
    for i, seed in enumerate(seed_results.keys()):
        mean = individual_seed_results['predicted_exact_out_of_all_108'][i]
        plt.plot(epochs, mean, label=f'Seed {seed}', linewidth=2)
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Exact Accuracy (1 out of 108)', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.title(f'Exact accuracy (1 out of 108) across seeds ({hop}-hop)', fontsize=16)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1)
    plt.tight_layout()
    if args.normalized_reward:
        plt.savefig(f'{exp_typ}_{hop}_exact_accuracy_across_seeds_normalized_reward.png')
        plt.savefig(f'{exp_typ}_{hop}_exact_accuracy_across_seeds_normalized_reward.pdf', bbox_inches='tight')
    else:
        plt.savefig(f'{exp_typ}_{hop}_exact_accuracy_across_seeds.png')
        plt.savefig(f'{exp_typ}_{hop}_exact_accuracy_across_seeds.pdf', bbox_inches='tight')

    # Exit after plotting individual seeds
    exit()

fig = plt.figure(figsize=(12, 8))
for metric, label in metrics:
    mean = averaged_results[metric]
    sem = std_errors[metric]
    # colors = held_out_colors if exp_typ == 'held_out' else colors
    if exp_typ == 'composition':
        colors = colors_composition
    elif exp_typ == 'held_out':
        colors = held_out_colors
    else:
        colors = colors
    if exp_typ == 'held_out':
        plt.plot(epochs, mean, label=label, linewidth=2, linestyle=linestyles.get(metric, 'solid'), color=colors.get(metric, 'gold'))
        plt.fill_between(epochs, mean - sem, mean + sem, alpha=0.2, color=colors.get(metric, 'gold'))
    else:
        # if metric == 'predicted_in_context_correct_half_exact_accuracies':
        #     # For P(C | A ∩ B), use dashed line
        #     plt.plot(epochs, mean, label=label, linewidth=2, linestyle='dashed', color=colors.get(hop, {}).get(metric, 'gold'))
        #     plt.fill_between(epochs, mean - sem, mean + sem, alpha=0.2, color=colors.get(hop, {}).get(metric, 'gold'))
        # else:
        plt.plot(epochs, mean, label=label, linewidth=2, color=colors.get(hop, {}).get(metric, 'gold'), linestyle=args.custom_linestyle if args.custom_linestyle else linestyles.get(metric, 'solid'))
        plt.fill_between(epochs, mean - sem, mean + sem, alpha=0.2, color=colors.get(hop, {}).get(metric, 'gold'))
    # Add text annotations at specific epochs
    if args.annotated_epochs:
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
plt.legend(fontsize=18, loc='center right', ncol=1, frameon=True)
plt.grid(True, alpha=0.3)
plt.ylim(0, 1)
# plt.tight_layout()
custom_output_file = args.custom_output_file
if custom_output_file is not None:
    plt.savefig(f'{custom_output_file}.png')
    plt.savefig(f'{custom_output_file}.pdf', bbox_inches='tight')
else:
    if args.get_output_file_from_input_path:
        file_paths = composition_file_paths if exp_typ == 'composition' else decomposition_file_paths if exp_typ == 'decomposition' else None
        assert len(file_paths[hop]) == 1, "Currently only supports single file path to get output file name."
        input_path = file_paths[hop][0]
        # Extract only the 'complete_graph/scheduler_cosine/wd_0.1_lr_0.0001/eta_min_8.5e-05' and the seed value from the input path. Use regex.
        match = re.search(r'complete_graph/(.+?)/seed_(\d+)', input_path)
        if match:
            # Make sure to replace '/' with '_' in the extracted part.
            extracted_part = match.group(1).replace('/', '_')
            seed_value = match.group(2)
            output_file_name = f"{exp_typ}_{hop}hop_{extracted_part}_seed_{seed_value}_phasic_learning_of_latent_structure.png"
            plt.savefig(output_file_name)
            plt.savefig(output_file_name.replace('.png', '.pdf'), bbox_inches='tight')
    else:
        plt.savefig(f'Jan_2_{exp_typ}_{hop}_phasic_learning_of_latent_structure.png')
        plt.savefig(f'Jan_2_{exp_typ}_{hop}_phasic_learning_of_latent_structure.pdf', bbox_inches='tight')