import json
import numpy as np
from collections import defaultdict

def parse_stone_state(stone_state_str):
    """Extract stone state string from query format."""
    # Format: "{color: X, size: Y, roundness: Z, reward: W} POTION -> {expected output}"
    if '->' in stone_state_str:
        parts = stone_state_str.split('->')
        if len(parts) == 2:
            return parts[1].strip()
    return stone_state_str

def extract_support_stones(prompt_text):
    """Extract all stone states (S1-S8) from the stone ID mapping in prompt."""
    support_stones = set()
    
    # Find the mapping section between "Stone ID Mapping:" and "Analyze the patterns"
    if "Stone ID Mapping:" in prompt_text:
        mapping_section = prompt_text.split("Stone ID Mapping:")[1].split("Analyze the patterns")[0]
        
        # Extract stone IDs (S1, S2, etc.)
        import re
        stone_ids = re.findall(r'(S\d+):', mapping_section)
        support_stones.update(stone_ids)
    
    return support_stones

def calculate_in_support_metrics(results_file):
    """
    Calculate in-support accuracy: whether predicted stone ID is in the support set.
    
    Args:
        results_file: Path to JSON file with LLM evaluation results
    
    Returns:
        dict with in-support accuracy and detailed breakdown
    """
    # Load results
    with open(results_file, 'r') as f:
        data = json.load(f)
    
    detailed_results = data['detailed_results']
    
    # Metrics
    total_predictions = 0
    in_support_count = 0
    correct_predictions = 0
    
    # Group by episode for detailed analysis
    episode_metrics = defaultdict(lambda: {
        'total': 0, 
        'in_support': 0, 
        'correct': 0
    })
    
    for result in detailed_results:
        episode_id = result['episode_id']
        predicted_stone_id = result['predicted_output']  # e.g., "S3"
        expected_stone_id = result['expected_stone_id']  # e.g., "S5"
        is_correct = result['correct']  # stone_id_correct
        
        # Extract support stones from prompt (stored in result if available)
        # For this analysis, we know support has 8 stones (S1-S8)
        # These are defined in the prompt's "Stone ID Mapping" section
        support_stones = {f"S{i}" for i in range(1, 9)}  # S1-S8
        
        total_predictions += 1
        episode_metrics[episode_id]['total'] += 1
        
        # Check if prediction is in support set
        if predicted_stone_id in support_stones:
            in_support_count += 1
            episode_metrics[episode_id]['in_support'] += 1
        
        # Check if correct
        if is_correct:
            correct_predictions += 1
            episode_metrics[episode_id]['correct'] += 1
    
    # Calculate overall metrics
    in_support_accuracy = in_support_count / total_predictions if total_predictions > 0 else 0
    exact_accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
    
    # Conditional: correct given in-support
    correct_given_in_support = (correct_predictions / in_support_count 
                                if in_support_count > 0 else 0)
    
    results = {
        'total_predictions': total_predictions,
        'in_support_count': in_support_count,
        'in_support_accuracy': in_support_accuracy,
        'exact_accuracy': exact_accuracy,
        'correct_given_in_support': correct_given_in_support,
        'episode_metrics': dict(episode_metrics)
    }
    
    return results

# Usage
results_file = '/home/rsaha/projects/def-afyshe-ab/rsaha/dm_alchemy/prompting_results/meta-llama_Llama-3.1-8B-Instruct_complete_graph_generated_data_enhanced_qnodes_in_snodesdecompositional_chemistry_samples_167424_80_unique_stones_train_shop_2_qhop_1_combined_20250930_003212_results.json'
metrics = calculate_in_support_metrics(results_file)

print(f"Total predictions: {metrics['total_predictions']}")
print(f"In-support accuracy (8/108): {metrics['in_support_accuracy']:.3f}")
print(f"Exact accuracy (1/8): {metrics['exact_accuracy']:.3f}")
print(f"P(correct | in-support): {metrics['correct_given_in_support']:.3f}")
print(f"\nNumber of episodes: {len(metrics['episode_metrics'])}")