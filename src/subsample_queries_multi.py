#!/usr/bin/env python3
"""
Script to subsample queries from a source JSON file to match the distribution 
of a reference JSON file, maintaining balance across input stone states.

This version handles large multi-episode datasets from the dm_alchemy project.
Works by analyzing the average queries per stone state per episode in the reference file,
then applying that distribution to each episode in the source file.

Usage:
    python src/subsample_queries_multi.py --reference ref.json --source src.json --output out.json
"""

import json
import argparse
from collections import defaultdict
from typing import Dict, List, Set, Tuple
import random
import os


def extract_input_stone_state(example_str: str) -> str:
    """Extract the input stone state from an example string."""
    first_brace_end = example_str.find("}")
    if first_brace_end == -1:
        raise ValueError(f"Malformed example string: {example_str}")
    return example_str[:first_brace_end + 1]


def analyze_reference_distribution(reference_file: str) -> Dict[str, float]:
    """
    Analyze the reference file to determine the average target distribution of input stone states
    per episode.
    
    Args:
        reference_file: Path to the reference JSON file
    
    Returns:
        Dict mapping input stone state -> average queries per episode
    """
    print(f"Analyzing reference distribution from {reference_file}...")
    
    with open(reference_file, 'r') as f:
        reference_data = json.load(f)
    
    # Count queries per state across all episodes that have queries
    stone_state_counts = defaultdict(int)
    episodes_with_state = defaultdict(set)
    episodes_with_queries = 0
    
    for episode_id, episode_content in reference_data["episodes"].items():
        if not episode_content or not episode_content.get("query", []):
            continue
            
        episodes_with_queries += 1
        queries = episode_content["query"]
        episode_state_counts = defaultdict(int)
        
        # Count states in this episode
        for query in queries:
            input_state = extract_input_stone_state(query)
            episode_state_counts[input_state] += 1
            episodes_with_state[input_state].add(episode_id)
        
        # Add to global counts
        for state, count in episode_state_counts.items():
            stone_state_counts[state] += count
    
    # Calculate average queries per state per episode
    avg_queries_per_state = {}
    for state, total_count in stone_state_counts.items():
        num_episodes_with_state = len(episodes_with_state[state])
        avg_queries_per_state[state] = total_count / num_episodes_with_state
    
    print(f"Reference analysis complete:")
    print(f"  Episodes with queries: {episodes_with_queries}")
    print(f"  Unique input stone states: {len(stone_state_counts)}")
    print(f"  Total queries across all episodes: {sum(stone_state_counts.values())}")
    
    return avg_queries_per_state


def subsample_queries_for_episode(source_queries: List[str], target_distribution: Dict[str, float]) -> List[str]:
    """
    Subsample queries from a single episode to match the target distribution.
    
    Args:
        source_queries: List of query strings from the source episode
        target_distribution: Target average count for each input stone state
    
    Returns:
        List of subsampled query strings
    """
    # Group source queries by input stone state
    queries_by_state = defaultdict(list)
    for query in source_queries:
        input_state = extract_input_stone_state(query)
        queries_by_state[input_state].append(query)
    
    # Only subsample for states that exist in both target and source
    subsampled_queries = []
    
    for state, target_count_float in target_distribution.items():
        available_queries = queries_by_state.get(state, [])
        
        if not available_queries:
            continue  # Skip states not present in this episode
        
        # Convert float target to int (round to nearest)
        target_count = round(target_count_float) # Essentially this is for the reference file which is the 2-hop composition.
        
        if len(available_queries) <= target_count:
            # Take all available queries if we don't have more than needed
            selected_queries = available_queries
        else:
            # Randomly sample the required number of queries
            # selected_queries = random.sample(available_queries, target_count)
            # Select the first 'target_count' queries to maintain some order
            selected_queries = available_queries[:target_count]
        
        subsampled_queries.extend(selected_queries)
    
    return subsampled_queries


def main():
    parser = argparse.ArgumentParser(
        description="Subsample queries to match reference distribution across input stone states"
    )
    parser.add_argument("--reference_file", "-r", type=str, required=False,
                        default='src/data/complete_graph_generated_data_enhanced_qnodes_in_snodes/compositional_chemistry_samples_167424_80_unique_stones_train_shop_1_qhop_2_.json',
                        help="Path to reference JSON file")
    parser.add_argument("--source_file", "-s", type=str, required=False,
                        default='src/data/complete_graph_generated_data_enhanced_qnodes_in_snodes/compositional_chemistry_samples_167424_80_unique_stones_train_shop_1_qhop_5_.json',
                        help="Path to source JSON file to subsample from")
    parser.add_argument("--output_dir", "-o", type=str, required=False,
                        default='src/data/subsampled_balanced_complete_graph_generated_data_enhanced_qnodes_in_snodes',
                        help="Path to output the subsampled JSON file")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducible subsampling")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Print detailed analysis")
    parser.add_argument("--sample-episodes", type=int, default=None,
                        help="Only process first N episodes (for testing)")
    parser.add_argument("--skip-empty", action="store_true", default=False,
                        help="Skip episodes with no queries")
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    random.seed(args.seed)
    
    print(f"=== Multi-Episode Query Subsampling ===")
    print(f"Reference: {args.reference_file}")
    print(f"Source: {args.source_file}")
    print(f"Output: {args.output_dir}")
    print(f"Random seed: {args.seed}")
    
    # Analyze reference distribution
    for seed in [0, 1, 2, 3, 4]:
        args.reference = args.reference_file.replace('.json', f'seed_{seed}.json')
        args.source = args.source_file.replace('.json', f'seed_{seed}.json')
        print(f"\nProcessing seed {seed}...")
        target_distribution = analyze_reference_distribution(args.reference)
        
        if args.verbose:
            print(f"\nTarget distribution (avg queries per episode per state):")
            for state, count in sorted(target_distribution.items()):
                print(f"  {state}: {count:.1f}")
        
        # Load source file
        print(f"\nLoading source file: {args.source}")
        with open(args.source, 'r') as f:
            source_data = json.load(f)
        
        # Create output data structure
        output_data = {
            "metadata": source_data.get("metadata", {}),
            "episodes": {}
        }
        
        # Add subsampling info to metadata
        if "subsampling_info" not in output_data["metadata"]:
            output_data["metadata"]["subsampling_info"] = {}
        
        output_data["metadata"]["subsampling_info"].update({
            "reference_file": os.path.basename(args.reference),
            "source_file": os.path.basename(args.source),
            "random_seed": args.seed,
            "target_distribution": {k: round(v, 2) for k, v in target_distribution.items()}
        })
        
        # Assert that the number of episodes in source matches reference
        print(f"Source episodes: {len(source_data['episodes'])}")
        print(f"Reference episodes: {len(json.load(open(args.reference))['episodes'])}")
        if len(source_data['episodes']) != len(json.load(open(args.reference))['episodes']):
            print(f"⚠️ WARNING: Number of episodes in source and reference do not match!")
        
        # Process episodes
        total_source_queries = 0
        total_subsampled_queries = 0
        episodes_processed = 0
        episodes_with_queries = 0
        episodes_skipped_empty = 0
        
        source_episodes = list(source_data["episodes"].items())
        if args.sample_episodes:
            source_episodes = source_episodes[:args.sample_episodes]
            print(f"Processing first {args.sample_episodes} episodes only")
        
        print(f"\nProcessing {len(source_episodes)} episodes...")
        
        for episode_id, source_episode in source_episodes:
            episodes_processed += 1
            
            if not source_episode:  # Empty episode
                print(f"Episode {episode_id} is empty.")
                if not args.skip_empty:
                    output_data["episodes"][episode_id] = {}
                else:
                    episodes_skipped_empty += 1
                continue
            
            source_queries = source_episode.get("query", [])
            if not source_queries:  # Episode with no queries
                print(f"Episode {episode_id} has no queries.")
                if not args.skip_empty:
                    output_data["episodes"][episode_id] = source_episode
                else:
                    episodes_skipped_empty += 1
                continue
            
            episodes_with_queries += 1
            total_source_queries += len(source_queries)
            
            # Subsample queries for this episode
            subsampled_queries = subsample_queries_for_episode(source_queries, target_distribution)
            total_subsampled_queries += len(subsampled_queries)
            
            # Create output episode (preserve all other fields)
            output_episode = source_episode.copy()
            output_episode["query"] = subsampled_queries
            output_data["episodes"][episode_id] = output_episode
            
            # Progress reporting
            if episodes_processed % 1000 == 0:
                print(f"  Processed {episodes_processed:,} episodes... "
                    f"({episodes_with_queries:,} with queries)")
            
            if args.verbose and episodes_with_queries <= 3:  # Show details for first few episodes
                print(f"\nEpisode {episode_id}:")
                print(f"  Source queries: {len(source_queries)}")
                print(f"  Subsampled queries: {len(subsampled_queries)}")
                # Show distribution for this episode
                episode_dist = defaultdict(int)
                for query in subsampled_queries:
                    state = extract_input_stone_state(query)
                    episode_dist[state] += 1
                for state, count in sorted(episode_dist.items()):
                    target_avg = target_distribution.get(state, 0)
                    print(f"    {state}: {count} (target: {target_avg:.1f})")
        
        # Create output directory if needed
        
        args.output = os.path.join(args.output_dir, os.path.basename(args.source))
        
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        
        # Save output
        print(f"\nSaving subsampled data to: {args.output}")
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        # Final statistics
        reduction_ratio = total_subsampled_queries / max(total_source_queries, 1)
        
        print(f"\n=== Subsampling Results ===")
        print(f"Episodes processed: {episodes_processed:,}")
        print(f"Episodes with queries: {episodes_with_queries:,}")
        if episodes_skipped_empty > 0:
            print(f"Episodes skipped (empty): {episodes_skipped_empty:,}")
        print(f"Episodes in output: {len(output_data['episodes']):,}")
        print(f"Total source queries: {total_source_queries:,}")
        print(f"Total subsampled queries: {total_subsampled_queries:,}")
        print(f"Reduction ratio: {reduction_ratio:.3f}")
        
        if episodes_with_queries > 0:
            print(f"Avg queries per episode (source): {total_source_queries / episodes_with_queries:.1f}")
            print(f"Avg queries per episode (subsampled): {total_subsampled_queries / episodes_with_queries:.1f}")
        
        print(f"\n✅ Subsampling completed successfully!")
        print(f"Output file: {args.output}")


if __name__ == "__main__":
    main()
