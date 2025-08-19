#!/usr/bin/env python3
"""
Test script to verify equivalence between alchemy_datagen.py and enhanced_chemistry_generator.py
for the same chemistry configurations.

This script tests:
1. Whether TempAlchemyEnv.apply_potion() generates the same transitions as AlchemyEnv.apply_potion()
2. Whether structural parts from graph analysis match actual transition behavior
3. Whether perceived stone/potion representations are consistent across both approaches
"""

import sys
import os
import numpy as np
from itertools import product
from typing import List, Tuple, Dict, Any
import pickle
import hashlib
from tqdm import tqdm
import re
# Add the project paths
sys.path.append('/home/rsaha/projects/dm_alchemy')
sys.path.append('/home/rsaha/projects/dm_alchemy/explicit_implicit_icl')
sys.path.append('/home/rsaha/projects/dm_alchemy/src/data')

# Import from alchemy_datagen.py
from explicit_implicit_icl.alchemy_datagen import AlchemyEnv, AlchemyFactory

# Import from enhanced_chemistry_generator.py
from src.data.enhanced_chemistry_generator import (
    TempAlchemyEnv, 
    generate_transitions_for_chemistry,
    generate_canonical_data_with_transitions
)

# Import dm_alchemy types
from dm_alchemy.types import utils as type_utils
from dm_alchemy.types.stones_and_potions import possible_perceived_potions


def create_test_chemistry(factory: AlchemyFactory, chemistry_id: int):
    """Create a test chemistry using AlchemyFactory."""
    return factory[chemistry_id]


def compare_apply_potion_functions(alchemy_env: AlchemyEnv, temp_env: TempAlchemyEnv) -> Dict[str, Any]:
    """
    Compare apply_potion results between AlchemyEnv and TempAlchemyEnv.
    
    Returns:
        Dictionary with comparison results and statistics.
    """
    print("  Comparing apply_potion() functions...")
    
    potions = alchemy_env.possible_perceived_potions()
    stones = alchemy_env.possible_perceived_stones()
    
    results = {
        "total_tests": 0,
        "matching_results": 0,
        "mismatched_results": 0,
        "alchemy_valid_transitions": 0,
        "temp_valid_transitions": 0,
        "mismatches": []
    }
    
    for p, s in product(potions, stones):
        results["total_tests"] += 1
        
        # Test alchemy_datagen.py version
        alchemy_result = alchemy_env.apply_potion(s, p)
        
        # Test enhanced_chemistry_generator.py version
        temp_result = temp_env.apply_potion(s, p)
        
        # Handle different null representations
        # alchemy_datagen returns original stone when no transition
        # enhanced_chemistry_generator returns None
        alchemy_valid = not np.array_equal(alchemy_result.perceived_coords, s.perceived_coords) or alchemy_result.reward != s.reward
        temp_valid = temp_result is not None
        
        if alchemy_valid:
            results["alchemy_valid_transitions"] += 1
        if temp_valid:
            results["temp_valid_transitions"] += 1
        
        # Compare results
        if alchemy_valid and temp_valid:
            # Both have valid transitions - compare results
            if (np.array_equal(alchemy_result.perceived_coords, temp_result.perceived_coords) and 
                alchemy_result.reward == temp_result.reward):
                results["matching_results"] += 1
            else:
                results["mismatched_results"] += 1
                results["mismatches"].append({
                    "input_stone": tuple(s.perceived_coords),
                    "input_stone_reward": s.reward,
                    "input_potion": p.index(),
                    "alchemy_result": (tuple(alchemy_result.perceived_coords), alchemy_result.reward),
                    "temp_result": (tuple(temp_result.perceived_coords), temp_result.reward)
                })
        elif not alchemy_valid and not temp_valid:
            # Both have no valid transition
            results["matching_results"] += 1
        else:
            # One has valid transition, other doesn't
            results["mismatched_results"] += 1
            results["mismatches"].append({
                "input_stone": tuple(s.perceived_coords),
                "input_stone_reward": s.reward,
                "input_potion": p.index(),
                "alchemy_result": (tuple(alchemy_result.perceived_coords), alchemy_result.reward) if alchemy_valid else "no_transition",
                "temp_result": (tuple(temp_result.perceived_coords), temp_result.reward) if temp_valid else "no_transition"
            })
    
    return results


def compare_transition_matrix_vs_direct_chemistry(alchemy_env: AlchemyEnv, chemistry) -> Dict[str, Any]:
    """
    Compare structural representations derived from:
    1. Transition matrix approach (like alchemy_datagen.py)
    2. Direct chemistry graph analysis (like enhanced_chemistry_generator.py)
    
    This is the real test of equivalence between the two approaches.
    """
    print("  Comparing transition matrix vs direct chemistry approaches...")
    
    # APPROACH 1: Generate transitions matrix and extract structural representation from it
    # This mimics the alchemy_datagen.py approach
    transitions_from_matrix = []
    potions = alchemy_env.possible_perceived_potions()
    stones = alchemy_env.possible_perceived_stones()
    
    for p, s in product(potions, stones):
        s_ = alchemy_env.apply_potion(s, p)
        # Only include valid transitions (where result differs from input)
        if not (np.array_equal(s_.perceived_coords, s.perceived_coords) and s_.reward == s.reward):
            transition_tuple = (
                tuple(s.perceived_coords),
                s.reward,
                p.index(),
                tuple(s_.perceived_coords),
                s_.reward
            )
            transitions_from_matrix.append(transition_tuple)
    
    # APPROACH 2: Generate transitions directly from chemistry structure
    # This mimics the enhanced_chemistry_generator.py approach
    transitions_from_chemistry = generate_transitions_for_chemistry(chemistry)
    
    # Sort both for comparison
    transitions_from_matrix = sorted(transitions_from_matrix)
    transitions_from_chemistry = sorted(transitions_from_chemistry)
    
    results = {
        "matrix_transition_count": len(transitions_from_matrix),
        "chemistry_transition_count": len(transitions_from_chemistry),
        "transitions_match": transitions_from_matrix == transitions_from_chemistry,
        "matrix_only": [],
        "chemistry_only": [],
        "common_transitions": 0
    }
    
    # Find differences
    matrix_set = set(transitions_from_matrix)
    chemistry_set = set(transitions_from_chemistry)
    
    results["matrix_only"] = list(matrix_set - chemistry_set)
    results["chemistry_only"] = list(chemistry_set - matrix_set)
    results["common_transitions"] = len(matrix_set & chemistry_set)
    
    return results


def compare_structural_representations(chemistry, canonical_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compare structural parts derived from graph vs. actual transitions.
    
    Returns:
        Dictionary with comparison results.
    """
    print("  Comparing structural representations...")
    
    structural_string = canonical_data["structural_string"]
    transition_structural_string = canonical_data["transition_structural_string"]
    
    # Parse both structural representations
    structural_parts = structural_string.split("||")
    transition_parts = transition_structural_string.split("||")
   
    # Remove chemistry mapping parts from structural_parts for fair comparison
    chemistry_mapping_keywords = ["POTION_MAP:", "POTION_DIR:", "STONE_MAP:", "ROTATION:"]
    filtered_structural_parts = [
        part for part in structural_parts 
        if not any(part.startswith(keyword) for keyword in chemistry_mapping_keywords)
    ]
    
    # Join first before splitting again.
    filtered_structural_parts = ''.join(filtered_structural_parts)
    transition_parts = ''.join(transition_parts)
    
    # Split on pipe to get individual parts.
    filtered_structural_parts = filtered_structural_parts.split('|')
    transition_parts = transition_parts.split('|')
    
    results = {
        "structural_parts_count": len(filtered_structural_parts),
        "transition_parts_count": len(transition_parts),
        "parts_match": set(filtered_structural_parts) == set(transition_parts),
        "structural_only": list(set(filtered_structural_parts) - set(transition_parts)),
        "transition_only": list(set(transition_parts) - set(filtered_structural_parts)),
        "common_parts": len(set(filtered_structural_parts) & set(transition_parts))
    }
    
    return results


def test_single_chemistry(chemistry_id: int) -> Dict[str, Any]:
    """Test a single chemistry configuration."""
    print(f"\nTesting Chemistry ID: {chemistry_id}")
    
    # Create chemistry using AlchemyFactory
    factory = AlchemyFactory()
    alchemy_env = factory[chemistry_id]
    
    # Create equivalent chemistry object for enhanced generator
    chemistry = type_utils.Chemistry(
        potion_map=alchemy_env.potion_map,
        stone_map=alchemy_env.stone_map,
        graph=alchemy_env.graph,
        rotation=alchemy_env.rotation
    )
    
    # Create TempAlchemyEnv
    temp_env = TempAlchemyEnv(chemistry)
    
    # Run comparisons
    apply_potion_results = compare_apply_potion_functions(alchemy_env, temp_env)
    transition_results = compare_transition_matrix_vs_direct_chemistry(alchemy_env, chemistry)
    
    # Generate canonical data
    canonical_data = generate_canonical_data_with_transitions(chemistry)
    structural_results = compare_structural_representations(chemistry, canonical_data)
    
    return {
        "chemistry_id": chemistry_id,
        "apply_potion_comparison": apply_potion_results,
        "transition_comparison": transition_results,
        "structural_comparison": structural_results,
        "canonical_data_summary": {
            "num_transitions": len(canonical_data["transitions"]),
            "is_complete": canonical_data["summary"]["is_complete"]
        }
    }


def run_equivalence_tests(num_test_chemistries: int = 5) -> None:
    """Run equivalence tests on multiple chemistry configurations."""
    print("=" * 80)
    print("EQUIVALENCE VERIFICATION TEST")
    print("=" * 80)
    print(f"Testing {num_test_chemistries} chemistry configurations...")
    
    all_results = []
    
    for i in range(num_test_chemistries):
        try:
            result = test_single_chemistry(i)
            all_results.append(result)
        except Exception as e:
            print(f"Error testing chemistry {i}: {e}")
            continue
    
    # Summarize results
    print("\n" + "=" * 80)
    print("SUMMARY OF RESULTS")
    print("=" * 80)
    
    total_tests = len(all_results)
    apply_potion_perfect_matches = 0
    transition_perfect_matches = 0
    structural_perfect_matches = 0
    
    for result in all_results:
        chem_id = result["chemistry_id"]
        apply_potion = result["apply_potion_comparison"]
        transition = result["transition_comparison"]
        structural = result["structural_comparison"]
        
        print(f"\nChemistry {chem_id}:")
        print(f"  Apply Potion: {apply_potion['matching_results']}/{apply_potion['total_tests']} matches")
        print(f"  Transitions: {'✓' if transition['transitions_match'] else '✗'} "
              f"({transition['matrix_transition_count']} vs {transition['chemistry_transition_count']})")
        print(f"  Structural: {'✓' if structural['parts_match'] else '✗'} "
              f"({structural['structural_parts_count']} vs {structural['transition_parts_count']})")
        
        if apply_potion['mismatched_results'] == 0:
            apply_potion_perfect_matches += 1
        if transition['transitions_match']:
            transition_perfect_matches += 1
        if structural['parts_match']:
            structural_perfect_matches += 1
            
        # Show mismatches if any
        if apply_potion['mismatched_results'] > 0:
            print(f"    Apply Potion Mismatches: {apply_potion['mismatched_results']}")
            for mismatch in apply_potion['mismatches'][:3]:  # Show first 3
                print(f"      {mismatch}")
        
        if not transition['transitions_match']:
            print(f"    Transition Mismatches:")
            print(f"      Matrix only: {len(transition['matrix_only'])}")
            print(f"      Chemistry only: {len(transition['chemistry_only'])}")
            
        if not structural['parts_match']:
            print(f"    Structural Mismatches:")
            print(f"      Structural only: {len(structural['structural_only'])}")
            print(f"      Transition only: {len(structural['transition_only'])}")
    
    print(f"\n" + "=" * 80)
    print(f"FINAL SUMMARY ({total_tests} chemistries tested)")
    print(f"Apply Potion Perfect Matches: {apply_potion_perfect_matches}/{total_tests}")
    print(f"Transition Perfect Matches: {transition_perfect_matches}/{total_tests}")
    print(f"Structural Perfect Matches: {structural_perfect_matches}/{total_tests}")
    
    if apply_potion_perfect_matches == total_tests and transition_perfect_matches == total_tests:
        print("\n✅ ALL TESTS PASSED - Functions are equivalent!")
    else:
        print("\n❌ SOME TESTS FAILED - Functions may not be equivalent!")
    
    print("=" * 80)


def test_structural_string_uniqueness(pickle_file_path: str = "enhanced_chemistries_with_transitions.pkl") -> None:
    """
    Load enhanced chemistries from pickle file and test uniqueness of structural strings.
    This verifies that the hashing approach using structural_string produces unique identifiers.
    """
    print("=" * 80)
    print("STRUCTURAL STRING UNIQUENESS TEST")
    print("=" * 80)
    
    try:
        # Load the pickle file
        print(f"Loading chemistries from: {pickle_file_path}")
        with open(pickle_file_path, 'rb') as f:
            chemistries_data = pickle.load(f)
        
        print(f"Loaded {len(chemistries_data)} chemistry entries")
        
        # Extract structural strings and create hashes
        structural_strings = []
        structural_hashes = set()
        duplicate_info = []
        
        # Chemistry mapping keywords to filter out (as highlighted by user)
        chemistry_mapping_keywords = ["POTION_MAP:", "POTION_DIR:", "STONE_MAP:", "ROTATION:"]
        
        for i, chemistry_data in tqdm(chemistries_data.items()):
            if 'structural_string' in chemistry_data:
                structural_string = chemistry_data['structural_string']
                
                # Filter out chemistry mapping parts (as highlighted by user)
                structural_parts = structural_string.split("||")
                filtered_structural_parts = [
                    part for part in structural_parts 
                    if not any(part.startswith(keyword) for keyword in chemistry_mapping_keywords)
                ]
                filtered_structural_string = "||".join(filtered_structural_parts)
                # Remove coordinate tuples using regex
                filtered_structural_string = re.sub(r'\([^)]*\)', '', filtered_structural_string)
                
                # Create hash of the filtered structural string
                structural_hash = hashlib.md5(filtered_structural_string.encode()).hexdigest()
                
                # Check for duplicates
                if structural_hash in structural_hashes:
                    duplicate_info.append({
                        'index': i,
                        'hash': structural_hash,
                        'structural_string': filtered_structural_string[:100] + "..." if len(filtered_structural_string) > 100 else filtered_structural_string
                    })
                    print("Warning: Duplicate structural string found!")
                else:
                    structural_hashes.add(structural_hash)
                
                structural_strings.append(filtered_structural_string)
            else:
                print(f"Warning: Chemistry {i} missing structural_string data")
        
        # Report results
        total_chemistries = len(structural_strings)
        unique_hashes = len(structural_hashes)
        num_duplicates = len(duplicate_info)
        
        print(f"\nResults:")
        print(f"  Total chemistries processed: {total_chemistries}")
        print(f"  Unique structural hashes: {unique_hashes}")
        print(f"  Duplicate structural strings: {num_duplicates}")
        print(f"  Uniqueness rate: {(unique_hashes/total_chemistries)*100:.2f}%" if total_chemistries > 0 else "  No chemistries to process")
        
        if num_duplicates > 0:
            print(f"\n❌ DUPLICATES FOUND!")
            print(f"The following {num_duplicates} chemistries have duplicate structural strings:")
            for dup in duplicate_info[:5]:  # Show first 5 duplicates
                print(f"  Index {dup['index']}: Hash {dup['hash'][:8]}... -> {dup['structural_string']}")
            if num_duplicates > 5:
                print(f"  ... and {num_duplicates - 5} more duplicates")
        else:
            print(f"\n✅ ALL STRUCTURAL STRINGS ARE UNIQUE!")
            print("The structural_string-based hashing approach successfully creates unique identifiers.")
        
        # Additional statistics
        if structural_strings:
            avg_length = sum(len(s) for s in structural_strings) / len(structural_strings)
            min_length = min(len(s) for s in structural_strings)
            max_length = max(len(s) for s in structural_strings)
            
            print(f"\nStructural String Statistics:")
            print(f"  Average length: {avg_length:.2f} characters")
            print(f"  Min length: {min_length} characters")
            print(f"  Max length: {max_length} characters")
        
    except FileNotFoundError:
        print(f"❌ ERROR: Pickle file '{pickle_file_path}' not found!")
        print("Make sure to run the enhanced chemistry generator first to create the pickle file.")
    except Exception as e:
        print(f"❌ ERROR: Failed to process pickle file: {e}")
    
    print("=" * 80)


def test_full_canonical_string_uniqueness(pickle_file_path: str = "enhanced_chemistries_with_transitions.pkl") -> None:
    """
    Test uniqueness of full canonical strings (including transitions) from pickle file.
    This tests the complete uniqueness detection approach.
    """
    print("=" * 80)
    print("FULL CANONICAL STRING UNIQUENESS TEST")
    print("=" * 80)
    
    try:
        # Load the pickle file
        print(f"Loading chemistries from: {pickle_file_path}")
        with open(pickle_file_path, 'rb') as f:
            chemistries_data = pickle.load(f)
        
        print(f"Loaded {len(chemistries_data)} chemistry entries")
        
        # Extract full canonical strings and create hashes
        canonical_strings = []
        canonical_hashes = set()
        duplicate_info = []
        
        for i, chemistry_data in enumerate(chemistries_data):
            if 'canonical_data' in chemistry_data and 'canonical_string' in chemistry_data['canonical_data']:
                canonical_string = chemistry_data['canonical_data']['canonical_string']
                
                # Create hash of the full canonical string
                canonical_hash = hashlib.md5(canonical_string.encode()).hexdigest()
                
                # Check for duplicates
                if canonical_hash in canonical_hashes:
                    duplicate_info.append({
                        'index': i,
                        'hash': canonical_hash,
                        'canonical_string': canonical_string[:100] + "..." if len(canonical_string) > 100 else canonical_string
                    })
                else:
                    canonical_hashes.add(canonical_hash)
                
                canonical_strings.append(canonical_string)
            else:
                print(f"Warning: Chemistry {i} missing canonical_string data")
        
        # Report results
        total_chemistries = len(canonical_strings)
        unique_hashes = len(canonical_hashes)
        num_duplicates = len(duplicate_info)
        
        print(f"\nResults:")
        print(f"  Total chemistries processed: {total_chemistries}")
        print(f"  Unique canonical hashes: {unique_hashes}")
        print(f"  Duplicate canonical strings: {num_duplicates}")
        print(f"  Uniqueness rate: {(unique_hashes/total_chemistries)*100:.2f}%" if total_chemistries > 0 else "  No chemistries to process")
        
        if num_duplicates > 0:
            print(f"\n❌ DUPLICATES FOUND!")
            print(f"The following {num_duplicates} chemistries have duplicate canonical strings:")
            for dup in duplicate_info[:5]:  # Show first 5 duplicates
                print(f"  Index {dup['index']}: Hash {dup['hash'][:8]}... -> {dup['canonical_string']}")
            if num_duplicates > 5:
                print(f"  ... and {num_duplicates - 5} more duplicates")
        else:
            print(f"\n✅ ALL CANONICAL STRINGS ARE UNIQUE!")
            print("The enhanced canonical string approach successfully creates unique identifiers.")
        
    except FileNotFoundError:
        print(f"❌ ERROR: Pickle file '{pickle_file_path}' not found!")
        print("Make sure to run the enhanced chemistry generator first to create the pickle file.")
    except Exception as e:
        print(f"❌ ERROR: Failed to process pickle file: {e}")
    
    print("=" * 80)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test equivalence and uniqueness of enhanced chemistry generator")
    parser.add_argument(
        "--test_type", 
        type=str, 
        default="structural_uniqueness",
        choices=["equivalence", "structural_uniqueness", "canonical_uniqueness", "all"],
        help="Type of test to run"
    )
    parser.add_argument(
        "--num_chemistries", 
        type=int, 
        default=167424,
        help="Number of chemistries to test for equivalence testing"
    )
    parser.add_argument(
        "--pickle_file", 
        type=str, 
        default="src/data/enhanced_chemistries_with_transitions.pkl",
        help="Path to the pickle file containing enhanced chemistries"
    )
    
    args = parser.parse_args()
    
    if args.test_type == "equivalence":
        # Run the equivalence tests
        run_equivalence_tests(num_test_chemistries=args.num_chemistries)
    elif args.test_type == "structural_uniqueness":
        # Test structural string uniqueness
        test_structural_string_uniqueness(args.pickle_file)
    elif args.test_type == "canonical_uniqueness":
        # Test full canonical string uniqueness
        test_full_canonical_string_uniqueness(args.pickle_file)
    elif args.test_type == "all":
        # Run all tests
        print("Running all tests...\n")
        run_equivalence_tests(num_test_chemistries=args.num_chemistries)
        print("\n")
        test_structural_string_uniqueness(args.pickle_file)
        print("\n")
        test_full_canonical_string_uniqueness(args.pickle_file)
