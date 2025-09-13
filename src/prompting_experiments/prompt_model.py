#!/usr/bin/env python3
"""
Chemistry prompting experiments using vLLM.
This script evaluates LLMs on chemistry transformation prediction tasks
using in-context learning with support/query splits.
"""

import json
import argparse
import requests
import time
import random
from typing import Dict, List, Tuple, Any, Optional
import re
from dataclasses import dataclass
from pathlib import Path
import sys
import os

# Add parent directory to path to import data utilities
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'data'))


@dataclass
class ChemistryExample:
    """Represents a single chemistry transformation example."""
    input_stone: str
    potions: List[str]
    output_stone: str
    raw_text: str


@dataclass
class EvaluationResults:
    """Store evaluation results."""
    total_queries: int
    correct_predictions: int
    accuracy: float
    failed_predictions: int
    detailed_results: List[Dict[str, Any]]


class ChemistryPromptEvaluator:
    """Evaluator for chemistry transformation prompting experiments."""
    
    def __init__(self, 
                 vllm_url: str = "http://localhost:8000/v1/completions",
                 model_name: str = "meta-llama/Llama-2-7b-chat-hf",
                 temperature: float = 0.0,
                 max_tokens: int = 200):
        """Initialize the evaluator."""
        self.vllm_url = vllm_url
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        
    def parse_chemistry_example(self, example_text: str) -> ChemistryExample:
        """Parse a chemistry example from text format."""
        # Pattern to match: {stone_state} POTION(S) -> {stone_state}
        pattern = r'(\{[^}]+\})\s+([A-Z\s]+?)\s+->\s+(\{[^}]+\})'
        match = re.match(pattern, example_text.strip())
        
        if not match:
            raise ValueError(f"Could not parse chemistry example: {example_text}")
        
        input_stone = match.group(1)
        potions_str = match.group(2).strip()
        output_stone = match.group(3)
        
        # Split potions by spaces (assuming they are space-separated)
        potions = potions_str.split()
        
        return ChemistryExample(
            input_stone=input_stone,
            potions=potions,
            output_stone=output_stone,
            raw_text=example_text.strip()
        )
    
    def create_prompt(self, support_examples: List[str], query_input: str, query_potions: List[str]) -> str:
        """Create the prompt for in-context learning."""
        
        # Parse support examples
        support_parsed = []
        for example in support_examples:
            try:
                parsed = self.parse_chemistry_example(example)
                support_parsed.append(parsed)
            except ValueError as e:
                print(f"Warning: Skipping malformed support example: {e}")
                continue
        
        # Build the prompt
        prompt_parts = []
        prompt_parts.append("You are an expert in magical chemistry. Given stone states and potions, predict the resulting stone state after applying the potions.")
        prompt_parts.append("\nHere are some examples of transformations:\n")
        
        # Add support examples
        for i, example in enumerate(support_parsed, 1):
            potions_str = " ".join(example.potions)
            prompt_parts.append(f"Example {i}: {example.input_stone} {potions_str} -> {example.output_stone}")
        
        # Add the query
        query_potions_str = " ".join(query_potions)
        prompt_parts.append(f"\nNow, predict the output for:")
        prompt_parts.append(f"{query_input} {query_potions_str} -> ")
        
        return "\n".join(prompt_parts)
    
    def query_vllm(self, prompt: str) -> Optional[str]:
        """Send a query to the vLLM server."""
        payload = {
            "model": self.model_name.split("/")[-1],  # Use just model name
            "prompt": prompt,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": 1.0,
            "stop": ["\n", "Example", "Now,"]
        }
        
        try:
            response = requests.post(self.vllm_url, json=payload, timeout=30)
            response.raise_for_status()
            result = response.json()
            
            if "choices" in result and len(result["choices"]) > 0:
                return result["choices"][0]["text"].strip()
            else:
                print(f"Unexpected response format: {result}")
                return None
                
        except requests.exceptions.RequestException as e:
            print(f"Error querying vLLM server: {e}")
            return None
        except json.JSONDecodeError as e:
            print(f"Error parsing response: {e}")
            return None
    
    def extract_predicted_output(self, model_response: str) -> Optional[str]:
        """Extract the predicted stone state from model response."""
        if not model_response:
            return None
        
        # Look for a stone state pattern in the response
        pattern = r'\{[^}]+\}'
        matches = re.findall(pattern, model_response)
        
        if matches:
            # Return the first (or most complete) match
            return matches[0]
        
        return None
    
    def evaluate_single_example(self, 
                               support_examples: List[str], 
                               query_example: str) -> Dict[str, Any]:
        """Evaluate a single query example."""
        
        try:
            # Parse the query example
            query_parsed = self.parse_chemistry_example(query_example)
            
            # Create prompt with support examples and query input
            prompt = self.create_prompt(
                support_examples, 
                query_parsed.input_stone, 
                query_parsed.potions
            )
            
            # Query the model
            model_response = self.query_vllm(prompt)
            
            if model_response is None:
                return {
                    "query": query_example,
                    "expected_output": query_parsed.output_stone,
                    "model_response": None,
                    "predicted_output": None,
                    "correct": False,
                    "error": "Failed to get model response"
                }
            
            # Extract predicted output
            predicted_output = self.extract_predicted_output(model_response)
            
            # Check if prediction is correct
            correct = predicted_output == query_parsed.output_stone
            
            return {
                "query": query_example,
                "expected_output": query_parsed.output_stone,
                "model_response": model_response,
                "predicted_output": predicted_output,
                "correct": correct,
                "prompt": prompt
            }
            
        except Exception as e:
            return {
                "query": query_example,
                "expected_output": None,
                "model_response": None,
                "predicted_output": None,
                "correct": False,
                "error": str(e)
            }
    
    def evaluate_dataset(self, data: Dict[str, List[str]]) -> EvaluationResults:
        """Evaluate the entire dataset."""
        
        support_examples = data["support"]
        query_examples = data["query"]
        
        print(f"Evaluating {len(query_examples)} queries with {len(support_examples)} support examples")
        
        detailed_results = []
        correct_count = 0
        failed_count = 0
        
        for i, query in enumerate(query_examples):
            print(f"Processing query {i+1}/{len(query_examples)}")
            
            result = self.evaluate_single_example(support_examples, query)
            detailed_results.append(result)
            
            if "error" in result:
                failed_count += 1
                print(f"  Error: {result['error']}")
            elif result["correct"]:
                correct_count += 1
                print(f"  ✓ Correct")
            else:
                print(f"  ✗ Incorrect - Expected: {result['expected_output']}, Got: {result['predicted_output']}")
            
            # Small delay to avoid overwhelming the server
            time.sleep(0.1)
        
        total_queries = len(query_examples)
        accuracy = correct_count / total_queries if total_queries > 0 else 0.0
        
        return EvaluationResults(
            total_queries=total_queries,
            correct_predictions=correct_count,
            accuracy=accuracy,
            failed_predictions=failed_count,
            detailed_results=detailed_results
        )
    
    def save_results(self, results: EvaluationResults, output_file: str):
        """Save evaluation results to a file."""
        output_data = {
            "summary": {
                "total_queries": results.total_queries,
                "correct_predictions": results.correct_predictions,
                "accuracy": results.accuracy,
                "failed_predictions": results.failed_predictions
            },
            "detailed_results": results.detailed_results
        }
        
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"Results saved to {output_file}")


def load_data_from_file(file_path: str) -> Dict[str, List[str]]:
    """Load chemistry data from a JSON file."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # Handle different possible data structures
    if isinstance(data, dict):
        if "support" in data and "query" in data:
            return data
        elif "episodes" in data:
            # Handle episode-based structure - use first episode
            first_episode = list(data["episodes"].values())[0]
            return {
                "support": first_episode["support"],
                "query": first_episode["query"]
            }
    
    raise ValueError("Data format not recognized. Expected dict with 'support' and 'query' keys")


def create_sample_data() -> Dict[str, List[str]]:
    """Create sample data for testing."""
    return {
        "support": [
            "{color: purple, size: small, roundness: pointy, reward: 1} YELLOW PINK -> {color: blue, size: large, roundness: medium_round, reward: 1}",
            "{color: purple, size: small, roundness: pointy, reward: 1} GREEN YELLOW -> {color: red, size: large, roundness: medium_round, reward: 1}",
            "{color: blue, size: small, roundness: medium_round, reward: -1} GREEN CYAN -> {color: red, size: small, roundness: medium_round, reward: -1}",
        ],
        "query": [
            "{color: purple, size: small, roundness: pointy, reward: 1} PINK -> {color: blue, size: small, roundness: medium_round, reward: -1}",
            "{color: blue, size: small, roundness: medium_round, reward: -1} YELLOW -> {color: blue, size: large, roundness: medium_round, reward: 1}",
        ]
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate chemistry transformation prediction")
    parser.add_argument("--data", type=str,
                        help="Path to JSON file with support/query data")
    parser.add_argument("--output", type=str, default="chemistry_evaluation_results.json",
                        help="Output file for results")
    parser.add_argument("--vllm-url", type=str, default="http://localhost:8000/v1/completions",
                        help="vLLM server URL")
    parser.add_argument("--model-name", type=str, default="meta-llama/Llama-2-7b-chat-hf",
                        help="Model name")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Sampling temperature")
    parser.add_argument("--max-tokens", type=int, default=200,
                        help="Maximum tokens in response")
    parser.add_argument("--use-sample-data", action="store_true",
                        help="Use built-in sample data for testing")
    
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = ChemistryPromptEvaluator(
        vllm_url=args.vllm_url,
        model_name=args.model_name,
        temperature=args.temperature,
        max_tokens=args.max_tokens
    )
    
    # Load data
    if args.use_sample_data:
        print("Using sample data for testing")
        data = create_sample_data()
    elif args.data:
        print(f"Loading data from {args.data}")
        data = load_data_from_file(args.data)
    else:
        print("Error: Must provide --data or --use-sample-data")
        sys.exit(1)
    
    print(f"Loaded {len(data['support'])} support examples and {len(data['query'])} query examples")
    
    # Run evaluation
    results = evaluator.evaluate_dataset(data)
    
    # Print summary
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"Total Queries: {results.total_queries}")
    print(f"Correct Predictions: {results.correct_predictions}")
    print(f"Failed Predictions: {results.failed_predictions}")
    print(f"Accuracy: {results.accuracy:.3f} ({results.accuracy*100:.1f}%)")
    
    # Save results
    evaluator.save_results(results, args.output)


if __name__ == "__main__":
    main()