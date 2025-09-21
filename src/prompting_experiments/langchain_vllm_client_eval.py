#!/usr/bin/env python3
"""
LangChain + vLLM evaluation script using a separate, persistent vLLM server.
This script acts as a client to a running vLLM OpenAI-compatible server.
"""

import json
import argparse
import time
import re
import sys
import os
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from tqdm import tqdm
from datetime import datetime

# Check for required packages
try:
    from langchain_community.llms import VLLMOpenAI
    from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate
    from langchain_core.example_selectors import LengthBasedExampleSelector
    from langchain_core.output_parsers import BaseOutputParser
    from langchain_core.tracers import LangChainTracer
    from langsmith import Client
    from langsmith.evaluation import evaluate, LangChainStringEvaluator
    LANGCHAIN_AVAILABLE = True
    LANGSMITH_AVAILABLE = True
except ImportError as e:
    print("❌ Missing required packages!")
    print("Please install with:")
    print("  pip install langchain langchain-community langchain-core openai langsmith")
    print(f"\nError details: {e}")
    LANGCHAIN_AVAILABLE = False
    LANGSMITH_AVAILABLE = False


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


class ChemistryOutputParser(BaseOutputParser):
    """Parser for chemistry transformation outputs."""
    def parse(self, text: str) -> str:
        # First try to find text enclosed in square brackets
        bracket_pattern = r'\[([^\]]+)\]'
        bracket_matches = re.findall(bracket_pattern, text)
        if bracket_matches:
            return bracket_matches[0].strip()
        
        # Fallback to original pattern if no brackets found
        pattern = r'\{[^}]+\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[0]
        return text.strip()

    @property
    def _type(self) -> str:
        return "chemistry_output_parser"


class ChemistryPromptEvaluator:
    """Evaluator that connects to a remote vLLM server."""

    def __init__(self,
                 vllm_url: str = "http://localhost:8000/v1",
                 model_name: str = "meta-llama/Llama-3.2-1B",
                 temperature: float = 0.0,
                 max_tokens: int = 200,
                 enable_langsmith: bool = False,
                 experiment_name: str = None,
                 data_file_name: str = None):
        """Initialize the evaluator."""
        if not LANGCHAIN_AVAILABLE:
            raise ImportError("LangChain packages not available. Please install requirements.")

        # LangSmith setup
        self.enable_langsmith = enable_langsmith and LANGSMITH_AVAILABLE
        self.langsmith_client = None
        self.experiment_name = experiment_name or f"alchemy_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.data_file_name = data_file_name or "chemistry_eval_results.json"
        
        if self.enable_langsmith:
            try:
                # Initialize LangSmith client
                self.langsmith_client = Client()
                print(f"🔗 LangSmith enabled. Experiment: {self.experiment_name}")
                
                # Set environment variables for tracing
                os.environ["LANGCHAIN_TRACING_V2"] = "true"
                os.environ["LANGCHAIN_PROJECT"] = self.experiment_name
                
                # Create dataset in LangSmith if it doesn't exist
                self._setup_langsmith_dataset()
                
            except Exception as e:
                print(f"⚠️ Warning: Could not initialize LangSmith: {e}")
                self.enable_langsmith = False

        # Create vLLM client using official LangChain integration for OpenAI-compatible servers
        served_model_name = model_name.split("/")[-1]
        self.llm = VLLMOpenAI(
            openai_api_base=vllm_url,
            model_name=served_model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            openai_api_key="EMPTY",
        )
        self.model_name = served_model_name 
        self.output_parser = ChemistryOutputParser()
        self.example_template = PromptTemplate(
            input_variables=["input_stone", "potions", "output_stone"],
            template="{input_stone} {potions} -> {output_stone}"
        )

    def _setup_langsmith_dataset(self):
        """Setup LangSmith dataset for evaluation tracking."""
        try:
            dataset_name = f"dataset_{self.experiment_name}"
            # Check if dataset exists, create if not
            try:
                dataset = self.langsmith_client.read_dataset(dataset_name=dataset_name)
                print(f"📊 Using existing LangSmith dataset: {dataset_name}")
            except:
                dataset = self.langsmith_client.create_dataset(
                    dataset_name=dataset_name,
                    description=self.data_file_name
                )
                print(f"📊 Created new LangSmith dataset: {dataset_name}")
            
            self.dataset_name = dataset_name
        except Exception as e:
            print(f"⚠️ Warning: Could not setup LangSmith dataset: {e}")

    def _log_example_to_langsmith(self, 
                                  query_example: str, 
                                  support_examples: List[str], 
                                  result: Dict[str, Any]):
        """Log individual evaluation example to LangSmith."""
        if not self.enable_langsmith or not self.langsmith_client:
            return
            
        try:
            # Create example in dataset
            example_data = {
                "query": query_example,
                "support_count": len(support_examples),
                "expected_output": result.get("expected_output"),
                "model_response": result.get("model_response"),
                "predicted_output": result.get("predicted_output"),
                "correct": result.get("correct"),
                "prompt": result.get("prompt", ""),  # Truncate for storage
            }
            
            self.langsmith_client.create_example(
                inputs={"query": query_example, "support_examples": support_examples},  # Limit support examples
                outputs={"expected": result.get("expected_output")},
                dataset_name=self.dataset_name,
                metadata=example_data
            )
        except Exception as e:
            print(f"⚠️ Warning: Could not log to LangSmith: {e}")

    def _log_episode_to_langsmith(self, 
                                  episode_id: str, 
                                  support_examples: List[str], 
                                  episode_results: List[Dict[str, Any]]):
        """Log entire episode results to LangSmith in batch."""
        if not self.enable_langsmith or not self.langsmith_client:
            return
            
        try:
            # Calculate episode-level metrics
            total_queries = len(episode_results)
            correct_queries = sum(1 for r in episode_results if r.get("correct", False))
            failed_queries = sum(1 for r in episode_results if "error" in r and r["error"])
            episode_accuracy = correct_queries / total_queries if total_queries > 0 else 0.0
            
            # Prepare episode summary
            episode_summary = {
                "episode_id": episode_id,
                "total_queries": total_queries,
                "correct_queries": correct_queries,
                "failed_queries": failed_queries,
                "accuracy": episode_accuracy,
                "support_count": len(support_examples),
                "timestamp": datetime.now().isoformat()
            }
            
            # Create a single dataset example for the entire episode
            episode_inputs = {
                "episode_id": episode_id,
                "support_examples": support_examples[:10],  # Limit to first 10 for storage
                "query_count": total_queries
            }
            
            episode_outputs = {
                "accuracy": episode_accuracy,
                "correct_count": correct_queries,
                "total_count": total_queries
            }
            
            # Include sample results for debugging (limit to first 5 and any errors)
            sample_results = []
            error_results = []
            
            for i, result in enumerate(episode_results):
                sample_results.append({
                    "query": result.get("query", ""),
                    "expected": result.get("expected_output", ""),
                    "predicted": result.get("predicted_output", ""),
                    "correct": result.get("correct", False)
                })
                
                if "error" in result and result["error"]:  # All error cases
                    error_results.append({
                        "query": result.get("query", ""),
                        "error": result.get("error", "")
                    })
            
            episode_metadata = {
                **episode_summary,
                "sample_results": sample_results,
                "error_results": error_results,
                "model": self.model_name
            }
            
            # Create single example in LangSmith dataset for this episode
            self.langsmith_client.create_example(
                inputs=episode_inputs,
                outputs=episode_outputs,
                dataset_name=self.dataset_name,
                metadata=episode_metadata
            )
            
            print(f"📊 Logged episode {episode_id} to LangSmith: {correct_queries}/{total_queries} correct ({episode_accuracy:.3f})")
            
        except Exception as e:
            print(f"⚠️ Warning: Could not log episode to LangSmith: {e}")

    def parse_chemistry_example(self, example_text: str) -> ChemistryExample:
        """Parse a chemistry example from text format."""
        pattern = r'(\{[^}]+\})\s+([A-Z\s]+?)\s+->\s+(\{[^}]+\})'
        match = re.match(pattern, example_text.strip())
        if not match:
            raise ValueError(f"Could not parse chemistry example: {example_text}")
        input_stone, potions_str, output_stone = match.groups()
        return ChemistryExample(input_stone, potions_str.strip().split(), output_stone, example_text.strip())

    def create_few_shot_prompt(self, support_examples: List[str]) -> FewShotPromptTemplate:
        """Create a few-shot prompt template."""
        examples = []
        for example_text in support_examples:
            try:
                parsed = self.parse_chemistry_example(example_text)
                # Escape the braces in the stone descriptions to prevent format errors
                escaped_input_stone = parsed.input_stone.replace("{", "{{").replace("}", "}}")
                escaped_output_stone = parsed.output_stone.replace("{", "{{").replace("}", "}}")

                examples.append({
                    "input_stone": escaped_input_stone,
                    "potions": " ".join(parsed.potions),
                    "output_stone": escaped_output_stone
                })
            except ValueError as e:
                print(f"Warning: Skipping malformed support example: {e}")
        
        # This template now expects escaped strings for stone descriptions
        escaped_example_template = PromptTemplate(
            input_variables=["input_stone", "potions", "output_stone"],
            template="{input_stone} {potions} -> {output_stone}"
        )

        example_selector = LengthBasedExampleSelector(examples=examples, example_prompt=escaped_example_template, max_length=3000)
        
        return FewShotPromptTemplate(
            example_selector=example_selector,
            example_prompt=escaped_example_template,
            prefix="You are an expert in magical chemistry. Given stone states and potions, predict the resulting stone state after applying the potions.\n\nHere are some examples of transformations:",
            suffix="Now, predict the output stone state for the following example when a potion is applied to an input stone state. Just print the output stone state. \n{input_stone} {potions} -> ",
            input_variables=["input_stone", "potions"],
        )

    def evaluate_single_example(self, support_examples: List[str], query_example: str) -> Dict[str, Any]:
        """Evaluate a single query example."""
        try:
            query_parsed = self.parse_chemistry_example(query_example)
            few_shot_prompt = self.create_few_shot_prompt(support_examples)
            
            prompt = few_shot_prompt.format(input_stone=query_parsed.input_stone, potions=" ".join(query_parsed.potions))
            
            # Add LangSmith tracing context
            model_response = self.llm.invoke(prompt)
                
            predicted_output = self.output_parser.parse(model_response)
            
            correct = predicted_output == query_parsed.output_stone
            result = {
                "query": query_example, 
                "expected_output": query_parsed.output_stone, 
                "model_response": model_response, 
                "predicted_output": predicted_output, 
                "correct": correct, 
                "prompt": prompt
            }
            
            # Remove individual logging - we'll batch this at episode level
            # if self.enable_langsmith:
            #     self._log_example_to_langsmith(query_example, support_examples, result)
            
            return result
            
        except Exception as e:
            return {"query": query_example, "expected_output": None, "model_response": None, "predicted_output": None, "correct": False, "error": str(e)}

    def evaluate_dataset(self, dataset: Dict[str, List[str]]) -> EvaluationResults:
        """Evaluate the entire dataset."""
        detailed_results, correct_count, failed_count = [], 0, 0
        episodes = dataset.get("episodes", [])
        
        # Log experiment start to LangSmith
        if self.enable_langsmith:
            self._log_experiment_start(dataset)
        
        for episode_id, episode in tqdm(episodes.items(), desc="Evaluating episodes"):
            
            support_examples, query_examples = episode["support"], episode["query"]
            episode_results, episode_correct_count, episode_failed_count = [], 0, 0
            
            for i, query in enumerate(query_examples):
                print(f"⚗️  Processing query {i+1}/{len(query_examples)}")
                result = self.evaluate_single_example(support_examples, query)
                episode_results.append(result)
                if "error" in result and result["error"]:
                    episode_failed_count += 1
                    print(f"  ❌ Error: {result['error']}")
                elif result["correct"]:
                    episode_correct_count += 1
                    print(f"  ✅ Correct, Predicted: {result['predicted_output']}")
                else:
                    print(f"  ❌ Incorrect - Expected: {result['expected_output']}, Got: {result['predicted_output']}")
            
            # Log entire episode to LangSmith after processing all queries
            if self.enable_langsmith:
                self._log_episode_to_langsmith(episode_id, support_examples, episode_results)
                
            total_queries = len(query_examples)
            detailed_results.extend(episode_results)
            correct_count += episode_correct_count
            failed_count += episode_failed_count
        
        accuracy = correct_count / len(detailed_results) if detailed_results else 0.0
        results = EvaluationResults(len(detailed_results), correct_count, accuracy, failed_count, detailed_results)
        
        # Log experiment completion to LangSmith
        if self.enable_langsmith:
            self._log_experiment_completion(results)
        
        return results

    def _log_experiment_start(self, dataset: Dict[str, Any]):
        """Log experiment start to LangSmith."""
        if not self.enable_langsmith:
            return
            
        try:
            metadata = {
                "model": self.model_name,
                "num_episodes": len(dataset.get("episodes", {})),
                "start_time": datetime.now().isoformat(),
                "experiment_type": "chemistry_transformation_prediction"
            }
            print(f"📝 Logged experiment start to LangSmith: {self.experiment_name}")
        except Exception as e:
            print(f"⚠️ Warning: Could not log experiment start: {e}")

    def _log_experiment_completion(self, results: EvaluationResults):
        """Log experiment completion to LangSmith.""" 
        if not self.enable_langsmith:
            return
            
        try:
            metadata = {
                "accuracy": results.accuracy,
                "total_queries": results.total_queries,
                "correct_predictions": results.correct_predictions,
                "failed_predictions": results.failed_predictions,
                "completion_time": datetime.now().isoformat()
            }
            print(f"📊 Logged experiment completion to LangSmith with accuracy: {results.accuracy:.3f}")
        except Exception as e:
            print(f"⚠️ Warning: Could not log experiment completion: {e}")

    def save_results(self, results: EvaluationResults, output_file: str):
        """Save evaluation results to a file."""
        output_data = {"summary": results.__dict__, "detailed_results": results.detailed_results, "metadata": {"llm_type": "langchain_vllm_client", "model_name": self.model_name}}
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"💾 Results saved to {output_file}")



def load_data_from_file(file_path: str, use_all_episodes: bool = False) -> Dict[str, List[str]]:
    """Load chemistry data from a JSON file."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    if isinstance(data, dict):
        if "support" in data and "query" in data:
            return data
        elif "episodes" in data:
            if use_all_episodes:
                return data
            else:
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
            "{color: blue, size: small, roundness: medium_round, reward: -1} YELLOW CYAN -> {color: purple, size: large, roundness: pointy, reward: 3}",
            "{color: purple, size: large, roundness: pointy, reward: 3} GREEN PINK -> {color: purple, size: large, roundness: round, reward: -1}",
        ],
        "query": [
            "{color: purple, size: small, roundness: pointy, reward: 1} PINK -> {color: blue, size: small, roundness: medium_round, reward: -1}",
            "{color: blue, size: small, roundness: medium_round, reward: -1} YELLOW -> {color: blue, size: large, roundness: medium_round, reward: 1}",
        ]
    }


def main():
    if not LANGCHAIN_AVAILABLE:
        sys.exit(1)

    parser = argparse.ArgumentParser(description="Chemistry evaluation client for a remote vLLM server.")
    parser.add_argument("--data", type=str, help="Path to JSON file with support/query data", 
                        default='/home/rsaha/projects/dm_alchemy/src/data/complete_graph_generated_data_enhanced_qnodes_in_snodes/decompositional_chemistry_samples_167424_80_unique_stones_val_shop_2_qhop_1_seed_0.json')
    parser.add_argument("--output", type=str, default="langchain_vllm_client_results.json", help="Output file for results")
    parser.add_argument("--vllm-url", type=str, default="http://localhost:8000/v1", help="vLLM server OpenAI-compatible API URL")
    parser.add_argument("--model-name", type=str, default="meta-llama/Llama-3.2-1B", help="Model name being served")
    parser.add_argument("--use-sample-data", action="store_true", help="Use built-in sample data for testing", default=False)
    parser.add_argument("--max_new_tokens", type=int, default=512, help="Max new tokens to generate")
    
    # LangSmith arguments
    parser.add_argument("--enable-langsmith", action="store_true", help="Enable LangSmith monitoring", default=True)
    
    args = parser.parse_args()

    # Set up LangSmith API key if provided
    keys = json.load(open('keys.json'))
    os.environ["LANGSMITH_API_KEY"] = keys['langsmith']
        
    # Create experiment_name.
    if args.enable_langsmith:
        file_path = ''.join(args.data.split('/')[-2:]).replace('.json', '') if args.data else "sample_data"
        experiment_name = f"{args.model_name.replace('/', '_')}_{file_path}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    evaluator = ChemistryPromptEvaluator(
        vllm_url=args.vllm_url,
        model_name=args.model_name,
        max_tokens=args.max_new_tokens,
        enable_langsmith=args.enable_langsmith,
        experiment_name=experiment_name
    )

    if args.use_sample_data:
        data = create_sample_data()
    elif args.data:
        data = load_data_from_file(args.data, use_all_episodes=True)
    else:
        print("❌ Error: Must provide --data or --use-sample-data")
        sys.exit(1)

    results = evaluator.evaluate_dataset(data)
    print(f"\n🏆 EVALUATION COMPLETE: Accuracy: {results.accuracy:.3f}")
    
    if args.enable_langsmith:
        print(f"📊 View results in LangSmith: https://smith.langchain.com/projects/{evaluator.experiment_name}")
        
    # Change args.output according the to the experiment name
    # Create a new directory for the prompting results if it doesn't exist
    if not os.path.exists('prompting_results'):
        os.makedirs('prompting_results')
    args.output = f'prompting_results/{evaluator.experiment_name}_results.json'
    
    evaluator.save_results(results, args.output)


if __name__ == "__main__":
    main()