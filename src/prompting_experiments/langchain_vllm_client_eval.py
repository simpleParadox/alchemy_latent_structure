#!/usr/bin/env python3
"""
LangChain + vLLM evaluation script using a separate, persistent vLLM server.
This script acts as a client to a running vLLM OpenAI-compatible server with wandb logging.
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
import json

# Check for required packages
try:
    from langchain_community.llms import VLLMOpenAI
    from langchain_openai import ChatOpenAI
    from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate, ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
    from langchain_core.example_selectors import LengthBasedExampleSelector
    from langchain_core.output_parsers import BaseOutputParser
    from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
    LANGCHAIN_AVAILABLE = True
except ImportError as e:
    print("❌ Missing required LangChain packages!")
    print("Please install with:")
    print("  pip install langchain langchain-community langchain-core openai")
    print(f"\nError details: {e}")
    LANGCHAIN_AVAILABLE = False

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError as e:
    print("❌ Missing wandb package!")
    print("Please install with:")
    print("  pip install wandb")
    print(f"\nError details: {e}")
    WANDB_AVAILABLE = False



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
    
    def __init__(self, use_stone_ids: bool = False):
        self.use_stone_ids = use_stone_ids
    
    def parse(self, text: str) -> str:
        if self.use_stone_ids:
            # Pattern for stone ID: "Therefore, the output stone ID is: SX"
            id_pattern = r"Therefore, the output stone ID is:\s*(S\d+)"
            match = re.search(id_pattern, text, re.DOTALL)
            if match:
                return match.group(1).strip()
            
            # Fallback: look for any SX pattern
            fallback_pattern = r'S\d+'
            matches = re.findall(fallback_pattern, text)
            if matches:
                return matches[-1].strip()
        else:
            # Original stone state parsing
            specific_pattern = r"Therefore, the output stone state is:\s*(\{.*?\})"
            match = re.search(specific_pattern, text, re.DOTALL)
            if match:
                return match.group(1).strip()

            fallback_pattern = r'\{[^}]+\}'
            matches = re.findall(fallback_pattern, text)
            if matches:
                return matches[-1].strip()
        
        return text.strip()

    @property
    def _type(self) -> str:
        return "chemistry_output_parser"

def evaluate_feature_level(self, expected_stone_state: str, predicted_stone_state: str) -> Dict[str, Any]:
    """Evaluate individual features between expected and predicted stone states."""
    try:
        # Parse both stone states to extract features
        expected_features = self.parse_stone_features(expected_stone_state)
        predicted_features = self.parse_stone_features(predicted_stone_state)
        
        feature_evaluation = {
            "color_correct": expected_features.get("color") == predicted_features.get("color"),
            "size_correct": expected_features.get("size") == predicted_features.get("size"),
            "roundness_correct": expected_features.get("roundness") == predicted_features.get("roundness"),
            "reward_correct": expected_features.get("reward") == predicted_features.get("reward"),
        }
        
        feature_evaluation["features_correct_count"] = sum(feature_evaluation.values())
        feature_evaluation["features_accuracy"] = feature_evaluation["features_correct_count"] / 4.0
        
        return feature_evaluation
        
    except Exception as e:
        return {"error": str(e), "features_correct_count": 0, "features_accuracy": 0.0}

def parse_stone_features(self, stone_state: str) -> Dict[str, str]:
    """Parse individual features from a stone state string."""
    features = {}
    
    # Extract features using regex
    color_match = re.search(r'color:\s*([^,}]+)', stone_state)
    size_match = re.search(r'size:\s*([^,}]+)', stone_state)
    roundness_match = re.search(r'roundness:\s*([^,}]+)', stone_state)
    reward_match = re.search(r'reward:\s*([^,}]+)', stone_state)
    
    if color_match:
        features["color"] = color_match.group(1).strip()
    if size_match:
        features["size"] = size_match.group(1).strip()
    if roundness_match:
        features["roundness"] = roundness_match.group(1).strip()
    if reward_match:
        features["reward"] = reward_match.group(1).strip()
    
    return features

class ChemistryPromptEvaluator:
    """Evaluator that connects to a remote vLLM server."""

    def __init__(self,
                 vllm_url: str = "http://localhost:8000/v1",
                 model_name: str = "meta-llama/Llama-3.2-1B",
                 temperature: float = 0.0,
                 max_tokens: int = 200,
                 enable_wandb: bool = False,
                 experiment_name: str = None,
                 data_file_name: str = None,
                 api_key: str = "EMPTY",
                 use_chat_api: bool = True,
                 provider: str = "vllm",
                 shop_length: int = 2,
                 qhop_length: int = 1,
                 data_split_seed=0,
                 batch_size: int = 1,  # NEW: Batch size for inference
                 enable_batch_inference: bool = False,  # NEW: Enable/disable batching
                 enable_reasoning: bool = True,  # NEW: Enable/disable reasoning
                 use_stone_ids: bool = False):  # NEW: Use stone IDs
        """Initialize the evaluator.
        
        Args:
            use_chat_api: If True, use ChatOpenAI with /v1/chat/completions endpoint.
                         If False, use VLLMOpenAI with /v1/completions endpoint.
            provider: "vllm" for local vLLM server, "fireworks" for FireworksAI API.
        """
        if not LANGCHAIN_AVAILABLE:
            raise ImportError("LangChain packages not available. Please install requirements.")

        # W&B setup
        self.enable_wandb = enable_wandb and WANDB_AVAILABLE
        self.experiment_name = experiment_name or f"alchemy_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.data_file_name = data_file_name or "chemistry_eval_results.json"
        
        # Store all results for W&B table
        self.wandb_table_data = []
        
        if self.enable_wandb:
            try:
                # Initialize wandb
                wandb.init(
                    project="alchemy_llm_evaluation",
                    name=self.experiment_name,
                    config={
                        "model_name": model_name,
                        "temperature": temperature,
                        "max_tokens": max_tokens,
                        "vllm_url": vllm_url,
                        "data_file": self.data_file_name,
                        "use_chat_api": use_chat_api,
                        "provider": provider,
                        "shop_length": shop_length,
                        "qhop_length": qhop_length,
                        "data_split_seed": data_split_seed
                    }
                )
                print(f"🔗 W&B enabled. Experiment: {self.experiment_name}")
                
            except Exception as e:
                print(f"⚠️ Warning: Could not initialize W&B: {e}")
                self.enable_wandb = False

        # Store configuration
        self.use_chat_api = use_chat_api
        self.provider = provider
        
        # Create LLM client based on configuration
        served_model_name = model_name.split("/")[-1]
        
        if provider == "fireworks":
            # Use FireworksAI API
            if use_chat_api:
                try:
                    from langchain_fireworks import ChatFireworks
                    self.llm = ChatFireworks(
                        model=model_name,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        timeout=None,
                        api_key=api_key,
                    )
                    print(f"🔥 Using FireworksAI ChatFireworks interface for {model_name}")
                except ImportError:
                    print("❌ langchain_fireworks not available. Install with: pip install langchain-fireworks")
                    # Fallback to OpenAI-compatible interface
                    self.llm = ChatOpenAI(
                        base_url="https://api.fireworks.ai/inference/v1",
                        model=model_name,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        api_key=api_key,
                    )
                    print(f"🔥 Using FireworksAI via ChatOpenAI interface for {model_name}")
            else:
                # Completions interface for Fireworks (less common)
                self.llm = VLLMOpenAI(
                    openai_api_base="https://api.fireworks.ai/inference/v1",
                    model_name=model_name,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    openai_api_key=api_key,
                )
                print(f"🔥 Using FireworksAI completions interface for {model_name}")
            
        else:  # vllm provider
            if use_chat_api:
                # Use ChatOpenAI for chat interface (/v1/chat/completions)
                self.llm = ChatOpenAI(
                    base_url=vllm_url,
                    model=served_model_name,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    api_key=api_key,
                )
                print(f"💬 Using vLLM ChatOpenAI interface (/v1/chat/completions) for {model_name}")
            else:
                # Use VLLMOpenAI for completions interface (/v1/completions) - legacy mode
                self.llm = VLLMOpenAI(
                    openai_api_base=vllm_url,
                    model_name=served_model_name,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    openai_api_key=api_key,
                )
                print(f"📝 Using vLLM completions interface (/v1/completions) for {model_name}")
                
        self.model_name = served_model_name 
        self.full_model_name = model_name
        self.output_parser = ChemistryOutputParser()
        self.example_template = PromptTemplate(
            input_variables=["input_stone", "potions", "output_stone"],
            template="{input_stone} {potions} -> {output_stone}"
        )
        self.batch_size = batch_size
        self.enable_batch_inference = enable_batch_inference
        self.enable_reasoning = enable_reasoning  # NEW
        self.use_stone_ids = use_stone_ids  # NEW

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
            prefix="Your task is to understand the latent structure from a set of support examples and predict the output of a new query example by leveraing the latent structure. \nYou will be given many examples where there will be input stone states containing four features, and single or multiple potions of different colors will be applied to the input stone state. Applying the potion(s) on the input stone states will change its feature(s) to give the output stone state.\nHere are the support examples that show the input / output mappings:",
            suffix="Now, predict the output stone state for the following example when a potion is applied to an input stone state. Just print the output stone state. \n{input_stone} {potions} -> ",
            input_variables=["input_stone", "potions"],
        )

    def create_chat_prompt_template(self, support_examples: List[str]) -> ChatPromptTemplate:
        """
        Create a chat-based prompt template that safely selects few-shot examples
        to fit within the context window.
        """
        
        # System message with high-level task description
        system_message = """You are an expert in understanding latent structures in chemistry transformations. Your task is to analyze support examples showing how potions transform input stone states to output stone states, then predict the output for new examples."""
        

        # 1. Create a template for a single example line to measure length.
        #    Note: The variables don't matter here, it's just for the selector.
        example_prompt = PromptTemplate(
            input_variables=["input_stone", "potions", "output_stone"],
            template="{input_stone} {potions} -> {output_stone}"
        )

        # 2. Manually parse and prepare all available examples for the selector.
        all_parsed_examples = []
        for example_text in support_examples:
            try:
                parsed = self.parse_chemistry_example(example_text)
                all_parsed_examples.append({
                    "input_stone": parsed.input_stone,
                    "potions": " ".join(parsed.potions),
                    "output_stone": parsed.output_stone,
                    "raw_text": parsed.raw_text
                })
            except ValueError:
                print(f"Warning: Skipping malformed support example: {example_text}")
                continue # Skip malformed examples

        # 3. Use LengthBasedExampleSelector to pick a safe number of examples.
        #    This replicates the logic from the old few-shot template.
        example_selector = LengthBasedExampleSelector(
            examples=[ex for ex in all_parsed_examples],
            example_prompt=example_prompt,
            max_length=10000  # Use a safe length limit
        )
        
        # The selector needs an empty input dict to select based on length
        selected_examples = example_selector.select_examples({})
        
        # 4. Build the final examples string from the *selected* examples' raw text.
        examples_string = "\n".join([ex['raw_text'] for ex in selected_examples])

        # 5. Construct the final prompt template with the safely selected examples.
        human_message_prompt = HumanMessagePromptTemplate.from_template(
            "Each stone state has four features: color, size, roundness, and reward value. Potions are represented by color names (e.g., RED, BLUE, GREEN, YELLOW, ORANGE, PINK) and can be applied individually or in combination. The application of the potion(s) modifies the features of the input stone to produce the output stone. \n \n Analyze the patterns in the support examples to understand the transformation rules, then apply this knowledge to the output of a new query example where potion(s) are applied to an input stone state.
            
             Do not print reasoning steps, only predict 
             Print the output stone state for the query example by saying: 'Therefore, the output stone state is:'"
            "Here are some examples of transformations:\n"
            "---START EXAMPLES---\n"
            "{examples}\n"
            "---END EXAMPLES---\n\n"
            "Now, predict the output for the following example:\n"
            "{input_stone} {potions} ->"
        )

        chat_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=system_message),
            human_message_prompt
        ])

        # Pre-fill the {examples} placeholder with our dynamically selected examples
        return chat_prompt.partial(examples=examples_string)

    def evaluate_single_example(self, support_examples: List[str], query_example: str, episode_id: str) -> Dict[str, Any]:
        """Evaluate a single query example."""
        try:
            query_parsed = self.parse_chemistry_example(query_example)
            
            if self.use_chat_api:
                # Use chat-based prompting
                chat_prompt = self.create_chat_prompt_template(support_examples)
                # For chat API, no need to escape braces
                formatted_prompt = chat_prompt.format_messages(
                    input_stone=query_parsed.input_stone, 
                    potions=" ".join(query_parsed.potions)
                )
                # # Get model response
                # messages = [
                #     ('system', formatted_prompt[0].content),
                #     ('human', formatted_prompt[1].content)
                # ]
                
                
                model_response = self.llm.invoke(formatted_prompt)
                # Extract content from AIMessage
                if hasattr(model_response, 'content'):
                    model_response_text = model_response.content
                    # print(f"Model response (raw): {model_response_text}")
                else:
                    model_response_text = str(model_response)
                prompt_text = f"Chat messages: {[msg.content for msg in formatted_prompt]}"
            else:
                # Use traditional few-shot prompting (legacy mode)
                few_shot_prompt = self.create_few_shot_prompt(support_examples)
                # Escape braces in query input for prompt formatting
                escaped_query_input = query_parsed.input_stone.replace("{", "{{").replace("}", "}}")
                prompt_text = few_shot_prompt.format(input_stone=escaped_query_input, potions=" ".join(query_parsed.potions))
                # Get model response
                model_response = self.llm.invoke(prompt_text)
                model_response_text = str(model_response)
            
            predicted_output = self.output_parser.parse(model_response_text)
            
            correct = predicted_output == query_parsed.output_stone
            result = {
                "episode_id": episode_id,
                "query": query_example, 
                "expected_output": query_parsed.output_stone, 
                "model_response": model_response_text, 
                "predicted_output": predicted_output, 
                "correct": correct, 
                "prompt": prompt_text
            }
            
            # Store data for W&B table
            if self.enable_wandb:
                self.wandb_table_data.append({
                    "episode_id": episode_id,
                    "input_prompt": prompt_text,
                    "full_model_output": model_response_text,
                    "extracted_output": predicted_output,
                    "expected_output": query_parsed.output_stone,
                    "correct": correct,
                    "model_name": self.full_model_name,
                    "api_type": "chat" if self.use_chat_api else "completions"
                })
            
            return result
            
        except Exception as e:
            error_result = {
                "episode_id": episode_id,
                "query": query_example, 
                "expected_output": None, 
                "model_response": None, 
                "predicted_output": None, 
                "correct": False, 
                "error": str(e)
            }
            
            # Store error data for W&B table
            if self.enable_wandb:
                self.wandb_table_data.append({
                    "episode_id": episode_id,
                    "input_prompt": f"Error during prompt creation for query: {query_example}",
                    "full_model_output": str(e),
                    "extracted_output": "ERROR",
                    "expected_output": "N/A",
                    "correct": False,
                    "model_name": self.full_model_name,
                    "api_type": "chat" if self.use_chat_api else "completions"
                })
            
            return error_result

    def evaluate_batch_examples(self, support_examples: List[str], query_examples: List[str], episode_id: str) -> List[Dict[str, Any]]:
        """Evaluate multiple query examples in batch with stone ID support."""
        try:
            # Parse all queries first
            parsed_queries = []
            for query_example in query_examples:
                try:
                    parsed_queries.append((self.parse_chemistry_example(query_example), query_example))
                except ValueError as e:
                    print(f"Warning: Skipping malformed query: {e}")
                    continue
            
            if not parsed_queries:
                return []
            
            if self.use_chat_api:
                # Create chat prompt template once - pass all query examples for complete stone ID mapping
                all_query_texts = [q[1] for q in parsed_queries]  # Extract raw query texts
                chat_prompt = self.create_chat_prompt_template(support_examples, all_query_texts)
                
                # Prepare all prompts for batch processing
                formatted_prompts = []
                for parsed_query, _ in parsed_queries:
                    # Format the query input based on stone ID mode
                    if self.use_stone_ids:
                        # Convert input stone to ID for prompt
                        input_stone_display = self.stone_to_id_mapping.get(parsed_query.input_stone, parsed_query.input_stone)
                    else:
                        input_stone_display = parsed_query.input_stone
                    
                    prompt_messages = chat_prompt.format_messages(
                        input_stone=input_stone_display,
                        potions=" ".join(parsed_query.potions)
                    )
                    formatted_prompts.append(prompt_messages)
                
                # Batch inference using LangChain's batch method
                batch_responses = self.llm.batch(formatted_prompts)
                
                # Process batch responses
                batch_results = []
                for i, (parsed_query, original_query) in enumerate(parsed_queries):
                    try:
                        model_response = batch_responses[i]
                        if hasattr(model_response, 'content'):
                            model_response_text = model_response.content
                        else:
                            model_response_text = str(model_response)
                        
                        predicted_output = self.output_parser.parse(model_response_text)
                        prompt_text = f"Chat messages: {[msg.content for msg in formatted_prompts[i]]}"
                        
                        # Dual evaluation logic for batch processing
                        if self.use_stone_ids:
                            # Primary evaluation: Stone ID matching
                            expected_stone_id = self.stone_to_id_mapping.get(parsed_query.output_stone, None)
                            stone_id_correct = predicted_output == expected_stone_id
                            
                            # Secondary evaluation: Feature-level analysis
                            predicted_stone_state = self.id_to_stone_mapping.get(predicted_output, None)
                            feature_level_evaluation = None
                            
                            if predicted_stone_state:
                                feature_level_evaluation = self.evaluate_feature_level(
                                    parsed_query.output_stone, predicted_stone_state
                                )
                            
                            result = {
                                "episode_id": episode_id,
                                "query": original_query,
                                "expected_output": parsed_query.output_stone,
                                "expected_stone_id": expected_stone_id,
                                "model_response": model_response_text,
                                "predicted_output": predicted_output,
                                "predicted_stone_state": predicted_stone_state,
                                "correct": stone_id_correct,
                                "stone_id_correct": stone_id_correct,
                                "feature_level_evaluation": feature_level_evaluation,
                                "prompt": prompt_text
                            }
                        else:
                            # Standard evaluation: Full stone state matching
                            correct = predicted_output == parsed_query.output_stone
                            result = {
                                "episode_id": episode_id,
                                "query": original_query,
                                "expected_output": parsed_query.output_stone,
                                "model_response": model_response_text,
                                "predicted_output": predicted_output,
                                "correct": correct,
                                "prompt": prompt_text
                            }
                        
                        batch_results.append(result)
                        
                        # Store data for W&B table
                        if self.enable_wandb:
                            wandb_data = {
                                "episode_id": episode_id,
                                "input_prompt": prompt_text,
                                "full_model_output": model_response_text,
                                "extracted_output": predicted_output,
                                "expected_output": parsed_query.output_stone,
                                "correct": result["correct"],
                                "model_name": self.full_model_name,
                                "api_type": "chat" if self.use_chat_api else "completions",
                                "use_stone_ids": self.use_stone_ids
                            }
                            
                            if self.use_stone_ids:
                                wandb_data.update({
                                    "expected_stone_id": result.get("expected_stone_id"),
                                    "predicted_stone_state": result.get("predicted_stone_state"),
                                    "stone_id_correct": result.get("stone_id_correct"),
                                    "feature_evaluation": result.get("feature_level_evaluation")
                                })
                                
                            self.wandb_table_data.append(wandb_data)
                    
                    except Exception as e:
                        # Handle individual errors within batch
                        error_result = {
                            "episode_id": episode_id,
                            "query": original_query,
                            "expected_output": parsed_query.output_stone,
                            "model_response": str(e),
                            "predicted_output": None,
                            "correct": False,
                            "error": str(e)
                        }
                        batch_results.append(error_result)
                        
                        if self.enable_wandb:
                            self.wandb_table_data.append({
                                "episode_id": episode_id,
                                "input_prompt": f"Error in batch processing for query: {original_query}",
                                "full_model_output": str(e),
                                "extracted_output": "ERROR",
                                "expected_output": parsed_query.output_stone,
                                "correct": False,
                                "model_name": self.full_model_name,
                                "api_type": "chat" if self.use_chat_api else "completions"
                            })
                
                return batch_results
                
            else:
                # Fallback to individual processing for non-chat API
                individual_results = []
                for parsed_query, original_query in parsed_queries:
                    result = self.evaluate_single_example(support_examples, original_query, episode_id)
                    individual_results.append(result)
                return individual_results
                
        except Exception as e:
            print(f"❌ Batch evaluation failed: {e}")
            # Fallback to individual processing
            individual_results = []
            for _, original_query in parsed_queries:
                result = self.evaluate_single_example(support_examples, original_query, episode_id)
                individual_results.append(result)
            return individual_results

    def evaluate_dataset(self, dataset: Dict[str, List[str]]) -> EvaluationResults:
        """Evaluate the entire dataset with optional batch processing."""
        detailed_results, correct_count, failed_count = [], 0, 0
        episodes = dataset.get("episodes", [])
        
        print(f"🧪 Starting evaluation of {len(episodes)} episodes")
        if hasattr(self, 'enable_batch_inference') and self.enable_batch_inference:
            print(f"🚀 Batch inference enabled with batch size: {self.batch_size}")
        
        for episode_id, episode in tqdm(episodes.items(), desc="Evaluating episodes"):
            support_examples, query_examples = episode["support"], episode["query"]
            
            if hasattr(self, 'enable_batch_inference') and self.enable_batch_inference and len(query_examples) > 1:
                # Use batch processing for multiple queries
                print(f"⚗️  Processing {len(query_examples)} queries in batches for episode {episode_id}")
                
                # Split queries into batches
                batches = [query_examples[i:i + self.batch_size] 
                          for i in range(0, len(query_examples), self.batch_size)]
                
                episode_results = []
                for batch_idx, batch_queries in enumerate(batches):
                    print(f"   📦 Processing batch {batch_idx + 1}/{len(batches)} ({len(batch_queries)} queries)")
                    batch_results = self.evaluate_batch_examples(support_examples, batch_queries, episode_id)
                    episode_results.extend(batch_results)
            else:
                # Use individual processing (original method)
                episode_results = []
                for i, query in enumerate(query_examples):
                    print(f"⚗️  Processing query {i+1}/{len(query_examples)} in episode {episode_id}")
                    result = self.evaluate_single_example(support_examples, query, episode_id)
                    episode_results.append(result)
            
            # Count results
            episode_correct_count = sum(1 for r in episode_results if r.get("correct", False))
            episode_failed_count = sum(1 for r in episode_results if "error" in r and r["error"])
            
            # Print episode summary
            for result in episode_results:
                if "error" in result and result["error"]:
                    print(f"  ❌ Error: {result['error']}")
                elif result["correct"]:
                    print(f"  ✅ Correct, Predicted: {result['predicted_output']}")
                else:
                    print(f"  ❌ Incorrect - Expected: {result['expected_output']}, Got: {result['predicted_output']}")
            
            detailed_results.extend(episode_results)
            correct_count += episode_correct_count
            failed_count += episode_failed_count
            
            # Log episode-level metrics to W&B
            # if self.enable_wandb:
            #     episode_accuracy = episode_correct_count / len(query_examples) if query_examples else 0.0
            #     wandb.log({
            #         f"episode_{episode_id}_accuracy": episode_accuracy,
            #         f"episode_{episode_id}_correct": episode_correct_count,
            #         f"episode_{episode_id}_total": len(query_examples),
            #         "step": int(episode_id) if episode_id.isdigit() else len(detailed_results)
            #     })
        
        accuracy = correct_count / len(detailed_results) if detailed_results else 0.0
        results = EvaluationResults(len(detailed_results), correct_count, accuracy, failed_count, detailed_results)
        
        # Log final results and create W&B table
        if self.enable_wandb:
            self._log_to_wandb(results)
        
        return results

    def _log_to_wandb(self, results: EvaluationResults):
        """Log final results and create W&B table."""
        try:
            # Log summary metrics
            wandb.log({
                "final_accuracy": results.accuracy,
                "total_queries": results.total_queries,
                "correct_predictions": results.correct_predictions,
                "failed_predictions": results.failed_predictions,
            })
            
            # Create and log the main results table
            table = wandb.Table(
                columns=[
                    "episode_id", 
                    "input_prompt", 
                    "full_model_output", 
                    "extracted_output",
                    "expected_output",
                    "correct", 
                    "model_name"
                ],
                data=[[
                    row["episode_id"],
                    row["input_prompt"], 
                    row["full_model_output"],
                    row["extracted_output"],
                    row["expected_output"],
                    row["correct"],
                    row["model_name"]
                ] for row in self.wandb_table_data]
            )
            
            wandb.log({"evaluation_results": table})
            
            # Create accuracy distribution by episode
            episode_accuracies = {}
            for row in self.wandb_table_data:
                episode_id = row["episode_id"]
                if episode_id not in episode_accuracies:
                    episode_accuracies[episode_id] = {"correct": 0, "total": 0}
                episode_accuracies[episode_id]["total"] += 1
                if row["correct"]:
                    episode_accuracies[episode_id]["correct"] += 1
            
            # Log episode accuracy distribution
            episode_acc_data = []
            for episode_id, stats in episode_accuracies.items():
                acc = stats["correct"] / stats["total"] if stats["total"] > 0 else 0.0
                episode_acc_data.append([episode_id, acc, stats["correct"], stats["total"]])
            
            episode_table = wandb.Table(
                columns=["episode_id", "accuracy", "correct", "total"],
                data=episode_acc_data
            )
            
            wandb.log({"episode_accuracies": episode_table})
            
            print(f"📊 Logged {len(self.wandb_table_data)} examples to W&B")
            print(f"🔗 View results at: {wandb.run.url}")
            
        except Exception as e:
            print(f"⚠️ Warning: Could not log to W&B: {e}")

    def save_results(self, results: EvaluationResults, output_file: str):
        """Save evaluation results to a file."""
        output_data = {
            "summary": results.__dict__, 
            "detailed_results": results.detailed_results, 
            "metadata": {
                "llm_type": "langchain_vllm_client", 
                "model_name": self.full_model_name,
                "experiment_name": self.experiment_name
            }
        }
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"💾 Results saved to {output_file}")

    def finish_wandb(self):
        """Properly finish W&B run."""
        if self.enable_wandb:
            wandb.finish()


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
        "episodes": {
            "0": {
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
        }
    }


def main():
    if not LANGCHAIN_AVAILABLE:
        sys.exit(1)

    parser = argparse.ArgumentParser(description="Chemistry evaluation client for a remote vLLM server.")
    parser.add_argument("--data", type=str, help="Path to JSON file with support/query data", 
                        default='/home/rsaha/projects/dm_alchemy/src/data/complete_graph_generated_data_enhanced_qnodes_in_snodes/decompositional_chemistry_samples_167424_80_unique_stones_val_shop_2_qhop_1.json')
    parser.add_argument("--data_split_seed", type=int, default=0, help="Data split seed")
    parser.add_argument("--output", type=str, default="langchain_vllm_client_results.json", help="Output file for results")
    parser.add_argument("--vllm_url", type=str, default="http://localhost:8000/v1", help="vLLM server OpenAI-compatible API URL")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-8B-Instruct", help="Model name being served")
    parser.add_argument("--use_sample_data", action="store_true", help="Use built-in sample data for testing", default=False)
    parser.add_argument("--max_new_tokens", type=int, default=1000, help="Max new tokens to generate")
    
    # Chat API arguments
    parser.add_argument("--use_chat_api", action="store_true", help="Use ChatOpenAI (/v1/chat/completions) instead of VLLMOpenAI (/v1/completions)", default=True)
    parser.add_argument("--provider", type=str, default="vllm", choices=["vllm", "fireworks"], help="API provider: vllm or fireworks")
    
    # Batch inference arguments
    parser.add_argument("--enable_batch_inference", action="store_true", help="Enable batch inference for faster processing", default=False)
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for batch inference")
    
    # W&B arguments
    parser.add_argument("--enable_wandb", action="store_true", help="Enable W&B monitoring", default=False)
    
    # NEW: Stone ID and reasoning arguments
    parser.add_argument("--use_stone_ids", action="store_true", help="Use stone IDs (S1, S2, ...) instead of full stone states", default=False)
    parser.add_argument("--enable_reasoning", action="store_true", help="Allow model to show reasoning steps", default=True)
    parser.add_argument("--disable_reasoning", action="store_true", help="Prohibit model from showing reasoning steps", default=False)
    
    args = parser.parse_args()
    
    # Handle reasoning flag logic
    if args.disable_reasoning:
        args.enable_reasoning = False

    # Add the 'data_split_seed' to the data filename (as _seed_X) before the .json extension
    if args.data and args.data.endswith('.json'):
        args.data = args.data.replace('.json', f'_seed_{args.data_split_seed}.json')
        print("Using data file:", args.data)
    # Create experiment_name
    if args.enable_wandb:
        file_path = ''.join(args.data.split('/')[-2:]).replace('.json', '') if args.data else "sample_data"
        experiment_name = f"{args.model_name.replace('/', '_')}_{file_path}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    else:
        experiment_name = f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Handle API configuration
    api_key = "EMPTY"
    if args.provider == "fireworks":
        args.vllm_url = 'https://api.fireworks.ai/inference/v1'
        try:
            with open('src/prompting_experiments/keys.json', 'r') as f:
                keys = json.load(f)
                api_key = keys.get('fireworks_ai',"EMPTY")
                assert api_key != "EMPTY", "FireworksAI API key not found in keys.json"
                os.environ['FIREWORKS_API_KEY'] = api_key
        except FileNotFoundError:
            assert False, "keys.json file not found. Please create keys.json with your API keys."
        args.provider = "fireworks"
    
    # Print configuration info
    api_type = "chat" if args.use_chat_api else "completions"
    print(f"🔧 Configuration: {args.provider} provider, {api_type} API, model: {args.model_name}")

    # Get the shop, qhop, and seed from the filename if possible
    shop_length, qhop_length, data_split_seed = None, None, None
    if args.data:
        shop_match = re.search(r'shop_(\d+)', args.data)
        qhop_match = re.search(r'qhop_(\d+)', args.data)
        seed_match = re.search(r'seed_(\d+)', args.data)
        if shop_match:
            shop_length = int(shop_match.group(1))
        if qhop_match:
            qhop_length = int(qhop_match.group(1))
        if seed_match:
            data_split_seed = int(seed_match.group(1))
        print(f"Detected shop_length: {shop_length}, qhop_length: {qhop_length}, data_split_seed: {data_split_seed}")
    
    evaluator = ChemistryPromptEvaluator(
        vllm_url=args.vllm_url,
        model_name=args.model_name,
        max_tokens=args.max_new_tokens,
        enable_wandb=args.enable_wandb,
        experiment_name=experiment_name,
        data_file_name=args.data,
        api_key=api_key,
        use_chat_api=args.use_chat_api,
        provider=args.provider,
        shop_length=shop_length,
        qhop_length=qhop_length,
        data_split_seed=data_split_seed,
        batch_size=args.batch_size,  # NEW
        enable_batch_inference=args.enable_batch_inference,  # NEW
        enable_reasoning=args.enable_reasoning,  # NEW
        use_stone_ids=args.use_stone_ids         # NEW
    )

    try:
        if args.use_sample_data:
            data = create_sample_data()
        elif args.data:
            data = load_data_from_file(args.data, use_all_episodes=True)
        else:
            print("❌ Error: Must provide --data or --use-sample-data")
            sys.exit(1)

        results = evaluator.evaluate_dataset(data)
        print(f"\n🏆 EVALUATION COMPLETE: Accuracy: {results.accuracy:.3f}")
        
        # Create output directory and save results
        if not os.path.exists('prompting_results'):
            os.makedirs('prompting_results')
        args.output = f'prompting_results/{evaluator.experiment_name}_results.json'
        
        evaluator.save_results(results, args.output)
        
    finally:
        # Always properly finish W&B run
        evaluator.finish_wandb()


if __name__ == "__main__":
    main()