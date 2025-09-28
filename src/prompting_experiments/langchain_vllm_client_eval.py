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
import random
import numpy as np
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
    use_stone_ids: bool = False
    def __init__(self, use_stone_ids: bool = False, **kwargs):
        super().__init__(use_stone_ids=use_stone_ids, **kwargs)
    
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
                 batch_size: int = 1,
                 enable_batch_inference: bool = False,
                 enable_reasoning: bool = True,  # NEW: Control reasoning generation
                 use_stone_ids: bool = False,   # NEW: Use stone IDs instead of full features
                 batch_mode: str = "global"):  # NEW: Add batch_mode parameter
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
        self.example_template = PromptTemplate(
            input_variables=["input_stone", "potions", "output_stone"],
            template="{input_stone} {potions} -> {output_stone}"
        )
        self.batch_size = batch_size
        self.enable_batch_inference = enable_batch_inference
        self.enable_reasoning = enable_reasoning  # NEW
        self.use_stone_ids = use_stone_ids  # NEW
        self.batch_mode = batch_mode  # NEW: Store it

        # NEW: Stone ID configuration
        self.enable_reasoning = enable_reasoning
        self.use_stone_ids = use_stone_ids
        
        # Will be populated when processing examples
        self.stone_to_id_mapping = {}
        self.id_to_stone_mapping = {}
        
        # Update output parser with new configuration
        self.output_parser = ChemistryOutputParser(use_stone_ids=use_stone_ids)
        
        # NEW: Add reasoning control to vLLM if possible
        if not enable_reasoning and provider == "vllm":
            try:
                # Try to set up reasoning suppression via API parameters
                # This might require additional configuration depending on vLLM version
                pass
            except:
                print("⚠️ Could not configure reasoning suppression via API, using prompt-based approach")
    
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
                escaped_input_stone = parsed.input_stone.replace("{", "{").replace("}", "}")
                escaped_output_stone = parsed.output_stone.replace("{", "{").replace("}", "}")

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
    def create_chat_prompt_template(self, support_examples: List[str], query_examples: List[str] = None,
                                    episode_id=None) -> ChatPromptTemplate:
        """
        Create a chat-based prompt template with optional stone ID mapping and reasoning control.
        """
        # System message
        if episode_id is None or episode_id not in self.stone_to_id_mapping:
            raise ValueError(f"Mappings not found for episode {episode_id}")
        if self.use_stone_ids:
            system_message = (
                "You are an expert in understanding latent structures in chemistry transformations. "
                "Your task is to analyze support examples showing how potions transform input stone states to output stone states, "
                "then predict the output stone ID for new examples."
            )
        else:
            system_message = (
                "You are an expert in understanding latent structures in chemistry transformations. "
                "Your task is to analyze support examples showing how potions transform input stone states to output stone states, "
                "then predict the output for new examples."
            )

        # Parse support examples
        example_prompt = PromptTemplate(
            input_variables=["input_stone", "potions", "output_stone"],
            template="{input_stone} {potions} -> {output_stone}"
        )

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
                continue

        # Stone ID mapping (optional)
        if self.use_stone_ids:
            stone_to_id_mapping = self.stone_to_id_mapping[episode_id]
            id_to_stone_mapping = self.id_to_stone_mapping[episode_id]
            mapping_display = "Stone ID Mapping:\n"
            for stone_state, stone_id in sorted(stone_to_id_mapping.items(), key=lambda x: x[1]):
                mapping_display += f"{stone_id}: {stone_state}\n"
            mapping_display += "\n"
        else:
            mapping_display = ""

        # Example selection
        example_selector = LengthBasedExampleSelector(
            examples=all_parsed_examples,
            example_prompt=example_prompt,
            max_length=10000
        )
        selected_examples = example_selector.select_examples({})

        # Build examples string (with or without IDs)
        # if self.use_stone_ids:
        #     examples_lines = []
        #     for ex in selected_examples:
        #         input_id = self.stone_to_id_mapping[episode_id].get(ex['input_stone'], ex['input_stone'])
        #         output_id = self.stone_to_id_mapping[episode_id].get(ex['output_stone'], ex['output_stone'])
        #         examples_lines.append(f"{input_id} {ex['potions']} -> {output_id}")
        #     examples_string = "\n".join(examples_lines)
        # else:
        random.shuffle(selected_examples)
        examples_string = "\n".join([ex['raw_text'] for ex in selected_examples])

        # Instructions (escape {stone_state})
        if self.use_stone_ids:
            if self.enable_reasoning:
                instruction = (
                    "Analyze the patterns in the support examples to understand the transformation rules, "
                    "then apply this knowledge to predict the output stone ID for the query example.\n\n"
                    "You may show your reasoning, but must end your response with: "
                    "'Therefore, the output stone ID is: [SX]' where X is the stone number."
                )
            else:
                instruction = (
                    "Analyze the patterns in the support examples to understand the transformation rules, "
                    "then predict the output stone ID for the query example.\n\n"
                    "Do not show reasoning steps. Respond only with: "
                    "'Therefore, the output stone ID is: [SX]' where X is the stone number."
                )
            query_format = "Now, predict the output stone ID for the following example:\n{input_stone} {potions} ->"
        else:
            if self.enable_reasoning:
                instruction = (
                    "Analyze the patterns in the support examples to understand the transformation rules, "
                    "then apply this knowledge to predict the output stone state for the query example.\n\n"
                    "You may show your reasoning, but must end your response with: "
                    "'Therefore, the output stone state is: stone_state', where stone_state is the full stone description including the curly braces."
                )
            else:
                instruction = (
                    "Analyze the patterns in the support examples to understand the transformation rules, "
                    "then predict the output stone state for the query example.\n\n"
                    "Do not show reasoning steps. Respond only with: "
                    "'Therefore, the output stone state is: stone_state', where stone_state is the full stone description including the curly braces."
                )
            query_format = "Now, predict the output for the following example:\n{input_stone} {potions} ->"

        # Escape braces in dynamic inserts (examples & mapping display)
        def escape_braces(s: str) -> str:
            return s.replace("{", "{").replace("}", "}")
        examples_string_escaped = escape_braces(examples_string)
        mapping_display_escaped = escape_braces(mapping_display)

        # Template with placeholders for mapping_display & examples
        human_template = (
            "Each stone state has four features: color, size, roundness, and reward value. "
            "Potions are represented by color names (e.g., RED, BLUE, GREEN, YELLOW, ORANGE, PINK) "
            "and can be applied individually or in combination. The application of the potion(s) "
            "modifies the features of the input stone to produce the output stone.\n\n"
            "{mapping_display}"
            f"{instruction}\n\n"
            "Here are some examples of transformations:\n"
            "---START EXAMPLES---\n"
            "{examples}\n"
            "---END EXAMPLES---\n\n"
            f"{query_format}"
        )

        human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

        chat_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=system_message),
            human_message_prompt
        ])

        # Inject escaped dynamic content
        return chat_prompt.partial(
            examples=examples_string_escaped,
            mapping_display=mapping_display_escaped
        )

    def create_stone_id_mapping(self, support_examples: List[str], query_examples: List[str] = None,
                                episode_id=None) -> tuple:
        """Create mapping from stone states to IDs using S1, S2, ... format."""
        if episode_id is None:
            raise ValueError("episode_id must be provided for episode-specific mappings")
        unique_stones = set()
        
        # Extract all unique stone states from support examples
        for example_text in support_examples:
            try:
                parsed = self.parse_chemistry_example(example_text)
                unique_stones.add(parsed.input_stone)
                unique_stones.add(parsed.output_stone)
            except ValueError:
                assert False, f"Support example malformed: {example_text}"
        
        # Also include query examples if provided
        if query_examples:
            for query_text in query_examples:
                try:
                    parsed = self.parse_chemistry_example(query_text)
                    unique_stones.add(parsed.input_stone)
                    unique_stones.add(parsed.output_stone)
                except ValueError:
                    assert False, f"Query example malformed: {query_text}"
        
        # Create bidirectional mapping with S1, S2, ... format
        stone_to_id = {}
        id_to_stone = {}
        
        for i, stone_state in enumerate(sorted(unique_stones)):
            stone_id = f"S{i+1}"
            stone_to_id[stone_state] = stone_id
            id_to_stone[stone_id] = stone_state
        self.stone_to_id_mapping[episode_id] = stone_to_id
        self.id_to_stone_mapping[episode_id] = id_to_stone
        
        print(f"📋 Created stone ID mapping with {len(unique_stones)} unique stones")
        return stone_to_id, id_to_stone

    def evaluate_single_example(self, support_examples: List[str], query_example: str, episode_id: str) -> Dict[str, Any]:
        """Evaluate a single query example with dual evaluation for stone IDs."""
        try:
            query_parsed = self.parse_chemistry_example(query_example)
            
            if self.use_chat_api:
                # Use chat-based prompting - pass query examples for complete mapping
                query_examples_for_mapping = [query_example]
                chat_prompt = self.create_chat_prompt_template(support_examples, query_examples_for_mapping, episode_id)
                
                # Format the query input based on stone ID mode
                if self.use_stone_ids:
                    # Convert input stone to ID for prompt
                    input_stone_display = self.stone_to_id_mapping[episode_id].get(query_parsed.input_stone, query_parsed.input_stone)
                else:
                    input_stone_display = query_parsed.input_stone
                    
                formatted_prompt = chat_prompt.format_messages(
                    input_stone=input_stone_display, 
                    potions=" ".join(query_parsed.potions)
                )
                
                model_response = self.llm.invoke(formatted_prompt)
                if hasattr(model_response, 'content'):
                    model_response_text = model_response.content
                else:
                    model_response_text = str(model_response)
                prompt_text = f"Chat messages: {[msg.content for msg in formatted_prompt]}"
            else:
                few_shot_prompt = self.create_few_shot_prompt(support_examples)
                escaped_query_input = query_parsed.input_stone.replace("{", "{{").replace("}", "}}")
                prompt_text = few_shot_prompt.format(input_stone=escaped_query_input, potions=" ".join(query_parsed.potions))
                model_response = self.llm.invoke(prompt_text)
                model_response_text = str(model_response)
            
            predicted_output = self.output_parser.parse(model_response_text)
            
            # Dual evaluation logic
            if self.use_stone_ids:
                # Primary evaluation: Stone ID matching
                expected_stone_id = self.stone_to_id_mapping[episode_id].get(query_parsed.output_stone, None)
                stone_id_correct = predicted_output == expected_stone_id
                
                # Secondary evaluation: Feature-level analysis
                predicted_stone_state = self.id_to_stone_mapping[episode_id].get(predicted_output, None)
                feature_level_evaluation = None
                if predicted_stone_state is None:
                    print(f"⚠️ Warning: Predicted stone ID '{predicted_output}' not found in mapping")
                
                if predicted_stone_state:
                    feature_level_evaluation = self.evaluate_feature_level(
                        query_parsed.output_stone, predicted_stone_state
                    )
                
                result = {
                    "episode_id": episode_id,
                    "query": query_example,
                    "expected_output": query_parsed.output_stone,
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
                wandb_data = {
                    "episode_id": episode_id,
                    "input_prompt": prompt_text,
                    "full_model_output": model_response_text,
                    "extracted_output": predicted_output,
                    "expected_output": query_parsed.output_stone,
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
            
            return result
            
        except Exception as e:
            # Error handling remains the same
            error_result = {
                "episode_id": episode_id,
                "query": query_example,
                "expected_output": None,
                "model_response": None,
                "predicted_output": None,
                "correct": False,
                "error": str(e)
            }
            
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
            chat_prompt = self.create_chat_prompt_template(support_examples, all_query_texts, episode_id)
            
            # Prepare all prompts for batch processing
            formatted_prompts = []
            for parsed_query, _ in parsed_queries:
                # Format the query input based on stone ID mode
                # if self.use_stone_ids:
                #     # Convert input stone to ID for prompt
                #     input_stone_display = self.stone_to_id_mapping[episode_id].get(parsed_query.input_stone, parsed_query.input_stone)
                # else:
                input_stone_display = parsed_query.input_stone
                
                prompt_messages = chat_prompt.format_messages(
                    input_stone=input_stone_display,
                    potions=" ".join(parsed_query.potions)
                )
                formatted_prompts.append(prompt_messages)
            
            # Batch inference
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
                        expected_stone_id = self.stone_to_id_mapping[episode_id].get(parsed_query.output_stone, None)
                        stone_id_correct = predicted_output == expected_stone_id
                        
                        # Secondary evaluation: Feature-level analysis
                        predicted_stone_state = self.id_to_stone_mapping[episode_id].get(predicted_output, None)
                        feature_level_evaluation = None
                        if predicted_stone_state is None:
                            print(f"⚠️ Warning: Predicted stone ID '{predicted_output}' not found in mapping")
                        
                        
                        if predicted_stone_state:
                            feature_level_evaluation = self.evaluate_feature_level(
                                parsed_query.output_stone, predicted_stone_state
                            )
                        print("Prompt:")
                        print(prompt_text)
                        print("\nModel Response:")
                        print(model_response_text)
                        print("\nPredicted Output:", predicted_output)
                        print("Expected Output:", parsed_query.output_stone)
                        print("Stone ID Correct:", stone_id_correct)
                        print("Feature Evaluation:", feature_level_evaluation)
                        
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
                
        # except Exception as e:
            # print(f"❌ Batch evaluation failed: {e}")
            # # Fallback to individual processing
            # individual_results = []
            # for _, original_query in parsed_queries:
            #     result = self.evaluate_single_example(support_examples, original_query, episode_id)
            #     individual_results.append(result)
            # return individual_results

    def evaluate_dataset(self, dataset: Dict[str, List[str]]) -> EvaluationResults:
        """Evaluate the entire dataset with batch processing (global or per-episode)."""
        detailed_results, correct_count, failed_count = [], 0, 0
        episodes = dataset.get("episodes", [])
        
        print(f"🧪 Starting evaluation of {len(episodes)} episodes with batch_mode='{self.batch_mode}'")
        if hasattr(self, 'enable_batch_inference') and self.enable_batch_inference:
            print(f"🚀 Batch inference enabled with batch size: {self.batch_size}")
        
        if self.batch_mode == "global":
            # NEW: Global batching mode - collect all queries across episodes
            all_queries = []
            for episode_id, episode in tqdm(episodes.items(), desc="Processing episodes for global batching"):
                support_examples = episode["support"]
                query_examples = episode["query"]
                # Create a mapping here for each episode separately.
                self.create_stone_id_mapping(support_examples, query_examples, episode_id)
                for query in query_examples:
                    all_queries.append((episode_id, support_examples, query))
            
            total_queries = len(all_queries)
            print(f"📊 Collected {total_queries} total queries across all episodes for global batching")
            
            if not (hasattr(self, 'enable_batch_inference') and self.enable_batch_inference) or total_queries <= 1:
                print("⚠️ Batching disabled or not needed; processing individually")
                for episode_id, support_examples, query in all_queries:
                    result = self.evaluate_single_example(support_examples, query, episode_id)
                    detailed_results.append(result)
            else:
                # Generate prompts for all queries (each with its episode's support)
                formatted_prompts = []
                query_metadata = []  # Track (episode_id, support_examples, query, parsed_query)
                for episode_id, support_examples, query in all_queries:
                    # try:
                    query_parsed = self.parse_chemistry_example(query)
                    if self.use_chat_api:
                        query_examples_for_mapping = [query]
                        chat_prompt = self.create_chat_prompt_template(support_examples, query_examples_for_mapping, episode_id)
                        
                        input_stone_display = (
                            self.stone_to_id_mapping[episode_id].get(query_parsed.input_stone, query_parsed.input_stone)
                            if self.use_stone_ids else query_parsed.input_stone
                        )
                        
                        prompt_messages = chat_prompt.format_messages(input_stone=input_stone_display, potions=" ".join(query_parsed.potions))
                        formatted_prompts.append(prompt_messages)
                        query_metadata.append((episode_id, support_examples, query, query_parsed))
                    else:
                        # For non-chat API, fall back to individual
                        result = self.evaluate_single_example(support_examples, query, episode_id)
                        detailed_results.append(result)
                        continue
                    # except Exception as e:
                    #     print("Exception during prompt creation:", e)
                    #     error_result = {
                    #         "episode_id": episode_id,
                    #         "query": query,
                    #         "expected_output": None,
                    #         "model_response": str(e),
                    #         "predicted_output": None,
                    #         "correct": False,
                    #         "error": str(e)
                    #     }
                    #     detailed_results.append(error_result)
                    #     if self.enable_wandb:
                    #         self.wandb_table_data.append({
                    #             "episode_id": episode_id,
                    #             "input_prompt": f"Error during prompt creation for query: {query}",
                    #             "full_model_output": str(e),
                    #             "extracted_output": "ERROR",
                    #             "expected_output": "N/A",
                    #             "correct": False,
                    #             "model_name": self.full_model_name,
                    #             "api_type": "chat" if self.use_chat_api else "completions"
                    #         })
                    #     continue
                
                # Split into global batches and process
                batches = [formatted_prompts[i:i + self.batch_size] for i in range(0, len(formatted_prompts), self.batch_size)]
                batch_metadata = [query_metadata[i:i + self.batch_size] for i in range(0, len(query_metadata), self.batch_size)]
                
                for batch_idx, (batch_prompts, batch_meta) in enumerate(tqdm(zip(batches, batch_metadata), 
                                                                              desc="Processing global batches",
                                                                              total=len(batches),
                                                                              unit="batch")):
                    print(f"   📦 Processing global batch {batch_idx + 1}/{len(batches)} ({len(batch_prompts)} queries)")
                    try:
                        batch_responses = self.llm.batch(batch_prompts)
                        
                        for i, (response, (episode_id, support_examples, query, query_parsed)) in enumerate(zip(batch_responses, batch_meta)):
                            try:
                                if hasattr(response, 'content'):
                                    model_response_text = response.content
                                else:
                                    model_response_text = str(response)
                                
                                predicted_output = self.output_parser.parse(model_response_text)
                                prompt_text = f"Chat messages: {[msg.content for msg in batch_prompts[i]]}"
                                
                                # Dual evaluation logic (same as before)
                                if self.use_stone_ids:
                                    expected_stone_id = self.stone_to_id_mapping[episode_id].get(query_parsed.output_stone, None)
                                    if expected_stone_id is None:
                                        import pdb; pdb.set_trace()
                                    stone_id_correct = predicted_output == expected_stone_id
                                    # This is a way to check if the predicted ID exists in the mapping
                                    # and see if the corresponding full feature stone state matches to the ground truth mapping.
                                    predicted_stone_state = self.id_to_stone_mapping[episode_id].get(predicted_output, None)
                                    feature_level_evaluation = (
                                        self.evaluate_feature_level(query_parsed.output_stone, predicted_stone_state)
                                        if predicted_stone_state else None
                                    )
                                    result = {
                                        "episode_id": episode_id,
                                        "query": query,
                                        "expected_output": query_parsed.output_stone,
                                        "expected_stone_id": expected_stone_id,
                                        "model_response": model_response_text,
                                        "predicted_output": predicted_output,
                                        "predicted_stone_state": predicted_stone_state,
                                        "correct": stone_id_correct,
                                        "stone_id_correct": stone_id_correct,
                                        "feature_level_evaluation": feature_level_evaluation,
                                        "prompt": prompt_text
                                    }
                                    
                                    # Print the model prompt and the model response.
                                    print(f"Prompt:\n{prompt_text}\nModel Response:\n{model_response_text}\nPredicted Output: {predicted_output}\nExpected Output: {query_parsed.output_stone}\nStone ID Correct: {stone_id_correct}\nFeature Evaluation: {feature_level_evaluation}\n---")
                                    
                                    # Print results for stone ID mode
                                    if stone_id_correct:
                                        print(f"  ✅ Episode {episode_id}: Correct - Predicted ID: {predicted_output}, Expected ID: {expected_stone_id}")
                                    else:
                                        print(f"  ❌ Episode {episode_id}: Incorrect - Predicted ID: {predicted_output}, Expected ID: {expected_stone_id}")
                                        print(f"      Expected stone: {query_parsed.output_stone}")
                                        if predicted_stone_state:
                                            print(f"      Predicted stone: {predicted_stone_state}")
                                else:
                                    correct = predicted_output == query_parsed.output_stone
                                    result = {
                                        "episode_id": episode_id,
                                        "query": query,
                                        "expected_output": query_parsed.output_stone,
                                        "model_response": model_response_text,
                                        "predicted_output": predicted_output,
                                        "correct": correct,
                                        "prompt": prompt_text
                                    }
                                    # Print results for full stone state mode
                                    if correct:
                                        print(f"  ✅ Episode {episode_id}: Correct - Predicted: {predicted_output}")
                                    else:
                                        print(f"  ❌ Episode {episode_id}: Incorrect - Expected: {query_parsed.output_stone}")
                                        print(f"      Predicted: {predicted_output}")
                                
                                
                                detailed_results.append(result)
                                
                                # W&B logging
                                if self.enable_wandb:
                                    wandb_data = {
                                        "episode_id": episode_id,
                                        "input_prompt": prompt_text,
                                        "full_model_output": model_response_text,
                                        "extracted_output": predicted_output,
                                        "expected_output": query_parsed.output_stone,
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
                                print(f"❌ Error processing query in batch: {e}")
                                error_result = {
                                    "episode_id": episode_id,
                                    "query": query,
                                    "expected_output": query_parsed.output_stone,
                                    "model_response": str(e),
                                    "predicted_output": None,
                                    "correct": False,
                                    "error": str(e)
                                }
                                detailed_results.append(error_result)
                                if self.enable_wandb:
                                    self.wandb_table_data.append({
                                        "episode_id": episode_id,
                                        "input_prompt": f"Error in global batch processing for query: {query}",
                                        "full_model_output": str(e),
                                        "extracted_output": "ERROR",
                                        "expected_output": query_parsed.output_stone,
                                        "correct": False,
                                        "model_name": self.full_model_name,
                                        "api_type": "chat" if self.use_chat_api else "completions"
                                    })
                    
                    except Exception as e:
                        print(f"❌ Global batch {batch_idx + 1} failed: {e}. Falling back to per-episode processing for this batch.")
                        # Fallback: Process each query in the batch individually
                        for episode_id, support_examples, query in batch_meta:
                            result = self.evaluate_single_example(support_examples, query, episode_id)
                            detailed_results.append(result)
            correct_count = sum(1 for r in detailed_results if r.get("correct", False))
            failed_count = sum(1 for r in detailed_results if "error" in r and r["error"])
        
        else:  # batch_mode == "per_episode" (original logic)
            # ORIGINAL: Per-episode batching
            for episode_id, episode in tqdm(episodes.items(), desc="Evaluating episodes"):
                support_examples, query_examples = episode["support"], episode["query"]
                
                if hasattr(self, 'enable_batch_inference') and self.enable_batch_inference and len(query_examples) > 1:
                    print(f"⚗️  Processing {len(query_examples)} queries in batches for episode {episode_id}")
                    
                    batches = [query_examples[i:i + self.batch_size] 
                              for i in range(0, len(query_examples), self.batch_size)]
                    
                    episode_results = []
                    for batch_idx, batch_queries in enumerate(batches):
                        print(f"   📦 Processing batch {batch_idx + 1}/{len(batches)} ({len(batch_queries)} queries)")
                        batch_results = self.evaluate_batch_examples(support_examples, batch_queries, episode_id)
                        episode_results.extend(batch_results)
                else:
                    episode_results = []
                    for i, query in enumerate(query_examples):
                        print(f"⚗️  Processing query {i+1}/{len(query_examples)} in episode {episode_id}")
                        result = self.evaluate_single_example(support_examples, query, episode_id)
                        episode_results.append(result)
                
                # Count and log per-episode
                episode_correct_count = sum(1 for r in episode_results if r.get("correct", False))
                episode_failed_count = sum(1 for r in episode_results if "error" in r and r["error"])
                
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
        
        # Aggregate and log final results (same as before)
        accuracy = correct_count / len(detailed_results) if detailed_results else 0.0
        results = EvaluationResults(len(detailed_results), correct_count, accuracy, failed_count, detailed_results)
        
        if self.enable_wandb:
            wandb.log({"batch_mode": self.batch_mode})
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
                        default='/home/rsaha/projects/dm_alchemy/src/data/complete_graph_generated_data_enhanced_qnodes_in_snodes/decompositional_chemistry_samples_167424_80_unique_stones_train_shop_2_qhop_1.json')
    parser.add_argument("--val_data", type=str, help="Path to JSON file with support/query data", 
                        default='/home/rsaha/projects/dm_alchemy/src/data/complete_graph_generated_data_enhanced_qnodes_in_snodes/decompositional_chemistry_samples_167424_80_unique_stones_val_shop_2_qhop_1.json')
    parser.add_argument("--output", type=str, default="langchain_vllm_client_results.json", help="Output file for results")
    parser.add_argument("--vllm_url", type=str, default="http://localhost:8000/v1", help="vLLM server OpenAI-compatible API URL")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.1-8B-Instruct", help="Model name being served")
    parser.add_argument("--use_sample_data", type=str, default="false", help="Use built-in sample data for testing")
    parser.add_argument("--max_new_tokens", type=int, default=1000, help="Max new tokens to generate")
    
    # Chat API arguments
    parser.add_argument("--use_chat_api", type=str, default="true", help="Use ChatOpenAI (/v1/chat/completions) instead of VLLMOpenAI (/v1/completions)")
    parser.add_argument("--provider", type=str, default="vllm", choices=["vllm", "fireworks"], help="API provider: vllm or fireworks")
    
    # Batch inference arguments
    parser.add_argument("--enable_batch_inference", type=str, default="true", help="Enable batch inference for faster processing")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for batch inference")
    
    # W&B arguments
    parser.add_argument("--enable_wandb", type=str, default="false", help="Enable W&B monitoring")
    
    # NEW: Stone ID and reasoning arguments
    parser.add_argument("--use_stone_ids", type=str, default="false", help="Use stone IDs (S1, S2, ...) instead of full stone states")
    parser.add_argument("--enable_reasoning", type=str, default="true", help="Allow model to show reasoning steps")
    parser.add_argument("--disable_reasoning", type=str, default="false", help="Prohibit model from showing reasoning steps")
    parser.add_argument("--batch_mode", type=str, default="global", choices=["global", "per_episode"], 
                        help="Batch mode: 'global' (across episodes) or 'per_episode' (within each episode)")
    
    
    parser.add_argument("--temperature", type=float, default=0.0, help="Temperature for generation (not used currently)")
    
    args = parser.parse_args()
    
    # Convert string arguments to boolean
    def str_to_bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')
    
    args.use_sample_data = str_to_bool(args.use_sample_data)
    args.use_chat_api = str_to_bool(args.use_chat_api)
    args.enable_batch_inference = str_to_bool(args.enable_batch_inference)
    args.enable_wandb = str_to_bool(args.enable_wandb)
    args.use_stone_ids = str_to_bool(args.use_stone_ids)
    args.enable_reasoning = str_to_bool(args.enable_reasoning)
    args.disable_reasoning = str_to_bool(args.disable_reasoning)
    
    # Handle reasoning flag logic
    if args.disable_reasoning:
        args.enable_reasoning = False
        
    print("Configuration:", args)
    
    
    
    

    if args.data and args.data.endswith('.json'):
        args.data = args.data.replace('.json', f'_combined.json')
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
        print(f"Detected shop_length: {shop_length}, qhop_length: {qhop_length}")
        
    random.seed(42)
    np.random.seed(42)
    
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
        batch_size=args.batch_size,  
        enable_batch_inference=args.enable_batch_inference,  # NEW
        enable_reasoning=args.enable_reasoning,  
        use_stone_ids=args.use_stone_ids,    
        batch_mode=args.batch_mode,
        
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