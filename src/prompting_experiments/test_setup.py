#!/usr/bin/env python3
"""
Simple test script to verify the chemistry prompting setup works correctly.
"""

import sys
import time
import requests
import json
from prompt_model import ChemistryPromptEvaluator, create_sample_data

def test_server_connection(url="http://localhost:8000/v1/completions"):
    """Test if vLLM server is accessible."""
    try:
        # Simple test request
        payload = {
            "model": "meta-llama/Llama-3.2-3B-Instruct",
            "prompt": "Hello, world!",
            "max_tokens": 10,
            "temperature": 0.0
        }
        
        response = requests.post(url, json=payload, timeout=10)
        response.raise_for_status()
        result = response.json()
        
        if "choices" in result and len(result["choices"]) > 0:
            print("✓ vLLM server is accessible and responding")
            return True
        else:
            print("✗ vLLM server responded but format is unexpected")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"✗ Cannot connect to vLLM server: {e}")
        return False
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return False

def test_prompt_parsing():
    """Test the example parsing functionality."""
    try:
        evaluator = ChemistryPromptEvaluator()
        
        # Test parsing
        example = "{color: purple, size: small, roundness: pointy, reward: 1} YELLOW PINK -> {color: blue, size: large, roundness: medium_round, reward: 1}"
        parsed = evaluator.parse_chemistry_example(example)
        
        expected_input = "{color: purple, size: small, roundness: pointy, reward: 1}"
        expected_potions = ["YELLOW", "PINK"]
        expected_output = "{color: blue, size: large, roundness: medium_round, reward: 1}"
        
        if (parsed.input_stone == expected_input and 
            parsed.potions == expected_potions and 
            parsed.output_stone == expected_output):
            print("✓ Example parsing works correctly")
            return True
        else:
            print(f"✗ Example parsing failed:")
            print(f"  Expected input: {expected_input}")
            print(f"  Got input: {parsed.input_stone}")
            print(f"  Expected potions: {expected_potions}")
            print(f"  Got potions: {parsed.potions}")
            print(f"  Expected output: {expected_output}")
            print(f"  Got output: {parsed.output_stone}")
            return False
            
    except Exception as e:
        print(f"✗ Example parsing test failed: {e}")
        return False

def test_prompt_creation():
    """Test prompt creation functionality."""
    try:
        evaluator = ChemistryPromptEvaluator()
        data = create_sample_data()
        
        # Test creating a prompt
        query_example = data["query"][0]
        parsed_query = evaluator.parse_chemistry_example(query_example)
        
        prompt = evaluator.create_prompt(
            data["support"][:3],  # Use first 3 support examples
            parsed_query.input_stone,
            parsed_query.potions
        )
        
        # Check that prompt contains expected elements
        if ("You are an expert in magical chemistry" in prompt and 
            "Example 1:" in prompt and
            parsed_query.input_stone in prompt and 
            " ".join(parsed_query.potions) in prompt):
            print("✓ Prompt creation works correctly")
            return True
        else:
            print("✗ Prompt creation failed - missing expected elements")
            print(f"Prompt preview: {prompt[:200]}...")
            return False
            
    except Exception as e:
        print(f"✗ Prompt creation test failed: {e}")
        return False

def main():
    print("Testing Chemistry Prompting Setup")
    print("=" * 40)
    
    # Test 1: Check if server is running
    print("\n1. Testing server connection...")
    server_ok = test_server_connection()
    
    # Test 2: Test example parsing
    print("\n2. Testing example parsing...")
    parsing_ok = test_prompt_parsing()
    
    # Test 3: Test prompt creation
    print("\n3. Testing prompt creation...")
    prompt_ok = test_prompt_creation()
    
    # Summary
    print("\n" + "=" * 40)
    print("TEST SUMMARY")
    print("=" * 40)
    
    if server_ok and parsing_ok and prompt_ok:
        print("✓ All tests passed! The setup is ready for evaluation.")
        print("\nTo run a full evaluation, use:")
        print("  python prompt_model.py --use-sample-data")
    else:
        print("✗ Some tests failed. Please check the issues above.")
        if not server_ok:
            print("\n  To start the server:")
            print("    ./run_vllm_server.sh")
    
    return server_ok and parsing_ok and prompt_ok

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
