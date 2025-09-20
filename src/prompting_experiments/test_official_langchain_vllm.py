#!/usr/bin/env python3
"""
Test script for the official LangChain VLLM integration (direct model loading).
"""

import sys
from official_langchain_vllm import test_connection, ChemistryPromptEvaluator, create_sample_data, LANGCHAIN_AVAILABLE

def test_direct_vllm_integration():
    """Test the direct LangChain + vLLM integration."""
    print("Testing Official LangChain + vLLM Integration (Direct Model Loading)")
    print("=" * 70)
    
    if not LANGCHAIN_AVAILABLE:
        print("❌ LangChain packages not available")
        return False
    
    # Test 1: Basic model loading test
    print("\n1. Testing model loading...")
    model_name = "meta-llama/Llama-3.2-1B"  # Use a smaller model for testing
    if not test_connection(model_name):
        print("💡 Tip: Make sure the model is available locally or adjust the model name")
        return False
    
    # Test 2: Chemistry evaluator initialization
    print("\n2. Testing chemistry evaluator initialization...")
    try:
        evaluator = ChemistryPromptEvaluator(
            model_name=model_name,
            max_tokens=50,  # Smaller for faster testing
            tensor_parallel_size=1
        )
        print(f"✅ Chemistry evaluator created successfully")
        print(f"   Model: {evaluator.model_name}")
    except Exception as e:
        print(f"❌ Failed to create chemistry evaluator: {e}")
        return False
    
    # Test 3: Example parsing
    print("\n3. Testing example parsing...")
    try:
        sample_example = "{color: purple, size: small, roundness: pointy, reward: 1} YELLOW PINK -> {color: blue, size: large, roundness: medium_round, reward: 1}"
        parsed = evaluator.parse_chemistry_example(sample_example)
        print(f"✅ Example parsing successful")
        print(f"   Input: {parsed.input_stone}")
        print(f"   Potions: {parsed.potions}")
        print(f"   Output: {parsed.output_stone}")
    except Exception as e:
        print(f"❌ Example parsing failed: {e}")
        return False
    
    # Test 4: Few-shot prompt creation
    print("\n4. Testing few-shot prompt creation...")
    try:
        sample_data = create_sample_data()
        few_shot_prompt = evaluator.create_few_shot_prompt(sample_data["support"])
        test_prompt = few_shot_prompt.format(
            input_stone="{color: purple, size: small, roundness: pointy, reward: 1}",
            potions="GREEN"
        )
        print(f"✅ Few-shot prompt creation successful")
        print(f"   Prompt length: {len(test_prompt)} characters")
        print(f"   Preview: {test_prompt[:200]}...")
    except Exception as e:
        print(f"❌ Few-shot prompt creation failed: {e}")
        return False
    
    # Test 5: Single example evaluation (optional, may take time)
    print("\n5. Testing single example evaluation (optional)...")
    try:
        sample_data = create_sample_data()
        result = evaluator.evaluate_single_example(
            sample_data["support"][:2],  # Use fewer examples for faster testing
            sample_data["query"][0]
        )
        print(f"✅ Single example evaluation successful")
        print(f"   Query: {result['query'][:80]}...")
        print(f"   Predicted: {result['predicted_output']}")
        print(f"   Expected: {result['expected_output']}")
        print(f"   Correct: {result['correct']}")
    except Exception as e:
        print(f"⚠️  Single example evaluation failed (this may be due to model loading issues): {e}")
        # Don't return False here as this test is optional
    
    print("\n" + "=" * 70)
    print("✅ Core LangChain + vLLM tests passed!")
    print("\n📋 Usage Examples:")
    print("  # Test model loading:")
    print(f"  python official_langchain_vllm.py --test-model --model-name {model_name}")
    print("  ")
    print("  # Run evaluation with sample data:")
    print("  python official_langchain_vllm.py --use-sample-data")
    print("  ")
    print("  # Run evaluation with your data:")
    print("  python official_langchain_vllm.py --data your_data.json")
    
    print("\n💡 Note: This approach loads the model directly (no separate server needed)")
    print("   Pros: Simpler setup, more control over model parameters")
    print("   Cons: Uses more memory, slower startup time")
    
    return True


if __name__ == "__main__":
    success = test_direct_vllm_integration()
    sys.exit(0 if success else 1)
