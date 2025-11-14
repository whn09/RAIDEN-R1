#!/usr/bin/env python3
"""
Quick test script for SGLang data generator

Tests the data generation pipeline with a small sample.
"""

import sys
from pathlib import Path
import json

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from data.sglang_generator import SGLangGenerator


def test_single_sample():
    """Test generating a single sample"""
    print("=" * 70)
    print("Testing SGLang Data Generator - Single Sample")
    print("=" * 70)

    # Initialize generator
    print("\n1. Initializing SGLang generator...")
    try:
        generator = SGLangGenerator(
            base_url="http://localhost:30000",
            model_name="test-model"
        )
        print("   ✓ Generator initialized successfully")
    except Exception as e:
        print(f"   ✗ Failed to initialize generator: {e}")
        print("\nPlease ensure:")
        print("  - SGLang server is running on localhost:30000")
        print("  - Run: ./scripts/deploy_sglang.sh --model-path /path/to/model")
        return 1

    # Create a test character profile
    test_profile = {
        "Name": "Alice",
        "Age": "25",
        "Personality": "friendly and outgoing",
        "Occupation": "software engineer",
        "Hobbies": ["reading", "hiking", "coding"],
        "Background": "Grew up in Seattle, loves coffee and rainy days"
    }

    print("\n2. Test character profile:")
    print(json.dumps(test_profile, indent=2, ensure_ascii=False))

    # Generate WH questions
    print("\n3. Generating WH-questions...")
    try:
        questions = generator.generate_wh_questions(
            character_profile=test_profile,
            num_questions=2
        )
        print(f"   ✓ Generated {len(questions)} questions")
        for i, q in enumerate(questions, 1):
            print(f"\n   Question {i}:")
            print(f"     Q: {q['question']}")
            print(f"     Type: {q.get('question_type', 'unknown')}")
            print(f"     Difficulty: {q.get('difficulty', 'unknown')}")
            print(f"     Focus: {q.get('focus', 'unknown')}")
    except Exception as e:
        print(f"   ✗ Failed to generate questions: {e}")
        return 1

    if not questions:
        print("   ✗ No questions generated")
        return 1

    # Generate answer with CoT for first question
    print("\n4. Generating answer with Chain-of-Thought...")
    try:
        question = questions[0]['question']
        reasoning, answer = generator.generate_answer_with_cot(
            character_profile=test_profile,
            conversation_history=None,
            question=question
        )
        print(f"   ✓ Answer generated")
        print(f"\n   Question: {question}")
        print(f"\n   Reasoning:")
        print(f"   {reasoning[:200]}...")
        print(f"\n   Answer:")
        print(f"   {answer}")
    except Exception as e:
        print(f"   ✗ Failed to generate answer: {e}")
        return 1

    # Extract keywords
    print("\n5. Extracting keywords...")
    try:
        keywords = generator.extract_keywords(
            question=question,
            answer=answer,
            character_profile=test_profile
        )
        print(f"   ✓ Extracted {len(keywords)} keywords: {keywords}")
    except Exception as e:
        print(f"   ✗ Failed to extract keywords: {e}")
        return 1

    # Generate complete sample
    print("\n6. Generating complete training sample...")
    try:
        sample = generator.generate_sample_from_profile(
            character_profile=test_profile,
            conversation_history=None,
            difficulty="medium"
        )
        print(f"   ✓ Sample generated successfully")
        print(f"\n   Character: {sample.character_name}")
        print(f"   Question: {sample.question}")
        print(f"   Answer: {sample.answer}")
        print(f"   Keywords: {sample.keywords}")
        print(f"   Validation method: {sample.validation_method}")
        print(f"   Difficulty: {sample.difficulty}")
    except Exception as e:
        print(f"   ✗ Failed to generate sample: {e}")
        return 1

    print("\n" + "=" * 70)
    print("All tests passed! ✓")
    print("=" * 70)
    print("\nYou can now run the full data generation with:")
    print("  python scripts/generate_data_with_sglang.py")

    return 0


if __name__ == "__main__":
    sys.exit(test_single_sample())
