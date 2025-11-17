#!/usr/bin/env python3
"""
Test OpenR1 Integration for RAIDEN-R1

This script verifies that OpenR1 is correctly integrated and ready to use
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

def test_openr1_installation():
    """Test if OpenR1 is installed"""
    print("=" * 70)
    print("Testing OpenR1 Installation")
    print("=" * 70)

    try:
        import open_r1
        print("✓ OpenR1 is installed")
        print(f"  Version: {getattr(open_r1, '__version__', 'unknown')}")

        # Test specific imports needed for training
        from open_r1.configs import GRPOConfig
        from trl import GRPOTrainer
        print("✓ Required classes (GRPOConfig, GRPOTrainer) can be imported")
        return True
    except ImportError as e:
        print(f"✗ OpenR1 or required dependencies not installed: {e}")
        print("\nTo install OpenR1 (must install from source):")
        print("  git clone https://github.com/huggingface/open-r1.git")
        print("  cd open-r1")
        print("  pip install -e \".[dev]\"")
        print("  cd ..")
        print("  pip install trl  # If not already installed")
        return False


def test_data_adapter():
    """Test RAIDEN data adapter"""
    print("\n" + "=" * 70)
    print("Testing Data Adapter")
    print("=" * 70)

    try:
        from training.openr1_adapter import RAIDENDataAdapter

        adapter = RAIDENDataAdapter()
        print("✓ Data adapter imported successfully")

        # Test sample conversion
        sample = {
            "character_name": "测试角色",
            "character_profile": {
                "Name": "测试角色",
                "Gender": "female",
                "Introduction": "一个测试角色",
                "Detailed Description": "详细描述"
            },
            "question": "你叫什么名字？",
            "answer": "我叫测试角色。",
            "keywords": ["测试角色"],
            "question_type": "what",
            "validation_method": "single_term_validation",
            "difficulty": "easy",
            "metadata": {}
        }

        converted = adapter.convert_sample_to_openr1(sample)
        print("✓ Sample conversion successful")
        print(f"  Prompt has {len(converted['prompt'])} messages")
        print(f"  Expected answer: {converted['expected_answer'][:30]}...")

        return True

    except Exception as e:
        print(f"✗ Data adapter test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_reward_function():
    """Test VRAR reward function"""
    print("\n" + "=" * 70)
    print("Testing Reward Function")
    print("=" * 70)

    try:
        from training.openr1_adapter import create_raiden_reward_function
        from transformers import AutoTokenizer

        print("  Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            "Qwen/Qwen2.5-0.5B-Instruct",
            trust_remote_code=True
        )

        reward_fn = create_raiden_reward_function(
            tokenizer=tokenizer,
            accuracy_weight=0.7,
            format_weight=0.3
        )
        print("✓ Reward function created successfully")

        # Test reward calculation
        test_prompts = [{
            "keywords": ["测试"],
            "validation_method": "single_term_validation",
            "expected_answer": "这是一个测试。"
        }]
        test_completions = ["这是一个测试回答。"]

        rewards = reward_fn(test_prompts, test_completions)
        print(f"✓ Reward calculation successful")
        print(f"  Test reward: {rewards[0]:.3f}")

        return True

    except Exception as e:
        print(f"✗ Reward function test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dependencies():
    """Test required dependencies"""
    print("\n" + "=" * 70)
    print("Testing Dependencies")
    print("=" * 70)

    dependencies = {
        "torch": "PyTorch",
        "transformers": "Transformers",
        "accelerate": "Accelerate",
        "datasets": "Datasets",
    }

    all_ok = True
    for module, name in dependencies.items():
        try:
            __import__(module)
            print(f"✓ {name} is installed")
        except ImportError:
            print(f"✗ {name} is not installed")
            all_ok = False

    return all_ok


def main():
    """Run all tests"""
    print("\n" + "=" * 70)
    print("RAIDEN-R1 OpenR1 Integration Test")
    print("=" * 70 + "\n")

    results = []

    # Test OpenR1 installation
    results.append(("OpenR1 Installation", test_openr1_installation()))

    # Test dependencies
    results.append(("Dependencies", test_dependencies()))

    # Test data adapter
    results.append(("Data Adapter", test_data_adapter()))

    # Test reward function (only if other tests pass)
    if all(r[1] for r in results):
        results.append(("Reward Function", test_reward_function()))

    # Summary
    print("\n" + "=" * 70)
    print("Test Summary")
    print("=" * 70)

    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {name}")

    all_passed = all(r[1] for r in results)

    print("\n" + "=" * 70)
    if all_passed:
        print("✓ All tests passed! OpenR1 integration is ready to use.")
        print("=" * 70)
        print("\nNext steps:")
        print("  1. Generate training data:")
        print("     python scripts/generate_data_with_sglang.py --language zh")
        print("  2. Start training:")
        print("     python scripts/train_with_openr1.py configs/openr1_config.yaml")
        return 0
    else:
        print("✗ Some tests failed. Please fix the issues above.")
        print("=" * 70)
        print("\nCommon fixes:")
        print("  - Install OpenR1 from source:")
        print("    git clone https://github.com/huggingface/open-r1.git")
        print("    cd open-r1 && pip install -e \".[dev]\" && cd ..")
        print("  - Install dependencies: pip install -r requirements.txt")
        print("  - Check Python version: Python 3.8+ required")
        return 1


if __name__ == "__main__":
    sys.exit(main())
