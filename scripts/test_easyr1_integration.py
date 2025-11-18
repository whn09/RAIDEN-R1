#!/usr/bin/env python3
"""
Test script to verify EasyR1 integration for RAIDEN-R1

This script checks:
1. Data format conversion
2. Reward function
3. Configuration files
"""

import sys
import json
from pathlib import Path

def test_data_conversion():
    """Test if data conversion script works"""
    print("Testing data conversion...")

    # Check if conversion script exists
    script_path = Path("scripts/convert_to_easyr1_format.py")
    if not script_path.exists():
        print("  ❌ Conversion script not found!")
        return False

    # Check if training data exists
    train_path = Path("data/training/train.json")
    if not train_path.exists():
        print("  ⚠️  Training data not found (expected for fresh install)")
        return True

    # Try to import and run a sample conversion
    try:
        # Add parent directory to path
        sys.path.insert(0, str(Path(__file__).parent.parent))
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "convert_to_easyr1_format",
            script_path
        )
        converter_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(converter_module)
        convert_sample = converter_module.convert_sample

        # Load a sample
        with open(train_path, 'r') as f:
            data = json.load(f)

        if len(data) > 0:
            sample = data[0]
            converted = convert_sample(sample)

            # Check required keys
            required_keys = ["problem", "answer", "images", "metadata"]
            if all(key in converted for key in required_keys):
                print("  ✓ Data conversion works!")
                print(f"    Sample problem length: {len(converted['problem'])}")
                print(f"    Sample answer: {converted['answer'][:50]}...")
                return True
            else:
                print(f"  ❌ Converted sample missing required keys: {required_keys}")
                return False

    except Exception as e:
        print(f"  ❌ Error testing conversion: {e}")
        return False

    return True


def test_reward_function():
    """Test if reward function works"""
    print("\nTesting reward function...")

    try:
        # Import reward function
        sys.path.insert(0, 'easyr1/reward_function')
        from raiden_vrar import compute_score

        # Test with sample data
        test_inputs = [
            {
                "response": "<think>This character is a doctor who eats lunch on the rooftop.</think>\n\nI started eating lunch on the rooftop after becoming a doctor.",
                "response_length": 100,
                "ground_truth": "我从成为医生后开始在屋上吃午餐。"
            }
        ]

        scores = compute_score(test_inputs)

        # Check score format
        if len(scores) == 1:
            score = scores[0]
            required_keys = ["overall", "accuracy", "format", "thinking"]

            if all(key in score for key in required_keys):
                print("  ✓ Reward function works!")
                print(f"    Overall: {score['overall']:.3f}")
                print(f"    Accuracy: {score['accuracy']:.3f}")
                print(f"    Format: {score['format']:.3f}")
                print(f"    Thinking: {score['thinking']:.3f}")
                return True
            else:
                print(f"  ❌ Score missing required keys: {required_keys}")
                return False
        else:
            print(f"  ❌ Expected 1 score, got {len(scores)}")
            return False

    except ImportError as e:
        print(f"  ❌ Failed to import reward function: {e}")
        return False
    except Exception as e:
        print(f"  ❌ Error testing reward function: {e}")
        return False


def test_config_files():
    """Test if configuration files are valid"""
    print("\nTesting configuration files...")

    try:
        import yaml

        config_path = Path("configs/easyr1_config.yaml")
        if not config_path.exists():
            print("  ❌ EasyR1 config file not found!")
            return False

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        # Check essential sections
        required_sections = ["data", "algorithm", "worker", "trainer"]
        if all(section in config for section in required_sections):
            print("  ✓ Configuration file is valid!")
            print(f"    Model: {config['worker']['actor']['model']['model_path']}")
            print(f"    GPUs: {config['trainer']['n_gpus_per_node']}")
            print(f"    Epochs: {config['trainer']['total_epochs']}")
            return True
        else:
            print(f"  ❌ Config missing required sections: {required_sections}")
            return False

    except ImportError:
        print("  ⚠️  PyYAML not installed, skipping config validation")
        return True
    except Exception as e:
        print(f"  ❌ Error testing config: {e}")
        return False


def test_prompt_template():
    """Test if prompt template exists"""
    print("\nTesting prompt template...")

    template_path = Path("easyr1/format_prompt/role_playing.jinja")
    if not template_path.exists():
        print("  ❌ Prompt template not found!")
        return False

    with open(template_path, 'r') as f:
        template_content = f.read()

    if "{{ content | trim }}" in template_content:
        print("  ✓ Prompt template is valid!")
        print(f"    Length: {len(template_content)} characters")
        return True
    else:
        print("  ❌ Prompt template appears invalid")
        return False


def test_easyr1_installation():
    """Test if EasyR1 is installed"""
    print("\nTesting EasyR1 installation...")

    try:
        import verl
        print(f"  ✓ EasyR1 (verl) is installed!")
        print(f"    Location: {verl.__file__}")
        return True
    except ImportError:
        print("  ⚠️  EasyR1 not installed. To install:")
        print("     git clone https://github.com/hiyouga/EasyR1.git")
        print("     cd EasyR1 && pip install -e .")
        return False


def main():
    print("=" * 60)
    print("EasyR1 Integration Test for RAIDEN-R1")
    print("=" * 60)

    results = {
        "Data Conversion": test_data_conversion(),
        "Reward Function": test_reward_function(),
        "Config Files": test_config_files(),
        "Prompt Template": test_prompt_template(),
        "EasyR1 Installation": test_easyr1_installation(),
    }

    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)

    all_passed = True
    for test_name, result in results.items():
        status = "✓ PASS" if result else "❌ FAIL"
        print(f"{test_name:.<40} {status}")
        if not result:
            all_passed = False

    print("=" * 60)

    if all_passed:
        print("\n✓ All tests passed! EasyR1 integration is ready.")
        print("\nNext steps:")
        print("  1. Generate or prepare training data")
        print("  2. Convert data: python scripts/convert_to_easyr1_format.py")
        print("  3. Start training: bash scripts/train_with_easyr1.sh")
        return 0
    else:
        print("\n⚠️  Some tests failed. Please fix the issues above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
