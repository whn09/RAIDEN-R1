#!/usr/bin/env python3
"""
Generate RAIDEN training data using SGLang-deployed models

This script uses locally deployed open-source models via SGLang for faster generation.
Supports models like MiniMax M2, GLM-4, Qwen2.5, etc.
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from data.sglang_generator import SGLangGenerator
from data.collection import DatasetBuilder


def main():
    parser = argparse.ArgumentParser(
        description="Generate RAIDEN training data using SGLang-deployed models"
    )

    # Input/output arguments
    parser.add_argument(
        "--profiles_file",
        type=str,
        default="./data/online_profiles.jsonl",
        help="Path to character profiles JSONL file"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="./data/generated_samples_sglang.json",
        help="Path to save generated samples"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./data/training",
        help="Directory for final training data"
    )

    # Generation parameters
    parser.add_argument(
        "--num_samples_per_profile",
        type=int,
        default=2,
        help="Number of SBK samples per profile"
    )
    parser.add_argument(
        "--include_cm",
        action="store_true",
        help="Generate Conversation Memory (CM) samples"
    )
    parser.add_argument(
        "--max_profiles",
        type=int,
        default=None,
        help="Maximum number of profiles to process (for testing)"
    )

    # SGLang parameters
    parser.add_argument(
        "--base_url",
        type=str,
        default="http://localhost:30000",
        help="SGLang server URL"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        help="Model name (optional, for metadata)"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="Request timeout in seconds"
    )

    # Language parameters
    parser.add_argument(
        "--language",
        type=str,
        default="zh",
        choices=["zh", "ja", "en", "ko"],
        help="Default language for generation (zh=Chinese, ja=Japanese, en=English, ko=Korean)"
    )
    parser.add_argument(
        "--no_auto_detect",
        action="store_true",
        help="Disable automatic language detection from character profiles"
    )

    # Dataset parameters
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.9,
        help="Ratio of training data (rest is validation)"
    )

    args = parser.parse_args()

    print("=" * 70)
    print("RAIDEN-R1 Data Generation with SGLang")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Profiles file: {args.profiles_file}")
    print(f"  Output file: {args.output_file}")
    print(f"  Output directory: {args.output_dir}")
    print(f"  Samples per profile: {args.num_samples_per_profile}")
    print(f"  Include CM samples: {args.include_cm}")
    print(f"  Max profiles: {args.max_profiles or 'all'}")
    print(f"  SGLang URL: {args.base_url}")
    print(f"  Model: {args.model_name or 'auto-detect'}")
    print(f"  Language: {args.language}")
    print(f"  Auto-detect language: {not args.no_auto_detect}")
    print("=" * 70)

    # Initialize generator
    print("\nInitializing SGLang generator...")
    try:
        generator = SGLangGenerator(
            base_url=args.base_url,
            model_name=args.model_name,
            timeout=args.timeout,
            language=args.language,
            auto_detect_language=not args.no_auto_detect
        )
    except Exception as e:
        print(f"Error: Failed to initialize SGLang generator: {e}")
        print("\nMake sure SGLang server is running:")
        print(f"  curl {args.base_url}/health")
        return 1

    # Generate dataset
    print("\nGenerating dataset from profiles...")
    try:
        generator.generate_dataset_from_profiles(
            profiles_file=args.profiles_file,
            output_file=args.output_file,
            num_samples_per_profile=args.num_samples_per_profile,
            include_conversation_memory=args.include_cm,
            max_profiles=args.max_profiles
        )
    except FileNotFoundError:
        print(f"Error: Profiles file not found at {args.profiles_file}")
        return 1
    except Exception as e:
        print(f"Error generating dataset: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Load generated samples into dataset builder
    print("\nPreparing dataset for training...")
    builder = DatasetBuilder()
    builder.load_from_json(args.output_file)

    # Print statistics
    stats = builder.get_statistics()
    print("\n" + "=" * 70)
    print("Dataset Statistics")
    print("=" * 70)
    print(f"Total samples: {stats['total_samples']}")
    print(f"Single-term validation samples: {stats['single_term_samples']}")
    print(f"Multi-term parsing samples: {stats['multi_term_samples']}")

    print(f"\nDifficulty distribution:")
    for difficulty, count in sorted(stats['difficulty_distribution'].items()):
        percentage = (count / stats['total_samples'] * 100) if stats['total_samples'] > 0 else 0
        print(f"  {difficulty}: {count} ({percentage:.1f}%)")

    print(f"\nQuestion type distribution:")
    for qtype, count in sorted(stats['question_types'].items()):
        percentage = (count / stats['total_samples'] * 100) if stats['total_samples'] > 0 else 0
        print(f"  {qtype}: {count} ({percentage:.1f}%)")

    # Export for training
    print(f"\nExporting dataset to {args.output_dir}...")
    builder.export_for_training(
        output_dir=args.output_dir,
        train_ratio=args.train_ratio
    )

    print("\n" + "=" * 70)
    print("Data generation completed successfully!")
    print("=" * 70)
    print(f"\nGenerated files:")
    print(f"  - Raw samples: {args.output_file}")
    print(f"  - Training set: {args.output_dir}/train.json")
    print(f"  - Validation set: {args.output_dir}/validation.json")
    print(f"  - Statistics: {args.output_dir}/dataset_stats.json")
    print("\nYou can now use these files for GRPO training.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
