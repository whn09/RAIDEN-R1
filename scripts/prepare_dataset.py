#!/usr/bin/env python3
"""
Script to prepare training dataset for RAIDEN-R1

Combines:
1. RAIDEN Benchmark samples (1000)
2. Custom challenging samples (1000)
"""

import argparse
import json
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from data.collection import DatasetBuilder


def main():
    parser = argparse.ArgumentParser(
        description="Prepare dataset for RAIDEN-R1 training"
    )
    parser.add_argument(
        "--benchmark_path",
        type=str,
        default=None,
        help="Path to RAIDEN Benchmark dataset (JSON file)"
    )
    parser.add_argument(
        "--num_custom_samples",
        type=int,
        default=1000,
        help="Number of custom challenging samples to generate"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./data",
        help="Output directory for prepared dataset"
    )
    parser.add_argument(
        "--characters_file",
        type=str,
        default=None,
        help="Optional JSON file with custom character profiles"
    )

    args = parser.parse_args()

    print("="*60)
    print("RAIDEN-R1 Dataset Preparation")
    print("="*60)

    # Initialize dataset builder
    builder = DatasetBuilder()

    # Load RAIDEN Benchmark if provided
    if args.benchmark_path:
        print(f"\nLoading RAIDEN Benchmark from: {args.benchmark_path}")
        try:
            builder.load_from_raiden_benchmark(args.benchmark_path)
            print(f"Loaded {len(builder.samples)} samples from benchmark")
        except FileNotFoundError:
            print(f"Warning: Benchmark file not found at {args.benchmark_path}")
            print("Continuing with custom samples only...")
        except Exception as e:
            print(f"Error loading benchmark: {e}")
            print("Continuing with custom samples only...")

    # Load custom characters if provided
    characters = None
    if args.characters_file:
        print(f"\nLoading character profiles from: {args.characters_file}")
        try:
            with open(args.characters_file, 'r') as f:
                characters = json.load(f)
            print(f"Loaded {len(characters)} character profiles")
        except Exception as e:
            print(f"Error loading characters: {e}")
            print("Using default characters...")

    # Generate custom challenging samples
    print(f"\nGenerating {args.num_custom_samples} challenging samples...")
    builder.generate_challenging_samples(
        num_samples=args.num_custom_samples,
        characters=characters
    )

    # Print statistics
    print("\n" + "="*60)
    print("Dataset Statistics")
    print("="*60)
    stats = builder.get_statistics()
    print(f"Total samples: {stats['total_samples']}")
    print(f"Single-term samples: {stats['single_term_samples']}")
    print(f"Multi-term samples: {stats['multi_term_samples']}")
    print(f"\nDifficulty distribution:")
    for difficulty, count in stats['difficulty_distribution'].items():
        print(f"  {difficulty}: {count}")
    print(f"\nSources:")
    for source, count in stats['sources'].items():
        print(f"  {source}: {count}")

    # Save dataset
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_file = output_dir / "train_dataset.json"
    print(f"\nSaving dataset to: {train_file}")
    builder.save_dataset(str(train_file))

    # Also create a small eval set (10% of data)
    eval_size = max(100, len(builder.samples) // 10)
    eval_samples = builder.samples[:eval_size]

    eval_builder = DatasetBuilder()
    for sample in eval_samples:
        eval_builder.add_sample(sample)

    eval_file = output_dir / "eval_dataset.json"
    print(f"Saving eval dataset to: {eval_file}")
    eval_builder.save_dataset(str(eval_file))

    print("\n" + "="*60)
    print("Dataset preparation completed successfully!")
    print("="*60)


if __name__ == "__main__":
    main()
