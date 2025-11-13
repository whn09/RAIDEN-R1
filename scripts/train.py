#!/usr/bin/env python3
"""
Main training script for RAIDEN-R1

Trains a model using GRPO with VRAR rewards
"""

import argparse
import json
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from training.grpo_trainer import GRPOTrainer, GRPOConfig
from data.collection import RolePlayingSample


def load_dataset(dataset_path: str):
    """Load dataset from JSON file"""
    print(f"Loading dataset from: {dataset_path}")

    with open(dataset_path, 'r') as f:
        data = json.load(f)

    samples = []
    for item in data:
        sample = RolePlayingSample(**item)
        samples.append(sample)

    print(f"Loaded {len(samples)} samples")
    return samples


def main():
    parser = argparse.ArgumentParser(
        description="Train RAIDEN-R1 model with GRPO"
    )

    # Dataset arguments
    parser.add_argument(
        "--train_data",
        type=str,
        required=True,
        help="Path to training dataset (JSON file)"
    )
    parser.add_argument(
        "--eval_data",
        type=str,
        default=None,
        help="Path to evaluation dataset (JSON file)"
    )

    # Model arguments
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen2.5-14B-Instruct",
        help="Base model name or path"
    )

    # Training arguments
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=3e-6,
        help="Learning rate"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size per GPU"
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=1,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Gradient accumulation steps"
    )
    parser.add_argument(
        "--num_samples_per_prompt",
        type=int,
        default=4,
        help="Number of samples per prompt for GRPO"
    )
    parser.add_argument(
        "--accuracy_weight",
        type=float,
        default=0.7,
        help="Weight for accuracy reward"
    )
    parser.add_argument(
        "--format_weight",
        type=float,
        default=0.3,
        help="Weight for format reward"
    )

    # Output arguments
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs",
        help="Output directory for checkpoints"
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=100,
        help="Save checkpoint every N steps"
    )

    # Other arguments
    parser.add_argument(
        "--use_bf16",
        action="store_true",
        default=True,
        help="Use bfloat16 precision"
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        default=True,
        help="Enable gradient checkpointing"
    )

    args = parser.parse_args()

    print("="*60)
    print("RAIDEN-R1 Training with GRPO")
    print("="*60)
    print(f"Model: {args.model_name}")
    print(f"Learning Rate: {args.learning_rate}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Epochs: {args.num_epochs}")
    print(f"Output Directory: {args.output_dir}")
    print("="*60)

    # Load datasets
    train_dataset = load_dataset(args.train_data)

    eval_dataset = None
    if args.eval_data:
        eval_dataset = load_dataset(args.eval_data)

    # Create GRPO config
    config = GRPOConfig(
        model_name=args.model_name,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_samples_per_prompt=args.num_samples_per_prompt,
        accuracy_weight=args.accuracy_weight,
        format_weight=args.format_weight,
        use_bf16=args.use_bf16,
        gradient_checkpointing=args.gradient_checkpointing,
        output_dir=args.output_dir,
        save_steps=args.save_steps
    )

    # Initialize trainer
    print("\nInitializing GRPO trainer...")
    trainer = GRPOTrainer(
        config=config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset
    )

    # Start training
    print("\nStarting training...")
    trainer.train()

    print("\n" + "="*60)
    print("Training completed successfully!")
    print(f"Model saved to: {args.output_dir}")
    print("="*60)


if __name__ == "__main__":
    main()
