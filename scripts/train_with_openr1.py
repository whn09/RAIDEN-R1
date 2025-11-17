#!/usr/bin/env python3
"""
Train RAIDEN-R1 using OpenR1's GRPO implementation

This script integrates OpenR1's GRPO framework with RAIDEN's VRAR rewards
"""

import sys
import yaml
import argparse
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    HfArgumentParser,
    TrainingArguments,
    set_seed
)
from accelerate import Accelerator

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from training.openr1_adapter import (
    RAIDENDataAdapter,
    create_raiden_reward_function
)

try:
    from open_r1.configs import GRPOConfig
    from trl import GRPOTrainer
    OPENR1_AVAILABLE = True
except ImportError as e:
    print(f"Warning: OpenR1 or TRL not installed: {e}")
    print("Please install with:")
    print("  1. git clone https://github.com/huggingface/open-r1.git")
    print("  2. cd open-r1 && pip install -e \".[dev]\" && cd ..")
    print("  3. pip install trl")
    OPENR1_AVAILABLE = False


@dataclass
class RAIDENTrainingArguments:
    """Arguments specific to RAIDEN training"""

    # Data arguments
    train_data_path: str = field(
        metadata={"help": "Path to training data JSON file"}
    )
    eval_data_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to evaluation data JSON file"}
    )

    # Model arguments
    model_name_or_path: str = field(
        default="Qwen/Qwen2.5-14B-Instruct",
        metadata={"help": "Path to pretrained model or model identifier"}
    )

    # Reward arguments
    accuracy_weight: float = field(
        default=0.7,
        metadata={"help": "Weight for accuracy reward in VRAR"}
    )
    format_weight: float = field(
        default=0.3,
        metadata={"help": "Weight for format reward in VRAR"}
    )

    # GRPO arguments
    num_samples_per_prompt: int = field(
        default=4,
        metadata={"help": "Number of samples per prompt for GRPO"}
    )
    kl_penalty: float = field(
        default=0.1,
        metadata={"help": "KL penalty coefficient"}
    )

    # System prompt
    system_prompt: Optional[str] = field(
        default=None,
        metadata={"help": "System prompt for role-playing"}
    )

    # Output
    output_dir: str = field(
        default="./outputs_openr1",
        metadata={"help": "Output directory"}
    )

    # Logging
    report_to: Optional[list] = field(
        default=None,
        metadata={"help": "List of integrations to report to (wandb, tensorboard, etc). Empty list disables all."}
    )

    # Training
    num_train_epochs: int = field(
        default=1,
        metadata={"help": "Number of training epochs"}
    )
    per_device_train_batch_size: int = field(
        default=4,
        metadata={"help": "Batch size per device"}
    )
    gradient_accumulation_steps: int = field(
        default=2,
        metadata={"help": "Gradient accumulation steps"}
    )
    learning_rate: float = field(
        default=3e-6,
        metadata={"help": "Learning rate"}
    )
    warmup_steps: int = field(
        default=100,
        metadata={"help": "Warmup steps"}
    )
    logging_steps: int = field(
        default=10,
        metadata={"help": "Logging steps"}
    )
    save_steps: int = field(
        default=100,
        metadata={"help": "Save checkpoint steps"}
    )
    eval_steps: int = field(
        default=50,
        metadata={"help": "Evaluation steps"}
    )
    bf16: bool = field(
        default=True,
        metadata={"help": "Use bfloat16 precision"}
    )
    gradient_checkpointing: bool = field(
        default=True,
        metadata={"help": "Enable gradient checkpointing"}
    )

    seed: int = field(
        default=42,
        metadata={"help": "Random seed"}
    )


def main():
    # Check OpenR1 availability
    if not OPENR1_AVAILABLE:
        print("Error: OpenR1 is not installed.")
        print("Please install it with: pip install open-r1")
        sys.exit(1)

    # Parse arguments - support both config files and command line
    parser = HfArgumentParser(RAIDENTrainingArguments)

    # Check if first argument is a config file
    if len(sys.argv) > 1 and (sys.argv[1].endswith('.yaml') or sys.argv[1].endswith('.yml')):
        args = parser.parse_yaml_file(yaml_file=sys.argv[1])[0]
    elif len(sys.argv) > 1 and sys.argv[1].endswith('.json'):
        args = parser.parse_json_file(json_file=sys.argv[1])[0]
    else:
        args = parser.parse_args_into_dataclasses()[0]

    # Set seed
    set_seed(args.seed)

    print("=" * 70)
    print("RAIDEN-R1 Training with OpenR1")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Model: {args.model_name_or_path}")
    print(f"  Training data: {args.train_data_path}")
    print(f"  Eval data: {args.eval_data_path or 'None'}")
    print(f"  Output directory: {args.output_dir}")
    print(f"  Accuracy weight: {args.accuracy_weight}")
    print(f"  Format weight: {args.format_weight}")
    print(f"  Num samples per prompt: {args.num_samples_per_prompt}")
    print(f"  KL penalty: {args.kl_penalty}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Batch size: {args.per_device_train_batch_size}")
    print(f"  Gradient accumulation: {args.gradient_accumulation_steps}")
    print(f"  Epochs: {args.num_train_epochs}")
    print(f"  BF16: {args.bf16}")
    print("=" * 70)

    # Initialize accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision="bf16" if args.bf16 else "no"
    )

    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        padding_side="left"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load datasets
    print("\nLoading and converting datasets...")
    data_adapter = RAIDENDataAdapter(system_prompt=args.system_prompt)

    train_dataset = data_adapter.load_raiden_dataset(args.train_data_path)
    print(f"  Loaded {len(train_dataset)} training samples")

    eval_dataset = None
    if args.eval_data_path:
        eval_dataset = data_adapter.load_raiden_dataset(args.eval_data_path)
        print(f"  Loaded {len(eval_dataset)} evaluation samples")

    # Load model
    print("\nLoading model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float32,
        trust_remote_code=True,
    )

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # Create reward function
    print("\nInitializing VRAR reward function...")
    reward_function = create_raiden_reward_function(
        tokenizer=tokenizer,
        accuracy_weight=args.accuracy_weight,
        format_weight=args.format_weight
    )

    # Configure GRPO
    print("\nConfiguring GRPO trainer...")
    grpo_config = GRPOConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps if eval_dataset else None,
        bf16=args.bf16,
        gradient_checkpointing=args.gradient_checkpointing,
        report_to=args.report_to if args.report_to is not None else [],
        # GRPO specific parameters (using TRL naming)
        num_generations=args.num_samples_per_prompt,  # Number of responses per prompt
        beta=args.kl_penalty,  # KL divergence penalty coefficient
        max_completion_length=2048,
        temperature=0.7,
        top_p=0.9,
        system_prompt=args.system_prompt,
    )

    # Create trainer
    print("\nInitializing GRPO trainer...")
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,  # TRL uses 'processing_class' for tokenizer
        args=grpo_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        reward_funcs=reward_function,  # TRL uses 'reward_funcs'
    )

    # Train
    print("\n" + "=" * 70)
    print("Starting training...")
    print("=" * 70 + "\n")

    trainer.train()

    # Save final model
    print("\nSaving final model...")
    output_path = Path(args.output_dir) / "final_model"
    trainer.save_model(str(output_path))
    tokenizer.save_pretrained(str(output_path))

    print("\n" + "=" * 70)
    print("Training completed successfully!")
    print("=" * 70)
    print(f"\nModel saved to: {output_path}")

    # Evaluation
    if eval_dataset:
        print("\nRunning final evaluation...")
        metrics = trainer.evaluate()
        print("\nEvaluation metrics:")
        for key, value in metrics.items():
            print(f"  {key}: {value}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
