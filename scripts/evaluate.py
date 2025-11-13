#!/usr/bin/env python3
"""
Evaluation script for RAIDEN-R1

Evaluates model on SBK and CM metrics
"""

import argparse
import json
from pathlib import Path
import sys
import torch
from tqdm import tqdm

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from evaluation.metrics import RAIDEN_R1_Evaluator, print_evaluation_results
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_eval_data(eval_path: str):
    """Load evaluation data"""
    print(f"Loading evaluation data from: {eval_path}")

    with open(eval_path, 'r') as f:
        data = json.load(f)

    print(f"Loaded {len(data)} evaluation samples")
    return data


def generate_response(model, tokenizer, prompt: str, max_length: int = 512):
    """Generate response from model"""
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        max_length=2048,
        truncation=True,
        padding=True
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_length,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id
        )

    response = tokenizer.decode(
        outputs[0][inputs['input_ids'].shape[1]:],
        skip_special_tokens=True
    )

    return response


def format_prompt(sample):
    """Format sample into prompt"""
    profile_str = json.dumps(sample['character_profile'], indent=2)

    prompt = f"""You are roleplaying as {sample['character']}.

Character Profile:
{profile_str}

User Query: {sample['query']}

Please provide a detailed response with step-by-step reasoning, staying in character.

Response:"""

    return prompt


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate RAIDEN-R1 model"
    )

    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to trained model"
    )
    parser.add_argument(
        "--eval_data",
        type=str,
        required=True,
        help="Path to evaluation dataset (JSON file)"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="./evaluation_results.json",
        help="Output file for results"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to evaluate (for testing)"
    )
    parser.add_argument(
        "--use_llm_judge",
        action="store_true",
        help="Use LLM-as-a-judge for evaluation (requires API key)"
    )
    parser.add_argument(
        "--sbk_weight",
        type=float,
        default=0.5,
        help="Weight for SBK score"
    )
    parser.add_argument(
        "--cm_weight",
        type=float,
        default=0.5,
        help="Weight for CM score"
    )

    args = parser.parse_args()

    print("="*60)
    print("RAIDEN-R1 Evaluation")
    print("="*60)
    print(f"Model: {args.model_path}")
    print(f"Eval Data: {args.eval_data}")
    print(f"SBK Weight: {args.sbk_weight}")
    print(f"CM Weight: {args.cm_weight}")
    print("="*60)

    # Load model and tokenizer
    print("\nLoading model...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto"
    )
    model.eval()

    print("Model loaded successfully")

    # Load evaluation data
    eval_data = load_eval_data(args.eval_data)

    if args.max_samples:
        eval_data = eval_data[:args.max_samples]
        print(f"Evaluating on {len(eval_data)} samples (limited)")

    # Initialize evaluator
    evaluator = RAIDEN_R1_Evaluator(
        sbk_weight=args.sbk_weight,
        cm_weight=args.cm_weight,
        use_llm_judge=args.use_llm_judge
    )

    # Generate responses and evaluate
    print("\nGenerating responses and evaluating...")

    eval_samples = []

    for sample in tqdm(eval_data, desc="Evaluating"):
        # Generate response
        prompt = format_prompt(sample)
        response = generate_response(model, tokenizer, prompt)

        # Prepare for evaluation
        eval_sample = {
            "response": response,
            "character_profile": sample.get("character_profile", {}),
            "query": sample.get("query", ""),
            "conversation_history": sample.get("conversation_history", None)
        }

        eval_samples.append(eval_sample)

    # Evaluate entire dataset
    print("\nComputing metrics...")
    results = evaluator.evaluate_dataset(
        eval_samples,
        output_file=args.output_file
    )

    # Print results
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"Average SBK (Script-Based Knowledge): {results['avg_sbk']:.4f} ({results['avg_sbk']*100:.2f}%)")
    print(f"Average CM (Conversation Memory):     {results['avg_cm']:.4f} ({results['avg_cm']*100:.2f}%)")
    print(f"Average Overall Score:                {results['avg_overall']:.4f} ({results['avg_overall']*100:.2f}%)")
    print(f"Number of Samples:                    {results['num_samples']}")
    print("="*60)

    print(f"\nDetailed results saved to: {args.output_file}")


if __name__ == "__main__":
    main()
