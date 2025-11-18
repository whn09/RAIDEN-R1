#!/usr/bin/env python3
"""
Convert RAIDEN-R1 dataset to EasyR1 format

RAIDEN-R1 format:
{
    "character_name": "...",
    "character_profile": {...},
    "conversation_history": [],
    "question": "...",
    "answer": "...",
    "keywords": [...],
    "question_type": "...",
    "validation_method": "...",
    "difficulty": "...",
    "metadata": {...}
}

EasyR1 format:
{
    "problem": "...",  # Combined character context + question
    "answer": "...",
    "images": [],  # Empty for text-only
    "metadata": {...}  # Additional metadata
}
"""

import json
import argparse
from pathlib import Path
from typing import Dict, Any, List


def format_character_context(
    character_name: str,
    character_profile: Dict[str, Any],
    conversation_history: List[Dict[str, str]] = None
) -> str:
    """Format character information as context for the problem"""
    context_parts = []

    # Add character profile
    context_parts.append(f"<Character Setting>")
    context_parts.append(f"Name: {character_profile.get('Name', character_name)}")

    if 'Gender' in character_profile:
        context_parts.append(f"Gender: {character_profile['Gender']}")

    if 'Introduction' in character_profile:
        context_parts.append(f"Introduction: {character_profile['Introduction']}")

    if 'Detailed Description' in character_profile:
        context_parts.append(f"Detailed Description: {character_profile['Detailed Description']}")

    context_parts.append("</Character Setting>")

    # Add conversation history if present
    if conversation_history:
        context_parts.append("\n<Conversation History>")
        for turn in conversation_history:
            role = turn.get('role', 'unknown')
            content = turn.get('content', '')
            context_parts.append(f"{role}: {content}")
        context_parts.append("</Conversation History>")

    return "\n".join(context_parts)


def convert_sample(sample: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a single RAIDEN-R1 sample to EasyR1 format"""

    # Format the problem with character context
    character_context = format_character_context(
        sample.get("character_name", ""),
        sample.get("character_profile", {}),
        sample.get("conversation_history", [])
    )

    # Combine context with question
    problem = f"{character_context}\n\n<Question>\n{sample.get('question', '')}\n</Question>"

    # Create EasyR1 format
    easyr1_sample = {
        "problem": problem,
        "answer": sample.get("answer", ""),
        "images": [],  # RAIDEN-R1 is text-only
        "metadata": {
            # Preserve original metadata
            "character_name": sample.get("character_name", ""),
            "keywords": sample.get("keywords", []),
            "question_type": sample.get("question_type", ""),
            "validation_method": sample.get("validation_method", ""),
            "difficulty": sample.get("difficulty", ""),
            "original_metadata": sample.get("metadata", {}),
            # Add reference to character profile for reward calculation
            "character_profile": sample.get("character_profile", {}),
        }
    }

    return easyr1_sample


def convert_dataset(
    input_file: Path,
    output_file: Path,
    verbose: bool = False
):
    """Convert entire dataset from RAIDEN-R1 to EasyR1 format"""

    print(f"Loading data from {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        raiden_data = json.load(f)

    print(f"Converting {len(raiden_data)} samples...")
    easyr1_data = []

    for i, sample in enumerate(raiden_data):
        try:
            converted = convert_sample(sample)
            easyr1_data.append(converted)

            if verbose and i < 3:  # Show first 3 samples
                print(f"\n=== Sample {i+1} ===")
                print(f"Problem (first 200 chars): {converted['problem'][:200]}...")
                print(f"Answer: {converted['answer']}")

        except Exception as e:
            print(f"Error converting sample {i}: {e}")
            continue

    # Save converted data
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(easyr1_data, f, ensure_ascii=False, indent=2)

    print(f"\nConversion complete!")
    print(f"Converted {len(easyr1_data)}/{len(raiden_data)} samples")
    print(f"Output saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert RAIDEN-R1 dataset to EasyR1 format"
    )
    parser.add_argument(
        "--train_input",
        type=str,
        default="./data/training/train.json",
        help="Path to RAIDEN-R1 training data"
    )
    parser.add_argument(
        "--val_input",
        type=str,
        default="./data/training/validation.json",
        help="Path to RAIDEN-R1 validation data"
    )
    parser.add_argument(
        "--train_output",
        type=str,
        default="./data/easyr1/train.json",
        help="Path to output EasyR1 training data"
    )
    parser.add_argument(
        "--val_output",
        type=str,
        default="./data/easyr1/validation.json",
        help="Path to output EasyR1 validation data"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print sample conversions"
    )

    args = parser.parse_args()

    # Convert training data
    if Path(args.train_input).exists():
        convert_dataset(
            Path(args.train_input),
            Path(args.train_output),
            args.verbose
        )
    else:
        print(f"Warning: Training file {args.train_input} not found")

    # Convert validation data
    if Path(args.val_input).exists():
        convert_dataset(
            Path(args.val_input),
            Path(args.val_output),
            args.verbose
        )
    else:
        print(f"Warning: Validation file {args.val_input} not found")


if __name__ == "__main__":
    main()
