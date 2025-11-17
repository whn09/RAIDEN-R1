"""
OpenR1 Adapter for RAIDEN-R1

Adapts RAIDEN's data format and VRAR rewards to OpenR1's GRPO framework
"""

import json
from typing import Dict, List, Any, Optional, Callable
from pathlib import Path
from datasets import Dataset
import torch

import sys
sys.path.append(str(Path(__file__).parent.parent))

from rewards.vrar import VRARCalculator, RoleAwarenessValidator
from data.collection import RolePlayingSample


class RAIDENDataAdapter:
    """
    Adapts RAIDEN data format to OpenR1's expected format
    """

    def __init__(self, system_prompt: Optional[str] = None):
        """
        Args:
            system_prompt: Optional system prompt for role-playing
        """
        self.system_prompt = system_prompt or (
            "You are a role-playing AI assistant. "
            "Answer questions in character based on the provided character profile."
        )

    def convert_sample_to_openr1(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert a RAIDEN sample to OpenR1 format

        Args:
            sample: RAIDEN sample dictionary

        Returns:
            OpenR1-compatible sample
        """
        character_name = sample.get("character_name", "Unknown")
        character_profile = sample.get("character_profile", {})
        question = sample.get("question", "")
        answer = sample.get("answer", "")
        keywords = sample.get("keywords", [])

        # Build character context
        character_context = self._build_character_context(character_name, character_profile)

        # Create prompt with character context
        prompt = f"{character_context}\n\nUser: {question}"

        # Build conversation for training
        conversation = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
        ]

        # For GRPO, we need the expected answer and validation info
        return {
            "prompt": conversation,
            "expected_answer": answer,
            "keywords": keywords,
            "character_name": character_name,
            "character_profile": character_profile,
            "question_type": sample.get("question_type", "what"),
            "validation_method": sample.get("validation_method", "single_term_validation"),
            "difficulty": sample.get("difficulty", "medium"),
            "metadata": sample.get("metadata", {})
        }

    def _build_character_context(
        self,
        character_name: str,
        character_profile: Dict[str, Any]
    ) -> str:
        """Build character context string"""
        context_parts = [f"Character: {character_name}"]

        if "Gender" in character_profile:
            context_parts.append(f"Gender: {character_profile['Gender']}")

        if "Introduction" in character_profile:
            context_parts.append(f"Introduction: {character_profile['Introduction']}")

        if "Detailed Description" in character_profile:
            context_parts.append(f"Description: {character_profile['Detailed Description']}")

        return "\n".join(context_parts)

    def load_raiden_dataset(self, json_path: str) -> Dataset:
        """
        Load RAIDEN dataset and convert to OpenR1 format

        Args:
            json_path: Path to RAIDEN JSON dataset

        Returns:
            HuggingFace Dataset object
        """
        # Load RAIDEN data
        with open(json_path, 'r', encoding='utf-8') as f:
            raiden_data = json.load(f)

        # Convert each sample
        openr1_data = [
            self.convert_sample_to_openr1(sample)
            for sample in raiden_data
        ]

        # Create HuggingFace Dataset
        return Dataset.from_list(openr1_data)


class RAIDENRewardFunction:
    """
    VRAR reward function compatible with OpenR1's GRPO
    """

    def __init__(
        self,
        tokenizer,
        accuracy_weight: float = 0.7,
        format_weight: float = 0.3,
        dataset_metadata: Dict[str, Any] = None
    ):
        """
        Args:
            tokenizer: Model tokenizer
            accuracy_weight: Weight for accuracy reward
            format_weight: Weight for format reward
            dataset_metadata: Mapping from prompt to metadata (keywords, validation_method, etc)
        """
        self.tokenizer = tokenizer
        self.vrar_calculator = VRARCalculator(
            accuracy_weight=accuracy_weight,
            format_weight=format_weight
        )
        self.validator = RoleAwarenessValidator()
        self.dataset_metadata = dataset_metadata or {}
        # Add __name__ attribute required by TRL's GRPOTrainer
        self.__name__ = "RAIDEN_VRAR"

    def __call__(
        self,
        prompts: List[Any],
        completions: List[str],
        **kwargs
    ) -> List[float]:
        """
        Calculate VRAR rewards for completions

        Args:
            prompts: List of prompts (can be dicts with metadata or conversation lists)
            completions: List of model completions

        Returns:
            List of reward scores
        """
        rewards = []

        for i, (prompt_data, completion) in enumerate(zip(prompts, completions)):
            try:
                # Handle different prompt formats
                if isinstance(prompt_data, dict):
                    # If it's a dict, extract metadata directly
                    keywords = prompt_data.get("keywords", [])
                    validation_method = prompt_data.get("validation_method", "single_term_validation")
                    expected_answer = prompt_data.get("expected_answer", "")
                elif isinstance(prompt_data, list):
                    # If it's a conversation list, try to find metadata using the prompt as key
                    prompt_key = str(prompt_data)
                    if prompt_key in self.dataset_metadata:
                        metadata = self.dataset_metadata[prompt_key]
                        keywords = metadata.get("keywords", [])
                        validation_method = metadata.get("validation_method", "single_term_validation")
                        expected_answer = metadata.get("expected_answer", "")
                    else:
                        # Fallback: use default values
                        if i == 0:  # Only print warning once
                            print(f"Warning: Could not find metadata for prompt, using defaults")
                        keywords = []
                        validation_method = "single_term_validation"
                        expected_answer = ""
                else:
                    print(f"Warning: Unknown prompt format type: {type(prompt_data)}")
                    keywords = []
                    validation_method = "single_term_validation"
                    expected_answer = ""

                # If no keywords available, give a default positive reward for non-empty completions
                if not keywords:
                    reward = 1.0 if len(completion.strip()) > 10 else 0.5
                    rewards.append(reward)
                    continue

                # Validate answer
                validation_result = self.validator.validate_answer(
                    response=completion,
                    keywords=keywords,
                    validation_method=validation_method
                )

                # Calculate VRAR reward
                reward = self.vrar_calculator.calculate_reward(
                    response=completion,
                    keywords=keywords,
                    validation_result=validation_result
                )

                rewards.append(reward)

            except Exception as e:
                print(f"Error calculating reward for sample {i}: {e}")
                import traceback
                traceback.print_exc()
                # Return neutral reward on error
                rewards.append(0.0)

        return rewards


def create_raiden_reward_function(
    tokenizer,
    accuracy_weight: float = 0.7,
    format_weight: float = 0.3,
    dataset_metadata: Dict[str, Any] = None
) -> Callable:
    """
    Factory function to create RAIDEN reward function for OpenR1

    Args:
        tokenizer: Model tokenizer
        accuracy_weight: Weight for accuracy reward
        format_weight: Weight for format reward
        dataset_metadata: Mapping from prompt to metadata

    Returns:
        Reward function compatible with OpenR1
    """
    reward_fn = RAIDENRewardFunction(
        tokenizer=tokenizer,
        accuracy_weight=accuracy_weight,
        format_weight=format_weight,
        dataset_metadata=dataset_metadata
    )

    return reward_fn
