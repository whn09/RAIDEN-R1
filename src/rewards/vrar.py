"""
Verifiable Role-Awareness Reward (VRAR) Mechanism

This module implements the reward calculation for RAIDEN-R1, including:
1. Accuracy Reward: Validates if the model's reasoning leads to correct answers
2. Format Reward: Validates if the output follows the expected format
"""

import re
import json
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass


@dataclass
class RewardResult:
    """Result of reward calculation"""
    total_reward: float
    accuracy_reward: float
    format_reward: float
    details: Dict[str, Any]


class VRARCalculator:
    """
    Verifiable Role-Awareness Reward Calculator

    Calculates rewards based on:
    - Accuracy: Whether the model correctly identifies role-specific information
    - Format: Whether the output follows the expected reasoning format
    """

    def __init__(
        self,
        accuracy_weight: float = 0.7,
        format_weight: float = 0.3,
        use_dynamic_parsing: bool = True
    ):
        """
        Args:
            accuracy_weight: Weight for accuracy reward (default: 0.7)
            format_weight: Weight for format reward (default: 0.3)
            use_dynamic_parsing: Whether to use multi-term dynamic parsing
        """
        self.accuracy_weight = accuracy_weight
        self.format_weight = format_weight
        self.use_dynamic_parsing = use_dynamic_parsing

    def calculate_reward(
        self,
        response: str,
        ground_truth: Dict[str, Any],
        validation_type: str = "single_term"
    ) -> RewardResult:
        """
        Calculate the total reward for a model response

        Args:
            response: Model's generated response
            ground_truth: Dictionary containing correct answers and expected format
            validation_type: "single_term" or "multi_term"

        Returns:
            RewardResult object with reward scores and details
        """
        # Calculate accuracy reward
        if validation_type == "single_term":
            accuracy_reward = self._calculate_single_term_accuracy(
                response, ground_truth
            )
        else:
            accuracy_reward = self._calculate_multi_term_accuracy(
                response, ground_truth
            )

        # Calculate format reward
        format_reward = self._calculate_format_reward(response, ground_truth)

        # Weighted total reward
        total_reward = (
            self.accuracy_weight * accuracy_reward +
            self.format_weight * format_reward
        )

        return RewardResult(
            total_reward=total_reward,
            accuracy_reward=accuracy_reward,
            format_reward=format_reward,
            details={
                "validation_type": validation_type,
                "response_length": len(response),
            }
        )

    def _calculate_single_term_accuracy(
        self,
        response: str,
        ground_truth: Dict[str, Any]
    ) -> float:
        """
        Single-Term Validation Strategy

        Validates if a single answer term is correct
        """
        expected_answer = ground_truth.get("answer", "")

        # Extract the final answer from response
        final_answer = self._extract_final_answer(response)

        # Exact match
        if final_answer.lower().strip() == str(expected_answer).lower().strip():
            return 1.0

        # Partial match (if expected answer is contained in response)
        if str(expected_answer).lower() in final_answer.lower():
            return 0.7

        # Check if key terms are present
        if self._check_key_terms(final_answer, ground_truth.get("key_terms", [])):
            return 0.5

        return 0.0

    def _calculate_multi_term_accuracy(
        self,
        response: str,
        ground_truth: Dict[str, Any]
    ) -> float:
        """
        Multi-Term Dynamic Parsing Strategy

        Dynamically parses and validates multiple terms/attributes
        """
        expected_terms = ground_truth.get("terms", {})

        if not expected_terms:
            # Fallback to single-term if no multi-term data provided
            return self._calculate_single_term_accuracy(response, ground_truth)

        total_terms = len(expected_terms)
        correct_terms = 0

        # Parse response for each expected term
        for term_name, term_value in expected_terms.items():
            if self._validate_term(response, term_name, term_value):
                correct_terms += 1

        return correct_terms / total_terms if total_terms > 0 else 0.0

    def _calculate_format_reward(
        self,
        response: str,
        ground_truth: Dict[str, Any]
    ) -> float:
        """
        Calculate format reward based on expected structure

        Expected format typically includes:
        - Reasoning steps
        - Clear conclusion
        - Proper formatting markers
        """
        reward = 0.0
        max_reward = 1.0

        # Check for reasoning structure
        if self._has_reasoning_structure(response):
            reward += 0.4

        # Check for conclusion marker
        if self._has_conclusion(response):
            reward += 0.3

        # Check for proper formatting (like numbered steps, markers)
        if self._has_proper_formatting(response):
            reward += 0.3

        return min(reward, max_reward)

    def _extract_final_answer(self, response: str) -> str:
        """Extract the final answer from the response"""
        # Look for common answer patterns
        patterns = [
            r"(?:final answer|answer|conclusion):\s*(.+?)(?:\n|$)",
            r"(?:therefore|thus|so),?\s*(?:the answer is)?\s*(.+?)(?:\n|$)",
            r"<answer>(.+?)</answer>",
            r"\*\*Answer:\*\*\s*(.+?)(?:\n|$)",
        ]

        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                return match.group(1).strip()

        # If no pattern found, return last line as potential answer
        lines = response.strip().split('\n')
        return lines[-1].strip() if lines else ""

    def _check_key_terms(self, text: str, key_terms: List[str]) -> bool:
        """Check if key terms are present in text"""
        if not key_terms:
            return False

        text_lower = text.lower()
        matches = sum(1 for term in key_terms if term.lower() in text_lower)
        return matches >= len(key_terms) * 0.5  # At least 50% of key terms

    def _validate_term(self, response: str, term_name: str, term_value: Any) -> bool:
        """Validate if a specific term/attribute is correctly mentioned"""
        response_lower = response.lower()
        term_name_lower = term_name.lower()
        term_value_lower = str(term_value).lower()

        # Check if both term name and value are mentioned
        if term_name_lower in response_lower and term_value_lower in response_lower:
            return True

        # Check if at least the value is correct
        if term_value_lower in response_lower:
            return True

        return False

    def _has_reasoning_structure(self, response: str) -> bool:
        """Check if response has clear reasoning structure"""
        # Look for step markers, numbered lists, or reasoning keywords
        patterns = [
            r'\d+\.',  # Numbered steps
            r'(?:first|second|third|finally)',  # Step keywords
            r'(?:because|since|therefore|thus)',  # Reasoning keywords
            r'step \d+',  # Explicit step markers
        ]

        matches = sum(1 for p in patterns if re.search(p, response, re.IGNORECASE))
        return matches >= 2

    def _has_conclusion(self, response: str) -> bool:
        """Check if response has a clear conclusion"""
        conclusion_patterns = [
            r'(?:in conclusion|to conclude|therefore|thus|finally)',
            r'(?:the answer is|the result is)',
            r'<answer>|</answer>',
            r'\*\*(?:answer|conclusion)\*\*',
        ]

        return any(re.search(p, response, re.IGNORECASE) for p in conclusion_patterns)

    def _has_proper_formatting(self, response: str) -> bool:
        """Check if response has proper formatting"""
        # Check for markdown formatting, structure, or clear organization
        formatting_indicators = [
            r'\*\*.+\*\*',  # Bold text
            r'\n\n',  # Paragraph breaks
            r'^\s*[-*]\s',  # Bullet points
            r'^\s*\d+\.\s',  # Numbered lists
        ]

        matches = sum(1 for p in formatting_indicators if re.search(p, response, re.MULTILINE))
        return matches >= 2


class RoleAwarenessValidator:
    """
    Validates role-awareness in model responses

    Checks if the model maintains consistency with:
    - Script-based knowledge (SBK)
    - Conversation memory (CM)
    """

    def __init__(self):
        self.vrar_calculator = VRARCalculator()

    def validate_script_knowledge(
        self,
        response: str,
        character_profile: Dict[str, Any]
    ) -> float:
        """
        Validate Script-Based Knowledge (SBK)

        Checks if response is consistent with character's background,
        personality, and knowledge from the original script
        """
        profile_terms = self._extract_profile_terms(character_profile)

        # Count how many profile elements are correctly reflected
        correct_count = 0
        total_count = len(profile_terms)

        for term_name, term_value in profile_terms.items():
            if self.vrar_calculator._validate_term(response, term_name, term_value):
                correct_count += 1

        return correct_count / total_count if total_count > 0 else 0.0

    def validate_conversation_memory(
        self,
        response: str,
        conversation_history: List[Dict[str, str]],
        current_query: str
    ) -> float:
        """
        Validate Conversation Memory (CM)

        Checks if response maintains consistency with previous conversation
        """
        # Extract key information from conversation history
        memory_points = self._extract_memory_points(conversation_history)

        if not memory_points:
            return 1.0  # No prior context to validate against

        # Check consistency with each memory point
        consistent_count = 0
        for point in memory_points:
            if self._is_consistent_with_memory(response, point):
                consistent_count += 1

        return consistent_count / len(memory_points)

    def _extract_profile_terms(self, profile: Dict[str, Any]) -> Dict[str, Any]:
        """Extract key terms from character profile"""
        terms = {}

        # Common profile fields
        for key in ["name", "occupation", "personality", "background", "skills"]:
            if key in profile:
                terms[key] = profile[key]

        return terms

    def _extract_memory_points(
        self,
        conversation_history: List[Dict[str, str]]
    ) -> List[Dict[str, str]]:
        """Extract key points from conversation history"""
        memory_points = []

        for turn in conversation_history:
            # Extract statements, claims, or important information
            if "assistant" in turn.get("role", ""):
                content = turn.get("content", "")
                # Simple extraction: sentences with key indicators
                sentences = content.split('.')
                for sent in sentences:
                    if any(kw in sent.lower() for kw in ["i am", "i have", "my", "i can"]):
                        memory_points.append({"content": sent.strip()})

        return memory_points

    def _is_consistent_with_memory(self, response: str, memory_point: Dict[str, str]) -> bool:
        """Check if response is consistent with a memory point"""
        memory_content = memory_point.get("content", "").lower()
        response_lower = response.lower()

        # Simple consistency check: key terms from memory appear in response
        # or response doesn't contradict memory

        # Extract key terms from memory
        words = memory_content.split()
        key_words = [w for w in words if len(w) > 3]  # Simple filter

        if not key_words:
            return True

        # Check if some key words appear (loose consistency check)
        matches = sum(1 for w in key_words if w in response_lower)
        return matches >= len(key_words) * 0.3  # At least 30% match
