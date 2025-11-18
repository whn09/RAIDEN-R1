"""
EasyR1-compatible Reward Function for RAIDEN-R1
Based on Verifiable Role-Awareness Reward (VRAR) mechanism

This reward function evaluates:
1. Role-awareness accuracy: Does the response match character knowledge?
2. Format quality: Does the response have proper reasoning structure?
"""

import re
from typing import Any, Dict, List


# Metadata required by EasyR1
REWARD_NAME = "raiden_vrar"
REWARD_TYPE = "batch"  # Process rewards in batch for efficiency


def calculate_role_awareness_reward(
    response: str,
    ground_truth: str,
    keywords: List[str],
    validation_method: str = "single_term_validation"
) -> float:
    """
    Calculate role-awareness accuracy reward

    Args:
        response: Model's generated response
        ground_truth: Expected correct answer
        keywords: Key terms that should appear in response
        validation_method: Validation strategy to use

    Returns:
        Accuracy score between 0.0 and 1.0
    """
    response_lower = response.lower()
    ground_truth_lower = ground_truth.lower()

    # 1. Exact match (highest reward)
    if ground_truth_lower in response_lower:
        return 1.0

    # 2. Keyword matching for multi-term validation
    if validation_method == "multi_term_parsing" and keywords:
        keyword_matches = sum(
            1 for keyword in keywords
            if keyword.lower() in response_lower
        )
        keyword_ratio = keyword_matches / len(keywords) if keywords else 0.0

        # Partial credit based on keyword coverage
        if keyword_ratio >= 0.8:
            return 0.9
        elif keyword_ratio >= 0.6:
            return 0.7
        elif keyword_ratio >= 0.4:
            return 0.5
        elif keyword_ratio >= 0.2:
            return 0.3

    # 3. Semantic similarity check (simple token overlap)
    ground_truth_tokens = set(ground_truth_lower.split())
    response_tokens = set(response_lower.split())

    if ground_truth_tokens and response_tokens:
        overlap = len(ground_truth_tokens & response_tokens)
        overlap_ratio = overlap / len(ground_truth_tokens)

        if overlap_ratio >= 0.6:
            return 0.6
        elif overlap_ratio >= 0.3:
            return 0.3

    return 0.0


def calculate_format_reward(response: str) -> float:
    """
    Calculate format quality reward

    Checks for:
    - Reasoning structure (thinking process)
    - Clear conclusion
    - Proper formatting

    Args:
        response: Model's generated response

    Returns:
        Format score between 0.0 and 1.0
    """
    reward = 0.0

    # 1. Check for reasoning/thinking structure (30%)
    thinking_patterns = [
        r'<think>.*?</think>',  # Explicit thinking tags
        r'(?:let me think|let\'s think|thinking about)',
        r'(?:first|second|third|finally)',
        r'(?:step \d+|步骤\s*\d+)',
        r'(?:because|since|therefore|thus|所以|因为)',
    ]

    has_thinking = any(
        re.search(pattern, response, re.IGNORECASE | re.DOTALL)
        for pattern in thinking_patterns
    )
    if has_thinking:
        reward += 0.3

    # 2. Check for clear structure/organization (30%)
    structure_patterns = [
        r'\d+\.',  # Numbered steps
        r'^\s*[-*]\s',  # Bullet points
        r'\n\n',  # Paragraph breaks
        r'\*\*.+?\*\*',  # Bold text
    ]

    structure_matches = sum(
        1 for pattern in structure_patterns
        if re.search(pattern, response, re.MULTILINE)
    )

    if structure_matches >= 3:
        reward += 0.3
    elif structure_matches >= 2:
        reward += 0.2
    elif structure_matches >= 1:
        reward += 0.1

    # 3. Check for conclusion/answer (40%)
    conclusion_patterns = [
        r'(?:in conclusion|to conclude|in summary|总之|综上)',
        r'(?:the answer is|答案是)',
        r'(?:therefore|thus|所以)',
        r'<answer>.*?</answer>',
        r'\*\*(?:answer|答案)\*\*',
    ]

    has_conclusion = any(
        re.search(pattern, response, re.IGNORECASE | re.DOTALL)
        for pattern in conclusion_patterns
    )
    if has_conclusion:
        reward += 0.4
    elif len(response.strip()) > 10:  # Has some content at least
        reward += 0.2

    return min(reward, 1.0)


def calculate_thinking_quality_reward(response: str) -> float:
    """
    Calculate the quality of reasoning/thinking process

    Rewards responses that show clear step-by-step reasoning
    similar to OpenAI's O1 thinking pattern

    Args:
        response: Model's generated response

    Returns:
        Thinking quality score between 0.0 and 1.0
    """
    reward = 0.0

    # Check for explicit thinking tags
    think_match = re.search(r'<think>(.*?)</think>', response, re.DOTALL)
    if think_match:
        thinking_content = think_match.group(1)

        # Reward longer, more detailed thinking
        thinking_length = len(thinking_content.strip())
        if thinking_length > 200:
            reward += 0.5
        elif thinking_length > 100:
            reward += 0.3
        elif thinking_length > 50:
            reward += 0.2

        # Check for reasoning quality indicators
        quality_indicators = [
            r'(?:analyze|consider|evaluate|分析|考虑)',
            r'(?:if|then|else|如果|那么)',
            r'(?:however|but|although|但是|然而)',
            r'(?:compare|contrast|对比|比较)',
        ]

        quality_count = sum(
            1 for pattern in quality_indicators
            if re.search(pattern, thinking_content, re.IGNORECASE)
        )

        if quality_count >= 3:
            reward += 0.3
        elif quality_count >= 2:
            reward += 0.2
        elif quality_count >= 1:
            reward += 0.1

    return min(reward, 1.0)


def compute_score(
    reward_inputs: List[Dict[str, Any]],
    accuracy_weight: float = 0.5,
    format_weight: float = 0.3,
    thinking_weight: float = 0.2
) -> List[Dict[str, float]]:
    """
    Compute rewards for a batch of responses

    This is the main entry point called by EasyR1

    Args:
        reward_inputs: List of dicts containing:
            - response: Model's generated response
            - response_length: Length of response
            - ground_truth: Expected answer
        accuracy_weight: Weight for role-awareness accuracy (default: 0.5)
        format_weight: Weight for format quality (default: 0.3)
        thinking_weight: Weight for thinking quality (default: 0.2)

    Returns:
        List of score dicts containing:
            - overall: Overall weighted score
            - accuracy: Role-awareness accuracy score
            - format: Format quality score
            - thinking: Thinking quality score
    """
    scores = []

    for reward_input in reward_inputs:
        response = reward_input["response"]
        ground_truth = reward_input["ground_truth"]

        # Extract metadata if available
        # Note: EasyR1 passes ground_truth as string, but we can pass metadata
        # through the dataset's "metadata" field
        keywords = []
        validation_method = "single_term_validation"

        # Try to extract metadata if ground_truth is a dict-like string
        # This is a workaround since EasyR1 converts everything to string
        if isinstance(ground_truth, str):
            # Simple case: ground_truth is just the answer string
            answer = ground_truth
        else:
            # If somehow dict is preserved
            answer = ground_truth.get("answer", str(ground_truth))
            keywords = ground_truth.get("keywords", [])
            validation_method = ground_truth.get("validation_method", "single_term_validation")

        # Calculate individual reward components
        accuracy_score = calculate_role_awareness_reward(
            response, answer, keywords, validation_method
        )
        format_score = calculate_format_reward(response)
        thinking_score = calculate_thinking_quality_reward(response)

        # Calculate weighted overall score
        overall_score = (
            accuracy_weight * accuracy_score +
            format_weight * format_score +
            thinking_weight * thinking_score
        )

        scores.append({
            "overall": overall_score,
            "accuracy": accuracy_score,
            "format": format_score,
            "thinking": thinking_score,
        })

    return scores


# Example usage for testing
if __name__ == "__main__":
    # Test case
    test_inputs = [
        {
            "response": "<think>This character is a doctor who eats lunch on the rooftop. Based on the profile, they started this habit after becoming a doctor to find peace while viewing the distant mountains.</think>\n\nThe answer is: I started eating lunch on the rooftop after becoming a doctor, as that place allows me to calmly gaze at the distant mountains.",
            "response_length": 100,
            "ground_truth": "我从成为医生后开始在屋上吃午餐，那里能让我静心眺望远山。"
        }
    ]

    results = compute_score(test_inputs)

    print("Test Results:")
    for i, result in enumerate(results):
        print(f"\nSample {i+1}:")
        print(f"  Overall: {result['overall']:.3f}")
        print(f"  Accuracy: {result['accuracy']:.3f}")
        print(f"  Format: {result['format']:.3f}")
        print(f"  Thinking: {result['thinking']:.3f}")
