"""
Evaluation Metrics for RAIDEN-R1

Implements:
1. Script-Based Knowledge (SBK)
2. Conversation Memory (CM)
"""

import json
import re
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent.parent))

from rewards.vrar import RoleAwarenessValidator


@dataclass
class EvaluationResult:
    """Result of evaluation"""
    sbk_score: float
    cm_score: float
    overall_score: float
    details: Dict[str, Any]


class ScriptBasedKnowledgeEvaluator:
    """
    Script-Based Knowledge (SBK) Evaluator

    Evaluates whether the model's responses are consistent with
    the character's background, personality, and knowledge from
    the original script/profile.
    """

    def __init__(self, use_llm_judge: bool = False, llm_client=None):
        """
        Args:
            use_llm_judge: Whether to use LLM-as-a-judge (e.g., Claude 3.5)
            llm_client: LLM client for evaluation (if use_llm_judge=True)
        """
        self.use_llm_judge = use_llm_judge
        self.llm_client = llm_client
        self.validator = RoleAwarenessValidator()

    def evaluate(
        self,
        response: str,
        character_profile: Dict[str, Any],
        query: str
    ) -> float:
        """
        Evaluate Script-Based Knowledge

        Args:
            response: Model's generated response
            character_profile: Character's profile information
            query: The query that was asked

        Returns:
            SBK score (0.0 to 1.0)
        """
        if self.use_llm_judge and self.llm_client:
            return self._llm_judge_evaluation(response, character_profile, query)
        else:
            return self._rule_based_evaluation(response, character_profile)

    def _rule_based_evaluation(
        self,
        response: str,
        character_profile: Dict[str, Any]
    ) -> float:
        """Rule-based SBK evaluation"""
        # Use the validator from rewards module
        return self.validator.validate_script_knowledge(response, character_profile)

    def _llm_judge_evaluation(
        self,
        response: str,
        character_profile: Dict[str, Any],
        query: str
    ) -> float:
        """
        LLM-as-a-judge evaluation (e.g., Claude 3.5)

        This would call an LLM to judge the quality of role-playing
        """
        profile_str = json.dumps(character_profile, indent=2)

        judge_prompt = f"""You are evaluating a role-playing conversation for Script-Based Knowledge (SBK).

Character Profile:
{profile_str}

User Query: {query}

Character Response: {response}

Evaluate how well the response demonstrates knowledge consistent with the character's profile.
Consider:
1. Does the response reflect the character's background?
2. Is the personality consistent?
3. Are the character's known skills/knowledge accurately represented?
4. Are there any contradictions with the character's profile?

Provide a score from 0.0 to 1.0, where:
- 1.0 = Perfect consistency with character profile
- 0.8-0.9 = Very good, minor inconsistencies
- 0.6-0.7 = Good, some inconsistencies
- 0.4-0.5 = Fair, notable inconsistencies
- 0.0-0.3 = Poor, major contradictions

Return only a single float number between 0.0 and 1.0."""

        # In actual implementation, this would call the LLM
        # For now, fallback to rule-based
        if self.llm_client:
            try:
                # score = self.llm_client.evaluate(judge_prompt)
                # return float(score)
                pass
            except Exception as e:
                print(f"LLM judge error: {e}")

        return self._rule_based_evaluation(response, character_profile)


class ConversationMemoryEvaluator:
    """
    Conversation Memory (CM) Evaluator

    Evaluates whether the model maintains consistency with
    the conversation history and remembers previous statements.
    """

    def __init__(self, use_llm_judge: bool = False, llm_client=None):
        """
        Args:
            use_llm_judge: Whether to use LLM-as-a-judge
            llm_client: LLM client for evaluation
        """
        self.use_llm_judge = use_llm_judge
        self.llm_client = llm_client
        self.validator = RoleAwarenessValidator()

    def evaluate(
        self,
        response: str,
        conversation_history: List[Dict[str, str]],
        query: str
    ) -> float:
        """
        Evaluate Conversation Memory

        Args:
            response: Model's generated response
            conversation_history: Previous conversation turns
            query: Current query

        Returns:
            CM score (0.0 to 1.0)
        """
        if self.use_llm_judge and self.llm_client:
            return self._llm_judge_evaluation(response, conversation_history, query)
        else:
            return self._rule_based_evaluation(response, conversation_history)

    def _rule_based_evaluation(
        self,
        response: str,
        conversation_history: List[Dict[str, str]]
    ) -> float:
        """Rule-based CM evaluation"""
        return self.validator.validate_conversation_memory(
            response, conversation_history, ""
        )

    def _llm_judge_evaluation(
        self,
        response: str,
        conversation_history: List[Dict[str, str]],
        query: str
    ) -> float:
        """LLM-as-a-judge evaluation for conversation memory"""
        history_str = self._format_conversation_history(conversation_history)

        judge_prompt = f"""You are evaluating a role-playing conversation for Conversation Memory (CM).

Conversation History:
{history_str}

Current Query: {query}

Character Response: {response}

Evaluate how well the response maintains consistency with the conversation history.
Consider:
1. Does the response acknowledge previous statements?
2. Is there consistency with earlier claims or information shared?
3. Does the character remember what was discussed?
4. Are there any contradictions with previous turns?

Provide a score from 0.0 to 1.0, where:
- 1.0 = Perfect memory, fully consistent
- 0.8-0.9 = Very good consistency
- 0.6-0.7 = Good, minor lapses
- 0.4-0.5 = Fair, some inconsistencies
- 0.0-0.3 = Poor, major contradictions or memory failures

Return only a single float number between 0.0 and 1.0."""

        # In actual implementation, this would call the LLM
        if self.llm_client:
            try:
                # score = self.llm_client.evaluate(judge_prompt)
                # return float(score)
                pass
            except Exception as e:
                print(f"LLM judge error: {e}")

        return self._rule_based_evaluation(response, conversation_history)

    def _format_conversation_history(
        self,
        history: List[Dict[str, str]]
    ) -> str:
        """Format conversation history for display"""
        formatted = []
        for i, turn in enumerate(history):
            role = turn.get("role", "unknown")
            content = turn.get("content", "")
            formatted.append(f"{role.capitalize()}: {content}")

        return "\n".join(formatted)


class RAIDEN_R1_Evaluator:
    """
    Complete evaluator for RAIDEN-R1

    Combines SBK and CM metrics
    """

    def __init__(
        self,
        sbk_weight: float = 0.5,
        cm_weight: float = 0.5,
        use_llm_judge: bool = False,
        llm_client=None
    ):
        """
        Args:
            sbk_weight: Weight for SBK score
            cm_weight: Weight for CM score
            use_llm_judge: Whether to use LLM-as-a-judge
            llm_client: LLM client for evaluation
        """
        self.sbk_weight = sbk_weight
        self.cm_weight = cm_weight

        self.sbk_evaluator = ScriptBasedKnowledgeEvaluator(
            use_llm_judge=use_llm_judge,
            llm_client=llm_client
        )

        self.cm_evaluator = ConversationMemoryEvaluator(
            use_llm_judge=use_llm_judge,
            llm_client=llm_client
        )

    def evaluate(
        self,
        response: str,
        character_profile: Dict[str, Any],
        query: str,
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> EvaluationResult:
        """
        Complete evaluation of a response

        Args:
            response: Model's generated response
            character_profile: Character profile
            query: User query
            conversation_history: Optional conversation history

        Returns:
            EvaluationResult with SBK, CM, and overall scores
        """
        # Evaluate SBK
        sbk_score = self.sbk_evaluator.evaluate(
            response, character_profile, query
        )

        # Evaluate CM
        cm_score = 1.0  # Default if no history
        if conversation_history:
            cm_score = self.cm_evaluator.evaluate(
                response, conversation_history, query
            )

        # Calculate overall score
        overall_score = (
            self.sbk_weight * sbk_score +
            self.cm_weight * cm_score
        )

        return EvaluationResult(
            sbk_score=sbk_score,
            cm_score=cm_score,
            overall_score=overall_score,
            details={
                "sbk_weight": self.sbk_weight,
                "cm_weight": self.cm_weight,
                "has_conversation_history": conversation_history is not None
            }
        )

    def evaluate_dataset(
        self,
        dataset: List[Dict[str, Any]],
        output_file: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Evaluate an entire dataset

        Args:
            dataset: List of evaluation samples
            output_file: Optional file to save detailed results

        Returns:
            Dictionary with aggregate metrics
        """
        sbk_scores = []
        cm_scores = []
        overall_scores = []

        results = []

        for i, sample in enumerate(dataset):
            response = sample.get("response", "")
            character_profile = sample.get("character_profile", {})
            query = sample.get("query", "")
            conversation_history = sample.get("conversation_history", None)

            result = self.evaluate(
                response, character_profile, query, conversation_history
            )

            sbk_scores.append(result.sbk_score)
            cm_scores.append(result.cm_score)
            overall_scores.append(result.overall_score)

            results.append({
                "sample_id": i,
                "sbk_score": result.sbk_score,
                "cm_score": result.cm_score,
                "overall_score": result.overall_score,
                **result.details
            })

        # Calculate aggregate metrics
        aggregate = {
            "avg_sbk": sum(sbk_scores) / len(sbk_scores) if sbk_scores else 0.0,
            "avg_cm": sum(cm_scores) / len(cm_scores) if cm_scores else 0.0,
            "avg_overall": sum(overall_scores) / len(overall_scores) if overall_scores else 0.0,
            "num_samples": len(dataset)
        }

        # Save detailed results if requested
        if output_file:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, 'w') as f:
                json.dump({
                    "aggregate": aggregate,
                    "detailed_results": results
                }, f, indent=2)

            print(f"Detailed results saved to {output_file}")

        return aggregate


def print_evaluation_results(result: EvaluationResult):
    """Print evaluation results in a readable format"""
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"Script-Based Knowledge (SBK): {result.sbk_score:.4f}")
    print(f"Conversation Memory (CM):     {result.cm_score:.4f}")
    print(f"Overall Score:                {result.overall_score:.4f}")
    print("="*50 + "\n")
