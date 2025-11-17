"""
Dataset Collection and Management for RAIDEN-R1

This module provides classes for managing role-playing training samples
and building datasets for GRPO training.
"""

import json
import random
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from pathlib import Path
import time


@dataclass
class RolePlayingSample:
    """
    A single role-playing training sample for RAIDEN-R1

    Attributes:
        character_name: Name of the character
        character_profile: Full character profile dictionary
        conversation_history: List of previous conversation turns
        question: The question to be answered
        answer: The expected answer (with reasoning)
        keywords: List of keywords for validation
        question_type: Type of question (what, who, where, when, why, how)
        validation_method: "single_term_validation" or "multi_term_parsing"
        difficulty: "easy", "medium", or "hard"
        metadata: Additional metadata (reasoning, focus, etc.)
    """
    character_name: str
    character_profile: Dict[str, Any]
    conversation_history: List[Dict[str, str]]
    question: str
    answer: str
    keywords: List[str]
    question_type: str
    validation_method: str
    difficulty: str
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert sample to dictionary"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RolePlayingSample':
        """Create sample from dictionary"""
        return cls(**data)

    def to_training_format(self) -> Dict[str, Any]:
        """Convert to training format for GRPO"""
        return {
            "character_name": self.character_name,
            "character_profile": self.character_profile,
            "conversation_history": self.conversation_history,
            "question": self.question,
            "answer": self.answer,
            "keywords": self.keywords,
            "question_type": self.question_type,
            "validation_method": self.validation_method,
            "difficulty": self.difficulty,
            "metadata": self.metadata
        }


class DatasetBuilder:
    """
    Builder for RAIDEN-R1 training datasets

    Supports:
    - Loading from RAIDEN Benchmark
    - Loading from generated samples
    - Generating challenging samples
    - Splitting into train/validation sets
    - Export for training
    """

    def __init__(self):
        self.samples: List[RolePlayingSample] = []

    def add_sample(self, sample: RolePlayingSample) -> None:
        """Add a single sample to the dataset"""
        self.samples.append(sample)

    def load_from_json(self, json_path: str) -> None:
        """
        Load samples from a JSON file

        Args:
            json_path: Path to JSON file containing samples
        """
        print(f"Loading samples from: {json_path}")

        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Handle both list and single object formats
        if isinstance(data, dict):
            data = [data]

        for item in data:
            sample = RolePlayingSample.from_dict(item)
            self.add_sample(sample)

        print(f"Loaded {len(data)} samples")

    def load_from_raiden_benchmark(self, benchmark_path: str) -> None:
        """
        Load samples from RAIDEN Benchmark dataset

        Args:
            benchmark_path: Path to RAIDEN Benchmark JSON file
        """
        print(f"Loading RAIDEN Benchmark from: {benchmark_path}")

        with open(benchmark_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Convert benchmark format to RolePlayingSample format
        for item in data:
            sample = self._convert_benchmark_to_sample(item)
            self.add_sample(sample)

        print(f"Loaded {len(data)} samples from benchmark")

    def _convert_benchmark_to_sample(self, benchmark_item: Dict[str, Any]) -> RolePlayingSample:
        """Convert benchmark format to RolePlayingSample"""
        return RolePlayingSample(
            character_name=benchmark_item.get("character_name", "Unknown"),
            character_profile=benchmark_item.get("character_profile", {}),
            conversation_history=benchmark_item.get("conversation_history", []),
            question=benchmark_item.get("question", ""),
            answer=benchmark_item.get("answer", ""),
            keywords=benchmark_item.get("keywords", []),
            question_type=benchmark_item.get("question_type", "what"),
            validation_method=benchmark_item.get("validation_method", "single_term_validation"),
            difficulty=benchmark_item.get("difficulty", "medium"),
            metadata={
                "source": "raiden_benchmark",
                **benchmark_item.get("metadata", {})
            }
        )

    def generate_challenging_samples(
        self,
        num_samples: int = 1000,
        characters: Optional[List[Dict[str, Any]]] = None
    ) -> None:
        """
        Generate challenging role-playing samples

        Args:
            num_samples: Number of samples to generate
            characters: Optional list of character profiles to use
        """
        print(f"Generating {num_samples} challenging samples...")

        # Use default characters if not provided
        if characters is None:
            characters = self._get_default_characters()

        question_types = ["what", "who", "where", "when", "why", "how"]
        difficulties = ["easy", "medium", "hard"]

        for i in range(num_samples):
            # Select random character
            character = random.choice(characters)

            # Generate sample parameters
            question_type = random.choice(question_types)
            difficulty = random.choice(difficulties)

            # Create sample
            sample = self._generate_sample(
                character=character,
                question_type=question_type,
                difficulty=difficulty
            )

            self.add_sample(sample)

            if (i + 1) % 100 == 0:
                print(f"  Generated {i + 1}/{num_samples} samples")

    def _get_default_characters(self) -> List[Dict[str, Any]]:
        """Get default character profiles"""
        return [
            {
                "name": "勇敢的骑士",
                "profile": {
                    "occupation": "骑士",
                    "personality": "勇敢、正直、忠诚",
                    "background": "来自北方王国的骑士，一生致力于保护弱小",
                    "skills": ["剑术", "骑术", "领导力"]
                }
            },
            {
                "name": "智慧的法师",
                "profile": {
                    "occupation": "魔法师",
                    "personality": "聪明、谨慎、好学",
                    "background": "在魔法学院学习多年，掌握多种元素魔法",
                    "skills": ["火焰魔法", "冰霜魔法", "占卜术"]
                }
            },
            {
                "name": "狡猾的盗贼",
                "profile": {
                    "occupation": "盗贼",
                    "personality": "机智、灵活、神秘",
                    "background": "在城市的阴影中长大，精通潜行和开锁",
                    "skills": ["潜行", "开锁", "飞刀术"]
                }
            }
        ]

    def _generate_sample(
        self,
        character: Dict[str, Any],
        question_type: str,
        difficulty: str
    ) -> RolePlayingSample:
        """Generate a single sample"""
        name = character.get("name", "Unknown")
        profile = character.get("profile", {})

        # Generate question based on type
        question = self._generate_question(profile, question_type)

        # Generate answer
        answer = self._generate_answer(profile, question, question_type)

        # Extract keywords
        keywords = self._extract_keywords(answer)

        # Determine validation method
        validation_method = "multi_term_parsing" if len(keywords) > 1 else "single_term_validation"

        return RolePlayingSample(
            character_name=name,
            character_profile=profile,
            conversation_history=[],
            question=question,
            answer=answer,
            keywords=keywords,
            question_type=question_type,
            validation_method=validation_method,
            difficulty=difficulty,
            metadata={
                "source": "generated",
                "generated_at": time.time()
            }
        )

    def _generate_question(self, profile: Dict[str, Any], question_type: str) -> str:
        """Generate a question based on profile and type"""
        questions = {
            "what": [
                f"What is your occupation?",
                f"What are your main skills?",
                f"What is your personality like?"
            ],
            "who": [
                f"Who are you?",
                f"Who taught you your skills?"
            ],
            "where": [
                f"Where did you grow up?",
                f"Where did you learn your skills?"
            ],
            "when": [
                f"When did you start your journey?",
                f"When did you master your skills?"
            ],
            "why": [
                f"Why did you choose this path?",
                f"Why is your personality like this?"
            ],
            "how": [
                f"How did you develop your skills?",
                f"How do you use your abilities?"
            ]
        }

        return random.choice(questions.get(question_type, questions["what"]))

    def _generate_answer(self, profile: Dict[str, Any], question: str, question_type: str) -> str:
        """Generate an answer based on profile"""
        occupation = profile.get("occupation", "unknown")
        personality = profile.get("personality", "unknown")
        background = profile.get("background", "unknown")
        skills = profile.get("skills", [])

        if "occupation" in question.lower():
            return f"I am a {occupation}. {background}"
        elif "skill" in question.lower():
            skills_str = ", ".join(skills) if skills else "various skills"
            return f"My main skills include {skills_str}."
        elif "personality" in question.lower():
            return f"My personality is {personality}."
        else:
            return f"As a {occupation}, {background} My personality is {personality}."

    def _extract_keywords(self, answer: str) -> List[str]:
        """Extract keywords from answer"""
        # Simple keyword extraction (in production, use more sophisticated methods)
        words = answer.split()
        keywords = [w.strip('.,!?') for w in words if len(w) > 4 and w.isalpha()]
        return keywords[:3] if keywords else []

    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics"""
        if not self.samples:
            return {
                "total_samples": 0,
                "single_term_samples": 0,
                "multi_term_samples": 0,
                "difficulty_distribution": {},
                "question_types": {},
                "sources": {}
            }

        stats = {
            "total_samples": len(self.samples),
            "single_term_samples": sum(1 for s in self.samples if s.validation_method == "single_term_validation"),
            "multi_term_samples": sum(1 for s in self.samples if s.validation_method == "multi_term_parsing"),
            "difficulty_distribution": {},
            "question_types": {},
            "sources": {}
        }

        # Count difficulties
        for sample in self.samples:
            diff = sample.difficulty
            stats["difficulty_distribution"][diff] = stats["difficulty_distribution"].get(diff, 0) + 1

        # Count question types
        for sample in self.samples:
            qtype = sample.question_type
            stats["question_types"][qtype] = stats["question_types"].get(qtype, 0) + 1

        # Count sources
        for sample in self.samples:
            source = sample.metadata.get("source", "unknown")
            stats["sources"][source] = stats["sources"].get(source, 0) + 1

        return stats

    def save_dataset(self, output_path: str) -> None:
        """
        Save dataset to JSON file

        Args:
            output_path: Path to save the dataset
        """
        data = [sample.to_dict() for sample in self.samples]

        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        print(f"Saved {len(data)} samples to: {output_path}")

    def export_for_training(
        self,
        output_dir: str,
        train_ratio: float = 0.9
    ) -> None:
        """
        Export dataset for training with train/validation split

        Args:
            output_dir: Directory to save train and validation sets
            train_ratio: Ratio of training data (default: 0.9)
        """
        print(f"Exporting dataset to: {output_dir}")

        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Shuffle samples
        shuffled_samples = self.samples.copy()
        random.shuffle(shuffled_samples)

        # Split into train/validation
        split_idx = int(len(shuffled_samples) * train_ratio)
        train_samples = shuffled_samples[:split_idx]
        val_samples = shuffled_samples[split_idx:]

        # Save train set
        train_data = [sample.to_training_format() for sample in train_samples]
        train_file = output_path / "train.json"
        with open(train_file, 'w', encoding='utf-8') as f:
            json.dump(train_data, f, ensure_ascii=False, indent=2)
        print(f"  Saved {len(train_data)} training samples to: {train_file}")

        # Save validation set
        val_data = [sample.to_training_format() for sample in val_samples]
        val_file = output_path / "validation.json"
        with open(val_file, 'w', encoding='utf-8') as f:
            json.dump(val_data, f, ensure_ascii=False, indent=2)
        print(f"  Saved {len(val_data)} validation samples to: {val_file}")

        # Save statistics
        stats = self.get_statistics()
        stats_file = output_path / "dataset_stats.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        print(f"  Saved statistics to: {stats_file}")

    def merge_with(self, other: 'DatasetBuilder') -> None:
        """Merge another dataset into this one"""
        self.samples.extend(other.samples)
        print(f"Merged datasets. Total samples: {len(self.samples)}")

    def filter_by_difficulty(self, difficulty: str) -> 'DatasetBuilder':
        """Create a new dataset with only specified difficulty"""
        filtered = DatasetBuilder()
        for sample in self.samples:
            if sample.difficulty == difficulty:
                filtered.add_sample(sample)
        return filtered

    def filter_by_question_type(self, question_type: str) -> 'DatasetBuilder':
        """Create a new dataset with only specified question type"""
        filtered = DatasetBuilder()
        for sample in self.samples:
            if sample.question_type == question_type:
                filtered.add_sample(sample)
        return filtered
