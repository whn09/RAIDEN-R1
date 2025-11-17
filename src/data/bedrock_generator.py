"""
AWS Bedrock Data Generator for RAIDEN-R1

Generates role-playing training data using AWS Bedrock Claude 3.5 Sonnet.
Implements SBK (Script-Based Knowledge) and CM (Conversation Memory) generation strategies.
"""

import json
import time
import random
import re
from typing import Dict, List, Any, Optional
import boto3
from botocore.exceptions import ClientError

from .collection import RolePlayingSample
from .language_utils import (
    LanguageDetector,
    PromptTemplates,
    translate_question_type,
    translate_focus_area,
    translate_difficulty,
    extract_question_from_response,
    extract_answer_from_response
)


class BedrockDataGenerator:
    """
    Generate RAIDEN training data using AWS Bedrock Claude
    """

    # Question types based on RAIDEN paper
    WH_QUESTION_TYPES = ["what", "who", "where", "when", "why", "how"]

    # Difficulty levels
    DIFFICULTY_LEVELS = ["easy", "medium", "hard"]

    def __init__(
        self,
        region_name: str = "us-east-1",
        model_id: str = "us.anthropic.claude-3-5-sonnet-20241022-v2:0",
        language: str = "zh",
        auto_detect_language: bool = True,
        enable_thinking: bool = False
    ):
        """
        Initialize Bedrock data generator

        Args:
            region_name: AWS region for Bedrock
            model_id: Claude model ID
            language: Default language for generation
            auto_detect_language: Auto-detect language from profiles
            enable_thinking: Enable thinking mode (mainly for consistency with SGLang)
        """
        self.region_name = region_name
        self.model_id = model_id
        self.default_language = language
        self.auto_detect_language = auto_detect_language
        self.enable_thinking = enable_thinking

        # Initialize Bedrock client
        self.bedrock = boto3.client(
            service_name='bedrock-runtime',
            region_name=region_name
        )

        print(f"Initialized Bedrock generator with model: {model_id}")
        print(f"Default language: {language}, Auto-detect: {auto_detect_language}")
        print(f"Enable thinking: {enable_thinking}")

    def generate_dataset_from_profiles(
        self,
        profiles_file: str,
        output_file: str,
        num_samples_per_profile: int = 2,
        include_conversation_memory: bool = False,
        max_profiles: Optional[int] = None
    ) -> None:
        """
        Generate dataset from character profiles file

        Args:
            profiles_file: Path to JSONL file with character profiles
            output_file: Path to save generated samples
            num_samples_per_profile: Number of SBK samples per profile
            include_conversation_memory: Whether to generate CM samples
            max_profiles: Maximum number of profiles to process (for testing)
        """
        print(f"\nGenerating dataset from profiles: {profiles_file}")
        print(f"Samples per profile (SBK): {num_samples_per_profile}")
        print(f"Include CM samples: {include_conversation_memory}")

        # Load character profiles
        profiles = self._load_profiles(profiles_file, max_profiles)
        print(f"Loaded {len(profiles)} character profiles")

        all_samples = []
        total_profiles = len(profiles)

        for idx, profile in enumerate(profiles, 1):
            print(f"\n[{idx}/{total_profiles}] Processing profile: {profile.get('character_profile', {}).get('Name', 'Unknown')}")

            try:
                # Generate SBK samples
                for i in range(num_samples_per_profile):
                    print(f"  Generating SBK sample {i+1}/{num_samples_per_profile}...")
                    sample = self.generate_sbk_sample(profile)
                    if sample:
                        all_samples.append(sample.to_dict())
                        time.sleep(0.5)  # Rate limiting

                # Generate CM sample if requested
                if include_conversation_memory:
                    print(f"  Generating CM sample...")
                    cm_sample = self.generate_cm_sample(profile)
                    if cm_sample:
                        all_samples.append(cm_sample.to_dict())
                        time.sleep(0.5)

            except Exception as e:
                print(f"  Error processing profile: {e}")
                continue

        # Save all samples
        print(f"\n\nSaving {len(all_samples)} samples to: {output_file}")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_samples, f, ensure_ascii=False, indent=2)

        print("Dataset generation completed!")

    def generate_sbk_sample(
        self,
        profile: Dict[str, Any],
        difficulty: Optional[str] = None
    ) -> Optional[RolePlayingSample]:
        """
        Generate Script-Based Knowledge sample

        Tests model's understanding of character background, personality, etc.

        Args:
            profile: Character profile dictionary
            difficulty: Difficulty level (random if None)

        Returns:
            RolePlayingSample or None if generation failed
        """
        # Detect language
        language = self._detect_language(profile)

        # Random difficulty if not specified
        if difficulty is None:
            difficulty = random.choice(self.DIFFICULTY_LEVELS)

        # Random question type (SBK focuses on what, who, where, when)
        question_type = random.choice(["what", "who", "where", "when"])

        # Extract character info
        character_name = profile.get('character_profile', {}).get('Name', 'Unknown')
        character_profile = profile.get('character_profile', {})

        try:
            # Generate question
            question = self._generate_question(
                character_profile=character_profile,
                conversation_history=[],
                question_type=question_type,
                focus="script_knowledge",
                difficulty=difficulty,
                language=language
            )

            # Generate answer with CoT reasoning
            answer, reasoning = self._generate_answer(
                character_profile=character_profile,
                conversation_history=[],
                question=question,
                language=language
            )

            # Extract keywords
            keywords = self._extract_keywords(answer, language)

            # Determine validation method
            validation_method = "multi_term_parsing" if len(keywords) > 1 else "single_term_validation"

            return RolePlayingSample(
                character_name=character_name,
                character_profile=character_profile,
                conversation_history=[],
                question=question,
                answer=answer,
                keywords=keywords,
                question_type=question_type,
                validation_method=validation_method,
                difficulty=difficulty,
                metadata={
                    "source": "bedrock_sbk",
                    "model": self.model_id,
                    "language": language,
                    "reasoning": reasoning,
                    "focus": "script_knowledge",
                    "generated_at": time.time()
                }
            )

        except Exception as e:
            print(f"    Error generating SBK sample: {e}")
            return None

    def generate_cm_sample(
        self,
        profile: Dict[str, Any],
        difficulty: Optional[str] = None
    ) -> Optional[RolePlayingSample]:
        """
        Generate Conversation Memory sample

        Tests model's memory of conversation context

        Args:
            profile: Character profile dictionary
            difficulty: Difficulty level (random if None)

        Returns:
            RolePlayingSample or None if generation failed
        """
        # Detect language
        language = self._detect_language(profile)

        # Random difficulty if not specified
        if difficulty is None:
            difficulty = random.choice(["medium", "hard"])  # CM is typically harder

        # CM focuses on why and how
        question_type = random.choice(["why", "how"])

        # Extract character info
        character_name = profile.get('character_profile', {}).get('Name', 'Unknown')
        character_profile = profile.get('character_profile', {})

        try:
            # Generate simulated conversation history
            conversation_history = self._generate_conversation_history(
                character_profile=character_profile,
                language=language
            )

            # Generate question about conversation
            question = self._generate_question(
                character_profile=character_profile,
                conversation_history=conversation_history,
                question_type=question_type,
                focus="conversation_memory",
                difficulty=difficulty,
                language=language
            )

            # Generate answer with CoT reasoning
            answer, reasoning = self._generate_answer(
                character_profile=character_profile,
                conversation_history=conversation_history,
                question=question,
                language=language
            )

            # Extract keywords
            keywords = self._extract_keywords(answer, language)

            # CM typically uses multi-term parsing
            validation_method = "multi_term_parsing"

            return RolePlayingSample(
                character_name=character_name,
                character_profile=character_profile,
                conversation_history=conversation_history,
                question=question,
                answer=answer,
                keywords=keywords,
                question_type=question_type,
                validation_method=validation_method,
                difficulty=difficulty,
                metadata={
                    "source": "bedrock_cm",
                    "model": self.model_id,
                    "language": language,
                    "reasoning": reasoning,
                    "focus": "conversation_memory",
                    "generated_at": time.time()
                }
            )

        except Exception as e:
            print(f"    Error generating CM sample: {e}")
            return None

    def _generate_question(
        self,
        character_profile: Dict[str, Any],
        conversation_history: List[Dict[str, str]],
        question_type: str,
        focus: str,
        difficulty: str,
        language: str
    ) -> str:
        """Generate a question using Claude"""
        # Format character info
        character_info = json.dumps(character_profile, ensure_ascii=False, indent=2)

        # Format conversation history
        conv_text = self._format_conversation_history(conversation_history)

        # Translate parameters to target language
        question_type_translated = translate_question_type(question_type, language)
        focus_translated = translate_focus_area(focus, language)
        difficulty_translated = translate_difficulty(difficulty, language)

        # Get prompt
        prompt = PromptTemplates.get_question_prompt(
            language=language,
            character_info=character_info,
            conversation_history=conv_text,
            question_type=question_type_translated,
            focus=focus_translated,
            difficulty=difficulty_translated
        )

        # Call Claude
        response = self._call_claude(prompt)

        # Extract actual question from response (filter out thinking)
        question = extract_question_from_response(response, language)

        return question

    def _generate_answer(
        self,
        character_profile: Dict[str, Any],
        conversation_history: List[Dict[str, str]],
        question: str,
        language: str
    ) -> tuple[str, str]:
        """
        Generate concise answer based on character information

        Returns:
            (answer, reasoning) tuple - reasoning is empty for SBK
        """
        # Format character info
        character_info = json.dumps(character_profile, ensure_ascii=False, indent=2)

        # Get prompt
        prompt = PromptTemplates.get_answer_prompt(
            language=language,
            character_info=character_info,
            question=question
        )

        # Call Claude
        response = self._call_claude(prompt, max_tokens=300)

        # Extract actual answer from response (filter out thinking)
        answer = extract_answer_from_response(response, language)

        return answer, ""

    def _extract_keywords(self, answer: str, language: str) -> List[str]:
        """Extract keywords from answer"""
        prompt = PromptTemplates.get_keyword_prompt(
            language=language,
            answer=answer
        )

        try:
            response = self._call_claude(prompt)

            # Parse JSON response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group(0))
                keywords = data.get("keywords", [])
                return keywords[:3]  # Max 3 keywords

        except Exception as e:
            print(f"      Error extracting keywords: {e}")

        # Fallback: simple extraction
        words = answer.split()
        return [w.strip('.,!?;:()[]{}') for w in words if len(w) > 3][:3]

    def _generate_conversation_history(
        self,
        character_profile: Dict[str, Any],
        language: str,
        num_turns: int = 3
    ) -> List[Dict[str, str]]:
        """Generate simulated conversation history"""
        # Simple simulated history for CM testing
        history = []

        character_name = character_profile.get('Name', 'Character')

        # Language-specific templates
        templates = {
            'zh': [
                {"role": "user", "content": f"你好，{character_name}！"},
                {"role": "assistant", "content": f"你好！我是{character_name}。"},
            ],
            'ja': [
                {"role": "user", "content": f"こんにちは、{character_name}さん！"},
                {"role": "assistant", "content": f"こんにちは！私は{character_name}です。"},
            ],
            'en': [
                {"role": "user", "content": f"Hello, {character_name}!"},
                {"role": "assistant", "content": f"Hello! I'm {character_name}."},
            ],
            'ko': [
                {"role": "user", "content": f"안녕하세요, {character_name}님!"},
                {"role": "assistant", "content": f"안녕하세요! 저는 {character_name}입니다."},
            ]
        }

        history = templates.get(language, templates['zh'])

        return history[:num_turns]

    def _call_claude(self, prompt: str, max_tokens: int = 2000) -> str:
        """Call Claude via Bedrock"""
        request_body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.7
        }

        try:
            response = self.bedrock.invoke_model(
                modelId=self.model_id,
                body=json.dumps(request_body)
            )

            response_body = json.loads(response['body'].read())
            return response_body['content'][0]['text']

        except ClientError as e:
            print(f"Error calling Bedrock: {e}")
            raise

    def _load_profiles(self, profiles_file: str, max_profiles: Optional[int] = None) -> List[Dict[str, Any]]:
        """Load character profiles from JSONL file"""
        profiles = []

        with open(profiles_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    raw_profile = json.loads(line)
                    # Parse the profile from the prompt field
                    parsed_profile = self._parse_profile(raw_profile)
                    if parsed_profile:
                        profiles.append(parsed_profile)

                    if max_profiles and len(profiles) >= max_profiles:
                        break

        return profiles

    def _parse_profile(self, raw_profile: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Parse character profile from the prompt field

        Expected format in prompt:
        <Character Setting>
        Name: XXX
        Gender: XXX
        Introduction: XXX
        Detailed Description: XXX
        </Character Setting>
        """
        prompt = raw_profile.get('prompt', '')

        # Extract character setting section
        character_setting_match = re.search(
            r'<Character Setting>(.*?)</Character Setting>',
            prompt,
            re.DOTALL
        )

        if not character_setting_match:
            return None

        setting_text = character_setting_match.group(1)

        # Extract fields
        name_match = re.search(r'Name:\s*(.+?)(?:\n|$)', setting_text)
        gender_match = re.search(r'Gender:\s*(.+?)(?:\n|$)', setting_text)
        intro_match = re.search(r'Introduction:\s*(.+?)(?:\n|Detailed Description:)', setting_text, re.DOTALL)
        detailed_match = re.search(r'Detailed Description:\s*(.+?)(?:\n</Character Setting>|$)', setting_text, re.DOTALL)

        # Build character profile
        character_profile = {
            'Name': name_match.group(1).strip() if name_match else 'Unknown',
            'Gender': gender_match.group(1).strip() if gender_match else 'unknown',
            'Introduction': intro_match.group(1).strip() if intro_match else '',
            'Detailed Description': detailed_match.group(1).strip() if detailed_match else ''
        }

        return {
            'prompt': prompt,
            'character_profile': character_profile
        }

    def _detect_language(self, profile: Dict[str, Any]) -> str:
        """Detect language from profile"""
        if self.auto_detect_language:
            return LanguageDetector.detect_from_profile(profile.get('character_profile', {}))
        return self.default_language

    def _format_conversation_history(self, history: List[Dict[str, str]]) -> str:
        """Format conversation history for prompt"""
        if not history:
            return "无 / None"

        lines = []
        for turn in history:
            role = turn.get('role', 'unknown')
            content = turn.get('content', '')
            lines.append(f"{role}: {content}")

        return '\n'.join(lines)

    def _extract_tag_content(self, text: str, tag: str) -> str:
        """Extract content from XML-like tags"""
        pattern = f"<{tag}>(.*?)</{tag}>"
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return ""
