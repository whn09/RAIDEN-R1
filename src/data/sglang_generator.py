"""
SGLang Local Data Generator for RAIDEN-R1

Generates role-playing training data using locally deployed models via SGLang.
10-100x faster than cloud APIs with comparable quality.

Supports models:
- MiniMax M2 (Recommended for role-playing)
- GLM-4/GLM-4.6 (Fast and efficient)
- Qwen2.5-14B/32B (Strong reasoning)
- DeepSeek-V2.5 (High quality)
- Yi-1.5-34B (Multilingual)
"""

import json
import time
import random
import re
from typing import Dict, List, Any, Optional
import requests
from openai import OpenAI

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


class SGLangGenerator:
    """
    Generate RAIDEN training data using SGLang-deployed models
    """

    # Question types based on RAIDEN paper
    WH_QUESTION_TYPES = ["what", "who", "where", "when", "why", "how"]

    # Difficulty levels
    DIFFICULTY_LEVELS = ["easy", "medium", "hard"]

    def __init__(
        self,
        base_url: str = "http://localhost:30000",
        model_name: Optional[str] = None,
        timeout: int = 120,
        language: str = "zh",
        auto_detect_language: bool = True,
        enable_thinking: bool = False
    ):
        """
        Initialize SGLang data generator

        Args:
            base_url: SGLang server URL
            model_name: Model name for metadata (auto-detected if None)
            timeout: Request timeout in seconds
            language: Default language for generation
            auto_detect_language: Auto-detect language from profiles
            enable_thinking: Enable thinking mode for models that support it (GLM-4.6, MiniMax M2, etc.)
                            Set to False to disable thinking output for cleaner responses
        """
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.default_language = language
        self.auto_detect_language = auto_detect_language
        self.enable_thinking = enable_thinking

        # Initialize OpenAI client for SGLang
        self.client = OpenAI(
            api_key="EMPTY",  # SGLang doesn't need real API key
            base_url=f"{self.base_url}/v1",
            timeout=timeout
        )

        # Test connection and get model info
        self.model_name = model_name or self._get_model_info()

        print(f"Initialized SGLang generator")
        print(f"  Server: {self.base_url}")
        print(f"  Model: {self.model_name}")
        print(f"  Default language: {language}")
        print(f"  Auto-detect: {auto_detect_language}")
        print(f"  Enable thinking: {enable_thinking}")

    def _get_model_info(self) -> str:
        """Get model information from SGLang server"""
        try:
            response = requests.get(
                f"{self.base_url}/get_model_info",
                timeout=10
            )
            if response.status_code == 200:
                model_info = response.json()
                return model_info.get('model_path', 'unknown').split('/')[-1]
        except Exception as e:
            print(f"Warning: Could not get model info: {e}")

        return "unknown"

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

                # Generate CM sample if requested
                if include_conversation_memory:
                    print(f"  Generating CM sample...")
                    cm_sample = self.generate_cm_sample(profile)
                    if cm_sample:
                        all_samples.append(cm_sample.to_dict())

            except Exception as e:
                print(f"  Error processing profile: {e}")
                continue

            # Save periodically
            if idx % 10 == 0:
                self._save_samples(all_samples, output_file)
                print(f"  Saved checkpoint: {len(all_samples)} samples")

        # Final save
        print(f"\n\nSaving {len(all_samples)} samples to: {output_file}")
        self._save_samples(all_samples, output_file)
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
                    "source": "sglang_sbk",
                    "model": self.model_name,
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
                    "source": "sglang_cm",
                    "model": self.model_name,
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
        """Generate a question using local model"""
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

        # Call model
        response = self._call_model(prompt, max_tokens=4000)

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

        # Call model
        response = self._call_model(prompt, max_tokens=4000)

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
            response = self._call_model(prompt, max_tokens=4000)

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

    def _call_model(self, prompt: str, max_tokens: int = 2000, temperature: float = 0.7) -> str:
        """Call SGLang model via OpenAI SDK"""
        messages = [
            {
                "role": "user",
                "content": prompt
            }
        ]

        # Prepare extra_body for thinking control
        extra_body = {}
        if not self.enable_thinking:
            extra_body = {
                "chat_template_kwargs": {
                    "enable_thinking": False,  # GLM-4.6, Qwen3
                    "thinking": False,          # DeepSeek
                }
            }
            print(f"\n[DEBUG] Calling with thinking disabled: {extra_body}")

        try:
            # Use OpenAI SDK (properly handles extra_body)
            completion = self.client.chat.completions.create(
                model=self.model_name or "default",
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                extra_body=extra_body if extra_body else None
            )

            # Extract content
            content = completion.choices[0].message.content

            print(f'\n[DEBUG] Raw content from model (first 200 chars):\n{content[:200] if content else "None"}...\n')

            # Filter out thinking tags as backup
            if not self.enable_thinking and content:
                original_length = len(content)
                content = self._filter_thinking_tags(content)
                filtered_length = len(content)
                if original_length != filtered_length:
                    print(f'[DEBUG] Filtered thinking tags: {original_length - filtered_length} chars removed')

            print(f'[DEBUG] Final content (first 200 chars):\n{content[:200] if content else "None"}...\n')

            return content or ""

        except Exception as e:
            print(f"Error calling SGLang: {e}")
            raise

    def _filter_thinking_tags(self, text: str) -> str:
        """
        Filter out thinking/reasoning tags from model output

        Some models (like MiniMax M2, GLM-4.6) output their internal thinking process
        wrapped in tags like <think>, <thinking>, <thought>, etc.
        This method removes those tags and their content.

        Note: We preserve <reasoning> and <answer> tags as those are part of our
        expected output format defined in prompts.

        Args:
            text: Raw model output

        Returns:
            Cleaned text without thinking tags
        """
        # Only filter internal thinking tags, not our prompt-required tags
        thinking_patterns = [
            r'<think>.*?</think>',
            r'<thinking>.*?</thinking>',
            r'<thought>.*?</thought>',
        ]

        cleaned_text = text
        for pattern in thinking_patterns:
            cleaned_text = re.sub(pattern, '', cleaned_text, flags=re.DOTALL)

        # Remove extra whitespace that may be left after removing tags
        cleaned_text = re.sub(r'\n\s*\n\s*\n', '\n\n', cleaned_text)
        cleaned_text = cleaned_text.strip()

        return cleaned_text

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

    def _save_samples(self, samples: List[Dict[str, Any]], output_file: str) -> None:
        """Save samples to file"""
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(samples, f, ensure_ascii=False, indent=2)

    def test_connection(self) -> bool:
        """Test connection to SGLang server"""
        try:
            response = requests.get(
                f"{self.base_url}/health",
                timeout=5
            )
            return response.status_code == 200
        except Exception as e:
            print(f"Connection test failed: {e}")
            return False
