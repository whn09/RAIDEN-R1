"""
RAIDEN-R1 Data Generation Module

This module provides tools for generating and managing role-playing training data
for the RAIDEN-R1 framework.

Components:
- collection: Dataset management and sample definitions
- bedrock_generator: AWS Bedrock data generation (cloud)
- sglang_generator: SGLang local data generation (10-100x faster)
- language_utils: Multilingual support (zh/ja/en/ko)

Usage:
    from data.collection import DatasetBuilder, RolePlayingSample
    from data.bedrock_generator import BedrockDataGenerator
    from data.sglang_generator import SGLangGenerator
"""

from .collection import (
    RolePlayingSample,
    DatasetBuilder
)

from .bedrock_generator import BedrockDataGenerator
from .sglang_generator import SGLangGenerator

from .language_utils import (
    LanguageDetector,
    PromptTemplates,
    translate_question_type,
    translate_focus_area,
    translate_difficulty
)

__all__ = [
    # Collection classes
    'RolePlayingSample',
    'DatasetBuilder',

    # Generators
    'BedrockDataGenerator',
    'SGLangGenerator',

    # Language utilities
    'LanguageDetector',
    'PromptTemplates',
    'translate_question_type',
    'translate_focus_area',
    'translate_difficulty',
]

__version__ = '1.0.0'
