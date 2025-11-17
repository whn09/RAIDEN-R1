"""
Multilingual Support Utilities for RAIDEN-R1

Supports language detection and prompts for:
- Chinese (zh) - Default
- Japanese (ja)
- English (en)
- Korean (ko)
"""

import re
from typing import Dict, Any, Optional


class LanguageDetector:
    """
    Detects language from text using Unicode character ranges
    """

    @staticmethod
    def detect_language(text: str) -> str:
        """
        Detect the primary language of text

        Args:
            text: Input text to analyze

        Returns:
            Language code: 'zh', 'ja', 'en', or 'ko'
        """
        if not text:
            return 'zh'  # Default to Chinese

        # Count character types
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
        japanese_chars = len(re.findall(r'[\u3040-\u309f\u30a0-\u30ff]', text))  # Hiragana + Katakana
        korean_chars = len(re.findall(r'[\uac00-\ud7af]', text))  # Hangul
        latin_chars = len(re.findall(r'[a-zA-Z]', text))

        # Determine primary language
        char_counts = {
            'zh': chinese_chars,
            'ja': japanese_chars + chinese_chars,  # Japanese uses both
            'ko': korean_chars,
            'en': latin_chars
        }

        # If Japanese-specific characters found, prioritize it
        if japanese_chars > 0:
            return 'ja'

        # If Korean characters found, it's Korean
        if korean_chars > chinese_chars and korean_chars > latin_chars:
            return 'ko'

        # If mostly Latin characters, it's English
        if latin_chars > chinese_chars and latin_chars > 10:
            return 'en'

        # Default to Chinese (most common for role-playing content)
        return 'zh'

    @staticmethod
    def detect_from_profile(profile: Dict[str, Any]) -> str:
        """
        Detect language from character profile

        Args:
            profile: Character profile dictionary

        Returns:
            Detected language code
        """
        # Combine text from profile fields
        text_parts = []

        if isinstance(profile, dict):
            for key, value in profile.items():
                if isinstance(value, str):
                    text_parts.append(value)
                elif isinstance(value, dict):
                    for v in value.values():
                        if isinstance(v, str):
                            text_parts.append(v)

        combined_text = ' '.join(text_parts)
        return LanguageDetector.detect_language(combined_text)


class PromptTemplates:
    """
    Multilingual prompt templates for data generation
    """

    # Question generation prompts
    QUESTION_GENERATION = {
        'zh': """根据角色信息生成一个问题。

角色信息：
{character_info}

要求：生成一个User向角色提问的简短问题，用"你"开头。

重要：直接输出问题，不要输出任何思考过程、分析或解释。只输出最终的问题。

例如：
你叫什么名字？
你的性格是怎样的？
你有什么爱好？

现在生成一个{question_type}类型的问题：""",

        'ja': """キャラクター情報に基づいて質問を生成してください。

キャラクター情報：
{character_info}

要求：Userがキャラクターに尋ねる短い質問を生成。

直接質問を出力、説明不要。例：
あなたの名前は何ですか？
あなたの性格はどうですか？
あなたの趣味は何ですか？

{question_type}タイプの質問を生成：""",

        'en': """Generate a question based on the character information.

Character Information:
{character_info}

Requirement: Generate a short question that User asks the character.

Output the question directly, no explanation. Examples:
What's your name?
What's your personality like?
What are your hobbies?

Now generate a {question_type} type question:""",

        'ko': """캐릭터 정보를 바탕으로 질문을 생성하세요.

캐릭터 정보:
{character_info}

요구사항: User가 캐릭터에게 묻는 짧은 질문 생성.

질문을 직접 출력, 설명 불필요. 예시:
당신의 이름은 무엇인가요?
당신의 성격은 어떤가요?
당신의 취미는 무엇인가요?

{question_type} 유형의 질문 생성:"""
    }

    # Answer generation with CoT reasoning
    ANSWER_GENERATION = {
        'zh': """你扮演这个角色，用第一人称回答User的问题。严格根据角色信息回答。

角色信息：
{character_info}

User问：{question}

要求：
1. 用第一人称（我/我的/我叫）回答
2. 根据上面的角色信息回答
3. 回答要简短（1-2句话）
4. 直接输出答案，不要解释

例如：
问：你叫什么名字？
答：我叫小明。

问：你的性格怎么样？
答：我性格开朗活泼。

现在回答：""",

        'ja': """あなたはこのキャラクターを演じて、一人称でUserの質問に答えます。キャラクター情報に基づいて答えてください。

キャラクター情報：
{character_info}

User：{question}

要求：
1. 一人称（私/僕）で答える
2. 上記のキャラクター情報に基づく
3. 短く答える（1-2文）
4. 直接答えを出力、説明不要

例：
User：あなたの名前は？
答：私は太郎です。

User：あなたの性格は？
答：私は明るい性格です。

今答えてください：""",

        'en': """You play this character and answer User's question in first person. Answer based on the character information.

Character information:
{character_info}

User: {question}

Requirements:
1. Answer in first person (I/I'm/My)
2. Based on the character information above
3. Keep it short (1-2 sentences)
4. Output answer directly, no explanation

Examples:
User: What's your name?
Answer: I'm Tom.

User: What's your personality?
Answer: I'm cheerful and outgoing.

Now answer:""",

        'ko': """당신은 이 캐릭터를 연기하여 1인칭으로 User 질문에 답합니다. 캐릭터 정보를 바탕으로 답변하세요.

캐릭터 정보:
{character_info}

User: {question}

요구사항:
1. 1인칭(저는/나는)으로 답변
2. 위 캐릭터 정보에 기반
3. 짧게 답변(1-2문장)
4. 답변을 직접 출력, 설명 불필요

예시:
User: 당신의 이름은?
답: 저는 철수입니다.

User: 당신의 성격은?
답: 저는 밝은 성격입니다.

지금 답변하세요:"""
    }

    # Keyword extraction prompts
    KEYWORD_EXTRACTION = {
        'zh': """请从以下答案中提取关键词用于验证。

答案：
{answer}

要求：
1. 提取最重要的1-3个关键词或短语
2. 关键词应该是可验证的事实
3. 返回JSON格式：{{"keywords": ["关键词1", "关键词2"]}}

请只返回JSON，不要包含其他内容。""",

        'ja': """以下の回答から検証用のキーワードを抽出してください。

回答：
{answer}

要件：
1. 最も重要な1-3個のキーワードまたはフレーズを抽出する
2. キーワードは検証可能な事実であるべきです
3. JSON形式で返す：{{"keywords": ["キーワード1", "キーワード2"]}}

JSONのみを返してください。他の内容は含めないでください。""",

        'en': """Please extract keywords from the following answer for verification.

Answer:
{answer}

Requirements:
1. Extract the 1-3 most important keywords or phrases
2. Keywords should be verifiable facts
3. Return in JSON format: {{"keywords": ["keyword1", "keyword2"]}}

Return only JSON, without any other content.""",

        'ko': """다음 답변에서 검증용 키워드를 추출해주세요.

답변:
{answer}

요구사항:
1. 가장 중요한 1-3개의 키워드 또는 구문을 추출합니다
2. 키워드는 검증 가능한 사실이어야 합니다
3. JSON 형식으로 반환: {{"keywords": ["키워드1", "키워드2"]}}

JSON만 반환하고 다른 내용은 포함하지 마세요."""
    }

    @staticmethod
    def get_question_prompt(language: str = 'zh', **kwargs) -> str:
        """Get question generation prompt in specified language"""
        template = PromptTemplates.QUESTION_GENERATION.get(language, PromptTemplates.QUESTION_GENERATION['zh'])
        return template.format(**kwargs)

    @staticmethod
    def get_answer_prompt(language: str = 'zh', **kwargs) -> str:
        """Get answer generation prompt in specified language"""
        template = PromptTemplates.ANSWER_GENERATION.get(language, PromptTemplates.ANSWER_GENERATION['zh'])
        return template.format(**kwargs)

    @staticmethod
    def get_keyword_prompt(language: str = 'zh', **kwargs) -> str:
        """Get keyword extraction prompt in specified language"""
        template = PromptTemplates.KEYWORD_EXTRACTION.get(language, PromptTemplates.KEYWORD_EXTRACTION['zh'])
        return template.format(**kwargs)


# Question type translations
QUESTION_TYPES = {
    'zh': {
        'what': '什么',
        'who': '谁',
        'where': '哪里',
        'when': '什么时候',
        'why': '为什么',
        'how': '如何'
    },
    'ja': {
        'what': '何',
        'who': '誰',
        'where': 'どこ',
        'when': 'いつ',
        'why': 'なぜ',
        'how': 'どのように'
    },
    'en': {
        'what': 'what',
        'who': 'who',
        'where': 'where',
        'when': 'when',
        'why': 'why',
        'how': 'how'
    },
    'ko': {
        'what': '무엇',
        'who': '누구',
        'where': '어디',
        'when': '언제',
        'why': '왜',
        'how': '어떻게'
    }
}

# Focus area translations
FOCUS_AREAS = {
    'zh': {
        'script_knowledge': '剧本知识（背景、性格、经历）',
        'conversation_memory': '对话记忆（上下文理解）'
    },
    'ja': {
        'script_knowledge': 'スクリプト知識（背景、性格、経験）',
        'conversation_memory': '会話記憶（コンテキスト理解）'
    },
    'en': {
        'script_knowledge': 'script knowledge (background, personality, experience)',
        'conversation_memory': 'conversation memory (context understanding)'
    },
    'ko': {
        'script_knowledge': '스크립트 지식 (배경, 성격, 경험)',
        'conversation_memory': '대화 기억 (맥락 이해)'
    }
}

# Difficulty level translations
DIFFICULTY_LEVELS = {
    'zh': {
        'easy': '简单',
        'medium': '中等',
        'hard': '困难'
    },
    'ja': {
        'easy': '簡単',
        'medium': '中級',
        'hard': '難しい'
    },
    'en': {
        'easy': 'easy',
        'medium': 'medium',
        'hard': 'hard'
    },
    'ko': {
        'easy': '쉬움',
        'medium': '중간',
        'hard': '어려움'
    }
}


def translate_question_type(qtype: str, language: str = 'zh') -> str:
    """Translate question type to specified language"""
    return QUESTION_TYPES.get(language, QUESTION_TYPES['zh']).get(qtype, qtype)


def translate_focus_area(focus: str, language: str = 'zh') -> str:
    """Translate focus area to specified language"""
    return FOCUS_AREAS.get(language, FOCUS_AREAS['zh']).get(focus, focus)


def translate_difficulty(difficulty: str, language: str = 'zh') -> str:
    """Translate difficulty level to specified language"""
    return DIFFICULTY_LEVELS.get(language, DIFFICULTY_LEVELS['zh']).get(difficulty, difficulty)


def extract_question_from_response(response: str, language: str) -> str:
    """
    Extract actual question from model response, filtering out thinking process

    Handles cases where model outputs thinking in English before the actual question
    """
    # If response starts with English thinking pattern, it's not the actual question
    if response.startswith(("The user", "The question", "We need", "First", "I need")):
        # Model is thinking in English, need to extract the actual question
        # Look for question marks in target language
        lines = response.split('\n')
        for line in lines:
            line = line.strip()
            # Chinese/Japanese question
            if language in ['zh', 'ja'] and '？' in line and not line.startswith(('The', 'So', 'We', 'But', 'I ')):
                # Extract just the question part
                if '？' in line:
                    question = line.split('？')[0] + '？'
                    return question.strip()
            # English question
            elif language == 'en' and '?' in line and not line.startswith(('The', 'So', 'We', 'But', 'First')):
                question = line.split('?')[0] + '?'
                return question.strip()

        # Fallback: return a default question
        default_questions = {
            'zh': '你叫什么名字？',
            'ja': 'あなたの名前は何ですか？',
            'en': "What's your name?",
            'ko': '당신의 이름은 무엇인가요?'
        }
        return default_questions.get(language, "What's your name?")

    # Response looks clean, return as is
    return response.strip()


def extract_answer_from_response(response: str, language: str) -> str:
    """
    Extract actual answer from model response, filtering out thinking process
    """
    # If response starts with English thinking pattern
    if response.startswith(("The user", "The question", "We need", "First", "I need", "So")):
        # Model is thinking, need to extract actual answer
        lines = response.split('\n')

        # Look for lines that start with or contain first-person pronouns
        first_person_patterns = {
            'zh': ['我', '我的', '我是', '我叫'],
            'ja': ['私', '僕', '私は', '僕は'],
            'en': ['I', "I'm", 'My', 'I am'],
            'ko': ['저는', '나는', '제', '내']
        }

        patterns = first_person_patterns.get(language, first_person_patterns['zh'])

        # First pass: lines starting with first person (highest priority)
        for line in lines:
            line = line.strip()
            if any(line.startswith(p) for p in patterns):
                return line

        # Second pass: lines containing first person but not starting with thinking
        for line in lines:
            line = line.strip()
            if line and not line.startswith(('The', 'So', 'We', 'But', 'First', 'This', 'It')):
                if any(p in line for p in patterns):
                    return line

        # Third pass: any non-empty line that looks like an answer (not thinking)
        for line in lines:
            line = line.strip()
            if line and len(line) > 3 and not line.startswith(('The', 'So', 'We', 'But', 'First', 'This', 'It', 'To', 'From', 'For')):
                return line

        # Fallback: return a default answer
        default_answers = {
            'zh': '我是这个角色。',
            'ja': '私はこのキャラクターです。',
            'en': "I'm this character.",
            'ko': '저는 이 캐릭터입니다.'
        }
        return default_answers.get(language, "I'm this character.")

    # Response looks clean, return as is
    return response.strip()
