# RAIDEN-R1: Improving Role-awareness of LLMs via GRPO with Verifiable Reward

Implementation of the RAIDEN-R1 framework for improving role-awareness in large language models through Group Relative Policy Optimization (GRPO) with Verifiable Role-Awareness Rewards (VRAR).

## Paper

**Title**: RAIDEN-R1: Improving Role-awareness of LLMs via GRPO with Verifiable Reward

**Authors**: Zongsheng Wang, Kaili Sun, Bowen Wu, Qun Yu, Ying Li, Baoxun Wang

**arXiv**: https://arxiv.org/html/2505.10218v1

## Overview

RAIDEN-R1 addresses role consistency in role-playing conversational agents through:

1. **Verifiable Role-Awareness Reward (VRAR)**: Quantifiable reward mechanism for role-aware training
2. **GRPO Training**: Using Group Relative Policy Optimization to improve role consistency
3. **High-quality Dataset**: Role-aware data with Script-Based Knowledge (SBK) and Conversation Memory (CM)

## Project Structure

```
raiden-r1/
├── src/
│   ├── data/                    # Data processing and generation
│   │   ├── sglang_generator.py  # SGLang local generator (10-100x faster)
│   │   ├── bedrock_generator.py # AWS Bedrock data generator
│   │   ├── language_utils.py    # Multilingual support (zh/ja/en/ko)
│   │   └── collection.py        # Dataset management
│   ├── training/
│   │   └── grpo_trainer.py      # GRPO implementation
│   ├── evaluation/              # Evaluation metrics (SBK, CM)
│   └── rewards/
│       └── vrar.py              # Verifiable Role-Awareness Rewards
├── configs/
│   └── grpo_config.yaml         # Training configuration
├── data/
│   ├── online_profiles.jsonl    # Character profiles
│   └── training/                # Generated training data
├── scripts/                     # Training and generation scripts
└── accelerate_config.yaml       # Multi-GPU training config
```

## Quick Start

### 1. Installation

```bash
pip install -r requirements.txt
```

**Requirements**:
- Python 3.8+
- PyTorch 2.0+
- Transformers
- 8x NVIDIA H800/H200 GPUs for training

### 2. Data Generation

**Recommended: SGLang + GLM-4.6** (10-100x faster than cloud APIs)

```bash
# Download GLM-4.6 model
huggingface-cli download zai-org/GLM-4.6 --local-dir /path/to/GLM-4.6

# Start SGLang server
python -m sglang.launch_server \
    --model-path /path/to/GLM-4.6 \
    --tp-size 8 \
    --port 30000 \
    --chat-template glm4 \
    --trust-remote-code

# Generate training data (Chinese by default)
python scripts/generate_data_with_sglang.py \
    --profiles_file ./data/online_profiles.jsonl \
    --num_samples_per_profile 2 \
    --include_cm \
    --language zh

# For other languages
python scripts/generate_data_with_sglang.py --language ja  # Japanese
python scripts/generate_data_with_sglang.py --language en  # English
python scripts/generate_data_with_sglang.py --language ko  # Korean
```

**Alternative: AWS Bedrock (for quick prototyping)**

```bash
python scripts/generate_data_with_bedrock.py \
    --num_samples_per_profile 2 \
    --language zh
```

### 3. Training

```bash
# Multi-GPU training (8x H200/H800)
accelerate launch --config_file accelerate_config.yaml \
    scripts/train.py --config configs/grpo_config.yaml

# Monitor training
watch -n 1 nvidia-smi
tail -f outputs/training.log
```

**Training Configuration** (`configs/grpo_config.yaml`):
- Base Model: `Qwen/Qwen2.5-14B-Instruct`
- Learning Rate: `3e-6` with cosine scheduler
- Batch Size: `8` per GPU (effective: 128 with 8 GPUs)
- Precision: `bf16`
- Epochs: `1`

### 4. Evaluation

```bash
python scripts/evaluate.py \
    --model_path ./outputs/epoch_0 \
    --eval_data ./data/training/validation.json \
    --output_file ./evaluation_results.json
```

## Data Generation Details

### Why GLM-4.6?

- ✅ **Excellent role-playing quality**
- ✅ **Fast inference with SGLang**
- ✅ **Supports thinking mode control** (clean outputs)
- ✅ **Native Chinese support**
- ✅ **Open-source and free**

### Supported Models (SGLang)

- **GLM-4.6** (Recommended) - Best balance of speed and quality
- **MiniMax M2** - Strong for role-playing
- **Qwen2.5-14B/32B** - Strong reasoning
- **DeepSeek-V2.5** - High quality

### Multilingual Support

RAIDEN-R1 supports multiple languages with auto-detection:

- **Chinese (zh)**: Default, native support ✓
- **Japanese (ja)**: 日本語対応
- **English (en)**: Full support
- **Korean (ko)**: 한국어 지원

```bash
# Use specific language
python scripts/generate_data_with_sglang.py --language zh

# Auto-detect from character profiles
python scripts/generate_data_with_sglang.py --auto_detect
```

### Generation Parameters

```bash
python scripts/generate_data_with_sglang.py \
    --profiles_file ./data/online_profiles.jsonl \
    --output_file ./data/generated_samples.json \
    --num_samples_per_profile 2 \
    --include_cm \
    --language zh \
    --max_profiles 100  # Limit for testing
```

### Input Data Format

Character profiles should be in JSONL format (`data/online_profiles.jsonl`). Each line contains a character with structured information:

```jsonl
{"prompt": "...[角色扮演系统提示词]...\n\n<Character Setting>\nName: 小明\nGender: male\nIntroduction: 一个阳光开朗的高中生，喜欢打篮球和画画。\nDetailed Description: 小明今年17岁，就读于某高中二年级。性格外向活泼，总是能给周围的人带来欢笑。课余时间喜欢和朋友一起打篮球，也热爱绘画创作。梦想是成为一名职业插画师。\n</Character Setting>\n\n...[其他设定]..."}
```

**Key Fields**:
- `Name`: Character name
- `Gender`: male/female
- `Introduction`: Brief introduction (1-2 sentences)
- `Detailed Description`: Detailed character background, personality, goals, etc.

The system will automatically parse the `<Character Setting>` block and generate training data based on this information.

## Training Data Format

```json
{
  "character_name": "角色名称",
  "character_profile": {
    "Name": "角色名称",
    "Gender": "female",
    "Introduction": "简介",
    "Detailed Description": "详细描述"
  },
  "conversation_history": [],
  "question": "你叫什么名字？",
  "answer": "我叫小明。",
  "keywords": ["小明"],
  "question_type": "what",
  "validation_method": "single_term_validation",
  "difficulty": "easy",
  "metadata": {
    "source": "sglang_sbk",
    "model": "GLM-4.6",
    "language": "zh"
  }
}
```

## Results

The RAIDEN-R1 14B-GRPO model achieves:
- **SBK (Script-Based Knowledge)**: 88.04%
- **CM (Conversation Memory)**: 88.65%

## Troubleshooting

### Low GPU Utilization
```bash
# Ensure using accelerate for multi-GPU
accelerate launch --config_file accelerate_config.yaml scripts/train.py --config configs/grpo_config.yaml

# Increase batch size in configs/grpo_config.yaml
batch_size: 8
gradient_accumulation_steps: 2
```

### Out of Memory (OOM)
```bash
# Reduce batch size in configs/grpo_config.yaml
batch_size: 4

# Use fewer GPUs
accelerate launch --num_processes=4 scripts/train.py --config configs/grpo_config.yaml
```

### SGLang Thinking Mode Issues

If the model outputs thinking tags (`<think>...</think>`), ensure:
1. SGLang server started with `--chat-template glm4`
2. Using latest code with OpenAI SDK integration
3. Not using `--enable_thinking` flag

## Advanced Configuration

### SGLang Server Optimization

```bash
# For production with 8 GPUs
python -m sglang.launch_server \
    --model-path /path/to/GLM-4.6 \
    --tp-size 8 \
    --ep-size 8 \
    --port 30000 \
    --chat-template glm4 \
    --mem-fraction-static 0.85 \
    --trust-remote-code
```

### Training with Custom Data

```bash
# Generate more samples per profile
python scripts/generate_data_with_sglang.py \
    --num_samples_per_profile 5 \
    --include_cm

# Use custom training/validation split
python scripts/generate_data_with_sglang.py \
    --train_ratio 0.8  # 80% train, 20% validation
```

## Citation

```bibtex
@article{wang2025raiden,
  title={RAIDEN-R1: Improving Role-awareness of LLMs via GRPO with Verifiable Reward},
  author={Wang, Zongsheng and Sun, Kaili and Wu, Bowen and Yu, Qun and Li, Ying and Wang, Baoxun},
  journal={arXiv preprint arXiv:2505.10218},
  year={2025}
}
```

## Acknowledgments

- Open-R1 library
- RAIDEN Benchmark
- Qwen2.5-14B-Instruct base model
- SGLang framework
