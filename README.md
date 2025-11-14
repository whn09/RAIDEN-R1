# RAIDEN-R1: Improving Role-awareness of LLMs via GRPO with Verifiable Reward

This repository contains the implementation of the RAIDEN-R1 framework for improving role-awareness in large language models through Group Relative Policy Optimization (GRPO) with Verifiable Role-Awareness Rewards (VRAR).

## Paper

**Title**: RAIDEN-R1: Improving Role-awareness of LLMs via GRPO with Verifiable Reward

**Authors**: Zongsheng Wang, Kaili Sun, Bowen Wu, Qun Yu, Ying Li, Baoxun Wang

**arXiv**: https://arxiv.org/html/2505.10218v1

## Overview

RAIDEN-R1 addresses the challenge of maintaining role consistency in role-playing conversational agents (RPCAs) through:

1. **Verifiable Role-Awareness Reward (VRAR)**: A quantifiable reward mechanism for role-aware training
2. **GRPO Training**: Using Group Relative Policy Optimization to improve role consistency
3. **High-quality CoT Dataset**: Role-aware Chain-of-Thought data through multi-LLM collaboration

## Key Features

- Two data collection strategies:
  - Single-Term Validation
  - Multi-Term Dynamic Parsing
- Two reward mechanisms:
  - Accuracy reward
  - Format reward
- Evaluation metrics:
  - Script-Based Knowledge (SBK)
  - Conversation Memory (CM)

## Project Structure

```
raiden-r1/
├── src/
│   ├── data/                    # Data processing and generation
│   │   ├── bedrock_generator.py # AWS Bedrock data generator
│   │   ├── sglang_generator.py  # SGLang local generator (10-100x faster)
│   │   ├── language_utils.py    # Multilingual support (zh/ja/en/ko)
│   │   └── collection.py        # Dataset management
│   ├── models/                  # Model definitions
│   ├── training/                # Training with GRPO
│   │   └── grpo_trainer.py      # GRPO implementation
│   ├── evaluation/              # Evaluation metrics (SBK, CM)
│   └── rewards/                 # VRAR reward mechanisms
│       └── vrar.py              # Verifiable Role-Awareness Rewards
├── configs/
│   ├── grpo_config.yaml         # Training configuration
│   └── sglang_models.yaml       # SGLang model configurations
├── data/                        # Dataset directory
│   ├── online_profiles.jsonl    # Character profiles
│   └── training/                # Generated training data
├── scripts/                     # Utility scripts
│   ├── train.py                 # Main training script
│   ├── generate_data_with_bedrock.py
│   ├── generate_data_with_sglang.py
│   └── deploy_sglang.sh         # One-click SGLang deployment
├── accelerate_config.yaml       # Multi-GPU training config
├── SGLANG_QUICKSTART.md         # SGLang setup guide
├── BEDROCK_QUICKSTART.md        # Bedrock setup guide
└── README.md
```

## Installation

```bash
pip install -r requirements.txt
```

## Requirements

- Python 3.8+
- PyTorch 2.0+
- Transformers
- 8x NVIDIA H800 GPUs (or equivalent) for training
- Base model: Qwen2.5-14B-Instruct

## Training

RAIDEN-R1 uses GRPO (Group Relative Policy Optimization) with Verifiable Role-Awareness Rewards for training.

> 📖 **详细训练指南**: 查看 [TRAINING_GUIDE.md](TRAINING_GUIDE.md) 获取完整的训练步骤、配置说明和故障排除。

### Multi-GPU Training (Recommended)

For best performance with 8x H200 GPUs:

```bash
# Configure accelerate for multi-GPU training
accelerate launch --config_file accelerate_config.yaml \
    scripts/train.py --config configs/grpo_config.yaml
```

### Single-GPU Training

For testing or smaller setups:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train.py \
    --config configs/grpo_config.yaml
```

### Training Configuration

Edit `configs/grpo_config.yaml` to customize:

- **Model**: `Qwen/Qwen2.5-14B-Instruct` (base model)
- **Learning Rate**: `3e-6` with cosine scheduler
- **Batch Size**: `8` per GPU (effective batch: 128 with 8 GPUs)
- **Gradient Accumulation**: `2` steps
- **Precision**: `bf16` (mixed precision)
- **Epochs**: `1` (as per paper)

### Training Data Format

The training script expects JSON files with the following structure:

```json
{
  "character_name": "角色名称",
  "character_profile": {...},
  "conversation_history": [...],
  "question": "问题",
  "answer": "答案",
  "keywords": ["关键词1", "关键词2"],
  "question_type": "what",
  "validation_method": "multi_term_parsing",
  "difficulty": "medium",
  "metadata": {...}
}
```

Generate training data using the data generation methods described below.

## Evaluation

```bash
python scripts/evaluate.py --model_path <path_to_model> --eval_data <path_to_eval_data>

python scripts/evaluate.py \
  --model_path ./outputs/epoch_0 \
  --eval_data ./data/test_training/validation.json \
  --output_file ./evaluation_results.json

python scripts/evaluate.py \
  --model_path Qwen/Qwen2.5-14B-Instruct \
  --eval_data ./data/test_training/validation.json \
  --output_file ./evaluation_results_origin.json
```

## Data Generation

RAIDEN-R1 supports two data generation methods:

### Method 1: AWS Bedrock (Cloud API)
- **Best for**: Quick prototyping, small datasets (<500 samples)
- **Quality**: Highest (Claude 3.5 Sonnet)
- **Speed**: Baseline
- **Cost**: ~$30/1000 samples

```bash
# Quick start
./example_usage.sh

# Or manual
python scripts/generate_data_with_bedrock.py \
    --profiles_file ./data/online_profiles.jsonl \
    --num_samples_per_profile 2
```

See [BEDROCK_QUICKSTART.md](BEDROCK_QUICKSTART.md) for details.

### Method 2: SGLang Local Deployment (Recommended)
- **Best for**: Production, large datasets (1000+ samples)
- **Quality**: High (open-source models)
- **Speed**: ⚡ **10-100x faster** than cloud APIs
- **Cost**: $0 API cost (GPU cost only)

```bash
# Deploy SGLang server

hf download MiniMaxAI/MiniMax-M2 --local-dir /opt/dlami/nvme/MiniMax-M2/

# python -m sglang.launch_server \
#     --model-path /opt/dlami/nvme/MiniMax-M2 \
#     --tp-size 8 \
#     --ep-size 8 \
#     --tool-call-parser minimax-m2 \
#     --trust-remote-code \
#     --host 0.0.0.0 \
#     --reasoning-parser minimax-append-think \
#     --port 30000 \
#     --mem-fraction-static 0.85

./scripts/deploy_sglang.sh \
    --model-path /opt/dlami/nvme/MiniMax-M2/ \
    --model-name minimax-m2 \
    --tp-size 8 \
    --ep-size 8

# Generate data
python scripts/generate_data_with_sglang.py \
    --num_samples_per_profile 2 \
    --include_cm \
    --model_name MiniMax-M2 \
    --language zh

hf download zai-org/GLM-4.6 --local-dir /opt/dlami/nvme/GLM-4.6/

# python -m sglang.launch_server \
#     --model-path /opt/dlami/nvme/GLM-4.6 \
#     --tp-size 8 \
#     --ep-size 8 \
#     --trust-remote-code \
#     --host 0.0.0.0 \
#     --port 30000 \
#     --mem-fraction-static 0.85

./scripts/deploy_sglang.sh \
    --model-path /opt/dlami/nvme/GLM-4.6/ \
    --model-name glm-4.6 \
    --tp-size 8 \
    --ep-size 8

# Generate data
python scripts/generate_data_with_sglang.py \
    --num_samples_per_profile 2 \
    --include_cm \
    --model_name GLM-4.6 \
    --language zh
```

See [SGLANG_QUICKSTART.md](SGLANG_QUICKSTART.md) for details.

**Comparison**: See [GENERATION_COMPARISON.md](GENERATION_COMPARISON.md)

### Supported Models (SGLang)
- **MiniMax M2** (Recommended for role-playing)
- **GLM-4** (9B, fastest)
- **Qwen2.5-14B/32B** (Strong reasoning)
- **DeepSeek-V2.5** (High quality)
- **Yi-1.5-34B** (Multilingual)

### Multilingual Support

RAIDEN-R1 supports generating training data in multiple languages with **Chinese as the default**:

- **Chinese (zh)**: Default language ✓
- **Japanese (ja)**: 日本語
- **English (en)**: English
- **Korean (ko)**: 한국어

**Auto-detection**: The system automatically detects the language of character profiles using Unicode character ranges (Hiragana/Katakana for Japanese, Hangul for Korean, etc.)

```bash
# Generate Chinese data (default)
python scripts/generate_data_with_sglang.py --language zh

# Generate Japanese data
python scripts/generate_data_with_sglang.py --language ja

# Auto-detect from character profiles (recommended)
python scripts/generate_data_with_sglang.py
```

## Dataset

The training dataset consists of:
- 1,000 samples from RAIDEN Benchmark
- 1,000 challenging role-playing samples
- Generated using Script-Based Knowledge (SBK) and Conversation Memory (CM) strategies

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Generate Training Data

```bash
# Option A: Using SGLang (Recommended - Fast)
./scripts/deploy_sglang.sh --model-path /path/to/model --tp-size 8
python scripts/generate_data_with_sglang.py --num_samples_per_profile 2 --include_cm

# Option B: Using AWS Bedrock
python scripts/generate_data_with_bedrock.py --num_samples_per_profile 2 --include_cm
```

### 3. Train Model

```bash
# Multi-GPU training (8x H200)
accelerate launch --config_file accelerate_config.yaml \
    scripts/train.py --config configs/grpo_config.yaml
```

### 4. Monitor Training

```bash
# Watch GPU utilization
watch -n 1 nvidia-smi

# Check training logs
tail -f outputs/training.log
```

## Troubleshooting

### GPU Utilization is Low

**Problem**: GPUs show 0% utilization or low memory usage

**Solution**:
```bash
# Use accelerate for multi-GPU training
accelerate launch --config_file accelerate_config.yaml scripts/train.py --config configs/grpo_config.yaml

# Increase batch size in configs/grpo_config.yaml
batch_size: 8  # Increase from 4
gradient_accumulation_steps: 2
```

### device_map Conflict Error

**Problem**: `ValueError: You can't train a model that has been loaded with device_map='auto'`

**Solution**: Fixed! Ensure you're using the latest code. The model loading has been updated to work with accelerate distributed training.

### DistributedDataParallel AttributeError

**Problem**: `AttributeError: 'DistributedDataParallel' object has no attribute 'generate'`

**Solution**: Fixed! The code now uses `accelerator.unwrap_model()` to access the original model's `generate()` method.

### Import Errors

**Problem**: `ImportError: cannot import name 'RolePlayingSample'`

**Solution**: The `RolePlayingSample` class is now in `src/data/collection.py`. Ensure you're using the latest code.

### Training Data Format Issues

**Problem**: `AttributeError: 'str' object has no attribute 'get'`

**Solution**: Ensure training data follows the correct format (see Training Data Format section). Regenerate data using the provided scripts.

### Out of Memory (OOM)

**Problem**: CUDA out of memory errors

**Solution**:
```bash
# Reduce batch size
batch_size: 4  # In configs/grpo_config.yaml

# Enable gradient checkpointing (already enabled by default)
gradient_checkpointing: true

# Use fewer GPUs
accelerate launch --num_processes=4 scripts/train.py --config configs/grpo_config.yaml
```

## Results

The RAIDEN-R1 14B-GRPO model achieves:
- **SBK (Script-Based Knowledge)**: 88.04%
- **CM (Conversation Memory)**: 88.65%

## Citation

```bibtex
@article{wang2025raiden,
  title={RAIDEN-R1: Improving Role-awareness of LLMs via GRPO with Verifiable Reward},
  author={Wang, Zongsheng and Sun, Kaili and Wu, Bowen and Yu, Qun and Li, Ying and Wang, Baoxun},
  journal={arXiv preprint arXiv:2505.10218},
  year={2025}
}
```

## License

[Add appropriate license]

## Acknowledgments

This implementation uses:
- Open-R1 library
- RAIDEN Benchmark
- Qwen2.5-14B-Instruct base model
