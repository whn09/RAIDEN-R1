# OpenR1 Integration Guide for RAIDEN-R1

This guide explains how to use OpenR1's GRPO implementation for training RAIDEN-R1, as mentioned in the original paper.

## What is OpenR1?

[OpenR1](https://github.com/huggingface/open-r1) is Hugging Face's fully open reproduction of DeepSeek-R1, providing a production-ready GRPO (Group Relative Policy Optimization) implementation.

**Why OpenR1?**
- ✅ Paper-accurate: RAIDEN paper explicitly uses OpenR1
- ✅ Optimized: Battle-tested GRPO implementation
- ✅ Maintained: Active development by Hugging Face
- ✅ Flexible: Easy to integrate custom rewards

## Installation

```bash
# Install OpenR1
pip install open-r1

# Or install from source for latest features
git clone https://github.com/huggingface/open-r1.git
cd open-r1
pip install -e .
```

## Quick Start

### 1. Prepare Your Data

Generate training data using GLM-4.6 (recommended):

```bash
python scripts/generate_data_with_sglang.py \
    --language zh \
    --num_samples_per_profile 2 \
    --include_cm \
    --output_file ./data/generated_samples.json
```

This creates `./data/training/train.json` and `./data/training/validation.json`.

### 2. Configure Training

Edit `configs/openr1_config.yaml`:

```yaml
train_data_path: "./data/training/train.json"
eval_data_path: "./data/training/validation.json"
model_name_or_path: "Qwen/Qwen2.5-14B-Instruct"

# VRAR weights
accuracy_weight: 0.7
format_weight: 0.3

# Training params
num_train_epochs: 1
per_device_train_batch_size: 4
learning_rate: 3.0e-6
```

### 3. Train

```bash
# Single command
python scripts/train_with_openr1.py \
    --train_data_path ./data/training/train.json \
    --eval_data_path ./data/training/validation.json \
    --model_name_or_path Qwen/Qwen2.5-14B-Instruct \
    --output_dir ./outputs_openr1

# Or with config file (recommended)
python scripts/train_with_openr1.py configs/openr1_config.yaml
```

### 4. Monitor

```bash
# Watch GPU usage
watch -n 1 nvidia-smi

# View logs
tail -f outputs_openr1/training.log
```

## Architecture

The OpenR1 integration consists of three components:

### 1. Data Adapter (`src/training/openr1_adapter.py`)

Converts RAIDEN's data format to OpenR1's expected format:

```python
from training.openr1_adapter import RAIDENDataAdapter

adapter = RAIDENDataAdapter()
train_dataset = adapter.load_raiden_dataset("./data/training/train.json")
```

**Input (RAIDEN format)**:
```json
{
  "character_name": "小明",
  "question": "你叫什么名字？",
  "answer": "我叫小明。",
  "keywords": ["小明"],
  "character_profile": {...}
}
```

**Output (OpenR1 format)**:
```python
{
  "prompt": [
    {"role": "system", "content": "You are a role-playing AI..."},
    {"role": "user", "content": "Character: 小明\n...\nUser: 你叫什么名字？"}
  ],
  "expected_answer": "我叫小明。",
  "keywords": ["小明"],
  ...
}
```

### 2. Reward Function (`src/training/openr1_adapter.py`)

Implements VRAR (Verifiable Role-Awareness Rewards) compatible with OpenR1:

```python
from training.openr1_adapter import create_raiden_reward_function

reward_fn = create_raiden_reward_function(
    tokenizer=tokenizer,
    accuracy_weight=0.7,
    format_weight=0.3
)
```

The reward function:
1. Validates answers against keywords
2. Calculates accuracy reward (0.7 weight)
3. Calculates format reward (0.3 weight)
4. Returns combined VRAR score

### 3. Training Script (`scripts/train_with_openr1.py`)

Orchestrates the training process:

```python
# Load data
train_dataset = adapter.load_raiden_dataset(train_data_path)

# Create reward function
reward_fn = create_raiden_reward_function(tokenizer, 0.7, 0.3)

# Initialize GRPO trainer
trainer = GRPOTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    reward_function=reward_fn,
    args=grpo_config
)

# Train
trainer.train()
```

## Configuration Options

### Core Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model_name_or_path` | Qwen/Qwen2.5-14B-Instruct | Base model |
| `accuracy_weight` | 0.7 | VRAR accuracy weight |
| `format_weight` | 0.3 | VRAR format weight |
| `num_samples_per_prompt` | 4 | GRPO group size |
| `kl_penalty` | 0.1 | KL divergence penalty |

### Training Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_train_epochs` | 1 | Training epochs |
| `per_device_train_batch_size` | 4 | Batch size per GPU |
| `gradient_accumulation_steps` | 2 | Gradient accumulation |
| `learning_rate` | 3e-6 | Learning rate |
| `warmup_steps` | 100 | Warmup steps |

### System

| Parameter | Default | Description |
|-----------|---------|-------------|
| `bf16` | true | Use bfloat16 precision |
| `gradient_checkpointing` | true | Enable checkpointing |
| `output_dir` | ./outputs_openr1 | Output directory |

## Comparison: Custom vs OpenR1

### Custom GRPO (`scripts/train.py`)

**Pros**:
- ✅ Simple implementation
- ✅ Easy to understand and modify
- ✅ No additional dependencies
- ✅ Good for learning and experimentation

**Cons**:
- ❌ Less optimized
- ❌ May have edge cases
- ❌ Not paper-accurate

### OpenR1 (`scripts/train_with_openr1.py`)

**Pros**:
- ✅ Paper-accurate (as cited in RAIDEN)
- ✅ Production-tested
- ✅ Optimized performance
- ✅ Active maintenance by HF
- ✅ Community support

**Cons**:
- ❌ Additional dependency
- ❌ More complex internals
- ❌ Requires understanding OpenR1 API

## Troubleshooting

### ImportError: No module named 'open_r1'

```bash
pip install open-r1
# Or from source:
pip install git+https://github.com/huggingface/open-r1.git
```

### Data Format Errors

Ensure your data follows RAIDEN format:
- Use `generate_data_with_sglang.py` to generate compatible data
- Check that JSON has required fields: `character_name`, `question`, `answer`, `keywords`

### Memory Issues

Reduce batch size or enable gradient checkpointing:

```yaml
per_device_train_batch_size: 2  # Reduce from 4
gradient_checkpointing: true     # Enable if not already
```

### Performance Tips

1. **Use bf16**: Always enable for A100/H100/H800 GPUs
2. **Tune batch size**: Maximize GPU utilization without OOM
3. **Gradient accumulation**: Increase effective batch size
4. **Warmup steps**: Important for stable training start

## Advanced Usage

### Custom System Prompt

```yaml
system_prompt: "你是一个专业的角色扮演AI助手。请根据角色设定回答问题，保持角色的性格、背景和知识一致性。"
```

### Multi-GPU Training

OpenR1 automatically supports distributed training via Accelerate:

```bash
# Configure accelerate
accelerate config

# Train
accelerate launch scripts/train_with_openr1.py configs/openr1_config.yaml
```

### Custom Reward Weights

Experiment with different VRAR weights:

```yaml
accuracy_weight: 0.8  # Emphasize answer correctness
format_weight: 0.2    # De-emphasize format

# Or
accuracy_weight: 0.6  # Balance both aspects
format_weight: 0.4
```

## References

- [OpenR1 GitHub](https://github.com/huggingface/open-r1)
- [RAIDEN Paper](https://arxiv.org/abs/2505.10218)
- [GRPO Paper](https://arxiv.org/abs/2402.03300)

## Support

If you encounter issues with:
- **RAIDEN integration**: Open an issue in this repo
- **OpenR1 library**: Check [OpenR1's issues](https://github.com/huggingface/open-r1/issues)
