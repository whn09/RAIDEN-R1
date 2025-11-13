# RAIDEN-R1 Quick Start Guide

This guide will help you get started with RAIDEN-R1 training and evaluation.

## Installation

1. Clone the repository and install dependencies:

```bash
cd raiden-r1
pip install -r requirements.txt
```

## Step 1: Prepare Dataset

Generate or prepare your training dataset using the provided script:

```bash
python scripts/prepare_dataset.py \
    --num_custom_samples 1000 \
    --characters_file configs/example_characters.json \
    --output_dir ./data
```

If you have the RAIDEN Benchmark dataset, you can include it:

```bash
python scripts/prepare_dataset.py \
    --benchmark_path /path/to/raiden_benchmark.json \
    --num_custom_samples 1000 \
    --characters_file configs/example_characters.json \
    --output_dir ./data
```

This will create:
- `data/train_dataset.json` - Training dataset
- `data/eval_dataset.json` - Evaluation dataset

## Step 2: Train the Model

Train RAIDEN-R1 using GRPO:

### Basic Training

```bash
python scripts/train.py \
    --train_data ./data/train_dataset.json \
    --eval_data ./data/eval_dataset.json \
    --model_name Qwen/Qwen2.5-14B-Instruct \
    --output_dir ./outputs
```

### Full Training with Custom Parameters

```bash
python scripts/train.py \
    --train_data ./data/train_dataset.json \
    --eval_data ./data/eval_dataset.json \
    --model_name Qwen/Qwen2.5-14B-Instruct \
    --learning_rate 3e-6 \
    --batch_size 4 \
    --num_epochs 1 \
    --num_samples_per_prompt 4 \
    --accuracy_weight 0.7 \
    --format_weight 0.3 \
    --output_dir ./outputs \
    --use_bf16 \
    --gradient_checkpointing
```

### Hardware Requirements

The paper uses:
- 8x NVIDIA H800 GPUs
- bf16 precision
- Batch size of 4 per GPU

For smaller setups, you can:
- Reduce `--batch_size` to 1 or 2
- Increase `--gradient_accumulation_steps` to compensate
- Use a smaller base model

## Step 3: Evaluate the Model

Evaluate your trained model on the test set:

```bash
python scripts/evaluate.py \
    --model_path ./outputs/epoch_0 \
    --eval_data ./data/eval_dataset.json \
    --output_file ./evaluation_results.json
```

With custom evaluation weights:

```bash
python scripts/evaluate.py \
    --model_path ./outputs/epoch_0 \
    --eval_data ./data/eval_dataset.json \
    --output_file ./evaluation_results.json \
    --sbk_weight 0.5 \
    --cm_weight 0.5
```

## Understanding the Metrics

### Script-Based Knowledge (SBK)
Measures how well the model maintains consistency with the character's:
- Background and history
- Personality traits
- Known skills and abilities
- Character-specific knowledge

**Target**: 88.04% (as reported in paper)

### Conversation Memory (CM)
Measures how well the model maintains consistency with:
- Previous statements in conversation
- Earlier claims or information shared
- Conversation context and flow

**Target**: 88.65% (as reported in paper)

## Customization

### Adding Custom Characters

Edit `configs/example_characters.json` to add your own characters:

```json
{
  "name": "Your Character",
  "occupation": "Their occupation",
  "personality": "Personality description",
  "background": "Background story",
  "skills": ["skill1", "skill2"],
  "traits": {
    "key": "value"
  }
}
```

### Adjusting Reward Weights

The VRAR reward consists of two components:

- **Accuracy Weight** (default: 0.7): How much to weight correctness of answers
- **Format Weight** (default: 0.3): How much to weight proper reasoning format

Adjust these in training:

```bash
python scripts/train.py \
    --accuracy_weight 0.8 \
    --format_weight 0.2 \
    ...
```

### GRPO Parameters

Key GRPO parameters:

- `--num_samples_per_prompt`: Number of responses to generate per prompt (default: 4)
  - Higher = more diverse group, better GRPO signal, but slower
  - Lower = faster training, less diverse

- `--learning_rate`: Learning rate (default: 3e-6)
  - Paper uses 3e-6 with cosine scheduler

## Troubleshooting

### Out of Memory

If you run out of GPU memory:

1. Reduce batch size: `--batch_size 1`
2. Enable gradient checkpointing: `--gradient_checkpointing`
3. Use gradient accumulation: `--gradient_accumulation_steps 8`
4. Reduce `--num_samples_per_prompt` to 2

### Slow Training

1. Increase batch size if you have GPU memory
2. Reduce `--num_samples_per_prompt` to 2
3. Use mixed precision: `--use_bf16`

### Poor Results

1. Check if your dataset quality is good
2. Increase training epochs (paper uses 1 epoch on 2000 samples)
3. Adjust reward weights
4. Try different `--num_samples_per_prompt` values

## Expected Results

Based on the paper, the RAIDEN-R1 14B-GRPO model achieves:

- **SBK**: 88.04%
- **CM**: 88.65%

Your results may vary depending on:
- Base model quality
- Dataset quality and size
- Training hyperparameters
- Hardware configuration

## Citation

If you use this implementation in your research, please cite:

```bibtex
@article{wang2025raiden,
  title={RAIDEN-R1: Improving Role-awareness of LLMs via GRPO with Verifiable Reward},
  author={Wang, Zongsheng and Sun, Kaili and Wu, Bowen and Yu, Qun and Li, Ying and Wang, Baoxun},
  journal={arXiv preprint arXiv:2505.10218},
  year={2025}
}
```

## Support

For issues and questions:
- Check the main README.md
- Review the paper at https://arxiv.org/html/2505.10218v1
- Examine the code in `src/` directory

## Next Steps

1. Experiment with different characters
2. Try different base models (e.g., Llama, Mistral)
3. Adjust GRPO hyperparameters
4. Collect your own role-playing dataset
5. Implement LLM-as-a-judge evaluation (Claude 3.5 API)
