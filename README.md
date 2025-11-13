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
│   ├── data/              # Data processing and collection
│   ├── models/            # Model definitions
│   ├── training/          # Training scripts and GRPO implementation
│   ├── evaluation/        # Evaluation metrics (SBK, CM)
│   └── rewards/           # VRAR reward mechanisms
├── configs/               # Configuration files
├── data/                  # Dataset directory
├── scripts/               # Utility scripts
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

```bash
python scripts/train.py --config configs/grpo_config.yaml
```

## Evaluation

```bash
python scripts/evaluate.py --model_path <path_to_model> --eval_data <path_to_eval_data>
```

## Dataset

The training dataset consists of:
- 1,000 samples from RAIDEN Benchmark
- 1,000 challenging role-playing samples

## Training Details

- **Base Model**: Qwen2.5-14B-Instruct
- **Learning Rate**: 3e-6 with cosine scheduler
- **Batch Size**: 4 per GPU
- **Training Duration**: 1 epoch
- **Precision**: bf16
- **Hardware**: 8x NVIDIA H800 GPUs

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
