# Quick Start: Training RAIDEN-R1 with EasyR1

This is a minimal guide to get started with EasyR1 for RAIDEN-R1 training.

## Prerequisites

- 8x NVIDIA H800/H200 GPUs (or 4x for Qwen2.5-7B)
- Python 3.9+
- CUDA 12.x

## Installation (5 minutes)

### Option 1: Docker (Recommended)

```bash
# Pull pre-built image with all dependencies
docker pull hiyouga/verl:ngc-th2.8.0-cu12.9-vllm0.11.0

# Run container
docker run -it --ipc=host --gpus=all \
    -v $(pwd):/workspace \
    hiyouga/verl:ngc-th2.8.0-cu12.9-vllm0.11.0

cd /workspace
```

### Option 2: Local Installation

```bash
# 1. Clone and install EasyR1
git clone https://github.com/hiyouga/EasyR1.git
cd EasyR1
pip install -e .
cd ..

# 2. Install additional dependencies
pip install -r easyr1_requirements.txt
```

## Verify Installation

```bash
python3 scripts/test_easyr1_integration.py
```

You should see:
```
✓ Data Conversion......................... ✓ PASS
✓ Reward Function......................... ✓ PASS
✓ Config Files............................ ✓ PASS
✓ Prompt Template......................... ✓ PASS
✓ EasyR1 Installation..................... ✓ PASS
```

## Training (3 steps)

### Step 1: Prepare Data

If you have RAIDEN-R1 training data:

```bash
python3 scripts/convert_to_easyr1_format.py \
    --train_input ./data/training/train.json \
    --val_input ./data/training/validation.json \
    --train_output ./data/easyr1/train.json \
    --val_output ./data/easyr1/validation.json \
    --verbose
```

### Step 2: Configure Training

Edit `configs/easyr1_config.yaml` if needed:

```yaml
# For smaller GPUs, use Qwen2.5-7B
worker:
  actor:
    model:
      model_path: Qwen/Qwen2.5-7B-Instruct  # or Qwen2.5-14B-Instruct

# Adjust GPU count
trainer:
  n_gpus_per_node: 8  # or 4 for smaller setup
```

### Step 3: Start Training

```bash
bash scripts/train_with_easyr1.sh
```

That's it! Training will start automatically.

## Monitor Training

In separate terminals:

```bash
# GPU utilization
watch -n 1 nvidia-smi

# Training logs
tail -f checkpoints/easyr1/qwen2_5_14b_raiden_grpo/logs/trainer.log

# TensorBoard (optional)
tensorboard --logdir checkpoints/easyr1/qwen2_5_14b_raiden_grpo/tensorboard
```

## After Training

Merge checkpoint to Hugging Face format:

```bash
bash scripts/merge_easyr1_checkpoint.sh \
    checkpoints/easyr1/qwen2_5_14b_raiden_grpo/global_step_100/actor
```

The merged model will be in:
```
checkpoints/easyr1/qwen2_5_14b_raiden_grpo/global_step_100/actor_merged/
```

## Troubleshooting

### Out of Memory?

Reduce batch size in `configs/easyr1_config.yaml`:

```yaml
data:
  rollout_batch_size: 128  # Reduce from 256

worker:
  actor:
    global_batch_size: 64   # Reduce from 128
  rollout:
    gpu_memory_utilization: 0.5  # Reduce from 0.6
```

### Training too slow?

Enable more optimizations:

```yaml
worker:
  actor:
    padding_free: true  # Already enabled
    dynamic_batching: true  # Already enabled
  rollout:
    tensor_parallel_size: 4  # Increase if you have more GPUs
```

## Expected Results

After 1 epoch (~8-12 hours on 8x H800):
- **SBK (Script-Based Knowledge)**: ~88%
- **CM (Conversation Memory)**: ~88%

## Next Steps

- See [EASYR1_GUIDE.md](EASYR1_GUIDE.md) for detailed documentation
- Check [configs/easyr1_config.yaml](configs/easyr1_config.yaml) for all options
- Adjust reward weights in the config if needed

## Common Commands

```bash
# Test integration
python3 scripts/test_easyr1_integration.py

# Convert data
python3 scripts/convert_to_easyr1_format.py --verbose

# Train
bash scripts/train_with_easyr1.sh

# Monitor
tail -f checkpoints/easyr1/*/logs/trainer.log

# Merge checkpoint
bash scripts/merge_easyr1_checkpoint.sh <checkpoint_path>
```

## Support

- EasyR1 GitHub: https://github.com/hiyouga/EasyR1
- RAIDEN-R1 Paper: https://arxiv.org/abs/2505.10218
- Issues: Open an issue on GitHub
