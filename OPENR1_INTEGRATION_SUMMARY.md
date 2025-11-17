# OpenR1 Integration Summary

This document summarizes the OpenR1 integration for RAIDEN-R1.

## What Was Added

### 1. Core Integration Files

#### `src/training/openr1_adapter.py`
- **Purpose**: Adapts RAIDEN data format to OpenR1's expected format
- **Key Classes**:
  - `RAIDENDataAdapter`: Converts RAIDEN samples to OpenR1 format
  - `RAIDENRewardFunction`: Implements VRAR rewards for OpenR1's GRPO
- **Function**: `create_raiden_reward_function()` - Factory for reward functions

#### `scripts/train_with_openr1.py`
- **Purpose**: Main training script using OpenR1's GRPO
- **Features**:
  - Command-line argument parsing
  - Distributed training support via Accelerate
  - VRAR reward integration
  - Model checkpointing and evaluation
- **Usage**:
  ```bash
  python scripts/train_with_openr1.py configs/openr1_config.yaml
  ```

#### `configs/openr1_config.yaml`
- **Purpose**: Configuration file for OpenR1 training
- **Parameters**:
  - Model settings (Qwen2.5-14B-Instruct)
  - VRAR weights (accuracy: 0.7, format: 0.3)
  - GRPO parameters (samples: 4, KL penalty: 0.1)
  - Training hyperparameters (lr: 3e-6, batch_size: 4)

### 2. Documentation

#### `OPENR1_GUIDE.md`
- **Purpose**: Comprehensive guide to using OpenR1 with RAIDEN
- **Sections**:
  - Quick Start
  - Architecture explanation
  - Configuration options
  - Troubleshooting
  - Advanced usage

### 3. Testing

#### `scripts/test_openr1_integration.py`
- **Purpose**: Verify OpenR1 integration is working correctly
- **Tests**:
  - OpenR1 installation
  - Data adapter functionality
  - Reward function calculation
  - Required dependencies
- **Usage**:
  ```bash
  python scripts/test_openr1_integration.py
  ```

### 4. Updated Files

#### `requirements.txt`
- Added OpenR1 dependency (commented out, optional):
  ```
  # open-r1>=0.1.0  # Uncomment to use OpenR1
  ```

#### `README.md`
- Added "Training" section with two options:
  - Option A: Custom GRPO (original implementation)
  - Option B: OpenR1 GRPO (paper-accurate)
- Added comparison table
- Added link to OPENR1_GUIDE.md

## Architecture Overview

```
RAIDEN Data (JSON)
    ↓
RAIDENDataAdapter
    ↓
OpenR1 Format (HuggingFace Dataset)
    ↓
GRPOTrainer (OpenR1)
    ↓
RAIDENRewardFunction (VRAR)
    ↓
Trained Model
```

## Key Differences from Custom GRPO

| Aspect | Custom GRPO | OpenR1 GRPO |
|--------|-------------|-------------|
| Implementation | Custom in-house | HuggingFace library |
| Paper accuracy | Approximate | Exact (as cited) |
| Optimization | Basic | Production-grade |
| Maintenance | Manual | Community-maintained |
| Complexity | Simple | More complex |
| Performance | Good | Optimized |
| Customization | Easy | Moderate |

## Quick Start

### 1. Install OpenR1

```bash
# OpenR1 must be installed from source
git clone https://github.com/huggingface/open-r1.git
cd open-r1
pip install -e ".[dev]"
cd ..
```

### 2. Test Integration

```bash
python scripts/test_openr1_integration.py
```

### 3. Generate Data

```bash
python scripts/generate_data_with_sglang.py \
    --language zh \
    --num_samples_per_profile 2
```

### 4. Train

```bash
# Simple command line
python scripts/train_with_openr1.py \
    --train_data_path ./data/training/train.json \
    --model_name_or_path Qwen/Qwen2.5-14B-Instruct \
    --output_dir ./outputs_openr1

# Or with config
python scripts/train_with_openr1.py configs/openr1_config.yaml
```

## When to Use Which Implementation?

### Use Custom GRPO (`scripts/train.py`) if:
- ✅ You want simplicity
- ✅ You're experimenting with modifications
- ✅ You want to understand the implementation details
- ✅ You don't want extra dependencies

### Use OpenR1 GRPO (`scripts/train_with_openr1.py`) if:
- ✅ You want paper-accurate results
- ✅ You need production-grade performance
- ✅ You want community support
- ✅ You're doing serious experiments or deployment

## Implementation Notes

### Data Format Conversion

RAIDEN stores data with character profiles and keywords:
```json
{
  "character_name": "小明",
  "character_profile": {...},
  "question": "你叫什么名字？",
  "answer": "我叫小明。",
  "keywords": ["小明"]
}
```

OpenR1 expects conversation format:
```python
{
  "prompt": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."}
  ]
}
```

The adapter handles this conversion automatically.

### Reward Function

The VRAR reward is calculated as:
```python
reward = accuracy_weight * accuracy_reward + format_weight * format_reward
```

Where:
- `accuracy_reward`: Based on keyword matching (0 or 1)
- `format_reward`: Based on response format quality (0 to 1)

Default weights: 0.7 for accuracy, 0.3 for format (as per paper).

### GRPO Parameters

Key GRPO parameters explained:
- `num_samples_per_prompt`: How many responses to generate per prompt (default: 4)
- `kl_penalty`: Penalty for diverging from reference model (default: 0.1)
- Higher samples → better exploration, slower training
- Higher KL penalty → stays closer to base model

## Troubleshooting

### Common Issues

1. **ImportError: No module named 'open_r1'**
   ```bash
   # OpenR1 must be installed from source
   git clone https://github.com/huggingface/open-r1.git
   cd open-r1
   pip install -e ".[dev]"
   cd ..
   ```

2. **CUDA out of memory**
   - Reduce `per_device_train_batch_size` in config
   - Enable `gradient_checkpointing: true`

3. **Data format errors**
   - Regenerate data with `generate_data_with_sglang.py`
   - Run `test_openr1_integration.py` to verify

4. **Slow training**
   - Ensure bf16 is enabled on compatible GPUs
   - Check GPU utilization with `nvidia-smi`
   - Consider using more gradient accumulation steps

## Files Summary

**New files**:
- `src/training/openr1_adapter.py` (217 lines)
- `scripts/train_with_openr1.py` (246 lines)
- `scripts/test_openr1_integration.py` (202 lines)
- `configs/openr1_config.yaml` (42 lines)
- `OPENR1_GUIDE.md` (418 lines)
- `OPENR1_INTEGRATION_SUMMARY.md` (this file)

**Modified files**:
- `requirements.txt` (added OpenR1 dependency)
- `README.md` (added OpenR1 training option)

**Total**: 6 new files, 2 modified files, ~1100 lines of code and documentation

## Testing

Run the test suite to verify everything works:

```bash
# Test OpenR1 integration
python scripts/test_openr1_integration.py

# Test with small model (faster)
python scripts/train_with_openr1.py \
    --train_data_path ./data/training/train.json \
    --model_name_or_path Qwen/Qwen2.5-0.5B-Instruct \
    --output_dir ./test_outputs \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2
```

## Support

- **RAIDEN issues**: [Open an issue](https://github.com/your-repo/RAIDEN-R1/issues)
- **OpenR1 issues**: [OpenR1 GitHub](https://github.com/huggingface/open-r1/issues)
- **Documentation**: See `OPENR1_GUIDE.md` for detailed usage

## References

- [RAIDEN Paper](https://arxiv.org/abs/2505.10218)
- [OpenR1 Repository](https://github.com/huggingface/open-r1)
- [GRPO Paper](https://arxiv.org/abs/2402.03300)
