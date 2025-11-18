#!/bin/bash
# Training script for RAIDEN-R1 with EasyR1
# This script uses the EasyR1 framework to train role-aware LLMs with GRPO

set -e  # Exit on error
set -x  # Print commands

# Configuration
MODEL_PATH=Qwen/Qwen2.5-14B-Instruct  # Change to Qwen2.5-7B-Instruct for smaller GPUs
CONFIG_FILE=configs/easyr1_config.yaml
EXPERIMENT_NAME=qwen2_5_14b_raiden_grpo

# Number of GPUs (adjust based on your hardware)
N_GPUS=8

# Check if EasyR1 is installed
if ! python3 -c "import verl" 2>/dev/null; then
    echo "Error: EasyR1 (verl) is not installed!"
    echo "Please install EasyR1 first:"
    echo "  git clone https://github.com/hiyouga/EasyR1.git"
    echo "  cd EasyR1"
    echo "  pip install -e ."
    exit 1
fi

# Check if data has been converted
if [ ! -f "./data/easyr1/train.json" ]; then
    echo "Converting RAIDEN-R1 data to EasyR1 format..."
    python3 scripts/convert_to_easyr1_format.py \
        --train_input ./data/training/train.json \
        --val_input ./data/training/validation.json \
        --train_output ./data/easyr1/train.json \
        --val_output ./data/easyr1/validation.json \
        --verbose
fi

echo "Starting RAIDEN-R1 training with EasyR1..."
echo "Model: $MODEL_PATH"
echo "Config: $CONFIG_FILE"
echo "GPUs: $N_GPUS"
echo ""

# Run training
python3 -m verl.trainer.main \
    config=${CONFIG_FILE} \
    data.train_files=./data/easyr1/train.json \
    data.val_files=./data/easyr1/validation.json \
    worker.actor.model.model_path=${MODEL_PATH} \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.n_gpus_per_node=${N_GPUS}

echo ""
echo "Training complete!"
echo "Checkpoints saved to: ./checkpoints/easyr1/${EXPERIMENT_NAME}"
echo ""
echo "To merge the checkpoint to Hugging Face format:"
echo "  python3 scripts/model_merger.py --local_dir checkpoints/easyr1/${EXPERIMENT_NAME}/global_step_XXX/actor"
