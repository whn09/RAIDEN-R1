#!/bin/bash
# Merge EasyR1 checkpoint to Hugging Face format
# This script merges the FSDP checkpoint into a single Hugging Face compatible model

set -e

if [ -z "$1" ]; then
    echo "Usage: $0 <checkpoint_path>"
    echo ""
    echo "Example:"
    echo "  $0 checkpoints/easyr1/qwen2_5_14b_raiden_grpo/global_step_100/actor"
    echo ""
    echo "Available checkpoints:"
    ls -d checkpoints/easyr1/*/global_step_*/actor 2>/dev/null || echo "  No checkpoints found"
    exit 1
fi

CHECKPOINT_PATH=$1

if [ ! -d "$CHECKPOINT_PATH" ]; then
    echo "Error: Checkpoint path does not exist: $CHECKPOINT_PATH"
    exit 1
fi

echo "Merging checkpoint from: $CHECKPOINT_PATH"
echo ""

# Check if EasyR1's model_merger script exists
if [ ! -f "scripts/model_merger.py" ]; then
    # Try to find it in EasyR1 installation
    EASYR1_PATH=$(python3 -c "import verl; import os; print(os.path.dirname(verl.__file__) + '/../scripts')" 2>/dev/null)
    if [ -f "$EASYR1_PATH/model_merger.py" ]; then
        python3 "$EASYR1_PATH/model_merger.py" --local_dir "$CHECKPOINT_PATH"
    else
        echo "Error: model_merger.py not found!"
        echo "Please ensure EasyR1 is properly installed."
        exit 1
    fi
else
    python3 scripts/model_merger.py --local_dir "$CHECKPOINT_PATH"
fi

echo ""
echo "Merge complete!"
echo "Merged model saved to: ${CHECKPOINT_PATH}_merged"
