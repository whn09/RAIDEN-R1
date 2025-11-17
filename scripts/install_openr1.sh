#!/bin/bash
# Quick installation script for OpenR1

set -e

echo "=============================================="
echo "Installing OpenR1 for RAIDEN-R1"
echo "=============================================="
echo ""

# Check if OpenR1 is already installed
if python -c "from open_r1.configs import GRPOConfig; from trl import GRPOTrainer" 2>/dev/null; then
    echo "✓ OpenR1 is already installed and working"
    python -c "import open_r1; print(f'  Version: {getattr(open_r1, \"__version__\", \"unknown\")}')"
    echo ""
    read -p "Reinstall? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Skipping installation."
        exit 0
    fi
fi

# Create temp directory
TEMP_DIR=$(mktemp -d)
cd "$TEMP_DIR"

echo "Cloning OpenR1 repository..."
git clone https://github.com/huggingface/open-r1.git
cd open-r1

echo ""
echo "Installing OpenR1..."
pip install -e ".[dev]"

echo ""
echo "Verifying installation..."
if python -c "from open_r1.configs import GRPOConfig; from trl import GRPOTrainer; import open_r1; print('✓ All required imports working')" 2>/dev/null; then
    echo "✓ OpenR1 installed successfully!"
    python -c "import open_r1; print(f'  Version: {getattr(open_r1, \"__version__\", \"unknown\")}')"
else
    echo "✗ Installation failed. Please install manually:"
    echo "  git clone https://github.com/huggingface/open-r1.git"
    echo "  cd open-r1"
    echo "  pip install -e \".[dev]\""
    echo "  pip install trl"
    exit 1
fi

# Clean up
cd ~
rm -rf "$TEMP_DIR"

echo ""
echo "=============================================="
echo "Installation complete!"
echo "=============================================="
echo ""
echo "Next steps:"
echo "  1. Test integration: python scripts/test_openr1_integration.py"
echo "  2. Generate data: python scripts/generate_data_with_sglang.py --language zh"
echo "  3. Train model: python scripts/train_with_openr1.py configs/openr1_config.yaml"
echo ""
