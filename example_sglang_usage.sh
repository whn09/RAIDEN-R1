#!/bin/bash
# Complete example for RAIDEN-R1 data generation with SGLang

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${GREEN}=================================="
echo "RAIDEN-R1 SGLang 数据生成完整示例"
echo -e "==================================${NC}"
echo ""

# Configuration
MODEL_PATH="/path/to/your/model"  # 修改为你的模型路径
MODEL_NAME="your-model"            # 修改为你的模型名称
PORT=30000
TP_SIZE=8

echo -e "${YELLOW}请先修改此脚本中的 MODEL_PATH 和 MODEL_NAME${NC}"
echo ""
echo "推荐配置:"
echo "  - MiniMax M2: MODEL_PATH=/path/to/MiniMax-Text-01"
echo "  - GLM-4: MODEL_PATH=/path/to/glm-4-9b-chat"
echo "  - Qwen2.5-14B: MODEL_PATH=/path/to/Qwen2.5-14B-Instruct"
echo ""

read -p "按 Enter 继续，或 Ctrl+C 取消..."

# Step 1: Check GPU
echo ""
echo -e "${BLUE}步骤 1: 检查 GPU 状态...${NC}"
if ! command -v nvidia-smi &> /dev/null; then
    echo -e "${RED}✗ nvidia-smi 未找到${NC}"
    exit 1
fi

GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
echo -e "${GREEN}✓ 检测到 $GPU_COUNT 块 GPU${NC}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Step 2: Install SGLang if needed
echo ""
echo -e "${BLUE}步骤 2: 检查 SGLang 安装...${NC}"
if ! python -c "import sglang" 2>/dev/null; then
    echo -e "${YELLOW}安装 SGLang...${NC}"
    pip install "sglang[all]" --quiet
fi
echo -e "${GREEN}✓ SGLang 已安装${NC}"

# Step 3: Deploy SGLang server
echo ""
echo -e "${BLUE}步骤 3: 部署 SGLang 服务器...${NC}"

if [ "$MODEL_PATH" = "/path/to/your/model" ]; then
    echo -e "${RED}✗ 请先修改脚本中的 MODEL_PATH${NC}"
    exit 1
fi

./scripts/deploy_sglang.sh \
    --model-path "$MODEL_PATH" \
    --model-name "$MODEL_NAME" \
    --port "$PORT" \
    --tp-size "$TP_SIZE"

if [ $? -ne 0 ]; then
    echo -e "${RED}✗ 服务器部署失败${NC}"
    exit 1
fi

# Step 4: Test the generator
echo ""
echo -e "${BLUE}步骤 4: 测试数据生成器...${NC}"
python scripts/test_sglang_generator.py

if [ $? -ne 0 ]; then
    echo -e "${RED}✗ 测试失败${NC}"
    exit 1
fi

# Step 5: Generate test dataset
echo ""
echo -e "${BLUE}步骤 5: 生成测试数据集 (10 个角色)...${NC}"

python scripts/generate_data_with_sglang.py \
    --profiles_file ./data/online_profiles.jsonl \
    --output_file ./data/test_generated_samples_sglang.json \
    --output_dir ./data/test_training_sglang \
    --num_samples_per_profile 1 \
    --max_profiles 10 \
    --base_url "http://localhost:$PORT" \
    --model_name "$MODEL_NAME"

if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}=================================="
    echo "✓ 测试数据生成完成！"
    echo -e "==================================${NC}"
    echo ""
    echo "生成的文件："
    echo "  - 原始样本: ./data/test_generated_samples_sglang.json"
    echo "  - 训练集: ./data/test_training_sglang/train.json"
    echo "  - 验证集: ./data/test_training_sglang/validation.json"
    echo "  - 统计信息: ./data/test_training_sglang/dataset_stats.json"
    echo ""

    # Show statistics
    if [ -f "./data/test_training_sglang/dataset_stats.json" ]; then
        echo -e "${BLUE}数据集统计:${NC}"
        cat ./data/test_training_sglang/dataset_stats.json | jq
    fi

    echo ""
    echo -e "${GREEN}下一步:${NC}"
    echo ""
    echo "1. 查看生成的数据质量:"
    echo "   cat ./data/test_training_sglang/train.json | jq '.[0]'"
    echo ""
    echo "2. 生成完整数据集:"
    echo "   python scripts/generate_data_with_sglang.py \\"
    echo "     --profiles_file ./data/online_profiles.jsonl \\"
    echo "     --output_dir ./data/training \\"
    echo "     --num_samples_per_profile 2 \\"
    echo "     --include_cm \\"
    echo "     --base_url http://localhost:$PORT"
    echo ""
    echo "3. 停止 SGLang 服务器:"
    echo "   kill \$(cat sglang_server_info.json | jq -r .pid)"
    echo ""
    echo "4. 查看服务器日志:"
    echo "   tail -f logs/sglang_*.log"
else
    echo ""
    echo -e "${RED}✗ 数据生成失败${NC}"
    exit 1
fi
