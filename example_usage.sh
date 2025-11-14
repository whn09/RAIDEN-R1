#!/bin/bash
# Example usage script for RAIDEN-R1 data generation with AWS Bedrock

echo "=================================="
echo "RAIDEN-R1 数据生成示例"
echo "=================================="
echo ""

# Step 1: Configure AWS credentials (if not already configured)
echo "步骤 1: 检查 AWS 配置..."
if aws sts get-caller-identity &>/dev/null; then
    echo "✓ AWS 凭证已配置"
else
    echo "✗ AWS 凭证未配置"
    echo "请运行: aws configure"
    exit 1
fi

# Step 2: Install dependencies
echo ""
echo "步骤 2: 安装依赖..."
if ! python -c "import boto3" &>/dev/null; then
    echo "安装 boto3..."
    pip install boto3 --quiet
fi
echo "✓ 依赖已安装"

# Step 3: Test the generator
echo ""
echo "步骤 3: 测试数据生成器..."
python scripts/test_bedrock_generator.py
if [ $? -ne 0 ]; then
    echo "✗ 测试失败，请检查配置"
    exit 1
fi

# Step 4: Generate small sample dataset (5 samples per profile, first 10 profiles)
echo ""
echo "步骤 4: 生成小规模测试数据集..."
echo "  - 使用前 10 个角色配置"
echo "  - 每个角色生成 2 个 SBK 样本"
echo "  - 包含 CM (对话记忆) 样本"
echo ""

python scripts/generate_data_with_bedrock.py \
    --profiles_file ./data/online_profiles.jsonl \
    --output_file ./data/test_generated_samples.json \
    --output_dir ./data/test_training \
    --num_samples_per_profile 1 \
    --include_cm \
    --region us-east-1

if [ $? -eq 0 ]; then
    echo ""
    echo "=================================="
    echo "✓ 数据生成完成！"
    echo "=================================="
    echo ""
    echo "生成的文件："
    echo "  - 原始样本: ./data/test_generated_samples.json"
    echo "  - 训练集: ./data/test_training/train.json"
    echo "  - 验证集: ./data/test_training/validation.json"
    echo "  - 统计信息: ./data/test_training/dataset_stats.json"
    echo ""
    echo "下一步："
    echo "  1. 查看生成的数据质量"
    echo "  2. 调整参数进行完整数据集生成"
    echo "  3. 使用生成的数据进行 GRPO 训练"
    echo ""
    echo "完整数据生成命令:"
    echo "  python scripts/generate_data_with_bedrock.py \\"
    echo "    --profiles_file ./data/online_profiles.jsonl \\"
    echo "    --output_file ./data/generated_samples.json \\"
    echo "    --output_dir ./data/training \\"
    echo "    --num_samples_per_profile 2 \\"
    echo "    --include_cm"
else
    echo ""
    echo "✗ 数据生成失败"
    echo "请查看错误信息并参考 DATA_GENERATION_GUIDE.md"
    exit 1
fi
