# RAIDEN-R1 快速开始 - AWS Bedrock 数据生成

使用 AWS Bedrock Claude 3.5 Sonnet 快速生成 RAIDEN 训练数据。

## 快速开始 (3 步骤)

### 1. 配置 AWS 凭证

```bash
# 方法 1: 使用 AWS CLI
aws configure

# 方法 2: 环境变量
export AWS_ACCESS_KEY_ID="your-access-key"
export AWS_SECRET_ACCESS_KEY="your-secret-key"
export AWS_DEFAULT_REGION="us-east-1"
```

### 2. 安装依赖

```bash
pip install boto3
```

### 3. 运行示例脚本

```bash
./example_usage.sh
```

这将：
- ✓ 检查 AWS 配置
- ✓ 测试数据生成器
- ✓ 生成小规模测试数据集
- ✓ 显示生成的文件位置

## 手动运行

### 测试连接

```bash
python scripts/test_bedrock_generator.py
```

### 生成完整数据集

```bash
python scripts/generate_data_with_bedrock.py \
    --profiles_file ./data/online_profiles.jsonl \
    --output_file ./data/generated_samples.json \
    --output_dir ./data/training \
    --num_samples_per_profile 2 \
    --include_cm
```

## 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--profiles_file` | 角色配置文件 (JSONL) | `./data/online_profiles.jsonl` |
| `--num_samples_per_profile` | 每个角色的样本数 | `2` |
| `--include_cm` | 生成对话记忆样本 | `False` |
| `--region` | AWS 区域 | `us-east-1` |
| `--language` | 默认语言 (zh/ja/en/ko) | `zh` (中文) |
| `--no_auto_detect` | 禁用自动语言检测 | `False` |
| `--output_file` | 输出文件路径 | `./data/generated_samples.json` |
| `--output_dir` | 训练数据目录 | `./data/training` |

## 🌍 多语言支持

RAIDEN-R1 支持多语言数据生成，默认使用**中文**。

### 支持的语言

- **中文 (zh)**: 默认语言 ✓
- **日文 (ja)**: Japanese
- **英文 (en)**: English
- **韩文 (ko)**: Korean

### 语言配置示例

```bash
# 使用中文生成（默认）
python scripts/generate_data_with_bedrock.py \
    --language zh

# 使用日文生成
python scripts/generate_data_with_bedrock.py \
    --language ja

# 使用英文生成
python scripts/generate_data_with_bedrock.py \
    --language en
```

### 自动语言检测

系统会自动检测角色配置文件的语言：

```bash
# 自动检测语言（推荐）
python scripts/generate_data_with_bedrock.py

# 强制使用指定语言
python scripts/generate_data_with_bedrock.py \
    --language zh \
    --no_auto_detect
```

## 生成的文件

```
data/
├── generated_samples.json      # 原始样本
└── training/
    ├── train.json             # 训练集 (90%)
    ├── validation.json        # 验证集 (10%)
    └── dataset_stats.json     # 统计信息
```

## 数据格式示例

```json
{
  "character_name": "角色名称",
  "question": "What is the character's favorite hobby?",
  "answer": "The character enjoys reading and hiking.",
  "keywords": ["reading", "hiking"],
  "question_type": "what",
  "validation_method": "multi_term_parsing",
  "difficulty": "medium",
  "metadata": {
    "reasoning": "Based on the character profile...",
    "focus": "script_knowledge"
  }
}
```

## 两种数据类型

### 1. SBK (Script-Based Knowledge)
测试角色基础知识：
- What is the character's name?
- Where does the character live?
- What are their hobbies?

### 2. CM (Conversation Memory)
测试对话记忆能力：
- Why did the character respond that way?
- How did the conversation progress?
- What was discussed earlier?

## 成本估算

**Claude 3.5 Sonnet 定价**:
- 输入: $3/百万 tokens
- 输出: $15/百万 tokens

**示例** (2000 个样本):
- 每样本 ~2500 tokens
- 总成本: ~$55

## 故障排除

### AWS 凭证错误
```bash
aws configure
aws sts get-caller-identity  # 验证配置
```

### Bedrock 访问被拒
1. 在 AWS Console 启用 Bedrock
2. 启用 Claude 3.5 Sonnet 模型
3. 检查 IAM 权限

### 数据质量问题
- 检查 `online_profiles.jsonl` 格式
- 调整 temperature 参数
- 优化角色描述质量

## 详细文档

完整文档请参考: [DATA_GENERATION_GUIDE.md](DATA_GENERATION_GUIDE.md)

## 下一步

1. **查看数据**: 检查 `data/training/train.json`
2. **评估质量**: 查看 `data/training/dataset_stats.json`
3. **开始训练**: 使用 `scripts/train.py`
4. **评估模型**: 使用 `scripts/evaluate.py`

## 支持

- 📖 [详细文档](DATA_GENERATION_GUIDE.md)
- 📝 [RAIDEN 论文](https://arxiv.org/html/2505.10218v1)
- 🔧 [AWS Bedrock 文档](https://docs.aws.amazon.com/bedrock/)
