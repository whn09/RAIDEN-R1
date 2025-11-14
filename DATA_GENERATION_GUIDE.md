# RAIDEN-R1 数据生成指南

本指南介绍如何使用 AWS Bedrock 的 Claude 3.5 Sonnet 模型生成 RAIDEN-R1 训练数据。

## 概述

根据 RAIDEN 论文，我们实现了两种数据生成策略：

1. **Script-Based Knowledge (SBK)**: 测试模型对角色背景、性格、偏好等基础知识的理解
2. **Conversation Memory (CM)**: 测试模型对对话历史和上下文的记忆能力

## 前置要求

### 1. AWS 配置

确保你的 AWS 凭证已配置：

```bash
# 方法1：使用 AWS CLI 配置
aws configure

# 方法2：设置环境变量
export AWS_ACCESS_KEY_ID="your-access-key"
export AWS_SECRET_ACCESS_KEY="your-secret-key"
export AWS_DEFAULT_REGION="us-east-1"
```

### 2. AWS Bedrock 访问权限

确保你的 AWS 账户：
- 在 `us-east-1` 区域开启了 Bedrock 访问权限
- 已启用 Claude 3.5 Sonnet 模型 (`anthropic.claude-3-5-sonnet-20241022-v2:0`)

### 3. 安装依赖

```bash
pip install -r requirements.txt
```

主要依赖：
- `boto3>=1.34.0`: AWS SDK for Python
- `anthropic>=0.7.0`: Anthropic API (可选，用于对比测试)

## 数据生成流程

### 方法概述

根据论文，数据生成包括以下步骤：

1. **问题生成**: 使用 Claude 3.5 Sonnet 生成 WH-questions (What, Who, Where, When, Why, How)
2. **答案生成**: 使用 Chain-of-Thought (CoT) 推理生成答案
3. **关键词提取**: 从答案中提取验证关键词
4. **验证方法选择**:
   - Single-Term Validation: 单个关键词的简单验证
   - Multi-Term Dynamic Parsing: 多个关键词的动态解析验证

### 快速测试

首先运行测试脚本确保一切正常：

```bash
python scripts/test_bedrock_generator.py
```

这个脚本会：
- 测试与 AWS Bedrock 的连接
- 生成测试问题
- 生成带有 CoT 推理的答案
- 提取关键词
- 生成完整的训练样本

### 生成完整数据集

使用主脚本从 `online_profiles.jsonl` 生成完整数据集：

```bash
python scripts/generate_data_with_bedrock.py \
    --profiles_file ./data/online_profiles.jsonl \
    --output_file ./data/generated_samples.json \
    --output_dir ./data/training \
    --num_samples_per_profile 2 \
    --include_cm
```

#### 参数说明

**输入/输出参数：**
- `--profiles_file`: 角色配置文件路径 (JSONL 格式)
- `--output_file`: 生成样本的保存路径
- `--output_dir`: 训练数据输出目录

**生成参数：**
- `--num_samples_per_profile`: 每个角色生成的 SBK 样本数量 (默认: 2)
- `--include_cm`: 是否生成 CM (对话记忆) 样本

**AWS Bedrock 参数：**
- `--region`: AWS 区域 (默认: us-east-1)
- `--model_id`: Claude 模型 ID (默认: anthropic.claude-3-5-sonnet-20241022-v2:0)

**数据集参数：**
- `--train_ratio`: 训练集比例 (默认: 0.9)

### 生成的文件

运行完成后，会生成以下文件：

```
data/
├── generated_samples.json      # 原始生成的样本
└── training/
    ├── train.json             # 训练集
    ├── validation.json        # 验证集
    └── dataset_stats.json     # 数据集统计信息
```

## 数据格式

### 输入格式 (online_profiles.jsonl)

每行一个 JSON 对象，包含角色配置：

```json
{
  "prompt": "角色设定文本...",
  "character_profile": {
    "Name": "角色名称",
    "Gender": "性别",
    "Introduction": "简介",
    "Detailed Description": "详细描述"
  }
}
```

### 输出格式

生成的训练样本格式：

```json
{
  "character_name": "角色名称",
  "character_profile": {...},
  "conversation_history": [...],
  "question": "生成的问题",
  "answer": "模型答案",
  "keywords": ["关键词1", "关键词2"],
  "question_type": "what",
  "validation_method": "single_term_validation",
  "difficulty": "medium",
  "metadata": {
    "reasoning": "CoT推理过程",
    "focus": "script_knowledge",
    "generated_at": 1234567890.0
  }
}
```

## 数据生成策略

### 1. Script-Based Knowledge (SBK)

测试模型对角色基础知识的理解：

- **问题类型**: What, Who, Where, When 为主
- **难度**: Easy to Medium
- **验证**: 关键词匹配
- **示例问题**:
  - "What is the character's favorite hobby?"
  - "Where did the character grow up?"
  - "What is the character's occupation?"

### 2. Conversation Memory (CM)

测试模型对对话上下文的记忆：

- **问题类型**: Why, How 为主
- **难度**: Medium to Hard
- **验证**: 多关键词动态解析
- **示例问题**:
  - "Why did the character respond angrily in the previous message?"
  - "How did the character's mood change during the conversation?"

## 自定义配置

### 修改问题类型分布

编辑 `src/data/bedrock_generator.py` 中的 `WH_QUESTION_TYPES`:

```python
WH_QUESTION_TYPES = [
    "what",   # 角色属性
    "who",    # 角色关系
    "where",  # 地点信息
    "when",   # 时间信息
    "why",    # 原因推理
    "how"     # 方式推理
]
```

### 调整难度分布

在生成时指定难度：

```python
difficulty = random.choice(["easy", "medium", "hard"])
sample = generator.generate_sample_from_profile(
    character_profile,
    conversation_history=None,
    difficulty=difficulty
)
```

### 自定义提示词

修改 `BedrockDataGenerator` 类中的提示词模板来调整生成行为。

## 成本估算

使用 AWS Bedrock Claude 3.5 Sonnet 的成本：

- **输入**: $3 / 百万 tokens
- **输出**: $15 / 百万 tokens

每个样本大约需要：
- SBK 样本: ~2000-3000 tokens (输入 + 输出)
- CM 样本: ~3000-4000 tokens (包含对话历史)

**示例计算** (生成 2000 个样本):
- 假设平均每个样本 2500 tokens
- 总 tokens: 2000 × 2500 = 5M tokens
- 假设输入/输出比例 1:2
- 成本: (5M × 1/3 × $3 + 5M × 2/3 × $15) / 1M ≈ $55

## 最佳实践

### 1. 分批生成

对于大量数据，建议分批生成以便监控和调试：

```bash
# 第一批: 100个角色
python scripts/generate_data_with_bedrock.py \
    --profiles_file ./data/batch1.jsonl \
    --num_samples_per_profile 2

# 第二批: 下100个角色
python scripts/generate_data_with_bedrock.py \
    --profiles_file ./data/batch2.jsonl \
    --num_samples_per_profile 2
```

### 2. 质量检查

生成后检查数据质量：

```python
import json

with open('./data/training/train.json', 'r') as f:
    samples = json.load(f)

# 检查关键词覆盖
samples_with_keywords = sum(1 for s in samples if s['keywords'])
print(f"Samples with keywords: {samples_with_keywords}/{len(samples)}")

# 检查难度分布
difficulties = {}
for s in samples:
    diff = s['difficulty']
    difficulties[diff] = difficulties.get(diff, 0) + 1
print(f"Difficulty distribution: {difficulties}")
```

### 3. 错误处理

脚本包含重试机制，但如果遇到持续错误：

1. 检查 AWS 凭证和权限
2. 检查 Bedrock 服务配额
3. 验证输入数据格式
4. 查看详细错误日志

## 故障排除

### 问题: AWS 凭证错误

```
Error: Unable to locate credentials
```

**解决方案**:
```bash
aws configure
# 或设置环境变量
export AWS_ACCESS_KEY_ID="..."
export AWS_SECRET_ACCESS_KEY="..."
```

### 问题: Bedrock 访问被拒绝

```
Error: AccessDeniedException
```

**解决方案**:
1. 在 AWS Console 中启用 Bedrock 访问
2. 确保 IAM 角色有 `bedrock:InvokeModel` 权限
3. 检查模型是否在你的区域可用

### 问题: 生成质量不佳

**解决方案**:
1. 调整 temperature 参数 (降低以获得更确定的输出)
2. 改进角色描述的质量
3. 增加提示词的详细程度
4. 检查并优化提示词模板

### 问题: 生成速度慢

**解决方案**:
1. 减少 `num_samples_per_profile`
2. 使用更快的模型 (如果可用)
3. 分批并行处理
4. 检查网络连接

## 下一步

数据生成完成后，可以：

1. **数据质量评估**: 使用验证集评估生成质量
2. **开始训练**: 使用生成的数据进行 GRPO 训练
3. **模型评估**: 在 SBK 和 CM 基准上评估模型性能

参考:
- [训练指南](QUICKSTART.md)
- [评估脚本](scripts/evaluate.py)
- [RAIDEN 论文](https://arxiv.org/html/2505.10218v1)

## 参考资料

- **RAIDEN 论文**: https://arxiv.org/html/2505.10218v1
- **AWS Bedrock 文档**: https://docs.aws.amazon.com/bedrock/
- **Claude API 参考**: https://docs.anthropic.com/claude/reference

## 联系与支持

如有问题，请：
1. 检查本文档的故障排除部分
2. 查看项目 Issues
3. 参考论文原文了解方法细节
