# RAIDEN-R1: 通过GRPO与可验证奖励提升大语言模型的角色感知能力

实现RAIDEN-R1框架，通过带有可验证角色感知奖励（VRAR）的组相对策略优化（GRPO）来提升大语言模型的角色感知能力。

[English](README.md) | 简体中文

## 论文

**标题**: RAIDEN-R1: Improving Role-awareness of LLMs via GRPO with Verifiable Reward

**作者**: Zongsheng Wang, Kaili Sun, Bowen Wu, Qun Yu, Ying Li, Baoxun Wang

**arXiv**: https://arxiv.org/html/2505.10218v1

## 概述

RAIDEN-R1 通过以下方法解决角色扮演对话智能体的角色一致性问题：

1. **可验证角色感知奖励（VRAR）**：用于角色感知训练的可量化奖励机制
2. **GRPO训练**：使用组相对策略优化提升角色一致性
3. **高质量数据集**：包含脚本知识（SBK）和对话记忆（CM）的角色感知数据

## 项目结构

```
raiden-r1/
├── src/
│   ├── data/                    # 数据处理和生成
│   │   ├── sglang_generator.py  # SGLang本地生成器（快10-100倍）
│   │   ├── bedrock_generator.py # AWS Bedrock数据生成器
│   │   ├── language_utils.py    # 多语言支持（中/日/英/韩）
│   │   └── collection.py        # 数据集管理
│   ├── training/
│   │   ├── grpo_trainer.py      # 自定义GRPO实现
│   │   └── openr1_adapter.py    # OpenR1适配器（论文准确）
│   ├── evaluation/              # 评估指标（SBK、CM）
│   └── rewards/
│       └── vrar.py              # 可验证角色感知奖励
├── configs/
│   ├── grpo_config.yaml         # 自定义训练配置
│   └── openr1_config.yaml       # OpenR1训练配置
├── data/
│   ├── online_profiles.jsonl    # 角色档案
│   └── training/                # 生成的训练数据
├── scripts/                     # 训练和生成脚本
└── accelerate_config.yaml       # 多GPU训练配置
```

## 快速开始

### 1. 安装

```bash
pip install -r requirements.txt
```

**环境要求**:
- Python 3.8+
- PyTorch 2.0+
- Transformers
- 训练需要 8x NVIDIA H800/H200 GPU

**GPU显存要求**:
- 自定义GRPO: Qwen2.5-14B (bf16) 每卡约需100GB显存
- OpenR1 GRPO: Qwen2.5-14B 每卡约需140-160GB显存（因需生成多个响应）
  - 建议：对于140GB显存的GPU，使用Qwen2.5-7B模型
  - 或减少生成数量（`num_samples_per_prompt: 2` 而非 4）

### 2. 数据生成

**推荐：SGLang + GLM-4.6**（比云API快10-100倍）

```bash
# 下载GLM-4.6模型
huggingface-cli download zai-org/GLM-4.6 --local-dir /path/to/GLM-4.6

# 启动SGLang服务器
python -m sglang.launch_server \
    --model-path /path/to/GLM-4.6 \
    --tp-size 8 \
    --port 30000 \
    --chat-template glm4 \
    --trust-remote-code

# 生成训练数据（默认中文）
python scripts/generate_data_with_sglang.py \
    --profiles_file ./data/online_profiles.jsonl \
    --num_samples_per_profile 2 \
    --include_cm \
    --language zh

# 其他语言
python scripts/generate_data_with_sglang.py --language ja  # 日语
python scripts/generate_data_with_sglang.py --language en  # 英语
python scripts/generate_data_with_sglang.py --language ko  # 韩语
```

**备选方案：AWS Bedrock（用于快速原型验证）**

```bash
python scripts/generate_data_with_bedrock.py \
    --num_samples_per_profile 2 \
    --language zh
```

### 3. 训练

RAIDEN-R1 提供两种训练实现：

> 📖 **详细的OpenR1指南**：查看 [OPENR1_GUIDE.md](OPENR1_GUIDE.md) 获取完整文档

#### 方案 A：自定义GRPO实现（默认）

简单易定制：

```bash
# 多GPU训练（8x H200/H800）
accelerate launch --config_file accelerate_config.yaml \
    scripts/train.py --config configs/grpo_config.yaml

# 监控训练
watch -n 1 nvidia-smi
tail -f outputs/training.log
```

#### 方案 B：OpenR1 GRPO（论文准确 ⭐）

使用论文中提到的Hugging Face OpenR1库：

```bash
# 安装OpenR1（必须从源码安装）
git clone https://github.com/huggingface/open-r1.git
cd open-r1
pip install -e ".[dev]"
cd ..

# 验证安装
python scripts/test_openr1_integration.py

# 使用OpenR1训练（单GPU）
python scripts/train_with_openr1.py configs/openr1_config.yaml

# 或使用命令行参数
python scripts/train_with_openr1.py \
    --train_data_path ./data/training/train.json \
    --eval_data_path ./data/training/validation.json \
    --model_name_or_path Qwen/Qwen2.5-14B-Instruct \
    --output_dir ./outputs_openr1 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --learning_rate 3e-6

# 多GPU训练（推荐，8x H200/H800）
accelerate launch --config_file accelerate_config.yaml \
    scripts/train_with_openr1.py configs/openr1_config.yaml

# 监控训练
watch -n 1 nvidia-smi
tail -f outputs_openr1/training.log
```

**对比**:
- **自定义GRPO**：更简单，易于调试，适合实验
- **OpenR1**：与论文实现一致，性能优化，社区维护

**训练配置** (`configs/grpo_config.yaml` 或 `configs/openr1_config.yaml`):
- 基础模型：`Qwen/Qwen2.5-14B-Instruct`
- 学习率：`3e-6` (余弦调度器)
- 批次大小：每GPU `4`（8卡有效批次：64）
- 精度：`bf16`
- 训练轮次：`1`
- GRPO采样数：每提示 `4` 个

### 4. 评估

```bash
python scripts/evaluate.py \
    --model_path ./outputs/epoch_0 \
    --eval_data ./data/training/validation.json \
    --output_file ./evaluation_results.json
```

## 数据生成详解

### 为什么选择GLM-4.6？

- ✅ **优秀的角色扮演质量**
- ✅ **SGLang推理速度快**
- ✅ **支持thinking模式控制**（输出干净）
- ✅ **原生中文支持**
- ✅ **开源免费**

### 支持的模型（SGLang）

- **GLM-4.6**（推荐）- 速度与质量的最佳平衡
- **MiniMax M2** - 角色扮演能力强
- **Qwen2.5-14B/32B** - 推理能力强
- **DeepSeek-V2.5** - 高质量输出

### 多语言支持

RAIDEN-R1 支持多语言自动检测：

- **中文（zh）**：默认语言，原生支持 ✓
- **日语（ja）**：日本語対応
- **英语（en）**：完整支持
- **韩语（ko）**：한국어 지원

```bash
# 使用指定语言
python scripts/generate_data_with_sglang.py --language zh

# 从角色档案自动检测
python scripts/generate_data_with_sglang.py --auto_detect
```

### 生成参数

```bash
python scripts/generate_data_with_sglang.py \
    --profiles_file ./data/online_profiles.jsonl \
    --output_file ./data/generated_samples.json \
    --num_samples_per_profile 2 \
    --include_cm \
    --language zh \
    --max_profiles 100  # 测试时限制数量
```

### 输入数据格式

角色档案应为JSONL格式（`data/online_profiles.jsonl`）。每行包含一个结构化的角色信息：

```jsonl
{"prompt": "...[角色扮演系统提示词]...\n\n<Character Setting>\nName: 小明\nGender: male\nIntroduction: 一个阳光开朗的高中生，喜欢打篮球和画画。\nDetailed Description: 小明今年17岁，就读于某高中二年级。性格外向活泼，总是能给周围的人带来欢笑。课余时间喜欢和朋友一起打篮球，也热爱绘画创作。梦想是成为一名职业插画师。\n</Character Setting>\n\n...[其他设定]..."}
```

**关键字段**:
- `Name`：角色名称
- `Gender`：性别（male/female）
- `Introduction`：简介（1-2句话）
- `Detailed Description`：详细的角色背景、性格、目标等

系统会自动解析 `<Character Setting>` 块并基于这些信息生成训练数据。

## 训练数据格式

```json
{
  "character_name": "角色名称",
  "character_profile": {
    "Name": "角色名称",
    "Gender": "female",
    "Introduction": "简介",
    "Detailed Description": "详细描述"
  },
  "conversation_history": [],
  "question": "你叫什么名字？",
  "answer": "我叫小明。",
  "keywords": ["小明"],
  "question_type": "what",
  "validation_method": "single_term_validation",
  "difficulty": "easy",
  "metadata": {
    "source": "sglang_sbk",
    "model": "GLM-4.6",
    "language": "zh"
  }
}
```

## 实验结果

RAIDEN-R1 14B-GRPO 模型达到：
- **SBK（脚本知识）**：88.04%
- **CM（对话记忆）**：88.65%

## 常见问题

### GPU利用率低

```bash
# 确保使用accelerate进行多GPU训练
accelerate launch --config_file accelerate_config.yaml scripts/train.py --config configs/grpo_config.yaml

# 在configs/grpo_config.yaml中增加批次大小
batch_size: 8
gradient_accumulation_steps: 2
```

### 显存不足（OOM）

```bash
# 在configs/grpo_config.yaml中减小批次大小
batch_size: 4

# 使用更少的GPU
accelerate launch --num_processes=4 scripts/train.py --config configs/grpo_config.yaml
```

### SGLang Thinking模式问题

如果模型输出thinking标签（`<think>...</think>`），请确保：
1. SGLang服务器启动时使用 `--chat-template glm4`
2. 使用最新代码（已集成OpenAI SDK）
3. 不使用 `--enable_thinking` 标志

## 高级配置

### SGLang服务器优化

```bash
# 生产环境配置（8 GPU）
python -m sglang.launch_server \
    --model-path /path/to/GLM-4.6 \
    --tp-size 8 \
    --ep-size 8 \
    --port 30000 \
    --chat-template glm4 \
    --mem-fraction-static 0.85 \
    --trust-remote-code
```

### 使用自定义数据训练

```bash
# 生成更多样本
python scripts/generate_data_with_sglang.py \
    --num_samples_per_profile 5 \
    --include_cm

# 使用自定义训练/验证集分割比例
python scripts/generate_data_with_sglang.py \
    --train_ratio 0.8  # 80%训练，20%验证
```

## 引用

```bibtex
@article{wang2025raiden,
  title={RAIDEN-R1: Improving Role-awareness of LLMs via GRPO with Verifiable Reward},
  author={Wang, Zongsheng and Sun, Kaili and Wu, Bowen and Yu, Qun and Li, Ying and Wang, Baoxun},
  journal={arXiv preprint arXiv:2505.10218},
  year={2025}
}
```

## 致谢

- Open-R1 库
- RAIDEN 基准测试
- Qwen2.5-14B-Instruct 基础模型
- SGLang 框架

## 相关文档

- [English README](README.md) - 英文版README
- [OpenR1集成指南](OPENR1_GUIDE.md) - OpenR1使用详解（英文）
- [OpenR1集成摘要](OPENR1_INTEGRATION_SUMMARY.md) - 集成概述（英文）

## 许可证

[待添加]
