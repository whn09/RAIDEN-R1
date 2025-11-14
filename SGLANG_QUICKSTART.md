# RAIDEN-R1 快速开始 - SGLang 本地部署

使用 SGLang 部署开源模型进行快速数据生成，比云端 API 快 10-100 倍！

## 🚀 为什么用 SGLang？

- ⚡ **超快速度**: 本地推理，无网络延迟
- 💰 **零成本**: 无 API 调用费用
- 🔒 **数据安全**: 数据不离开本地
- 📊 **可扩展**: 充分利用 8 块 H100 GPU

## 快速开始 (2 步骤)

### 1. 部署 SGLang 服务器

```bash
# 方法 1: 使用推荐的 MiniMax M2 (角色扮演效果好)
./scripts/deploy_sglang.sh \
    --model-path /path/to/MiniMax-Text-01 \
    --model-name minimax-m2 \
    --tp-size 8

# 方法 2: 使用 GLM-4 (9B, 更快)
./scripts/deploy_sglang.sh \
    --model-path /path/to/glm-4-9b-chat \
    --model-name glm-4 \
    --tp-size 4

# 方法 3: 使用 Qwen2.5-14B (推理能力强)
./scripts/deploy_sglang.sh \
    --model-path /path/to/Qwen2.5-14B-Instruct \
    --model-name qwen2.5-14b \
    --tp-size 4
```

服务器启动后会自动测试并保存配置到 `sglang_server_info.json`

### 2. 生成数据

```bash
# 使用已部署的服务器生成数据
python scripts/generate_data_with_sglang.py \
    --profiles_file ./data/online_profiles.jsonl \
    --output_dir ./data/training \
    --num_samples_per_profile 2 \
    --include_cm
```

## 📋 推荐模型配置

### p5en.48xlarge (8x H100 80GB)

| 模型 | TP Size | 速度 | 质量 | 用途 |
|------|---------|------|------|------|
| **MiniMax M2** | 8 | ⚡⚡⚡ | ⭐⭐⭐⭐⭐ | 角色扮演推荐 |
| **GLM-4 (9B)** | 4 | ⚡⚡⚡⚡⚡ | ⭐⭐⭐⭐ | 快速生成 |
| **Qwen2.5-14B** | 4 | ⚡⚡⚡⚡ | ⭐⭐⭐⭐⭐ | 推理能力强 |
| **Qwen2.5-32B** | 8 | ⚡⚡⚡ | ⭐⭐⭐⭐⭐ | 最高质量 |
| **DeepSeek-V2.5** | 8 | ⚡⚡⚡ | ⭐⭐⭐⭐⭐ | 推理专家 |

### 选择建议

- **快速测试**: GLM-4 (9B) with tp-size=2
- **生产环境**: MiniMax M2 or Qwen2.5-14B with tp-size=8
- **最高质量**: Qwen2.5-32B or DeepSeek-V2.5 with tp-size=8

## 🛠️ 安装依赖

```bash
# 安装 SGLang
pip install "sglang[all]"

# 或从源码安装最新版
pip install git+https://github.com/sgl-project/sglang.git
```

## 📊 性能对比

### 生成 1000 个样本

| 方法 | 时间 | 成本 | 速度 |
|------|------|------|------|
| AWS Bedrock | ~2-3 小时 | ~$30 | 基准 |
| SGLang (MiniMax M2) | ~10-15 分钟 | $0 | **10-12x 快** |
| SGLang (GLM-4) | ~5-8 分钟 | $0 | **20-30x 快** |

## 🎯 使用示例

### 快速测试 (10 个角色)

```bash
# 1. 部署快速模型
./scripts/deploy_sglang.sh \
    --model-path /models/glm-4-9b-chat \
    --model-name glm-4 \
    --tp-size 2

# 2. 生成测试数据
python scripts/generate_data_with_sglang.py \
    --profiles_file ./data/online_profiles.jsonl \
    --max_profiles 10 \
    --num_samples_per_profile 1 \
    --output_file ./data/test_samples.json
```

### 完整生成 (所有角色)

```bash
# 1. 部署生产模型
./scripts/deploy_sglang.sh \
    --model-path /models/MiniMax-Text-01 \
    --model-name minimax-m2 \
    --tp-size 8

# 2. 生成完整数据集
python scripts/generate_data_with_sglang.py \
    --profiles_file ./data/online_profiles.jsonl \
    --output_dir ./data/training \
    --num_samples_per_profile 2 \
    --include_cm
```

### 批量生成优化

```bash
# 使用更多 GPU 加速
./scripts/deploy_sglang.sh \
    --model-path /models/Qwen2.5-14B-Instruct \
    --tp-size 8 \
    --memory-fraction 0.95

# 增加批次处理
python scripts/generate_data_with_sglang.py \
    --num_samples_per_profile 3 \
    --include_cm \
    --timeout 180
```

## 🌍 多语言支持

RAIDEN-R1 支持多语言数据生成，默认使用**中文**。

### 支持的语言

- **中文 (zh)**: 默认语言
- **日文 (ja)**: Japanese
- **英文 (en)**: English
- **韩文 (ko)**: Korean

### 语言配置

```bash
# 使用中文生成（默认）
python scripts/generate_data_with_sglang.py \
    --language zh

# 使用日文生成
python scripts/generate_data_with_sglang.py \
    --language ja

# 使用英文生成
python scripts/generate_data_with_sglang.py \
    --language en
```

### 自动语言检测

默认情况下，系统会**自动检测**角色配置文件的语言：

```bash
# 自动检测语言（推荐）
python scripts/generate_data_with_sglang.py \
    --profiles_file ./data/online_profiles.jsonl

# 禁用自动检测，强制使用指定语言
python scripts/generate_data_with_sglang.py \
    --language zh \
    --no_auto_detect
```

### 语言检测机制

- **Hiragana/Katakana** (ひらがな・カタカナ) → 日文
- **Hangul** (한글) → 韩文
- **CJK 字符** + **无 Kana** → 中文
- **ASCII 为主** → 英文

### 多语言示例

```bash
# 例1：中文角色配置（自动检测为中文）
# 角色名：林黛玉
# 结果：生成中文问题和答案

# 例2：日文角色配置（自动检测为日文）
# 角色名：桜木花道
# 结果：生成日文问题和答案

# 例3：混合配置（强制使用中文）
python scripts/generate_data_with_sglang.py \
    --language zh \
    --no_auto_detect
```

## 📁 生成的文件

```
data/
├── generated_samples_sglang.json  # 原始样本
└── training/
    ├── train.json                 # 训练集 (90%)
    ├── validation.json            # 验证集 (10%)
    └── dataset_stats.json         # 统计信息

logs/
└── sglang_minimax-m2_20250114_120000.log  # 服务器日志

sglang_server_info.json            # 服务器配置信息
```

## 🔧 高级配置

### 调整推理参数

编辑脚本中的参数或创建配置文件：

```python
# 在 sglang_generator.py 中调整
generator = SGLangGenerator(
    base_url="http://localhost:30000",
    model_name="minimax-m2",
    timeout=180,  # 增加超时时间
    max_retries=5  # 增加重试次数
)
```

### 多服务器并行

在不同端口部署多个模型并行生成：

```bash
# 服务器 1: GLM-4 on port 30000
./scripts/deploy_sglang.sh \
    --model-path /models/glm-4-9b-chat \
    --port 30000 \
    --tp-size 4

# 服务器 2: Qwen2.5 on port 30001
./scripts/deploy_sglang.sh \
    --model-path /models/Qwen2.5-14B-Instruct \
    --port 30001 \
    --tp-size 4

# 分别生成不同批次
python scripts/generate_data_with_sglang.py \
    --base-url http://localhost:30000 \
    --max_profiles 25 &

python scripts/generate_data_with_sglang.py \
    --base-url http://localhost:30001 \
    --max_profiles 25 &
```

### 性能优化

```bash
# 启用 torch compile 加速
# 已在 deploy_sglang.sh 中默认启用

# 调整 GPU 内存占用
./scripts/deploy_sglang.sh \
    --memory-fraction 0.95  # 使用更多 GPU 内存

# 减少 TP size 以支持更大批次
./scripts/deploy_sglang.sh \
    --tp-size 4  # 使用 4 GPU 而不是 8
```

## 🐛 故障排除

### 问题: 服务器启动失败

```bash
# 检查 GPU 状态
nvidia-smi

# 检查端口占用
lsof -i :30000

# 查看详细日志
tail -f logs/sglang_*.log
```

### 问题: Out of Memory

```bash
# 减少 tensor parallel size
./scripts/deploy_sglang.sh --tp-size 4

# 减少内存占用
./scripts/deploy_sglang.sh --memory-fraction 0.8

# 使用更小的模型
./scripts/deploy_sglang.sh --model-path /models/glm-4-9b-chat
```

### 问题: 生成速度慢

```bash
# 1. 检查是否启用了 tensor parallelism
cat sglang_server_info.json | grep tp_size

# 2. 使用更快的模型
./scripts/deploy_sglang.sh --model-path /models/glm-4-9b-chat

# 3. 减少 max_tokens
python scripts/generate_data_with_sglang.py --timeout 60
```

### 问题: 生成质量不佳

```bash
# 1. 使用更大的模型
./scripts/deploy_sglang.sh --model-path /models/Qwen2.5-32B-Instruct

# 2. 调整采样参数（在代码中）
# temperature: 0.7 -> 0.9 (更多样化)
# top_p: 0.9 -> 0.95 (更高质量)
```

## 📈 监控和管理

### 查看服务器状态

```bash
# 检查健康状态
curl http://localhost:30000/health

# 查看服务器信息
cat sglang_server_info.json | jq

# 实时查看日志
tail -f logs/sglang_*.log
```

### 停止服务器

```bash
# 方法 1: 使用保存的 PID
kill $(cat sglang_server_info.json | jq -r .pid)

# 方法 2: 直接 kill 端口
lsof -ti :30000 | xargs kill -9

# 方法 3: kill 所有 SGLang 进程
pkill -f "sglang.launch_server"
```

### 资源监控

```bash
# GPU 使用情况
watch -n 1 nvidia-smi

# 显存使用
nvidia-smi --query-gpu=memory.used,memory.total --format=csv

# 进程监控
htop
```

## 🔄 与 Bedrock 对比

### 何时使用 SGLang

✅ **适合 SGLang**:
- 需要快速生成大量数据
- 有充足的 GPU 资源
- 关注成本控制
- 数据安全要求高
- 需要自定义模型

✅ **适合 Bedrock**:
- 小规模数据生成
- 无本地 GPU 资源
- 需要 Claude 特定能力
- 快速原型验证

### 混合使用

```bash
# 1. 用 SGLang 生成大量基础数据
python scripts/generate_data_with_sglang.py \
    --num_samples_per_profile 2

# 2. 用 Bedrock 生成高质量验证集
python scripts/generate_data_with_bedrock.py \
    --num_samples_per_profile 1 \
    --max_profiles 50
```

## 📖 配置参考

完整的模型配置请参考: [configs/sglang_models.yaml](configs/sglang_models.yaml)

## 🆘 获取帮助

- 📘 SGLang 文档: https://github.com/sgl-project/sglang
- 📕 详细数据生成文档: [DATA_GENERATION_GUIDE.md](DATA_GENERATION_GUIDE.md)
- 📝 RAIDEN 论文: https://arxiv.org/html/2505.10218v1

## 下一步

1. ✅ 部署 SGLang 服务器
2. ✅ 生成测试数据验证
3. ✅ 生成完整训练数据集
4. 🚀 开始 GRPO 训练

Happy generating! 🎉
