# RAIDEN-R1 训练指南

详细的 GRPO 训练步骤和配置说明。

## 📋 前置要求

### 硬件要求
- **GPU**: 8x NVIDIA H200/H800 (建议)
- **显存**: 每个 GPU 至少 40GB
- **内存**: 512GB+ RAM
- **存储**: 200GB+ 可用空间

### 软件要求
- Python 3.8+
- PyTorch 2.0+
- CUDA 12.0+
- Transformers
- Accelerate

## 🚀 快速开始

### 1. 准备训练数据

#### 方法 A: 使用 SGLang 生成（推荐）

```bash
# 部署 SGLang 服务器
./scripts/deploy_sglang.sh \
    --model-path /path/to/model \
    --model-name minimax-m2 \
    --tp-size 8

# 生成训练数据
python scripts/generate_data_with_sglang.py \
    --profiles_file ./data/online_profiles.jsonl \
    --output_dir ./data/training \
    --num_samples_per_profile 2 \
    --include_cm \
    --language zh
```

#### 方法 B: 使用 AWS Bedrock

```bash
# 配置 AWS 凭证
export AWS_ACCESS_KEY_ID="your-key"
export AWS_SECRET_ACCESS_KEY="your-secret"
export AWS_DEFAULT_REGION="us-east-1"

# 生成数据
python scripts/generate_data_with_bedrock.py \
    --profiles_file ./data/online_profiles.jsonl \
    --output_dir ./data/training \
    --num_samples_per_profile 2 \
    --include_cm \
    --language zh
```

### 2. 配置训练参数

编辑 `configs/grpo_config.yaml`:

```yaml
# 模型配置
model_name: "Qwen/Qwen2.5-14B-Instruct"

# 训练超参数
learning_rate: 3.0e-6
batch_size: 8              # 每个 GPU 的批次大小
num_epochs: 1
gradient_accumulation_steps: 2  # 有效批次 = 8 * 8 * 2 = 128

# GRPO 特定参数
num_samples_per_prompt: 4  # 每个提示的响应数
kl_penalty: 0.1           # KL 散度惩罚系数

# 奖励权重
accuracy_weight: 0.7      # 准确性奖励权重
format_weight: 0.3        # 格式奖励权重

# 硬件配置
use_bf16: true
gradient_checkpointing: true

# 数据路径
train_data: "./data/training/train.json"
eval_data: "./data/training/validation.json"
```

### 3. 启动训练

#### 多 GPU 训练（推荐）

```bash
# 使用所有 8 个 GPU
accelerate launch --config_file accelerate_config.yaml \
    scripts/train.py --config configs/grpo_config.yaml
```

#### 单 GPU 训练（测试用）

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train.py \
    --config configs/grpo_config.yaml
```

#### 自定义 GPU 数量

```bash
# 使用 4 个 GPU
accelerate launch --num_processes=4 \
    scripts/train.py --config configs/grpo_config.yaml

# 指定特定 GPU
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
    --num_processes=4 \
    scripts/train.py --config configs/grpo_config.yaml
```

## 📊 监控训练

### GPU 使用情况

```bash
# 实时监控
watch -n 1 nvidia-smi

# 详细监控
nvidia-smi dmon -s u

# 查看特定 GPU
nvidia-smi -i 0
```

### 训练日志

```bash
# 查看实时日志
tail -f outputs/training.log

# 查看 TensorBoard
tensorboard --logdir outputs/tensorboard
```

### 检查点

训练过程会自动保存检查点：

```
outputs/
├── checkpoint-100/
├── checkpoint-200/
└── final_model/
```

## ⚙️ 高级配置

### 优化 GPU 利用率

如果 GPU 利用率低于 60%：

```yaml
# configs/grpo_config.yaml
batch_size: 16  # 增加批次大小
gradient_accumulation_steps: 4
num_samples_per_prompt: 8  # 增加 GRPO 采样数
```

### 减少显存使用

如果遇到 OOM 错误：

```yaml
# configs/grpo_config.yaml
batch_size: 4  # 减少批次大小
gradient_accumulation_steps: 8
gradient_checkpointing: true
max_length: 1024  # 减少最大长度
```

### 加速训练

```yaml
# configs/grpo_config.yaml
use_bf16: true  # 使用混合精度
gradient_checkpointing: true
num_workers: 8  # 数据加载线程数
```

## 🔧 故障排除

### 问题 1: GPU 利用率为 0%

**症状**: `nvidia-smi` 显示所有 GPU 利用率和显存都是 0

**原因**: 没有使用多 GPU 训练

**解决方案**:
```bash
# 使用 accelerate
accelerate launch --config_file accelerate_config.yaml \
    scripts/train.py --config configs/grpo_config.yaml
```

### 问题 2: device_map 冲突错误

**症状**:
```
ValueError: You can't train a model that has been loaded with `device_map='auto'` in any distributed mode.
```

**原因**: 模型使用 `device_map="auto"` 加载，这与 accelerate 的分布式训练冲突

**解决方案**: 已修复！确保使用最新代码。模型加载代码已移除 `device_map="auto"`，由 accelerate 管理设备分配。

### 问题 2.5: DistributedDataParallel 缺少 generate 方法

**症状**:
```
AttributeError: 'DistributedDataParallel' object has no attribute 'generate'
```

**原因**: accelerate 将模型包装为 DistributedDataParallel，需要通过 `unwrap_model` 访问原始模型方法

**解决方案**: 已修复！代码现在使用 `accelerator.unwrap_model(self.model)` 来访问 `generate()` 方法。

### 问题 3: 导入错误

**症状**: `ImportError: cannot import name 'RolePlayingSample'`

**解决方案**:
```bash
# 确保使用最新代码
git pull
python -c "import sys; sys.path.append('src'); from data.collection import RolePlayingSample; print('✓ Import successful')"
```

### 问题 3: 奖励计算错误

**症状**: `AttributeError: 'str' object has no attribute 'get'`

**原因**: 训练数据格式不正确

**解决方案**:
```bash
# 重新生成训练数据
python scripts/generate_data_with_sglang.py \
    --profiles_file ./data/online_profiles.jsonl \
    --output_dir ./data/training
```

### 问题 4: 显存溢出 (OOM)

**症状**: `CUDA out of memory`

**解决方案**:
```bash
# 1. 减少批次大小
# 在 configs/grpo_config.yaml 中修改:
batch_size: 4

# 2. 使用更少的 GPU
accelerate launch --num_processes=4 \
    scripts/train.py --config configs/grpo_config.yaml

# 3. 减少序列长度
max_length: 1024
```

### 问题 5: 训练速度慢

**症状**: 每步训练时间超过 10 秒

**解决方案**:
```bash
# 1. 检查数据加载
# 增加数据加载线程数
num_workers: 8

# 2. 使用更大的批次
batch_size: 16
gradient_accumulation_steps: 1

# 3. 确保使用 bf16
use_bf16: true
```

## 📈 预期性能

### 8x H200 GPU 配置

| 批次大小 | 有效批次 | GPU 利用率 | 步/秒 | 预计时间 (1 epoch) |
|---------|---------|-----------|-------|-------------------|
| 4       | 64      | 40-60%    | 0.5   | 2-3 小时          |
| 8       | 128     | 70-90%    | 0.8   | 1-2 小时          |
| 16      | 256     | 85-95%    | 1.0   | <1 小时           |

### 显存使用

| 模型大小 | 批次大小 | 显存/GPU | 总显存需求 |
|---------|---------|---------|-----------|
| 14B     | 4       | 30-35GB | 240-280GB |
| 14B     | 8       | 40-50GB | 320-400GB |
| 14B     | 16      | 60-70GB | 480-560GB |

## 📝 训练检查清单

- [ ] 训练数据已生成（`data/training/train.json`）
- [ ] 验证数据已生成（`data/training/validation.json`）
- [ ] 配置文件已更新（`configs/grpo_config.yaml`）
- [ ] Accelerate 配置正确（`accelerate_config.yaml`）
- [ ] GPU 可用性已检查（`nvidia-smi`）
- [ ] Python 环境已激活
- [ ] 所有依赖已安装（`pip install -r requirements.txt`）

## 🎯 最佳实践

### 1. 数据质量

- 确保训练数据包含高质量的角色配置
- 平衡 SBK 和 CM 样本（1:1 比例）
- 使用多种难度级别（easy:medium:hard = 3:5:2）

### 2. 超参数调优

- 从论文推荐的参数开始
- 监控验证集性能
- 调整学习率和批次大小

### 3. 监控指标

- 训练损失下降趋势
- 验证集准确率
- GPU 利用率（目标 >70%）
- 显存使用情况

### 4. 保存策略

- 每 100 步保存一次检查点
- 保留最佳验证性能的模型
- 定期备份到云存储

## 📚 更多资源

- [RAIDEN 论文](https://arxiv.org/html/2505.10218v1)
- [SGLang 文档](SGLANG_QUICKSTART.md)
- [Bedrock 文档](BEDROCK_QUICKSTART.md)
- [数据生成对比](GENERATION_COMPARISON.md)

## 💡 提示

1. **首次训练**: 建议先用少量数据（~100 样本）测试整个流程
2. **GPU 利用率**: 使用 `accelerate launch` 可以显著提高 GPU 利用率
3. **数据生成**: SGLang 比 Bedrock 快 10-100 倍，推荐用于大规模生成
4. **多语言**: 系统默认使用中文，支持自动语言检测
5. **检查点**: 训练可能需要 1-2 小时，建议使用 `tmux` 或 `screen`

---

如有问题，请查看 [GitHub Issues](https://github.com/anthropics/claude-code/issues) 或参考主 README。
