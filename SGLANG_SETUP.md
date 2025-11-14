# SGLang 快速部署指南

## 📋 前置条件

✅ p5en.48xlarge 实例 (8x H100 80GB)
✅ CUDA 和 nvidia-smi 可用
✅ Python 3.8+

## 🚀 一键部署

### 步骤 1: 下载模型（如果还没有）

推荐模型下载来源：
- HuggingFace
- ModelScope
- 本地模型仓库

```bash
# 示例：下载 GLM-4 (9B)
# 选择合适的下载方式
```

### 步骤 2: 部署 SGLang

```bash
# 方法 1: 使用脚本（推荐）
./scripts/deploy_sglang.sh \
    --model-path /path/to/your/model \
    --model-name your-model-name \
    --tp-size 8

# 方法 2: 手动启动
python -m sglang.launch_server \
    --model-path /path/to/your/model \
    --host 0.0.0.0 \
    --port 30000 \
    --tp-size 8 \
    --mem-fraction-static 0.9
```

**推荐配置**:

| 模型 | TP Size | 端口 | 命令 |
|------|---------|------|------|
| MiniMax M2 | 8 | 30000 | `--tp-size 8` |
| GLM-4 (9B) | 4 | 30000 | `--tp-size 4` |
| Qwen2.5-14B | 4 | 30000 | `--tp-size 4` |
| Qwen2.5-32B | 8 | 30000 | `--tp-size 8` |

### 步骤 3: 验证部署

```bash
# 检查健康状态
curl http://localhost:30000/health

# 测试推理
curl -X POST http://localhost:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "your-model",
    "messages": [{"role": "user", "content": "你好"}],
    "max_tokens": 50
  }'
```

### 步骤 4: 生成数据

```bash
# 快速测试
python scripts/test_sglang_generator.py

# 生成数据
python scripts/generate_data_with_sglang.py \
    --profiles_file ./data/online_profiles.jsonl \
    --num_samples_per_profile 2 \
    --include_cm
```

## 📊 性能预期

### p5en.48xlarge (8x H100)

| 模型 | tokens/秒 | 样本/小时 | 1000样本时间 |
|------|-----------|-----------|--------------|
| MiniMax M2 (TP=8) | ~2000 | ~800 | 10-15分钟 |
| GLM-4 (TP=4) | ~4000 | ~1600 | 5-8分钟 |
| Qwen2.5-14B (TP=4) | ~2500 | ~1000 | 8-12分钟 |

## 🔧 常见问题

### 1. 模型加载失败
```bash
# 检查模型路径
ls -la /path/to/model

# 检查 GPU 内存
nvidia-smi

# 减少 TP size
./scripts/deploy_sglang.sh --tp-size 4
```

### 2. OOM (显存不足)
```bash
# 减少内存占用
./scripts/deploy_sglang.sh --memory-fraction 0.8

# 使用更小的模型
./scripts/deploy_sglang.sh --model-path /models/glm-4-9b-chat
```

### 3. 生成速度慢
```bash
# 增加 TP size
./scripts/deploy_sglang.sh --tp-size 8

# 使用更快的模型
./scripts/deploy_sglang.sh --model-path /models/glm-4-9b-chat
```

## 📝 快速命令参考

```bash
# 部署
./scripts/deploy_sglang.sh --model-path MODEL_PATH --tp-size 8

# 测试
python scripts/test_sglang_generator.py

# 生成
python scripts/generate_data_with_sglang.py

# 停止
kill $(cat sglang_server_info.json | jq -r .pid)

# 查看日志
tail -f logs/sglang_*.log

# 监控 GPU
watch -n 1 nvidia-smi
```

## 🎯 推荐工作流

1. **快速验证** (10个角色):
   ```bash
   ./scripts/deploy_sglang.sh --model-path MODEL_PATH --tp-size 4
   python scripts/generate_data_with_sglang.py --max_profiles 10
   ```

2. **完整生成** (所有角色):
   ```bash
   ./scripts/deploy_sglang.sh --model-path MODEL_PATH --tp-size 8
   python scripts/generate_data_with_sglang.py --include_cm
   ```

3. **批量并行** (多个服务器):
   ```bash
   # GPU 0-3
   CUDA_VISIBLE_DEVICES=0,1,2,3 ./scripts/deploy_sglang.sh \
       --model-path MODEL1 --port 30000

   # GPU 4-7
   CUDA_VISIBLE_DEVICES=4,5,6,7 ./scripts/deploy_sglang.sh \
       --model-path MODEL2 --port 30001
   ```

## 📖 更多文档

- [详细快速开始指南](SGLANG_QUICKSTART.md)
- [与 Bedrock 对比](GENERATION_COMPARISON.md)
- [完整数据生成文档](DATA_GENERATION_GUIDE.md)

---

**关键优势**: 比 Bedrock 快 10-100 倍，零 API 成本！ 🚀
