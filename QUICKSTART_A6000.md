# A6000 GPU 快速开始指南

## 🎯 一键启动优化训练

### 1️⃣ 验证优化配置
```bash
# 快速检查所有优化是否正确应用
bash scripts/check_optimization.sh
```

### 2️⃣ 开始训练
```bash
# Medium模型预训练 (自动应用所有A6000优化)
python3 scripts/train.py --mode pretrain --config medium

# 如果使用uv包管理器
uv run python scripts/train.py --mode pretrain --config medium
```

### 3️⃣ 监控训练
```bash
# 终端1: 监控GPU使用率
watch -n 1 nvidia-smi

# 终端2: 查看训练日志
tail -f logs/training.log
```

## ✅ 优化检查清单

训练开始时应该看到：
```
=== MEDIUM 模型配置 ===
设备: cuda
GPU: NVIDIA RTX A6000 (48.0 GB)
批量大小: 32
梯度累积: 4
有效批量: 128
混合精度: True
...
开始训练，最大步数: 100000
Batch size: 32, 梯度累积: 4, 有效batch: 128
✅ 启用混合精度训练 (FP16)
```

## 📊 预期性能指标

### GPU监控 (nvidia-smi)
- **GPU利用率**: 70-90% ✅
- **显存使用**: 20-25GB (FP16模式) ✅
- **温度**: 根据散热情况

### 训练速度
- **每步时间**: 比优化前快2-2.5倍 ✅
- **数据加载**: 无明显等待 ✅

## 🔧 已应用的优化

| 优化项 | 配置 | 效果 |
|--------|------|------|
| Batch Size | 32 | GPU利用率↑ |
| 梯度累积 | 4步 | 有效batch=128 |
| Data Workers | 8个 | 并行加载 |
| Prefetch | 4批/worker | 预取32批 |
| 混合精度 | FP16 | 显存↓40% |
| Pin Memory | 启用 | 传输加速 |
| Non-blocking | 启用 | 异步传输 |

## 📁 优化文档

- **详细指南**: `docs/A6000_OPTIMIZATION_GUIDE.md`
- **优化总结**: `OPTIMIZATION_SUMMARY.md`
- **验证脚本**: `scripts/verify_optimization.py`
- **快速检查**: `scripts/check_optimization.sh`

## 💡 常用命令

### 训练命令
```bash
# 预训练
python3 scripts/train.py --mode pretrain --config medium

# SFT微调
python3 scripts/train.py --mode sft --config medium

# 从checkpoint恢复
python3 scripts/train.py --mode pretrain --config medium --auto-resume

# 自定义参数
python3 scripts/train.py --mode pretrain --config medium \
    --batch-size 32 --learning-rate 3e-4 --max-steps 50000
```

### 监控命令
```bash
# GPU实时监控
nvidia-smi dmon -s um

# 显存详情
nvidia-smi --query-gpu=memory.used,memory.total --format=csv

# 训练进度
tail -f logs/training.log | grep "Step"
```

## ⚙️ 调优建议

### 如果显存不足 (OOM)
```bash
# 减小batch size
python3 scripts/train.py --mode pretrain --config medium --batch-size 24
```

### 如果GPU利用率仍然低
```bash
# 增加data workers
# 编辑 config/training_config.py
num_workers = 12  # 改为12
```

### 如果想要更快训练
```bash
# 减小序列长度
# 编辑 config/training_config.py
max_seq_len = 1024  # 从2048改为1024
```

## 🎯 下一步

1. **监控首个epoch**: 确认GPU利用率达到70-90%
2. **检查显存**: 确认在20-25GB左右（FP16）
3. **观察损失**: 确认loss正常下降
4. **长期训练**: 如果一切正常，继续完整训练

## 📞 故障排查

### GPU利用率仍然低 (<50%)
1. 检查数据加载: `scripts/check_optimization.sh`
2. 增加workers: 编辑`config/training_config.py`
3. 检查是否有其他进程占用GPU

### 显存溢出 (OOM)
1. 降低batch size: `--batch-size 24`
2. 减小序列长度: 修改`max_seq_len`
3. 禁用混合精度: 编辑配置文件

### 训练不稳定
1. 降低学习率: `--learning-rate 1e-4`
2. 检查数据质量
3. 查看梯度norm是否异常

---

**快速支持**:
- 详细文档: `docs/A6000_OPTIMIZATION_GUIDE.md`
- 优化总结: `OPTIMIZATION_SUMMARY.md`
