# MiniGPT 增强训练脚本说明

## 🚀 新功能概览

`train_optimized.py` 脚本已经进行了重大增强，支持：

- ✅ **全量数据训练** - 使用完整的数据集进行训练
- ✅ **重新训练Tokenizer** - 从多个数据源重新构建词汇表
- ✅ **自定义数据集** - 灵活选择训练数据文件
- ✅ **实时损失曲线** - 自动绘制并保存训练损失图表
- ✅ **灵活参数配置** - 命令行覆盖所有训练参数
- ✅ **资源监控** - 防止系统过载
- 🚀 **多线程优化** - PyTorch原生多线程和性能优化
- 🔥 **模型编译** - PyTorch 2.0+ torch.compile 加速
- ⚡ **数据加载优化** - 智能DataLoader worker配置

## 📊 使用方法

### 基础用法

```bash
# 使用默认medium配置训练
python scripts/train_optimized.py --config medium

# 使用全量数据训练
python scripts/train_optimized.py --config medium --use-full-data

# 重新训练tokenizer
python scripts/train_optimized.py --config medium --retrain-tokenizer
```

### 高级用法

```bash
# 完整的自定义训练
python scripts/train_optimized.py \
    --config medium \
    --use-full-data \
    --retrain-tokenizer \
    --tokenizer-vocab-size 20000 \
    --learning-rate 3e-5 \
    --max-steps 8000 \
    --batch-size 2 \
    --warmup-steps 800 \
    --output-dir "checkpoints/my_model" \
    --save-steps 400 \
    --plot-loss
```

## 🔧 参数说明

### 数据相关参数

- `--use-full-data`: 使用全量数据集 (pretrain_hq.jsonl, sft_1024.jsonl, sft_512.jsonl, r1_mix_1024.jsonl)
- `--data-files`: 指定具体的数据文件列表
- `--max-data-size`: 限制数据条数 (0表示不限制)

### Tokenizer相关参数

- `--retrain-tokenizer`: 重新训练tokenizer
- `--tokenizer-vocab-size`: Tokenizer词汇表大小 (默认: 15000)
- `--tokenizer-samples`: 训练tokenizer使用的样本数量 (默认: 100000)

### 训练参数

- `--learning-rate`: 学习率
- `--max-steps`: 最大训练步数
- `--batch-size`: 批次大小
- `--warmup-steps`: 预热步数
- `--save-steps`: 保存检查点的步数间隔

### 输出和可视化

- `--output-dir`: 输出目录
- `--plot-loss`: 启用实时损失曲线绘制

### 系统资源

- `--max-cpu`: 最大CPU使用率 (%)
- `--max-memory`: 最大内存使用率 (%)
- `--disable-monitoring`: 禁用资源监控

### 多线程和性能优化

- `--num-threads`: PyTorch线程数 (不指定则自动优化)
- `--dataloader-workers`: DataLoader worker数量 (不指定则自动优化)
- `--enable-compile`: 启用PyTorch模型编译优化 (需要PyTorch 2.0+)
- `--disable-optimizations`: 禁用所有性能优化

## 📈 损失曲线功能

启用 `--plot-loss` 参数后，训练过程中会自动：

1. **实时保存损失曲线**: 每次保存检查点时生成损失图表
2. **双重视图**: 包含完整训练历史和最近1000步的详细视图
3. **统计信息**: 显示当前、最小、最大、平均损失值
4. **自动保存**: 保存到 `{output_dir}/plots/` 目录

生成的文件：
- `loss_curve_step_{step}.png` - 每个检查点的损失曲线
- `loss_curve_latest.png` - 最新的损失曲线
- `loss_curve_final_step_{step}.png` - 最终训练完成的损失曲线

## 🎯 推荐配置

### 全新训练 (推荐，含多线程优化)

```bash
python scripts/train_optimized.py \
    --config medium \
    --use-full-data \
    --retrain-tokenizer \
    --tokenizer-vocab-size 20000 \
    --learning-rate 3e-5 \
    --max-steps 8000 \
    --batch-size 2 \
    --warmup-steps 800 \
    --save-steps 400 \
    --plot-loss \
    --output-dir "checkpoints/mac_medium_v2" \
    --num-threads 6 \
    --dataloader-workers 2 \
    --enable-compile
```

### 快速验证

```bash
python scripts/train_optimized.py \
    --config small \
    --data-files "pretrain_200.jsonl" \
    --max-data-size 10000 \
    --max-steps 1000 \
    --batch-size 8 \
    --plot-loss \
    --output-dir "checkpoints/test_run"
```

### 资源受限环境

```bash
python scripts/train_optimized.py \
    --config small \
    --data-files "sft_mini_512.jsonl" \
    --max-data-size 50000 \
    --batch-size 4 \
    --max-steps 3000 \
    --max-cpu 70 \
    --max-memory 70 \
    --plot-loss
```

## 🚀 性能优化功能

### 多线程优化

脚本会自动应用以下PyTorch性能优化：

1. **线程数优化**: 自动检测系统核心数并设置最佳线程数
2. **环境变量优化**: 设置 OMP_NUM_THREADS, MKL_NUM_THREADS 等
3. **JIT融合优化**: 启用 PyTorch JIT 融合策略
4. **MPS优化**: Mac GPU (Metal Performance Shaders) 专项优化
5. **MKL-DNN优化**: Intel 数学核心库优化

### 模型编译加速

使用 `--enable-compile` 启用 PyTorch 2.0+ 的 torch.compile:

- **MPS设备**: 使用 "reduce-overhead" 模式
- **CUDA设备**: 使用 "max-autotune" 模式
- **自动回退**: 不支持时自动跳过

### 数据加载优化

- **智能Worker配置**: 根据设备类型和批次大小自动优化
- **持久化Workers**: 减少worker重启开销
- **预取因子**: 优化数据管道吞吐量

## 🚨 重要提示

1. **清理旧数据**: 如果要全新训练，建议先删除旧的检查点和tokenizer文件
2. **资源监控**: 建议保持资源监控开启，防止系统卡死
3. **批次大小**: Mac环境下建议使用较小的批次大小 (2-4)
4. **学习率**: 新训练建议使用较小的学习率 (3e-5 到 5e-5)
5. **预热步数**: 充分的预热有助于训练稳定性
6. **多线程设置**: 线程数不宜过多，建议4-8个线程
7. **模型编译**: MPS设备上启用编译可显著提速

## 📂 输出文件结构

```
checkpoints/your_model/
├── tokenizer.pkl                    # 训练的分词器
├── checkpoint_step_400.pt           # 定期检查点
├── checkpoint_step_800.pt
├── ...
├── final_model.pt                   # 最终模型
└── plots/                           # 损失曲线图片
    ├── loss_curve_step_400.png
    ├── loss_curve_latest.png
    └── loss_curve_final_step_xxx.png
```

## 🔄 故障排除

### 常见问题

1. **内存不足**: 减小 `--batch-size` 或 `--max-data-size`
2. **训练不收敛**: 降低 `--learning-rate` 或增加 `--warmup-steps`
3. **系统卡死**: 检查 `--max-cpu` 和 `--max-memory` 设置
4. **Tokenizer训练失败**: 检查数据文件格式或减少 `--tokenizer-samples`

### 监控训练进度

使用监控脚本实时查看训练状态：

```bash
python scripts/monitor_training.py
```

或者查看损失曲线：

```bash
python scripts/plot_training_curves.py --checkpoint-dir checkpoints/your_model
```

---

**祝训练顺利！** 🎉