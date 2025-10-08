# TensorBoard 快速开始

## 🎯 5分钟快速上手

### 1. 启动训练（自动启用TensorBoard）

```bash
# 训练模型（自动生成TensorBoard日志）
make train-sft
```

训练完成后会显示：
```
📊 TensorBoard日志: /path/to/runs/sft_medium_20250108_143052
💡 查看训练过程: tensorboard --logdir=/path/to/runs/sft_medium_20250108_143052
```

### 2. 启动TensorBoard服务

```bash
# 一键启动（最简单）
make tensorboard
```

访问: http://localhost:6006

### 3. 查看训练指标

TensorBoard界面会显示：
- **Training/** - 训练指标（Loss, LR, 梯度范数等）
- **Performance/** - 性能指标（速度, GPU内存, CPU使用等）

## 📍 环境差异

### 本地环境
- TensorBoard日志: `项目根目录/runs/`
- 启动命令: `make tensorboard`

### 云GPU环境 (OpenBayes)
- TensorBoard日志: `/openbayes/home/tf_dir/`
- 平台会自动在界面显示TensorBoard链接
- 也可手动启动: `make tensorboard`

## 🛠️ 常用命令

```bash
# 启动TensorBoard
make tensorboard

# 查看状态
make tensorboard-status

# 停止服务
make tensorboard-stop

# 列出所有训练日志
make tensorboard-list

# 清理30天前的旧日志
make tensorboard-clean
```

## 📖 完整文档

详细使用指南请查看: [docs/TENSORBOARD_GUIDE.md](docs/TENSORBOARD_GUIDE.md)

## ❓ 常见问题

**Q: TensorBoard显示空白？**
A: 检查日志目录是否有内容：`make tensorboard-list`

**Q: 端口被占用？**
A: 使用其他端口：`python scripts/tensorboard_manager.py start --port 6007`

**Q: 云GPU环境找不到日志？**
A: 确认路径：`ls -la /openbayes/home/tf_dir/`
