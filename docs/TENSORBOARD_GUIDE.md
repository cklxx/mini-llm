# TensorBoard 监控指南

本文档提供MiniGPT项目TensorBoard监控的完整使用指南。

## 🎯 概览

MiniGPT项目已完整集成TensorBoard监控系统，支持：
- ✅ 自动检测本地/云GPU环境
- ✅ 统一的日志目录管理
- ✅ 完整的训练指标记录
- ✅ 便捷的管理脚本
- ✅ 轻量级监控模式

## 📂 TensorBoard日志路径

### 本地环境
```
项目根目录/runs/
  ├── sft_medium_20250108_143052/     # SFT训练 (medium配置)
  ├── pretrain_medium_20250108_120000/ # 预训练
  └── dpo_medium_20250108_160000/      # DPO训练
```

### 云GPU环境 (OpenBayes等)
```
/openbayes/home/tf_dir/
  ├── sft_medium_20250108_143052/
  └── ...
```

**自动检测逻辑:**
- 检测到 `/openbayes/home` 目录 → 使用 `/openbayes/home/tf_dir`
- 本地环境 → 使用 `项目根目录/runs/`

## 🚀 快速开始

### 1. 启动训练（自动记录TensorBoard日志）

```bash
# 训练会自动在runs/目录生成TensorBoard日志
make train-sft

# 或使用完整命令
python scripts/train.py --mode sft --config medium
```

**训练完成后会显示:**
```
📊 TensorBoard日志: /path/to/runs/sft_medium_20250108_143052
💡 查看训练过程: tensorboard --logdir=/path/to/runs/sft_medium_20250108_143052
```

### 2. 启动TensorBoard服务

#### 方式一：使用Makefile（推荐）
```bash
# 启动TensorBoard (默认端口6006)
make tensorboard

# 查看状态
make tensorboard-status

# 停止服务
make tensorboard-stop

# 列出所有日志
make tensorboard-list

# 清理30天前的旧日志
make tensorboard-clean
```

#### 方式二：使用管理脚本
```bash
# 启动（默认读取runs/目录）
python scripts/tensorboard_manager.py start

# 指定端口和日志目录
python scripts/tensorboard_manager.py start --port 6007 --logdir runs/

# 指定特定训练日志
python scripts/tensorboard_manager.py start --logdir runs/sft_medium_20250108_143052/

# 停止服务
python scripts/tensorboard_manager.py stop

# 重启服务
python scripts/tensorboard_manager.py restart

# 查看状态
python scripts/tensorboard_manager.py status

# 列出所有日志
python scripts/tensorboard_manager.py list

# 清理旧日志（保留最近30天）
python scripts/tensorboard_manager.py clean --days 30

# 模拟清理（不实际删除）
python scripts/tensorboard_manager.py clean --days 30 --dry-run
```

#### 方式三：手动启动
```bash
# 基础启动
tensorboard --logdir=runs/

# 指定端口
tensorboard --logdir=runs/ --port 6007

# 允许远程访问
tensorboard --logdir=runs/ --host 0.0.0.0 --port 6006
```

### 3. 访问TensorBoard

- **本地访问:** http://localhost:6006
- **远程访问:** http://<服务器IP>:6006

## 📊 记录的训练指标

### 核心训练指标
- `Training/Loss` - 训练损失
- `Training/LearningRate` - 学习率
- `Training/GradientNorm` - 梯度范数
- `Training/ParameterNorm` - 参数范数
- `Training/WeightUpdateRatio` - 权重更新比例（轻量级模式下每10步记录）

### 性能指标
- `Performance/SamplesPerSec` - 训练速度（样本/秒）
- `Performance/GPUMemoryGB` - GPU内存使用（GB）
- `Performance/CPUUsagePercent` - CPU使用率（%）
- `Performance/RAMUsageGB` - 内存使用（GB）

### 轻量级模式优化
为提升训练性能，项目默认启用轻量级监控模式：
- ✅ 每步记录关键指标（Loss, LR）
- ✅ 每10步记录完整指标
- ✅ 自动跳过耗时的权重更新分析
- ⚡ 降低监控开销 ~90%

## 🔧 高级配置

### 在代码中自定义配置

#### 1. 修改TensorBoard配置 (`config/training_config.py`)
```python
class BaseConfig:
    def __init__(self):
        # TensorBoard配置
        self.tensorboard_dir = "custom_logs"  # 自定义日志目录
        self.enable_tensorboard = True        # 启用/禁用
        self.tensorboard_flush_secs = 60      # 刷新间隔（秒）
```

#### 2. 调整监控粒度 (`scripts/train.py`)
```python
monitor = TrainingMonitor(
    model=model,
    log_dir=tensorboard_dir,
    enable_tensorboard=True,
    lightweight_mode=False,  # 禁用轻量级模式（完整监控）
    log_interval=1           # 每步记录（轻量级时有效）
)
```

### 云GPU环境特殊配置

#### OpenBayes平台
```bash
# 确认TensorBoard路径
ls -la /openbayes/home/tf_dir/

# 训练时会自动使用该路径
python scripts/train.py --mode sft --config medium

# 平台会自动识别并在界面显示TensorBoard链接
```

#### 其他云平台
如果平台需要特定路径，修改 `config/training_config.py`:
```python
# 添加自定义云平台检测
cloud_tb_dir = "/your/cloud/platform/tb_dir"
if os.path.exists("/your/cloud/platform") and os.access("/your/cloud/platform", os.W_OK):
    self.tensorboard_dir = cloud_tb_dir
    print(f"🌐 检测到云平台，TensorBoard日志: {cloud_tb_dir}")
```

## 📈 TensorBoard使用技巧

### 1. 对比多次训练

```bash
# 同时查看多个训练日志
tensorboard --logdir_spec=\
run1:runs/sft_medium_20250108_120000,\
run2:runs/sft_medium_20250108_140000,\
run3:runs/sft_medium_20250108_160000
```

### 2. 平滑曲线显示

在TensorBoard界面：
- 左侧面板找到 "Smoothing" 滑块
- 调整到 0.6-0.8 可平滑噪声
- 调整到 0 显示原始数据

### 3. 自定义时间范围

- 在图表下方拖动时间轴
- 点击 "Relative" / "Wall" 切换时间显示方式
- 使用鼠标滚轮缩放

### 4. 下载数据

- 点击图表右下角下载图标
- 可下载CSV格式原始数据
- 可下载SVG/PNG格式图片

## 🛠️ 故障排查

### 问题1: TensorBoard显示"No dashboards are active"

**原因:** 日志目录为空或路径不正确

**解决:**
```bash
# 检查日志目录
ls -la runs/

# 确认是否有事件文件
find runs/ -name "events.out.tfevents.*"

# 使用正确的日志路径启动
tensorboard --logdir=runs/sft_medium_20250108_143052/
```

### 问题2: 端口被占用

**错误信息:** `TensorBoard failed to bind to port`

**解决:**
```bash
# 使用其他端口
make tensorboard --port 6007

# 或查找并关闭占用进程
lsof -ti:6006 | xargs kill
```

### 问题3: 云GPU环境找不到TensorBoard

**检查步骤:**
```bash
# 1. 确认路径
echo $TENSORBOARD_DIR
ls -la /openbayes/home/tf_dir/

# 2. 确认写入权限
touch /openbayes/home/tf_dir/test.txt
rm /openbayes/home/tf_dir/test.txt

# 3. 查看训练输出是否显示正确路径
# 应显示: 🌐 检测到云GPU环境，TensorBoard日志: /openbayes/home/tf_dir
```

### 问题4: 指标更新不及时

**原因:** 刷新间隔设置过长

**解决:**
```python
# 方式1: 修改配置文件
self.tensorboard_flush_secs = 10  # 降低到10秒

# 方式2: 启动时指定
python scripts/tensorboard_manager.py start --reload-interval 10
```

### 问题5: GPU内存显示为0

**原因:** 非CUDA设备或MPS设备

**说明:**
- Apple Silicon (MPS): GPU内存监控为近似值
- CPU训练: GPU内存显示为0（正常）

## 📚 日志管理最佳实践

### 1. 定期清理旧日志

```bash
# 每月清理一次（保留30天）
make tensorboard-clean

# 手动清理（保留最近7天）
python scripts/tensorboard_manager.py clean --days 7

# 模拟运行（查看将要删除的内容）
python scripts/tensorboard_manager.py clean --days 30 --dry-run
```

### 2. 重要实验备份

```bash
# 备份重要训练日志
cp -r runs/sft_medium_20250108_143052 ~/backups/best_run/

# 或压缩保存
tar -czf sft_medium_best.tar.gz runs/sft_medium_20250108_143052/
```

### 3. 日志命名约定

项目自动按以下格式命名:
```
{mode}_{config}_{timestamp}/
```

例如:
- `sft_medium_20250108_143052` - SFT训练, medium配置, 2025年1月8日14:30:52
- `pretrain_large_20250108_120000` - 预训练, large配置

### 4. 磁盘空间监控

```bash
# 查看日志目录大小
du -sh runs/

# 查看各个训练日志大小
du -sh runs/*/

# 清理前检查将释放的空间
python scripts/tensorboard_manager.py clean --days 30 --dry-run
```

## 🎓 进阶用法

### 1. 自定义标量记录

如需记录额外指标，修改 `src/training/training_monitor.py`:

```python
def log_custom_metric(self, step, name, value):
    """记录自定义指标"""
    if self.tensorboard_writer:
        self.tensorboard_writer.add_scalar(f'Custom/{name}', value, step)
```

### 2. 记录模型图结构

```python
# 在训练开始时添加
if self.tensorboard_writer:
    dummy_input = torch.randn(1, 512).to(device)
    self.tensorboard_writer.add_graph(model, dummy_input)
```

### 3. 记录直方图

```python
# 记录参数分布
for name, param in model.named_parameters():
    self.tensorboard_writer.add_histogram(f'Parameters/{name}', param, step)

# 记录梯度分布
for name, param in model.named_parameters():
    if param.grad is not None:
        self.tensorboard_writer.add_histogram(f'Gradients/{name}', param.grad, step)
```

## 📖 参考资源

- **TensorBoard官方文档:** https://www.tensorflow.org/tensorboard
- **PyTorch TensorBoard教程:** https://pytorch.org/tutorials/intermediate/tensorboard_tutorial.html
- **项目源码:**
  - 配置文件: `config/training_config.py`
  - 监控系统: `src/training/training_monitor.py`
  - 管理脚本: `scripts/tensorboard_manager.py`
  - 训练脚本: `scripts/train.py`

## ❓ 常见问题

**Q: 为什么要用轻量级模式？**

A: 完整监控模式下，每步记录所有指标会增加15-20%的训练开销。轻量级模式降低到2-3%，同时保留关键指标。

**Q: 如何在训练中途启用完整监控？**

A: 修改 `scripts/train.py` 中的 `lightweight_mode=False`，然后恢复训练。

**Q: TensorBoard支持多GPU训练吗？**

A: 支持。多GPU训练时，监控器会记录主GPU (GPU:0) 的指标。

**Q: 如何导出TensorBoard数据到论文？**

A: 在TensorBoard界面点击下载按钮，可导出CSV数据或SVG图片，然后用matplotlib/Excel处理。

---

**更新日期:** 2025-01-08
**版本:** v1.0
**维护者:** MiniGPT Team
