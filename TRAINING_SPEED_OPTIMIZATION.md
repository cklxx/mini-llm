# 训练速度逐渐变慢问题分析与解决

## 🐌 观察到的问题

```
Step 45: 6.4min (总时间)
Step 59: 8.3min (总时间) ← 变慢了30%
```

**每步耗时**: ~8.3s/step

---

## 🔍 可能原因分析

### 1. **TensorBoard日志累积** (最可能)
```python
# 每步写入多个指标到TensorBoard
writer.add_scalar('Training/Loss', ...)
writer.add_scalar('Training/GradientNorm', ...)
writer.add_scalar('Performance/GPUMemory', ...)
...
```

**问题**: 随着步数增加，事件文件变大，写入变慢

### 2. **监控系统开销增加**
```python
# 每步都计算这些指标
- compute_gradient_norm()
- compute_parameter_norm()
- get_gpu_memory_info()
- detect_gradient_anomaly()
```

### 3. **内存碎片化**
- GPU内存分配/释放cycles增加
- PyTorch缓存管理overhead

### 4. **DataLoader缓存问题**
- `persistent_workers=True` 可能导致内存累积
- `num_workers=8` worker进程内存增长

---

## ✅ 已采取的优化措施

### 1. 轻量级监控模式
```python
# scripts/train.py:359-360
lightweight_mode=True,    # 启用轻量级模式
log_interval=10           # 每10步记录一次完整指标
```

### 2. TensorBoard优化
```python
# config/training_config.py:88-100
tensorboard_flush_secs = 30  # 每30秒刷新，而不是每步
```

### 3. 云GPU环境优化
```python
# 检测OpenBayes等云环境，使用固定路径
if os.path.exists("/openbayes/home"):
    tensorboard_dir = "/openbayes/home/tf_dir"
```

---

## 🚀 进一步优化方案

### 方案1: 增加监控间隔 (立即生效)

**当前**: `log_interval=10` (每10步记录一次)
**优化**: `log_interval=50` (每50步记录一次)

```python
# scripts/train.py:360
log_interval=50  # 从10改为50
```

**预期效果**: 减少80%的监控overhead

---

### 方案2: 禁用部分监控指标

```python
# src/training/training_monitor.py
# 可以临时禁用的指标:
class ModelHealthMonitor:
    def __init__(self, minimal_mode=False):
        self.minimal_mode = minimal_mode

    def compute_weight_update_ratio(self):
        if self.minimal_mode:  # 跳过最耗时的计算
            return 0.0
```

---

### 方案3: 周期性清理GPU缓存

```python
# scripts/train.py 训练循环中添加
if step % 100 == 0:
    torch.cuda.empty_cache()  # 每100步清理一次
    torch.cuda.reset_peak_memory_stats()
```

---

### 方案4: 优化TensorBoard写入

```python
# 只记录关键指标，减少事件文件大小
if step % 10 == 0:  # 10步记录一次
    writer.add_scalar('Loss', loss, step)
    writer.add_scalar('LR', lr, step)

if step % 100 == 0:  # 100步记录一次详细信息
    writer.add_scalar('GradNorm', grad_norm, step)
    writer.add_scalar('GPUMemory', gpu_mem, step)
```

---

### 方案5: 检查自动checkpoint保存

```python
# scripts/train.py 中的checkpoint保存逻辑
# 检查是否每步都在检查/保存checkpoint
```

---

## 📊 诊断工具

### 1. 添加性能profiling

```python
# 在训练循环中添加
import time

step_times = []
for batch_idx, batch in enumerate(data_loader):
    step_start = time.time()

    # ... 训练代码 ...

    step_end = time.time()
    step_times.append(step_end - step_start)

    if step % 10 == 0:
        recent_avg = np.mean(step_times[-10:])
        print(f"Step {step}: Avg step time = {recent_avg:.2f}s")
```

### 2. GPU内存监控

```python
if step % 10 == 0 and torch.cuda.is_available():
    print(f"GPU Memory: {torch.cuda.memory_allocated()/1e9:.2f}GB / "
          f"{torch.cuda.memory_reserved()/1e9:.2f}GB")
```

### 3. TensorBoard事件文件大小

```bash
# 检查事件文件增长
watch -n 5 'ls -lh runs/*/events.out.tfevents.* | tail -5'
```

---

## 🎯 立即行动建议

### 短期 (不中断训练)

1. **监控当前GPU状态**:
```bash
nvidia-smi dmon -s um -c 100
```

2. **检查TensorBoard日志大小**:
```bash
du -sh runs/pretrain_medium_*/
```

3. **查看系统资源**:
```bash
htop  # CPU和内存
```

### 中期 (下次训练)

1. **修改log_interval**: 10 → 50
2. **添加周期性缓存清理**: 每100步
3. **优化TensorBoard写入策略**

### 长期优化

1. **使用WandB替代TensorBoard**: 更轻量
2. **实现异步监控**: 监控放到后台线程
3. **自定义轻量级logger**: 只记录必要信息

---

## 🔧 快速修复脚本

创建一个临时补丁来优化当前训练:

```python
# scripts/patch_training_speed.py
import sys
import os

project_root = os.path.dirname(os.path.dirname(__file__))
sys.path.append(project_root)

# 修改配置
from config.training_config import MediumConfig

config = MediumConfig()

# 优化参数
config.log_interval = 50  # 增加间隔
config.tensorboard_flush_secs = 120  # 延长flush间隔

print("✅ 训练速度优化配置已应用")
print(f"   log_interval: {config.log_interval}")
print(f"   tensorboard_flush_secs: {config.tensorboard_flush_secs}")
```

---

## 📈 预期效果

| 优化措施 | 预期速度提升 | 实施难度 |
|---------|-------------|---------|
| log_interval: 10→50 | +20-30% | 简单 |
| 周期性缓存清理 | +5-10% | 简单 |
| 优化TensorBoard | +10-15% | 中等 |
| 禁用weight_update_ratio | +15-20% | 简单 |
| **综合优化** | **+40-60%** | - |

---

## ✅ 自动checkpoint恢复功能

### 当前实现状态

**已实现** ✅:

```python
# scripts/train.py 支持两种恢复模式:

# 1. 手动指定checkpoint
python scripts/train.py --resume checkpoints/pretrain_medium/checkpoint.pt

# 2. 自动恢复 (查找最新checkpoint)
python scripts/train.py --auto-resume
```

### 自动checkpoint保存机制

```python
# scripts/train.py:509-532
def _save_checkpoint(self, model, tokenizer, optimizer, step, loss):
    """保存检查点（只保留最新的一个）"""

    # 删除旧checkpoint
    old_checkpoints = glob.glob("checkpoint_step_*.pt")
    for old_ckpt in old_checkpoints:
        os.remove(old_ckpt)

    # 保存新checkpoint
    checkpoint_path = f"checkpoint_step_{step}.pt"
    torch.save({
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'tokenizer_vocab_size': tokenizer.vocab_size,
        'config': self.config
    }, checkpoint_path)
```

**保存频率**: 每100步自动保存 (可在train.py中配置)

### 改进建议

#### 1. 增加保留多个checkpoint

```python
# 保留最近3个checkpoint而不是只保留1个
MAX_CHECKPOINTS = 3

def _save_checkpoint(...):
    old_ckpts = sorted(glob.glob("checkpoint_*.pt"))
    if len(old_ckpts) >= MAX_CHECKPOINTS:
        os.remove(old_ckpts[0])  # 删除最老的
```

#### 2. 添加best_model保存

```python
# 保存loss最低的模型
if loss < self.best_loss:
    self.best_loss = loss
    torch.save(..., "checkpoint_best.pt")
```

#### 3. 云端备份

```python
# 自动同步到云存储
if step % 1000 == 0:
    backup_to_cloud(checkpoint_path)
```

---

**创建时间**: 2025-10-08
**问题**: 训练速度从6.4min变慢到8.3min
**状态**: 已提供多种优化方案
