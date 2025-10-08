# Checkpoint恢复与多设备优化功能

本文档介绍 `train_optimized.py` 脚本新增的checkpoint恢复功能和多设备优化功能（支持CUDA、MPS、CPU）。

## Checkpoint恢复功能

### 功能特性

1. **手动指定checkpoint恢复**
   ```bash
   python scripts/train_optimized.py --resume-from-checkpoint checkpoints/mac_medium/checkpoint_step_4000.pt
   ```

2. **自动恢复最新checkpoint**
   ```bash
   python scripts/train_optimized.py --auto-resume
   ```

3. **智能checkpoint查找**
   - 自动扫描输出目录中的所有checkpoint文件
   - 按步数排序，选择最新的checkpoint
   - 支持`final_model.pt`和`checkpoint_step_*.pt`文件

### 恢复的状态信息

- ✅ 模型权重状态 (`model_state_dict`)
- ✅ 优化器状态 (`optimizer_state_dict`)
- ✅ 训练步数 (`step`)
- ✅ 损失历史 (`loss_history`)
- ✅ 步数历史 (`step_history`)
- ✅ 训练配置 (`config`)

### 使用示例

#### 1. 从特定checkpoint恢复
```bash
python scripts/train_optimized.py \
    --config medium \
    --resume-from-checkpoint checkpoints/mac_medium/checkpoint_step_4000.pt \
    --max-steps 8000
```

#### 2. 自动恢复最新checkpoint
```bash
python scripts/train_optimized.py \
    --config medium \
    --output-dir checkpoints/mac_medium \
    --auto-resume \
    --max-steps 8000
```

## 多设备优化支持

### 设备检测优先级

脚本会按优先级自动检测和选择最佳设备：

1. **CUDA设备** (英伟达GPU) - 最高优先级
2. **MPS设备** (Mac Apple Silicon GPU)  
3. **CPU设备** - 最低优先级

```python
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
```

## CUDA设备优化

### CUDA设备检测和配置

脚本会自动检测英伟达GPU的CUDA支持并应用专门优化：

```python
if torch.cuda.is_available():
    device_count = torch.cuda.device_count()
    device_name = torch.cuda.get_device_name()
    print(f"检测到CUDA设备: {device_name}")
```

### CUDA特定优化

#### 1. cuDNN优化
```python
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True  # 自动寻找最优算法
torch.backends.cudnn.deterministic = False  # 允许非确定性提高性能
```

#### 2. CUDA内存管理
```python
torch.cuda.empty_cache()  # 清空缓存
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
```

#### 3. 批量大小和学习率优化
```python
# CUDA设备可以使用更大的batch_size
config.pretrain.batch_size = min(original_batch * 2, 16)
# CUDA设备可以使用更高的学习率
config.pretrain.learning_rate = max(original_lr, 1e-4)
```

#### 4. DataLoader优化
```python
# CUDA设备使用更多worker和pin_memory
optimal_workers = min(8, cpu_count)
pin_memory = True  # 加速GPU-CPU数据传输
```

#### 5. 模型编译优化
```python
# CUDA设备使用最激进的编译模式
compiled_model = torch.compile(model, mode="max-autotune", dynamic=True)
```

### CUDA优化启动脚本
```bash
./scripts/start_cuda_optimized_training.sh
```

该脚本包含以下CUDA优化配置：
- `--batch-size 8` - 适合CUDA设备的较大批量大小
- `--learning-rate 1e-4` - 稳定的学习率
- `--dataloader-workers 8` - 充分利用CPU-GPU并行性
- `--num-threads 8` - 多线程优化
- `--enable-compile` - 启用CUDA模型编译
- `--auto-resume` - 自动恢复checkpoint

## MPS设备优化

### MPS设备检测

脚本会自动检测Mac设备的MPS（Metal Performance Shaders）支持：

### MPS特定优化

#### 1. 内存管理优化
```python
# 减少内存使用
os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'
# 启用CPU回退
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
# 限制GPU内存使用
torch.mps.set_per_process_memory_fraction(0.8)
```

#### 2. 批量大小自动调整
```python
# MPS设备推荐使用较小的batch_size
config.pretrain.batch_size = min(original_batch, 4)
```

#### 3. 学习率自动优化
```python
# MPS设备使用稍低的学习率保持稳定性
config.pretrain.learning_rate = min(original_lr, 5e-5)
```

#### 4. DataLoader优化
```python
# MPS设备使用更少的worker避免瓶颈
recommended_workers = min(recommended_workers, 1)
```

#### 5. 模型编译优化
```python
# MPS设备使用专门的编译模式
compiled_model = torch.compile(model, mode="reduce-overhead", dynamic=False)
```

### MPS优化启动脚本

为了方便使用，提供了专门的MPS优化启动脚本：

```bash
./scripts/start_mps_optimized_training.sh
```

该脚本包含以下优化配置：
- `--batch-size 2` - 适合MPS设备的批量大小
- `--learning-rate 3e-5` - 稳定的学习率
- `--dataloader-workers 1` - 减少worker数量
- `--num-threads 4` - 适合Apple Silicon的线程数
- `--enable-compile` - 启用MPS模型编译
- `--auto-resume` - 自动恢复checkpoint

## 性能对比

### 多设备训练性能对比

| 设备类型 | 批量大小 | 步数/秒 | 内存使用 | DataLoader Workers | 推荐用途 |
|---------|---------|---------|---------|-------------------|---------|
| CUDA    | 8-16    | ~8.0    | 4-24GB  | 8               | 高性能训练 |
| MPS     | 2-4     | ~2.5    | 8-12GB  | 1               | 日常训练 |
| CPU     | 1-2     | ~1.0    | 4-8GB   | 2-4             | 轻量训练 |

### 设备特定优化建议

1. **CUDA设备**
   - 使用较大的batch_size (8-16)
   - 启用最激进的模型编译优化
   - 使用更多DataLoader worker (8)
   - 启用pin_memory加速数据传输
   - 监控GPU内存和显存使用情况

2. **MPS设备**
   - 使用较小的batch_size (2-4)
   - 启用模型编译优化
   - 使用1个DataLoader worker
   - 监控GPU内存使用情况

3. **CPU设备**
   - 使用最小的batch_size (1-2)
   - 可以使用更多的worker (2-4)
   - 关注CPU和内存使用率

## 故障排除

### 常见问题

1. **CUDA相关问题**
   
   **CUDA不可用**
   ```
   CUDA available: False
   ```
   解决方案：
   - 检查是否安装了支持CUDA的PyTorch版本
   - 验证NVIDIA GPU驱动程序是否正确安装
   - 使用 `nvidia-smi` 检查GPU状态

   **CUDA内存不足**
   ```
   RuntimeError: CUDA out of memory
   ```
   解决方案：
   - 减少batch_size (例如从8降到4或2)
   - 使用 `torch.cuda.empty_cache()` 清理缓存
   - 减少模型大小或序列长度

   **CUDA编译失败**
   ```
   ⚠️  CUDA模型编译失败，尝试 reduce-overhead 模式
   ```
   解决方案：脚本会自动降级编译模式或使用原始模型。

2. **MPS相关问题**

   **MPS编译失败**
   ```
   ⚠️  MPS模型编译失败，使用原始模型: xxx
   ```
   解决方案：脚本会自动回退到非编译模式，训练可以继续进行。

   **MPS内存不足**
   ```
   RuntimeError: MPS backend out of memory
   ```
   解决方案：减少batch_size或重启Python进程释放MPS内存。

3. **通用问题**

   **Checkpoint加载失败**
   ```
   检查点文件不存在: xxx.pt
   ```
   解决方案：检查文件路径是否正确，或使用`--auto-resume`自动查找。

### 调试命令

查看CUDA设备状态：
```python
import torch
print("CUDA available:", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda)
print("GPU count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("GPU name:", torch.cuda.get_device_name(0))
    print("GPU memory:", torch.cuda.get_device_properties(0).total_memory / 1024**3, "GB")
```

查看MPS设备状态：
```python
import torch
print("MPS available:", torch.backends.mps.is_available())
print("MPS built:", torch.backends.mps.is_built())
```

查看设备选择逻辑：
```python
import torch
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("选择CUDA设备")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    print("选择MPS设备")
else:
    device = torch.device("cpu")
    print("选择CPU设备")
print(f"当前设备: {device}")
```

查看可用的checkpoint：
```bash
ls -la checkpoints/mac_medium*/checkpoint_step_*.pt
```

## 总结

新的checkpoint恢复和MPS优化功能大大提升了训练的便利性和效率：

- 🔄 **无缝恢复训练** - 支持手动和自动checkpoint恢复
- 🚀 **MPS设备优化** - 专门针对Apple Silicon设备优化
- 📊 **智能参数调整** - 自动优化batch_size和学习率
- 🛡️ **稳定性保障** - 异常处理和回退机制
- 📈 **性能提升** - 模型编译和内存管理优化

这些功能让在Mac设备上训练大型语言模型变得更加高效和可靠。