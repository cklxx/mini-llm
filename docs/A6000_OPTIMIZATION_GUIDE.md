# A6000 GPU 训练优化指南

## 硬件配置
- **GPU**: NVIDIA A6000 (48GB VRAM)
- **CPU**: 16核
- **内存**: 60GB
- **存储**: 200GB 工作空间

## 优化前的问题

### 🔴 性能瓶颈
1. **GPU利用率低**: 仅30%
2. **显存占用大**: 未充分利用48GB显存
3. **数据加载慢**: 单进程加载(num_workers=0)
4. **Batch Size过小**: batch_size=12对于48GB显存太保守

### 🔴 根本原因
- 数据加载成为瓶颈，GPU等待数据
- 未启用混合精度训练
- 梯度累积配置不合理
- 未充分利用多核CPU

## 优化方案

### ✅ 1. Batch Size优化 (training_config.py:137-138)
```python
# 优化前
batch_size = 12
gradient_accumulation_steps = max(1, 128 // 12) = 10

# 优化后
batch_size = 32  # 提升2.7倍
gradient_accumulation_steps = 4
# 有效batch = 32 × 4 = 128 (保持不变)
```

**预期效果**:
- 更高的GPU利用率
- 更好的内存带宽利用
- 更稳定的梯度更新

### ✅ 2. 数据加载优化 (training_config.py:107-114)
```python
# 优化前
num_workers = 0  # 单进程
prefetch_factor = None

# 优化后
num_workers = 8  # 使用8个worker进程 (16核CPU的一半)
prefetch_factor = 4  # 每个worker预取4个batch
pin_memory = True
persistent_workers = True
```

**预期效果**:
- 数据加载并行化，消除CPU-GPU数据传输瓶颈
- 预取机制确保GPU始终有数据可处理
- GPU利用率预计提升至70-90%

### ✅ 3. 混合精度训练 (scripts/train.py:316-320)
```python
# 启用FP16自动混合精度
scaler = torch.cuda.amp.GradScaler()
with torch.cuda.amp.autocast():
    outputs = model(input_ids)
    loss = criterion(...)
```

**预期效果**:
- 显存占用减少约40-50%
- 训练速度提升30-40%
- 支持更大的batch size
- Tensor Core加速

### ✅ 4. 梯度累积优化 (scripts/train.py:360-446)
```python
# 优化后的梯度累积逻辑
for batch_idx, batch in enumerate(data_loader):
    # 前向+反向传播
    loss = loss / accumulation_steps
    loss.backward()

    # 只在累积步数达到时更新参数
    if (batch_idx + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

**预期效果**:
- 减少optimizer.step()调用频率
- 更稳定的大batch训练
- 保持有效batch size = 128

### ✅ 5. 数据传输优化 (scripts/train.py:375)
```python
# 优化前
batch = batch.to(self.device)

# 优化后
batch = batch.to(self.device, non_blocking=True)
```

**预期效果**:
- CPU到GPU异步数据传输
- 与计算并行执行

## 性能预期

### 训练速度提升
| 指标 | 优化前 | 优化后 | 提升 |
|------|--------|--------|------|
| GPU利用率 | ~30% | 70-90% | **2-3倍** |
| Batch处理速度 | 基线 | 2-3倍 | **2-3倍** |
| 每步训练时间 | 基线 | 0.4-0.5倍 | **2-2.5倍** |
| 显存占用 | ~30GB | ~20-25GB | **节省20-30%** |

### 资源利用
- **GPU显存**: 从30GB降至20-25GB (混合精度)
- **GPU利用率**: 从30%提升至70-90%
- **CPU利用率**: 8个worker进程充分利用多核
- **数据吞吐**: 8×预取保证持续数据供应

## 使用方法

### 1. 直接训练（自动应用优化）
```bash
# Medium模型 (针对A6000优化)
python3 scripts/train.py --mode pretrain --config medium

# 查看配置信息
python3 -c "from config.training_config import get_medium_config; get_medium_config()"
```

### 2. 监控训练性能
```bash
# 使用nvidia-smi监控GPU
watch -n 1 nvidia-smi

# 训练日志会显示:
# - Batch size: 32
# - 梯度累积: 4
# - 有效batch: 128
# - 混合精度: True
```

### 3. 性能基准测试
```bash
# 测试数据加载速度
python3 -c "
from config.training_config import get_medium_config
config = get_medium_config()
print(f'Workers: {config.num_workers}')
print(f'Prefetch: {config.prefetch_factor}')
print(f'Batch: {config.batch_size}')
print(f'Accumulation: {config.gradient_accumulation_steps}')
"
```

## 故障排查

### 如果显存不足
```python
# 在training_config.py中调整
batch_size = 24  # 降低batch size
gradient_accumulation_steps = 5  # 增加累积步数
```

### 如果数据加载慢
```python
# 增加worker数量
num_workers = 12  # 可以尝试更多worker
prefetch_factor = 6  # 增加预取
```

### 如果训练不稳定
```python
# 禁用混合精度
mixed_precision = False

# 或使用梯度裁剪
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

## 进一步优化建议

### 1. 启用PyTorch编译 (仅PyTorch 2.0+)
```python
model = torch.compile(model, mode="reduce-overhead")
```

### 2. 使用Flash Attention 2
```bash
pip install flash-attn --no-build-isolation
```

### 3. 分布式训练 (多GPU)
```bash
torchrun --nproc_per_node=2 scripts/train.py --mode pretrain
```

### 4. 优化数据集加载
- 使用内存映射(mmap)加载大文件
- 预处理数据并缓存token化结果
- 使用WebDataset格式

## 验证清单

- [x] Batch size从12提升至32
- [x] 梯度累积从10降至4
- [x] DataLoader worker从0提升至8
- [x] 启用prefetch_factor=4
- [x] 启用pin_memory=True
- [x] 启用persistent_workers=True
- [x] 启用混合精度训练(FP16)
- [x] 优化梯度累积逻辑
- [x] 启用non_blocking数据传输

## 参考资料
- [PyTorch Performance Tuning Guide](https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html)
- [NVIDIA A6000 Specifications](https://www.nvidia.com/en-us/design-visualization/rtx-a6000/)
- [Mixed Precision Training](https://pytorch.org/docs/stable/amp.html)
- [Efficient DataLoader](https://pytorch.org/docs/stable/data.html#single-and-multi-process-data-loading)
