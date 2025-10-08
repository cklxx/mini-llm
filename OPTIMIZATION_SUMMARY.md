# A6000 GPU 训练优化总结

## 🎯 优化目标
- **问题**: GPU利用率仅30%，显存占用大
- **硬件**: NVIDIA A6000 (48GB VRAM), 16核CPU, 60GB内存
- **目标**: 提升GPU利用率至70-90%，加速训练2-3倍

## 📊 优化对比

### 1. Batch Size配置
| 项目 | 优化前 | 优化后 | 改进 |
|------|--------|--------|------|
| batch_size | 12 | **32** | **+167%** |
| gradient_accumulation | 10 | **4** | 优化流程 |
| 有效batch | 120 | **128** | 保持稳定 |

**文件**: `config/training_config.py:137-138`

### 2. 数据加载优化
| 项目 | 优化前 | 优化后 | 改进 |
|------|--------|--------|------|
| num_workers | 0 | **8** | **并行加载** |
| prefetch_factor | None | **4** | **预取32个batch** |
| pin_memory | False | **True** | **加速传输** |
| persistent_workers | False | **True** | **减少开销** |

**文件**: `config/training_config.py:107-114`, `scripts/train.py:196-205`

### 3. 混合精度训练
| 项目 | 优化前 | 优化后 | 改进 |
|------|--------|--------|------|
| 精度 | FP32 | **FP16** | **显存减半** |
| Tensor Core | 未使用 | **启用** | **2-3倍加速** |
| GradScaler | 无 | **启用** | **稳定训练** |

**文件**: `scripts/train.py:316-320, 385-401`

### 4. 其他优化
| 优化项 | 状态 | 说明 |
|--------|------|------|
| 梯度累积逻辑 | ✅ | 减少optimizer.step()调用 |
| Non-blocking传输 | ✅ | CPU-GPU异步传输 |
| 梯度检查点 | ✅ | 节省显存 |
| TF32加速 | ✅ | Ampere架构优化 |

## 🚀 预期性能提升

### 训练速度
- **GPU利用率**: 30% → **70-90%** (提升 **2-3倍**)
- **每步训练时间**: 基线 → **0.4-0.5倍** (快 **2-2.5倍**)
- **吞吐量**: 基线 → **2-3倍**

### 资源利用
- **显存占用**: ~30GB → **~20-25GB** (节省 **20-30%**)
- **显存利用率**: ~62% → **~50%** (更高效)
- **CPU利用率**: ~10% → **~50%** (8 workers)

### 训练效率
- **数据加载瓶颈**: 消除 (8 workers + 预取)
- **GPU空闲时间**: 大幅减少
- **batch处理速度**: 提升2-3倍

## 📝 修改文件清单

### 1. `config/training_config.py`
```python
# Line 137-150: Batch size优化
batch_size = 32  # 12 → 32
gradient_accumulation_steps = 4  # 10 → 4

# Line 107-114: 数据加载优化
num_workers = 8  # 0 → 8
prefetch_factor = 4  # None → 4
```

### 2. `scripts/train.py`
```python
# Line 196-205: DataLoader配置
num_workers=config.num_workers,  # 新增
pin_memory=config.pin_memory,  # 新增
persistent_workers=config.persistent_workers,  # 新增
prefetch_factor=config.prefetch_factor,  # 新增

# Line 316-325: 混合精度训练
scaler = torch.cuda.amp.GradScaler()  # 新增

# Line 356-449: 训练循环重构
- 添加梯度累积逻辑
- 添加混合精度支持
- 添加non-blocking传输
```

### 3. 新增文档
- `docs/A6000_OPTIMIZATION_GUIDE.md`: 详细优化指南
- `scripts/verify_optimization.py`: 验证脚本
- `scripts/check_optimization.sh`: 快速检查脚本

## 🔧 使用方法

### 1. 验证优化配置
```bash
# 快速检查
bash scripts/check_optimization.sh

# 详细验证（需要PyTorch环境）
python3 scripts/verify_optimization.py
```

### 2. 开始训练
```bash
# Medium模型 (自动应用所有优化)
python3 scripts/train.py --mode pretrain --config medium

# 查看配置
python3 -c "from config.training_config import get_medium_config; get_medium_config()"
```

### 3. 监控性能
```bash
# 实时监控GPU
watch -n 1 nvidia-smi

# 查看训练日志
tail -f logs/training.log
```

## 📈 性能监控指标

### 期望看到的改进
- ✅ GPU利用率: 70-90%
- ✅ GPU显存: 20-25GB (FP16)
- ✅ 每步时间: 减少50-60%
- ✅ 数据加载: 无明显等待

### 如何验证优化生效
1. **GPU利用率**:
   ```bash
   nvidia-smi dmon -s u
   # 应该看到sm列(GPU利用率)在70-90%
   ```

2. **显存使用**:
   ```bash
   nvidia-smi
   # Memory-Usage应该在20-25GB左右
   ```

3. **训练日志**:
   ```
   开始训练，最大步数: 100000
   Batch size: 32, 梯度累积: 4, 有效batch: 128
   ✅ 启用混合精度训练 (FP16)
   ```

## 🎓 优化原理

### 为什么GPU利用率低？
1. **数据加载慢** (num_workers=0)
   - 单进程加载，GPU等待数据
   - **解决**: 8个worker并行加载

2. **Batch size太小** (batch_size=12)
   - GPU算力未充分利用
   - **解决**: 提升至32

3. **未使用混合精度**
   - 显存和带宽浪费
   - **解决**: 启用FP16

### 优化如何提升性能？

```
优化前:
[CPU加载] → 等待 → [GPU计算(30%)] → 等待 → [CPU加载] → ...
                ↑ 数据瓶颈        ↑ batch太小

优化后:
[Worker1] → 预取4批
[Worker2] → 预取4批     [GPU计算(85%)]
[Worker3] → 预取4批  →  持续供应   → FP16加速
...                     大batch(32)   Tensor Core
[Worker8] → 预取4批
```

## ⚠️ 故障排查

### 如果显存不足
```python
# 调整 config/training_config.py
batch_size = 24  # 降低
gradient_accumulation_steps = 5  # 增加
```

### 如果数据加载慢
```python
# 调整 config/training_config.py
num_workers = 12  # 增加
prefetch_factor = 6  # 增加
```

### 如果训练不稳定
```python
# 禁用混合精度
mixed_precision = False
```

## 🎯 进一步优化建议

### 1. 编译优化 (PyTorch 2.0+)
```python
model = torch.compile(model, mode="reduce-overhead")
```

### 2. Flash Attention 2
```bash
pip install flash-attn --no-build-isolation
```

### 3. 数据预处理优化
- 预先tokenize数据并缓存
- 使用mmap加载大文件
- WebDataset格式

### 4. 分布式训练
```bash
torchrun --nproc_per_node=2 scripts/train.py
```

## 📚 参考资料
- [PyTorch Performance Tuning](https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html)
- [Mixed Precision Training](https://pytorch.org/docs/stable/amp.html)
- [Efficient DataLoader](https://pytorch.org/docs/stable/data.html)
- [NVIDIA A6000 Specs](https://www.nvidia.com/en-us/design-visualization/rtx-a6000/)

---

**优化完成日期**: 2025-10-08
**优化版本**: v1.0
**预期加速比**: 2-3x
**显存节省**: 20-30%
