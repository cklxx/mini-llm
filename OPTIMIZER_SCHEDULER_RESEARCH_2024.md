# 2024-2025 LLM优化器与学习率调度器研究报告

## 📊 研究摘要

基于2024-2025最新论文和实践,本报告调研了适合Transformer/LLM训练的最优配置。

**核心发现**:
- **Muon优化器**: 相比AdamW节省48%计算量,达到相同效果
- **Warmup-Stable-Decay调度器**: 无需预设总步数,支持持续训练
- **混合优化策略**: Muon(2D参数) + AdamW(1D参数) = 最优配置

---

## 🔬 优化器对比 (2024)

### 1. **Muon** (⭐ 推荐用于大模型)

**论文**: https://arxiv.org/abs/2502.16982 (2024)

**核心特点**:
- Momentum Orthogonalized by Newton-Schulz
- 对权重矩阵使用Newton-Schulz正交化
- 相比AdamW节省~48%计算量(FLOPs)
- **Kimi-2 (1T参数)**使用此优化器

**性能数据**:
```
训练效率: Muon达到目标loss只需AdamW的52%训练步数
内存占用: 与SGD-momentum相当 (远低于AdamW)
训练稳定性: loss曲线更平滑
```

**推荐配置**:
```python
# 混合模式 (最优)
opts = get_hybrid_optimizer(
    model,
    muon_lr=0.02,      # Muon学习率是AdamW的20倍
    adamw_lr=1e-3,
    weight_decay=0.01
)

# 使用
loss.backward()
opts['muon'].step()
opts['adamw'].step()
scheduler.step()  # 调度器
opts['muon'].zero_grad()
opts['adamw'].zero_grad()
```

**适用场景**:
- ✅ 大模型预训练 (>1B参数)
- ✅ 计算资源受限
- ✅ 需要更快收敛
- ❌ 极小模型 (<100M) - 增益不明显

---

### 2. **AdamW** (⭐ 仍是黄金标准)

**论文**: Decoupled Weight Decay Regularization (2019)

**为什么AdamW仍然重要**:
- GPT-3, Llama, Chinchilla, BLOOM都使用AdamW
- 经过充分验证,稳定可靠
- 广泛支持,易于调试
- 与各种调度器兼容性最好

**推荐配置** (基于GPT-3):
```python
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=3e-4,
    betas=(0.9, 0.95),  # GPT-3配置 (预训练)
    # betas=(0.9, 0.999),  # BERT配置 (微调)
    eps=1e-8,
    weight_decay=0.1
)
```

**参数说明**:
- `β1=0.9`: 一阶矩(动量)
- `β2=0.95`: 预训练用 | `β2=0.999`: 微调用
- `weight_decay=0.1`: 较大值有助于防止过拟合

---

### 3. **Sophia** (二阶优化器)

**论文**: https://arxiv.org/abs/2305.14342 (2023)

**核心特点**:
- 使用对角Hessian估计 (二阶信息)
- 在相同步数下达到更低loss
- 适合大模型预训练

**性能数据**:
```
540M模型 + Sophia = 770M模型 + AdamW (相同步数)
收敛速度: 比AdamW快~30%
```

**推荐配置**:
```python
optimizer = Sophia(
    model.parameters(),
    lr=1e-4,
    betas=(0.965, 0.99),
    rho=0.04,  # Hessian裁剪参数
    weight_decay=0.1
)
```

**适用场景**:
- ✅ 大模型预训练 (125M-13B)
- ✅ 有充足计算资源
- ❌ 微调任务 - 不如AdamW
- ❌ 极小batch - 效果不佳

---

### 4. **Lion** (内存高效)

**论文**: https://arxiv.org/abs/2302.06675 (2023)

**核心特点**:
- 基于符号(sign)的更新
- 内存占用仅为AdamW的一半
- 在某些NLP任务上优于AdamW

**性能数据**:
```
内存占用: 1x参数量 (AdamW为2x)
收敛速度: 与AdamW相当或更快
```

**推荐配置**:
```python
optimizer = Lion(
    model.parameters(),
    lr=1e-4,  # Lion的lr通常比AdamW低10倍
    betas=(0.9, 0.99),
    weight_decay=0.01
)
```

**适用场景**:
- ✅ 内存受限环境
- ✅ 大batch训练
- ❌ 某些任务表现不稳定

---

## 📈 学习率调度器对比 (2024)

### 1. **Warmup + Cosine Decay** (⭐ 推荐)

**使用者**: GPT-3, Llama, Chinchilla, BLOOM, Pythia

**特点**:
- 业界标准配置
- warmup防止初期不稳定
- cosine平滑下降到10% peak lr

**代码**:
```python
from src.model.optimizers import get_warmup_cosine_schedule

scheduler = get_warmup_cosine_schedule(
    optimizer,
    num_warmup_steps=4000,    # 总步数的5-10%
    num_training_steps=100000,
    num_cycles=0.5,            # 半周期
    min_lr_ratio=0.1           # 最低降到10%
)
```

**学习率曲线**:
```
Step 0-4000:     0 → peak_lr      (线性warmup)
Step 4000-100k:  peak_lr → 0.1*peak_lr  (cosine decay)
```

**推荐参数**:
- Warmup: 总步数的5-10% (太长浪费,太短不稳定)
- Min LR: 10% peak_lr (GPT-3配置)
- Cycles: 0.5 (半周期,单调下降)

---

### 2. **Inverse Sqrt** (Transformer原始)

**使用者**: 原始Transformer论文 "Attention is All You Need"

**特点**:
- 经典配置,简单有效
- warmup后按 `1/√step` 衰减
- 永不降到0,适合持续训练

**代码**:
```python
from src.model.optimizers import get_inverse_sqrt_schedule

scheduler = get_inverse_sqrt_schedule(
    optimizer,
    num_warmup_steps=4000
)
```

**学习率曲线**:
```
Step 0-4000:   线性warmup
Step 4000+:    lr ∝ 1/√step
```

**适用场景**:
- ✅ 经典Transformer架构
- ✅ 不确定总训练步数
- ❌ 现代LLM - cosine更优

---

### 3. **Warmup-Stable-Decay (WSD)** (⭐ 2024最新)

**论文**: https://arxiv.org/abs/2410.05192 (2024)

**核心创新**:
- **无需预设总训练步数**
- 支持持续训练和中途checkpoint
- 3阶段: Warmup → Stable → Decay

**代码**:
```python
from src.model.optimizers import get_wsd_schedule

scheduler = get_wsd_schedule(
    optimizer,
    num_warmup_steps=4000,
    num_stable_steps=80000,  # 长时间稳定训练
    num_decay_steps=16000,   # 最后衰减
    min_lr_ratio=0.1
)
```

**学习率曲线**:
```
Step 0-4k:      0 → peak_lr           (warmup)
Step 4k-84k:    peak_lr (常数)         (stable)
Step 84k-100k:  peak_lr → 0.1*peak_lr (decay)
```

**优势**:
- 可以随时分叉出checkpoint并快速衰减
- 适合不确定训练时长的场景
- 主分支可以无限期stable训练

**适用场景**:
- ✅ 持续预训练
- ✅ 探索性训练
- ✅ 需要灵活性
- ❌ 明确训练预算 - cosine更直接

---

## 🎯 推荐配置方案

### 方案1: 小模型 (<500M参数) - 标准配置

```python
# 优化器
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=3e-4,
    betas=(0.9, 0.95),
    weight_decay=0.1
)

# 调度器
scheduler = get_warmup_cosine_schedule(
    optimizer,
    num_warmup_steps=4000,
    num_training_steps=100000,
    min_lr_ratio=0.1
)
```

**理由**: AdamW经过充分验证,稳定可靠

---

### 方案2: 中型模型 (500M-5B参数) - 平衡配置

```python
# 优化器: Muon + AdamW混合
opts = get_hybrid_optimizer(
    model,
    muon_lr=0.02,
    adamw_lr=1e-3,
    weight_decay=0.01
)

# 调度器: Warmup + Cosine
scheduler_muon = get_warmup_cosine_schedule(
    opts['muon'], 4000, 100000
)
scheduler_adamw = get_warmup_cosine_schedule(
    opts['adamw'], 4000, 100000
)

# 训练循环
loss.backward()
opts['muon'].step()
opts['adamw'].step()
scheduler_muon.step()
scheduler_adamw.step()
opts['muon'].zero_grad()
opts['adamw'].zero_grad()
```

**理由**: Muon提升效率,混合使用兼顾稳定性

---

### 方案3: 大模型 (>5B参数) - 激进优化

```python
# 优化器: Muon主导
opts = get_hybrid_optimizer(
    model,
    muon_lr=0.03,  # 更高学习率
    adamw_lr=1.5e-3,
    weight_decay=0.01
)

# 调度器: WSD (灵活性)
scheduler_muon = get_wsd_schedule(
    opts['muon'],
    num_warmup_steps=8000,
    num_stable_steps=200000,
    num_decay_steps=50000
)
scheduler_adamw = get_wsd_schedule(
    opts['adamw'],
    num_warmup_steps=8000,
    num_stable_steps=200000,
    num_decay_steps=50000
)
```

**理由**:
- Muon节省48%计算量
- WSD支持持续训练
- 适合探索性预训练

---

### 方案4: 微调任务 - 保守配置

```python
# 优化器
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=5e-5,  # 微调用更小lr
    betas=(0.9, 0.999),  # β2=0.999用于微调
    weight_decay=0.01
)

# 调度器: 线性warmup + 线性decay
def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=500,
    num_training_steps=10000
)
```

**理由**: 微调需要保守,防止灾难性遗忘

---

## 📊 性能对比表

| 优化器 | 内存占用 | 收敛速度 | 稳定性 | 推荐场景 |
|--------|---------|---------|--------|---------|
| **Muon** | 1x | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | 大模型预训练 |
| **AdamW** | 2x | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 通用(黄金标准) |
| **Sophia** | 2x | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | 大模型预训练 |
| **Lion** | 1x | ⭐⭐⭐⭐ | ⭐⭐⭐ | 内存受限 |

| 调度器 | 复杂度 | 效果 | 灵活性 | 推荐场景 |
|--------|-------|------|--------|---------|
| **Warmup+Cosine** | 简单 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | 标准预训练 |
| **Inverse Sqrt** | 简单 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | 经典Transformer |
| **WSD** | 中等 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 持续训练 |

---

## 🔬 实验建议

### 对比实验设置

```python
# 实验1: AdamW基线
exp1_optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95))
exp1_scheduler = get_warmup_cosine_schedule(exp1_optimizer, 4000, 100000)

# 实验2: Muon混合
exp2_opts = get_hybrid_optimizer(model, muon_lr=0.02, adamw_lr=1e-3)
exp2_scheduler_muon = get_warmup_cosine_schedule(exp2_opts['muon'], 4000, 100000)
exp2_scheduler_adamw = get_warmup_cosine_schedule(exp2_opts['adamw'], 4000, 100000)

# 实验3: WSD调度器
exp3_optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95))
exp3_scheduler = get_wsd_schedule(exp3_optimizer, 4000, 80000, 16000)
```

### 评估指标

```python
# 记录
- Training Loss曲线
- 验证集Perplexity
- GPU内存峰值
- 训练速度 (tokens/sec)
- 收敛步数

# TensorBoard可视化
writer.add_scalars('Loss', {
    'AdamW': loss_adamw,
    'Muon': loss_muon,
    'WSD': loss_wsd
}, step)
```

---

## ✅ 快速开始

### 当前项目使用

**已实现** (scripts/train.py):
```python
# 当前配置
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=3e-4,
    weight_decay=0.01,
    betas=(0.9, 0.95),
    eps=1e-8
)

scheduler = get_lr_scheduler(
    optimizer,
    warmup_steps=4000,
    max_steps=100000
)
# ✅ Warmup + Cosine Decay已实现
```

### 升级到Muon (可选)

```python
# 修改scripts/train.py
# 替换optimizer部分为:

from src.model.optimizers import get_hybrid_optimizer, get_warmup_cosine_schedule

# 创建混合优化器
opts = get_hybrid_optimizer(
    model,
    muon_lr=0.02,
    adamw_lr=1e-3,
    weight_decay=0.01
)

# 创建调度器
scheduler_muon = get_warmup_cosine_schedule(
    opts['muon'],
    num_warmup_steps=4000,
    num_training_steps=100000
)
scheduler_adamw = get_warmup_cosine_schedule(
    opts['adamw'],
    num_warmup_steps=4000,
    num_training_steps=100000
)

# 训练循环中
loss.backward()
opts['muon'].step()
opts['adamw'].step()
scheduler_muon.step()
scheduler_adamw.step()
opts['muon'].zero_grad()
opts['adamw'].zero_grad()
```

---

## 📚 参考文献

1. **Muon** (2024)
   - 论文: https://arxiv.org/abs/2502.16982
   - 作者: Kimi AI团队

2. **Sophia** (2023)
   - 论文: https://arxiv.org/abs/2305.14342
   - 实验: 125M-13B参数模型

3. **Lion** (2023)
   - 论文: https://arxiv.org/abs/2302.06675
   - Google Research

4. **WSD调度器** (2024)
   - 论文: https://arxiv.org/abs/2410.05192
   - 特点: 无需预设总步数

5. **GPT-3配置** (2020)
   - AdamW with β2=0.95
   - Warmup + Cosine Decay to 10%

6. **Llama/Chinchilla** (2022-2023)
   - AdamW with β2=0.95
   - Cosine Decay to 10%

---

**最后更新**: 2025-10-08
**作者**: MiniGPT Team
**基于**: 2024-2025最新研究
