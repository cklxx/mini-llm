# 🏋️ 训练与优化

`src/training` 模块封装了从数据集到优化器的完整训练逻辑。本篇文档介绍关键类、默认策略以及如何扩展。

## 数据集封装
- **LanguageModelingDataset**：接收纯文本列表，自动编码、截断/填充到 `max_length`，并将标签对齐至下一个 token。
- **ConversationDataset**：面向监督微调 (SFT)，输入/输出分别编码并拼接成 `input_ids` 与 `labels`，只对助手回复部分计算损失。

## 训练器
### PreTrainer
- 优化器：`AdamW(lr=1e-4, weight_decay=0.01)`
- 学习率调度：`CosineAnnealingLR`（`T_max=1000`）
- 训练循环：前向 → 计算交叉熵损失（忽略 PAD）→ 反向 → 梯度裁剪（`max_norm=1.0`）→ `optimizer.step()` → `scheduler.step()`
- 进度：使用 `tqdm` 打印 batch 级进度与统计信息

### SFTTrainer
- 继承自 `PreTrainer`
- 调整学习率为 `5e-5`
- 重写 `compute_loss`，通过掩码确保只对标签中非 PAD 的位置累积损失，适合对话类任务

### DPOTrainer
- 同时维护策略模型与参考模型，冻结参考模型参数
- `compute_dpo_loss` 计算选择/拒绝样本相对概率差，支持可调的 `beta`
- 与 `create_trainer('dpo', ...)` 工厂函数配合使用

## 内存与性能优化
`memory_optimizer.py` 提供一系列工具：

- **MemoryConfig**：集中管理 AMP、梯度累积、动态 batch、内存监控等开关
- **MemoryMonitor**：收集 CUDA/MPS/CPU 内存占用，并支持周期性清理缓存
- **MixedPrecisionManager**：封装 `torch.cuda.amp` 的 `autocast` 与 `GradScaler`
- **GradientAccumulator**：实现梯度累积与自动梯度裁剪
- **DynamicBatchSizer**：根据 OOM 情况自动缩减 batch 大小

在实际训练中，可以结合这些组件：
```python
import torch
from src.training.memory_optimizer import MemoryConfig, MemoryOptimizer

# 假设 dataloader、tokenizer、trainer、device 已就绪
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
memory_opt = MemoryOptimizer(model, MemoryConfig(enable_amp=True, gradient_accumulation_steps=4), device)

for step, batch in enumerate(dataloader):
    if isinstance(batch, dict):
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
    else:
        input_ids = batch.to(device)
        labels = torch.cat(
            [input_ids[:, 1:], torch.full((input_ids.size(0), 1), tokenizer.pad_id, device=device)],
            dim=1,
        )

    with memory_opt.optimize_step_context(optimizer) as ctx:
        logits = model(input_ids)
        loss = trainer.compute_loss(logits, labels)
        scaled_loss = memory_opt.compute_loss(loss)
        memory_opt.backward(scaled_loss)

        if ctx["should_update"]:
            # 参数更新由上下文自动完成
            pass
```

## 工厂函数
`create_trainer(training_type, ...)` 根据字符串创建预训练、SFT 或 DPO 训练器，便于在脚本中切换不同阶段的训练逻辑。

## 扩展建议
1. 新的训练阶段：继承 `PreTrainer` 并实现自定义的 `compute_loss`
2. 自定义优化器/调度器：在子类 `__init__` 中替换优化器或调度器
3. 新的内存策略：扩展 `MemoryConfig` 字段并在 `TrainingContext` 中读取

通过以上模块，Mini-LLM 可以覆盖从基础语言模型训练到偏好优化的常见需求，同时保持代码结构清晰易于修改。
