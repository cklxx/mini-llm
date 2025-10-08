# 第 04 章 · 训练与优化策略

mini-llm 在训练阶段提供了一套可直接复用的 Trainer、内存优化与监控工具。本章帮助你理解这些模块如何协同工作，并给出调整建议。

## 4.1 训练器骨架
- `PreTrainer` 位于 `src/training/trainer.py`，封装了语言模型预训练的核心循环：取 batch、前向、计算损失、反向、优化器 step、调度器 step。
- `LanguageModelingDataset`/`ConversationDataset` 作为数据入口，分别适配纯文本和 SFT 对话格式，返回 PyTorch 张量或字典供 DataLoader 直接使用。
- 每个 epoch 会记录 `total_loss` 并输出进度条，可快速评估训练是否收敛。

## 4.2 优化器与调度器
- 默认优化器为 `torch.optim.AdamW`，在 `PreTrainer.__init__` 中初始化，权重衰减设为 0.01。
- 学习率调度器使用 `CosineAnnealingLR`，通过 `self.scheduler.step()` 在每个 iteration 更新。
- 若需要替换，可在初始化时注入自定义优化器/调度器，或根据配置继承 `PreTrainer` 后覆写对应逻辑。

## 4.3 内存与混合精度管理
- `MemoryConfig` 描述 AMP、梯度累积、动态批大小等选项，默认启用 `enable_amp=True` 与 `max_grad_norm=1.0`。
- `MixedPrecisionManager` 判断硬件能力后自动启用 `torch.cuda.amp.autocast` 与 `GradScaler`，并提供 `scale_loss` / `autocast_context` API。
- `MemoryMonitor` 提供 `get_memory_info()` 与 `force_cleanup()`，可定期清理显存，避免 OOM。

## 4.4 训练过程监控
- `TrainingMonitor` 在 `src/training/training_monitor.py` 中整合系统指标与模型健康信息，通过 `TrainingMetrics` 数据类存储 loss、学习率、梯度范数等。
- `ModelHealthMonitor.compute_gradient_norm()` 与 `compute_weight_update_ratio()` 帮助捕捉梯度爆炸/消失等异常。
- 内置 TensorBoard 写入 (`SummaryWriter`) 与 Matplotlib 可视化，支持生成轻量化仪表板。

## 4.5 常用调参建议
- **梯度累积**：在显存紧张时，将 `MemoryConfig.gradient_accumulation_steps` 设置为 >1，保持全局 batch 不变。
- **AMP**：若在 CPU/MPS 训练，可关闭 `enable_amp` 避免不支持的路径。
- **监控频率**：对于长时间训练任务，建议将 `TrainingMonitor` 的记录间隔设为 10~20 step，平衡日志量与洞察力。

> 实践提示：mini-llm 支持在 `MemoryConfig` 中启用 `adaptive_batch_size`，结合监控器的 OOM 检测可实现半自动的 batch 调整，适合探索性实验。
