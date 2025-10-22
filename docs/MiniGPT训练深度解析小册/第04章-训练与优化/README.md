# 第 04 章 · 训练与优化策略

mini-llm 在训练阶段通过 `training.pipeline` 提供了统一的训练管道、内存优化与监控工具。本章帮助你理解这些模块如何协同工作，并给出调整建议。

## 4.1 训练器骨架
- `TrainingPipeline` 位于 `src/training/pipeline/pipeline.py`，负责保存配置快照、解析数据、构建模型并驱动 `TrainingLoopRunner` 执行梯度更新，是单一入口的训练 orchestrator。【F:src/training/pipeline/pipeline.py†L23-L213】
- `TrainingLoopRunner` 处理批次调度、梯度累积、DPO 特殊逻辑以及评估调用，内部与 `CheckpointManager` 协作完成中断恢复与最终模型落盘。【F:src/training/pipeline/training_loop.py†L18-L620】
- `LanguageModelingDataset`/`ConversationDataset`/`DPODataset` 作为数据入口，分别适配纯文本、SFT 对话与偏好样本，返回 PyTorch 张量或字典供 DataLoader 直接使用，源码位于 `src/training/datasets/`。

## 4.2 优化器与调度器
- 默认优化器为 `torch.optim.AdamW`，通过 `TrainingPipeline._create_optimizer` 按配置创建，可在配置或子类中覆写学习率、权重衰减等参数。【F:src/training/pipeline/pipeline.py†L200-L213】
- 学习率调度器使用余弦退火 + warmup，`_build_scheduler` 会根据 `max_steps` 与 `warmup_steps` 生成 LambdaLR；在恢复训练时会回放进度。【F:src/training/pipeline/pipeline.py†L190-L209】
- 若需自定义优化逻辑，可扩展 `TrainingPipeline`，或在调用 `train()` 时传入 `target_epochs`、`model_override` 等参数组合自己的循环。【F:src/training/pipeline/pipeline.py†L160-L213】

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
