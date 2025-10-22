# 📋 训练体系剩余工作清单

在完成训练流水线模块化改造后，我们继续推进训练工程的可观测性与稳定性。当前 `TrainingPipeline` 已统一入口与配置快照能力，但仍有若干重点事项需要排期落实。

## ✅ 最新完成

- **统一训练入口**：`TrainingPipeline` 串联数据采样、模型初始化、调度器与监控，启动即生成配置快照与数据集统计，便于复现实验。【F:src/training/pipeline/pipeline.py†L23-L125】
- **监控基线**：`TrainingMonitor` 默认输出损失、PPL、梯度范数与系统资源指标，可直接接入 TensorBoard 观察训练过程。【F:src/training/training_monitor.py†L120-L332】

## 1. 损失函数与训练模式完善

- **DPO/RLHF 专用逻辑**：当前 `TrainingLoopRunner` 仍使用交叉熵损失并假设 batch 仅包含 `input_ids`/`labels`，需要按 `mode` 区分 `(chosen, rejected)` 或强化学习轨迹并接入对应的优化器流程。【F:src/training/pipeline/training_loop.py†L79-L214】
- **DPO 数据集结构**：`DatasetPreparer._create_dpo_dataset` 目前只返回 `chosen` 文本，需要扩展为成对样本、奖励模型输入以及采样权重，才能支撑真实偏好优化。【F:src/training/pipeline/data_manager.py†L196-L214】

## 2. 评估与质量验证

- **离线指标扩展**：验证阶段仍仅产出加权 loss 与 perplexity，需要补充 BLEU/ROUGE、拒答率、身份关键词命中率等指标，并持久化样例级别分数。【F:src/training/pipeline/training_loop.py†L151-L214】
- **回归失败阈值与外部告警**：提示回归尚未整合，需要在训练控制流中加上阈值判断、训练中止选项以及 Webhook/IM 通知能力。【F:src/training/training_monitor.py†L408-L525】

## 3. 训练工程化增强

- **动态 batch / 梯度累积自适应**：目前未在逼近阈值时动态调节 batch size 或梯度累积步数，可复用 `memory_optimizer.py` 的策略实现自适应调度。【F:src/training/memory_optimizer.py†L1-L247】【F:src/training/pipeline/training_loop.py†L56-L214】
- **分布式/多 GPU 支持**：训练器仍以单卡为主，需要封装 DDP 初始化、梯度同步与分布式检查点策略，使训练环境可扩展至多 GPU。【F:src/training/pipeline/pipeline.py†L153-L213】

## 4. 自动化与可观测性

- **告警与实验元数据**：监控器已能保存 JSON 摘要，但尚未与 WandB/Weights & Biases 或内部实验追踪系统联动，需要扩展 run metadata、关键指标阈值与告警管道。【F:src/training/training_monitor.py†L120-L525】
- **训练后评估流水线**：完成训练后仍缺乏自动化的离线评估脚本（如批量生成对比、困惑度统计），需要在 `CheckpointManager.save_final` 之后串联评估与报告生成，实现训练->评估闭环。【F:src/training/pipeline/checkpointing.py†L95-L133】

完成上述事项后，Mini-LLM 的训练体系将在多阶段优化、质量监控和大规模部署方面具备更完善的能力，为进一步改进模型表现打下基础。
