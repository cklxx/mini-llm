# MiniGPT 训练效果分析与优化方案

> **最新进展**：训练数据配额、验证拆分、损失掩码、学习率调度、显存监控以及提示回归基线均已在近期迭代中上线，以下分析聚焦仍待解决的痛点与后续优化路径。

## 当前问题分析

- **偏好学习路径缺失**：SFT 管线虽已稳定，但 DPO/RLHF 模式依旧使用交叉熵损失，缺少对 `(chosen, rejected)` 样本与奖励信号的处理，导致偏好训练退化。【F:src/training/pipeline/training_loop.py†L79-L214】【F:src/training/pipeline/data_manager.py†L196-L214】
- **指标体系不完整**：现阶段仅有 loss、perplexity 与基础梯度监控，缺乏 BLEU/ROUGE、拒答率等侧重生成质量的指标，也未设置阈值来触发早停与告警。【F:src/training/pipeline/training_loop.py†L133-L214】【F:src/training/training_monitor.py†L120-L470】
- **自动评估闭环未建立**：训练结束后没有自动执行的离线评估、批量生成比对或报告归档，难以及时判断模型是否可发布或需回滚。【F:src/training/pipeline/checkpointing.py†L95-L133】
- **多卡与自适应显存策略缺位**：当前仍缺乏自动调节 batch size/梯度累积与分布式训练支持，限制了更大模型或多 GPU 扩展。【F:src/training/memory_optimizer.py†L1-L247】【F:src/training/pipeline/pipeline.py†L153-L213】

## 优化方案

### 1. 数据层面
1. **引入可配置的数据集配额**：在 `_get_data_paths` 的基础上，为每个数据源添加采样比例或最大样本数限制，避免身份类语料在加载后占据主导地位，可通过重构 `setup_data_loader` 将不同数据集先各自采样再合并实现。【F:src/training/pipeline/data_manager.py†L123-L224】
2. **改进对话拼接策略**：在 `ConversationDataset` 中显式插入角色标记（如 `<|user|>` / `<|assistant|>`），按轮次交替构造序列，并对每轮 user->assistant 的输出单独掩码，保持模型学习响应的清晰边界。【F:src/training/datasets/conversation.py†L10-L114】
3. **增加训练/验证拆分与打分集**：在数据加载阶段切分出固定比例验证集，或允许传入独立验证文件，配合监控器追踪验证损失、防止过拟合。【F:scripts/train.py†L228-L273】

### 2. 模型与优化
1. **对齐调度与梯度累积**：读取 `config` 中的 `learning_rate`、`warmup_steps`、`gradient_accumulation_steps`，在训练循环中实现线性 warmup + Cosine decay，并按累积步数再执行一次 `optimizer.step()`，同时缩放 loss，确保大批量等效训练稳定。【F:config/training_config.py†L138-L166】【F:src/training/pipeline/pipeline.py†L170-L213】
2. **启用混合精度与梯度检查点**：利用 `config` 的 `mixed_precision` 和 `gradient_checkpointing` 标志，在训练主循环中加入 `torch.cuda.amp.autocast` 与 GradScaler，并在模型构建时开启 checkpoint，以减小显存占用并允许更大 batch；当前实现需统一接口。【F:config/training_config.py†L118-L134】【F:src/training/pipeline/pipeline.py†L170-L213】
3. **正则化与数据增强**：考虑在 SFT 阶段加入 label smoothing 或 dropout 调整，配合对话尾部随机截断、噪声注入等方式提升泛化能力，缓解模型对身份模板的依赖。【F:src/training/datasets/conversation.py†L10-L114】【F:src/training/pipeline/pipeline.py†L170-L213】

### 3. 工程管控
1. **指标体系贯通**：继续扩展 `TrainingMonitor`，在现有 loss/perplexity/回归基础上加入 BLEU、拒答率等指标，并在回调中集中写入 TensorBoard 与 run 摘要。【F:src/training/training_monitor.py†L408-L525】
2. **失败阈值与自动回滚**：在 `TrainingLoopRunner` 中根据验证 loss、回归通过率设置阈值，触发早停或保存回滚点，同时调用监控器进行外部告警。【F:src/training/pipeline/training_loop.py†L133-L214】
3. **配置快照完善**：强化 `TrainingPipeline` 的配置快照，将提示回归、显存监控、评估脚本等参数一并持久化，保证实验可复现。【F:src/training/pipeline/pipeline.py†L41-L79】

## 验证方案

1. **自动化离线指标**
   - 在每次训练阶段结束后运行 `scripts/evaluate_perplexity.py`（需新增）对验证集计算 perplexity 与 token-level accuracy，记录到 `training_monitor`。验证目标：SFT perplexity 逐步下降并保持在预设区间。
   - 使用带身份问答、通用问答的多维提示集，通过批量生成比对参考答案，统计 BLEU/ROUGE 及答非所问率。确保身份类模板输出比例显著下降。

2. **在线生成回归**
   - 扩展 `TrainingMonitor` 提供提示回归回调，在训练过程中定期执行固定提示集回归，记录通过率并对失败样例落盘，后续可叠加关键词匹配、拒答检测等规则强化验收。【F:src/training/training_monitor.py†L408-L525】
   - 扩展 `scripts/generate.py` 或新增评测脚本，对关键提示做基准对比，必要时将差异推送到审阅队列或触发外部告警。【F:scripts/generate.py†L1-L135】

3. **监控与报警**
   - 通过 `TrainingMonitor` 汇总的指标将训练曲线同步到 TensorBoard 或简易 dashboard，设定当验证损失连续 N 次上升或生成回归失败数量超阈时发出提醒。【F:scripts/train.py†L228-L273】
   - 在训练脚本中增加对数据加载统计、掩码比例、有效 batch 数的日志校验，验证每次改动是否达到预期。

通过以上数据、优化策略与验证体系的组合，可以系统性定位并缓解当前模型在身份语料上过拟合的问题，持续提升通用问答质量。
