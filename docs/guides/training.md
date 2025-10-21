# 🏋️ 训练流水线概览

`src/training` 目录包含从数据采样、模型初始化到日志监控的完整训练体系。本篇文档梳理新的 `training.pipeline` 架构以及扩展点，帮助你快速对齐核心组件职责。

## 总体结构

训练入口位于 [`scripts/train.py`](../scripts/train.py)，它会调用 [`training.pipeline.cli`](../src/training/pipeline/cli.py) 解析命令行参数并构造 `MiniGPTTrainer`。`MiniGPTTrainer` 再协同以下模块完成一次完整 run：【F:src/training/pipeline/app.py†L25-L162】

- **TrainingEnvironment**：设置随机种子、检测设备、创建输出目录并持久化配置快照与数据集统计，保证实验可复现。【F:src/training/pipeline/environment.py†L10-L64】
- **TokenizerManager**：在需要时训练或加载分词器，支持缓存复用与输出目录对齐。【F:src/training/pipeline/tokenizer_manager.py†L1-L118】
- **DatasetPreparer / DataResolver**：解析多重数据路径、按文件配置采样比例/最大样本数，并自动划分训练与验证集。【F:src/training/pipeline/data_manager.py†L24-L520】
- **TrainingLoopRunner**：实现梯度累积、学习率调度、验证评估、早停及 smoke test 生成的训练主循环。【F:src/training/pipeline/training_loop.py†L18-L214】
- **CheckpointManager**：统一管理检查点恢复、pretrain 权重回退与最终模型持久化。【F:src/training/pipeline/checkpointing.py†L11-L133】
- **TrainingMonitor**：记录训练/验证损失、PPL、梯度范数与系统资源指标，并写入 TensorBoard 与总结报告。【F:src/training/training_monitor.py†L120-L332】

这一组合让 `scripts/train.py` 成为薄封装，方便通过 CLI 切换预训练、SFT、DPO、RLHF 等模式，同时复用相同的工程骨架。【F:scripts/train.py†L1-L21】【F:src/training/pipeline/cli.py†L8-L117】

## 数据与标签构造

### 对话监督微调（SFT）

`ConversationDataset` 接受统一的对话消息列表或 `input`/`output` 结构，自动插入角色标记并仅对助手回复计算损失，支持轮次截断增强用于缓解长对话过拟合。【F:src/training/datasets/conversation.py†L10-L145】

关键特性：

- 系统/用户/助手标记可在配置中自定义，标签对齐时会为非助手位置填充 `pad_id`；
- 自动补齐 `bos/eos` 并根据 `max_length` 截断或填充；
- `conversation_augmentation` 支持按概率裁剪尾部若干轮 assistant 回复，提高指令覆盖度。【F:src/training/datasets/conversation.py†L94-L145】

### 语言建模与偏好数据

`LanguageModelingDataset` 则面向纯文本或偏好数据 `chosen` 字段，直接产出 `(input, target, loss_mask)`，其中 loss mask 会屏蔽 PAD，保持与 MiniMind 预训练脚本一致。对于 DPO 模式当前仍返回语言建模视角的数据，后续可在 `DatasetPreparer._create_dpo_dataset` 中扩展为双通道样本。【F:src/training/datasets/language_modeling.py†L11-L115】【F:src/training/pipeline/data_manager.py†L473-L520】

## 训练循环细节

- **梯度累积与调度器**：`TrainingLoopRunner` 根据 `gradient_accumulation_steps` 缩放 loss 并在指定步数更新参数，同时启用线性 warmup + 余弦退火学习率计划。【F:src/training/pipeline/training_loop.py†L56-L132】【F:src/training/pipeline/app.py†L110-L173】
- **显存监控与自动清理**：`MemoryHooks` 在训练开始与每步优化后检查显存/内存占用，超过阈值会触发缓存清理并输出资源摘要，可通过 `MINIGPT_MEMORY_*` 环境变量调整策略。【F:src/training/pipeline/memory_hooks.py†L1-L87】【F:src/training/pipeline/app.py†L25-L204】
- **混合精度与梯度检查点**：当配置开启 `mixed_precision` 或 `gradient_checkpointing` 时，训练器会自动启用 AMP 与模型梯度检查点，以降低显存压力。【F:src/training/pipeline/app.py†L137-L204】
- **验证与早停**：按照 `eval_steps` 触发验证，计算加权平均 loss / perplexity，并结合 `early_stopping_patience` 与 `early_stopping_delta` 决定是否提前终止训练。【F:src/training/pipeline/training_loop.py†L133-L200】
- **中断恢复**：支持 Ctrl+C 优雅中断；若开启 `--auto-resume` 会在启动时自动加载最新检查点或回退到预训练权重。【F:src/training/pipeline/app.py†L205-L239】【F:src/training/pipeline/checkpointing.py†L35-L133】

## 监控与日志

`TrainingMonitor` 默认以轻量模式运行，每隔 `log_interval` 步采集完整指标，同时持续向 TensorBoard 写入 loss、学习率、速度、显存使用等信息。训练完成后会保存实时曲线和摘要 JSON，便于快速回顾 run 的健康状况。`log_regression` 会在每次提示回归后记录通过率与样例，帮助追踪身份/事实类提示是否退化。【F:src/training/training_monitor.py†L120-L525】

若希望实时可视化，可在配置中打开 `enable_real_time_plots`，监控器会单独启动绘图线程生成仪表板。

## 扩展指引

1. **新增数据源**：在配置 `dataset_sampling` 中添加新文件名及采样策略，即可让 `DatasetPreparer` 自动合并；如需自定义解析逻辑，可扩展 `_extract_text`。【F:src/training/pipeline/data_manager.py†L68-L520】
2. **自定义训练逻辑**：可以编写新的 `TrainingLoopRunner` 子类或在 `MiniGPTTrainer.train` 中注入自定义回调（例如奖励模型评估），保持主循环解耦。
3. **完善 DPO/RLHF**：当前 DPO/RLHF 模式仍采用语言建模式损失，可在 `TrainingLoopRunner.run` 里根据 `mode` 分支实现对应的 pairwise 或强化学习更新，并扩展 `DatasetPreparer._create_dpo_dataset` 产出 `(chosen, rejected)` 样本。
4. **提示回归与告警**：`RegressionSuite` 会按配置间隔执行固定提示回归测试并产出报告，结合 `TrainingMonitor.log_regression` 能够快速接入自定义告警（如 Slack/Webhook）或进一步的质量分析。【F:src/training/pipeline/regression_suite.py†L1-L112】

通过上述模块化设计，可以在不打破现有训练脚本的前提下快速试验新的优化策略，同时保持实验的可复现性与可观测性。
