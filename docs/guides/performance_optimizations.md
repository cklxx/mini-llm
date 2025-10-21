# MiniMind 性能优化实践总结

为了对齐 [MiniMind](https://github.com/jingyaogong/minimind) 项目在训练与推理阶段的高效表现，本项目吸收了其关键优化策略。以下内容梳理了已经整合到 `mini-llm` 中的改动及其使用建议，帮助你在 GPU 资源有限的环境下仍然获得稳定的吞吐率。

## 数据加载与填充策略

- **分布式采样与 pinned memory**：`DatasetPreparer` 会在检测到 `torch.distributed` 已初始化时启用 `DistributedSampler`，并自动关闭 DataLoader 的重复 `shuffle`。同时默认启用 pinned memory、持久化 worker、prefetch 等策略，确保预处理与 GPU 计算高度重叠。【F:src/training/pipeline/data_manager.py†L11-L21】【F:src/training/pipeline/data_manager.py†L333-L378】
- **静态长度对齐**：预训练与 SFT 数据集在进入模型前都会统一到固定的 `max_length`，避免动态 shape 触发额外的 kernel 启动与缓存抖动。这与 MiniMind 的 `padding='max_length'` 做法一致，确保 GPU 始终处理规则批次。【F:src/training/datasets/language_modeling.py†L30-L84】【F:src/training/datasets/conversation.py†L19-L124】
- **高吞吐缓存**：沿用 MiniMind 的做法，在 BaseConfig 中自动推导 `prefetch_factor` 与 `num_workers`，并允许通过环境变量覆盖。配合新的采样器逻辑，在多机/多卡上也能保持数据管线稳定。【F:config/training_config.py†L372-L432】
- **DPO/RLHF 清单与并行筛选**：数据解析器现在同样支持 DPO 与 RLHF 模式的 manifest，并在高性能路径中并行过滤 `(chosen, rejected)` 偏好样本，缓存到磁盘后即可复用，避免每次重启都要重新解码 JSON。【F:src/training/pipeline/data_manager.py†L53-L154】【F:src/data/high_performance_loader.py†L247-L328】

## 偏好优化与 RLHF

- **专用采样控制**：基础配置新增对 `dpo.jsonl` 的采样比例、上限与验证集开关，默认关闭验证分片，确保全部样本投入偏好学习。【F:config/training_config.py†L228-L247】
- **统一数据入口**：DPO/RLHF 训练可通过 manifest 声明数据文件，既兼容仓库内置语料，也允许在云端挂载路径后自动发现并缓存处理结果。【F:src/training/pipeline/data_manager.py†L38-L123】

## 推理 & 评测优化

- **统一的推理配置**：BaseConfig 新增 `MINIGPT_INFERENCE_*` 系列环境变量，可设置温度、Top-k/Top-p、重复惩罚、历史轮数与自动 `autocast` 精度。默认值即与 MiniMind 推理脚本保持一致，可直接用于训练日志内置的回归与行业评测流程。【F:config/training_config.py†L344-L365】【F:config/training_config.py†L403-L432】
- **推理助手重写**：`TextGenerator` 现在在推理过程中使用 `torch.inference_mode()` 与可选的 CUDA autocast，并支持 `tokenizer.apply_chat_template` 与历史上下文裁剪，等同于 MiniMind 的 ChatML 模板构造方式。生成后的文本会回写给回归测试与基准评估，避免重复实现。【F:src/inference/generator.py†L1-L196】
- **回归评估对齐推理路径**：回归套件使用新的 `TextGenerator`，统一温度、Top-k/Top-p、重复惩罚与 chat template 控制，结果更贴近真实推理表现，也避免手写循环导致的 GPU 同步开销。【F:src/training/pipeline/regression_suite.py†L6-L153】
- **基准评测开启 autocast**：行业评测在语言模型与多项选择任务中使用 `torch.inference_mode()` 与条件式 autocast，结合 non-blocking 张量迁移，复用了 MiniMind 评测脚本的高效路径，使推理端的优化在 benchmark 数值上可观测。【F:src/evaluation/benchmark_suite.py†L5-L726】

## 实用配置建议

1. **快速复现 MiniMind 推理体验**：设置环境变量

   ```bash
   export MINIGPT_INFERENCE_TEMPERATURE=0.85
   export MINIGPT_INFERENCE_TOP_P=0.85
   export MINIGPT_INFERENCE_HISTORY_TURNS=4
   export MINIGPT_INFERENCE_DTYPE=bf16
   ```

   在单卡 RTX 4090 上即可重现 MiniMind 推理时的流畅度，并自动在回归测试与行业评测中生效。

2. **分布式训练注意事项**：使用 `torchrun` 启动时无需额外改动，DataLoader 会检测 `torch.distributed` 状态并启用 `DistributedSampler`。若需自定义 worker 数，请通过 `MINIGPT_DATA_MAX_WORKERS` 覆盖。

3. **保持批次稳定性**：确保原始语料长度与配置的 `max_seq_len` 合理匹配。若存在大量超长样本，可在数据清洗阶段提前裁剪，避免在训练时出现过多被 `drop_last` 的批次。

通过上述改动，`mini-llm` 的训练与推理流程已尽可能贴近 MiniMind 的高效实现，同时保留了原有的可配置性与诊断能力。

