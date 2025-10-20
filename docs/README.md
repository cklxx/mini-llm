# 📚 Mini-LLM 文档索引

本目录收录了与 Mini-LLM 相关的所有参考文档，按照“从入门到进阶”的顺序组织，帮助你快速定位所需信息。

## 导航

### 入门与操作指南

- [guides/getting_started.md](guides/getting_started.md)：环境要求、安装方式以及最小训练/推理示例
- [guides/dataset_preparation.md](guides/dataset_preparation.md)：原始语料清洗、采样与 JSONL 规范
- [guides/model.md](guides/model.md)：模型配置、Transformer 组件、可选特性（GQA、RoPE、MoE 等）
- [guides/data.md](guides/data.md)：数据格式约定、SFT/预训练/DPO 加载器与分词器训练流程
- [guides/training.md](guides/training.md)：训练器、内存优化、混合精度与梯度累积
- [guides/inference.md](guides/inference.md)：`TextGenerator` 推理入口与生成策略配置
- [guides/rlhf.md](guides/rlhf.md)：RLHF 管道（SFT → 奖励模型 → PPO）的结构与定制说明

### 研究分析与对比

- [research/foundation_analysis.md](research/foundation_analysis.md)：基础模型设计与实验考量
- [research/data_analysis.md](research/data_analysis.md)：数据源画像、分布统计与质量评估
- [research/code_model_training_research.md](research/code_model_training_research.md)：代码/模型训练的技术调研
- [research/minimind_comparison.md](research/minimind_comparison.md)：与 MiniMind 框架的架构与流程对比
- [research/nanochat_research.md](research/nanochat_research.md)：NanoChat 方案评估与经验总结

### 规划路线

- [planning/training_optimization_plan.md](planning/training_optimization_plan.md)：训练优化路线图与优先级
- [planning/training_remaining_work.md](planning/training_remaining_work.md)：待办事项清单与状态跟踪

### 案例实践

- [case_studies/qwen_identity_finetune.md](case_studies/qwen_identity_finetune.md)：Qwen 指令对齐与身份微调案例
- [MiniGPT训练深度解析小册/](MiniGPT训练深度解析小册/README.md)：以源码为索引的专题手册，覆盖数学基础、架构、数据、训练、对齐

## 使用建议

1. **第一次接触项目**：建议按照 `guides/getting_started.md → guides/model.md → guides/data.md` 的顺序阅读。
2. **准备训练**：重点关注 `guides/training.md` 与 `guides/data.md`，确保数据格式和训练循环配置正确。
3. **需要生成或上线**：查阅 `guides/inference.md`，了解不同采样策略对结果的影响。
4. **扩展 RLHF**：阅读 `guides/rlhf.md` 了解管道各阶段与可自定义的接口。

所有文档均以源码为依据，示例代码可直接在仓库环境中运行。如发现与代码不一致的地方，欢迎提交 Issue 或 PR 帮助我们持续改进。
