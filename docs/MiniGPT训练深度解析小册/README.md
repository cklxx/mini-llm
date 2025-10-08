# MiniGPT 训练深度解析小册

> 以代码为准绳，帮助你在最短的路径上理解 mini-llm 的核心实现。

本小册将仓库中的训练流程拆解为五个维度：数学基础、Transformer 架构、预训练与数据、训练与优化、生成与对齐。每一章都指向对应的源码模块，方便你一边阅读文档一边在代码里定位实现。

## 使用方式
1. **先读总览，再进源码。** 每章的第一节都会给出关键文件列表，建议配合 IDE 的代码跳转一起查看。
2. **关注“实践提示”。** 我们结合 `src/training`、`src/model` 等目录中的工具，给出操作化的建议，帮助你快速落地实验。
3. **适合的读者。** 如果你已经熟悉 Transformer，但想了解 mini-llm 在工程和训练细节上的具体落地，这份小册将作为实战向的参考资料。

## 章节导航
- [第 01 章：数学基础与理论框架](第01章-数学基础与理论框架/README.md)
- [第 02 章：Transformer 核心架构](第02章-Transformer核心架构/README.md)
- [第 03 章：预训练与数据流程](第03章-预训练与数据/README.md)
- [第 04 章：训练与优化策略](第04章-训练与优化/README.md)
- [第 05 章：生成与对齐实践](第05章-生成与对齐/README.md)

## 仓库总览速查
| 模块 | 说明 | 关键文件 |
| --- | --- | --- |
| 模型构建 | Transformer 层、注意力、并行优化 | `src/model/transformer.py`, `src/model/gqa.py`, `src/model/moe.py` |
| 数据处理 | 数据集抽象、tokenizer 适配 | `src/data/dataset_loader.py`, `src/tokenizer/tokenizer_manager.py` |
| 训练流程 | 训练循环、优化器、监控 | `src/training/trainer.py`, `src/training/memory_optimizer.py`, `src/training/training_monitor.py` |
| 推理与服务 | 生成策略、聊天接口 | `src/inference/generator.py` |
| 对齐 (RLHF) | 奖励模型、策略优化 | `src/rl/reward_model/reward_trainer.py`, `src/rl/ppo/ppo_trainer.py` |

> **温馨提示**：所有章节都默认你已经读过顶层的 [docs/getting_started.md](../getting_started.md) 和 [docs/training.md](../training.md)。如果你还没配置好环境，请先完成基础教程。
