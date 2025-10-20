# Mini-LLM 与 MiniMind 架构与训练方案对比

本报告聚焦 Mini-LLM 与 MiniMind 两个开源小型大语言模型项目，从模型架构、超参数设计以及训练流程三个维度总结主要异同，帮助读者快速把握两者的工程侧重点。

## 概览

- **共同点**：两者都围绕现代 Transformer 解码器搭建，提供 RoPE 旋转位置编码、Grouped-Query Attention（GQA）和可选的 Mixture-of-Experts（MoE）前馈层，同时支持 Flash Attention 等显存优化手段。【F:docs/guides/model.md†L1-L49】【F:src/model/config.py†L17-L88】【F:src/model/config.py†L214-L264】（MiniMind 参见其文档与源码：`docs/01_MiniMindConfig.md`、`docs/07_ModelArchitecture.md`、`model/model_minimind.py`）
- **差异概览**：Mini-LLM 更强调可插拔的配置族与训练流水线抽象，便于在不同资源环境间切换；MiniMind 则沿用 HuggingFace `PretrainedConfig`/`PreTrainedModel` 框架，聚焦于极简脚本化训练与多阶段任务覆盖（预训练、SFT、LoRA、DPO、蒸馏）。

## 模型架构对比

| 方面 | Mini-LLM | MiniMind |
| --- | --- | --- |
| 基础模块 | `MiniGPT` 由嵌入层、RoPE 位置编码、TransformerBlock 列表、末端 RMSNorm 与共享词表投影组成；可在 block 级别选择 GQA、SwiGLU 或 MoE。【F:docs/guides/model.md†L11-L49】 | `MiniMindModel` 使用相同的嵌入 → TransformerBlock → RMSNorm → 线性输出结构，默认共享嵌入权重；源码直接继承 `PreTrainedModel`，并在 `MiniMindForCausalLM` 中封装推理接口（来源：MiniMind `docs/07_ModelArchitecture.md`、`model/model_minimind.py`）。 |
| 注意力机制 | 默认启用 RoPE，`use_gqa` 控制 KV 头压缩；可通过配置切换 Flash Attention 与梯度检查点。【F:src/model/config.py†L29-L65】【F:docs/guides/model.md†L27-L41】 | 同样提供 RoPE/GQA/Flash Attention；注意力模块在 `Attention` 中手动实现 KV repeat 与可选 Flash 路径（来源：MiniMind `model/model_minimind.py`）。 |
| 前馈层 | 默认使用 SwiGLU，可按需替换为 MoE；MoE 支持路由/共享专家及辅助损失权重调节。【F:docs/guides/model.md†L31-L41】【F:src/model/config.py†L66-L88】 | 支持密集层与 MoE，MoE 参数与 Mini-LLM 类似但整体集成在 HuggingFace 风格模型中（来源：MiniMind `docs/01_MiniMindConfig.md`、`model/model_minimind.py`）。 |
| 位置长度 | 预设配置最高支持 4096（`foundation`/`large`）或 2048（`small_30m`/`medium`），聚焦教学与桌面实验规模。【F:src/model/config.py†L214-L264】 | 默认 `max_position_embeddings=32768` 以兼容长上下文实验，同时在推理脚本中对 8K 输出窗口做限制（来源：MiniMind `docs/01_MiniMindConfig.md`、`eval_model.py` 第 103-123 行注释）。 |

**总结**：Mini-LLM 将多种现代组件包装为配置开关，便于课堂/实验扩展；MiniMind 更贴近 HuggingFace 生态，强调对高上下文长度和 MoE 的直接支持。

## 超参数与模型规模

### 默认配置

- **Mini-LLM**：`MiniGPTConfig` 默认 `hidden_size=512`、`num_hidden_layers=6`、`num_attention_heads=8`，并默认开启 `use_gqa=True`、`dropout=0.1`、`attention_dropout=0.1`，同时在初始化时若启用 GQA 自动设置 `num_key_value_heads = num_attention_heads // 4`。【F:src/model/config.py†L17-L65】
- **MiniMind**：`MiniMindConfig` 默认 `hidden_size=512`、`num_hidden_layers=8`、`num_attention_heads=8`，但 `dropout=0.0`、`num_key_value_heads=2`（固定 4:1 GQA 比例）、`max_position_embeddings=32768`、`flash_attn=True`，并沿用 `rope_theta=1e6` 以覆盖长上下文（来源：MiniMind `model/model_minimind.py`）。

### 预设/变体

- **Mini-LLM 预设族**：提供 `tiny`（128×8 层）、`small`（384×12 层）、`small_30m`（384×13 层、上下文 2K）、`medium`（384×20 层，瘦长架构 + Flash Attention）、`foundation`（768×24 层，梯度检查点）、`large`（768×32 层）、`moe`（384×12 层 MoE）。这些预设覆盖 1M 至 350M 参数级别，方便按算力选择。【F:src/model/config.py†L170-L252】
- **MiniMind 系列**：仓库注释列出 `MiniMind2-Small (26M)`（512×8）、`MiniMind2 (104M)`（768×16）、`MiniMind2-MoE (145M)`（640×8 且 `use_moe=True`）等规模，并在推理/评测脚本中通过命令行切换隐藏维度、层数和 MoE 开关（来源：MiniMind `eval_model.py` 第 103-123 行注释）。

**超参数差异要点**：Mini-LLM 通过代码层面的预设函数管理多模型族，强调教学中快速切换；MiniMind 借助脚本参数和 HuggingFace 配置默认值覆盖多个公开模型权重，并将上下文长度与 Flash Attention 默认值调高以支持长序列研究。

## 训练方案

### Mini-LLM 训练流水线

Mini-LLM 的 `training.pipeline` 架构将训练划分为环境配置、数据准备、训练循环、检查点与监控等模块：

- `TrainingEnvironment` 管理随机种子、设备检测、输出目录与配置快照。【F:docs/guides/training.md†L8-L40】
- `TokenizerManager` 负责分词器训练/缓存；`DatasetPreparer` 则按文件采样比例、最大样本数与验证拆分生成数据集。【F:docs/guides/training.md†L40-L74】
- `TrainingLoopRunner` 实现梯度累积、线性 warmup + 余弦退火调度、验证评估与早停；同时结合 `MemoryHooks` 进行显存监控。【F:docs/guides/training.md†L74-L120】
- `TrainingMonitor` 汇总损失、PPL、梯度范数、资源占用并输出 TensorBoard 与 JSON 摘要，支持提示回归评估。【F:docs/guides/training.md†L120-L154】

整体流程通过 `scripts/train.py` 与 CLI 参数切换预训练、SFT、DPO、RLHF 等模式，强调可复现性和监控覆盖。近期 medium 训练预设同步引入瘦长超参（384×20 层）与更激进的 `lr=5e-4`、`warmup_steps=10000`，向 MiniMind 默认调度对齐以提升小模型收敛速度。【F:docs/guides/training.md†L1-L74】【F:config/training_config.py†L197-L243】【F:src/model/config.py†L227-L252】

### MiniMind 训练脚本

MiniMind 的预训练脚本采用更直接的单文件实现：

- `trainer/train_pretrain.py` 手动创建 `MiniMindForCausalLM`、`PretrainDataset` 和 `DataLoader`，可选 DDP；默认批量 32、梯度累积 8、学习率 5e-4、Epoch=1，并使用 `GradScaler` 结合 `bfloat16/float16` 混合精度（来源：MiniMind `trainer/train_pretrain.py` 第 1-160 行）。
- 学习率通过 `get_lr` 实现余弦调度并叠加 10% 基线；每步将交叉熵乘以掩码再平均，并加上 MoE 辅助损失；按 `accumulation_steps` 执行梯度缩放、裁剪与优化器更新，同时定期保存半精度权重（来源：MiniMind `trainer/train_pretrain.py` 第 20-118 行）。
- 通过命令行参数调节隐藏维度、层数、是否启用 MoE 及数据路径，辅以 `wandb` 可选记录（来源：MiniMind `trainer/train_pretrain.py` 第 119-204 行）。

其他阶段（SFT、LoRA、DPO、蒸馏）复用类似脚本化范式，体现“原生 PyTorch + HuggingFace” 的轻量工程思路。

### 对比总结

- **工程抽象**：Mini-LLM 以面向对象管线封装，适合课程演示与多阶段实验；MiniMind 倾向可读性强的脚本，便于初学者直接修改。
- **调度与监控**：Mini-LLM 集成早停、回归测试、内存警戒线等监控机制；MiniMind 聚焦于核心训练环节，监控以日志与可选 wandb 为主。
- **自动化程度**：Mini-LLM 的配置文件提供数据采样、内存优化等默认策略，MiniMind 更多依赖命令行/脚本参数手动指定。

## 结论

Mini-LLM 在架构上突出“模块化 + 可配置”，通过多套预设与训练流水线帮助学习者理解大型模型训练的完整闭环。MiniMind 则以 HuggingFace 兼容的极简实现为核心，提供一套覆盖预训练到强化学习的脚本化流程和更长上下文的默认配置。根据学习或实验目标的不同，可选择更注重系统抽象（Mini-LLM）或更贴近生产模型格式与脚本化实践（MiniMind）的方案。【F:docs/guides/model.md†L1-L49】【F:docs/guides/training.md†L1-L154】【F:src/model/config.py†L17-L264】
