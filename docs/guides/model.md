# 🧠 模型与配置

Mini-LLM 的核心模型定义在 `src/model` 目录下，围绕 `MiniGPTConfig` 与 `MiniGPT` 两个类展开。本篇文档将概述配置项、主要模块以及可选特性，并结合源码说明每个设计选择的作用与限制。

## MiniGPTConfig

`MiniGPTConfig` 提供统一的超参数管理，默认值适合在桌面设备上快速实验。构造函数会把模型规模、注意力优化、MoE、生成等设置全部归档并在 `__init__` 结尾执行校验，确保如 `hidden_size` 与 `num_attention_heads` 的整除关系以及 GQA/MoE 的配置合法。【F:src/model/config.py†L7-L145】

常见字段分组如下：

- **模型规模**：`vocab_size`、`hidden_size`、`num_hidden_layers`、`num_attention_heads`、`intermediate_size` 直接决定参数量；若未显式给定 `intermediate_size`，会自动取 `hidden_size * 4` 以匹配 Transformer 经典比例。【F:src/model/config.py†L18-L87】
- **位置编码**：`use_rope`/`rope_theta` 控制是否启用旋转位置编码（RoPE）；禁用时模型会回退到传统正弦编码。【F:src/model/config.py†L27-L75】
- **归一化与激活**：`rms_norm_eps`、`hidden_act` 分别决定 RMSNorm 的数值稳定性与前馈网络激活函数（默认 `swiglu`）。【F:src/model/config.py†L24-L72】
- **注意力优化**：`use_gqa`/`num_key_value_heads` 支持 Grouped-Query Attention；当未指定 KV 头数时默认按 4:1 计算。【F:src/model/config.py†L33-L139】
- **MoE 扩展**：`use_moe`、`n_routed_experts`、`n_shared_experts`、`num_experts_per_tok`、`aux_loss_alpha` 配置稀疏专家数量、共享专家和负载均衡损失权重；若参数不合法会在 `_validate_config` 中抛出断言。【F:src/model/config.py†L45-L134】
- **推理默认值**：`max_generate_length`、`temperature`、`top_k`、`top_p` 会在 `MiniGPT.generate` 中作为默认参数使用。【F:src/model/config.py†L55-L115】【F:src/model/transformer.py†L451-L500】

项目提供的 `get_tiny`/`small`/`small_30m`/`medium`/`foundation` 等预设函数会复用以上字段组合出不同规模；例如 `get_small_config` 采用更深更窄的架构并开启 GQA，从而在保持参数量的同时降低显存峰值。【F:src/model/config.py†L159-L254】

> 需要按需修改配置时，可先调用 `get_config("medium")` 获得默认配置，再覆盖特定字段（如 `rope_theta` 或 `num_key_value_heads`），最后传入 `create_model` 复用训练脚本的自动打印与参数统计逻辑。【F:src/model/config.py†L257-L320】【F:src/model/transformer.py†L507-L547】

## MiniGPT 架构

`MiniGPT` 继承自 `torch.nn.Module`，按照典型的 Decoder-only 架构堆叠模块，并在初始化时根据配置选择 RoPE/传统位置编码以及 MoE 或密集前馈结构。【F:src/model/transformer.py†L314-L356】

核心组件：

1. **嵌入与位置编码**：`token_embedding` 将 token ID 映射到隐空间；若 `use_rope=False`，会启用 `PositionalEncoding` 注入正弦位置编码，否则由注意力模块内的 RoPE 完成位置注入。【F:src/model/transformer.py†L328-L408】
2. **TransformerBlock 列表**：构造时根据 `use_gqa` 决定是否实例化 `GroupedQueryAttention`，并在启用 MoE 时创建 `SparseMoE` 或 `SharedExpertMoE`，否则回退到 `SwiGLUFeedForward`。【F:src/model/transformer.py†L197-L307】
3. **归一化与输出层**：堆叠结束后经过 RMSNorm 并投影回词表维度；当配置 `tie_word_embeddings=True` 时会复用嵌入权重，节省参数量。【F:src/model/transformer.py†L338-L443】
4. **权重初始化**：`init_weights` 针对线性层、嵌入、RMSNorm 采用正态/常数初始化，保证训练初期稳定。【F:src/model/transformer.py†L354-L370】

### 配置派生与参数估算

- `create_model` 支持同时传入 `model_size` 与显式 `MiniGPTConfig`；当二者同时提供时，函数会优先使用传入的配置并在必要时同步词表大小，保持训练/推理一致性。【F:src/model/transformer.py†L507-L547】
- `MiniGPT.get_num_params()` 可用于快速估算不同配置的参数规模，训练脚本会在构建模型后打印总参数量与可训练参数量，帮助评估显存开销。【F:src/model/transformer.py†L501-L536】【F:src/training/pipeline/pipeline.py†L125-L170】
- 结合 `TrainingPipeline` 输出的配置快照，可以在 Notebook 中绘制“层数-参数量”曲线，用于教学展示模型缩放与资源消耗的关系。【F:src/training/pipeline/pipeline.py†L41-L79】

### TransformerBlock 关键路径

- **注意力**：`GroupedQueryAttention` 支持 GQA、RoPE 与 Flash Attention，默认优先使用；否则回退到兼容的 `MultiHeadAttention` 实现。【F:src/model/transformer.py†L202-L218】
- **MoE 与负载均衡**：当启用稀疏专家时，会根据配置构造共享或独立专家并在 `forward` 中返回辅助损失，方便在训练时加权。【F:src/model/transformer.py†L220-L305】
- **Residual + PreNorm**：块内部遵循 Pre-Norm 设计，先做 RMSNorm 再进入注意力或前馈，然后通过残差连接回主干，提升训练稳定性。【F:src/model/transformer.py†L262-L311】

## 前向传播与辅助损失

`MiniGPT.forward` 会自动创建因果掩码、生成 RoPE 所需的 `position_ids` 并遍历每个 TransformerBlock；若 `return_aux_loss=True` 且启用了 MoE，会额外返回累加后的负载均衡损失，供训练循环与主损失合并。【F:src/model/transformer.py†L379-L449】

## 文本生成

`MiniGPT.generate` 封装了最基本的自回归生成：循环前向推理、按温度缩放 logits、执行 Top-k 采样并在遇到 `eos_token_id` 时停止，适合作为快速调试入口；生产环境推荐使用更灵活的 `TextGenerator`，以便开启 Top-p、重复惩罚等策略。【F:src/model/transformer.py†L451-L500】【F:src/inference/generator.py†L125-L225】

## 与训练器的衔接

- 训练循环默认接收 `(batch_size, seq_len, vocab_size)` 的 logits，可直接用于交叉熵损失；需要 MoE 辅助损失时，训练器应将 `return_aux_loss=True` 并将返回的第二项按 `aux_loss_alpha` 加权后加到主损失中。【F:src/model/transformer.py†L437-L449】【F:src/training/pipeline/pipeline.py†L170-L213】
- 配置中的 `bos_token_id`/`eos_token_id`/`pad_token_id` 会被 `ConversationDataset` 和训练脚本读取，保持与分词器一致才能保证标签掩码正确。【F:src/model/config.py†L38-L115】【F:src/training/datasets/conversation.py†L53-L110】
- 当需要冻结部分层进行 LoRA/奖励模型微调时，可使用 `model.named_parameters()` 与配置中的层数信息配合筛选，例如冻结前 `n_layers-2` 层仅更新顶部注意力头；训练脚本会在 `create_model` 后输出层数信息以辅助筛选。【F:src/model/transformer.py†L197-L311】【F:src/model/transformer.py†L507-L547】

如需扩展模型（例如引入新的注意力实现或激活函数），建议：

1. 在 `MiniGPTConfig` 中增加可控开关并在 `_validate_config` 中补充断言。【F:src/model/config.py†L120-L139】
2. 在 `TransformerBlock` 中根据新开关选择实现，确保前向路径和辅助损失都被正确处理。【F:src/model/transformer.py†L197-L311】
3. 补充训练/推理文档或示例，保持代码与文档一致，减少团队间沟通成本。
