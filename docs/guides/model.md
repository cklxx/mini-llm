# 🧠 模型与配置

Mini-LLM 的核心模型定义在 `src/model` 目录下，围绕 `MiniGPTConfig` 与 `MiniGPT` 两个类展开。本篇文档将概述配置项、主要模块以及可选特性，并结合源码说明每个设计选择的作用与限制。

## MiniGPTConfig

`MiniGPTConfig` 提供统一的超参数管理，默认值现已与 MiniMind2-Small（26M）完全对齐：`vocab_size=6400`、`hidden_size=512`、`num_hidden_layers=8`、`num_attention_heads=8`、`num_key_value_heads=2`，并保持 `dropout=0.0` / `attention_dropout=0.0`、`flash_attn=True`、`rms_norm_eps=1e-5` 与 `hidden_act="silu"`（门控前馈使用 SiLU×线性 形式）。【F:src/model/config.py†L17-L145】构造函数会把模型规模、注意力优化、MoE、生成等设置全部归档并在 `__init__` 结尾执行校验，确保如 `hidden_size` 与 `num_attention_heads` 的整除关系以及 GQA/MoE 的配置合法。

常见字段分组如下：

- **模型规模**：`vocab_size`、`hidden_size`、`num_hidden_layers`、`num_attention_heads`、`intermediate_size` 直接决定参数量；若未显式给定 `intermediate_size`，会按照 MiniMind 的 `hidden_size * 8 / 3` 取整到 64 的策略推导前馈维度（例如 `hidden_size=512` 时得到 `intermediate_size=1408`）。【F:src/model/config.py†L18-L87】
- **位置编码**：`use_rope`/`rope_theta` 控制是否启用旋转位置编码（RoPE）；禁用时模型会回退到传统正弦编码。【F:src/model/config.py†L27-L75】
- **归一化与激活**：`rms_norm_eps`、`hidden_act` 分别决定 RMSNorm 的数值稳定性与前馈网络激活函数（默认 `silu`，与 MiniMind 统一）。【F:src/model/config.py†L24-L72】
- **注意力优化**：`use_gqa`/`num_key_value_heads` 支持 Grouped-Query Attention；默认固定 `num_key_value_heads=2`（4:1 比例），与 MiniMind 行为一致。【F:src/model/config.py†L33-L139】
- **MoE 扩展**：`use_moe`、`n_routed_experts`、`n_shared_experts`、`num_experts_per_tok`、`aux_loss_alpha` 配置稀疏专家数量、共享专家和负载均衡损失权重；若参数不合法会在 `_validate_config` 中抛出断言。【F:src/model/config.py†L45-L134】
- **推理默认值**：`max_generate_length`、`temperature`、`top_k`、`top_p` 会在 `MiniGPT.generate` 中作为默认参数使用。【F:src/model/config.py†L55-L115】【F:src/model/transformer.py†L312-L354】

项目提供的 `get_minimind_small_config` / `get_minimind_base_config` / `get_minimind_moe_config` 三个核心函数分别对应 MiniMind2-Small（26M 稠密）、MiniMind2（104M 稠密）与 MiniMind2-MoE（145M 稀疏）三套参数；旧的 `get_tiny`/`small`/`medium` 等名称作为别名返回相同结果，方便沿用既有脚本。【F:src/model/config.py†L146-L259】

> 需要按需修改配置时，可先调用 `get_config("minimind_small")` 获得默认配置，再覆盖特定字段（如 `use_moe` 或 `rope_scaling`），最后传入 `create_model` 复用训练脚本的自动打印与参数统计逻辑。【F:src/model/config.py†L314-L360】【F:src/model/transformer.py†L360-L381】

## MiniGPT 架构

`MiniGPT` 继承自 `torch.nn.Module`，按照典型的 Decoder-only 架构堆叠模块，并在初始化时根据配置选择 RoPE/传统位置编码以及 MoE 或密集前馈结构。【F:src/model/transformer.py†L267-L310】

核心组件：

1. **嵌入与位置编码**：`MiniMindModel` 在初始化时创建词嵌入并预计算 RoPE 的余弦/正弦缓存，前向传播时按当前序列长度切片后交给注意力层使用。【F:src/model/transformer.py†L203-L264】
2. **MiniMindBlock 列表**：每层采用 Pre-Norm → `MiniMindAttention`（支持 GQA、RoPE 与 Flash Attention）→ 残差的路径，然后再次 Pre-Norm 并接 `MiniMindFeedForward` 或基于 `build_moelayer_from_config` 的 MiniMind 式 MoE。【F:src/model/transformer.py†L154-L200】【F:src/model/moe.py†L35-L208】
3. **归一化与输出层**：堆叠结束后通过 RMSNorm，再由 `MiniGPT` 复用嵌入矩阵或独立 `lm_head` 映射回词表空间，默认启用权重共享以匹配 MiniMind。【F:src/model/transformer.py†L259-L310】

### 配置派生与参数估算

- `create_model` 支持同时传入 `model_size` 与显式 `MiniGPTConfig`；当二者同时提供时，函数会优先使用传入的配置并在必要时同步词表大小，保持训练/推理一致性。【F:src/model/transformer.py†L360-L381】
- `MiniGPT.get_num_params()` 可用于快速估算不同配置的参数规模，训练脚本会在构建模型后打印总参数量与可训练参数量，帮助评估显存开销。【F:src/model/transformer.py†L356-L381】【F:src/training/pipeline/pipeline.py†L125-L170】
- 结合 `TrainingPipeline` 输出的配置快照，可以在 Notebook 中绘制“层数-参数量”曲线，用于教学展示模型缩放与资源消耗的关系。【F:src/training/pipeline/pipeline.py†L41-L79】

### TransformerBlock 关键路径

- **注意力**：`MiniMindAttention` 复刻 MiniMind 的 GQA/Flash 路径，能在有缓存时复用 KV 并在 Flash 可用时调用 `scaled_dot_product_attention`，否则回退到遮罩 Softmax。【F:src/model/transformer.py†L71-L151】
- **MoE 与密集前馈**：`build_moelayer_from_config` 会在 `use_moe=True` 时选择 `MiniMindMOEFeedForward`，实现 MiniMind 相同的 top-k 门控、共享专家与辅助损失累积；未启用时回退到门控前馈层。【F:src/model/transformer.py†L154-L200】【F:src/model/moe.py†L117-L208】
- **Pre-Norm 残差**：每层都在注意力与前馈前应用 RMSNorm，并在输出处做残差相加，保持训练稳定性。【F:src/model/transformer.py†L154-L200】

## 前向传播与辅助损失

`MiniGPT.forward` 会委托 `MiniMindModel` 处理可选的注意力掩码与 KV 缓存，拿到隐藏状态、缓存列表和累加的 MoE 辅助损失；随后根据是否共享嵌入选择线性映射，并在 `return_aux_loss=True` 且启用 MoE 时将辅助损失追加到输出元组。【F:src/model/transformer.py†L267-L310】

## 文本生成

`MiniGPT.generate` 按 MiniMind 的推理流程循环调用模型（可复用 KV 缓存），对最后一个 token 的 logits 进行温度缩放与 Top-k 截断后采样，并在生成到 `eos_token_id` 时提前停止，适合作为快速调试入口；生产环境仍推荐使用更完整的 `TextGenerator` 封装。【F:src/model/transformer.py†L312-L354】【F:src/inference/generator.py†L125-L225】

## 与训练器的衔接

- 训练循环默认接收 `(batch_size, seq_len, vocab_size)` 的 logits，可直接用于交叉熵损失；需要 MoE 辅助损失时，训练器应将 `return_aux_loss=True` 并将返回的第二项按 `aux_loss_alpha` 加权后加到主损失中。【F:src/model/transformer.py†L267-L310】【F:src/training/pipeline/pipeline.py†L170-L213】
- 配置中的 `bos_token_id`/`eos_token_id`/`pad_token_id` 会被 `ConversationDataset` 和训练脚本读取，保持与分词器一致才能保证标签掩码正确。【F:src/model/config.py†L38-L115】【F:src/training/datasets/conversation.py†L53-L110】
- 当需要冻结部分层进行 LoRA/奖励模型微调时，可使用 `model.named_parameters()` 与配置中的层数信息配合筛选，例如冻结前 `n_layers-2` 层仅更新顶部注意力头；训练脚本会在 `create_model` 后输出层数信息以辅助筛选。【F:src/model/transformer.py†L154-L200】【F:src/model/transformer.py†L360-L381】

如需扩展模型（例如引入新的注意力实现或激活函数），建议：

1. 在 `MiniGPTConfig` 中增加可控开关并在 `_validate_config` 中补充断言。【F:src/model/config.py†L120-L139】
2. 在 `TransformerBlock` 中根据新开关选择实现，确保前向路径和辅助损失都被正确处理。【F:src/model/transformer.py†L154-L200】
3. 补充训练/推理文档或示例，保持代码与文档一致，减少团队间沟通成本。
