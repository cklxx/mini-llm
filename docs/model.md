# 🧠 模型与配置

Mini-LLM 的核心模型定义在 `src/model` 目录下，围绕 `MiniGPTConfig` 与 `MiniGPT` 两个类展开。本篇文档将概述配置项、主要模块以及可选特性。

## MiniGPTConfig
`MiniGPTConfig` 提供统一的超参数管理，默认值适合在桌面设备上快速实验。重要字段如下：

- **模型规模**：`vocab_size`、`hidden_size`、`num_hidden_layers`、`num_attention_heads`、`intermediate_size`
- **位置编码**：`use_rope` 控制是否启用 RoPE，`rope_theta` 决定旋转频率
- **归一化与激活**：`rms_norm_eps`、`hidden_act`（默认 `swiglu`）
- **注意力优化**：`use_gqa`/`num_key_value_heads` 控制 Grouped-Query Attention
- **MoE 支持**：`use_moe` 控制是否用稀疏专家替换 FFN，`n_routed_experts`/`n_shared_experts`/`num_experts_per_tok`/`aux_loss_alpha` 细化路由与负载均衡
- **训练辅助**：`dropout`、`attention_dropout`、`gradient_checkpointing`、`flash_attn`
- **生成参数**：`max_generate_length`、`temperature`、`top_k`、`top_p`

框架额外提供多种预设（`tiny`/`small`/`small_30m`/`medium`/`foundation`/`large`/`moe`），便于按资源快速选择合适的规模。其中：

- `small_30m`：13 层 × 384 hidden、GQA 4:1、≈30M 参数，默认上下文长度扩展到 2K，并开启 Flash Attention，适合单卡轻量级预训练或指令微调。
- `foundation`：24 层 × 768 hidden（约 2.1 亿参数），在 GQA (16Q→4KV)、Flash Attention 与梯度检查点的配合下兼顾训练吞吐与显存占用。

## MiniGPT 模型结构
`MiniGPT` 继承自 `torch.nn.Module`，实现了标准的 Transformer Decoder 结构：

1. **嵌入层**：`nn.Embedding` 将 token ID 映射为向量，可选择与输出层权重共享
2. **位置编码**：默认启用 RoPE，兼容传统正弦位置编码
3. **TransformerBlock 列表**：每个 block 根据配置组装注意力（含 GQA）、前馈（SwiGLU 或 MoE）与 RMSNorm 组件
4. **输出层**：最后经 RMSNorm 再映射回词表维度

### TransformerBlock 关键组件
- **注意力层**：默认使用 `OptimizedMultiHeadAttention`；当 `use_gqa=True` 时启用 `GroupedQueryAttention`
- **RoPE 支持**：`RotaryPositionEmbedding` 与 `apply_rotary_pos_emb` 在注意力计算中注入位置信息
- **SwiGLU 前馈网络**：`SwiGLUFeedForward` 替代传统 `GELU`
- **RMSNorm**：`RMSNorm` 替代 `LayerNorm`，减少均值计算成本
- **MoE（可选）**：当 `use_moe=True` 时，密集 FFN 会被 `SparseMoE`/`SharedExpertMoE` 取代，可通过配置切换共享专家、top-k 等参数

## 文本生成
`MiniGPT.generate` 简化封装了因果掩码、温度采样、Top-k/Top-p 的基本逻辑；若需更复杂的解码，可使用 `src/inference/generator.TextGenerator`。

## 与训练器的衔接
模型前向默认返回 `(batch_size, seq_len, vocab_size)` 的 logits，可直接送入 `PreTrainer`/`SFTTrainer` 的交叉熵损失计算。若在配置中启用 MoE，可通过 `model(input_ids, return_aux_loss=True)` 额外获得负载均衡辅助损失，用于与主损失加权合并。配置对象中定义的 `bos_token_id`/`eos_token_id`/`pad_token_id` 与分词器保持一致即可。

如需扩展模型（例如引入新的注意力实现或激活函数），建议：
1. 在 `MiniGPTConfig` 中增加可控开关
2. 在 `TransformerBlock` 中根据配置进行分支选择
3. 补充训练与推理文档，保持文档-代码一致
