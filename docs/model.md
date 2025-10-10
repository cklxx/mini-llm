# 🧠 模型与配置

Mini-LLM 的核心模型定义在 `src/model` 目录下，围绕 `MiniGPTConfig` 与 `MiniGPT` 两个类展开。本篇文档将概述配置项、主要模块以及可选特性。

## MiniGPTConfig
`MiniGPTConfig` 提供统一的超参数管理，默认值适合在桌面设备上快速实验。重要字段如下：

- **模型规模**：`vocab_size`、`hidden_size`、`num_hidden_layers`、`num_attention_heads`、`intermediate_size`
- **位置编码**：`use_rope` 控制是否启用 RoPE，`rope_theta` 决定旋转频率
- **归一化与激活**：`rms_norm_eps`、`hidden_act`（默认 `swiglu`）
- **注意力优化**：`use_gqa`/`num_key_value_heads` 控制 Grouped-Query Attention
- **MoE 支持**：`use_moe` 及相关专家数量、辅助损失配置
- **训练辅助**：`dropout`、`attention_dropout`、`gradient_checkpointing`、`flash_attn`
- **生成参数**：`max_generate_length`、`temperature`、`top_k`、`top_p`

框架额外提供多种预设（`tiny`/`small`/`medium`/`foundation`/`large`/`moe`），便于按资源快速选择合适的规模。其中 `foundation` 配置采用 24 层 × 768 宽度的架构（约 2.1 亿参数），在 GQA (16Q→4KV)、Flash Attention 与梯度检查点的配合下兼顾训练吞吐与显存占用。

## MiniGPT 模型结构
`MiniGPT` 继承自 `torch.nn.Module`，实现了标准的 Transformer Decoder 结构：

1. **嵌入层**：`nn.Embedding` 将 token ID 映射为向量，可选择与输出层权重共享
2. **位置编码**：默认启用 RoPE，兼容传统正弦位置编码
3. **TransformerBlock 列表**：每个 block 包含注意力层、SwiGLU 前馈网络、RMSNorm 与可选的 GQA/MoE
4. **输出层**：最后经 RMSNorm 再映射回词表维度

### TransformerBlock 关键组件
- **注意力层**：默认使用 `OptimizedMultiHeadAttention`；当 `use_gqa=True` 时启用 `GroupedQueryAttention`
- **RoPE 支持**：`RotaryPositionEmbedding` 与 `apply_rotary_pos_emb` 在注意力计算中注入位置信息
- **SwiGLU 前馈网络**：`SwiGLUFeedForward` 替代传统 `GELU`
- **RMSNorm**：`RMSNorm` 替代 `LayerNorm`，减少均值计算成本
- **MoE（可选）**：`ExpertMLP` 与 `MoELayer` 提供混合专家结构

## 文本生成
`MiniGPT.generate` 简化封装了因果掩码、温度采样、Top-k/Top-p 的基本逻辑；若需更复杂的解码，可使用 `src/inference/generator.TextGenerator`。

## 与训练器的衔接
模型前向返回 `(batch_size, seq_len, vocab_size)` 的 logits，可直接送入 `PreTrainer`/`SFTTrainer` 的交叉熵损失计算。配置对象中定义的 `bos_token_id`/`eos_token_id`/`pad_token_id` 与分词器保持一致即可。

如需扩展模型（例如引入新的注意力实现或激活函数），建议：
1. 在 `MiniGPTConfig` 中增加可控开关
2. 在 `TransformerBlock` 中根据配置进行分支选择
3. 补充训练与推理文档，保持文档-代码一致
