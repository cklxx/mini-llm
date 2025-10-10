# 第 02 章 · Transformer 核心架构

本章聚焦 mini-llm 的模型搭建方式，解析配置、注意力优化以及可选的 MOE 扩展。推荐配合 `src/model/config.py`、`src/model/transformer.py`、`src/model/gqa.py` 和 `src/model/moe.py` 阅读。

## 2.1 配置驱动的模型构建
- `MiniGPTConfig` 将所有结构化超参集中管理，初始化时自动校验 `hidden_size % num_attention_heads == 0` 等约束。
- `get_tiny_config` / `get_small_config` / `get_small_30m_config` 提供了可直接复现的参考设置，便于在实验中快速切换模型规模。
- 常用调参入口：`use_rope` 控制位置编码，`use_gqa`/`num_key_value_heads` 控制注意力头共享，`use_moe` 触发专家路由。

## 2.2 解码器堆叠
- `MiniGPT` 采用纯解码器结构，核心逻辑在 `forward` 中：
  1. 嵌入输入 token；
  2. 根据配置决定是否注入正弦位置编码或 RoPE；
  3. 逐层调用 `TransformerBlock`，每层包含自注意力 + SwiGLU 前馈；
  4. 通过 `RMSNorm` 和输出线性层生成 logits。
- `create_causal_mask` 使用下三角矩阵实现因果约束，推理时无需额外写 mask 逻辑。

## 2.3 Grouped-Query Attention (GQA)
- `GroupedQueryAttention` 在查询头之间共享键值头，`num_queries_per_kv = num_heads // num_key_value_heads` 控制共享比例。
- 当 `flash_attn=True` 且 PyTorch 版本支持时，内部走 `F.scaled_dot_product_attention` 的高效实现。
- RoPE 支持：在 `forward` 中根据 `position_ids` 调用 `apply_rotary_pos_emb`，可与增量解码缓存 `past_key_value` 协同工作。

## 2.4 SwiGLU 前馈与 RMSNorm
- `SwiGLUFeedForward` 用 `F.silu(gate) * up` 实现门控激活，并在输出前应用 dropout，提高表示能力。
- `RMSNorm` 作为预归一化层消除了均值中心化，使深层模型训练更加稳定。

## 2.5 Mixture-of-Experts (可选)
- `src/model/moe.py` 中的 `Router`、`Expert`、`SparseMoE` 组合实现稀疏专家路由，`top_k` 和 `capacity_factor` 控制每个 token 激活的专家数量及冗余空间。
- 若要在自定义模型中集成 MoE，可将 `TransformerBlock` 的前馈部分替换为 `SparseMoE`，并在 `MiniGPTConfig` 里开启 `use_moe` 以统一管理超参。

> 实践提示：调试推理性能时，先在配置中开启 `use_gqa=True`，若显存仍吃紧再考虑启用 MOE（需要额外的分布式通信支持）。
