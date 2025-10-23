# 第 02 章 · Transformer 核心架构

本章聚焦 mini-llm 的模型搭建方式，解析配置、注意力优化以及可选的 MOE 扩展。推荐配合 `src/model/config.py`、`src/model/transformer.py` 与 `src/model/moe.py` 阅读。

## 2.1 配置驱动的模型构建
- `MiniGPTConfig` 将所有结构化超参集中管理，初始化时自动校验 `hidden_size % num_attention_heads == 0` 等约束。
- `get_tiny_config` / `get_small_config` / `get_small_30m_config` 提供了可直接复现的参考设置，便于在实验中快速切换模型规模。
- 常用调参入口：`use_rope` 控制位置编码，`use_gqa`/`num_key_value_heads` 控制注意力头共享，`use_moe` 触发专家路由。

## 2.2 解码器堆叠
- `MiniMindModel` 负责嵌入输入 token、应用 dropout，并在初始化阶段预计算 RoPE 的 cos/sin 缓存；前向时会按当前序列长度切片后传给各层使用。【F:src/model/transformer.py†L203-L264】
- 每个 `MiniMindBlock` 采用 Pre-Norm 架构：先进入 `MiniMindAttention`（完成 GQA/Flash 注意力与残差），再经过第二个 RMSNorm 进入密集前馈或 MoE。【F:src/model/transformer.py†L154-L200】
- `MiniGPT.forward` 在层堆叠结束后通过 RMSNorm 与共享嵌入（或可选独立 `lm_head`）生成 logits，并在需要时返回 KV 缓存与 MoE 辅助损失。【F:src/model/transformer.py†L267-L310】

## 2.3 MiniMindAttention (GQA + RoPE)
- `MiniMindAttention` 在查询头之间共享键值头，`num_key_value_heads` 决定共享比例，内部通过 `repeat_kv` 复用 KV 并保持与 MiniMind 一致的头维拆分。【F:src/model/transformer.py†L71-L151】
- 当 `flash_attn=True` 且环境支持时，注意力调用 `F.scaled_dot_product_attention`；否则回退到遮罩 Softmax 并合并因果与可选的 padding 掩码。【F:src/model/transformer.py†L127-L148】
- RoPE 支持：每层根据当前位置切片 cos/sin 并对 Q/K 执行旋转；增量解码时会与 `past_key_value` 串联以复用历史缓存。【F:src/model/transformer.py†L111-L121】

## 2.4 MiniMindFeedForward 与 RMSNorm
- `MiniMindFeedForward` 使用 SiLU 门控的三线性结构（gate/up/down），与 MiniMind 默认实现一致，并在输出前应用 dropout。【F:src/model/moe.py†L35-L50】
- 若启用 MoE，`MiniMindMOEFeedForward` 会复用相同的门控前馈作为专家，同时提供 top-k 路由与可选共享专家，辅助损失在前向中累加。【F:src/model/transformer.py†L154-L200】【F:src/model/moe.py†L117-L208】
- `RMSNorm` 在注意力与前馈前作为预归一化出现，帮助深层模型保持训练稳定。【F:src/model/transformer.py†L154-L200】

## 2.5 Mixture-of-Experts (可选)
- `MiniMindMOEFeedForward` 由 `MiniMindMoEGate`、路由专家与可选共享专家组成，`num_experts_per_tok`、`n_routed_experts` 等超参全部与 MiniMind 对齐。【F:src/model/moe.py†L53-L208】
- 在 `MiniMindBlock` 中启用 MoE 只需在配置里设置 `use_moe=True`，其余参数将由 `build_moelayer_from_config` 自动读取并注入模型。【F:src/model/transformer.py†L154-L200】【F:src/model/moe.py†L193-L208】

> 实践提示：调试推理性能时，先在配置中开启 `use_gqa=True`，若显存仍吃紧再考虑启用 MOE（需要额外的分布式通信支持）。
