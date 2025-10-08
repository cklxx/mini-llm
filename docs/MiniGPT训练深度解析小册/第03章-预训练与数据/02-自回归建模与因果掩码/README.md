# 3.2 自回归与因果掩码

> 从 `MiniGPT.forward` 的实现入手，阐释自回归建模的关键行。

## 代码走读
| 代码位置 | 原理解析 | 这么做的理由 |
| --- | --- | --- |
| `causal_mask = self.create_causal_mask(seq_len)` | 生成下三角矩阵，确保每个位置只能看到过去。 | 符合自回归假设，防止信息泄漏。 |
| `if position_ids is None and self.use_rope:`<br>`position_ids = torch.arange(seq_len, ...)` | 为 RoPE 准备位置索引。 | 在推理时也能处理不同长度输入，保证旋转编码连续。 |
| `for transformer_block in self.transformer_blocks: x = transformer_block(x, causal_mask, position_ids)` | 逐层传递同一掩码和位置。 | 所有层共享同一自回归约束，保证全局一致。 |
| `if self.lm_head is not None: logits = self.lm_head(x)` | 映射到词表维度，输出条件概率分布。 | 为语言建模提供最终预测。 |

## 实践建议
- 若想适配双向任务，可改写 `create_causal_mask` 为全 1，并在外部控制掩码逻辑。
- 推理时可缓存 `transformer_block` 的 KV，以减少重复计算。 
