# 2.1 TransformerBlock 中的注意力数学

> 聚焦 `TransformerBlock.forward` 的注意力路径，逐行解释如何将数学公式落地成可微代码。

## 代码走读
| 代码位置 | 原理解析 | 这么做的理由 |
| --- | --- | --- |
| `normalized_x = self.norm1(x)` | Pre-LN 结构先对输入做 RMSNorm，使其均方范数接近 1。 | 保障后续注意力计算的数值稳定，减轻梯度消失与爆炸。 |
| `attn_output, _ = self.attention(...)` | 当启用 GQA 时，调用 `GroupedQueryAttention`，内部复用少量键/值头来服务更多查询头。 | 减少显存和算力开销，同时保持注意力表达力，适配大模型常用的 GQA 结构。 |
| `attn_output = self.attention(normalized_x, normalized_x, normalized_x, mask)` | 关闭 GQA 时回退到标准多头注意力，输入相同张量表示自注意力。 | 保留兼容性，方便对照验证。 |
| `x = x + self.dropout(attn_output)` | 残差连接 + dropout，对注意力输出进行正则化并与原表示融合。 | 残差保证梯度可以直接传到浅层，dropout 提升泛化能力。 |
| `normalized_x = self.norm2(x)` | 第二次 RMSNorm，为后续前馈网络提供稳定输入。 | 维持层间统计一致性，避免前馈层放大激活。 |

## 调试建议
- 可通过 `self.attention(..., return_attn_weights=True)`（GQA 支持）检查注意力矩阵是否符合预期。
- 观察 `self.dropout.p`，在推理时应切换到 `model.eval()` 关闭随机性。 
