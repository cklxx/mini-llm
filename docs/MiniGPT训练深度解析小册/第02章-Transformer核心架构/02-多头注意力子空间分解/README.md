# 2.2 多头注意力的子空间解耦

> 对 `MultiHeadAttention.forward` 逐行注释，展示多头机制如何并行化子空间表示。

## 代码走读
| 代码位置 | 原理解析 | 这么做的理由 |
| --- | --- | --- |
| `self.w_q = nn.Linear(d_model, d_model)` 等 | 三个线性层分别生成查询、键、值矩阵。 | 允许模型为不同任务维度学习独立投影，提高表达力。 |
| `Q = Q.reshape(...).transpose(1, 2)` | 将批次、序列、头、维度重新排列成 `(batch, heads, seq, d_k)`。 | 方便后续按头并行地执行矩阵乘法，充分利用 GPU。 |
| `attention_output = F.scaled_dot_product_attention(Q, K, V, mask)` | 调用 PyTorch 内核执行多头注意力。 | 内置实现已做 GPU 优化并支持 Flash Attention，减少实现复杂度。 |
| `attention_output = attention_output.transpose(1, 2).contiguous()` | 把注意力结果恢复到 `(batch, seq, heads, d_k)`。 | 为重组成 `d_model` 做准备，确保内存布局连续以加速。 |
| `output = self.w_o(attention_output)` | 通过线性层混合各个头的信息。 | 让不同头之间的信息交互，形成最终的上下文表示。 |

## 小结
- 每个注意力头相当于在不同子空间内执行近似的低秩分解。
- 若需要扩展到 Flash Attention，可在配置中开启 `flash_attn`，实现时复用相同接口。 
