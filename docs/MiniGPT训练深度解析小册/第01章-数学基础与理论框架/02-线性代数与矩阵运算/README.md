# 1.2 线性代数驱动的表示变换

> 解析 `src/model/transformer.py` 与 `src/model/rope.py` 中的矩阵操作，理解每一行线性代数对模型稳定性的贡献。

## 代码走读
| 代码位置 | 原理解析 | 这么做的理由 |
| --- | --- | --- |
| `Q = Q.reshape(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)` | 使用张量重排把隐藏向量拆成 `n_heads × d_k`，对应线性代数中的块矩阵分解。 | 让每个注意力头独立处理子空间，充分利用并行性并降低单头的维度，提升表达能力。 |
| `scores = torch.matmul(query, key.transpose(-2, -1))` | 经典矩阵乘法实现 `QK^T`，本质是计算不同 token 表示之间的内积。 | 内积衡量相似度，直接对应向量空间中的投影关系，是注意力权重的基础。 |
| `cos = cos.unsqueeze(1)`<br>`sin = sin.unsqueeze(1)` | 在 RoPE 中扩展维度以便广播，等价于对每个注意力头复制旋转矩阵。 | 让相同角频率作用到所有头上，保证不同头对相对位置的感知一致。 |
| `rotate_half(q)` | 将张量拆成两半并交换符号，等价于复数空间中的乘以 `i` 操作。 | 与 `cos`、`sin` 相乘时形成二维旋转矩阵，实现 RoPE 的几何解释。 |
| `attention_output = attention_output.reshape(batch_size, seq_len, d_model)` | 把多头结果拼回原始维度，相当于矩阵块的拼接还原。 | 复原输入形状，便于后续残差连接与前馈层继续处理。 |

## 进一步阅读
- 建议配合 `torch.einsum` 做等价推导，帮助理解张量重排的维度意义。
- 可以在调试时打印 `Q.norm(dim=-1)`，观察不同头的向量范数是否平衡。 
