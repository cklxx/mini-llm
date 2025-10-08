# 1.1 信息论视角下的注意力概率

> 结合 `src/model/transformer.py` 中的实现，从熵与概率的角度拆解注意力公式的每一行代码。

## 代码走读
| 代码位置 | 原理解析 | 这么做的理由 |
| --- | --- | --- |
| `Q = self.w_q(query)`<br>`K = self.w_k(key)`<br>`V = self.w_v(value)` | 线性层将输入投影到查询、键、值三个子空间，相当于对序列信息进行不同视角的编码，准备计算条件分布。 | 将不同的信息通道分离，便于后续用点积模拟“条件互信息”，同时保持可学习性。 |
| `attention_output = F.scaled_dot_product_attention(Q, K, V, mask)` | PyTorch 的内置函数先计算 `Q @ K^T / sqrt(d_k)`，再调用 Softmax 获得概率分布，最后加权 `V`。这个过程实现了最大熵原理下的最优注意力分配。 | 使用框架内核可获得更稳定的数值和更快的实现，同时自动应用 mask，保证注意力只落在允许的位置上。 |
| `scores = torch.matmul(query, key.transpose(-2, -1)) / self.scale` | 在 `src/model/rope.py` 的 `RoPEAttention` 中显式计算注意力分数，体现了点积对应互信息估计，除以 `sqrt(d_k)` 控制熵的尺度。 | 显式写法便于调试和插入自定义逻辑，例如对注意力矩阵做可视化或统计。 |
| `attn_weights = torch.softmax(scores, dim=-1)` | Softmax 将未归一化得分转为概率分布，实现 Shannon 熵最大化约束下的权重求解。 | 保证每行权重和为 1，使注意力符合概率意义，方便解释与分析。 |
| `attn_output = torch.matmul(attn_weights, value)` | 将概率权重对 `V` 做加权求和，相当于对信息源进行期望计算，输出即“信息熵最小化”的表示。 | 通过加权期望聚合上下文，保证梯度可以反向传递到 `V` 和上游投影层，实现端到端训练。 |

## 实验提示
- 想要记录注意力熵，可在 `TransformerBlock.forward` 中对 `attn_weights` 做 `-(p * log p).sum(dim=-1)`。
- 若希望调节注意力的温度，可以在调用 `scaled_dot_product_attention` 前后分别乘以或除以系数，模拟不同熵约束。 
