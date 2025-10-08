# 2.5 SwiGLU 前馈网络拆解

> 详细注释 `SwiGLUFeedForward` 的每一行，理解激活函数与维度扩展的配合。

## 代码走读
| 代码位置 | 原理解析 | 这么做的理由 |
| --- | --- | --- |
| `self.w_gate = nn.Linear(dim, hidden_dim, bias=False)` | 生成门控向量。 | 用于控制通过量，模仿 GLU 结构。 |
| `self.w_up = nn.Linear(dim, hidden_dim, bias=False)` | 生成候选向量。 | 与门控做逐元素相乘形成非线性组合。 |
| `gate = self.w_gate(x)`<br>`up = self.w_up(x)` | 并行计算门和值。 | 降低一次前向的线性层调用次数。 |
| `gated_output = F.silu(gate) * up` | SiLU 激活提供平滑的非线性，乘上 `up` 实现自适应缩放。 | 比 ReLU 更平滑，梯度在零点连续，训练更稳定。 |
| `output = self.w_down(self.dropout(gated_output))` | dropout 后线性映射回原维度。 | Dropout 提升泛化，线性层保证残差相加时维度一致。 |

## 实战建议
- 如需缩减计算量，可将 `hidden_dim` 设置为 `int(2.67 * dim)`，符合 SwiGLU 原论文建议。
- 若推理时不想引入随机性，记得调用 `model.eval()` 关闭 dropout。 
