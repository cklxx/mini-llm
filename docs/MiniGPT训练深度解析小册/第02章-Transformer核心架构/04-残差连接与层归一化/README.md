# 2.4 残差 + 归一化的稳定器

> 阅读 `RMSNorm` 与 `TransformerBlock` 的残差实现，理解每一行对梯度流的作用。

## 代码走读
| 代码位置 | 原理解析 | 这么做的理由 |
| --- | --- | --- |
| `variance = hidden_states.pow(2).mean(-1, keepdim=True)` | RMSNorm 通过均方根估计尺度，无需减均值。 | 保持计算简单且稳定，适合大模型的半精度训练。 |
| `hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)` | 对输入做尺度归一化。 | 避免小方差导致的放大，`epsilon` 防止除零。 |
| `return self.weight * hidden_states.to(input_dtype)` | 应用可学习的缩放参数并恢复原 dtype。 | 允许模型重新调整尺度，同时兼容混合精度。 |
| `x = x + self.dropout(attn_output)` | 注意力残差连接。 | 提供信息短路，保持原始表示并促进梯度传播。 |
| `x = x + self.dropout(ff_output)` | 前馈层残差连接。 | 让非线性层只需学习增量，避免梯度衰减。 |

## 实践建议
- 若在训练中观察到 `nan`，可尝试增大 `rms_norm_eps` 以提高数值稳定性。
- 当模型出现梯度爆炸时，优先检查是否误删了残差或 dropout。 
