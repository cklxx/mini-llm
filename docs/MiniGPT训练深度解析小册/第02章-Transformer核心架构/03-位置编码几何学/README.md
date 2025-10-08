# 2.3 RoPE 的几何含义

> 从 `src/model/rope.py` 的实现出发，一行行理解旋转位置编码的推导。

## 代码走读
| 代码位置 | 原理解析 | 这么做的理由 |
| --- | --- | --- |
| `inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))` | 构造频率向量，相当于把角频率按指数衰减排列。 | 让低维捕捉长周期， 高维捕捉短周期，兼顾局部与全局位置感知。 |
| `self.register_buffer("cos_cached", emb.cos(), ...)` | 预计算并缓存 cos/sin，避免每次前向重复计算。 | 提升推理效率，降低数值误差。 |
| `if position_ids is not None: cos = cos[position_ids]` | 根据位置索引提取对应角度。 | 支持批量中每个样本拥有不同长度或位置跳跃。 |
| `q_embed = (q * cos) + (rotate_half(q) * sin)` | 将查询向量与旋转矩阵相乘，实现二维平面旋转。 | 在保持范数不变的同时注入相对位置信息，兼容自注意力需求。 |
| `rotate_half` 函数 | 切分并交换张量的前后半部分，对应复数乘以 `i`。 | 提供高效的旋转实现，无需显式构建大矩阵。 |

## 实践提醒
- 若序列长度超过缓存，可留意 `_set_cos_sin_cache` 会自动扩容，但也意味着需要额外显存。
- 可在调试中打印 `cos_cached[0]`，验证频率是否按指数衰减。 
