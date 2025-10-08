# 1.3 优化理论与梯度下降

> 对照 `src/training/trainer.py`，逐行理解预训练器如何实现梯度下降与学习率调度。

## 代码走读
| 代码位置 | 原理解析 | 这么做的理由 |
| --- | --- | --- |
| `self.optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)` | 采用 AdamW，结合自适应二阶动量与 L2 正则（通过权重衰减实现）。 | AdamW 在大模型训练中表现稳定，能在小批量下保持收敛速度，并抑制过拟合。 |
| `self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=1000)` | 使用余弦退火调度学习率，模拟热退火过程。 | 有助于在训练后期细化模型，避免震荡，同时无需手动设定多阶段超参。 |
| `loss.backward()` | 根据链式法则反向传播梯度。 | 将损失对参数的偏导数累积到各层，为梯度下降提供方向信息。 |
| `torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)` | 对梯度范数做裁剪，实现梯度约束。 | 防止梯度爆炸，保持优化稳定，尤其在混合精度训练或长序列场景下。 |
| `self.optimizer.step()`<br>`self.scheduler.step()` | 先更新参数，再更新学习率。 | 保证本步使用旧学习率完成更新，然后准备下一个 step 的学习率。 |

## 实战提示
- 若使用分布式训练，需在 `loss.backward()` 之后插入 `scaler.unscale_` 等操作以兼容梯度缩放。
- 可以记录 `self.scheduler.get_last_lr()`，观察余弦曲线是否与预期一致。 
