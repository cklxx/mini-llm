# 4.2 梯度管理与稳定技巧

> 总结 `train_epoch` 中的梯度控制细节，确保训练稳定。

## 代码走读
| 代码位置 | 原理解析 | 这么做的理由 |
| --- | --- | --- |
| `self.optimizer.zero_grad()` | 清空上一批次的梯度。 | 防止梯度累积导致更新方向错误。 |
| `loss.backward()` | 计算当前批次的梯度。 | 依据链式法则传播误差，为参数更新提供方向。 |
| `torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)` | 对梯度做范数裁剪。 | 避免梯度爆炸，保护训练稳定。 |
| `self.scheduler.step()` | 每个 step 调整学习率。 | 让学习率按余弦曲线下降，缓解训练后期震荡。 |
| `progress_bar.set_postfix({'loss': ..., 'lr': ...})` | 实时记录损失与学习率。 | 便于观察梯度是否稳定，及时定位异常。 |

## 实战建议
- 在混合精度训练中，可结合 `GradScaler`，在 `loss.backward()` 前后插入缩放/反缩放流程。
- 若梯度常被裁剪到 1.0，可适当减小学习率或提高裁剪阈值。 
