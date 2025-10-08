# 4.3 验证评估与可视化

> 以 `PreTrainer.validate`、`save_checkpoint` 与 `plot_training_curve` 为例，解释评估与可视化流程。

## 代码走读
| 代码位置 | 原理解析 | 这么做的理由 |
| --- | --- | --- |
| `self.model.eval()`<br>`with torch.no_grad():` | 验证时关闭 dropout 并禁用梯度。 | 获取可靠的评估指标并减少显存占用。 |
| `labels = torch.cat([input_ids[:, 1:], pad_col], dim=1)` | 构建验证标签。 | 复用训练时的自回归损失逻辑，保证一致性。 |
| `if val_loss < best_val_loss: self.save_checkpoint(...)` | 保存最佳模型。 | 实现早停策略，锁定最优泛化性能。 |
| `plt.plot(epochs, self.train_losses, 'b-')` 等 | 绘制训练/验证损失曲线。 | 可视化训练动态，帮助定位过拟合或欠拟合。 |
| `plt.savefig(os.path.join(save_dir, 'training_curve.png'))` | 将图像写入磁盘。 | 方便在实验记录中留存证据。 |

## 实战建议
- 若在无图形界面环境中运行，请确保 `matplotlib` 使用 `Agg` 后端。
- 可以在保存 checkpoint 时附带超参配置，便于复现实验。 
