# 1.4 统计学习与泛化分析

> 使用 `src/training/training_monitor.py` 和 `src/training/trainer.py` 的接口，说明如何从统计角度监控模型。

## 代码走读
| 代码位置 | 原理解析 | 这么做的理由 |
| --- | --- | --- |
| `progress_bar.set_postfix({'loss': f'{loss.item():.4f}', 'lr': ...})` | 在训练循环中实时记录损失和学习率，相当于在线估计经验风险。 | 让我们能够观察风险随 step 的变化趋势，判断是否过拟合或欠拟合。 |
| `self.train_losses.append(train_loss)` | 保存每个 epoch 的平均损失。 | 构建经验风险序列，后续可绘制学习曲线评估收敛性。 |
| `if val_dataloader:`<br>`val_loss = self.validate(...)` | 通过独立验证集评估泛化误差。 | 对照训练损失与验证损失可发现过拟合迹象，指导正则化策略。 |
| `plt.plot(epochs, self.train_losses, 'b-', label='训练损失')` | 在 `plot_training_curve` 中绘制曲线，可视化经验风险。 | 直观呈现模型在训练与验证集上的表现，支持统计学习中的偏差-方差分析。 |
| `self.save_checkpoint(os.path.join(save_dir, "best_model.pt"))` | 保存验证损失最低的模型。 | 实现早停思想，防止后续训练导致泛化性能下降。 |

## 实践建议
- 可将验证集划分为多折，结合 `validate` 函数进行交叉验证。
- 若损失震荡，尝试调低学习率或增大 batch size，以降低方差。 
