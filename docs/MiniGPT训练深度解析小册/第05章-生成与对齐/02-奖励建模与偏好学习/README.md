# 5.2 奖励建模与偏好学习

> 逐行解析 `RewardModel` 与 `RewardTrainer` 的关键实现，理解奖励模型训练流程。

## 代码走读
| 代码位置 | 原理解析 | 这么做的理由 |
| --- | --- | --- |
| `class RewardHead(nn.Module)` | 定义多层感知机，将隐藏状态映射为标量奖励。 | 为人类偏好建模提供可学习的评分函数。 |
| `sequence_lengths = attention_mask.sum(dim=1) - 1` | 根据注意力掩码找到最后一个有效 token。 | 奖励通常针对完整回复，因此选择序列末尾的表示。 |
| `reward = self.reward_head(last_hidden_states)` | 生成奖励分数。 | 将语义表示转化为单标量，便于比较优劣。 |
| `self.preference_loss = create_preference_loss(...)` | 组合排序损失与温度参数。 | 逼近人类偏好排序，实现 Bradley-Terry 等模型。 |
| `self.optimizer = optim.AdamW(...)` | 为奖励模型配置优化器。 | 使用稳定的优化策略，支持正则化。 |

## 实战建议
- 如果奖励模型训练不稳定，可开启 `freeze_backbone` 只微调奖励头。
- 建议在每轮训练后评估奖励模型是否真正拉开好/坏样本的得分差距。 
