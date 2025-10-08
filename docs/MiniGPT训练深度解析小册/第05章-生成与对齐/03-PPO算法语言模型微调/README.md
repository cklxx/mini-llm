# 5.3 PPO 微调语言模型

> 对 `PPOTrainer` 的初始化与缓冲区实现进行逐行注释，理解策略优化的关键步骤。

## 代码走读
| 代码位置 | 原理解析 | 这么做的理由 |
| --- | --- | --- |
| `class PPOExperienceBuffer` | 定义经验缓冲区，存储状态、动作、奖励等。 | 解决语言模型生成序列无法一次性放入内存的问题。 |
| `self.states.append(state)` 等 | 将轨迹数据逐项压入缓冲区。 | 保存完整轨迹，为后续优势估计提供原始数据。 |
| `if len(self.states) > self.max_size: ... pop(0)` | 限制缓冲区容量。 | 避免内存无限增长，同时保留最近的策略分布。 |
| `self.reference_model = self._create_reference_model()` | 构建冻结的参考策略。 | PPO 中需要 KL 约束，参考模型提供旧策略概率。 |
| `self.pg_computer = create_policy_gradient_computer(...)` | 创建策略梯度计算器，内部包含 GAE 与裁剪目标。 | 将复杂的优势计算与损失封装，保持主循环清晰。 |

## 实战建议
- 若观察到 KL 过大，可调低 `clip_ratio` 或增大 `target_kl` 的触发频率。
- 经验缓冲区可根据显存调整 `max_size`，同时注意保持批次覆盖足够多的对话样本。 
