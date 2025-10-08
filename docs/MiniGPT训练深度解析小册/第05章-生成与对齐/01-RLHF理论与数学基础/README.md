# 5.1 RLHF 理论基石

> 从 `src/rl/rlhf_pipeline.py` 的配置与初始化切入，梳理强化学习与人类反馈结合的关键步骤。

## 代码走读
| 代码位置 | 原理解析 | 这么做的理由 |
| --- | --- | --- |
| `class RLHFConfig` | 用 dataclass 定义超参，包括 SFT、RM、PPO 三阶段。 | 清晰组织 RLHF 流程的全部变量，便于实验追踪。 |
| `self._setup_logging()` | 配置日志格式与级别。 | 在长流程训练中提供可追溯的记录。 |
| `self.device = self._setup_device()` | 根据硬件选择 CPU/CUDA/MPS。 | 确保后续训练组件使用统一设备。 |
| `self._save_config()` | 将配置写入磁盘。 | 留下实验配置，符合强化学习实验的复现要求。 |
| `self.logger.info("RLHF管道初始化完成")` | 标记初始化结束。 | 方便检查流程是否按预期启动。 |

## 实战建议
- 若需要分布式训练，可扩展 `_setup_device` 以支持 `cuda:{rank}`。
- 记录配置文件时可附带 Git 提交号，便于追踪代码版本。 
