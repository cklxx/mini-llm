# 4.1 训练循环逐行拆解

> 以 `PreTrainer.train_epoch` 为例，拆解一个 epoch 的关键步骤。

## 代码走读
| 代码位置 | 原理解析 | 这么做的理由 |
| --- | --- | --- |
| `self.model.train()` | 切换到训练模式，启用 dropout 等随机性。 | 确保训练与推理行为差异明确。 |
| `progress_bar = tqdm(dataloader, desc="预训练")` | 使用进度条封装迭代器。 | 便于观察训练进度与动态指标。 |
| `for batch_idx, batch in enumerate(progress_bar):` | 遍历 DataLoader，获取每个批次。 | 逐批更新参数，支持大规模数据。 |
| `logits = self.model(input_ids)` | 前向传播。 | 获取当前模型对批次的预测。 |
| `self.optimizer.zero_grad()`<br>`loss.backward()`<br>`self.optimizer.step()` | 标准的梯度更新流程。 | 清零旧梯度、防止累积；反传计算梯度；更新参数。 |

## 实战建议
- 可在循环内加入 `if (batch_idx + 1) % grad_accum == 0:` 实现梯度累积。
- 若在多机环境中运行，请结合 `DistributedSampler` 保证数据划分均匀。 
