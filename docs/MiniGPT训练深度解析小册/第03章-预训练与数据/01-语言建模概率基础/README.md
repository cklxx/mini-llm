# 3.1 语言建模的概率建模

> 聚焦 `PreTrainer.compute_loss` 的实现，解释交叉熵损失如何衡量条件概率。

## 代码走读
| 代码位置 | 原理解析 | 这么做的理由 |
| --- | --- | --- |
| `logits = logits.reshape(-1, logits.size(-1))` | 将三维张量展平成 `(batch*seq, vocab)`。 | 交叉熵要求输入为二维 `[N, C]`，便于对每个 token 计算概率。 |
| `labels = labels.reshape(-1)` | 标签展平为 `(batch*seq,)`。 | 与 logits 对齐，实现逐 token 的条件概率匹配。 |
| `loss_fn = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_id)` | 定义交叉熵损失，并忽略 PAD。 | 避免填充 token 干扰梯度，符合语言模型只预测真实 token 的目标。 |
| `loss = loss_fn(logits, labels)` | 计算条件负对数似然。 | 最大化正确 token 的概率，最小化困惑度。 |

## 实战建议
- 如果想输出困惑度，可在训练循环中打印 `loss.exp()`。
- 当使用自定义 tokenizer 时，务必确认 `pad_id` 与数据集一致，否则忽略逻辑会失效。 
