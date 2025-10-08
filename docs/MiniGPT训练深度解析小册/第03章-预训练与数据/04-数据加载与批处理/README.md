# 3.4 数据加载与批处理

> 逐行解析 `LanguageModelingDataset.__getitem__`，理解输入构建与异常兜底策略。

## 代码走读
| 代码位置 | 原理解析 | 这么做的理由 |
| --- | --- | --- |
| `token_ids = self.tokenizer.encode(text, add_special_tokens=True)` | 将原始文本编码成 token 序列。 | 保留 BOS/EOS，方便语言模型训练。 |
| `if len(token_ids) < 2: token_ids = [bos, eos]` | 确保至少有输入与目标。 | 避免过短样本导致 shift 操作失败。 |
| `if len(token_ids) > self.max_length: token_ids = token_ids[:self.max_length]` | 截断超长序列。 | 控制显存消耗，并保持 batch 中长度一致。 |
| `needed_padding = max(0, self.max_length - len(token_ids)); token_ids.extend([pad] * needed_padding)` | 对不足长度的样本补 PAD。 | 统一张量尺寸，便于堆叠成批次。 |
| `except Exception as e: ... return torch.tensor(default_tokens, ...)` | 捕获异常并返回安全默认值。 | 在脏数据或编码失败时保持训练不中断。 |

## 实践建议
- 若你使用流式数据，可将 `max_length` 设为更小值并配合 `DataLoader` 的 `drop_last` 控制批次。
- 可以扩展异常处理逻辑，把失败样本记录到日志，方便数据清洗。 
