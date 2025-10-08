# 5.4 生成策略与推理控制

> 解析 `src/inference/generator.py` 中的采样策略，实现推理阶段的逐行解释。

## 代码走读
| 代码位置 | 原理解析 | 这么做的理由 |
| --- | --- | --- |
| `class GenerationConfig` | 用 dataclass 定义生成超参。 | 统一管理温度、top-k、top-p 等策略参数。 |
| `self.model.eval()` | 初始化时将模型置为评估模式。 | 禁用 dropout，保证生成确定性。 |
| `next_token_logits = outputs[:, -1, :] / config.temperature` | 对最后一个 token 的 logits 做温度缩放。 | 控制分布的平滑度，温度越低越保守。 |
| `next_token_logits = self.top_k_filtering(next_token_logits, config.top_k)` | 保留概率最高的 K 个 token。 | 避免尾部分布噪声，提高可控性。 |
| `next_token = torch.multinomial(probs, num_samples=1)` | 从概率分布中采样下一个 token。 | 保持生成多样性，同时尊重概率权重。 |

## 实战建议
- 若需要贪心解码，可将 `config.do_sample` 设为 `False` 并把 `top_k` 置 0。
- 结合 `apply_repetition_penalty` 可以抑制循环输出，适合长对话场景。 
