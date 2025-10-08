# 第 01 章 · 数学基础与理论框架

本章梳理 mini-llm 中最常见的数学工具，并展示它们如何在源码中落地。阅读时请对照 `src/model/transformer.py`、`src/model/rope.py` 和 `src/training/trainer.py`。

## 1.1 注意力的概率视角
- **Scaled Dot-Product Attention**：在 `TransformerBlock.forward` 中，通过 `q @ k.transpose(-2, -1)` 计算相似度矩阵，随后除以 `math.sqrt(head_dim)` 完成缩放。
- **Softmax 归一化**：`torch.nn.functional.softmax` 将注意力权重解释为概率分布，保证每个 token 的权重和为 1。
- **Mask 处理**：推理时的因果 mask 由 `src/model/transformer.py` 中的 `Transformer.create_causal_mask` 生成，确保模型只关注历史 token。

> 实践提示：如果你想调试注意力权重，建议在 `TransformerBlock` 中插入 `torch.no_grad()` 的钩子，并输出 `attn_weights` 的统计量。

## 1.2 残差连接与层归一化
- **残差公式**：`y = x + f(x)`，对应代码中 `hidden_states = residual + self.dropout(output)`。
- **层归一化**：`LayerNorm(eps)` 在 mini-llm 里使用预归一化（Pre-LN）结构，相关实现位于 `src/model/transformer.py` 的 `RMSNorm`。
- **稳定性原因**：预归一化可以在深层网络中保持梯度稳定，避免训练前期的梯度爆炸。

## 1.3 RoPE 位置编码
- **旋转公式**：通过二维旋转矩阵嵌入位置信息，`src/model/rope.py` 中的 `apply_rope` 函数会在计算 q/k 时注入角频率。
- **频率选取**：`RotaryPositionEmbedding` 读取配置中的 `rope_theta` 并生成频率表，与 `LLaMA` 系列的实现保持一致。

## 1.4 损失函数与梯度
- **Cross Entropy**：`src/training/trainer.py` 中 `PreTrainer.compute_loss` 调用 `nn.CrossEntropyLoss`，默认使用 `ignore_index=self.tokenizer.pad_id` 以跳过 padding token。
- **梯度裁剪**：`torch.nn.utils.clip_grad_norm_` 在 `PreTrainer.train_epoch` 里保证梯度范数不超过 1.0。
- **学习率调度**：`PreTrainer` 默认启用 `optim.lr_scheduler.CosineAnnealingLR`，并在每个 step 调用 `self.scheduler.step()`。

> 若要复现论文中的数值，建议使用 `torch.autocast` 检查混合精度对梯度的影响，并结合训练监控 (`src/training/training_monitor.py`) 观察损失曲线。

## 1.5 进一步阅读
- 《Attention Is All You Need》：理解自注意力的原理。
- 《A Primer in BERTology》：了解预训练语言模型的损失建模方式。
- mini-llm [`docs/training.md`](../../training.md)：介绍完整的训练流水线。
