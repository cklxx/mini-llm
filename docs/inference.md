# 🗣️ 推理与文本生成

Mini-LLM 的推理组件位于 `src/inference/generator.py`。`TextGenerator` 在保持 API 简洁的同时，提供多种主流的解码策略。

## GenerationConfig
核心字段：
- `max_length`：最大生成步数
- `temperature`：采样温度，越大越随机
- `top_k` / `top_p`：Top-k 与 Nucleus (Top-p) 采样
- `repetition_penalty`：重复惩罚系数
- `num_beams` / `early_stopping`：束搜索参数
- `do_sample`：是否启用随机采样

## TextGenerator 功能
- **greedy_search**：逐步选择最大概率 token
- **sample_generate**：支持温度、Top-k、Top-p、重复惩罚的随机采样
- **beam_search**：维护多个候选序列并根据平均对数概率选择最优输出
- **辅助方法**：`apply_repetition_penalty`、`top_k_filtering`、`top_p_filtering`

## 使用示例
```python
import torch
from src.inference.generator import TextGenerator, GenerationConfig

text_generator = TextGenerator(model, tokenizer, device="cpu")
prompt_ids = tokenizer.encode("Mini-LLM", add_special_tokens=True)

config = GenerationConfig(
    max_length=60,
    temperature=0.8,
    top_k=40,
    top_p=0.95,
    repetition_penalty=1.1,
)
output_ids = text_generator.sample_generate(
    input_ids=torch.tensor([prompt_ids]),
    config=config
)
print(tokenizer.decode(output_ids[0].tolist()))
```

## 与模型集成的建议
1. 调用推理前请确保模型处于 `eval()` 模式，避免 Dropout 影响结果
2. 当使用 `beam_search` 时，可根据 `config.length_penalty` 调整短序列偏好
3. 对于长文本生成，建议结合 `MemoryOptimizer.optimize_model_for_inference()` 释放多余显存
4. 如需自定义策略（例如多样性惩罚、动态温度），可在 `TextGenerator` 基础上扩展

`TextGenerator` 只依赖 PyTorch 与分词器接口，可轻松移植到其他项目或脚本中。
