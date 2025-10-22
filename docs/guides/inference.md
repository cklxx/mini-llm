# 🗣️ 推理与文本生成

Mini-LLM 的推理组件位于 `src/inference/generator.py`。`TextGenerator` 在保持 API 简洁的同时，提供多种主流的解码策略，并在初始化时将模型移动到目标设备、切换到 `eval()` 模式，避免训练态副作用。【F:src/inference/generator.py†L27-L44】

## GenerationConfig 字段详解

`GenerationConfig` 为所有生成策略提供统一入口，表格列出了常用字段及其影响。【F:src/inference/generator.py†L12-L25】

| 字段 | 作用 | 调参建议 |
| ---- | ---- | -------- |
| `max_length` | 控制追加 token 的最大步数；到达上限即停止 | 训练阶段若上下文很长，可使用较小的 `max_length` 保持响应简短 |
| `temperature` | 对 logits 缩放，温度越高越随机 | 指令微调模型常用 `0.7~0.9`，若输出过于发散可下降 |
| `top_k` / `top_p` | 限制候选 token 数量或累计概率，提升多样性 | 与 `temperature` 联合使用；生成事实类回答时可调低 |
| `repetition_penalty` | 对已生成 token 的 logits 进行惩罚，防止复读 | 大于 1 会降低重复概率；过高可能导致语句不连贯【F:src/inference/generator.py†L45-L78】 |
| `num_beams` / `early_stopping` | 束搜索宽度与停止条件 | `num_beams>1` 时自动走 `beam_search` 分支，并在所有 beam 生成 EOS 时提前结束【F:src/inference/generator.py†L167-L225】 |
| `do_sample` | 是否启用采样模式 | False 时在 `sample_generate` 内会退化为 argmax，相当于贪心解码【F:src/inference/generator.py†L125-L166】 |

## 解码流程拆解

- **贪心搜索**：循环前向推理并选择最大概率 token，适合验证或确定性回复。【F:src/inference/generator.py†L103-L124】
- **随机采样**：先按温度缩放 logits，再依次应用重复惩罚、Top-k/Top-p 过滤，并调用 `torch.multinomial` 采样；若 `do_sample=False` 则会改为取最大概率，兼容采样和确定性场景。【F:src/inference/generator.py†L125-L166】
- **束搜索**：维护多个候选 beam，计算累计对数概率并实时更新最佳序列；当 `early_stopping=True` 且所有 beam 都生成 EOS 时提前返回。【F:src/inference/generator.py†L167-L225】
- **辅助工具**：`apply_repetition_penalty` 对已出现 token 调整 logits，`top_k_filtering`/`top_p_filtering` 负责裁剪不符合条件的候选。【F:src/inference/generator.py†L45-L102】

调用 `TextGenerator.generate` 时，会根据配置自动选择上述策略并在结束后解码为字符串，避免重复手动分支判断。【F:src/inference/generator.py†L227-L245】

### 批量与增量推理

- `sample_generate` 支持输入 `[batch, seq_len]` 的张量，可一次性对多条 prompt 生成回复；在教学场景中可直接用 `torch.stack` 合并批量以演示批处理效果。【F:src/inference/generator.py†L125-L166】
- 若希望在循环外逐 token 推理，可以参考 `TextGenerator.stream_generate` 等增量生成接口，在每步选择 argmax 并将新 token 拼接到输入中，演示了“增量 decode” 的写法。【F:src/inference/generator.py†L200-L262】
- 结合训练阶段保存的 `GenerationConfig` 默认值，可通过 `MiniGPT.generate` 快速对比两种实现的输出差异，帮助学生理解封装层的价值。【F:src/model/transformer.py†L451-L500】

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

1. `TextGenerator` 初始化阶段已经调用 `model.eval()`，若在外部切换回训练模式请在推理前再次设为 `eval()`，否则 Dropout 会带来随机噪声。【F:src/inference/generator.py†L38-L44】
2. 束搜索默认不会应用长度惩罚，可根据需求在 `GenerationConfig` 中添加 `length_penalty` 字段并在自定义分支内使用，避免 beam 偏向短句。【F:src/inference/generator.py†L167-L225】
3. 长文本生成时建议搭配 `MemoryOptimizer.optimize_model_for_inference()`，释放梯度缓存减少显存占用。【F:src/training/memory_optimizer.py†L21-L176】
4. 如需集成新的控制策略（多样性惩罚、动态温度等），可以继承 `TextGenerator` 并重写 `sample_generate`，复用现有的重复惩罚与过滤逻辑。

## 服务化与调试提示

- `TextGenerator.chat` 简化了多轮对话上下文的构造，适合作为 HTTP/CLI demo 的快速入口；真实服务中可以替换为结构化的对话历史并配合 `TokenizerManager` 统一特殊 token。【F:src/inference/generator.py†L227-L318】【F:src/training/pipeline/tokenizer_manager.py†L14-L118】
- 推理脚本若需要与训练阶段保持一致的监控指标，可复用 `TrainingMonitor.get_gradient_norm` 等工具在推理后检查权重变化是否异常。【F:src/training/training_monitor.py†L120-L170】
- 为保证再现性，建议在推理前载入训练目录下的 `training_config_snapshot.json`，将其中的 `max_generate_length`、`temperature` 等字段同步到 `GenerationConfig`，避免训练/推理参数不一致导致输出偏差。【F:src/training/pipeline/pipeline.py†L41-L79】【F:src/model/config.py†L55-L115】

`TextGenerator` 只依赖 PyTorch 与分词器接口，可轻松移植到其他项目或脚本中；在服务化场景中可结合批量前向和 KV Cache 进一步提升吞吐。
