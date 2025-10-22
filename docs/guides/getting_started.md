# 🚀 快速上手 Mini-LLM

本指南帮助你在本地环境中完成安装、准备最小数据集，并跑通一次训练与推理流程。

## 环境要求
- Python 3.10+（推荐 3.11，与 `pyproject.toml` 保持一致）
- PyTorch 2.1 及以上版本（包含 CUDA/MPS 支持时可自动检测）
- `pip` 或 `uv` 用于安装依赖

> 想确认设备是否被正确识别，可在克隆仓库后执行：
> ```bash
> python -c "from config.training_config import get_config;\
> from src.training.pipeline.pipeline import TrainingPipeline;\
> pipeline=TrainingPipeline(get_config('tiny'),'sft');print(pipeline.device)"
> ```
> 这段代码会输出 `cuda`/`mps`/`cpu` 并生成配置快照，验证环境写权限。【F:src/training/pipeline/pipeline.py†L23-L79】

## 安装步骤
```bash
git clone https://github.com/your-org/mini-llm.git
cd mini-llm
pip install -e .
```
> 如果你使用 `uv`，可以在仓库根目录执行 `uv sync`。

## 运行标准训练流水线
推荐通过脚本入口复现实验，它会自动解析配置、准备分词器、采样数据并保存检查点。

```bash
uv run python scripts/train.py --mode sft --config medium --auto-resume
```

常用参数：

- `--mode {pretrain,sft,dpo,rlhf}`：切换训练阶段；
- `--retrain-tokenizer`：强制重新训练并覆盖现有分词器；
- `--resume` / `--auto-resume`：从指定或最新检查点恢复训练；
- `--learning-rate`、`--batch-size`、`--warmup-steps`：命令行覆盖配置文件数值。【F:src/training/pipeline/cli.py†L12-L151】

脚本内部会构建 `TrainingPipeline`，它将 `DatasetPreparer`、`TrainingLoopRunner` 等模块串联起来，并在输出目录中持久化配置快照与数据集统计，方便追踪实验。【F:src/training/pipeline/pipeline.py†L23-L213】【F:src/training/pipeline/data_manager.py†L24-L199】

运行结束后，检查 `checkpoints/<mode>_<config>/`：

- `training_config_snapshot.json`：记录当次运行的所有超参，便于复现。【F:src/training/pipeline/pipeline.py†L41-L79】
- `dataset_stats.json`：列出每个数据文件的原始/采样数量，以及验证集占比，帮助排查样本不足。【F:src/training/pipeline/pipeline.py†L81-L125】【F:src/training/pipeline/data_manager.py†L146-L209】

## 最小化教学示例
若你想快速观察底层训练循环的每个阶段，推荐直接运行内置的全栈调试脚本：

```bash
uv run python scripts/debug_fullstack.py --mode pretrain --model-size tiny --prompt "你好，MiniGPT！"
```

脚本会自动：

- 载入 `TrainingPipeline` 并保存配置快照，确保任何调试都有完整上下文可追溯。【F:scripts/debug_fullstack.py†L1-L214】【F:src/training/pipeline/pipeline.py†L23-L213】
- 打印首个 batch 的原始文本、token ID、loss mask 等关键张量形状，为数据检查和异常定位提供依据。【F:scripts/debug_fullstack.py†L86-L140】【F:src/training/datasets/language_modeling.py†L11-L115】
- 复用正式训练循环的损失与梯度逻辑执行一次优化步骤，并输出梯度范数；随后以给定 `--prompt` 做推理，形成闭环。【F:scripts/debug_fullstack.py†L146-L189】【F:src/training/pipeline/training_loop.py†L18-L620】

若需在 Notebook 中手写代码，也可仿照脚本中对 `TrainingPipeline` 的使用方式，手动获取 `DatasetPreparer`、`TrainingLoopRunner` 等组件执行自定义实验。

## 推理示例
训练结束后可以直接复用模型进行文本生成：

```python
import torch
from src.inference.generator import TextGenerator, GenerationConfig

prompt = "Mini-LLM"
prompt_ids = tokenizer.encode(prompt, add_special_tokens=True)

text_generator = TextGenerator(model, tokenizer, device="cpu")
output_ids = text_generator.sample_generate(
    input_ids=torch.tensor([prompt_ids]),
    config=GenerationConfig(max_length=40, top_p=0.9, temperature=0.8)
)
print(tokenizer.decode(output_ids[0].tolist()))
```

`TextGenerator.sample_generate` 会在采样前应用温度缩放、重复惩罚、Top-k/Top-p 过滤，并根据 `do_sample` 自动选择采样或贪心策略；如需批量推理可直接传入多个 `prompt_ids`。【F:src/inference/generator.py†L125-L166】

## 常见问题排查

- **数据未被加载**：确认数据文件位于 `config.data_dir` 或 `MINIGPT_DATA_DIR` 所指向的目录，`DataResolver` 会按模式查找并打印缺失警告。【F:src/training/pipeline/data_manager.py†L28-L124】
- **显存不足**：在命令行加入 `--batch-size` 或调小配置中的 `gradient_accumulation_steps`；必要时缩短 `max_steps` 以快速验证流程。【F:config/training_config.py†L124-L205】
- **训练被中断**：若终端提示“收到中断信号”，训练器会保存最新 checkpoint，可用 `--auto-resume` 无缝继续。【F:src/training/pipeline/pipeline.py†L214-L230】
- **输出波动**：检查 `TrainingMonitor` 的梯度异常提示并适当降低学习率或延长 warmup。【F:src/training/training_monitor.py†L120-L470】

## 下一步
- 阅读 [model.md](model.md) 理解 `MiniGPTConfig` 可配置项
- 阅读 [data.md](data.md) 了解真实数据集需要遵循的字段格式
- 阅读 [training.md](training.md) 掌握混合精度、梯度累积、检查点等高级特性

至此，你已经可以在本地复现实验并对框架进行二次开发。
