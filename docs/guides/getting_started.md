# 🚀 快速上手 Mini-LLM

本指南帮助你在本地环境中完成安装、准备最小数据集，并跑通一次训练与推理流程。

## 环境要求
- Python 3.10+（推荐 3.11，与 `pyproject.toml` 保持一致）
- PyTorch 2.1 及以上版本（包含 CUDA/MPS 支持时可自动检测）
- `pip` 或 `uv` 用于安装依赖

> 想确认设备是否被正确识别，可在克隆仓库后执行：
> ```bash
> python -c "from src.training.pipeline.environment import TrainingEnvironment;\
> from config.training_config import get_config;\
> env=TrainingEnvironment(get_config('tiny'),'sft');print(env.device)"
> ```
> 这段代码会调用 `_setup_device`，输出 `cuda`/`mps`/`cpu`，同时在工作目录生成配置快照，验证环境写权限。【F:src/training/pipeline/environment.py†L12-L63】

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

脚本内部会构建 `MiniGPTTrainer`，它将 `TrainingEnvironment`、`DatasetPreparer`、`TrainingLoopRunner` 等模块串联起来，并在输出目录中持久化配置快照与数据集统计，方便追踪实验。【F:src/training/pipeline/app.py†L25-L204】【F:src/training/pipeline/data_manager.py†L24-L199】

运行结束后，检查 `checkpoints/<mode>_<config>/`：

- `training_config_snapshot.json`：记录当次运行的所有超参，便于复现。【F:src/training/pipeline/environment.py†L32-L63】
- `dataset_stats.json`：列出每个数据文件的原始/采样数量，以及验证集占比，帮助排查样本不足。【F:src/training/pipeline/environment.py†L56-L63】【F:src/training/pipeline/data_manager.py†L146-L209】
- `regression/`：若启用了回归评估，会保存固定提示的通过率，可用于对齐回归检查。【F:src/training/pipeline/regression_suite.py†L22-L147】

## 最小化教学示例
若你想快速理解底层训练循环，可以使用下列少量代码复现一个语言模型 batch 的训练：

```python
import torch
from torch.utils.data import DataLoader

from src.model.config import get_tiny_config
from src.model.transformer import MiniGPT
from src.tokenizer.bpe_tokenizer import BPETokenizer
from src.training.datasets import LanguageModelingDataset
from src.training.trainer import PreTrainer

texts = [
    "你好，Mini-LLM!",
    "Transformer 架构演示",
    "小模型也能训练",
]

tokenizer = BPETokenizer(vocab_size=256)
tokenizer.train(texts)

dataset = LanguageModelingDataset(texts, tokenizer, max_length=64)
dataloader = DataLoader(dataset, batch_size=2)

config = get_tiny_config()
model = MiniGPT(config)
trainer = PreTrainer(model, tokenizer, device="cpu")

loss = trainer.train_epoch(dataloader)
print(f"epoch loss: {loss:.4f}")
```

示例中各行代码与核心模块的对应关系：

- `get_tiny_config()` 返回约 1M 参数的默认配置，自动验证注意力头与隐藏维度的整除关系，适合 CPU 快速实验。【F:src/model/config.py†L159-L176】
- `MiniGPT(config)` 会根据配置选择是否启用 RoPE、GQA 以及 MoE，并构建完整的 Transformer Decoder。【F:src/model/transformer.py†L314-L443】
- `BPETokenizer` 在 `train(texts)` 时完成中文友好预处理与 BPE 合并，默认注册 `<PAD>/<UNK>/<BOS>/<EOS>` 四个特殊符号，对应的 ID 会在数据集和模型中复用。【F:src/tokenizer/bpe_tokenizer.py†L77-L199】
- `LanguageModelingDataset` 会在截断/填充后返回 `(input, target, loss_mask)`，完全复用 MiniMind 的预训练样本结构，loss mask 自动忽略 PAD 区域。【F:src/training/datasets/language_modeling.py†L11-L115】
- `PreTrainer` 初始化时绑定 `AdamW` 与余弦退火调度器，`train_epoch` 会在每个 batch 上执行前向、反向与梯度裁剪，演示最小化训练闭环。【F:src/training/trainer.py†L13-L204】

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
- **显存不足**：在命令行加入 `--batch-size` 或调小配置中的 `gradient_accumulation_steps`；同时确保 `MINIGPT_MEMORY_THRESHOLD` 设置合理以触发自动清理。【F:config/training_config.py†L124-L205】【F:src/training/pipeline/memory_hooks.py†L1-L78】
- **训练被中断**：若终端提示“收到中断信号”，训练器会保存最新 checkpoint，可用 `--auto-resume` 无缝继续。【F:src/training/pipeline/app.py†L118-L186】
- **输出退化**：启用回归评估或检查 `TrainingMonitor` 的梯度异常提示，必要时降低学习率或延长 warmup。【F:src/training/training_monitor.py†L120-L222】【F:src/training/pipeline/regression_suite.py†L22-L147】

## 下一步
- 阅读 [model.md](model.md) 理解 `MiniGPTConfig` 可配置项
- 阅读 [data.md](data.md) 了解真实数据集需要遵循的字段格式
- 阅读 [training.md](training.md) 掌握混合精度、梯度累积、检查点等高级特性

至此，你已经可以在本地复现实验并对框架进行二次开发。
