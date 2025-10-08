# 🚀 快速上手 Mini-LLM

本指南帮助你在本地环境中完成安装、准备最小数据集，并跑通一次训练与推理流程。

## 环境要求
- Python 3.10+（推荐 3.11，与 `pyproject.toml` 保持一致）
- PyTorch 2.1 及以上版本（包含 CUDA/MPS 支持时可自动检测）
- `pip` 或 `uv` 用于安装依赖

## 安装步骤
```bash
git clone https://github.com/your-org/mini-llm.git
cd mini-llm
pip install -e .
```
> 如果你使用 `uv`，可以在仓库根目录执行 `uv sync`。

## 最小训练示例
下面的示例展示如何使用仓库内置组件训练一个最小的语言模型 batch，并记录损失。

```python
import torch
from torch.utils.data import DataLoader

from src.model.config import get_tiny_config
from src.model.transformer import MiniGPT
from src.tokenizer.bpe_tokenizer import BPETokenizer
from src.training.trainer import LanguageModelingDataset, PreTrainer

# 1. 准备训练语料
texts = [
    "你好，Mini-LLM!",
    "Transformer 架构演示",
    "小模型也能训练",
]

# 2. 训练一个最小 BPE 分词器
#    （实际项目中请使用更大的语料并缓存 tokenizer）
tokenizer = BPETokenizer(vocab_size=256)
tokenizer.train(texts)

# 3. 构造数据集与 DataLoader
dataset = LanguageModelingDataset(texts, tokenizer, max_length=64)
dataloader = DataLoader(dataset, batch_size=2)

# 4. 初始化模型与训练器
config = get_tiny_config()
model = MiniGPT(config)
trainer = PreTrainer(model, tokenizer, device="cpu")

# 5. 运行一个 epoch 并查看损失
loss = trainer.train_epoch(dataloader)
print(f"epoch loss: {loss:.4f}")
```

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

## 下一步
- 阅读 [model.md](model.md) 理解 `MiniGPTConfig` 可配置项
- 阅读 [data.md](data.md) 了解真实数据集需要遵循的字段格式
- 阅读 [training.md](training.md) 掌握混合精度、梯度累积、检查点等高级特性

至此，你已经可以在本地复现实验并对框架进行二次开发。
