# Mini-LLM 训练框架

Mini-LLM 是一个面向教学与原型验证的小型语言模型训练框架。代码全部采用 PyTorch 实现，聚焦于让开发者快速理解并实验 Transformer 架构、数据流水线、训练循环以及 RLHF 相关组件。

## ✨ 项目亮点
- **模块化 Transformer 实现**：`MiniGPTConfig` 与 `MiniGPT` 提供可配置的隐藏层数、注意力头数、RoPE、GQA、SwiGLU、MoE 等现代组件，便于按需裁剪模型规模与特性。
- **数据与分词支持**：`ConversationDataLoader`/`PretrainDataLoader`/`DPODataLoader` 覆盖监督微调、预训练与偏好数据三种常见格式；`TokenizerManager` 与 `BPETokenizer` 支持中文友好的 BPE 训练和缓存复用。
- **训练工具链**：`PreTrainer` 与 `SFTTrainer` 等封装了损失计算、梯度裁剪、Cosine 调度器、混合精度、梯度累积等训练常见逻辑，`MemoryConfig`/`MemoryMonitor` 聚焦显存与内存优化。
- **推理与评估**：`TextGenerator` 提供贪心、Top-k、Top-p、Beam Search 等生成策略；`benchmarks/performance_benchmark.py` 可用于快速评估不同配置的性能。
- **RLHF 管道雏形**：`RLHFPipeline` 串联监督微调、奖励模型训练与 PPO 策略优化，展示 RLHF 端到端流程的关键步骤。

## 📁 仓库结构
```
mini-llm/
├── data/                    # 示例数据及配置
├── docs/                    # 项目文档（见下文）
├── src/
│   ├── benchmarks/          # 性能基准脚本
│   ├── data/                # 数据加载与切分
│   ├── inference/           # 文本生成工具
│   ├── model/               # 模型与配置实现
│   ├── rl/                  # 奖励模型与 PPO
│   ├── tokenizer/           # BPE 分词器与管理器
│   └── training/            # 训练循环与内存优化
├── tokenizers/              # 已训练分词器缓存
├── utils/                   # 预留工具模块
└── test_lightweight_monitor.py
```

## 🚀 快速开始
1. **安装依赖**
   ```bash
   git clone https://github.com/your-org/mini-llm.git
   cd mini-llm
   pip install -e .
   ```

2. **训练最小示例**（CPU 上运行一个 batch 演示完整流程）
   ```python
   from torch.utils.data import DataLoader

   from src.model.config import get_tiny_config
   from src.model.transformer import MiniGPT
   from src.tokenizer.bpe_tokenizer import BPETokenizer
   from src.training.trainer import LanguageModelingDataset, PreTrainer

   texts = ["你好，Mini-LLM!", "Transformer 架构演示", "小模型也能训练"]

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

3. **文本生成**
   ```python
   import torch
   from src.inference.generator import TextGenerator, GenerationConfig

   generator = TextGenerator(model, tokenizer)
   prompt_ids = tokenizer.encode("Mini-LLM", add_special_tokens=True)
   output_ids = generator.sample_generate(
       input_ids=torch.tensor([prompt_ids]),
       config=GenerationConfig(max_length=40, top_p=0.9)
   )
   print(tokenizer.decode(output_ids[0].tolist()))
   ```

## 📚 深入阅读
- [docs/README.md](docs/README.md)：文档索引与阅读指引
- [docs/getting_started.md](docs/getting_started.md)：环境配置与实践示例
- [docs/model.md](docs/model.md)：模型与配置说明
- [docs/data.md](docs/data.md)：数据与分词流程
- [docs/training.md](docs/training.md)：训练循环与内存优化
- [docs/inference.md](docs/inference.md)：推理策略与配置
- [docs/rlhf.md](docs/rlhf.md)：RLHF 流程概览与扩展思路

欢迎在阅读源码的同时配合文档理解每个组件的职责，便于根据自身需求进行裁剪或扩展。
