# Mini-LLM 训练框架

Mini-LLM 是一个面向教学与原型验证的小型语言模型训练框架。代码全部采用 PyTorch 实现，聚焦于让开发者快速理解并实验 Transformer 架构、数据流水线、训练循环以及 RLHF 相关组件。

## ✨ 项目亮点
- **模块化 Transformer 实现**：`MiniGPTConfig` 与 `MiniGPT` 提供可配置的隐藏层数、注意力头数、RoPE、GQA、SwiGLU、MoE 等现代组件，便于按需裁剪模型规模与特性。
- **训练流水线抽象**：`training.pipeline` 中的 `TrainingEnvironment`、`DatasetPreparer`、`TrainingLoopRunner`、`CheckpointManager` 和 `TrainingMonitor` 串联起设备初始化、数据采样、调度器/优化器、验证与早停等流程，可直接通过 `scripts/train.py` 复现完整训练回路。【F:src/training/pipeline/app.py†L25-L162】【F:src/training/pipeline/training_loop.py†L18-L214】
- **数据与分词支持**：`training.datasets` 提供语言建模与对话 SFT 数据集实现，支持角色标记、掩码策略与轮次截断增强；`TokenizerManager` 管理分词器训练与缓存复用，降低重复开销。【F:src/training/datasets/conversation.py†L10-L145】【F:src/training/pipeline/tokenizer_manager.py†L1-L118】
- **监控与实验追踪**：增强版 `TrainingMonitor` 记录训练/验证损失、PPL、系统资源与梯度健康指标，并在训练结束自动生成 TensorBoard 与可视化摘要。【F:src/training/training_monitor.py†L120-L332】
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
│   └── training/
│       ├── datasets/        # 预训练/SFT 数据集实现
│       ├── pipeline/        # 训练环境、数据、训练循环与 CLI
│       ├── memory_optimizer.py
│       └── trainer.py       # 教学用的轻量训练器
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

2. **运行训练流水线**（自动创建输出目录、采样数据并保存检查点）
   ```bash
   uv run python scripts/train.py --mode sft --config medium --auto-resume
   ```
   > 通过 `--mode pretrain` / `--mode dpo` / `--mode rlhf` 切换不同阶段，`--retrain-tokenizer` 可强制重新训练分词器。【F:scripts/train.py†L1-L21】【F:src/training/pipeline/cli.py†L8-L117】

3. **训练最小示例**（保留教学用途，便于理解基础训练循环）
   ```python
   from torch.utils.data import DataLoader

   from src.model.config import get_tiny_config
   from src.model.transformer import MiniGPT
   from src.tokenizer.bpe_tokenizer import BPETokenizer
   from src.training.datasets import LanguageModelingDataset
   from src.training.trainer import PreTrainer

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

4. **文本生成**
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
