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
   > `uv run python scripts/train.py`：调用顶层 CLI，负责拼装环境并执行训练主程序。【F:scripts/train.py†L1-L21】
   > `--mode sft`：选择监督微调流程，CLI 会自动带入分阶段默认的学习率与 warmup 策略，避免直接覆盖预训练表征。【F:src/training/pipeline/cli.py†L14-L83】
   > `--config medium`：载入中等规模的 `BaseConfig` 派生体，决定模型宽深、批量大小以及优化器超参。【F:config/training_config.py†L355-L413】
   > `--auto-resume`：触发自动检查点恢复逻辑，便于因中断而续训。【F:src/training/pipeline/cli.py†L88-L119】

   **想要快速体验？** 项目提供 Makefile 快捷命令用于常见训练场景：

   ```bash
   make train-sft        # 使用 small 配置运行监督微调，会自动重训分词器
   make train-pretrain   # 使用 medium 配置执行预训练流程
   make train-dpo        # 依赖已完成的 SFT 权重，启动 DPO 训练
   ```
   > 这些命令封装了常见的训练参数组合，可在初次体验或演示场景下直接复用；若需更细粒度控制，可继续使用上方的 CLI 参数自定义运行。【F:Makefile†L73-L107】

### 命令行字段详解

> `--mode {pretrain,sft,dpo,rlhf}`：控制训练阶段，内部 `apply_mode_defaults` 会针对不同目标调整最大步数、warmup 以及学习率。例如预训练保持 1e-4 学习率与 5% warmup，SFT 则降低学习率保护语言能力，RLHF 则进一步缩小步长以稳定 PPO。【F:src/training/pipeline/cli.py†L41-L82】
>
> `--config <size>`：映射到 `config/training_config.py` 中的配置类，涵盖模型维度、批量大小、优化器参数与生成策略。不同配置依据显存大小动态调整梯度累积与 DataLoader 参数，保证在教学环境中也能复现完整流程。【F:config/training_config.py†L202-L340】
>
> `--retrain-tokenizer`：强制执行 `TokenizerManager.setup` 内的重新训练逻辑，以便在数据集发生变化时更新词表，避免旧词表导致的未登录词问题。【F:src/training/pipeline/app.py†L39-L67】
>
> `--resume/--auto-resume`：与 `CheckpointManager` 协作，实现手动或自动加载最近检查点。恢复时会同步优化器与调度器状态，确保学习率曲线连续。【F:src/training/pipeline/app.py†L93-L150】
>
> `--learning-rate/--max-steps/--batch-size/--warmup-steps`：用于覆盖配置中的默认超参，CLI 会在应用模式默认后再处理这些覆盖，保证命令行显式值优先生效。【F:src/training/pipeline/cli.py†L98-L121】

> 💡 **为什么要区分这些字段？** 通过 CLI 控制训练阶段与模型规模可以让教学场景快速切换实验条件，同时保留合理的默认值以防止因参数设置不当导致发散或 OOM；而恢复与覆盖参数的能力则满足真实研发中的迭代需求。

#### 常见进阶覆写

- **环境变量热补丁**：`BaseConfig` 会在初始化时读取 `MINIGPT_TRAIN_SEED`、`MINIGPT_VAL_SPLIT`、`MINIGPT_MEMORY_THRESHOLD` 等环境变量，无需改动源码即可调整随机种子、验证集比例与内存阈值。【F:config/training_config.py†L118-L209】
- **数据配额控制**：在 JSONL 同名匹配的前提下，可通过 `dataset_sampling` 为特定文件指定采样比例与验证集占比，便于在组合多源数据时保持类别平衡；若希望统一缩放全部语料，可设置 `MINIGPT_GLOBAL_SAMPLE_RATIO`（默认 `0.5`）快速降低预处理样本量。【F:config/training_config.py†L124-L182】【F:src/training/pipeline/data_manager.py†L92-L212】
- **回归测试频率**：设置 `MINIGPT_REGRESSION_INTERVAL=0` 将使得 Regression Suite 每轮评估都执行，适合演示如何捕获指令退化；设置较大的间隔可降低训练开销。【F:config/training_config.py†L181-L206】【F:src/training/pipeline/regression_suite.py†L22-L87】

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

   > `from torch.utils.data import DataLoader`：引入 PyTorch 数据管道组件，用于批量化迭代样本。【F:src/training/trainer.py†L9-L58】
   > `from src.model.config import get_tiny_config`：拉取教学用最小配置，内部开启 GQA、RoPE 以演示现代结构选择的影响。【F:src/model/config.py†L159-L175】
   > `from src.model.transformer import MiniGPT`：导入 Transformer 主体，支持在不同配置间复用同一实现。【F:src/model/transformer.py†L314-L440】
   > `from src.tokenizer.bpe_tokenizer import BPETokenizer`：使用项目自带 BPE 分词器，便于快速训练新词表。【F:src/tokenizer/bpe_tokenizer.py†L1-L196】
   > `from src.training.datasets import LanguageModelingDataset`：选择基础语言建模数据集封装，自动补齐 PAD 并返回 `(X, Y, loss_mask)` 三元组，直接对齐 MiniMind 预训练损失。【F:src/training/datasets/language_modeling.py†L11-L123】
   > `from src.training.trainer import PreTrainer`：载入轻量训练循环，包含优化器、调度器与损失封装，方便课堂演示。【F:src/training/trainer.py†L13-L200】
   > `texts = [...]`：准备极小的原始语料，演示数据格式要求；数据集内部会自动处理 `dict`/`str` 并统一为文本输入。【F:src/training/datasets/language_modeling.py†L22-L63】
   > `tokenizer = BPETokenizer(vocab_size=256)`：实例化小词表，便于快速拟合；示例中 256 词表减少内存压力。【F:src/tokenizer/bpe_tokenizer.py†L25-L140】
   > `tokenizer.train(texts)`：直接在示例文本上拟合分词模型，展示离线训练流程的接口形式。【F:src/tokenizer/bpe_tokenizer.py†L76-L140】
   > `dataset = LanguageModelingDataset(...)`：将纯文本包装成固定长度 token 序列，执行截断/填充后产出 `(input, target, loss_mask)`。【F:src/training/datasets/language_modeling.py†L39-L115】
   > `dataloader = DataLoader(...)`：组建批次并在迭代时触发 `__getitem__`，可配合更多参数实现打乱或多进程加载。【F:src/training/trainer.py†L56-L154】
   > `config = get_tiny_config()`：获取模型超参（层数、头数、上下文长度等），确保 `MiniGPT` 正确初始化权重矩阵。【F:src/model/config.py†L159-L175】
   > `model = MiniGPT(config)`：构建模型实例；内部根据配置拼装注意力、前馈与嵌入层。【F:src/model/transformer.py†L314-L440】
   > `trainer = PreTrainer(model, tokenizer, device="cpu")`：封装优化器、调度器与损失，默认使用 AdamW + 余弦退火并将模型迁移到 CPU。【F:src/training/trainer.py†L16-L52】
   > `loss = trainer.train_epoch(dataloader)`：执行单轮训练，返回平均损失，内部自动生成右移标签并裁剪梯度。【F:src/training/trainer.py†L56-L154】
   > `print(f"epoch loss: {loss:.4f}")`：输出观测指标，帮助确认训练是否正常下降。【F:src/training/trainer.py†L127-L129】

   > ✅ **为什么逐行解释？** 初学者可将此示例作为调试脚本，快速理解模型、分词器、数据集与训练循环之间的协作关系，从而在迁移到完整管道时少走弯路。

## ✅ 功能测试（Feature Tests）

为确保最新特性在本地环境中稳定运行，可先安装测试依赖并执行内置的 PyTest 套件：

```bash
uv pip install .[test]
uv run pytest
```

测试脚本覆盖了 RoPE、GQA、深窄架构等核心模型特性以及训练/推理端到端流程，全部通过即表示主要功能均可跑通。【F:scripts/tests/test_architecture.py†L1-L320】【F:scripts/tests/test_training_inference.py†L1-L450】

### 🧪 烟雾测试：最小化运行完整训练与推理

若希望在本地快速确认整个流水线（预训练→SFT→DPO→RLHF→推理）能够跑通，可执行内置的烟雾测试脚本：

```bash
uv run python scripts/run_smoke_pipeline.py
```

脚本会自动准备一套极小的合成数据集，禁用 manifest 配置后依次运行四个训练阶段，并在最后使用 RLHF checkpoint 做一次单轮文本生成，以验证训练与推理链路协同正常。【F:scripts/run_smoke_pipeline.py†L1-L231】

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

5. **一键行业评测**
   ```bash
   uv run python scripts/run_benchmark_eval.py --checkpoint path/to/checkpoint_step_*.pt
   ```
   > 默认自动在与 checkpoint 同目录的 `tokenizer/tokenizer.json` 上下文中加载模型与分词器，并跑通 WikiText-2、LAMBADA、HellaSwag、ARC、Winogrande、PIQA、BoolQ 等评测任务，结果会写入同目录下带时间戳的 `benchmark_results_*.json`。如需自定义任务或缓存目录，可通过 `--tasks`、`--cache-dir`、`--disable-auto-download` 等参数覆盖默认值。【F:scripts/run_benchmark_eval.py†L24-L134】【F:scripts/run_benchmark_eval.py†L155-L210】

## 📚 深入阅读
- [docs/README.md](docs/README.md)：文档索引与阅读指引
- [docs/guides/getting_started.md](docs/guides/getting_started.md)：环境配置与实践示例
- [docs/guides/dataset_preparation.md](docs/guides/dataset_preparation.md)：数据清洗、采样与格式规范
- [docs/guides/model.md](docs/guides/model.md)：模型与配置说明
- [docs/guides/data.md](docs/guides/data.md)：数据与分词流程
- [docs/guides/training.md](docs/guides/training.md)：训练循环与内存优化
- [docs/guides/inference.md](docs/guides/inference.md)：推理策略与配置
- [docs/guides/rlhf.md](docs/guides/rlhf.md)：RLHF 流程概览与扩展思路
- [docs/research/minimind_comparison.md](docs/research/minimind_comparison.md)：与 MiniMind 的对比分析
- [docs/planning/training_optimization_plan.md](docs/planning/training_optimization_plan.md)：训练优化路线图

欢迎在阅读源码的同时配合文档理解每个组件的职责，便于根据自身需求进行裁剪或扩展。

## 🔁 训练阶段流程与设计动机

### 1. 预训练（Pretrain）
- **流程**：`MiniGPTTrainer` 会按照 `BaseConfig` 设定的数据目录加载海量无监督语料，训练目标为下一个 token 预测。【F:src/training/pipeline/app.py†L25-L162】【F:config/training_config.py†L105-L145】
- **关键设置**：CLI 默认学习率 1e-4 与 5% warmup，配合余弦退火调度，保持稳定学习曲线；混合精度与梯度检查点自动启用以降低显存占用。【F:src/training/pipeline/cli.py†L41-L64】【F:src/training/pipeline/app.py†L106-L150】
- **动机**：建立基础语言建模能力，为后续下游任务提供通用表征。如果跳过该阶段，SFT/RLHF 将缺乏知识语料支撑，收敛质量明显下降。

### 2. 监督微调（SFT）
- **流程**：在预训练权重基础上加载对话/任务数据（`ConversationDataset`），优化为条件生成任务，并在 CLI 中降低学习率、缩短 warmup。【F:src/training/datasets/conversation.py†L10-L145】【F:src/training/pipeline/cli.py†L65-L83】
- **关键设置**：使用较小步长（5e-5）与角色标记，引导模型遵循提示结构；`DatasetPreparer` 会根据角色 token 与增强策略平衡不同来源数据，减少灾难性遗忘。【F:src/training/pipeline/data_manager.py†L20-L188】【F:config/training_config.py†L147-L159】
- **动机**：将通用语言能力对齐到指令遵循任务，为 RLHF 提供更稳定的初始策略。直接在预训练模型上做 RL 会造成奖励模型难以学习有意义的偏好差异。

### 3. 奖励模型训练（Reward Modeling, RLHF 中间阶段）
- **流程**：`RLHFPipeline` 读取偏好数据，通过 `create_reward_model` 与 `create_reward_trainer` 拟合人类排序信号，并可选择冻结骨干以防退化。【F:src/rl/rlhf_pipeline.py†L37-L200】【F:src/rl/reward_model/reward_trainer.py†L1-L220】
- **关键设置**：奖励模型通常复制 SFT 权重并仅训练顶部头部；较小学习率与梯度裁剪避免过拟合，同时定期保存配置供后续 PPO 使用。【F:src/rl/rlhf_pipeline.py†L37-L200】
- **动机**：让强化学习阶段拥有可微的偏好信号。如果跳过奖励模型，将无法通过 PPO 量化响应好坏。

### 4. 强化学习（RL / PPO）
- **流程**：`RLHFPipeline` 内的 `create_ppo_trainer` 结合策略模型、价值模型与奖励模型执行 PPO，循环采样-评估-更新。【F:src/rl/ppo/ppo_trainer.py†L1-L220】【F:src/rl/rlhf_pipeline.py†L59-L200】
- **关键设置**：策略学习率设置为 1e-5、价值网络 3e-4，并通过 mini-batch 分解与 KL 奖励稳定训练；同时保持短 warmup，防止策略剧烈偏移。【F:src/rl/rlhf_pipeline.py†L59-L200】
- **动机**：在 SFT 基础上进一步优化响应质量，使模型在对话中更贴近人类偏好；PPO 的剪切目标可防止策略崩溃，是教学中易于解释的 RLHF 算法。

## 🔍 实验追踪与复现

- **配置快照**：`TrainingEnvironment` 会在启动时将 `BaseConfig` 展平成 JSON 并保存到 `training_config_snapshot.json`，即使后续修改默认值也能回溯当时的完整配置。【F:src/training/pipeline/environment.py†L12-L63】
- **数据统计**：每次构建数据集时都会记录原始样本量、采样后数量及验证集占比，并写入 `dataset_stats.json`；若某个数据源被采样为 0 条会在日志中立即提醒。【F:src/training/pipeline/app.py†L52-L64】【F:src/training/pipeline/data_manager.py†L92-L209】
- **训练监控**：默认启用的 `TrainingMonitor` 会跟踪损失、学习率、梯度范数与异常检测，必要时还可记录激活统计和回归测试结果，便于课堂演示常见训练波动。【F:src/training/training_monitor.py†L120-L222】
- **回归评估**：配置 `regression_eval_enabled=True` 后，训练循环会定期运行固定提示检查指令性能并将通过率写入 `regression/`，帮助快速识别对齐退化问题。【F:config/training_config.py†L181-L206】【F:src/training/pipeline/regression_suite.py†L22-L147】

## 🛡️ 稳定性与资源管理策略

- **学习率调度**：`MiniGPTTrainer` 统一使用 warmup + 余弦退火；恢复训练时会回放调度器状态并提示当前阶段，避免学习率突变导致震荡。【F:src/training/pipeline/app.py†L65-L160】
- **梯度累积与裁剪**：`TrainingLoopRunner` 按配置执行梯度累积并在每次优化步记录梯度范数、平均损失，为检测梯度爆炸提供第一手指标。【F:src/training/pipeline/training_loop.py†L34-L123】
- **内存守护**：`MemoryHooks` 基于阈值触发缓存清理或在 OOM 时立即释放显存，且会输出 allocator 配置，帮助在低显存环境中稳定运行。【F:config/training_config.py†L173-L205】【F:src/training/pipeline/memory_hooks.py†L1-L78】
- **中断恢复**：信号处理器捕获 Ctrl+C 后会优雅保存 checkpoint，并提示使用 `--auto-resume` 继续训练，降低课堂/实验中断带来的风险。【F:src/training/pipeline/app.py†L118-L186】

## ⚠️ 注意事项
- **数据质量**：预训练需要覆盖多领域语料，SFT/奖励阶段则应保证指令多样性与偏好标注一致，否则奖励模型会学习到噪声偏好。【F:config/training_config.py†L105-L173】【F:src/rl/rlhf_pipeline.py†L47-L200】
- **资源限制**：根据 GPU 显存自动调整批量与梯度累积；在资源较弱的机器上可通过 CLI 降低 `--max-steps` 或切换 `tiny`/`small` 配置。【F:config/training_config.py†L181-L340】【F:src/training/pipeline/cli.py†L41-L121】
- **安全中断**：训练过程中按 `Ctrl+C` 触发信号处理器，`MiniGPTTrainer` 会先保存检查点再退出，避免长时间训练成果丢失。【F:src/training/pipeline/app.py†L118-L150】
- **监控与调试**：默认启用 `TrainingMonitor` 与回归测试，建议定期查看 TensorBoard 与回归结果，快速定位损失震荡或输出退化问题。【F:src/training/pipeline/app.py†L69-L162】

> 💬 **为什么强调这些注意事项？** 真实研发中最常见的问题来自数据噪声、算力不足与中断恢复。提前在 README 中明确，可帮助读者在有限时间里优先规避高风险操作。
