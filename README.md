# MiniGPT训练框架

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.4+-red.svg)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![GPU Optimized](https://img.shields.io/badge/GPU-Optimized-green.svg)](#gpu优化)

**完整的大语言模型训练框架，支持预训练、监督微调(SFT)、DPO和RLHF全流程，针对NVIDIA GPU和Apple Silicon优化**

[📚 技术手册](docs/MiniGPT训练深度解析小册/) •
[🚀 快速开始](#-快速开始) •
[🍎 Mac优化版](README_MAC_OPTIMIZED.md) •
[📖 开发笔记](CLAUDE.md)

</div>

## ✨ 项目特色

### 🚀 完整训练流程
- **预训练（Pretrain）**: 从零开始训练语言模型
- **监督微调（SFT）**: 指令跟随和对话能力训练
- **DPO训练**: 直接偏好优化，无需奖励模型
- **RLHF流程**: 基于人类反馈的强化学习

### 🚀 GPU自动优化
- **智能设备检测**: 自动检测NVIDIA GPU、Apple Silicon MPS或CPU
- **动态配置调整**: 根据GPU显存自动调整批量大小和梯度累积
- **PyTorch 2.4优化**: 启用TensorFloat-32、Flash Attention、模型编译等优化
- **混合精度训练**: 支持FP16/BF16混合精度训练，提升性能和节省显存

### 🏗️ 模块化架构
- **可扩展设计**: 清晰的模块划分，易于扩展
- **配置驱动**: 灵活的配置系统支持多种训练场景
- **完整工具链**: 数据处理、训练、推理、评估一体化
- **现代架构**: 支持SwiGLU、GELU、MoE等业界主流技术

### 🤖 现代AI技术
- **RMSNorm优化**: 替代LayerNorm，减少计算量提升训练效率
- **SwiGLU激活函数**: 现代大模型标准激活函数，替代传统ReLU
- **先进架构组件**: 支持Transformer、多头注意力、位置编码等
- **高效训练技术**: 梯度检查点、梯度累积、学习率调度等

### 📚 深度技术手册
- **完整知识体系**: 8章技术手册覆盖从数学基础到工程实践
- **理论与实践结合**: 每章包含数学推导、代码实现和应用案例
- **系统性学习路径**: 从基础概念到高级技术的进阶路线
- **生产级指导**: 基于实际项目经验的最佳实践

## 🎯 适用场景

- **学习研究**: 深入理解大模型训练原理
- **快速原型**: 验证想法和算法效果
- **教学演示**: 完整的训练流程展示
- **Mac开发**: 在Mac设备上进行模型训练

## 📚 核心技术手册

> **深度解析手册是本项目的核心文档，从数学原理到工程实践，全面解析大模型训练技术**

### 🧮 [第01章 - 数学基础与理论框架](docs/MiniGPT训练深度解析小册/第01章-数学基础与理论框架/)
- **信息论与概率基础**: 语言建模的数学基础
- **线性代数与矩阵运算**: Transformer的核心数学工具
- **优化理论与梯度下降**: 训练算法的理论基础
- **统计学习理论**: 泛化能力的理论保证

### 🏗️ [第02章 - Transformer核心架构](docs/MiniGPT训练深度解析小册/第02章-Transformer核心架构/)
- **注意力机制数学原理**: 自注意力的完整推导
- **多头注意力子空间分解**: 并行处理的数学原理
- **位置编码几何学**: 序列位置的巧妙编码
- **残差连接与层归一化**: 深度网络训练稳定性

### 📖 [第03章 - 预训练理论与实现](docs/MiniGPT训练深度解析小册/第03章-预训练理论与实现/)
- **语言建模概率基础**: 自回归模型的数学基础
- **自回归建模与因果掩码**: 并行训练的技术实现
- **分词策略与信息压缩**: BPE算法的深度分析
- **优化算法深度解析**: AdamW与学习率调度

### 🎯 [第04章 - 监督微调深度解析](docs/MiniGPT训练深度解析小册/第04章-监督微调深度解析/)
- **任务适应理论框架**: 从预训练到任务特化
- **指令跟随与对话建模**: 人机交互的数学建模
- **损失函数设计与优化**: SFT训练的核心技术
- **评估指标与效果分析**: 微调效果的量化评估

### 🔄 [第05章 - 强化学习人类反馈](docs/MiniGPT训练深度解析小册/第05章-强化学习人类反馈/)
- **RLHF理论与数学基础**: 人类偏好学习的理论
- **奖励建模与偏好学习**: 人类反馈的数学建模
- **PPO算法语言模型微调**: 策略优化的具体实现
- **DPO与替代RLHF方法**: 直接偏好优化技术

### 🎲 [第06章 - 生成与解码策略](docs/MiniGPT训练深度解析小册/第06章-生成与解码策略/)
- **自回归生成数学原理**: 序列生成的概率建模
- **经典解码算法深度解析**: Greedy、Beam Search等算法
- **高级采样策略与控制**: Top-k、Top-p、Temperature采样
- **生成质量控制与优化**: 生成文本的质量评估与控制

### 📊 [第07章 - 评估与分析方法](docs/MiniGPT训练深度解析小册/第07章-评估与分析方法/)
- **自动评估指标深度解析**: BLEU、ROUGE、困惑度等指标
- **人类评估框架设计**: 主观评估的标准化方法
- **错误分析与诊断技术**: 模型失败案例的系统分析
- **基准测试与比较分析**: 标准数据集上的性能对比

### 🔧 [第08章 - 工程实践与优化](docs/MiniGPT训练深度解析小册/第08章-工程实践与优化/)
- **训练基础设施与可扩展性**: 分布式训练的工程实现
- **性能优化技术**: 内存优化、计算加速、混合精度
- **部署与生产系统**: 模型服务化的完整方案
- **监控与维护**: 生产环境的模型监控与更新

> **💡 提示**: 技术手册采用理论与实践相结合的方式，每章都包含详细的数学推导、代码实现和实际应用案例。强烈建议按章节顺序学习，建立完整的知识体系。

## 📋 目录

- [核心技术手册](#-核心技术手册)
- [安装配置](#-安装配置)
- [快速开始](#-快速开始)
- [训练流程](#-训练流程)
- [项目架构](#-项目架构)
- [配置说明](#-配置说明)
- [高级功能](#-高级功能)
- [文档资源](#-文档资源)
- [贡献指南](#-贡献指南)

## 🔧 安装配置

### 环境要求

- **Python**: 3.11+
- **PyTorch**: 2.4+
- **系统**: macOS/Linux/Windows
- **显卡**: NVIDIA GPU (推荐) / Apple Silicon / CPU
- **内存**: 最低4GB，推荐8GB+ (GPU显存根据模型大小调整)

### 方式一：使用UV（推荐）

```bash
# 克隆仓库
git clone https://github.com/your-repo/minigpt-training.git
cd minigpt-training

# 一键设置UV环境
uv sync

# 激活环境
source .venv/bin/activate
```

### 方式二：传统pip安装

```bash
# 安装依赖 (PyTorch 2.4 + NVIDIA GPU优化)
pip install -e .

# 手动安装PyTorch (CUDA版本)
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121

# 或安装CPU版本
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cpu
```

### GPU优化库安装 (可选)

```bash
# Flash Attention (NVIDIA GPU)
pip install flash-attn>=2.6.0

# XFormers (内存优化)
pip install xformers>=0.0.27

# DeepSpeed (分布式训练)
pip install deepspeed>=0.14.0
```

## 🚀 快速开始

### 1️⃣ 环境测试

```bash
# 测试GPU配置和模型创建
python -c "
from config.training_config import get_config
from src.model.transformer import create_model

config = get_config('tiny')
model = create_model(vocab_size=config.vocab_size, model_size='tiny')
print('✅ 环境配置正常，可以开始训练！')
"
```

### 2️⃣ 快速训练测试

```bash
# 运行完整的训练和推理测试
python test_training.py   # 训练测试
python test_inference.py  # 推理测试
```

### 3️⃣ 标准训练流程

#### 训练分词器和模型

```bash
# 方法1: 使用训练脚本
python scripts/train.py --mode sft --config small --retrain-tokenizer

# 方法2: 分步训练
# 1) 训练分词器
python scripts/train_tokenizer.py --vocab_size 10000 --data_path data/dataset/minimind_dataset/sft_mini_512.jsonl

# 2) 训练模型
python scripts/train.py --mode sft --config small
```

#### 监督微调 (SFT)

```bash
# 使用tiny配置快速验证
python scripts/train.py --mode sft --config tiny

# 使用small配置标准训练
python scripts/train.py --mode sft --config small

# 使用medium配置 (需要较大显存)
python scripts/train.py --mode sft --config medium

# 从检查点恢复训练
python scripts/train.py --mode sft --config small --resume checkpoints/checkpoint.pt
```

#### 预训练

```bash
# 预训练模型
python scripts/train.py --mode pretrain --config small

# 使用大数据集预训练
python scripts/train.py --mode pretrain --config medium --data_path data/dataset/minimind_dataset/pretrain_hq.jsonl
```

### 4️⃣ 推理和生成

#### 交互式对话

```bash
# 启动聊天模式
python scripts/generate.py \
    --model-path checkpoints/best_model.pt \
    --tokenizer-path checkpoints/tokenizer.pkl \
    --mode chat

# 示例对话:
# 用户: 你好
# 助手: 你好！有什么我可以帮助您的吗？
```

#### 单次推理

```bash
# 单个问题推理
python scripts/generate.py \
    --model-path checkpoints/best_model.pt \
    --tokenizer-path checkpoints/tokenizer.pkl \
    --mode single \
    --prompt "请介绍一下人工智能的发展历史"
```

#### 批量测试

```bash
# 批量测试生成质量
python scripts/generate.py \
    --model-path checkpoints/best_model.pt \
    --tokenizer-path checkpoints/tokenizer.pkl \
    --mode batch \
    --output results.jsonl
```

### 5️⃣ GPU优化配置

系统会自动检测您的硬件并优化配置：

```bash
# 自动检测并显示优化信息
python -c "
from config.training_config import get_config
config = get_config('small')  # 会显示GPU信息和优化配置
"
```

#### 不同GPU的推荐配置

| GPU型号 | 显存 | 推荐配置 | 批量大小 |
|---------|------|----------|----------|
| RTX 3090/4090 | 24GB | medium/large | 16-32 |
| RTX 3080/4080 | 16GB | small/medium | 8-16 |
| RTX 3060Ti/4060Ti | 12GB | tiny/small | 4-8 |
| Apple M1/M2 Pro | 统一内存 | tiny/small | 4-16 |
| CPU | 系统内存 | tiny | 2-4 |

### 6️⃣ 自定义配置

```bash
# 创建自定义配置
python -c "
from src.model.config import MiniGPTConfig
from src.model.transformer import create_model

config = MiniGPTConfig(
    vocab_size=32000,
    hidden_size=768,
    num_hidden_layers=12,
    num_attention_heads=12,
    intermediate_size=3072,
    max_position_embeddings=2048
)

model = create_model(config=config)
print(f'自定义模型参数量: {model.get_num_params():,}')
"
```

## 🔄 训练流程

### 训练阶段概览

1. **预训练（Pretrain）**
   - **目标**: 学习语言基础能力
   - **数据**: 大规模无标注文本
   - **损失**: 下一个token预测

2. **监督微调（SFT）**
   - **目标**: 学习指令跟随能力
   - **数据**: 指令-回答对
   - **损失**: 交叉熵损失（仅回答部分）

3. **DPO训练**
   - **目标**: 优化生成质量和偏好对齐
   - **数据**: 偏好对比数据
   - **损失**: DPO损失函数

4. **RLHF训练**
   - **目标**: 基于人类反馈持续优化
   - **方法**: PPO算法
   - **组件**: 奖励模型、价值模型、策略模型

## 🏗️ 项目架构

```
minigpt-training/
├── src/                    # 核心代码
│   ├── model/             # 模型定义
│   ├── training/          # 训练逻辑
│   ├── data/              # 数据处理
│   ├── tokenizer/         # 分词器
│   ├── inference/         # 推理生成
│   ├── rl/                # 强化学习
│   └── utils/             # 工具函数
├── config/                # 配置文件
├── scripts/               # 训练脚本
├── data/                  # 数据集
├── docs/                  # 文档
├── tests/                 # 测试代码
└── checkpoints/           # 模型检查点
```

### 核心模块说明

| 模块 | 功能 | 主要文件 |
|------|------|----------|
| `src.model` | Transformer模型实现 | `transformer.py` |
| `src.model` | 现代激活函数 | `activation_functions.py` |
| `src.model` | 现代优化器 | `optimizers.py` |
| `src.model` | MoE架构 | `moe.py` |
| `src.training` | 训练流程控制 | `trainer.py` |
| `src.data` | 数据加载和处理 | `dataset_loader.py` |
| `src.tokenizer` | BPE分词器 | `bpe_tokenizer.py` |
| `src.rl` | 强化学习训练 | `rlhf_pipeline.py` |
| `src.inference` | 文本生成 | `generator.py` |

## ⚙️ 配置说明

### 模型配置

```python
# 超小模型（Mac优化）
tiny_config = {
    "d_model": 128,
    "n_heads": 2,
    "n_layers": 4,
    "vocab_size": 5000
}

# 小模型（推荐学习）
small_config = {
    "d_model": 512,
    "n_heads": 8,
    "n_layers": 6,
    "vocab_size": 10000
}
```

### 训练配置

```python
# 预训练配置
pretrain_config = {
    "learning_rate": 1e-4,
    "batch_size": 32,
    "max_steps": 50000,
    "warmup_steps": 1000
}

# SFT配置
sft_config = {
    "learning_rate": 5e-5,
    "batch_size": 16,
    "max_epochs": 10,
    "gradient_accumulation_steps": 2
}
```

## 🚀 高级功能

### 现代激活函数

```python
# 选择激活函数
from src.model.activation_functions import get_feedforward_layer

# SwiGLU前馈网络（推荐）
feed_forward = get_feedforward_layer(d_model=512, hidden_dim=2048, feedforward_type="swiglu")

# GEGLU前馈网络
feed_forward = get_feedforward_layer(d_model=512, hidden_dim=2048, feedforward_type="geglu")

# 标准前馈网络 + 现代激活函数
feed_forward = get_feedforward_layer(d_model=512, hidden_dim=2048, feedforward_type="standard", activation="gelu")
```

### 现代优化器

```python
from src.model.optimizers import get_optimizer

# Lion优化器（推荐）
optimizer = get_optimizer("lion", model.parameters(), lr=1e-4, weight_decay=0.01)

# Sophia优化器（大模型训练推荐）
optimizer = get_optimizer("sophia", model.parameters(), lr=1e-4, weight_decay=0.1)

# Schedule-Free AdamW
optimizer = get_optimizer("adamw_schedule_free", model.parameters(), lr=1e-3)
```

### MoE架构

```python
from src.model.moe import create_moe_model

# 创建MoE模型
model = create_moe_model(
    vocab_size=10000,
    d_model=512,
    n_layers=6,
    num_experts=8,      # 专家数量
    top_k=2,           # 激活的专家数量
    moe_type="sparse", # 稀疏MoE或共享专家MoE
)

# 使用MoE Transformer块
from src.model.moe import MoETransformerBlock
moe_block = MoETransformerBlock(
    d_model=512,
    n_heads=8,
    d_ff=2048,
    num_experts=8,
    top_k=2,
    moe_type="sparse"
)
```

### LoRA微调

```python
# 启用LoRA
config.sft.use_lora = True
config.sft.lora_rank = 16
config.sft.lora_alpha = 32
```

### 混合精度训练

```python
# 启用FP16
config.optimization.use_fp16 = True
```

### 分布式训练

```bash
# 多GPU训练
python -m torch.distributed.launch --nproc_per_node=4 scripts/train.py
```

### 数据集概览

本项目包含完整的minimind数据集，位于 `data/dataset/minimind_dataset/` 目录：

| 文件名 | 大小 | 用途 | 描述 |
|--------|------|------|------|
| `pretrain_hq.jsonl` | 1.6GB | 预训练 | 高质量中文预训练数据 |
| `sft_mini_512.jsonl` | 1.1GB | SFT训练 | 精选SFT对话数据（推荐） |
| `sft_512.jsonl` | 7.0GB | SFT训练 | 完整SFT数据集（字符长度<512） |
| `sft_1024.jsonl` | 5.2GB | SFT训练 | Qwen2.5蒸馏对话数据（字符长度<1024） |
| `sft_2048.jsonl` | 8.3GB | SFT训练 | 扩展Qwen2.5对话数据（字符长度<2048） |
| `dpo.jsonl` | 867MB | DPO训练 | RLHF偏好数据 |
| `r1_mix_1024.jsonl` | 351MB | 推理训练 | DeepSeek-R1蒸馏推理数据 |
| `lora_medical.jsonl` | 33MB | 领域微调 | 医学领域Q&A数据 |
| `lora_identity.jsonl` | 22KB | 身份训练 | 自我认知数据 |

#### 快速开始推荐
- **快速验证**: `pretrain_minimal.jsonl` + `pretrain_test.jsonl`
- **标准训练**: `pretrain_hq.jsonl` + `sft_mini_512.jsonl`
- **完整训练**: 使用所有数据文件（~20GB，4B tokens）

### 自定义数据集

```jsonl
# 预训练数据格式
{"text": "这是一段用于预训练的文本..."}

# SFT数据格式
{"conversations": [
    {"role": "user", "content": "问题"},
    {"role": "assistant", "content": "回答"}
]}

# DPO数据格式
{"prompt": "提示", "chosen": "更好的回答", "rejected": "较差的回答"}
```

## 📚 文档资源

### 实用指南
- [Mac优化指南](README_MAC_OPTIMIZED.md) - Mac设备训练完整指南
- [开发笔记](CLAUDE.md) - 项目开发过程与思考记录
- [项目结构说明](PROJECT_STRUCTURE.md) - 项目文件组织和模块说明

## 🔍 性能对比

### 标准Transformer模型

| 配置 | 参数量 | 内存需求 | 训练时间 | 推荐用途 |
|------|--------|----------|----------|----------|
| Tiny | ~13K | ~0.2MB | 10-20分钟 | 快速验证/Mac优化 |
| Small | ~66K | ~0.8MB | 30-45分钟 | 学习研究/小规模实验 |
| Medium | ~2.5M | ~30MB | 2-4小时 | 中等规模训练 |
| Large | ~25M | ~300MB | 数小时 | 完整模型训练 |

### MoE模型

| 配置 | 总参数量 | 激活参数量 | 专家数量 | 内存需求 | 推荐用途 |
|------|----------|------------|----------|----------|----------|
| MoE-Small | ~200K | ~50K | 8 | ~2MB | MoE架构验证 |
| MoE-Medium | ~10M | ~2.5M | 16 | ~120MB | 中型MoE训练 |
| MoE-Large | ~100M | ~25M | 32 | ~1.2GB | 大型MoE训练 |

### 优化器性能对比

| 优化器 | 内存开销 | 收敛速度 | 推荐场景 |
|--------|----------|----------|----------|
| AdamW | 2x参数量 | 标准 | 通用训练 |
| Lion | 1x参数量 | 快速 | 资源受限环境 |
| Sophia | 2x参数量 | 超快 | 大模型预训练 |

## 🧪 测试和评估

### 运行测试

```bash
# 模型结构测试
python tests/test_correct_small.py

# 中等模型测试
python tests/test_medium_model.py

# 模型检查
python tests/inspect_model.py
```

### 性能评估

```bash
# 生成质量评估
python scripts/evaluate.py --model checkpoints/model.pt --dataset test

# 性能基准测试
python calculate_model_comparison.py
```

## 🤝 贡献指南

### 开发环境设置

```bash
# 1. Fork并克隆仓库
git clone https://github.com/your-username/minigpt-training.git

# 2. 创建开发分支
git checkout -b feature/your-feature

# 3. 安装开发依赖
pip install -e ".[dev]"

# 4. 运行测试
python -m pytest tests/
```

### 提交规范

```bash
# 提交信息格式
git commit -m "feat: 添加新功能"
git commit -m "fix: 修复bug"
git commit -m "docs: 更新文档"
```

### 贡献类型

- 🐛 **Bug修复**: 修复已知问题
- ✨ **新功能**: 添加新的训练方法或工具
- 📚 **文档**: 改进文档和示例
- 🎨 **优化**: 性能优化和代码重构
- 🧪 **测试**: 添加或改进测试用例

## 📄 许可证

本项目采用 [MIT License](LICENSE) 开源协议。

## 🙏 致谢

感谢所有为此项目做出贡献的开发者和研究者。

## 📞 支持与反馈

- **问题反馈**: [GitHub Issues](https://github.com/your-repo/minigpt-training/issues)
- **功能建议**: [GitHub Discussions](https://github.com/your-repo/minigpt-training/discussions)
- **文档问题**: 查看 [docs/](docs/) 目录或提交Issue

---

<div align="center">

**⭐ 如果这个项目对你有帮助，请给个Star支持一下！ ⭐**

</div> 