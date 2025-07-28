# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MiniGPT is a hand-implemented Transformer-based language model training project. It provides a complete pipeline for training small GPT models from scratch, including pretraining, supervised fine-tuning (SFT), and DPO training. The project is designed for educational purposes to understand Transformer architecture and language model training.

## Development Environment Setup

```bash
# Install dependencies using uv (recommended)
uv sync

# Alternative: pip install
pip install -e .
```

## Core Architecture

### Model Components
- **Transformer Model** (`src/model/transformer.py`): Hand-implemented Transformer with multi-head attention, position encoding, and feed-forward networks
- **BPE Tokenizer** (`src/tokenizer/bpe_tokenizer.py`): Byte Pair Encoding tokenizer for text preprocessing
- **Training Pipeline** (`src/training/trainer.py`): Supports pretrain, SFT, and DPO training modes
- **Text Generation** (`src/inference/generator.py`): Text generation with various sampling strategies

### Configuration System
- All training configurations are centralized in `config/training_config.py`
- Predefined model sizes: tiny (~1M params), small (~25M params), medium (~100M params)
- Configurable training modes: pretrain, sft, dpo

## Common Development Commands

### Training Commands

```bash
# Train tokenizer from scratch and run SFT
python scripts/train.py --mode sft --config small --retrain-tokenizer

# Standard SFT training
python scripts/train.py --mode sft --config small

# Pretraining
python scripts/train.py --mode pretrain --config small

# Resume from checkpoint
python scripts/train.py --mode sft --config small --resume checkpoints/checkpoint.pt
```

### Inference Commands

```bash
# Interactive chat mode
python scripts/generate.py \
    --model-path checkpoints/best_model.pt \
    --tokenizer-path checkpoints/tokenizer.pkl \
    --mode chat

# Single inference
python scripts/generate.py \
    --model-path checkpoints/best_model.pt \
    --tokenizer-path checkpoints/tokenizer.pkl \
    --mode single \
    --prompt "Hello, how are you?"

# Batch testing
python scripts/generate.py \
    --model-path checkpoints/best_model.pt \
    --tokenizer-path checkpoints/tokenizer.pkl \
    --mode batch
```

## Key Code Patterns

### Model Creation
```python
from src.model.transformer import create_model
model = create_model(vocab_size=10000, model_size="small")
```

### Training Configuration
```python
from config.training_config import get_small_config
config = get_small_config()
# Modify config as needed
config.device = "mps"  # or "cuda", "cpu"
```

### Device Detection
The project automatically detects and uses the best available device:
- Apple Silicon GPU (MPS) on M1/M2 Macs
- CUDA GPU if available
- CPU as fallback

## Data Pipeline

### Expected Data Format
- **SFT Data**: Conversation format in JSONL files (`sft_mini_512.jsonl`)
- **Pretrain Data**: Plain text data (`pretrain_hq.jsonl`) 
- **DPO Data**: Preference pairs (`dpo.jsonl`)

### Data Location
- Training data expected in `data/dataset/minimind_dataset/`
- Checkpoints saved to `checkpoints/`
- Training logs saved to `logs/`

### Dataset Files Description

The `data/dataset/minimind_dataset/` directory contains various training datasets for different training stages:

#### Core Training Data (Essential)
- **`pretrain_hq.jsonl`** (1.6GB): High-quality pretraining data extracted from 匠数大模型数据集. Contains ~1.6GB of Chinese text with character length <512
  - Format: `{"text": "content here..."}`
  - Use: Pretraining phase to establish basic language understanding

- **`sft_mini_512.jsonl`** (1.2GB): Minimal SFT dataset combining 匠数科技 and Qwen2.5 distilled data
  - Format: Conversation format with user/assistant roles
  - Use: Recommended for quick Zero model training (character length <512)

#### Additional SFT Datasets
- **`sft_512.jsonl`** (7.5GB): Full SFT data from 匠数科技, cleaned with character length <512
- **`sft_1024.jsonl`** (5.6GB): Qwen2.5 distilled conversations with character length <1024  
- **`sft_2048.jsonl`** (9GB): Extended Qwen2.5 distilled conversations with character length <2048

#### Specialized Training Data
- **`dpo.jsonl`** (909MB): RLHF preference data from Magpie-DPO dataset
  - Format: Contains "chosen" and "rejected" response pairs
  - Use: DPO training to align model with human preferences

- **`r1_mix_1024.jsonl`** (340MB): DeepSeek-R1 distilled reasoning data
  - Format: Same as SFT data but focused on reasoning tasks
  - Use: Training reasoning capabilities (character length <1024)

#### Domain-Specific Data
- **`lora_identity.jsonl`** (22.8KB): Self-recognition data ("你是谁？我是minimind...")
- **`lora_medical.jsonl`** (34MB): Medical Q&A dataset for domain specialization

#### Training Configuration Notes
- Match sequence length settings to data: `sft_512.jsonl` → `max_seq_len=512`
- Recommended quick start: `pretrain_hq.jsonl` + `sft_mini_512.jsonl`
- Full training: Use complete dataset combination (~20GB, 4B tokens)
- All SFT data uses conversation format:
  ```json
  {
    "conversations": [
      {"role": "user", "content": "question"},
      {"role": "assistant", "content": "answer"}
    ]
  }
  ```

## Hardware Considerations

- **Apple Silicon**: Optimized for MPS acceleration
- **Memory**: FP16 training support for memory efficiency
- **Batch Sizes**: Automatically adjusted based on model size
  - tiny: batch_size=16, seq_len=256
  - small: batch_size=32, seq_len=512  
  - medium: batch_size=16, seq_len=1024

## Model Sizes and Configurations

| Size | Parameters | d_model | layers | heads | Use Case |
|------|-----------|---------|--------|-------|----------|
| tiny | ~1M | 128 | 4 | 2 | Quick testing |
| small | ~25M | 512 | 6 | 8 | Development |
| medium | ~100M | 768 | 12 | 12 | Full training |

## Important Implementation Notes

- The Transformer implementation is educational and hand-coded for clarity
- All core components (attention, position encoding, etc.) are implemented from scratch
- Supports causal masking for autoregressive generation
- Includes proper weight initialization and layer normalization
- Generation supports temperature, top-k, and top-p sampling