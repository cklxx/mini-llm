# MiniLLM Documentation

Welcome to the MiniLLM documentation. This directory contains comprehensive information about the project, from quick start guides to detailed technical documentation.

## Documentation Structure

### 📚 [Guides](./guides/README.md)

Comprehensive guides for using and developing MiniLLM:

- **[Checkpoint Loading Guide](./guides/CHECKPOINT_LOADING.md)** - Load pre-trained models from various sources

### 📋 [Changelog](./changelog/README.md)

Version history and detailed change logs:

- **[Full Changelog](./changelog/CHANGELOG.md)** - Complete history of all changes
- Latest features, fixes, and improvements

### 🌐 [Detailed Documentation](./booklet_cn.md)

In-depth guide (in Chinese) covering:
- Detailed architecture overview
- Training pipeline explanation
- Data processing guide
- Advanced features

### 📜 [Community Guidelines](./CODE_OF_CONDUCT.md)

Our Code of Conduct outlining community standards and expectations.

## Quick Navigation

| Topic | Document |
|-------|----------|
| Project Overview | [Main README](../README.md) |
| Getting Started | [Main README - Quick Start](../README.md#quick-start) |
| Checkpoint Management | [Checkpoint Loading Guide](./guides/CHECKPOINT_LOADING.md) |
| Version History | [Changelog](./changelog/CHANGELOG.md) |
| Architecture & Details | [Booklet CN](./booklet_cn.md) |
| Community Standards | [Code of Conduct](./CODE_OF_CONDUCT.md) |

## Finding What You Need

**I want to...**

- **Get started quickly** → Start with [Main README](../README.md)
- **Load a pre-trained model** → Read [Checkpoint Loading Guide](./guides/CHECKPOINT_LOADING.md)
- **Continue training from OpenBayes** → Check [Checkpoint Loading Examples](./guides/CHECKPOINT_LOADING.md#examples)
- **Understand recent changes** → See [Changelog](./changelog/CHANGELOG.md)
- **Learn about architecture** → Read [Booklet CN](./booklet_cn.md)
- **Contribute** → Review [Code of Conduct](./CODE_OF_CONDUCT.md)

## Directory Structure

```
docs/
├── README.md                    # This file - documentation index
├── guides/                      # Usage and implementation guides
│   ├── README.md               # Guide index
│   └── CHECKPOINT_LOADING.md   # Checkpoint loading guide
├── changelog/                   # Version history
│   ├── README.md               # Changelog index
│   └── CHANGELOG.md            # Full changelog with version details
├── booklet_cn.md               # Detailed documentation (Chinese)
└── CODE_OF_CONDUCT.md          # Community guidelines
```

## Key Features

### Checkpoint Loading System

MiniLLM now supports multiple ways to load pre-trained checkpoints:

```bash
# Method 1: Explicit path
python trainer/train_full_sft.py --pretrained_path /path/to/checkpoint.pth

# Method 2: Environment variable
export MINILLM_PRETRAINED_PATH=/path/to/checkpoint.pth
python trainer/train_full_sft.py --data_path data/sft_mini_512.jsonl

# Method 3: Auto-detection from remote or local directories
python trainer/train_full_sft.py --data_path data/sft_mini_512.jsonl
```

For complete details, see [Checkpoint Loading Guide](./guides/CHECKPOINT_LOADING.md).

### Automatic Checkpoint Loading in run.sh

The main training script automatically detects and loads checkpoints:

```bash
scripts/run.sh              # Auto-load from multiple sources
scripts/run.sh --smoke-test # CPU-only smoke test
```

## Recent Updates

Latest version includes:

- ✅ **Automatic Checkpoint Loading** - Scripts auto-detect pre-trained models
- ✅ **Flexible Checkpoint Resolution** - Multiple loading strategies with priority ordering
- ✅ **Enhanced .gitignore** - Comprehensive patterns for Python/ML projects
- ✅ **Organized Documentation** - All docs properly categorized in this structure
- ✅ **Smoke Test Support** - Working CPU-only tests for validation

See [Changelog](./changelog/CHANGELOG.md) for complete version history.

## Getting Help

- **Having trouble with checkpoints?** → [Troubleshooting Guide](./guides/CHECKPOINT_LOADING.md#troubleshooting)
- **Want to know what's new?** → [Changelog](./changelog/CHANGELOG.md)
- **Found an issue?** → Report on GitHub

## Contributing

We welcome contributions! Before submitting:

1. Review our [Code of Conduct](./CODE_OF_CONDUCT.md)
2. Check the [Changelog](./changelog/CHANGELOG.md) for recent changes
3. Follow the existing code style and documentation patterns

Thank you for using MiniLLM!
