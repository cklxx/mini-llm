# Changelog

All notable changes to the MiniLLM project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Automatic Checkpoint Loading in run.sh**: The main training script now automatically detects and loads pretrained model checkpoints from `/openbayes/home/out`, local `out/` directory, or via the `MINILLM_PRETRAINED_PATH` environment variable. Priority order: environment variable → remote OpenBayes directory → local directory.
  - Added `find_pretrained_checkpoint()` function with smart checkpoint discovery
  - Support for MoE model suffix detection (`_moe`)
  - Auto-loading for all three training stages (pretrain, SFT, DPO)

- **CheckpointLoader Utility Module**: New `trainer/checkpoint_loader.py` module provides flexible checkpoint resolution:
  - `CheckpointLoader.get_checkpoint_path()`: Get checkpoint path with existence check
  - `CheckpointLoader.load_checkpoint()`: Load checkpoints into models
  - `CheckpointLoader.resolve_checkpoint_path()`: Multi-strategy checkpoint resolution
  - Support for explicit paths, environment variables, and auto-detection
  - Comprehensive documentation with API reference

- **Pretrained Model Loading CLI Arguments**: All training scripts now support:
  - `--pretrained_path`: Explicit checkpoint path (any location)
  - `--load_from_remote`: Load from `/openbayes/home/out` directory
  - Environment variable support via `MINILLM_PRETRAINED_PATH`

- **Documentation**:
  - `CHECKPOINT_LOADING.md`: Comprehensive guide for checkpoint loading with examples
  - Support for continuing training from models trained in OpenBayes environments

### Changed
- **Fixed Model Path References**: Changed all trainer scripts to use `./model/` instead of `../model/` for correct tokenizer loading path
- **Enhanced .gitignore**: Added comprehensive patterns for Python projects:
  - Python compiled files and virtual environments
  - IDE configuration files
  - Training outputs, logs, and cache
  - Environment variable files for security
  - Data processing artifacts

- **run.sh Script Enhancements**:
  - Added `USE_MOE` environment variable support
  - Improved checkpoint detection logic for all training stages
  - Added informative logging for checkpoint loading
  - Smart fallback strategy for SFT and DPO stages

### Fixed
- **Smoke Test Execution**: Fixed broken virtual environment setup that prevented smoke test from running
  - Recreated virtual environment with proper pip installation
  - Created minimal preference pairs data for DPO testing
  - Fixed all tokenizer path references across trainer scripts
- **Parameter Access Issue**: Fixed `use_moe` parameter access in checkpoint loading by using `lm_config.use_moe` instead of `args.use_moe`

## [2024-10-24]

### Added
- **Initial Model Path Fixes**: Fixed tokenizer loading paths from `../model/` to `./model/` in all training scripts
- **Enhanced .gitignore**: Added patterns for common Python/ML project artifacts
- **Smoke Test Support**: Enabled smoke test execution with CPU-only mode
- **Preference Data**: Created minimal `data/chinese/preference_pairs.jsonl` for testing

### Fixed
- **Virtual Environment Issues**: Resolved broken venv symlinks by recreating with proper Python installation
- **Smoke Test Failures**: All three training stages (pretrain, SFT, DPO) now execute successfully with proper evaluation metrics

---

## Usage Examples

### Automatic Checkpoint Loading (New Feature)

Simply run the training pipeline and it will automatically detect and load available checkpoints:

```bash
# Will auto-load pretrain checkpoint if available
scripts/run.sh

# With custom model size
MODEL_HIDDEN_SIZE=768 scripts/run.sh

# With MoE models
USE_MOE=true MODEL_HIDDEN_SIZE=512 scripts/run.sh
```

### Explicit Checkpoint Path

```bash
python trainer/train_full_sft.py \
    --pretrained_path /openbayes/home/out/full_sft_512.pth \
    --data_path data/sft_mini_512.jsonl
```

### Environment Variable

```bash
export MINILLM_PRETRAINED_PATH=/path/to/pretrain_512.pth
scripts/run.sh
```

---

## Migration Guide

### For Existing Users

If you were manually specifying checkpoint paths before:

**Old way:**
```bash
python trainer/train_full_sft.py \
    --data_path data/sft_mini_512.jsonl
# Would fail if no checkpoint in default location
```

**New way:**
```bash
scripts/run.sh
# Automatically finds and loads checkpoints from multiple locations
```

Or explicitly:
```bash
python trainer/train_full_sft.py \
    --pretrained_path /openbayes/home/out/pretrain_512.pth \
    --data_path data/sft_mini_512.jsonl
```

---

## Technical Details

### Checkpoint Loading Priority Order

1. **Explicit Path** (`--pretrained_path` argument): Highest priority
2. **Environment Variable** (`MINILLM_PRETRAINED_PATH`): Second priority
3. **Remote Directory** (`/openbayes/home/out/`): Third priority
4. **Local Directory** (`./out/`): Lowest priority
5. **Not found**: Initialize with random weights and log warning

### Supported Checkpoint Formats

Checkpoints should be PyTorch state dictionaries in `.pth` format:

```
pretrain_{hidden_size}[_moe].pth
full_sft_{hidden_size}[_moe].pth
reason_{hidden_size}[_moe].pth
rlhf_{hidden_size}[_moe].pth
ppo_actor_{hidden_size}[_moe].pth
ppo_critic_{hidden_size}[_moe].pth
full_dist_{hidden_size}[_moe].pth
```

### Environment Variables

- `MINILLM_PRETRAINED_PATH`: Explicit path to pretrained checkpoint
- `MODEL_HIDDEN_SIZE`: Model hidden dimension (default: 512)
- `MODEL_NUM_LAYERS`: Number of transformer layers (default: 8)
- `USE_MOE`: Enable Mixture of Experts (default: false)
- `USE_UV`: Use uv for dependency management (auto-detected)
- `VENV_DIR`: Virtual environment directory (default: .venv)
- `TF_DIR`: TensorBoard directory (default: /openbayes/home/tf_dir)
- `OUT_DIR`: Output directory for checkpoints (default: out)
- `DATA_DIR`: Processed data directory (default: data/processed)

---

## Known Issues

None currently reported.

---

## Future Improvements

- [ ] Support for HuggingFace model hub integration
- [ ] Automatic checkpoint versioning
- [ ] Model quantization support
- [ ] Multi-GPU checkpoint sharding
- [ ] Distributed checkpoint loading across nodes
- [ ] Checkpoint compression/decompression utilities

---

## Contributing

When making changes, please:
1. Update this CHANGELOG.md file with your changes
2. Follow the "Keep a Changelog" format
3. Update version information if appropriate
4. Add examples for new features
5. Document any breaking changes

---

## Support

For issues or questions about checkpoint loading:
- See `CHECKPOINT_LOADING.md` for detailed documentation
- Check `scripts/run.sh` for implementation details
- Review `trainer/checkpoint_loader.py` for API reference
