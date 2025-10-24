# MiniLLM Guides

This directory contains comprehensive guides for using and developing the MiniLLM project.

## Available Guides

### [Checkpoint Loading Guide](./CHECKPOINT_LOADING.md)

Complete guide for loading pre-trained model checkpoints from various sources:

- **Explicit checkpoint paths** - Specify exact paths
- **Environment variables** - Use `MINILLM_PRETRAINED_PATH`
- **Remote directories** - Load from `/openbayes/home/out`
- **Local directories** - Use local output directory
- **Fallback strategies** - Multiple resolution strategies with priority ordering

Key topics:
- Usage methods and examples
- Checkpoint naming conventions
- CheckpointLoader API reference
- Troubleshooting and best practices
- Integration with training pipelines

**Read this guide if you want to:**
- Load pre-trained models for fine-tuning
- Continue training from OpenBayes checkpoints
- Understand checkpoint management strategies
- Set up automated checkpoint loading in your training pipeline

## Other Documentation

For other documentation, see:

- [Main README](../../README.md) - Project overview and quick start
- [Changelog](../changelog/CHANGELOG.md) - Version history and changes
- [Booklet CN](../booklet_cn.md) - Detailed Chinese documentation (booklet)

## Contributing

When adding new guides:

1. Create a new markdown file in this directory
2. Add a descriptive section to this README
3. Include examples and best practices
4. Keep the guide focused on a specific topic

## Related Documentation Structure

```
docs/
├── guides/                      # Usage and implementation guides
│   ├── README.md               # This file
│   └── CHECKPOINT_LOADING.md   # Checkpoint loading guide
├── changelog/                   # Version history
│   ├── README.md               # Changelog index
│   └── CHANGELOG.md            # Full changelog
├── booklet_cn.md               # Detailed Chinese documentation
├── CODE_OF_CONDUCT.md          # Community guidelines
└── ../README.md                # Main project README
```
