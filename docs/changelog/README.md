# Changelog Directory

This directory contains detailed changelog information for the MiniLLM project.

## Files

- **CHANGELOG.md** - Main changelog file documenting all changes, fixes, and improvements to the project

## Format

The changelog follows the [Keep a Changelog](https://keepachangelog.com/) format and is organized by version and category (Added, Changed, Fixed, Deprecated, Removed, Security).

## Latest Changes

For the most recent updates, see [CHANGELOG.md](./CHANGELOG.md).

### Quick Summary of Recent Changes

✅ **Automatic Checkpoint Loading** - The `run.sh` script now automatically detects and loads pretrained checkpoints from `/openbayes/home/out`, local directories, or environment variables.

✅ **CheckpointLoader Utility** - New `trainer/checkpoint_loader.py` module for flexible checkpoint management across all training scripts.

✅ **Enhanced Documentation** - New `CHECKPOINT_LOADING.md` with comprehensive examples and best practices.

✅ **Fixed Model Path Issues** - Corrected tokenizer loading paths in all trainer scripts.

✅ **Working Smoke Test** - Smoke test now passes successfully through all three training stages.

## How to Use This Directory

### Finding Information
- Look for specific feature changes in CHANGELOG.md
- Check "Added" section for new features
- Check "Fixed" section for bug fixes
- Check "Changed" section for modifications to existing features

### Reporting Issues
If you encounter issues, check the "Known Issues" section in CHANGELOG.md first. If your issue is not listed, please report it with:
- Description of the problem
- Steps to reproduce
- Expected vs. actual behavior
- Your environment (Python version, OS, etc.)

### Contributing
When contributing changes, please:
1. Update CHANGELOG.md with your changes
2. Follow the format: Add entries under the "Unreleased" section
3. Categorize changes appropriately (Added, Changed, Fixed, etc.)
4. Include examples when documenting new features

## Version History

- **Latest**: Development version
  - See "Unreleased" section in CHANGELOG.md

- **2024-10-24**: First version with automatic checkpoint loading
  - Fixed smoke test failures
  - Added checkpoint loading support
  - Enhanced .gitignore

## Related Documentation

- [CHECKPOINT_LOADING.md](../CHECKPOINT_LOADING.md) - Guide for using checkpoint loading features
- [Main README](../../README.md) - Project overview
- [scripts/run.sh](../../scripts/run.sh) - Main training script with auto-loading logic
