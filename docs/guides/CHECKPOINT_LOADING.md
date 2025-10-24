# Checkpoint Loading Guide

This document describes how to load pre-trained model checkpoints from various sources, including `/openbayes/home/out`.

## Features

The checkpoint loading system supports:

1. **Explicit checkpoint paths** - Specify the exact path to a checkpoint
2. **Environment variables** - Use `MINILLM_PRETRAINED_PATH` for automated loading
3. **Remote directories** - Automatically load from `/openbayes/home/out`
4. **Local directories** - Default behavior using local output directory
5. **Fallback strategies** - Multiple resolution strategies with priority ordering

## Usage Methods

### Method 1: Explicit Path (Highest Priority)

Load a specific checkpoint file:

```bash
python trainer/train_full_sft.py \
    --pretrained_path /openbayes/home/out/full_sft_512.pth \
    --data_path data/sft_mini_512.jsonl
```

This also works with local paths:

```bash
python trainer/train_full_sft.py \
    --pretrained_path ./out/pretrain_512.pth \
    --data_path data/sft_mini_512.jsonl
```

### Method 2: Remote Loading Flag

Automatically load from `/openbayes/home/out` based on stage and hidden size:

```bash
python trainer/train_full_sft.py \
    --load_from_remote \
    --hidden_size 512 \
    --data_path data/sft_mini_512.jsonl
```

This will look for `/openbayes/home/out/pretrain_512.pth` (default pretrain stage for SFT).

### Method 3: Environment Variable

Set the environment variable and let the system find the checkpoint:

```bash
export MINILLM_PRETRAINED_PATH=/openbayes/home/out/full_sft_512.pth

python trainer/train_full_sft.py \
    --data_path data/sft_mini_512.jsonl
```

### Method 4: Auto-Detection (Default)

By default, the system looks for checkpoints in:

1. Environment variable `MINILLM_PRETRAINED_PATH` (if set)
2. Remote directory `/openbayes/home/out/{stage}_{hidden_size}.pth`
3. Local directory `./out/{stage}_{hidden_size}.pth`

```bash
python trainer/train_full_sft.py --data_path data/sft_mini_512.jsonl
```

## Checkpoint Naming Convention

Checkpoints follow standard naming patterns:

```
pretrain_{hidden_size}[_moe].pth       # Pretraining stage
full_sft_{hidden_size}[_moe].pth       # Full SFT stage
reason_{hidden_size}[_moe].pth         # Reasoning model
rlhf_{hidden_size}[_moe].pth           # DPO/RLHF stage
ppo_actor_{hidden_size}[_moe].pth      # PPO actor model
ppo_critic_{hidden_size}[_moe].pth     # PPO critic model
full_dist_{hidden_size}[_moe].pth      # Distillation model
```

Where:
- `{hidden_size}` is the model's hidden dimension (e.g., 512, 768)
- `_moe` suffix is added if the model uses Mixture of Experts

## Examples

### Example 1: Continue Pretrain from OpenBayes Output

```bash
# Use a checkpoint from OpenBayes training
python trainer/train_pretrain.py \
    --pretrained_path /openbayes/home/out/pretrain_512.pth \
    --data_path data/pretrain_hq.jsonl \
    --epochs 2
```

### Example 2: Fine-tune with Remote Checkpoint

```bash
# Automatically load pretrained model from remote directory
python trainer/train_full_sft.py \
    --load_from_remote \
    --hidden_size 512 \
    --data_path data/sft_mini_512.jsonl \
    --epochs 1
```

### Example 3: DPO with Environment Variable

```bash
# Set the checkpoint path via environment variable
export MINILLM_PRETRAINED_PATH=/openbayes/home/out/full_sft_512.pth

python trainer/train_dpo.py \
    --data_path data/dpo_pairs.jsonl \
    --hidden_size 512
```

### Example 4: Script-based Training Pipeline

```bash
#!/bin/bash
set -e

MODEL_SIZE=512
REMOTE_OUT="/openbayes/home/out"

echo "Stage 1: Pretrain"
python trainer/train_pretrain.py \
    --hidden_size $MODEL_SIZE \
    --data_path data/pretrain_hq.jsonl \
    --epochs 2

echo "Stage 2: SFT"
python trainer/train_full_sft.py \
    --hidden_size $MODEL_SIZE \
    --data_path data/sft_mini_512.jsonl \
    --epochs 1

echo "Stage 3: DPO"
python trainer/train_dpo.py \
    --hidden_size $MODEL_SIZE \
    --data_path data/dpo_pairs.jsonl \
    --pretrained_path ./out/full_sft_${MODEL_SIZE}.pth
```

## CheckpointLoader API

The `checkpoint_loader.py` module provides utilities for advanced usage:

```python
from checkpoint_loader import CheckpointLoader

# Get checkpoint path (with existence check)
path, exists = CheckpointLoader.get_checkpoint_path(
    stage='sft',
    hidden_size=512,
    use_moe=False,
    source='remote',
    remote_dir='/openbayes/home/out'
)

if exists:
    print(f"Found checkpoint at: {path}")

# Resolve checkpoint path with fallback strategies
resolved_path = CheckpointLoader.resolve_checkpoint_path(
    explicit_path=None,
    stage='sft',
    hidden_size=512,
    use_moe=False
)

# Load checkpoint into model
success = CheckpointLoader.load_checkpoint(
    model=my_model,
    checkpoint_path=resolved_path,
    device='cuda:0',
    strict=False
)
```

## Supported Training Scripts

The checkpoint loading feature is integrated into:

- `train_pretrain.py` - Pretraining stage
- `train_full_sft.py` - Full supervised fine-tuning
- `train_dpo.py` - Direct preference optimization
- `train_ppo.py` - PPO RLHF training
- `train_grpo.py` - Group relative policy optimization
- `train_spo.py` - Sequence preference optimization
- `train_distillation.py` - Model distillation
- `train_distill_reason.py` - Reasoning distillation
- `train_lora.py` - LoRA fine-tuning

All scripts support the same checkpoint loading arguments:

- `--pretrained_path`: Explicit checkpoint path
- `--load_from_remote`: Load from `/openbayes/home/out`

## Troubleshooting

### Checkpoint Not Found

If you see a warning "No pretrained checkpoint found":

1. Verify the checkpoint file exists: `ls -la /path/to/checkpoint.pth`
2. Check the hidden size matches: `--hidden_size 512`
3. Use `--pretrained_path /full/path/to/checkpoint.pth` for explicit paths
4. Enable verbose logging to see which paths were searched

### Wrong Model Loaded

1. Verify the checkpoint path with `--pretrained_path`
2. Check the model architecture matches (hidden_size, num_layers)
3. Use `strict=False` (default) if loading into models with slightly different architectures

### OpenBayes Path Access

If `/openbayes/home/out` is not accessible:

1. Check file permissions: `ls -la /openbayes/home/out/`
2. Use absolute paths or mounted paths instead
3. Copy checkpoints to local directory and use local paths

## Best Practices

1. **Always specify hidden_size** - Ensure it matches the checkpoint
2. **Use explicit paths when certain** - Avoid ambiguity with `--pretrained_path`
3. **Document your pipelines** - Include checkpoint paths in training scripts
4. **Version your checkpoints** - Use meaningful directory structures like `./out/v1/`, `./out/v2/`
5. **Test loading locally first** - Verify checkpoints work before deploying to remote systems

## Integration with Training Pipelines

To integrate checkpoint loading into your training scripts:

```python
from checkpoint_loader import CheckpointLoader

# In your training script's argument parser
parser.add_argument("--pretrained_path", type=str, default=None,
                    help="Path to pretrained checkpoint")
parser.add_argument("--load_from_remote", action="store_true",
                    help="Load from /openbayes/home/out")

# In your model initialization
ckp_path = CheckpointLoader.resolve_checkpoint_path(
    explicit_path=args.pretrained_path,
    stage='pretrain',  # or 'sft', 'rlhf', etc.
    hidden_size=args.hidden_size,
    use_moe=args.use_moe,
    remote_dir='/openbayes/home/out'
)

if ckp_path:
    CheckpointLoader.load_checkpoint(
        model=model,
        checkpoint_path=ckp_path,
        device=args.device
    )
```
