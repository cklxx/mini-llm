# MLX 训练 MiniLLM（≈200MB 预设）

本目录提供一个 **MLX** 版本的 MiniLLM（架构对齐 `model/model_minillm.py` 的 LLaMA-style：RMSNorm + RoPE + (GQA/MQA) Attention + SwiGLU FFN，权重绑定 lm_head/embedding）。

## 安装

```bash
python3 -m pip install -r mlx_train/requirements.txt
```

也可以使用 `uv` 初始化环境：

```bash
uv venv .venv_mlx --seed
uv pip install -r mlx_train/requirements.txt -p .venv_mlx/bin/python
```

## 一键脚本（推荐）

```bash
# 完整流程：下载数据 -> 预训练 -> SFT -> 推理
bash scripts/run_mlx.sh

# 只跑通：极小数据 + 少量步数 + 推理
bash scripts/run_mlx.sh --smoke-test
```

> `--smoke-test` 默认会自动删除输出目录（`out/mlx_smoke`），如需保留可设置 `CLEANUP_SMOKE=0`。

## 快速 Smoke（推荐先跑通）

使用仓库自带的 `data/chinese/identity_conversations.jsonl` 做 SFT 形式的 mask，跑几步验证：

```bash
python3 -m mlx_train.train \
  --data_path data/chinese/identity_conversations.jsonl \
  --task sft \
  --preset tiny \
  --seq_len 256 \
  --batch_size 2 \
  --accum_steps 1 \
  --max_steps 5 \
  --save_interval 2 \
  --log_interval 1 \
  --out_dir out/mlx_smoke
```

## 自动下载训练数据（MiniMind 数据集）

脚本支持在 `--data_path` 里直接写 `minimind:*`，会自动从 HuggingFace 的 `jingyaogong/minimind_dataset` 下载到 `--data_dir`（默认 `./dataset`）。

```bash
# Smoke：下载极小数据并跑通（不会拉取大文件）
python3 -m mlx_train.train \
  --data_path minimind:smoke \
  --task sft \
  --preset tiny \
  --seq_len 256 \
  --batch_size 2 \
  --max_steps 5 \
  --out_dir out/mlx_minimind_smoke
```

内置别名：

- `minimind:auto`：推荐训练数据（`pretrain`→`pretrain_hq.jsonl`，`sft`→`sft_mini_512.jsonl`）
- `minimind:small`：小数据（`lora_medical.jsonl`，约 32MB）
- `minimind:smoke`：极小数据（`lora_identity.jsonl`，约 23KB）
- DPO 数据：`minimind:dpo.jsonl`；兼容别名 `minimind:dpo_pairs.jsonl`（会在本地生成同名文件）
- 默认下载保护：`--max_download_mb=2048`（超过会报错，需显式放开）

如果本地访问 HuggingFace 较慢/不可用，可设置镜像端点：

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

## 训练 200MB（≈100M params）预设

```bash
python3 -m mlx_train.train \
  --data_path minimind:auto \
  --task pretrain \
  --preset 200mb \
  --seq_len 1024 \
  --batch_size 1 \
  --accum_steps 8 \
  --dtype bfloat16 \
  --out_dir out/mlx_200mb
```

完成预训练后，可用 `--init_from` 将权重带入 SFT：

```bash
python3 -m mlx_train.train \
  --data_path minimind:sft_mini_512.jsonl \
  --task sft \
  --preset 200mb \
  --init_from out/mlx_200mb/checkpoints/step_XXXXXXXX
```

如需下载更大的数据文件（例如 `sft_512.jsonl` / `sft_2048.jsonl`），请显式放开限制：

```bash
python3 -m mlx_train.train \
  --data_path minimind:sft_512.jsonl \
  --task sft \
  --preset 200mb \
  --max_download_mb 8000
```

## 输出与恢复

- checkpoint 目录：`<out_dir>/checkpoints/step_XXXXXXXX/`
  - `model.safetensors`：MLX 权重
  - `optimizer.npz`：优化器状态（用于恢复训练）
  - `config.json` / `state.json`
- 恢复训练：`--resume <out_dir>/checkpoints/step_XXXXXXXX`
- 默认只保留最近 3 个 checkpoint：`--keep_last_checkpoints 3`（设为 `0` 可关闭清理）

## 性能 / 瓶颈定位

分段计时（会额外同步，速度会更慢，适合定位瓶颈）：

```bash
python3 -m mlx_train.train \
  --data_path minimind:smoke \
  --task sft \
  --preset tiny \
  --seq_len 256 \
  --batch_size 2 \
  --max_steps 5 \
  --log_interval 1 \
  --profile_timing
```

GPU 侧（Metal）kernel 级别分析：生成 `.gputrace` 后用 Xcode 打开即可查看时间线。

```bash
python3 -m mlx_train.train \
  --data_path minimind:smoke \
  --task sft \
  --preset tiny \
  --seq_len 256 \
  --batch_size 2 \
  --max_steps 3 \
  --metal_capture out/trace.gputrace \
  --metal_capture_steps 1
```

## 说明

- 当前 MLX 路径 **未实现 MoE**（`use_moe=True` 会报错）。
- 200MB 预设指 fp16/bf16 权重大致占用（训练时显存/内存还会包含优化器状态与激活）。
