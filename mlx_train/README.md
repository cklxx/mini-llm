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
- `scripts/run_mlx.sh` 默认不会下载 DPO（MLX 还未实现 DPO 训练），如需预先下载可设置 `DOWNLOAD_DPO=1`。

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

## 推理 / 效果查看

```bash
TRANSFORMERS_VERBOSITY=error python3 -m mlx_train.infer \
  --checkpoint out/mlx/pretrain/checkpoints/step_XXXXXXXX \
  --prompt "请介绍一下自己。" \
  --temperature 0.7 --top_p 0.9 --max_new_tokens 200
```

## 一键查看模型参数量 / FLOPs / 设备峰值

```bash
# 自动选 out/mlx 下最新 checkpoint（优先 sft，其次 pretrain）
bash scripts/stats_mlx.sh

# 指定 checkpoint 并估算利用率（从训练日志里抄 tok/s 过来）
bash scripts/stats_mlx.sh \
  --checkpoint out/mlx/sft/checkpoints/step_XXXXXXXX \
  --batch_size 1 --seq_len 1024 --accum_steps 8 \
  --tok_s 3747
```

## Bench（与本地 Ollama Qwen3 对比）

需要本机已安装并启动 `ollama serve`，且本地有 `qwen3:0.6b`（默认对比模型）：

```bash
# 默认：多维度 synthetic bench（math_mcq/qa/logic/knowledge/sort/json/copy），并展示进度
bash scripts/bench_mlx.sh --checkpoint out/mlx/sft/checkpoints/step_XXXXXXXX

# 只跑数学四则选择题
bash scripts/bench_mlx.sh --suite math_mcq --checkpoint out/mlx/sft/checkpoints/step_XXXXXXXX

# 指定对比模型
bash scripts/bench_mlx.sh --ollama_model qwen3:8b
```

## 蒸馏（Ollama Qwen3:0.6b 合成数据 -> MLX SFT）

目标：用本机 `ollama` 的 `qwen3:0.6b` 生成偏 **知识问答 / 逻辑推理 / 数学计算** 的合成 SFT 数据，并在生成的同时启动 MLX 训练（先冷启动生成一部分数据再开训）。

前置：

```bash
ollama serve
ollama pull qwen3:0.6b
```

一键运行（默认：先生成 512 条，再并行生成 + 训练）：

```bash
bash scripts/run_mlx_distill_ollama.sh
```

常用环境变量（可按需覆盖）：

- `OLLAMA_URL`（默认 `http://127.0.0.1:11434`）
- `OLLAMA_MODEL`（默认 `qwen3:0.6b`）
- `COLD_SAMPLES`（默认 `512`，冷启动生成 N 条再开训）
- `TOTAL_SAMPLES`（默认 `20000`，目标总样本数；设为 `0` 表示持续生成）
- `GEN_WORKERS`（默认 `8`，并行推理 worker 数）
- `DATA_JSONL`（默认 `out/distill_ollama_qwen3_0.6b/synth.jsonl`）
- `OUT_DIR`（默认 `out/mlx_distill/qwen3_0.6b_sft`）
- `INIT_FROM`（可选：从某个 checkpoint 初始化 SFT，例如 `out/mlx/pretrain/checkpoints/step_XXXXXXXX`）
- `MAX_STEPS` / `SEQ_LEN` / `BATCH_SIZE` / `ACCUM_STEPS` / `DTYPE` / `PRESET`（训练超参）

数据生成器也可单独使用：

```bash
python3 -m mlx_train.distill_data_ollama --help
```

## GSM8K 强化学习（rollout/buffer/train）

实现放在独立目录：`mlx_train/rl_gsm8k/`（含 `README.md`）。

一键脚本（rollout -> train 循环；每轮用最新权重继续 rollout）：

```bash
CHECKPOINT=out/mlx/sft/checkpoints/step_XXXXXXXX bash scripts/run_mlx_rl_gsm8k.sh
```

脚本默认 `DOWNLOAD=auto`：如果本地没有数据会自动下载；在非 root 环境会自动使用 `dataset/gsm8k`（可用 `DATASET_DIR` 覆盖）。默认还会先做一小段 GSM8K 的 SFT warmup（`WARMUP_STEPS=200`；设 `WARMUP_STEPS=0` 可关闭；如需从头跑可设 `RESET_OUT=1` 或换 `OUT_DIR`）。Loop 参数：`ITERS` / `MAX_STEPS` / `STEPS_PER_ITER` / `NUM_ROLLOUTS` / `SAMPLES_PER_PROMPT`。训练结束后默认会追加跑 GSM8K `test`（`GSM8K_EVAL_NUM`）与 `mlx_train.bench`（`BENCH_SUITE`；`RUN_EVAL=0` / `RUN_BENCH=0` 可关闭）。

## 简单评测（loss / perplexity）

对一小段数据做 forward-only 的平均 loss / ppl（建议先用 `--max_batches` 小跑对比不同 checkpoint）：

```bash
TRANSFORMERS_VERBOSITY=error python3 -m mlx_train.eval \
  --checkpoint out/mlx/pretrain/checkpoints/step_XXXXXXXX \
  --task pretrain \
  --data_path minimind:pretrain_hq.jsonl \
  --data_dir dataset/minimind \
  --seq_len 1024 \
  --batch_size 1 \
  --max_batches 100 \
  --compile
```

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
