# GSM8K 强化学习（MiniLLM + MLX）

目标：用 **GSM8K** 题目做 rollout，按最终答案是否正确给 `reward∈{0,1}`，再用 **MLX** 做一个最小可跑通的 REINFORCE 训练闭环（rollout / buffer / train）。

## 1) 下载数据集

按你的要求（默认落到 `/root/gsm8k`）：

```bash
huggingface-cli download --repo-type dataset zhuzilin/gsm8k --local-dir /root/gsm8k
```

或用项目封装：

```bash
python3 -m mlx_train.rl_gsm8k.download --local_dir /root/gsm8k
```

如果你是在非 root 环境（例如 macOS 本机）跑脚本，推荐把数据下载到仓库内：

```bash
python3 -m mlx_train.rl_gsm8k.download --local_dir dataset/gsm8k
```

## 2) Rollout 生成（MLX infer 后端）

把 rollout 写入 JSONL buffer（可以单独运行；在 loop 模式下会每轮生成一个 `out_dir/buffer/iter_XXX.jsonl`）：

```bash
python3 -m mlx_train.rl_gsm8k.rollout \
  --checkpoint out/mlx/sft/checkpoints/step_XXXXXXXX \
  --dataset_dir /root/gsm8k \
  --out_buffer out/rl_gsm8k/buffer.jsonl \
  --num_rollouts 1024
```

## 3) 训练（MLX）

从 buffer 做 RL 更新并保存 checkpoint（配合 `--resume` 可多轮累计训练）：

```bash
python3 -m mlx_train.rl_gsm8k.train \
  --init_from out/mlx/sft/checkpoints/step_XXXXXXXX \
  --buffer_path out/rl_gsm8k/buffer.jsonl \
  --out_dir out/rl_gsm8k \
  --max_steps 1000
```

## 4) 一键跑通（rollout -> train -> rollout）

```bash
python3 -m mlx_train.rl_gsm8k.run \
  --checkpoint out/mlx/sft/checkpoints/step_XXXXXXXX \
  --dataset_dir /root/gsm8k \
  --out_dir out/rl_gsm8k \
  --iters 5 --max_steps 1000
```

如需在 Python loop 中启用 warmup：

```bash
python3 -m mlx_train.rl_gsm8k.run \
  --checkpoint out/mlx/sft/checkpoints/step_XXXXXXXX \
  --dataset_dir /root/gsm8k \
  --out_dir out/rl_gsm8k \
  --warmup_sft_steps 200
```

也可以用脚本（默认 `DOWNLOAD=auto`，数据不存在会自动下载；在非 root 环境会自动落到 `dataset/gsm8k`）：

```bash
CHECKPOINT=out/mlx/sft/checkpoints/step_XXXXXXXX bash scripts/run_mlx_rl_gsm8k.sh
```

### SFT warmup（推荐）

如果一开始 rollout `reward>0` 一直为 0，RL 会没有梯度（loss/grad_norm 可能长期为 0）。脚本默认会先做一小段 GSM8K 的 SFT warmup（`WARMUP_STEPS=200`）来 bootstrap 正样本。

- 关闭 warmup：`WARMUP_STEPS=0`
- warmup 的数据会写到：`$OUT_DIR/warmup_sft/gsm8k_sft.jsonl`
- 如果 `$OUT_DIR/checkpoints` 已经有旧 checkpoint，脚本会优先 resume 并跳过 warmup；要从头 warmup 可用 `RESET_OUT=1` 或换一个 `OUT_DIR`

Loop 相关常用参数：

- `ITERS`：轮数（默认 `5`）
- `MAX_STEPS`：总训练步数（默认 `1000`，均摊到每轮）
- `STEPS_PER_ITER`：每轮训练步数（可选；默认 `ceil(MAX_STEPS/ITERS)`）
- `NUM_ROLLOUTS`：每轮采样的 prompt 数
- `SAMPLES_PER_PROMPT`：每个 prompt 采样的 completion 数（>1 用于提高正样本概率）
- `MIN_POSITIVE`：每轮至少写入多少条 `reward>0`（不满足会继续采样，受 `MAX_TOTAL_ROLLOUTS` 限制）
- `MAX_TOTAL_ROLLOUTS`：每轮最多写入多少条 rollout（`0` 表示 `NUM_ROLLOUTS*SAMPLES_PER_PROMPT`）

脚本默认还会在训练后：

- 跑 GSM8K `test`（默认抽样 `GSM8K_EVAL_NUM=200`；设为 `0` 表示全量）
- 跑 `mlx_train.bench` 的 synthetic suite（默认 `BENCH_SUITE=all`，且 `--no_ollama`）

可通过环境变量关闭：

```bash
RUN_EVAL=0 RUN_BENCH=0 CHECKPOINT=... bash scripts/run_mlx_rl_gsm8k.sh
```

## Buffer 格式（JSONL）

每行是一条 rollout 记录，包含 `messages`（system/user）、`response`、`reward`、以及解析出的 `pred_final/ref_final` 等字段。
