from __future__ import annotations

import argparse
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import List, Optional


def _run(cmd: List[str]) -> None:
    print(f"[cmd] {' '.join(shlex.quote(c) for c in cmd)}", flush=True)
    subprocess.run(cmd, check=True)


def _latest_checkpoint(out_dir: Path) -> Optional[Path]:
    ckpt_dir = out_dir / "checkpoints"
    if not ckpt_dir.is_dir():
        return None
    ckpts = sorted([p for p in ckpt_dir.glob("step_*") if p.is_dir()])
    return ckpts[-1] if ckpts else None


_STEP_RE = re.compile(r"step_(\d+)$")


def _checkpoint_step(path: Optional[Path]) -> int:
    if path is None:
        return 0
    m = _STEP_RE.search(path.name)
    if not m:
        return 0
    try:
        return int(m.group(1))
    except Exception:
        return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="GSM8K RL loop: rollout -> train -> rollout (use latest weights each iter)."
    )
    parser.add_argument("--dataset_dir", type=str, default="/root/gsm8k")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--tokenizer_path", type=str, default="./model")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Initial checkpoint dir (used for rollout generation, and as --init_from for the 1st iter).",
    )
    parser.add_argument("--out_dir", type=str, required=True)

    parser.add_argument("--download", action=argparse.BooleanOptionalAction, default=False, help="Run dataset download first.")
    parser.add_argument("--repo_id", type=str, default="zhuzilin/gsm8k")
    parser.add_argument("--repo_type", type=str, default="dataset")

    parser.add_argument(
        "--warmup_sft_steps",
        type=int,
        default=0,
        help="Optional: run a short GSM8K SFT warmup (mlx_train.train --task sft) before RL to bootstrap positives.",
    )
    parser.add_argument("--warmup_sft_limit", type=int, default=2048, help="How many GSM8K samples to write for warmup (0 = all).")
    parser.add_argument("--warmup_sft_seq_len", type=int, default=1024)
    parser.add_argument("--warmup_sft_batch_size", type=int, default=4)
    parser.add_argument("--warmup_sft_lr", type=float, default=1e-4)
    parser.add_argument("--warmup_sft_answer_mode", type=str, default="full", choices=["full", "final"])
    parser.add_argument("--warmup_sft_out_dir", type=str, default=None, help="Default: <out_dir>/warmup_sft")
    parser.add_argument("--warmup_sft_regen", action=argparse.BooleanOptionalAction, default=False)

    parser.add_argument("--num_rollouts", type=int, default=512, help="Prompts per iter.")
    parser.add_argument("--samples_per_prompt", type=int, default=1)
    parser.add_argument("--min_positive", type=int, default=0)
    parser.add_argument("--max_total_rollouts", type=int, default=0, help="0 = num_rollouts*samples_per_prompt")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--max_new_tokens", type=int, default=256)

    parser.add_argument("--max_steps", type=int, default=1000, help="Total RL steps across all iters.")
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--steps_per_iter", type=int, default=None, help="Default: ceil(max_steps / iters).")
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--clean_buffers", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--run_eval", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gsm8k_eval_split", type=str, default="test")
    parser.add_argument("--gsm8k_eval_num", type=int, default=200, help="0 = all")

    parser.add_argument("--run_bench", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--bench_suite", type=str, default="all")
    parser.add_argument("--bench_max_new_tokens", type=int, default=32)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    buffer_dir = out_dir / "buffer"
    buffer_dir.mkdir(parents=True, exist_ok=True)

    py = sys.executable
    init_ckpt = Path(args.checkpoint)

    if bool(args.download):
        _run(
            [
                py,
                "-m",
                "mlx_train.rl_gsm8k.download",
                "--repo_id",
                str(args.repo_id),
                "--repo_type",
                str(args.repo_type),
                "--local_dir",
                str(args.dataset_dir),
            ]
        )

    warmup_steps = int(args.warmup_sft_steps)
    if warmup_steps > 0 and _latest_checkpoint(out_dir) is None:
        warmup_out_dir = (
            Path(args.warmup_sft_out_dir)
            if args.warmup_sft_out_dir is not None
            else (out_dir / "warmup_sft")
        )
        warmup_out_dir.mkdir(parents=True, exist_ok=True)
        warmup_jsonl = warmup_out_dir / "gsm8k_sft.jsonl"

        if bool(args.warmup_sft_regen) or not warmup_jsonl.exists():
            cmd = [
                py,
                "-m",
                "mlx_train.rl_gsm8k.prepare_sft",
                "--dataset_dir",
                str(args.dataset_dir),
                "--split",
                str(args.split),
                "--out_path",
                os.fspath(warmup_jsonl),
                "--answer_mode",
                str(args.warmup_sft_answer_mode),
            ]
            limit = int(args.warmup_sft_limit)
            if limit > 0:
                cmd.extend(["--limit", str(limit)])
            _run(cmd)

        warmup_latest = _latest_checkpoint(warmup_out_dir)
        if warmup_latest is None or _checkpoint_step(warmup_latest) < warmup_steps:
            train_cmd = [
                py,
                "-m",
                "mlx_train.train",
                "--task",
                "sft",
                "--tokenizer_path",
                str(args.tokenizer_path),
                "--data_path",
                os.fspath(warmup_jsonl),
                "--out_dir",
                os.fspath(warmup_out_dir),
                "--seq_len",
                str(int(args.warmup_sft_seq_len)),
                "--batch_size",
                str(int(args.warmup_sft_batch_size)),
                "--learning_rate",
                str(float(args.warmup_sft_lr)),
                "--epochs",
                "999999",
                "--max_steps",
                str(int(warmup_steps)),
            ]
            if warmup_latest is None:
                train_cmd.extend(["--init_from", os.fspath(init_ckpt)])
            else:
                train_cmd.extend(["--resume", os.fspath(warmup_latest)])
            _run(train_cmd)

        warmup_latest = _latest_checkpoint(warmup_out_dir)
        if warmup_latest is None:
            raise RuntimeError(f"Warmup finished but no checkpoint found under {warmup_out_dir}/checkpoints")
        init_ckpt = warmup_latest
        print(f"[warmup] init_ckpt={init_ckpt}", flush=True)

    iters = max(int(args.iters), 1)
    total_steps = max(int(args.max_steps), 1)
    steps_per_iter = (
        int(args.steps_per_iter)
        if args.steps_per_iter is not None
        else (total_steps + iters - 1) // iters
    )
    if steps_per_iter <= 0:
        raise ValueError("steps_per_iter must be > 0")

    for i in range(iters):
        latest = _latest_checkpoint(out_dir)
        cur_step = _checkpoint_step(latest)
        if cur_step >= total_steps:
            print(f"[loop] reached max_steps={total_steps} (cur_step={cur_step}); stop", flush=True)
            break

        rollout_ckpt = latest if latest is not None else init_ckpt
        next_step = min(total_steps, cur_step + steps_per_iter)
        iter_buf = buffer_dir / f"iter_{i:03d}.jsonl"
        if bool(args.clean_buffers) and iter_buf.exists():
            iter_buf.unlink()

        print(
            f"[loop] iter={i:03d} rollout_ckpt={rollout_ckpt} steps={cur_step}->{next_step} buffer={iter_buf}",
            flush=True,
        )

        _run(
            [
                py,
                "-m",
                "mlx_train.rl_gsm8k.rollout",
                "--dataset_dir",
                str(args.dataset_dir),
                "--split",
                str(args.split),
                "--tokenizer_path",
                str(args.tokenizer_path),
                "--checkpoint",
                os.fspath(rollout_ckpt),
                "--out_buffer",
                os.fspath(iter_buf),
                "--num_rollouts",
                str(int(args.num_rollouts)),
                "--samples_per_prompt",
                str(int(args.samples_per_prompt)),
                "--min_positive",
                str(int(args.min_positive)),
                "--max_total_rollouts",
                str(int(args.max_total_rollouts)),
                "--seed",
                str(int(args.seed) + i * 10007),
                "--temperature",
                str(float(args.temperature)),
                "--top_p",
                str(float(args.top_p)),
                "--max_new_tokens",
                str(int(args.max_new_tokens)),
            ]
        )

        train_cmd = [
            py,
            "-m",
            "mlx_train.rl_gsm8k.train",
            "--tokenizer_path",
            str(args.tokenizer_path),
            "--buffer_path",
            os.fspath(iter_buf),
            "--out_dir",
            str(args.out_dir),
            "--max_steps",
            str(int(next_step)),
            "--seq_len",
            str(int(args.seq_len)),
            "--batch_size",
            str(int(args.batch_size)),
        ]
        if latest is None:
            train_cmd.extend(["--init_from", os.fspath(init_ckpt)])
        else:
            train_cmd.extend(["--resume", os.fspath(latest)])
        _run(train_cmd)

    latest = _latest_checkpoint(out_dir)
    if latest is None:
        print(f"[warn] no checkpoint found under {out_dir}/checkpoints; skip eval/bench", flush=True)
        return

    if bool(args.run_eval):
        eval_buf = os.fspath(out_dir / f"eval_gsm8k_{args.gsm8k_eval_split}.jsonl")
        _run(
            [
                py,
                "-m",
                "mlx_train.rl_gsm8k.rollout",
                "--dataset_dir",
                str(args.dataset_dir),
                "--split",
                str(args.gsm8k_eval_split),
                "--tokenizer_path",
                str(args.tokenizer_path),
                "--checkpoint",
                os.fspath(latest),
                "--out_buffer",
                eval_buf,
                "--num_rollouts",
                str(int(args.gsm8k_eval_num)),
                "--seed",
                str(int(args.seed)),
                "--temperature",
                "0.0",
                "--top_p",
                "1.0",
                "--max_new_tokens",
                str(int(args.max_new_tokens)),
            ]
        )

    if bool(args.run_bench):
        _run(
            [
                py,
                "-m",
                "mlx_train.bench",
                "--checkpoint",
                os.fspath(latest),
                "--tokenizer_path",
                str(args.tokenizer_path),
                "--suite",
                str(args.bench_suite),
                "--seed",
                str(int(args.seed)),
                "--max_new_tokens",
                str(int(args.bench_max_new_tokens)),
                "--no_ollama",
            ]
        )


if __name__ == "__main__":
    main()
