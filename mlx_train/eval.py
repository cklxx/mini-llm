from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
from transformers import AutoTokenizer

from .config import MiniLLMConfig
from .data import make_batch_iterator, resolve_jsonl_paths
from .download import resolve_data_path_spec
from .model import MiniLLMForCausalLM


def load_config(checkpoint_dir: Path) -> MiniLLMConfig:
    config_path = checkpoint_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config.json in checkpoint dir: {checkpoint_dir}")
    with open(config_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return MiniLLMConfig(**data).finalize()


def loss_sum_and_tokens(
    model: MiniLLMForCausalLM, x: mx.array, y: mx.array, loss_mask: mx.array
) -> Tuple[mx.array, mx.array]:
    logits = model(x)  # [B, T, V]
    bsz, seq_len, vocab = logits.shape
    loss = nn.losses.cross_entropy(
        logits.reshape(bsz * seq_len, vocab),
        y.reshape(bsz * seq_len),
        reduction="none",
    ).reshape(bsz, seq_len)

    mask = loss_mask.astype(mx.float32)
    return mx.sum(loss * mask), mx.sum(mask)


def main() -> None:
    parser = argparse.ArgumentParser(description="MiniLLM (MLX) eval: average loss / perplexity")
    parser.add_argument("--tokenizer_path", type=str, default="./model")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Checkpoint directory produced by mlx_train/train.py (contains model.safetensors + config.json).",
    )
    parser.add_argument("--task", type=str, default="pretrain", choices=["pretrain", "sft"])
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="JSONL file/dir/glob; can be comma-separated. Supports minimind:* specs (downloads if needed).",
    )
    parser.add_argument("--data_dir", type=str, default="./dataset/minimind")
    parser.add_argument("--hf_repo_id", type=str, default="jingyaogong/minimind_dataset")
    parser.add_argument("--hf_endpoint", type=str, default=None)
    parser.add_argument("--force_download", action="store_true")
    parser.add_argument("--max_download_mb", type=int, default=2048)
    parser.add_argument("--seq_len", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--shuffle_buffer", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument(
        "--max_batches",
        type=int,
        default=100,
        help="Evaluate at most N batches (0 = full dataset; default: 100).",
    )
    parser.add_argument("--log_interval", type=int, default=20)
    parser.add_argument("--compile", action="store_true", help="Compile the forward loss for faster eval.")

    args = parser.parse_args()

    mx.random.seed(args.seed)

    ckpt = Path(args.checkpoint)
    if not ckpt.exists() or not ckpt.is_dir():
        raise FileNotFoundError(f"Checkpoint must be a directory: {ckpt}")

    resolved = resolve_data_path_spec(
        args.data_path,
        task=args.task,
        data_dir=args.data_dir,
        hf_repo_id=args.hf_repo_id,
        hf_endpoint=args.hf_endpoint,
        force_download=bool(args.force_download),
        max_download_mb=int(args.max_download_mb),
    )
    paths = resolve_jsonl_paths(resolved)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    cfg = load_config(ckpt)
    model = MiniLLMForCausalLM(cfg)
    model.load_weights(os.fspath(ckpt / "model.safetensors"))
    model.eval()

    def eval_step(x: mx.array, y: mx.array, m: mx.array) -> Tuple[mx.array, mx.array]:
        return loss_sum_and_tokens(model, x, y, m)

    eval_fn = eval_step
    if args.compile:
        eval_fn = mx.compile(eval_fn, inputs={"model": model})
        x0 = mx.zeros((args.batch_size, args.seq_len), dtype=mx.int32)
        y0 = mx.zeros((args.batch_size, args.seq_len), dtype=mx.int32)
        m0 = mx.ones((args.batch_size, args.seq_len), dtype=mx.float32)
        out0 = eval_fn(x0, y0, m0)
        mx.eval(out0)

    it = make_batch_iterator(
        paths=paths,
        tokenizer=tokenizer,
        task=args.task,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        shuffle_buffer=args.shuffle_buffer,
        seed=args.seed,
    )

    total_loss = 0.0
    total_tokens = 0.0
    batches = 0
    t0 = time.time()
    for batch in it:
        x = mx.array(batch.x, dtype=mx.int32)
        y = mx.array(batch.y, dtype=mx.int32)
        m = mx.array(batch.loss_mask, dtype=mx.float32)
        loss_sum, tokens = eval_fn(x, y, m)
        mx.eval(loss_sum, tokens)

        total_loss += float(loss_sum.item())
        total_tokens += float(tokens.item())
        batches += 1

        if args.log_interval > 0 and batches % args.log_interval == 0:
            avg = total_loss / max(total_tokens, 1.0)
            elapsed = max(time.time() - t0, 1e-9)
            tok_s = total_tokens / elapsed
            print(f"[eval] batches={batches} avg_loss={avg:.4f} tok/s={tok_s:.0f}")

        if args.max_batches > 0 and batches >= args.max_batches:
            break

    avg_loss = total_loss / max(total_tokens, 1.0)
    ppl = math.exp(min(avg_loss, 100.0))
    elapsed = max(time.time() - t0, 1e-9)
    tok_s = total_tokens / elapsed
    print(
        f"[eval] done batches={batches} tokens={int(total_tokens)} avg_loss={avg_loss:.4f} ppl={ppl:.2f} tok/s={tok_s:.0f}"
    )


if __name__ == "__main__":
    main()
