from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import time
from typing import Any, Dict, Optional

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import mlx.utils as mlx_utils
from transformers import AutoTokenizer

from .config import MiniLLMConfig, minillm_200mb
from .data import make_batch_iterator, resolve_jsonl_paths
from .download import resolve_data_path_spec
from .model import MiniLLMForCausalLM, count_parameters, parameters_bytes


def set_seed(seed: int) -> None:
    mx.random.seed(seed)


def cosine_lr(
    step: int,
    total_steps: int,
    base_lr: float,
    *,
    warmup_steps: int = 0,
    min_lr_ratio: float = 0.1,
) -> float:
    if total_steps <= 0:
        return base_lr
    if warmup_steps > 0 and step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return base_lr * (min_lr_ratio + (1.0 - min_lr_ratio) * cosine)


def save_optimizer_state(optimizer: optim.Optimizer, path: str) -> None:
    flat: Dict[str, Any] = {}
    mlx_utils.tree_flatten(optimizer.state, destination=flat)
    mx.savez(path, **flat)


def load_optimizer_state(optimizer: optim.Optimizer, path: str) -> None:
    flat = dict(mx.load(path))
    optimizer.state = mlx_utils.tree_unflatten(flat)


def loss_fn(
    model: MiniLLMForCausalLM, x: mx.array, y: mx.array, loss_mask: mx.array
) -> mx.array:
    logits = model(x)  # [B, T, V]
    bsz, seq_len, vocab = logits.shape
    loss = nn.losses.cross_entropy(
        logits.reshape(bsz * seq_len, vocab),
        y.reshape(bsz * seq_len),
        reduction="none",
    ).reshape(bsz, seq_len)

    mask = loss_mask.astype(mx.float32)
    denom = mx.maximum(mx.sum(mask), mx.array(1.0, dtype=mx.float32))
    return mx.sum(loss * mask) / denom


def make_config(args, tokenizer) -> MiniLLMConfig:
    if args.preset == "200mb":
        cfg = minillm_200mb()
    elif args.preset == "tiny":
        cfg = MiniLLMConfig(
            hidden_size=256,
            num_hidden_layers=4,
            num_attention_heads=4,
            num_key_value_heads=2,
            vocab_size=tokenizer.vocab_size,
            dropout=args.dropout,
            rope_theta=args.rope_theta,
            max_position_embeddings=args.max_position_embeddings,
            use_moe=False,
        ).finalize()
    elif args.preset == "custom":
        cfg = MiniLLMConfig(
            hidden_size=args.hidden_size,
            num_hidden_layers=args.num_hidden_layers,
            num_attention_heads=args.num_attention_heads,
            num_key_value_heads=args.num_key_value_heads,
            vocab_size=tokenizer.vocab_size
            if args.vocab_size is None
            else args.vocab_size,
            dropout=args.dropout,
            rope_theta=args.rope_theta,
            max_position_embeddings=args.max_position_embeddings,
            use_moe=False,
        ).finalize()
    else:
        raise ValueError(f"Unknown preset: {args.preset}")

    if cfg.vocab_size != tokenizer.vocab_size:
        raise ValueError(
            f"Config vocab_size={cfg.vocab_size} != tokenizer.vocab_size={tokenizer.vocab_size}"
        )

    return cfg


def prune_checkpoints(ckpt_dir: str, *, keep_last: int) -> None:
    if keep_last <= 0:
        return

    pat = re.compile(r"^step_(\d+)$")
    ckpts: list[tuple[int, str]] = []
    for name in os.listdir(ckpt_dir):
        m = pat.match(name)
        if not m:
            continue
        try:
            step = int(m.group(1))
        except ValueError:
            continue
        ckpts.append((step, os.path.join(ckpt_dir, name)))

    ckpts.sort(key=lambda x: x[0])
    for _, path in ckpts[:-keep_last]:
        shutil.rmtree(path, ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="MiniLLM (MLX) training")
    parser.add_argument("--tokenizer_path", type=str, default="./model")
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="JSONL file/dir/glob; can be comma-separated.",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./dataset",
        help="Download/cache directory when `data_path` contains URLs or `minimind:*` specs.",
    )
    parser.add_argument(
        "--hf_dataset_repo",
        type=str,
        default="jingyaogong/minimind_dataset",
        help="HuggingFace dataset repo used by `minimind:*` specs.",
    )
    parser.add_argument(
        "--hf_endpoint",
        type=str,
        default=None,
        help="Optional HuggingFace endpoint/mirror.",
    )
    parser.add_argument(
        "--force_download",
        action="store_true",
        help="Re-download remote datasets even if present.",
    )
    parser.add_argument(
        "--max_download_mb",
        type=int,
        default=2048,
        help="Safety guard for remote dataset downloads (MB); set 0 to disable.",
    )
    parser.add_argument(
        "--task", type=str, choices=["pretrain", "sft"], default="pretrain"
    )

    parser.add_argument(
        "--preset", type=str, choices=["200mb", "tiny", "custom"], default="200mb"
    )
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_hidden_layers", type=int, default=8)
    parser.add_argument("--num_attention_heads", type=int, default=8)
    parser.add_argument("--num_key_value_heads", type=int, default=2)
    parser.add_argument("--vocab_size", type=int, default=None)
    parser.add_argument("--max_position_embeddings", type=int, default=32768)
    parser.add_argument("--rope_theta", type=float, default=1_000_000.0)
    parser.add_argument("--dropout", type=float, default=0.0)

    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--accum_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--shuffle_buffer", type=int, default=2048)
    parser.add_argument(
        "--keep_last_checkpoints",
        type=int,
        default=3,
        help="Keep only the latest N checkpoints under out_dir/checkpoints (0 to disable pruning).",
    )

    parser.add_argument(
        "--dtype",
        type=str,
        choices=["float16", "bfloat16", "float32"],
        default="bfloat16",
    )
    parser.add_argument("--out_dir", type=str, default="./out/mlx")
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--save_interval", type=int, default=200)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument(
        "--profile_timing",
        action="store_true",
        help="Print per-step timing breakdown (adds extra synchronization; slower).",
    )
    parser.add_argument(
        "--profile_warmup_steps",
        type=int,
        default=2,
        help="Ignore the first N steps for timing stats when --profile_timing is enabled.",
    )
    parser.add_argument(
        "--metal_capture",
        type=str,
        default=None,
        help="Optional path to write a Metal capture (.gputrace).",
    )
    parser.add_argument(
        "--metal_capture_steps",
        type=int,
        default=1,
        help="How many optimizer steps to capture when --metal_capture is set.",
    )
    parser.add_argument(
        "--metal_capture_start_step",
        type=int,
        default=None,
        help="Which global step to start capture (default: first step after resume).",
    )

    parser.add_argument(
        "--init_from",
        type=str,
        default=None,
        help="Initialise weights from a checkpoint dir (containing model.safetensors) or a .safetensors file.",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Checkpoint directory produced by this script.",
    )

    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    ckpt_dir = os.path.join(args.out_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    set_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    if tokenizer.pad_token_id is None:
        raise ValueError("Tokenizer must define pad_token_id.")

    data_spec = resolve_data_path_spec(
        args.data_path,
        task=args.task,
        data_dir=args.data_dir,
        hf_repo_id=args.hf_dataset_repo,
        hf_endpoint=args.hf_endpoint,
        force_download=args.force_download,
        max_download_mb=args.max_download_mb,
    )
    paths = resolve_jsonl_paths(data_spec)

    cfg = make_config(args, tokenizer)
    model = MiniLLMForCausalLM(cfg)

    if args.init_from and args.resume:
        raise ValueError("`--init_from` and `--resume` are mutually exclusive.")

    start_step = 0
    resume_optimizer_path = None
    if args.resume:
        model_path = os.path.join(args.resume, "model.safetensors")
        opt_path = os.path.join(args.resume, "optimizer.npz")
        state_path = os.path.join(args.resume, "state.json")
        if not os.path.isfile(model_path):
            raise FileNotFoundError(model_path)
        model.load_weights(model_path)
        if os.path.isfile(state_path):
            with open(state_path, "r", encoding="utf-8") as f:
                state = json.load(f)
            start_step = int(state.get("step", 0))
        if os.path.isfile(opt_path):
            resume_optimizer_path = opt_path
        print(f"[resume] step={start_step} from {args.resume}")
    elif args.init_from:
        init_path = args.init_from
        if os.path.isdir(init_path):
            init_path = os.path.join(init_path, "model.safetensors")
        if not os.path.isfile(init_path):
            raise FileNotFoundError(init_path)
        model.load_weights(init_path)
        print(f"[init] loaded weights from {init_path}")

    # Dtype casting for memory/throughput.
    dtype_map = {"float16": mx.float16, "bfloat16": mx.bfloat16, "float32": mx.float32}
    model.apply(lambda p: p.astype(dtype_map[args.dtype]))

    params = model.parameters()
    n_params = count_parameters(params)
    n_bytes = parameters_bytes(params)
    print(
        f"[model] params={n_params / 1e6:.2f}M | approx_size={n_bytes / 1024 / 1024:.1f} MiB | dtype={args.dtype}"
    )

    optimizer = optim.AdamW(
        learning_rate=args.learning_rate, weight_decay=args.weight_decay
    )
    if resume_optimizer_path is not None:
        load_optimizer_state(optimizer, resume_optimizer_path)

    # Estimate total steps for lr scheduling (best-effort).
    total_steps = (
        args.max_steps if args.max_steps is not None else (args.epochs * 10**9)
    )

    value_and_grad = nn.value_and_grad(model, loss_fn)
    model.train()

    global_step = start_step
    t0 = time.time()
    seen_tokens = 0
    profile_timing = bool(args.profile_timing)
    timing_window_steps = 0
    timing_data_s = 0.0
    timing_to_mx_s = 0.0
    timing_fwd_bwd_s = 0.0
    timing_clip_s = 0.0
    timing_opt_s = 0.0
    timing_total_s = 0.0

    metal_capture_path: Optional[str] = args.metal_capture
    metal_capture_start_step = (
        start_step
        if args.metal_capture_start_step is None
        else int(args.metal_capture_start_step)
    )
    metal_capture_end_step: Optional[int] = None
    metal_capturing = False
    if metal_capture_path is not None and not metal_capture_path.endswith(".gputrace"):
        raise ValueError("--metal_capture must end with .gputrace")

    if args.max_steps is not None and start_step >= int(args.max_steps):
        print(
            f"[warn] start_step={start_step} >= max_steps={args.max_steps}; "
            "no training steps will run (use a fresh --out_dir or increase --max_steps)."
        )

    def save_checkpoint(step: int) -> str:
        path = os.path.join(ckpt_dir, f"step_{step:08d}")
        os.makedirs(path, exist_ok=True)
        model.save_weights(os.path.join(path, "model.safetensors"))
        save_optimizer_state(optimizer, os.path.join(path, "optimizer.npz"))
        with open(os.path.join(path, "config.json"), "w", encoding="utf-8") as f:
            json.dump(cfg.to_dict(), f, ensure_ascii=False, indent=2)
        with open(os.path.join(path, "state.json"), "w", encoding="utf-8") as f:
            json.dump(
                {"step": step, "seen_tokens": seen_tokens, "args": vars(args)},
                f,
                ensure_ascii=False,
                indent=2,
            )
        prune_checkpoints(ckpt_dir, keep_last=args.keep_last_checkpoints)
        return path

    try:
        for epoch in range(args.epochs):
            batch_iter = iter(
                make_batch_iterator(
                    paths=paths,
                    tokenizer=tokenizer,
                    task=args.task,
                    seq_len=args.seq_len,
                    batch_size=args.batch_size,
                    shuffle_buffer=args.shuffle_buffer,
                    seed=args.seed + epoch,
                )
            )

            while True:
                if args.max_steps is not None and global_step >= args.max_steps:
                    break

                if (
                    metal_capture_path is not None
                    and not metal_capturing
                    and global_step == metal_capture_start_step
                ):
                    try:
                        mx.metal.start_capture(metal_capture_path)
                    except Exception as e:
                        print(f"[metal] capture failed to start: {e}")
                        metal_capture_path = None
                        metal_capturing = False
                        metal_capture_end_step = None
                    else:
                        metal_capturing = True
                        metal_capture_end_step = global_step + int(
                            args.metal_capture_steps
                        )
                        print(
                            f"[metal] capture started: {metal_capture_path} (steps={args.metal_capture_steps})"
                        )

                step_t0 = time.perf_counter() if profile_timing else 0.0
                # Gradient accumulation across `accum_steps` micro-batches.
                grad_accum: Any = None
                loss_accum = mx.array(0.0, dtype=mx.float32)
                micro_batches = 0

                for _ in range(args.accum_steps):
                    try:
                        data_t0 = time.perf_counter() if profile_timing else 0.0
                        batch = next(batch_iter)
                        if profile_timing:
                            timing_data_s += time.perf_counter() - data_t0
                    except StopIteration:
                        break

                    to_mx_t0 = time.perf_counter() if profile_timing else 0.0
                    x = mx.array(batch.x, dtype=mx.int32)
                    y = mx.array(batch.y, dtype=mx.int32)
                    m = mx.array(batch.loss_mask, dtype=mx.float32)
                    if profile_timing:
                        mx.eval(x, y, m)
                        timing_to_mx_s += time.perf_counter() - to_mx_t0

                    fwd_bwd_t0 = time.perf_counter() if profile_timing else 0.0
                    loss, grads = value_and_grad(model, x, y, m)
                    if profile_timing:
                        mx.eval(loss, grads)
                        timing_fwd_bwd_s += time.perf_counter() - fwd_bwd_t0
                    loss_accum = loss_accum + loss.astype(mx.float32)
                    micro_batches += 1

                    if grad_accum is None:
                        grad_accum = grads
                    else:
                        grad_accum = mlx_utils.tree_map(
                            lambda a, b: a + b, grad_accum, grads
                        )

                if micro_batches == 0:
                    break  # finished this epoch

                assert grad_accum is not None
                grad_accum = mlx_utils.tree_map(lambda g: g / micro_batches, grad_accum)
                grad_norm: Optional[mx.array] = None
                if args.grad_clip > 0:
                    clip_t0 = time.perf_counter() if profile_timing else 0.0
                    grad_accum, grad_norm = optim.clip_grad_norm(
                        grad_accum, max_norm=args.grad_clip
                    )
                    if profile_timing:
                        mx.eval(grad_accum, grad_norm)
                        timing_clip_s += time.perf_counter() - clip_t0

                lr = cosine_lr(
                    global_step,
                    total_steps,
                    args.learning_rate,
                    warmup_steps=args.warmup_steps,
                )
                opt_t0 = time.perf_counter() if profile_timing else 0.0
                optimizer.learning_rate = lr
                optimizer.update(model, grad_accum)
                mx.eval(model.parameters(), optimizer.state, loss_accum)
                if profile_timing:
                    timing_opt_s += time.perf_counter() - opt_t0

                global_step += 1
                seen_tokens += int(args.batch_size) * int(args.seq_len) * micro_batches
                if profile_timing:
                    timing_total_s += time.perf_counter() - step_t0
                    if global_step > start_step + max(0, int(args.profile_warmup_steps)):
                        timing_window_steps += 1
                    else:
                        timing_data_s = 0.0
                        timing_to_mx_s = 0.0
                        timing_fwd_bwd_s = 0.0
                        timing_clip_s = 0.0
                        timing_opt_s = 0.0
                        timing_total_s = 0.0
                        timing_window_steps = 0

                if metal_capturing and metal_capture_end_step is not None:
                    if global_step >= metal_capture_end_step:
                        mx.metal.stop_capture()
                        metal_capturing = False
                        print(f"[metal] capture saved: {metal_capture_path}")
                        metal_capture_path = None

                if global_step % args.log_interval == 0:
                    dt = time.time() - t0
                    tok_s = seen_tokens / max(dt, 1e-6)
                    avg_loss = float(loss_accum.item()) / micro_batches
                    timing_msg = ""
                    if profile_timing and timing_window_steps > 0:
                        to_ms = 1000.0 / timing_window_steps
                        timing_msg = (
                            f" | step_ms={(timing_total_s * to_ms):.1f}"
                            f" data_ms={(timing_data_s * to_ms):.1f}"
                            f" to_mx_ms={(timing_to_mx_s * to_ms):.1f}"
                            f" fwd_bwd_ms={(timing_fwd_bwd_s * to_ms):.1f}"
                            f" clip_ms={(timing_clip_s * to_ms):.1f}"
                            f" opt_ms={(timing_opt_s * to_ms):.1f}"
                        )
                    print(
                        f"[train] step={global_step} epoch={epoch + 1}/{args.epochs} "
                        f"loss={avg_loss:.4f} lr={lr:.2e} tok/s={tok_s:.0f}{timing_msg}"
                    )

                if global_step % args.save_interval == 0:
                    path = save_checkpoint(global_step)
                    print(f"[ckpt] saved {path}")

            if args.max_steps is not None and global_step >= args.max_steps:
                break

    except KeyboardInterrupt:
        print("\n[train] interrupted, saving last checkpoint...")
    finally:
        path = save_checkpoint(global_step)
        print(f"[ckpt] saved {path}")


if __name__ == "__main__":
    main()
