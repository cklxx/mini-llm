from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import mlx.utils as mlx_utils
from transformers import AutoTokenizer

from ..config import MiniLLMConfig
from ..infer import load_config
from ..model import MiniLLMForCausalLM
from ..optim import make_optimizer
from .buffer import JsonlRolloutBuffer


def cosine_lr(
    step: int, total_steps: int, base_lr: float, *, warmup_steps: int = 0, min_lr_ratio: float = 0.1
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


def shuffle_stream(items: Iterable[Dict[str, Any]], *, buffer_size: int, seed: int) -> Iterator[Dict[str, Any]]:
    if buffer_size <= 0:
        yield from items
        return

    rng = random.Random(seed)
    buf: List[Dict[str, Any]] = []
    for item in items:
        if len(buf) < buffer_size:
            buf.append(item)
            continue
        idx = rng.randrange(len(buf))
        yield buf[idx]
        buf[idx] = item
    rng.shuffle(buf)
    yield from buf


def _pad_or_truncate(ids: List[int], *, length: int, pad_id: int) -> List[int]:
    if len(ids) >= length:
        return ids[:length]
    return ids + [pad_id] * (length - len(ids))


@dataclass(frozen=True)
class TokenizedRollout:
    x: List[int]
    y: List[int]
    mask: List[int]
    reward: float


def tokenize_rollout(
    *,
    tokenizer,
    messages: Sequence[Dict[str, Any]],
    response: str,
    reward: float,
    seq_len: int,
    pad_id: int,
) -> Optional[TokenizedRollout]:
    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    prompt_ids: List[int] = tokenizer.encode(prompt_text, add_special_tokens=False)

    resp_ids: List[int] = tokenizer.encode(str(response), add_special_tokens=False)
    ids = list(prompt_ids) + list(resp_ids)
    if tokenizer.eos_token_id is not None:
        ids.append(int(tokenizer.eos_token_id))

    ids = _pad_or_truncate(ids, length=seq_len + 1, pad_id=pad_id)
    x = ids[:-1]
    y = ids[1:]

    # Tokens to train: response tokens (+ optional EOS), predicted starting from the last prompt token.
    start = max(0, len(prompt_ids) - 1)
    end = min(seq_len, len(prompt_ids) + len(resp_ids))
    if start >= seq_len or end <= start:
        return None

    mask = [0] * seq_len
    for i in range(start, end):
        mask[i] = 1
    # Never train on padding.
    for i, tok in enumerate(y):
        if tok == pad_id:
            mask[i] = 0

    if sum(mask) == 0:
        return None
    return TokenizedRollout(x=x, y=y, mask=mask, reward=float(reward))


def rl_loss_fn(
    model: MiniLLMForCausalLM,
    x: mx.array,
    y: mx.array,
    mask: mx.array,
    advantages: mx.array,
) -> mx.array:
    logits = model(x)  # [B, T, V]
    bsz, seq_len, vocab = logits.shape
    per_tok = nn.losses.cross_entropy(
        logits.reshape(bsz * seq_len, vocab),
        y.reshape(bsz * seq_len),
        reduction="none",
    ).reshape(bsz, seq_len)

    mask_f = mask.astype(mx.float32)
    adv = advantages.astype(mx.float32).reshape(bsz, 1)

    denom = mx.maximum(mx.sum(mask_f), mx.array(1.0, dtype=mx.float32))
    return mx.sum(per_tok * mask_f * adv) / denom


def load_model_from_checkpoint(checkpoint_dir: Path) -> Tuple[MiniLLMConfig, MiniLLMForCausalLM]:
    cfg = load_config(checkpoint_dir)
    model = MiniLLMForCausalLM(cfg)
    model.load_weights(os.fspath(checkpoint_dir / "model.safetensors"))
    return cfg, model


def resolve_checkpoint_dir(path: str) -> Path:
    p = Path(path)
    if p.is_dir():
        return p
    raise FileNotFoundError(f"Expected checkpoint dir, got: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="MiniLLM (MLX) RL training on a rollout buffer (GSM8K reward).")
    parser.add_argument("--tokenizer_path", type=str, default="./model")
    parser.add_argument("--buffer_path", type=str, required=True, help="Rollout buffer JSONL path.")

    parser.add_argument("--resume", type=str, default=None, help="Resume from a checkpoint dir (step_XXXXXXXX).")
    parser.add_argument("--init_from", type=str, default=None, help="Init weights from a checkpoint dir.")

    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"])

    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--shuffle_buffer", type=int, default=0)
    parser.add_argument("--follow_buffer", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--min_reward", type=float, default=None, help="Filter rollouts with reward < min_reward.")

    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--adv_norm", action=argparse.BooleanOptionalAction, default=True, help="Normalize advantages per batch.")
    parser.add_argument("--seed", type=int, default=1337)

    parser.add_argument("--save_interval", type=int, default=200)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--keep_last_checkpoints", type=int, default=3)
    args = parser.parse_args()

    mx.random.seed(int(args.seed))
    random.seed(int(args.seed))

    out_dir = Path(args.out_dir)
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    if tokenizer.pad_token_id is None:
        raise ValueError("Tokenizer must have pad_token_id set (or an eos_token_id to fall back to).")
    pad_id = int(tokenizer.pad_token_id)

    start_step = 0
    resume_dir: Optional[Path] = None
    if args.resume:
        resume_dir = resolve_checkpoint_dir(str(args.resume))
        cfg, model = load_model_from_checkpoint(resume_dir)
        state_path = resume_dir / "state.json"
        if state_path.exists():
            try:
                with open(state_path, "r", encoding="utf-8") as f:
                    state = json.load(f)
                start_step = int(state.get("step") or 0)
            except Exception:
                start_step = 0
        print(f"[resume] step={start_step} from {resume_dir}", flush=True)
    else:
        if not args.init_from:
            raise ValueError("Either --resume or --init_from must be provided.")
        init_dir = resolve_checkpoint_dir(str(args.init_from))
        cfg, model = load_model_from_checkpoint(init_dir)
        print(f"[init] loaded weights from {init_dir}", flush=True)

    dtype_map = {"float16": mx.float16, "bfloat16": mx.bfloat16, "float32": mx.float32}
    model.apply(lambda p: p.astype(dtype_map[str(args.dtype)]))
    model.train()

    optimizer = make_optimizer(
        name="adamw",
        learning_rate=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
        state_dtype="float32",
    )
    if resume_dir is not None:
        opt_path = resume_dir / "optimizer.npz"
        if opt_path.exists() and opt_path.stat().st_size > 0:
            load_optimizer_state(optimizer, os.fspath(opt_path))

    buffer = JsonlRolloutBuffer(args.buffer_path)

    def save_checkpoint(step: int) -> str:
        path = ckpt_dir / f"step_{step:08d}"
        path.mkdir(parents=True, exist_ok=True)

        model_path = path / "model.safetensors"
        tmp_model = Path(os.fspath(model_path) + ".tmp.safetensors")
        model.save_weights(os.fspath(tmp_model))
        os.replace(tmp_model, model_path)

        opt_path = path / "optimizer.npz"
        tmp_opt = Path(os.fspath(opt_path) + ".tmp.npz")
        save_optimizer_state(optimizer, os.fspath(tmp_opt))
        os.replace(tmp_opt, opt_path)

        cfg_path = path / "config.json"
        tmp_cfg = Path(os.fspath(cfg_path) + ".tmp")
        with open(tmp_cfg, "w", encoding="utf-8") as f:
            json.dump(cfg.to_dict(), f, ensure_ascii=False, indent=2)
        os.replace(tmp_cfg, cfg_path)

        state_path = path / "state.json"
        tmp_state = Path(os.fspath(state_path) + ".tmp")
        with open(tmp_state, "w", encoding="utf-8") as f:
            json.dump({"step": step, "args": vars(args)}, f, ensure_ascii=False, indent=2)
        os.replace(tmp_state, state_path)

        prune_checkpoints(os.fspath(ckpt_dir), keep_last=int(args.keep_last_checkpoints))
        return os.fspath(path)

    loss_wrapped = lambda x, y, m, adv: rl_loss_fn(model, x, y, m, adv)
    value_and_grad = nn.value_and_grad(model, loss_wrapped)

    t0 = time.time()
    global_step = start_step

    def iter_tokenized(seed: int) -> Iterator[TokenizedRollout]:
        stream = buffer.iter(follow=bool(args.follow_buffer))
        stream2 = shuffle_stream(stream, buffer_size=int(args.shuffle_buffer), seed=seed)
        for obj in stream2:
            if args.min_reward is not None and float(obj.get("reward") or 0.0) < float(args.min_reward):
                continue
            msgs = obj.get("messages")
            resp = obj.get("response")
            if not isinstance(msgs, list) or not isinstance(resp, str):
                continue
            tokenized = tokenize_rollout(
                tokenizer=tokenizer,
                messages=msgs,
                response=resp,
                reward=float(obj.get("reward") or 0.0),
                seq_len=int(args.seq_len),
                pad_id=pad_id,
            )
            if tokenized is None:
                continue
            yield tokenized

    token_iter = iter_tokenized(seed=int(args.seed))

    while global_step < int(args.max_steps):
        batch: List[TokenizedRollout] = []
        while len(batch) < int(args.batch_size):
            try:
                batch.append(next(token_iter))
            except StopIteration:
                if bool(args.follow_buffer):
                    time.sleep(0.25)
                    continue
                token_iter = iter_tokenized(seed=int(args.seed) + global_step + 1)
                continue

        rewards = mx.array([b.reward for b in batch], dtype=mx.float32)
        adv = rewards - mx.mean(rewards)
        if bool(args.adv_norm):
            std = mx.sqrt(mx.mean(mx.square(adv)))
            adv = adv / mx.maximum(std, mx.array(1e-6, dtype=mx.float32))

        x = mx.array([b.x for b in batch], dtype=mx.int32)
        y = mx.array([b.y for b in batch], dtype=mx.int32)
        m = mx.array([b.mask for b in batch], dtype=mx.float32)

        lr = cosine_lr(global_step, int(args.max_steps), float(args.learning_rate), warmup_steps=int(args.warmup_steps))
        optimizer.learning_rate = lr

        loss, grads = value_and_grad(x, y, m, adv)
        mx.eval(loss, grads)

        if float(args.grad_clip) > 0:
            grads, grad_norm = optim.clip_grad_norm(grads, max_norm=float(args.grad_clip))
        else:
            grad_norm = mx.array(0.0, dtype=mx.float32)

        optimizer.update(model, grads)
        mx.eval(grad_norm)

        global_step += 1

        if int(args.log_interval) > 0 and global_step % int(args.log_interval) == 0:
            dt = time.time() - t0
            mean_r = float(mx.mean(rewards).item())
            print(
                f"[train] step={global_step} loss={float(loss.item()):.4f} "
                f"mean_reward={mean_r:.3f} lr={lr:.2e} grad_norm={float(grad_norm.item()):.3f} "
                f"elapsed_s={dt:.1f}",
                flush=True,
            )

        if int(args.save_interval) > 0 and global_step % int(args.save_interval) == 0:
            path = save_checkpoint(global_step)
            print(f"[ckpt] saved {path}", flush=True)

    path = save_checkpoint(global_step)
    print(f"[done] saved {path}", flush=True)


if __name__ == "__main__":
    main()
