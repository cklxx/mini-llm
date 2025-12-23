from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import mlx.core as mx
from transformers import AutoTokenizer

from .config import MiniLLMConfig
from .model import MiniLLMForCausalLM, LayerKVCache


def load_config(checkpoint_dir: Path) -> MiniLLMConfig:
    config_path = checkpoint_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config.json in checkpoint dir: {checkpoint_dir}")
    with open(config_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return MiniLLMConfig(**data).finalize()


def sample_next_token(
    logits: mx.array,
    *,
    temperature: float,
    top_p: float,
) -> int:
    if temperature <= 0:
        return int(mx.argmax(logits).item())

    logits = logits / temperature

    if top_p >= 1.0:
        token = mx.random.categorical(logits, axis=-1)
        return int(token.item())

    # Nucleus sampling (top_p) in sorted space, then map back.
    sorted_idx = mx.argsort(-logits, axis=-1)
    sorted_logits = mx.take_along_axis(logits, sorted_idx, axis=-1)
    sorted_probs = mx.softmax(sorted_logits, axis=-1)
    cumprobs = mx.cumsum(sorted_probs, axis=-1)

    # Remove tokens once cumprob passes top_p, but keep the first token that crosses the threshold.
    remove = cumprobs > top_p
    remove = mx.concatenate([mx.array([False]), remove[:-1]], axis=-1)
    neg_inf = mx.array(-1e9, dtype=sorted_logits.dtype)
    filtered_logits = mx.where(remove, neg_inf, sorted_logits)
    picked = mx.random.categorical(filtered_logits, axis=-1)
    picked_i = int(picked.item())
    return int(sorted_idx[picked_i].item())


def generate(
    model: MiniLLMForCausalLM,
    *,
    input_ids: List[int],
    eos_token_id: Optional[int],
    max_new_tokens: int,
    min_new_tokens: int,
    banned_token_ids: Optional[List[int]] = None,
    temperature: float,
    top_p: float,
    max_seq_len: Optional[int],
) -> List[int]:
    ids = list(input_ids)
    if max_new_tokens <= 0:
        return ids
    if max_seq_len is not None and len(ids) >= int(max_seq_len):
        return ids
    if not ids:
        raise ValueError("input_ids is empty; need at least 1 token for generation")

    cache_len = int(max_seq_len) if max_seq_len is not None else (len(ids) + int(max_new_tokens) + 1)
    cache: List[LayerKVCache] = model.allocate_kv_cache(batch_size=1, max_seq_len=cache_len)

    # Prefill: run the whole prompt once and populate KV cache.
    x0 = mx.array([ids], dtype=mx.int32)
    logits0, cache = model.forward_with_cache(x0, start_pos=0, cache=cache)
    logits = logits0[0, -1, :]

    produced = 0
    for _ in range(int(max_new_tokens)):
        if max_seq_len is not None and len(ids) >= int(max_seq_len):
            break

        if produced < int(min_new_tokens) and banned_token_ids:
            vocab = int(logits.shape[0])
            idx = mx.arange(vocab)
            mask = mx.zeros((vocab,), dtype=mx.bool_)
            for token_id in set(banned_token_ids):
                if 0 <= int(token_id) < vocab:
                    mask = mx.logical_or(mask, idx == int(token_id))
            logits = mx.where(mask, mx.array(-1e9, dtype=logits.dtype), logits)

        next_token = sample_next_token(logits, temperature=temperature, top_p=top_p)
        ids.append(int(next_token))
        produced += 1
        if eos_token_id is not None and produced >= int(min_new_tokens) and int(next_token) == int(eos_token_id):
            break
        if max_seq_len is not None and len(ids) >= int(max_seq_len):
            break

        # Decode: feed only the last generated token; KV cache makes it O(1) per step.
        pos = len(ids) - 1  # absolute position of the token we just appended
        x = mx.array([[int(next_token)]], dtype=mx.int32)
        logits1, cache = model.forward_with_cache(x, start_pos=int(pos), cache=cache)
        logits = logits1[0, -1, :]

    return ids


def main() -> None:
    parser = argparse.ArgumentParser(description="MiniLLM (MLX) inference")
    parser.add_argument("--tokenizer_path", type=str, default="./model")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Checkpoint directory produced by mlx_train/train.py (contains model.safetensors + config.json).",
    )
    parser.add_argument("--prompt", type=str, default="hi")
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--min_new_tokens", type=int, default=0, help="Force at least N new tokens before EOS stop.")
    parser.add_argument("--temperature", type=float, default=0.0, help="0 for greedy; >0 for sampling.")
    parser.add_argument("--top_p", type=float, default=1.0, help="Nucleus sampling threshold (only if temperature>0).")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--max_seq_len", type=int, default=None)

    args = parser.parse_args()

    mx.random.seed(args.seed)

    ckpt = Path(args.checkpoint)
    if not ckpt.exists() or not ckpt.is_dir():
        raise FileNotFoundError(f"Checkpoint must be a directory: {ckpt}")

    cfg = load_config(ckpt)
    model = MiniLLMForCausalLM(cfg)
    model.load_weights(os.fspath(ckpt / "model.safetensors"))
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    messages = [{"role": "user", "content": args.prompt}]
    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    prompt_ids: List[int] = tokenizer.encode(prompt_text, add_special_tokens=False)

    output_ids = generate(
        model,
        input_ids=prompt_ids,
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=args.max_new_tokens,
        min_new_tokens=args.min_new_tokens,
        banned_token_ids=list(tokenizer.all_special_ids),
        temperature=args.temperature,
        top_p=args.top_p,
        max_seq_len=args.max_seq_len,
    )

    response_ids = output_ids[len(prompt_ids) :]
    response_text = tokenizer.decode(response_ids, skip_special_tokens=True)
    text = response_text.strip()
    if text:
        print(text)
    else:
        print("[infer] empty response (try --temperature 0.7 --top_p 1.0 or set --min_new_tokens)")


if __name__ == "__main__":
    main()
