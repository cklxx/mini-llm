from __future__ import annotations

import argparse
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

import mlx.core as mx
from transformers import AutoTokenizer

from ..infer import generate, load_config
from ..model import MiniLLMForCausalLM
from .buffer import JsonlRolloutBuffer, RolloutRecord, sha1, utc_now_iso
from .dataset import load_gsm8k_list
from .reward import extract_final_answer, reward_gsm8k


DEFAULT_SYSTEM_PROMPT = (
    "你是一位严谨的数学助理。请逐步推理，并在最后一行输出最终答案，格式为：#### <整数答案>。"
)


def load_mlx_model(checkpoint_dir: Path, *, dtype: Optional[str] = None) -> MiniLLMForCausalLM:
    cfg = load_config(checkpoint_dir)
    model = MiniLLMForCausalLM(cfg)
    model.load_weights(os.fspath(checkpoint_dir / "model.safetensors"))
    if dtype:
        dtype_map = {"float16": mx.float16, "bfloat16": mx.bfloat16, "float32": mx.float32}
        if dtype not in dtype_map:
            raise ValueError(f"Unknown dtype: {dtype} (expected one of {sorted(dtype_map)})")
        model.apply(lambda p: p.astype(dtype_map[dtype]))
    model.eval()
    return model


def build_messages(question: str, *, system_prompt: str) -> List[Dict[str, str]]:
    return [{"role": "system", "content": system_prompt}, {"role": "user", "content": question}]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate GSM8K rollouts with the MLX infer backend.")
    parser.add_argument("--dataset_dir", type=str, default="/root/gsm8k")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--limit", type=int, default=None, help="Only load the first N dataset samples.")

    parser.add_argument("--tokenizer_path", type=str, default="./model")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Checkpoint directory produced by mlx_train/train.py (contains model.safetensors + config.json).",
    )
    parser.add_argument("--dtype", type=str, default=None, help="Optional cast: float16/bfloat16/float32")

    parser.add_argument("--out_buffer", type=str, required=True, help="Output JSONL buffer path (appends).")
    parser.add_argument("--num_rollouts", type=int, default=256, help="Number of prompts to sample (0 = all).")
    parser.add_argument(
        "--samples_per_prompt",
        type=int,
        default=1,
        help="How many completions to sample per prompt (use >1 to increase chance of positive rewards).",
    )
    parser.add_argument(
        "--min_positive",
        type=int,
        default=0,
        help="Keep sampling prompts (with replacement) until collecting at least N reward>0 rollouts (bounded by --max_total_rollouts).",
    )
    parser.add_argument(
        "--max_total_rollouts",
        type=int,
        default=0,
        help="Hard cap on total rollout records to write (0 = derived from num_rollouts*samples_per_prompt).",
    )
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--system_prompt", type=str, default=DEFAULT_SYSTEM_PROMPT)

    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--min_new_tokens", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--max_seq_len", type=int, default=None)
    parser.add_argument("--log_every", type=int, default=25)
    args = parser.parse_args()

    mx.random.seed(int(args.seed))
    rng = random.Random(int(args.seed))

    ckpt = Path(args.checkpoint)
    if not ckpt.exists() or not ckpt.is_dir():
        raise FileNotFoundError(f"Checkpoint must be a directory: {ckpt}")

    examples = load_gsm8k_list(str(args.dataset_dir), split=str(args.split), limit=args.limit)
    if not examples:
        raise RuntimeError("No GSM8K samples loaded (check --dataset_dir / --split).")
    rng.shuffle(examples)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = load_mlx_model(ckpt, dtype=args.dtype)

    buffer = JsonlRolloutBuffer(args.out_buffer)

    num_prompts = len(examples) if int(args.num_rollouts) <= 0 else min(len(examples), int(args.num_rollouts))
    samples_per_prompt = int(args.samples_per_prompt)
    if samples_per_prompt <= 0:
        raise ValueError("--samples_per_prompt must be > 0")

    min_positive = int(args.min_positive)
    if min_positive < 0:
        raise ValueError("--min_positive must be >= 0")

    max_total_rollouts = int(args.max_total_rollouts)
    if max_total_rollouts < 0:
        raise ValueError("--max_total_rollouts must be >= 0")
    if max_total_rollouts == 0:
        max_total_rollouts = int(num_prompts) * int(samples_per_prompt)

    ok = 0
    written = 0
    prompts_used = 0
    ex_idx = 0

    while written < max_total_rollouts and (prompts_used < num_prompts or ok < min_positive):
        if ex_idx >= len(examples):
            rng.shuffle(examples)
            ex_idx = 0
        ex = examples[ex_idx]
        ex_idx += 1
        prompts_used += 1

        messages = build_messages(ex.question, system_prompt=str(args.system_prompt))
        prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompt_ids: List[int] = tokenizer.encode(prompt_text, add_special_tokens=False)
        prompt_hash = sha1(prompt_text)

        for j in range(samples_per_prompt):
            if written >= max_total_rollouts:
                break

            # Make sampling reproducible and diverse across (prompt, sample).
            req_seed = int(args.seed) + prompts_used * 1000003 + j * 10007
            mx.random.seed(req_seed)

            out_ids = generate(
                model,
                input_ids=prompt_ids,
                eos_token_id=tokenizer.eos_token_id,
                max_new_tokens=int(args.max_new_tokens),
                min_new_tokens=int(args.min_new_tokens),
                banned_token_ids=list(tokenizer.all_special_ids),
                temperature=float(args.temperature),
                top_p=float(args.top_p),
                max_seq_len=args.max_seq_len,
            )
            resp_ids = out_ids[len(prompt_ids) :]
            response_text = tokenizer.decode(resp_ids, skip_special_tokens=True).strip()
            ref_final = extract_final_answer(ex.answer)
            pred_final = extract_final_answer(response_text)
            r = reward_gsm8k(response_text, ex.answer)

            record = RolloutRecord(
                example_id=ex.example_id,
                messages=messages,
                response=response_text,
                reward=float(r),
                reference_answer=ex.answer,
                pred_final=pred_final,
                ref_final=ref_final,
                meta={
                    **ex.meta,
                    "created_at": utc_now_iso(),
                    "checkpoint": os.fspath(ckpt),
                    "prompt_hash": prompt_hash,
                    "sample_idx": int(j),
                    "samples_per_prompt": int(samples_per_prompt),
                    "temperature": float(args.temperature),
                    "top_p": float(args.top_p),
                    "max_new_tokens": int(args.max_new_tokens),
                    "split": str(args.split),
                },
            )
            buffer.append(record)
            written += 1

            ok += int(r > 0.0)

            if int(args.log_every) > 0 and written % int(args.log_every) == 0:
                acc = ok / max(written, 1)
                print(
                    f"[rollout] wrote={written}/{max_total_rollouts} prompts={prompts_used} reward>0={ok} acc={acc:.3f}",
                    flush=True,
                )

    acc = ok / max(written, 1)
    print(
        f"[done] wrote={written} prompts={prompts_used} buffer={args.out_buffer} reward>0={ok} acc={acc:.3f}",
        flush=True,
    )
    if min_positive > 0 and ok < min_positive:
        print(
            f"[warn] min_positive not reached: want={min_positive} got={ok} "
            f"(cap max_total_rollouts={max_total_rollouts})",
            flush=True,
        )


if __name__ == "__main__":
    main()
