from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Any, Dict, Optional

from .dataset import load_gsm8k_list
from .reward import extract_final_answer


DEFAULT_SYSTEM_PROMPT = (
    "你是一位严谨的数学助理。请逐步推理，并在最后一行输出最终答案，格式为：#### <整数答案>。"
)


def _build_record(
    *,
    example_id: str,
    question: str,
    answer: str,
    meta: Dict[str, Any],
    system_prompt: str,
    answer_mode: str,
) -> Dict[str, Any]:
    assistant = answer.strip()
    if answer_mode == "final":
        final = extract_final_answer(answer)
        if final:
            assistant = f"#### {final}"

    return {
        "conversations": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question.strip()},
            {"role": "assistant", "content": assistant},
        ],
        "meta": {
            "example_id": example_id,
            **(meta or {}),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a GSM8K SFT JSONL (chat format) for mlx_train/train.py.")
    parser.add_argument("--dataset_dir", type=str, default="/root/gsm8k")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--out_path", type=str, required=True, help="Output JSONL path.")
    parser.add_argument("--limit", type=int, default=None, help="Write at most N samples (default: all).")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--shuffle", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--system_prompt", type=str, default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument(
        "--answer_mode",
        type=str,
        choices=["full", "final"],
        default="full",
        help="`full`: use original GSM8K rationale+final. `final`: keep only `#### <ans>` to reduce length.",
    )
    args = parser.parse_args()

    examples = load_gsm8k_list(str(args.dataset_dir), split=str(args.split), limit=None)
    if not examples:
        raise RuntimeError("No GSM8K samples loaded (check --dataset_dir / --split).")

    if bool(args.shuffle):
        random.Random(int(args.seed)).shuffle(examples)

    limit: Optional[int] = None if args.limit is None else int(args.limit)
    if limit is not None and limit > 0:
        examples = examples[:limit]

    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for ex in examples:
            rec = _build_record(
                example_id=ex.example_id,
                question=ex.question,
                answer=ex.answer,
                meta=ex.meta,
                system_prompt=str(args.system_prompt),
                answer_mode=str(args.answer_mode),
            )
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            written += 1

    print(f"[done] wrote={written} out={os.fspath(out_path)}", flush=True)


if __name__ == "__main__":
    main()

