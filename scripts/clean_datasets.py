"""
MiniGPT dataset cleaner.

This script processes JSONL datasets to remove placeholders, optional follow-up
requests, and repeated entries using signature hashing. It also strips
<think>...</think> blocks when requested and provides summary statistics.

Features
--------
- Works with JSONL datasets used for pretrain ("text" field) and SFT ("conversations" list).
- Deduplicates entries using normalized hashes.
- Drops samples containing placeholder / incomplete responses ("...", "省略", etc.).
- Optionally strips internal thinking tags (e.g. ``<think>``) from assistant replies.
- Filters assistant refusals above a configurable threshold.
- Provides summary statistics for auditing.

Example
-------
    python scripts/clean_datasets.py \
        --input data/sft_mini_512.jsonl \
        --output data/clean/sft_mini_512.cleaned.jsonl \
        --dataset-type sft --drop-placeholders --dedupe --strip-think

    python scripts/clean_datasets.py \
        --input data/pretrain_hq.jsonl \
        --output data/clean/pretrain_hq.cleaned.jsonl \
        --dataset-type pretrain --dedupe --max-refusal-count 2
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PLACEHOLDER_PATTERNS = [
    "……",  # Chinese ellipsis
    "...",
    "省略",
    "（省略）",
    "待补充",
    "TODO",
    "xxx",
    "XX",
    "文章过程省略",
]

FOLLOWUP_PATTERNS = [
    "请提供更多信息",
    "需要更多信息",
    "请提供具体",
    "请告知更多",
]

REFUSAL_PATTERNS = [
    "很抱歉",
    "抱歉",
    "无法",
    "不能",
    "不支持",
    "不方便",
    "作为一个AI",
]

THINK_TAG_PATTERN = re.compile(r"<think>.*?</think>", re.DOTALL)
ANSWER_WRAPPER_PATTERN = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)


@dataclass
class CleanConfig:
    dataset_type: str
    dedupe: bool = False
    drop_placeholders: bool = False
    drop_followups: bool = False
    strip_think: bool = False
    lowercase_signature: bool = False
    max_refusal_count: int | None = None


@dataclass
class Stats:
    total: int = 0
    written: int = 0
    duplicates: int = 0
    placeholder_drops: int = 0
    followup_drops: int = 0
    refusal_drops: int = 0
    think_stripped: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "total": self.total,
            "written": self.written,
            "duplicates": self.duplicates,
            "placeholder_drops": self.placeholder_drops,
            "followup_drops": self.followup_drops,
            "refusal_drops": self.refusal_drops,
            "think_stripped": self.think_stripped,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean MiniGPT JSONL datasets")
    parser.add_argument("--input", required=True, type=Path, help="Path to input JSONL file")
    parser.add_argument("--output", required=True, type=Path, help="Path to output JSONL file")
    parser.add_argument(
        "--dataset-type",
        choices=["sft", "pretrain"],
        required=True,
        help="Dataset flavour to apply corresponding cleaning rules",
    )
    parser.add_argument("--dedupe", action="store_true", help="Drop duplicate entries")
    parser.add_argument(
        "--drop-placeholders",
        action="store_true",
        help="Remove samples containing placeholder markers (省略, ... 等)",
    )
    parser.add_argument(
        "--drop-followups",
        action="store_true",
        help="Remove SFT samples where assistant回答仅索要更多信息",
    )
    parser.add_argument(
        "--strip-think",
        action="store_true",
        help="Strip <think>…</think> blocks and unwrap <answer> contents",
    )
    parser.add_argument(
        "--max-refusal-count",
        type=int,
        default=None,
        help=(
            "Maximum allowed occurrences of refusal phrases in an entry; entries "
            "exceeding the count are dropped"
        ),
    )
    parser.add_argument(
        "--lowercase-signature",
        action="store_true",
        help="Lowercase signature text before hashing (useful for English corpora)",
    )
    return parser.parse_args()


def normalize_signature(raw: str, lowercase: bool = False) -> bytes:
    text = raw.strip()
    if lowercase:
        text = text.lower()
    digest = hashlib.md5(text.encode("utf-8")).digest()
    return digest


def contains_placeholder(text: str) -> bool:
    return any(pat in text for pat in PLACEHOLDER_PATTERNS)


def is_followup_request(text: str) -> bool:
    trimmed = text.strip()
    window = trimmed if len(trimmed) <= 64 else trimmed[:64]
    return any(pat in window for pat in FOLLOWUP_PATTERNS)


def count_refusals(text: str) -> int:
    return sum(text.count(pat) for pat in REFUSAL_PATTERNS)


def strip_think_blocks(text: str, stats: Stats) -> str:
    if "<think>" not in text:
        return text
    new_text, replacements = THINK_TAG_PATTERN.subn("", text)
    if replacements:
        stats.think_stripped += replacements
    match = ANSWER_WRAPPER_PATTERN.search(new_text)
    if match:
        new_text = match.group(1).strip()
    return new_text.strip()


def load_json(line: str) -> dict[str, Any]:
    return json.loads(line)


def dump_json(obj: dict[str, Any]) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def clean_sft_entry(
    entry: dict[str, Any],
    cfg: CleanConfig,
    stats: Stats,
) -> tuple[dict[str, Any] | None, bytes | None]:
    convs = entry.get("conversations")
    if not isinstance(convs, list):
        return None, None
    cleaned_convs: list[dict[str, str]] = []
    for turn in convs:
        role = turn.get("role")
        content = turn.get("content", "")
        if not isinstance(content, str):
            return None, None
        if role == "assistant":
            if cfg.drop_placeholders and contains_placeholder(content):
                stats.placeholder_drops += 1
                return None, None
            if cfg.drop_followups and is_followup_request(content):
                stats.followup_drops += 1
                return None, None
            if cfg.strip_think:
                content = strip_think_blocks(content, stats)
        else:
            if cfg.strip_think:
                content = strip_think_blocks(content, stats)
        cleaned_convs.append({"role": role, "content": content})
    entry = {"conversations": cleaned_convs}

    signature: bytes | None = None
    if cfg.dedupe:
        signature = normalize_signature(
            dump_json(entry), lowercase=cfg.lowercase_signature
        )
    return entry, signature


def clean_pretrain_entry(
    entry: dict[str, Any],
    cfg: CleanConfig,
    stats: Stats,
) -> tuple[dict[str, Any] | None, bytes | None]:
    text = entry.get("text")
    if not isinstance(text, str):
        return None, None
    if cfg.drop_placeholders and contains_placeholder(text):
        stats.placeholder_drops += 1
        return None, None
    if cfg.max_refusal_count is not None:
        refusal_count = count_refusals(text)
        if refusal_count > cfg.max_refusal_count:
            stats.refusal_drops += 1
            return None, None
    if cfg.strip_think:
        text = strip_think_blocks(text, stats)
    entry = {"text": text}
    signature: bytes | None = None
    if cfg.dedupe:
        signature = normalize_signature(
            dump_json(entry), lowercase=cfg.lowercase_signature
        )
    return entry, signature


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def clean_file(args: argparse.Namespace) -> Stats:
    cfg = CleanConfig(
        dataset_type=args.dataset_type,
        dedupe=args.dedupe,
        drop_placeholders=args.drop_placeholders,
        drop_followups=args.drop_followups,
        strip_think=args.strip_think,
        lowercase_signature=args.lowercase_signature,
        max_refusal_count=args.max_refusal_count,
    )

    stats = Stats()
    seen_hashes: set[bytes] = set()
    ensure_parent_dir(args.output)

    cleaner = clean_sft_entry if cfg.dataset_type == "sft" else clean_pretrain_entry

    with args.input.open(encoding="utf-8") as src, args.output.open(
        "w", encoding="utf-8"
    ) as dst:
        for line in src:
            line = line.strip()
            if not line:
                continue
            stats.total += 1
            try:
                entry = load_json(line)
            except json.JSONDecodeError:
                stats.placeholder_drops += 1
                continue
            cleaned, signature = cleaner(entry, cfg, stats)
            if cleaned is None:
                continue
            if signature is not None:
                if signature in seen_hashes:
                    stats.duplicates += 1
                    continue
                seen_hashes.add(signature)
            dst.write(dump_json(cleaned) + "\n")
            stats.written += 1
    return stats


def main() -> None:
    args = parse_args()
    stats = clean_file(args)
    summary = stats.as_dict()
    summary_json = json.dumps(summary, ensure_ascii=False, indent=2)
    print(summary_json)


if __name__ == "__main__":
    main()
