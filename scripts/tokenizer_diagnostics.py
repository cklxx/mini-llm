#!/usr/bin/env python3
"""分词器健康度诊断工具。

功能：
- 校验 tokenizer 配置与元数据
- 统计随机样本的 UNK 占比
- 输出基础统计，帮助排查预训练/推理分词器不匹配问题
"""
from __future__ import annotations

import argparse
import json
import os
import pprint
import random
import sys
from collections.abc import Iterable, Iterator
from pathlib import Path

# 添加项目根目录和src目录到路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.append(PROJECT_ROOT)
sys.path.append(os.path.join(PROJECT_ROOT, "src"))

from tokenizer.bpe_tokenizer import BPETokenizer  # noqa: E402
from tokenizer.config_utils import canonicalize_tokenizer_config  # noqa: E402

try:
    import torch
except ImportError:  # pragma: no cover - fallback for minimal environments
    torch = None


def iter_texts_from_jsonl(path: str) -> Iterator[str]:
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "text" in record and record["text"]:
                yield str(record["text"])
            elif "conversations" in record:
                for turn in record["conversations"]:
                    content = turn.get("content")
                    if content:
                        yield str(content)
            elif "input" in record and "output" in record:
                yield f"{record['input']} {record['output']}"
            elif "chosen" in record:
                yield str(record["chosen"])


def iter_texts_from_plain(path: str) -> Iterator[str]:
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield line


def collect_texts(path: str, sample_size: int, seed: int) -> list[str]:
    if path.endswith(".jsonl"):
        iterator: Iterable[str] = iter_texts_from_jsonl(path)
    else:
        iterator = iter_texts_from_plain(path)

    if sample_size <= 0:
        return list(iterator)

    reservoir: list[str] = []
    rng = random.Random(seed)
    for idx, text in enumerate(iterator):
        if not text:
            continue
        if len(reservoir) < sample_size:
            reservoir.append(text)
        else:
            j = rng.randint(0, idx)
            if j < sample_size:
                reservoir[j] = text
    rng.shuffle(reservoir)
    return reservoir


def _resolve_tokenizer_location(path: str) -> str:
    candidate = Path(path).expanduser()
    if candidate.is_dir():
        json_path = candidate / "tokenizer.json"
        if not json_path.exists():
            raise FileNotFoundError(f"目录中缺少 tokenizer.json: {candidate}")
        return str(candidate)
    if candidate.exists():
        return str(candidate)
    raise FileNotFoundError(f"未找到分词器文件: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="MiniGPT 分词器诊断工具")
    parser.add_argument(
        "--tokenizer",
        required=True,
        help="tokenizer 路径（支持 tokenizer.json、tokenizer.pkl 或包含 tokenizer.json 的目录）",
    )
    parser.add_argument("--data", help="用于统计UNK率的数据文件 (jsonl 或 txt)")
    parser.add_argument("--sample-size", type=int, default=1000, help="随机抽样样本数")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--checkpoint", help="用于对比元数据的checkpoint路径")

    args = parser.parse_args()

    tokenizer_path = _resolve_tokenizer_location(args.tokenizer)
    tokenizer = BPETokenizer()
    tokenizer.load(tokenizer_path)

    print("=== Tokenizer 信息 ===")
    print(f"文件: {tokenizer_path}")
    print(f"词汇表大小: {tokenizer.vocab_size}")
    print(f"配置: {tokenizer.get_config()}")
    print(f"Checksum: {tokenizer.checksum() or 'N/A'}")
    print("特殊Token映射:")
    pprint.pprint(tokenizer.special_tokens_map(require_presence=bool(tokenizer.vocab)))

    if args.checkpoint:
        if torch is None:
            print("⚠️  未安装torch，无法加载checkpoint进行对比")
        elif not os.path.exists(args.checkpoint):
            raise FileNotFoundError(f"checkpoint不存在: {args.checkpoint}")
        else:
            checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
            raw_expected_config = checkpoint.get("tokenizer_config")
            expected_config = canonicalize_tokenizer_config(raw_expected_config)
            if expected_config:
                actual_config = canonicalize_tokenizer_config(tokenizer.get_config())
                mismatches = {
                    key: (expected_config[key], actual_config.get(key))
                    for key in expected_config
                    if actual_config.get(key) != expected_config[key]
                }
                if mismatches:
                    print("⚠️  配置不一致:")
                    for key, values in mismatches.items():
                        print(f"   {key}: checkpoint={values[0]} tokenizer={values[1]}")
                else:
                    print("✅ checkpoint 配置匹配")
            expected_special = checkpoint.get("tokenizer_special_tokens")
            if expected_special:
                special_mismatches = tokenizer.diff_special_tokens(expected_special)
                if special_mismatches:
                    print("⚠️  特殊token映射不一致:")
                    for name, (exp, act) in special_mismatches.items():
                        print(f"   {name}: checkpoint={exp} tokenizer={act}")
                else:
                    print("✅ 特殊token映射匹配")

    if not args.data:
        print("未提供数据文件，跳过 UNK 率统计。")
        return

    if not os.path.exists(args.data):
        raise FileNotFoundError(f"数据文件不存在: {args.data}")

    texts = collect_texts(args.data, args.sample_size, args.seed)
    if not texts:
        print("⚠️  样本为空，无法统计 UNK 率")
        return

    stats = tokenizer.compute_unk_statistics(texts, sample_size=len(texts))
    print("\n=== UNK 统计 ===")
    print(f"样本条数: {stats['sampled_texts']}")
    print(f"总token数: {stats['total_tokens']}")
    print(f"UNK数量: {stats['unk_tokens']}")
    print(f"UNK率: {stats['unk_rate']*100:.4f}%")

    if stats["unk_rate"] > 0.001:
        print(
            "⚠️  UNK率偏高，建议检查 tokenizer 配置、训练/推理规范化流程或是否启用 byte fallback。"
        )
    else:
        print("✅ UNK率处于正常范围")


if __name__ == "__main__":
    main()
