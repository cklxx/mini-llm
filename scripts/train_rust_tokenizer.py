#!/usr/bin/env python3
"""Train or reuse the Rust-based tokenizer."""

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.append(str(SRC_ROOT))

from tokenizer.tokenizer_manager import TokenizerConfig, TokenizerManager  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the RustBPE tokenizer")
    parser.add_argument(
        "--data",
        type=str,
        default=str(PROJECT_ROOT / "data" / "pretrain_hq.jsonl"),
        help="JSONL dataset used for tokenizer training",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=32000,
        help="Target vocabulary size including special tokens",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(PROJECT_ROOT / "tokenizers" / "rust_bpe"),
        help="Directory where the tokenizer artifacts will be written",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=str(PROJECT_ROOT / "tokenizers"),
        help="Tokenizer cache directory (avoids retraining when unchanged)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force retraining even if a cached tokenizer exists",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=150000,
        help="Metadata hint used to form the cache key (for reproducibility)",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    data_path = Path(args.data)
    if not data_path.exists():
        parser.error(f"Dataset not found: {data_path}")

    os.makedirs(args.cache_dir, exist_ok=True)
    config = TokenizerConfig(
        vocab_size=args.vocab_size,
        tokenizer_type="rust-bpe",
        max_samples=args.max_samples,
    )
    manager = TokenizerManager(cache_dir=args.cache_dir)
    tokenizer = manager.get_or_train_tokenizer(
        str(data_path),
        config,
        force_retrain=args.force,
    )

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(output_dir))
    print(f"✅ RustBPE tokenizer ready at: {output_dir}")


if __name__ == "__main__":
    main()
