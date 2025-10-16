#!/usr/bin/env python3
"""Train a SentencePiece tokenizer on one or more JSONL corpora.

Requirements:
    pip install sentencepiece

Example:
    python scripts/train_sentencepiece.py \
        --inputs data/processed/wiki_zh_full.cleaned.jsonl \
                  data/processed/chinacorpus_full.cleaned.jsonl \
                  data/processed/pretrain_hq.cleaned.jsonl \
        --text-field text \
        --model-prefix artifacts/minigpt_spm \
        --vocab-size 30000 \
        --character-coverage 0.9995 \
        --input-sentence-size 10000000
"""

from __future__ import annotations

import argparse
import json
import tempfile
from collections.abc import Iterable
from pathlib import Path

try:
    import sentencepiece as spm
except ImportError as exc:  # pragma: no cover
    raise SystemExit("sentencepiece not installed. Run `pip install sentencepiece`." ) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SentencePiece on JSONL corpora")
    parser.add_argument("--inputs", nargs="+", type=Path, required=True, help="JSONL files containing text field")
    parser.add_argument("--text-field", default="text", help="Field name to extract text from")
    parser.add_argument("--model-prefix", required=True, help="Output prefix for SentencePiece model")
    parser.add_argument("--vocab-size", type=int, default=32000)
    parser.add_argument("--character-coverage", type=float, default=0.9995)
    parser.add_argument("--input-sentence-size", type=int, default=10000000, help="Subsample size for SentencePiece trainer")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle sentences before training")
    return parser.parse_args()


def iter_text(inputs: Iterable[Path], field: str) -> Iterable[str]:
    for path in inputs:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                text = obj.get(field, "")
                if isinstance(text, str) and text.strip():
                    yield text.strip()


def main() -> None:
    args = parse_args()
    args.model_prefix = str(Path(args.model_prefix))

    with tempfile.NamedTemporaryFile(mode="w", delete=False, encoding="utf-8") as tmp:
        for text in iter_text(args.inputs, args.text_field):
            tmp.write(text.replace("\n", " ") + "\n")
        temp_path = tmp.name

    spm.SentencePieceTrainer.train(
        input=temp_path,
        model_prefix=args.model_prefix,
        vocab_size=args.vocab_size,
        character_coverage=args.character_coverage,
        shuffle_input_sentence=args.shuffle,
        input_sentence_size=args.input_sentence_size,
    )

    Path(temp_path).unlink(missing_ok=True)


if __name__ == "__main__":
    main()
