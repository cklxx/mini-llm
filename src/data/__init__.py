"""Data package helpers."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Optional

from src.tokenizer.bpe_tokenizer import BPETokenizer


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _default_tokenizer_candidates() -> list[Path]:
    root = _project_root()
    return [
        root / "tokenizers" / "minimind",
        root / "tokenizers" / "minimind" / "tokenizer.json",
    ]


def _resolve_tokenizer_path(path: Optional[str]) -> Path:
    if path:
        candidate = Path(path)
        if candidate.exists():
            return candidate
    for candidate in _default_tokenizer_candidates():
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "未找到默认的分词器资源，请确保 tokenizers/minimind/tokenizer.json 存在。"
    )


@lru_cache(maxsize=4)
def load_tokenizer(tokenizer_path: Optional[str] = None) -> BPETokenizer:
    """Load the shared tokenizer used across training and preprocessing."""

    resolved = _resolve_tokenizer_path(tokenizer_path)
    tokenizer = BPETokenizer()
    tokenizer.load(str(resolved))
    return tokenizer


__all__ = ["load_tokenizer"]
