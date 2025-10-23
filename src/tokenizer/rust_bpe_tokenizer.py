"""Rust-accelerated Byte Pair Encoding tokenizer integration.

This module wraps the ``rustbpe`` PyO3 extension borrowed from the
`nanochat <https://github.com/cklxx/nanochat>`_ project and exposes an
API compatible with the existing :class:`BPETokenizer` implementation.
It combines the fast Rust trainer with ``tiktoken`` for inference-time
encoding/decoding so the rest of the codebase can reuse the same
interfaces without modification.
"""

from __future__ import annotations

import base64
import hashlib
import json
import pickle
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import tiktoken

try:  # rust extension is optional during static analysis
    import rustbpe  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - handled at runtime
    rustbpe = None  # type: ignore

SPECIAL_TOKENS = ["<PAD>", "<UNK>", "<BOS>", "<EOS>"]
TOKENIZER_PICKLE = "tokenizer.pkl"
TOKENIZER_METADATA = "tokenizer_meta.json"


class RustBPETokenizer:
    """Tokenizer that trains with Rust but behaves like :class:`BPETokenizer`."""

    def __init__(self, vocab_size: int = 30000) -> None:
        if vocab_size <= len(SPECIAL_TOKENS):
            raise ValueError("vocab_size must be larger than the special token count")

        self.vocab_size = vocab_size
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"

        self.pad_id = -1
        self.unk_id = -1
        self.bos_id = -1
        self.eos_id = -1

        self._encoding: tiktoken.Encoding | None = None
        self._pattern: str | None = None
        self._storage_dir: Path | None = None
        self._checksum: str | None = None

        self._special_token_to_id: dict[str, int] = {}
        self.vocab: dict[str, int] = {}

    # ------------------------------------------------------------------
    # helpers
    def _ensure_extension(self) -> None:
        if rustbpe is None:  # pragma: no cover - defensive guard
            raise RuntimeError(
                "rustbpe 扩展未安装。请运行 `uv run maturin develop --manifest-path "
                "rustbpe/Cargo.toml --release` 构建分词器扩展。"
            )

    def _ensure_encoding(self) -> tiktoken.Encoding:
        if self._encoding is None:
            raise RuntimeError("Tokenizer 未初始化。请先调用 train() 或 load().")
        return self._encoding

    def _compute_checksum(
        self,
        pattern: str,
        mergeable_ranks: Sequence[tuple[bytes, int]],
        special_tokens: dict[str, int],
    ) -> str:
        hasher = hashlib.sha256()
        hasher.update(pattern.encode("utf-8"))
        for token_bytes, rank in sorted(mergeable_ranks, key=lambda item: item[1]):
            hasher.update(token_bytes)
            hasher.update(rank.to_bytes(4, "little", signed=False))
        for name, idx in sorted(special_tokens.items()):
            hasher.update(name.encode("utf-8"))
            hasher.update(idx.to_bytes(4, "little", signed=False))
        return hasher.hexdigest()

    def _record_special_tokens(self, special_tokens: dict[str, int]) -> None:
        self._special_token_to_id = special_tokens
        self.pad_id = special_tokens[self.pad_token]
        self.unk_id = special_tokens[self.unk_token]
        self.bos_id = special_tokens[self.bos_token]
        self.eos_id = special_tokens[self.eos_token]

    def _rebuild_vocab(
        self, mergeable_ranks: Sequence[tuple[bytes, int]], special_tokens: dict[str, int]
    ) -> None:
        vocab: dict[str, int] = {}
        for token_bytes, token_id in mergeable_ranks:
            try:
                token_str = token_bytes.decode("utf-8")
            except UnicodeDecodeError:
                token_str = base64.b64encode(token_bytes).decode("ascii")
            vocab[token_str] = token_id
        vocab.update(special_tokens)
        self.vocab = vocab

    # ------------------------------------------------------------------
    # public API compatible with BPETokenizer
    def train(self, texts: Iterable[str]) -> None:
        self._ensure_extension()

        iterator = self._ensure_iterator(texts)
        tokenizer = rustbpe.Tokenizer()
        vocab_without_special = self.vocab_size - len(SPECIAL_TOKENS)
        if vocab_without_special < 256:
            raise ValueError(
                "vocab_size 太小。Rust BPE 要求词表大小减去特殊 token 至少为 256。"
            )

        tokenizer.train_from_iterator(iterator, vocab_without_special, pattern=None)

        pattern = tokenizer.get_pattern()
        mergeable_ranks_list = tokenizer.get_mergeable_ranks()
        mergeable_ranks = [(bytes(item[0]), item[1]) for item in mergeable_ranks_list]
        mergeable_rank_dict = {token_bytes: idx for token_bytes, idx in mergeable_ranks}

        token_offset = len(mergeable_ranks)
        special_tokens = {
            name: token_offset + index for index, name in enumerate(SPECIAL_TOKENS)
        }

        encoding = tiktoken.Encoding(
            name="rustbpe",
            pat_str=pattern,
            mergeable_ranks=mergeable_rank_dict,
            special_tokens=special_tokens,
        )

        self._encoding = encoding
        self._pattern = pattern
        self.vocab_size = encoding.n_vocab
        self._record_special_tokens(special_tokens)
        self._rebuild_vocab(mergeable_ranks, special_tokens)
        self._checksum = self._compute_checksum(
            pattern,
            mergeable_ranks,
            special_tokens,
        )
        self._storage_dir = None

    def _ensure_iterator(self, texts: Iterable[str]) -> Iterator[str]:
        if isinstance(texts, Iterator):
            return texts
        return iter(texts)

    def encode(
        self,
        text: str,
        *,
        add_special_tokens: bool = True,
        return_tensors: str | None = None,
    ) -> list[int] | "torch.Tensor":
        encoding = self._ensure_encoding()
        token_ids = encoding.encode(text, allowed_special=set(self._special_token_to_id.keys()))
        if add_special_tokens:
            token_ids = [self.bos_id, *token_ids, self.eos_id]
        if return_tensors is None:
            return token_ids
        if return_tensors != "pt":
            raise ValueError("仅支持 return_tensors='pt'")
        import torch

        return torch.tensor([token_ids], dtype=torch.long)

    def decode(self, token_ids: Sequence[int]) -> str:
        encoding = self._ensure_encoding()
        filtered = [
            token_id
            for token_id in token_ids
            if token_id not in {self.pad_id, self.bos_id, self.eos_id}
        ]
        return encoding.decode(filtered)

    def save(self, path: str) -> None:
        encoding = self._ensure_encoding()
        dest = Path(path)
        if dest.suffix:
            raise ValueError("请提供目录路径用于保存 Rust BPE 分词器")
        dest.mkdir(parents=True, exist_ok=True)

        pickle_path = dest / TOKENIZER_PICKLE
        with open(pickle_path, "wb") as handle:
            pickle.dump(encoding, handle)

        mergeable_ranks = list(encoding._mergeable_ranks.items())  # type: ignore[attr-defined]
        pattern = getattr(encoding, "_pat_str", self._pattern) or ""
        special_tokens = dict(encoding._special_tokens)  # type: ignore[attr-defined]
        checksum = self._compute_checksum(mergeable_ranks=mergeable_ranks, pattern=pattern, special_tokens=special_tokens)

        metadata = {
            "tokenizer_type": "rust-bpe",
            "vocab_size": encoding.n_vocab,
            "pattern": pattern,
            "special_tokens": special_tokens,
            "checksum": checksum,
        }
        with open(dest / TOKENIZER_METADATA, "w", encoding="utf-8") as handle:
            json.dump(metadata, handle, ensure_ascii=False, indent=2)

        self._storage_dir = dest
        self._checksum = checksum
        self._pattern = pattern
        self._record_special_tokens(special_tokens)
        self._rebuild_vocab(mergeable_ranks, special_tokens)
        self.vocab_size = encoding.n_vocab

    def load(self, path: str) -> None:
        dest = Path(path)
        pickle_path = dest
        metadata_path = dest

        if dest.is_dir():
            pickle_path = dest / TOKENIZER_PICKLE
            metadata_path = dest / TOKENIZER_METADATA
        elif dest.suffix == ".pkl":
            metadata_path = dest.with_name(TOKENIZER_METADATA)
        elif dest.suffix == ".json":
            raise RuntimeError("请提供保存的 Rust BPE 目录或 tokenizer.pkl 文件")

        if not pickle_path.exists():
            raise FileNotFoundError(f"未找到 Rust BPE 分词器文件: {pickle_path}")

        with open(pickle_path, "rb") as handle:
            encoding: tiktoken.Encoding = pickle.load(handle)

        pattern = getattr(encoding, "_pat_str", None)
        special_tokens = dict(encoding._special_tokens)  # type: ignore[attr-defined]

        metadata: dict[str, Any] = {}
        if metadata_path.exists():
            with open(metadata_path, encoding="utf-8") as handle:
                metadata = json.load(handle)
            pattern = metadata.get("pattern", pattern)
            special_tokens = metadata.get("special_tokens", special_tokens)
            self._checksum = metadata.get("checksum")

        mergeable_ranks = list(encoding._mergeable_ranks.items())  # type: ignore[attr-defined]
        if pattern is None:
            pattern = ""
        if self._checksum is None:
            self._checksum = self._compute_checksum(
                mergeable_ranks=mergeable_ranks, pattern=pattern, special_tokens=special_tokens
            )

        self._encoding = encoding
        self._pattern = pattern
        self._record_special_tokens(special_tokens)
        self._rebuild_vocab(mergeable_ranks, special_tokens)
        self.vocab_size = encoding.n_vocab
        self._storage_dir = pickle_path.parent

    # ------------------------------------------------------------------
    # compatibility helpers used in checkpoints
    def get_vocab_size(self) -> int:
        return self.vocab_size

    def get_config(self) -> dict[str, Any]:
        return {
            "tokenizer_type": "rust-bpe",
            "vocab_size": self.vocab_size,
            "tokenizer_dir": str(self._storage_dir) if self._storage_dir else None,
            "pattern": self._pattern,
            "special_tokens": self.special_tokens_map(require_presence=False),
        }

    def checksum(self) -> str:
        return self._checksum or ""

    def special_tokens_map(self, *, require_presence: bool = True) -> dict[str, dict[str, Any]]:
        mapping: dict[str, dict[str, Any]] = {}
        encoding = self._ensure_encoding()
        for name in ["pad", "unk", "bos", "eos"]:
            token_name = getattr(self, f"{name}_token")
            token_id = getattr(self, f"{name}_id")
            present = token_id in encoding._special_tokens.values()  # type: ignore[attr-defined]
            mapping[name] = {"token": token_name, "id": token_id, "present": present}
            if require_presence and not present:
                raise ValueError(f"缺少必要的特殊 token: {name}")
        return mapping

    def diff_special_tokens(
        self, expected: dict[str, dict[str, Any]]
    ) -> dict[str, tuple[Any, Any]]:
        if not expected:
            return {}
        actual = self.special_tokens_map(require_presence=False)
        mismatches: dict[str, tuple[Any, Any]] = {}
        for name, expected_info in expected.items():
            actual_info = actual.get(name)
            if actual_info is None:
                mismatches[name] = (expected_info, None)
                continue
            keys_to_compare = {key for key in ("token", "id", "present") if key in expected_info}
            if not keys_to_compare:
                keys_to_compare = {"token", "id"}
            comparison = {k: expected_info.get(k) for k in keys_to_compare}
            actual_comp = {k: actual_info.get(k) for k in keys_to_compare}
            if comparison != actual_comp:
                mismatches[name] = (expected_info, actual_info)
        return mismatches

    def compute_unk_statistics(
        self, texts: Iterable[str], sample_size: int = 1000
    ) -> dict[str, Any]:
        encoding = self._ensure_encoding()
        unk_id = self.unk_id
        total_tokens = 0
        unk_tokens = 0
        sampled = 0
        iterator = self._ensure_iterator(texts)
        for text in iterator:
            if sample_size and sampled >= sample_size:
                break
            sampled += 1
            token_ids = encoding.encode(text, allowed_special=set(self._special_token_to_id.keys()))
            total_tokens += len(token_ids)
            unk_tokens += sum(1 for tid in token_ids if tid == unk_id)
        unk_rate = (unk_tokens / total_tokens) if total_tokens else 0.0
        return {
            "sampled_texts": sampled,
            "total_tokens": total_tokens,
            "unk_tokens": unk_tokens,
            "unk_rate": unk_rate,
        }

    def compute_unk_rate(self, texts: Iterable[str], sample_size: int = 1000) -> float:
        return self.compute_unk_statistics(texts, sample_size)["unk_rate"]

    def __len__(self) -> int:  # pragma: no cover - simple delegation
        return self.vocab_size


def train_rust_tokenizer_from_data(data_path: str, vocab_size: int = 30000) -> RustBPETokenizer:
    """Train a :class:`RustBPETokenizer` on a JSONL dataset."""

    print(f"📁 加载数据: {data_path}, 目标词汇表: {vocab_size:,}")
    texts: list[str] = []
    with open(data_path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "conversations" in data:
                for conv in data["conversations"]:
                    content = conv.get("content")
                    if content:
                        texts.append(content)
            elif "text" in data and data["text"]:
                texts.append(data["text"])
            elif "input" in data and "output" in data:
                texts.append(f"{data['input']} {data['output']}")
    tokenizer = RustBPETokenizer(vocab_size=vocab_size)
    tokenizer.train(texts)
    return tokenizer


__all__ = ["RustBPETokenizer", "train_rust_tokenizer_from_data"]
