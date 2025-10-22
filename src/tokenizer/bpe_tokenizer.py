"""Utilities for working with BPE tokenizers.

This module now delegates tokenizer training and serialization to the
HuggingFace `tokenizers`/`transformers` stack rather than maintaining a
project-local implementation.  It keeps a light wrapper so the rest of the
codebase can continue using ``BPETokenizer`` while benefiting from the battle-
tested reference implementation.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.normalizers import Lowercase, NFKC, Sequence as NormalizerSequence, Strip
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.processors import TemplateProcessing
from tokenizers.trainers import BpeTrainer

try:  # transformers is an optional dependency during linting
    from transformers import PreTrainedTokenizerFast
except ModuleNotFoundError:  # pragma: no cover - handled at runtime
    PreTrainedTokenizerFast = None  # type: ignore


def _ensure_transformers_available() -> None:
    if PreTrainedTokenizerFast is None:  # pragma: no cover - defensive guard
        raise RuntimeError(
            "加载或训练 HuggingFace 分词器需要安装 transformers 库。"
        )


class BPETokenizer:
    """A thin wrapper around :class:`PreTrainedTokenizerFast`.

    The wrapper keeps compatibility with the previous project API while relying
    entirely on the official HuggingFace tokenizer implementation for training,
    encoding and decoding.
    """

    def __init__(
        self,
        vocab_size: int = 30000,
        *,
        lowercase: bool = True,
        normalize_nfkc: bool = True,
        strip_spaces: bool = True,
        enable_byte_fallback: bool = False,
    ) -> None:
        self.vocab_size = vocab_size
        self.lowercase = lowercase
        self.normalize_nfkc = normalize_nfkc
        self.strip_spaces = strip_spaces
        self.enable_byte_fallback = enable_byte_fallback

        # Special tokens mirror the historical defaults.
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"

        self.pad_id = 0
        self.unk_id = 1
        self.bos_id = 2
        self.eos_id = 3

        self._hf_tokenizer: PreTrainedTokenizerFast | None = None
        self._hf_tokenizer_dir: Path | None = None
        self._hf_checksum: str | None = None

        # Provide a cached vocab for quick lookups and metadata.
        self.vocab: dict[str, int] = {}

    # ------------------------------------------------------------------
    def _using_hf(self) -> bool:
        return self._hf_tokenizer is not None

    def _require_hf_tokenizer(self) -> PreTrainedTokenizerFast:
        if self._hf_tokenizer is None:
            raise RuntimeError("Tokenizer 未加载。请先调用 train() 或 load().")
        return self._hf_tokenizer

    # ------------------------------------------------------------------
    def _build_normalizer(self) -> NormalizerSequence | Lowercase | NFKC | Strip | None:
        stages: list[Any] = []
        if self.normalize_nfkc:
            stages.append(NFKC())
        if self.strip_spaces:
            stages.append(Strip())
        if self.lowercase:
            stages.append(Lowercase())

        if not stages:
            return None
        if len(stages) == 1:
            return stages[0]
        return NormalizerSequence(stages)

    def _assign_hf_tokenizer(self, tokenizer: PreTrainedTokenizerFast) -> None:
        self._hf_tokenizer = tokenizer
        self.vocab = dict(tokenizer.get_vocab())
        self.vocab_size = tokenizer.vocab_size

        if tokenizer.pad_token is not None:
            self.pad_token = tokenizer.pad_token
        if tokenizer.unk_token is not None:
            self.unk_token = tokenizer.unk_token
        if tokenizer.bos_token is not None:
            self.bos_token = tokenizer.bos_token
        if tokenizer.eos_token is not None:
            self.eos_token = tokenizer.eos_token

        if tokenizer.pad_token_id is not None:
            self.pad_id = tokenizer.pad_token_id
        if tokenizer.unk_token_id is not None:
            self.unk_id = tokenizer.unk_token_id
        if tokenizer.bos_token_id is not None:
            self.bos_id = tokenizer.bos_token_id
        if tokenizer.eos_token_id is not None:
            self.eos_id = tokenizer.eos_token_id

        # byte fallback is not exposed by HuggingFace's BPE implementation.
        self.enable_byte_fallback = False

    # ------------------------------------------------------------------
    def train(self, texts: Iterable[str]) -> None:
        """Train a tokenizer using HuggingFace's implementation."""

        _ensure_transformers_available()

        tokenizer = Tokenizer(BPE(unk_token=self.unk_token))
        normalizer = self._build_normalizer()
        if normalizer is not None:
            tokenizer.normalizer = normalizer
        tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)

        special_tokens = [self.pad_token, self.unk_token, self.bos_token, self.eos_token]
        trainer = BpeTrainer(vocab_size=self.vocab_size, special_tokens=special_tokens)
        tokenizer.train_from_iterator(texts, trainer=trainer)

        bos_id = tokenizer.token_to_id(self.bos_token)
        eos_id = tokenizer.token_to_id(self.eos_token)
        pad_id = tokenizer.token_to_id(self.pad_token)
        unk_id = tokenizer.token_to_id(self.unk_token)

        for name, value in {
            "BOS": bos_id,
            "EOS": eos_id,
            "PAD": pad_id,
            "UNK": unk_id,
        }.items():
            if value is None:
                raise RuntimeError(f"训练分词器时缺少 {name} token，请检查训练数据和配置。")

        tokenizer.post_processor = TemplateProcessing(
            single=f"{self.bos_token} $A {self.eos_token}",
            pair=(
                f"{self.bos_token} $A {self.eos_token} "
                f"{self.bos_token} $B {self.eos_token}"
            ),
            special_tokens=[
                (self.bos_token, bos_id),
                (self.eos_token, eos_id),
            ],
        )
        tokenizer.enable_padding(pad_id=pad_id, pad_token=self.pad_token)

        hf_tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=tokenizer,
            bos_token=self.bos_token,
            eos_token=self.eos_token,
            unk_token=self.unk_token,
            pad_token=self.pad_token,
        )
        self._assign_hf_tokenizer(hf_tokenizer)
        self._hf_tokenizer_dir = None
        self._hf_checksum = None

    # ------------------------------------------------------------------
    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        tokenizer = self._require_hf_tokenizer()
        return tokenizer.encode(text, add_special_tokens=add_special_tokens)

    def decode(self, token_ids: list[int]) -> str:
        tokenizer = self._require_hf_tokenizer()
        if not token_ids:
            return ""
        return tokenizer.decode(token_ids, skip_special_tokens=True)

    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        tokenizer = self._require_hf_tokenizer()
        dest = Path(path)
        if dest.suffix:
            raise ValueError("HuggingFace 分词器必须保存到目录中，请提供目录路径")
        dest.mkdir(parents=True, exist_ok=True)
        tokenizer.save_pretrained(str(dest))
        self._hf_tokenizer_dir = dest
        self._hf_checksum = self._compute_hf_checksum(dest)
        print(f"分词器已保存到: {dest}")

    def load(self, path: str) -> None:
        path_obj = Path(path)
        if path_obj.is_file() and path_obj.suffix not in {".json"}:
            raise RuntimeError(
                "项目不再支持自定义的 pickle 分词器，请提供 HuggingFace tokenizer.json"
            )
        self._load_hf_tokenizer(path_obj)
        print(f"分词器已加载: {path}")

    def _load_hf_tokenizer(self, location: Path) -> None:
        _ensure_transformers_available()

        if location.is_dir():
            directory = location
            tokenizer_file = directory / "tokenizer.json"
        else:
            directory = location.parent
            tokenizer_file = location

        if not tokenizer_file.exists():
            raise FileNotFoundError(f"未找到 tokenizer.json: {tokenizer_file}")

        config_path = directory / "tokenizer_config.json"
        try:
            tokenizer = PreTrainedTokenizerFast.from_pretrained(str(directory))
        except OSError:
            config_kwargs: dict[str, Any] = {}
            if config_path.exists():
                with open(config_path, "r", encoding="utf-8") as handle:
                    config_kwargs = json.load(handle)
            tokenizer = PreTrainedTokenizerFast(tokenizer_file=str(tokenizer_file), **config_kwargs)

        self._assign_hf_tokenizer(tokenizer)
        self._hf_tokenizer_dir = directory
        self._hf_checksum = self._compute_hf_checksum(directory)

    def _compute_hf_checksum(self, directory: Path) -> str:
        hasher = hashlib.sha256()
        files = [directory / "tokenizer.json", directory / "tokenizer_config.json"]
        for file_path in files:
            if file_path.exists():
                hasher.update(file_path.name.encode("utf-8"))
                with open(file_path, "rb") as handle:
                    hasher.update(handle.read())
        return hasher.hexdigest()

    # ------------------------------------------------------------------
    def get_vocab_size(self) -> int:
        tokenizer = self._require_hf_tokenizer()
        return tokenizer.vocab_size

    def get_config(self) -> dict[str, Any]:
        if not self._using_hf():
            return {}
        return {
            "tokenizer_type": "huggingface",
            "vocab_size": self.vocab_size,
            "tokenizer_dir": str(self._hf_tokenizer_dir) if self._hf_tokenizer_dir else None,
            "special_tokens": self.special_tokens_map(require_presence=True),
        }

    def checksum(self) -> str:
        return self._hf_checksum or ""

    def special_tokens_map(self, *, require_presence: bool = True) -> dict[str, dict[str, Any]]:
        tokenizer = self._require_hf_tokenizer()
        mapping: dict[str, dict[str, Any]] = {}
        for name, fallback_token, fallback_id in [
            ("pad", self.pad_token, self.pad_id),
            ("unk", self.unk_token, self.unk_id),
            ("bos", self.bos_token, self.bos_id),
            ("eos", self.eos_token, self.eos_id),
        ]:
            token = getattr(tokenizer, f"{name}_token", None)
            token_id = getattr(tokenizer, f"{name}_token_id", None)
            mapping[name] = {
                "token": token if token is not None else fallback_token,
                "id": token_id if token_id is not None else fallback_id,
                "present": token is not None and token_id is not None,
            }
            if require_presence and token is None:
                raise ValueError(f"缺少必要的特殊token: {name}")
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
        tokenizer = self._require_hf_tokenizer()
        total_tokens = 0
        unk_tokens = 0
        sampled = 0
        unk_id = tokenizer.unk_token_id
        for text in texts:
            if sample_size and sampled >= sample_size:
                break
            sampled += 1
            token_ids = tokenizer.encode(text, add_special_tokens=False)
            total_tokens += len(token_ids)
            if unk_id is not None:
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

    def __len__(self) -> int:
        return self.get_vocab_size()


def train_tokenizer_from_data(data_path: str, vocab_size: int = 30000) -> BPETokenizer:
    """Train a tokenizer directly from a JSONL dataset using HuggingFace tooling."""

    print(f"📁 加载数据: {data_path}, 目标词汇表: {vocab_size:,}")
    texts: list[str] = []
    with open(data_path, encoding="utf-8") as f:
        for line in f:
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

    tokenizer = BPETokenizer(vocab_size=vocab_size)
    tokenizer.train(texts)
    return tokenizer


if __name__ == "__main__":  # pragma: no cover - manual smoke test
    sample_texts = [
        "Hello world! How are you?",
        "I am learning about BPE tokenization.",
        "This is a sample text for training.",
        "BPE stands for Byte Pair Encoding.",
    ]

    tokenizer = BPETokenizer(vocab_size=1000)
    tokenizer.train(sample_texts)

    test_text = "Hello! How are you doing?"
    token_ids = tokenizer.encode(test_text)
    decoded_text = tokenizer.decode(token_ids)

    print(f"原文: {test_text}")
    print(f"编码: {token_ids}")
    print(f"解码: {decoded_text}")
    print(f"词汇表大小: {tokenizer.get_vocab_size()}")
