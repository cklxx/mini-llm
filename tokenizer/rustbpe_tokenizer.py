"""RustBPE-based tokenizer integration for MiniLLM."""
from __future__ import annotations

import json
import os
import pickle
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, List, Sequence, Tuple, cast

import rustbpe  # type: ignore
import tiktoken  # type: ignore

from utils.conversation import Message, conversation_to_template, normalize_conversation

SPECIAL_TOKENS = [
    "<|bos|>",
    "<|user_start|>",
    "<|user_end|>",
    "<|assistant_start|>",
    "<|assistant_end|>",
    "<|python_start|>",
    "<|python_end|>",
    "<|output_start|>",
    "<|output_end|>",
]

SPLIT_PATTERN = (
    r"'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,2}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"
)


class RustBPETokenizer:
    """Wrapper around rustbpe (training) + tiktoken (fast inference)."""

    def __init__(self, encoding: tiktoken.Encoding, bos_token: str = "<|bos|>") -> None:
        self.enc = encoding
        self.bos_token_id = self.encode_special(bos_token)

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------
    @classmethod
    def train_from_iterator(
        cls,
        iterator: Iterable[str],
        vocab_size: int,
        *,
        pattern: str = SPLIT_PATTERN,
    ) -> "RustBPETokenizer":
        """Train a tokenizer using rustbpe and return a fast tiktoken wrapper."""

        tokenizer = cast(Any, rustbpe).Tokenizer()
        vocab_size_no_special = vocab_size - len(SPECIAL_TOKENS)
        if vocab_size_no_special < 256:
            raise ValueError(
                "Vocab size too small. After reserving special tokens we need at least 256 mergeable tokens."
            )
        tokenizer.train_from_iterator(iterator, vocab_size_no_special, pattern=pattern)
        mergeable_ranks = {bytes(k): v for k, v in tokenizer.get_mergeable_ranks()}
        tokens_offset = len(mergeable_ranks)
        special_tokens = {name: tokens_offset + i for i, name in enumerate(SPECIAL_TOKENS)}
        encoding = tiktoken.Encoding(
            name="rustbpe",
            pat_str=tokenizer.get_pattern(),
            mergeable_ranks=mergeable_ranks,
            special_tokens=special_tokens,
        )
        return cls(encoding)

    @classmethod
    def from_directory(cls, directory: os.PathLike[str] | str) -> "RustBPETokenizer":
        directory = os.fspath(directory)
        with open(os.path.join(directory, "tokenizer.pkl"), "rb") as f:
            encoding = pickle.load(f)
        return cls(encoding)

    # ------------------------------------------------------------------
    def save(self, directory: os.PathLike[str] | str) -> None:
        directory = os.fspath(directory)
        Path(directory).mkdir(parents=True, exist_ok=True)
        with open(os.path.join(directory, "tokenizer.pkl"), "wb") as f:
            pickle.dump(self.enc, f)
        meta = {
            "name": self.enc.name,
            "n_vocab": self.enc.n_vocab,
            "special_tokens": list(SPECIAL_TOKENS),
            "pattern": SPLIT_PATTERN,
        }
        with open(os.path.join(directory, "tokenizer_meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

    def get_vocab_size(self) -> int:
        return self.enc.n_vocab

    def get_bos_token_id(self) -> int:
        return self.bos_token_id

    # ------------------------------------------------------------------
    # Encoding helpers
    # ------------------------------------------------------------------
    @lru_cache(maxsize=64)
    def encode_special(self, token: str) -> int:
        return self.enc.encode_single_token(token)

    def encode(self, text: str, *, num_threads: int = 8) -> List[int]:
        return self.enc.encode_ordinary(text, num_threads=num_threads)

    def decode(self, ids: Sequence[int]) -> str:
        return self.enc.decode(ids)

    # ------------------------------------------------------------------
    def render_conversation(
        self,
        conversation: Sequence[Message | dict],
        *,
        max_tokens: int = 2048,
        add_bos: bool = True,
    ) -> Tuple[List[int], List[int]]:
        """Render a conversation into token ids and a supervision mask."""

        if conversation and isinstance(conversation[0], Message):
            normalized: List[Message] = list(conversation)  # type: ignore[arg-type]
        else:
            normalized = normalize_conversation({"conversations": conversation})  # type: ignore[arg-type]

        ids: List[int] = []
        mask: List[int] = []

        def add_tokens(token_ids: Sequence[int] | int, mask_value: int) -> None:
            nonlocal ids, mask
            if isinstance(token_ids, int):
                token_ids = [token_ids]
            ids.extend(token_ids)
            mask.extend([mask_value] * len(token_ids))

        if add_bos:
            add_tokens(self.get_bos_token_id(), 0)

        user_start = self.encode_special("<|user_start|>")
        user_end = self.encode_special("<|user_end|>")
        assistant_start = self.encode_special("<|assistant_start|>")
        assistant_end = self.encode_special("<|assistant_end|>")

        for message in normalized:
            if message.role == "system":
                value_ids = self.encode(message.content)
                add_tokens(value_ids, 0)
                continue
            if message.role == "user":
                add_tokens(user_start, 0)
                value_ids = self.encode(message.content)
                add_tokens(value_ids, 0)
                add_tokens(user_end, 0)
                continue
            if message.role == "assistant":
                add_tokens(assistant_start, 0)
                value_ids = self.encode(message.content)
                add_tokens(value_ids, 1)
                add_tokens(assistant_end, 1)
                continue
            raise ValueError(f"Unsupported role in conversation: {message.role}")

        ids = ids[:max_tokens]
        mask = mask[:max_tokens]
        return ids, mask

    def render_prompt_text(self, conversation: Sequence[Message | dict], *, include_bos: bool = True) -> str:
        """Convert a conversation to the textual template used during tokenization."""

        normalized: List[Message]
        if conversation and isinstance(conversation[0], Message):
            normalized = list(conversation)  # type: ignore[arg-type]
        else:
            normalized = normalize_conversation({"conversations": conversation})  # type: ignore[arg-type]
        return conversation_to_template(normalized, include_bos=include_bos)

    def visualize_tokenization(self, ids: Sequence[int], mask: Sequence[int]) -> str:
        tokens = []
        for token_id, mask_val in zip(ids, mask):
            token_str = self.decode([token_id])
            if mask_val:
                tokens.append(f"<A>{token_str}</A>")
            else:
                tokens.append(token_str)
        return "|".join(tokens)
