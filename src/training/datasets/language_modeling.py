"""Simplified language modeling dataset aligned with MiniMind preprocessing."""

from __future__ import annotations

from typing import Iterable, Sequence

import torch
from torch.utils.data import Dataset


class LanguageModelingDataset(Dataset):
    """Wrap plain text samples into fixed-length token tensors.

    This implementation mirrors the dataset pipeline used in
    https://github.com/jingyaogong/minimind by tokenising on-the-fly,
    padding/truncating to ``max_length`` and exposing the shifted labels
    expected by the training loop. Any tokenizer that provides ``encode``
    together with ``pad_id``/``bos_id``/``eos_id`` attributes can be used.
    """

    def __init__(
        self,
        texts: Sequence[str] | Iterable[str],
        tokenizer,
        max_length: int = 512,
    ) -> None:
        self.texts = [str(text) for text in texts]
        self.tokenizer = tokenizer
        self.max_length = max(2, int(max_length))

        # Cache frequently accessed ids to avoid attribute lookups during __getitem__
        self.pad_id = getattr(tokenizer, "pad_id", 0)
        self.bos_id = getattr(tokenizer, "bos_id", None)
        self.eos_id = getattr(tokenizer, "eos_id", None)

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        tokens = self._encode_with_padding(self.texts[idx])
        tensor = torch.tensor(tokens, dtype=torch.long)

        # Labels mirror the input sequence; the training loop performs the shift.
        labels = tensor.clone()
        attention_mask = (tensor != self.pad_id).long()

        return {
            "input_ids": tensor,
            "labels": labels,
            "attention_mask": attention_mask,
        }

    def _encode_with_padding(self, text: str) -> list[int]:
        tokens = self.tokenizer.encode(text, add_special_tokens=True)

        if len(tokens) < 2:
            tokens = self._fallback_tokens()

        if len(tokens) > self.max_length:
            tokens = tokens[: self.max_length]
            if self.eos_id is not None:
                tokens[-1] = self.eos_id
        elif len(tokens) < self.max_length:
            padding = [self.pad_id] * (self.max_length - len(tokens))
            tokens = tokens + padding

        return tokens

    def _fallback_tokens(self) -> list[int]:
        bos = self.bos_id if self.bos_id is not None else self.pad_id
        eos = self.eos_id if self.eos_id is not None else self.pad_id
        tokens = [bos, eos]
        if len(tokens) < self.max_length:
            tokens.extend([self.pad_id] * (self.max_length - len(tokens)))
        return tokens
