"""Language modeling dataset for pre-training with padding support."""

from __future__ import annotations

from typing import Iterable, Sequence

import torch
from torch.utils.data import Dataset


class LanguageModelingDataset(Dataset):
    """Tokenise raw text on-the-fly and produce (X, Y, loss_mask) tuples.

    The reference implementation returns three tensors per sample: the
    left-shifted inputs ``X``, the right-shifted targets ``Y`` and a
    ``loss_mask`` indicating which target positions should contribute to the
    loss. This implementation mirrors that contract while supporting both
    HuggingFace tokenisers (callable with ``padding='max_length'``) and the
    project-local :class:`~src.tokenizer.bpe_tokenizer.BPETokenizer`.
    """

    def __init__(
        self,
        texts: Sequence[str] | Iterable[str],
        tokenizer,
        max_length: int = 512,
    ) -> None:
        self.texts = [self._ensure_text(sample) for sample in texts]
        self.tokenizer = tokenizer
        self.max_length = max(2, int(max_length))

        self.pad_id = self._resolve_token_id("pad_token_id", "pad_id", default=0)
        self.bos_id = self._resolve_token_id("bos_token_id", "bos_id")
        self.eos_id = self._resolve_token_id("eos_token_id", "eos_id")

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        token_ids = self._tokenise(self.texts[idx])

        if token_ids.size(0) < 2:
            token_ids = self._fallback_tensor()

        if token_ids.size(0) != self.max_length:
            token_ids = self._pad_or_truncate(token_ids)

        loss_mask = (token_ids != self.pad_id).long()

        inputs = token_ids[:-1].clone()
        targets = token_ids[1:].clone()
        loss_mask = loss_mask[1:].clone()

        return inputs, targets, loss_mask

    # ------------------------------------------------------------------
    def _ensure_text(self, sample) -> str:
        if isinstance(sample, dict) and "text" in sample:
            sample = sample["text"]
        if sample is None:
            return ""
        return str(sample)

    def _resolve_token_id(self, *names: str, default: int | None = None) -> int | None:
        for name in names:
            value = getattr(self.tokenizer, name, None)
            if value is not None:
                return int(value)
        return default

    def _tokenise(self, text: str) -> torch.Tensor:
        # Prefer HuggingFace-style call semantics if available.
        if callable(self.tokenizer):
            try:
                encoding = self.tokenizer(
                    text,
                    max_length=self.max_length,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt",
                )
            except TypeError:
                encoding = None
            else:
                input_ids = getattr(encoding, "input_ids", None)
                if input_ids is None and isinstance(encoding, dict):
                    input_ids = encoding.get("input_ids")
                if input_ids is not None:
                    tensor = input_ids.squeeze(0)
                    if not isinstance(tensor, torch.Tensor):
                        tensor = torch.tensor(input_ids, dtype=torch.long)
                    return tensor.to(dtype=torch.long)

        if hasattr(self.tokenizer, "encode"):
            tokens = self.tokenizer.encode(text, add_special_tokens=True)
            tokens = tokens[: self.max_length]
            if len(tokens) < self.max_length:
                tokens.extend([self.pad_id] * (self.max_length - len(tokens)))
            return torch.tensor(tokens, dtype=torch.long)

        raise TypeError("Unsupported tokenizer interface for LanguageModelingDataset")

    def _pad_or_truncate(self, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.size(0) > self.max_length:
            tensor = tensor[: self.max_length]
            if self.eos_id is not None:
                tensor[-1] = self.eos_id
            return tensor

        padding = self.max_length - tensor.size(0)
        if padding <= 0:
            return tensor

        pad_tensor = torch.full((padding,), self.pad_id, dtype=torch.long)
        return torch.cat([tensor, pad_tensor], dim=0)

    def _fallback_tensor(self) -> torch.Tensor:
        bos = self.bos_id if self.bos_id is not None else self.pad_id
        eos = self.eos_id if self.eos_id is not None else self.pad_id
        base = [bos, eos]
        if len(base) < self.max_length:
            base.extend([self.pad_id] * (self.max_length - len(base)))
        return torch.tensor(base, dtype=torch.long)
