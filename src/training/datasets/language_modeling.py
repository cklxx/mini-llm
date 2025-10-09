"""Language modeling dataset utilities."""
from __future__ import annotations

from typing import List

import torch
from torch.utils.data import Dataset


class LanguageModelingDataset(Dataset):
    """Dataset that packs plain text into tokenized language modeling samples."""

    def __init__(self, texts: List[str], tokenizer, max_length: int = 512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        print(f"📊 数据集初始化: {len(texts)} 条文本, 最大长度: {max_length}")

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> torch.Tensor:
        try:
            text = self.texts[idx]

            token_ids = self.tokenizer.encode(text, add_special_tokens=True)
            if len(token_ids) < 2:
                token_ids = [self.tokenizer.bos_id, self.tokenizer.eos_id]

            if len(token_ids) > self.max_length:
                token_ids = token_ids[: self.max_length]
            else:
                needed_padding = max(0, self.max_length - len(token_ids))
                token_ids.extend([self.tokenizer.pad_id] * needed_padding)

            if len(token_ids) < 2:
                token_ids = [
                    self.tokenizer.bos_id,
                    self.tokenizer.eos_id,
                    *([self.tokenizer.pad_id] * max(0, self.max_length - 2)),
                ]

            return torch.tensor(token_ids, dtype=torch.long)

        except Exception as exc:  # pragma: no cover - defensive logging
            print(f"❌ 处理数据项 {idx} 时发生错误: {exc}")
            if "text" in locals():
                print(f"❌ 错误文本长度: {len(text)}")
                print(f"❌ 错误文本预览: {text[:200]}...")
            if "token_ids" in locals():
                print(f"❌ Token IDs 长度: {len(token_ids)}")
                print(f"❌ Token IDs: {token_ids[:10]}...")
            import traceback

            traceback.print_exc()
            default_tokens = [
                self.tokenizer.bos_id,
                self.tokenizer.eos_id,
                *([self.tokenizer.pad_id] * max(0, self.max_length - 2)),
            ]
            return torch.tensor(default_tokens, dtype=torch.long)
