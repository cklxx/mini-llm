"""Language modeling dataset utilities."""

from __future__ import annotations

import time
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset


class LanguageModelingDataset(Dataset):
    """Dataset that packs plain text into tokenized language modeling samples."""

    def __init__(
        self,
        texts: Sequence[str],
        tokenizer,
        max_length: int = 512,
        *,
        pretokenize: bool = True,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pretokenize = pretokenize

        if pretokenize:
            self._tokens = self._pretokenize_texts(list(texts))
            self.texts: list[str] | None = None
        else:
            self.texts = list(texts)
            self._tokens = None
        total = len(self._tokens) if self._tokens is not None else len(self.texts)
        print(f"📊 数据集初始化: {total} 条文本, 最大长度: {max_length}")

    def __len__(self) -> int:
        if self._tokens is not None:
            return self._tokens.shape[0]
        return len(self.texts) if self.texts is not None else 0

    def __getitem__(self, idx: int) -> torch.Tensor:
        if self._tokens is not None:
            return torch.from_numpy(self._tokens[idx])
        return self._encode_to_tensor(self.texts[idx], idx)

    # ------------------------------------------------------------------
    def _pretokenize_texts(self, texts: list[str]) -> np.ndarray:
        total = len(texts)
        tokens = np.empty((total, self.max_length), dtype=np.int64)
        start_time = time.time()

        print(f"  🔄 开始预编码 {total:,} 个样本...")

        # 单线程编码（简单稳定，避免multiprocessing问题）
        # tokenizer通常是C++实现，速度已经很快
        for idx, text in enumerate(texts):
            tokens[idx] = self._encode_numpy(text)

            # 定期显示进度
            if (idx + 1) % 5000 == 0 or idx == total - 1:
                elapsed = time.time() - start_time
                speed = (idx + 1) / elapsed if elapsed > 0 else 0.0
                eta = (total - idx - 1) / speed if speed > 0 else 0
                print(f"  🔄 预编码 {idx + 1:,}/{total:,} 样本 (速度 {speed:.1f}/s, 预计剩余 {eta/60:.1f}分钟)")

        elapsed = time.time() - start_time
        avg_speed = total / elapsed if elapsed > 0 else 0.0
        print(f"  ✅ 预编码完成: {total:,} 样本, 耗时 {elapsed:.1f}s, 平均速度 {avg_speed:.1f}/s")

        return tokens

    def _encode_numpy(self, text: str) -> np.ndarray:
        token_ids = self.tokenizer.encode(text, add_special_tokens=True)
        if len(token_ids) < 2:
            token_ids = [self.tokenizer.bos_id, self.tokenizer.eos_id]

        if len(token_ids) > self.max_length:
            token_ids = token_ids[: self.max_length]
        else:
            needed_padding = max(0, self.max_length - len(token_ids))
            if needed_padding:
                token_ids.extend([self.tokenizer.pad_id] * needed_padding)

        if len(token_ids) < 2:
            token_ids = [
                self.tokenizer.bos_id,
                self.tokenizer.eos_id,
                *([self.tokenizer.pad_id] * max(0, self.max_length - 2)),
            ]

        return np.asarray(token_ids, dtype=np.int64)

    def _encode_to_tensor(self, text: str, idx: int) -> torch.Tensor:
        try:
            return torch.from_numpy(self._encode_numpy(text))
        except Exception as exc:  # pragma: no cover - defensive logging
            print(f"❌ 处理数据项 {idx} 时发生错误: {exc}")
            if text:
                print(f"❌ 错误文本长度: {len(text)}")
                print(f"❌ 错误文本预览: {text[:200]}...")
            import traceback

            traceback.print_exc()
            default_tokens = [
                self.tokenizer.bos_id,
                self.tokenizer.eos_id,
                *([self.tokenizer.pad_id] * max(0, self.max_length - 2)),
            ]
            return torch.tensor(default_tokens, dtype=torch.long)
