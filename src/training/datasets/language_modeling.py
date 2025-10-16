"""Language modeling dataset utilities."""

from __future__ import annotations

import multiprocessing as mp
import time
from functools import partial
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset


def _encode_text_worker(text: str, tokenizer, max_length: int) -> np.ndarray:
    """多进程worker函数：编码单个文本"""
    token_ids = tokenizer.encode(text, add_special_tokens=True)
    if len(token_ids) < 2:
        token_ids = [tokenizer.bos_id, tokenizer.eos_id]

    if len(token_ids) > max_length:
        token_ids = token_ids[:max_length]
    else:
        needed_padding = max(0, max_length - len(token_ids))
        if needed_padding:
            token_ids.extend([tokenizer.pad_id] * needed_padding)

    if len(token_ids) < 2:
        token_ids = [
            tokenizer.bos_id,
            tokenizer.eos_id,
            *([tokenizer.pad_id] * max(0, max_length - 2)),
        ]

    return np.asarray(token_ids, dtype=np.int64)


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

        # 使用多进程加速预编码
        num_workers = min(mp.cpu_count(), 16)  # 最多使用16个进程

        if total < 1000 or num_workers <= 1:
            # 数据量小或单核，使用单线程
            for idx, text in enumerate(texts):
                tokens[idx] = self._encode_numpy(text)
                if (idx + 1) % 5000 == 0 or idx == total - 1:
                    elapsed = time.time() - start_time
                    speed = (idx + 1) / elapsed if elapsed > 0 else 0.0
                    print(f"  🔄 预编码 {idx + 1:,}/{total:,} 样本 (速度 {speed:.1f}/s)")
        else:
            # 数据量大，使用多进程
            print(f"  🚀 使用 {num_workers} 个进程并行预编码...")

            # 创建编码函数
            encode_func = partial(
                _encode_text_worker,
                tokenizer=self.tokenizer,
                max_length=self.max_length
            )

            # 分批处理并显示进度
            chunk_size = max(100, total // (num_workers * 10))
            with mp.Pool(processes=num_workers) as pool:
                results = []
                for i, encoded in enumerate(pool.imap(encode_func, texts, chunksize=chunk_size)):
                    tokens[i] = encoded
                    if (i + 1) % 5000 == 0 or i == total - 1:
                        elapsed = time.time() - start_time
                        speed = (i + 1) / elapsed if elapsed > 0 else 0.0
                        print(f"  🔄 预编码 {i + 1:,}/{total:,} 样本 (速度 {speed:.1f}/s)")

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
