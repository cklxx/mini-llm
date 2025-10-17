"""Language modeling dataset utilities."""

from __future__ import annotations

import concurrent.futures
import os
import pickle
import time
from multiprocessing import get_context
from typing import Any, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset


_WORKER_TOKENIZER: Any | None = None
_WORKER_MAX_LENGTH: int | None = None


def _init_pretokenize_worker(tokenizer_bytes: bytes, max_length: int) -> None:
    """Initialise worker local tokenizer state."""

    global _WORKER_TOKENIZER, _WORKER_MAX_LENGTH
    _WORKER_TOKENIZER = pickle.loads(tokenizer_bytes)
    _WORKER_MAX_LENGTH = max_length


def _worker_encode(index_and_text: tuple[int, str]) -> tuple[int, np.ndarray]:
    """Encode text inside a worker process."""

    global _WORKER_TOKENIZER, _WORKER_MAX_LENGTH
    if _WORKER_TOKENIZER is None or _WORKER_MAX_LENGTH is None:
        raise RuntimeError("预编码worker未正确初始化tokenizer状态")

    idx, text = index_and_text
    token_ids = _encode_with_tokenizer(_WORKER_TOKENIZER, text, _WORKER_MAX_LENGTH)
    return idx, np.asarray(token_ids, dtype=np.int64)


def _encode_with_tokenizer(tokenizer, text: str, max_length: int) -> list[int]:
    """Encode text using the provided tokenizer and max length constraints."""

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

    return token_ids


class LanguageModelingDataset(Dataset):
    """Dataset that packs plain text into tokenized language modeling samples."""

    def __init__(
        self,
        texts: Sequence[str],
        tokenizer,
        max_length: int = 512,
        *,
        pretokenize: bool = True,
        pretokenize_workers: int | None = None,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pretokenize = pretokenize
        self._explicit_worker_count = pretokenize_workers

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
    def _resolve_worker_count(self, total: int) -> int:
        if total <= 1:
            return 1

        if self._explicit_worker_count is not None:
            requested = max(1, self._explicit_worker_count)
        else:
            env_value = os.environ.get("MINIGPT_PRETOKENIZE_WORKERS")
            if env_value is not None:
                try:
                    requested = max(1, int(env_value))
                except ValueError:
                    requested = 1
            else:
                requested = min(16, os.cpu_count() or 1)

        return min(requested, total)

    def _pretokenize_texts(self, texts: list[str]) -> np.ndarray:
        total = len(texts)
        tokens = np.empty((total, self.max_length), dtype=np.int64)
        start_time = time.time()

        worker_count = self._resolve_worker_count(total)
        print(
            f"  🔄 开始预编码 {total:,} 个样本..."
            + (f" (并行 {worker_count} workers)" if worker_count > 1 else "")
        )

        if worker_count <= 1:
            self._pretokenize_texts_single(texts, tokens, start_time)
        else:
            try:
                serialized_tokenizer = pickle.dumps(self.tokenizer)
            except Exception as exc:  # pragma: no cover - defensive fallback
                print(f"  ⚠️ 无法序列化tokenizer以进行并行预编码: {exc}，回退到单线程模式。")
                self._pretokenize_texts_single(texts, tokens, start_time)
            else:
                mp_ctx = get_context("spawn")
                with concurrent.futures.ProcessPoolExecutor(
                    max_workers=worker_count,
                    mp_context=mp_ctx,
                    initializer=_init_pretokenize_worker,
                    initargs=(serialized_tokenizer, self.max_length),
                ) as executor:
                    # ``executor.map`` streams results back in submission order without
                    # building an unbounded future queue.  ``chunksize`` keeps each worker
                    # busy while avoiding the enormous memory spike that ``submit``ing the
                    # entire corpus at once would trigger when datasets contain millions of
                    # samples.
                    chunk_hint = max(1, min(64, total // max(worker_count, 1)))
                    result_iter = executor.map(
                        _worker_encode,
                        enumerate(texts),
                        chunksize=chunk_hint,
                    )
                    for completed, (idx, encoded) in enumerate(result_iter, start=1):
                        tokens[idx] = encoded
                        if completed % 5000 == 0 or completed == total:
                            elapsed = time.time() - start_time
                            speed = completed / elapsed if elapsed > 0 else 0.0
                            eta = (total - completed) / speed if speed > 0 else 0
                            print(
                                f"  🔄 预编码 {completed:,}/{total:,} 样本 (速度 {speed:.1f}/s, 预计剩余 {eta/60:.1f}分钟)"
                            )

        elapsed = time.time() - start_time
        avg_speed = total / elapsed if elapsed > 0 else 0.0
        print(f"  ✅ 预编码完成: {total:,} 样本, 耗时 {elapsed:.1f}s, 平均速度 {avg_speed:.1f}/s")

        return tokens

    def _pretokenize_texts_single(
        self, texts: list[str], tokens: np.ndarray, start_time: float
    ) -> np.ndarray:
        total = len(texts)
        for idx, text in enumerate(texts):
            tokens[idx] = self._encode_numpy(text)
            if (idx + 1) % 5000 == 0 or idx == total - 1:
                elapsed = time.time() - start_time
                speed = (idx + 1) / elapsed if elapsed > 0 else 0.0
                eta = (total - idx - 1) / speed if speed > 0 else 0
                print(
                    f"  🔄 预编码 {idx + 1:,}/{total:,} 样本 (速度 {speed:.1f}/s, 预计剩余 {eta/60:.1f}分钟)"
                )
        return tokens

    def _encode_numpy(self, text: str) -> np.ndarray:
        token_ids = _encode_with_tokenizer(self.tokenizer, text, self.max_length)
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
