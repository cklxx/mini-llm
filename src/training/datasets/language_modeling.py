"""Language modeling dataset utilities."""

from __future__ import annotations

import concurrent.futures
import hashlib
import os
import pickle
import threading
import time
import tempfile
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
        initial_pretokenize_items: int | None = None,
        background_pretokenize: bool = True,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pretokenize = pretokenize
        self._explicit_worker_count = pretokenize_workers
        self._initial_target = initial_pretokenize_items
        self._background_enabled = background_pretokenize

        self._tokens: np.ndarray | np.memmap | None = None
        self._ready_mask: np.ndarray | np.memmap | None = None
        self._token_cache_path: str | None = None
        self._token_tmp_path: str | None = None
        self._ready_cache_path: str | None = None
        self._ready_tmp_path: str | None = None
        self._background_thread: threading.Thread | None = None
        self._background_error: BaseException | None = None
        self._background_in_flight: bool = False
        self._progress_lock = threading.Lock()
        self._ready_high_water: int = -1
        self._last_progress_speed: float | None = None
        try:
            margin_env = os.environ.get("MINILLM_PRETOKENIZE_WAIT_MARGIN")
            self._ready_wait_margin = (
                max(0, int(margin_env)) if margin_env is not None else 2048
            )
        except (TypeError, ValueError):  # pragma: no cover - env parsing guard
            self._ready_wait_margin = 2048
        try:
            wait_env = os.environ.get("MINILLM_PRETOKENIZE_WAIT_TIMEOUT")
            self._ready_wait_timeout = (
                max(0.0, float(wait_env)) if wait_env is not None else 0.5
            )
        except (TypeError, ValueError):  # pragma: no cover - env parsing guard
            self._ready_wait_timeout = 0.5
        self._total_texts: int = 0

        if pretokenize:
            text_list = list(texts)
            self._total_texts = len(text_list)
            cache_path = self._build_cache_path(text_list)
            cached_tokens, cached_ready = self._load_cached_tokens(
                cache_path, len(text_list)
            )
            if cached_tokens is not None:
                self._token_cache_path = cache_path
                self._ready_cache_path = self._ready_cache_filename(cache_path)
                self._tokens = cached_tokens
                self._ready_mask = cached_ready
                self._mark_all_ready()
                self.texts: list[str] | None = None
            else:
                self.texts = text_list
                self._token_cache_path = cache_path
                self._ready_cache_path = (
                    self._ready_cache_filename(cache_path) if cache_path else None
                )
                self._tokens = self._pretokenize_incremental(text_list)
        else:
            self.texts = list(texts)
            self._tokens = None
            self._total_texts = len(self.texts)
        if self._tokens is not None and not self._total_texts:
            self._total_texts = self._tokens.shape[0]
        total = self._total_texts
        print(f"📊 数据集初始化: {total} 条文本, 最大长度: {max_length}")

    def __len__(self) -> int:
        return self._total_texts

    def __getitem__(self, idx: int) -> torch.Tensor:
        self._check_background_status()

        if self._tokens is not None:
            tokens = self._ensure_token_buffer()
            ready = self._ensure_ready_buffer()
            if ready is not None and ready[idx]:
                return torch.from_numpy(tokens[idx])
            if self._should_wait_for_index(idx) and self._wait_for_ready(idx, ready):
                return torch.from_numpy(tokens[idx])

        if self.texts is None:
            # Fallback when texts are unavailable but tokens missing (should be rare)
            tokens = self._ensure_token_buffer()
            return torch.from_numpy(tokens[idx])

        tensor = self._encode_to_tensor(self.texts[idx], idx)

        if self._tokens is not None:
            tokens = self._ensure_token_buffer()
            ready = self._ensure_ready_buffer()
            tokens[idx] = tensor.numpy()
            if ready is not None:
                ready[idx] = 1
                self._flush_ready_if_needed()
            self._flush_tokens_if_needed()
        self._update_ready_high_water(idx)
        return tensor

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

    def _pretokenize_incremental(self, texts: list[str]) -> np.ndarray | np.memmap:
        total = len(texts)
        if not self._total_texts:
            self._total_texts = total
        tokens, tmp_path = self._prepare_token_buffer(total, self._token_cache_path)
        ready, ready_tmp = self._prepare_ready_buffer(total, self._ready_cache_path)
        self._token_tmp_path = tmp_path
        self._ready_tmp_path = ready_tmp
        self._tokens = tokens
        self._ready_mask = ready

        worker_count = self._resolve_worker_count(total)
        mode_desc = (
            f" (并行 {worker_count} workers)" if worker_count > 1 else " (单线程)"
        )

        start = time.time()
        target = self._resolve_initial_target(total)
        if target <= 0:
            target = min(total, 1)

        print(
            f"  🔄 初始预编码 {target:,}/{total:,} 个样本...{mode_desc}"
        )
        self._pretokenize_range(0, target, texts, worker_count, start)
        self._flush_tokens_if_needed()
        self._flush_ready_if_needed()

        if target >= total:
            elapsed = time.time() - start
            speed = total / elapsed if elapsed > 0 else 0.0
            print(
                f"  ✅ 预编码完成: {total:,} 样本, 耗时 {elapsed:.1f}s, 平均速度 {speed:.1f}/s"
            )
            self._finalize_cache()
            self.texts = None
            self._background_in_flight = False
            return self._ensure_token_buffer()

        if self._background_enabled:
            self._background_in_flight = True
            self._background_thread = threading.Thread(
                target=self._background_worker,
                args=(target, texts, worker_count, start),
                name="LMBackgroundPretokenize",
                daemon=True,
            )
            self._background_thread.start()

        return self._ensure_token_buffer()

    def _background_worker(
        self,
        start_index: int,
        texts: list[str],
        worker_count: int,
        start_time: float,
    ) -> None:
        try:
            total = len(texts)
            self._pretokenize_range(start_index, total, texts, worker_count, start_time)
            self._flush_tokens_if_needed()
            self._flush_ready_if_needed()
            self._finalize_cache()
            elapsed = time.time() - start_time
            avg_speed = total / elapsed if elapsed > 0 else 0.0
            print(
                f"  ✅ 后台预编码完成: {total:,} 样本, 总耗时 {elapsed:.1f}s, 平均速度 {avg_speed:.1f}/s"
            )
            self.texts = None
        except BaseException as exc:  # pragma: no cover - defensive guard
            self._background_error = exc
            print(f"  ❌ 后台预编码失败: {exc}")
        finally:
            self._background_in_flight = False

    def _resolve_initial_target(self, total: int) -> int:
        if self._initial_target is None:
            return total
        return max(0, min(total, self._initial_target))

    def _background_active(self) -> bool:
        thread = self._background_thread
        if thread is not None and thread.is_alive():
            return True
        return self._background_in_flight

    def _wait_for_ready(
        self,
        idx: int,
        ready: np.ndarray | np.memmap | None,
    ) -> bool:
        if ready is None or self._ready_wait_timeout <= 0:
            return False
        if not self._background_active():
            return False

        deadline = time.time() + self._ready_wait_timeout
        sleep_interval = 0.05

        while True:
            if ready[idx]:
                return True
            self._check_background_status()
            if not self._background_active():
                return bool(ready[idx])
            now = time.time()
            if now >= deadline:
                break
            time.sleep(min(sleep_interval, max(0.0, deadline - now)))
        return bool(ready[idx])

    def _pretokenize_range(
        self,
        start_idx: int,
        end_idx: int,
        texts: list[str],
        worker_count: int,
        start_time: float,
    ) -> None:
        total = end_idx - start_idx
        if total <= 0:
            return

        tokens = self._ensure_token_buffer()
        ready = self._ensure_ready_buffer()

        total_texts = len(texts)

        if worker_count <= 1 or total < worker_count:
            for offset, text in enumerate(texts[start_idx:end_idx]):
                idx = start_idx + offset
                tokens[idx] = self._encode_numpy(text)
                if ready is not None:
                    ready[idx] = 1
                self._update_ready_high_water(idx)
                processed = idx + 1
                self._update_progress_metrics(processed, start_time)
                if ((processed) % 5000 == 0) or processed == total_texts:
                    self._log_progress(processed, total_texts, start_time)
        else:
            try:
                serialized_tokenizer = pickle.dumps(self.tokenizer)
            except Exception as exc:  # pragma: no cover - defensive fallback
                print(
                    f"  ⚠️ 无法序列化tokenizer以进行并行预编码: {exc}，回退到单线程模式。"
                )
                self._pretokenize_range(start_idx, end_idx, texts, 1, start_time)
                return

            mp_ctx = get_context("spawn")
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=worker_count,
                mp_context=mp_ctx,
                initializer=_init_pretokenize_worker,
                initargs=(serialized_tokenizer, self.max_length),
            ) as executor:
                chunk_hint = max(1, min(64, total // max(worker_count, 1)))
                enumerated = (
                    (start_idx + i, text)
                    for i, text in enumerate(texts[start_idx:end_idx])
                )
                result_iter = executor.map(
                    _worker_encode,
                    enumerated,
                    chunksize=chunk_hint,
                )
                processed = start_idx
                for idx, encoded in result_iter:
                    tokens[idx] = encoded
                    if ready is not None:
                        ready[idx] = 1
                    self._update_ready_high_water(idx)
                    processed += 1
                    self._update_progress_metrics(processed, start_time)
                    if (processed % 5000 == 0) or processed == total_texts:
                        self._log_progress(processed, total_texts, start_time)

        if end_idx < total_texts and ((end_idx % 5000) != 0):
            self._log_progress(end_idx, total_texts, start_time)

    def _log_progress(self, processed: int, total: int, start_time: float) -> None:
        elapsed = time.time() - start_time
        speed = processed / elapsed if elapsed > 0 else 0.0
        remaining = max(0, total - processed)
        eta = remaining / speed if speed > 0 else 0.0
        print(
            f"  🔄 预编码 {processed:,}/{total:,} 样本 (速度 {speed:.1f}/s, 预计剩余 {eta/60:.1f}分钟)"
        )
        self._update_progress_metrics(processed, start_time)

    def _ensure_token_buffer(self) -> np.ndarray | np.memmap:
        if self._tokens is not None:
            return self._tokens

        shape = (self._total_texts, self.max_length)
        if self._token_cache_path and os.path.exists(self._token_cache_path):
            self._tokens = np.memmap(
                self._token_cache_path,
                dtype=np.int64,
                mode="r+",
                shape=shape,
            )
            return self._tokens
        if self._token_tmp_path and os.path.exists(self._token_tmp_path):
            self._tokens = np.memmap(
                self._token_tmp_path,
                dtype=np.int64,
                mode="r+",
                shape=shape,
            )
            return self._tokens
        raise RuntimeError("预编码token缓冲未初始化")

    def _ensure_ready_buffer(self) -> np.ndarray | np.memmap | None:
        if self._ready_mask is None:
            return None
        if isinstance(self._ready_mask, np.ndarray):
            return self._ready_mask

        shape = (self._total_texts,)
        if self._ready_cache_path and os.path.exists(self._ready_cache_path):
            self._ready_mask = np.memmap(
                self._ready_cache_path,
                dtype=np.uint8,
                mode="r+",
                shape=shape,
            )
            return self._ready_mask
        if self._ready_tmp_path and os.path.exists(self._ready_tmp_path):
            self._ready_mask = np.memmap(
                self._ready_tmp_path,
                dtype=np.uint8,
                mode="r+",
                shape=shape,
            )
            return self._ready_mask
        return None

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

    def _prepare_token_buffer(
        self, total: int, cache_path: str | None
    ) -> tuple[np.ndarray, str | None]:
        if not cache_path:
            return np.empty((total, self.max_length), dtype=np.int64), None

        cache_dir = os.path.dirname(cache_path)
        try:
            os.makedirs(cache_dir, exist_ok=True)
        except OSError as exc:  # pragma: no cover - filesystem errors
            print(f"  ⚠️ 无法创建缓存目录 {cache_dir}: {exc}，跳过缓存。")
            return np.empty((total, self.max_length), dtype=np.int64), None

        fd, tmp_path = tempfile.mkstemp(prefix="cache-", suffix=".npy", dir=cache_dir)
        os.close(fd)
        token_buffer = np.memmap(
            tmp_path, dtype=np.int64, mode="w+", shape=(total, self.max_length)
        )
        return token_buffer, tmp_path

    def _prepare_ready_buffer(
        self, total: int, cache_path: str | None
    ) -> tuple[np.ndarray | np.memmap, str | None]:
        if not cache_path:
            return np.zeros(total, dtype=np.uint8), None

        cache_dir = os.path.dirname(cache_path)
        try:
            os.makedirs(cache_dir, exist_ok=True)
        except OSError as exc:  # pragma: no cover - filesystem errors
            print(f"  ⚠️ 无法创建缓存目录 {cache_dir}: {exc}，跳过缓存。")
            return np.zeros(total, dtype=np.uint8), None

        fd, tmp_path = tempfile.mkstemp(prefix="ready-", suffix=".npy", dir=cache_dir)
        os.close(fd)
        ready_buffer = np.memmap(tmp_path, dtype=np.uint8, mode="w+", shape=(total,))
        return ready_buffer, tmp_path

    def _cleanup_tmp_cache(self, tmp_path: str | None) -> None:
        if not tmp_path:
            return
        try:
            os.remove(tmp_path)
        except OSError:
            pass

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

    # ------------------------------------------------------------------
    def _build_cache_path(self, texts: Sequence[str]) -> str | None:
        cache_dir = os.environ.get("MINILLM_CACHE_DIR")
        if not cache_dir:
            home_dir = os.path.expanduser("~")
            if not home_dir or home_dir == "~":
                return None
            cache_dir = os.path.join(home_dir, ".cache", "mini-llm")

        cache_dir = os.path.join(cache_dir, "language_modeling")
        try:
            os.makedirs(cache_dir, exist_ok=True)
        except OSError as exc:  # pragma: no cover - filesystem errors are environment specific
            print(f"  ⚠️ 无法创建缓存目录 {cache_dir}: {exc}，跳过缓存。")
            return None

        hasher = hashlib.sha256()
        try:
            tokenizer_bytes = pickle.dumps(self.tokenizer)
        except Exception:  # pragma: no cover - pickle failure is rare
            tokenizer_bytes = repr(self.tokenizer).encode("utf-8", "ignore")
        hasher.update(hashlib.sha256(tokenizer_bytes).digest())
        hasher.update(str(self.max_length).encode("utf-8"))
        hasher.update(len(texts).to_bytes(8, "little"))
        for text in texts:
            encoded = text.encode("utf-8", "ignore")
            hasher.update(len(encoded).to_bytes(8, "little"))
            hasher.update(hashlib.sha1(encoded).digest())

        cache_key = hasher.hexdigest()
        return os.path.join(cache_dir, f"{cache_key}.npy")

    def _ready_cache_filename(self, cache_path: str | None) -> str | None:
        if not cache_path:
            return None
        return f"{cache_path}.ready"

    def _load_cached_tokens(
        self, cache_path: str | None, expected_rows: int
    ) -> tuple[np.ndarray | np.memmap | None, np.ndarray | np.memmap | None]:
        if not cache_path or not os.path.exists(cache_path):
            return None, None

        try:
            cached = np.load(cache_path, allow_pickle=False, mmap_mode="r+")
        except Exception as exc:  # pragma: no cover - depends on external state
            print(f"  ⚠️ 无法从缓存加载 {cache_path}: {exc}，重新预编码。")
            return None, None

        if not isinstance(cached, np.ndarray):
            print(f"  ⚠️ 缓存文件 {cache_path} 非数组格式，重新预编码。")
            return None, None

        if cached.dtype != np.int64:
            print(f"  ⚠️ 缓存文件 {cache_path} dtype 不匹配，重新预编码。")
            return None, None

        if cached.ndim != 2 or cached.shape[1] != self.max_length:
            print(f"  ⚠️ 缓存文件 {cache_path} 形状不匹配，重新预编码。")
            return None, None

        if cached.shape[0] != expected_rows:
            print(f"  ⚠️ 缓存文件 {cache_path} 行数不一致，重新预编码。")
            return None, None

        ready_path = self._ready_cache_filename(cache_path)
        ready_mask: np.ndarray | np.memmap | None = None
        if ready_path and os.path.exists(ready_path):
            try:
                ready_mask = np.memmap(
                    ready_path,
                    dtype=np.uint8,
                    mode="r+",
                    shape=(expected_rows,),
                )
            except Exception as exc:  # pragma: no cover - filesystem/environment specific
                print(
                    f"  ⚠️ 无法从缓存加载就绪掩码 {ready_path}: {exc}，将重置为全部可用。"
                )
                ready_mask = np.ones(expected_rows, dtype=np.uint8)
        else:
            ready_mask = np.ones(expected_rows, dtype=np.uint8)

        self._total_texts = expected_rows
        print(f"  💾 从缓存加载预编码数据: {cache_path}")
        return cached, ready_mask

    def _save_tokens_to_cache(
        self, cache_path: str | None, tokens: np.ndarray
    ) -> None:
        if cache_path is None:
            return

        cache_dir = os.path.dirname(cache_path)
        try:
            os.makedirs(cache_dir, exist_ok=True)
        except OSError as exc:  # pragma: no cover - filesystem errors
            print(f"  ⚠️ 无法创建缓存目录 {cache_dir}: {exc}，跳过缓存。")
            return

        tmp_path = None
        try:
            fd, tmp_path = tempfile.mkstemp(prefix="cache-", suffix=".npy", dir=cache_dir)
            with os.fdopen(fd, "wb") as tmp_file:
                np.save(tmp_file, tokens, allow_pickle=False)
            os.replace(tmp_path, cache_path)
        except Exception as exc:  # pragma: no cover - depends on filesystem state
            print(f"  ⚠️ 写入缓存失败 {cache_path}: {exc}，跳过缓存。")
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
        else:
            print(f"  💾 已缓存预编码结果: {cache_path}")

    def _flush_tokens_if_needed(self) -> None:
        if isinstance(self._tokens, np.memmap):
            self._tokens.flush()

    def _flush_ready_if_needed(self) -> None:
        if isinstance(self._ready_mask, np.memmap):
            self._ready_mask.flush()

    def _finalize_cache(self) -> None:
        if not self._token_cache_path or not self._token_tmp_path:
            return

        tokens = self._ensure_token_buffer()
        ready = self._ensure_ready_buffer()
        if isinstance(tokens, np.memmap):
            tokens.flush()
        if isinstance(ready, np.memmap):
            ready.flush()

        os.replace(self._token_tmp_path, self._token_cache_path)
        self._token_tmp_path = None
        if self._ready_tmp_path and self._ready_cache_path:
            os.replace(self._ready_tmp_path, self._ready_cache_path)
            self._ready_tmp_path = None

        self._tokens = np.memmap(
            self._token_cache_path,
            dtype=np.int64,
            mode="r+",
            shape=(self._total_texts, self.max_length),
        )
        if self._ready_cache_path:
            self._ready_mask = np.memmap(
                self._ready_cache_path,
                dtype=np.uint8,
                mode="r+",
                shape=(self._total_texts,),
            )
        else:
            self._ready_mask = np.ones(self._total_texts, dtype=np.uint8)
        self._mark_all_ready()
        self._background_in_flight = False

    def _check_background_status(self) -> None:
        thread = self._background_thread
        if thread is not None and not thread.is_alive():
            self._background_in_flight = False
        if self._background_error is not None:
            raise RuntimeError("后台预编码失败") from self._background_error

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        # ``np.memmap`` and threads are not picklable; reopen lazily in workers
        state["_tokens"] = None
        state["_ready_mask"] = None
        state["_background_thread"] = None
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:  # pragma: no cover - pickling path
        self.__dict__.update(state)
        if self.pretokenize:
            try:
                self._tokens = self._ensure_token_buffer()
            except RuntimeError:
                self._tokens = None
            if self._ready_cache_path or self._ready_tmp_path:
                self._ready_mask = self._ensure_ready_buffer()

    def _update_ready_high_water(self, idx: int) -> None:
        with self._progress_lock:
            if idx > self._ready_high_water:
                self._ready_high_water = idx

    def _update_progress_metrics(self, processed: int, start_time: float) -> None:
        if processed <= 0:
            return
        elapsed = time.time() - start_time
        if elapsed <= 0:
            return
        speed = processed / elapsed
        with self._progress_lock:
            self._last_progress_speed = speed

    def _mark_all_ready(self) -> None:
        if self._total_texts <= 0:
            return
        with self._progress_lock:
            self._ready_high_water = self._total_texts - 1
            self._last_progress_speed = None

    def _should_wait_for_index(self, idx: int) -> bool:
        if self._ready_wait_timeout <= 0:
            return False
        if not self._background_active():
            return False
        with self._progress_lock:
            high_water = self._ready_high_water
            speed = self._last_progress_speed
            margin = self._ready_wait_margin
        if high_water < 0:
            return True
        if idx <= high_water:
            return True
        distance = idx - high_water
        if margin and distance > margin:
            return False
        if speed is None or speed <= 0:
            return False
        expected = distance / speed
        return expected <= self._ready_wait_timeout

