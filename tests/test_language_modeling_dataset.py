"""Tests for LanguageModelingDataset pretokenization parallelism."""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

try:  # pragma: no cover - dependency guard for minimal environments
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover - skip when numpy absent
    import pytest

    pytest.skip("numpy is required for dataset tests", allow_module_level=True)

try:  # pragma: no cover - dependency guard for minimal environments
    import torch  # noqa: F401
except ModuleNotFoundError:  # pragma: no cover - skip when torch absent
    import pytest

    pytest.skip("torch is required for dataset tests", allow_module_level=True)

from src.tokenizer.bpe_tokenizer import BPETokenizer
from src.training.datasets.language_modeling import LanguageModelingDataset


def _build_tokenizer() -> BPETokenizer:
    tokenizer = BPETokenizer(vocab_size=256)
    corpus = [
        "你好，世界！",
        "Mini-LLM 让教学更简单。",
        "并行预编码应该与串行结果一致。",
        "PyTorch 数据加载测试",
    ]
    tokenizer.train(corpus)
    return tokenizer


def test_parallel_pretokenize_matches_serial() -> None:
    texts = [
        "欢迎使用 Mini-LLM",  # ensure typical unicode
        "并行进度需要保持稳定",
        "数据预处理验证",
        "第四条测试样本",
        "更多的测试数据",
        "第六条数据",
        "第七条数据",
        "第八条数据",
    ]

    tokenizer_parallel = _build_tokenizer()
    tokenizer_serial = _build_tokenizer()

    parallel_dataset = LanguageModelingDataset(
        texts=texts,
        tokenizer=tokenizer_parallel,
        max_length=32,
        pretokenize=True,
        pretokenize_workers=4,
        initial_pretokenize_items=len(texts),
        background_pretokenize=False,
    )

    serial_dataset = LanguageModelingDataset(
        texts=texts,
        tokenizer=tokenizer_serial,
        max_length=32,
        pretokenize=True,
        pretokenize_workers=1,
        initial_pretokenize_items=len(texts),
        background_pretokenize=False,
    )

    assert len(parallel_dataset) == len(serial_dataset)

    for idx in range(len(parallel_dataset)):
        parallel_item = parallel_dataset[idx].tolist()
        serial_item = serial_dataset[idx].tolist()
        assert parallel_item == serial_item


def test_pretokenize_uses_cache(tmp_path, monkeypatch) -> None:
    cache_dir = tmp_path / "cache"
    monkeypatch.setenv("MINILLM_CACHE_DIR", str(cache_dir))

    texts = [
        "缓存测试一",
        "缓存测试二",
        "缓存测试三",
    ]

    tokenizer = _build_tokenizer()

    dataset_first = LanguageModelingDataset(
        texts=texts,
        tokenizer=tokenizer,
        max_length=16,
        pretokenize=True,
        pretokenize_workers=2,
        initial_pretokenize_items=len(texts),
        background_pretokenize=False,
    )

    cached_tokens = dataset_first._tokens.copy()  # type: ignore[union-attr]

    def _fail(*args, **kwargs):  # pragma: no cover - should never be called
        raise AssertionError("预编码缓存未生效")

    monkeypatch.setattr(
        LanguageModelingDataset,
        "_pretokenize_incremental",
        _fail,
    )

    dataset_second = LanguageModelingDataset(
        texts=texts,
        tokenizer=tokenizer,
        max_length=16,
        pretokenize=True,
        pretokenize_workers=2,
        initial_pretokenize_items=len(texts),
        background_pretokenize=False,
    )

    np.testing.assert_array_equal(dataset_second._tokens, cached_tokens)  # type: ignore[union-attr]


def test_incremental_pretokenize_encodes_on_demand(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("MINILLM_CACHE_DIR", str(tmp_path))

    texts = [
        "预编码第一条",
        "预编码第二条",
        "需要动态编码的第三条",
        "需要动态编码的第四条",
    ]

    tokenizer = _build_tokenizer()

    dataset = LanguageModelingDataset(
        texts=texts,
        tokenizer=tokenizer,
        max_length=32,
        pretokenize=True,
        pretokenize_workers=2,
        initial_pretokenize_items=2,
        background_pretokenize=False,
    )

    assert dataset._ready_mask is not None  # type: ignore[attr-defined]
    ready_mask = dataset._ready_mask  # type: ignore[attr-defined]
    assert ready_mask[:2].sum() == 2
    assert ready_mask[2:].sum() == 0

    third = dataset[2]
    tokens = dataset._ensure_token_buffer()  # type: ignore[attr-defined]
    ready_mask = dataset._ensure_ready_buffer()  # type: ignore[attr-defined]
    assert ready_mask[2] == 1  # type: ignore[index]
    np.testing.assert_array_equal(tokens[2], third.numpy())  # type: ignore[index]


def test_waits_for_background_ready(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("MINILLM_CACHE_DIR", str(tmp_path))

    texts = [
        "等待测试一",
        "等待测试二",
        "等待测试三",
    ]

    tokenizer = _build_tokenizer()

    original_range = LanguageModelingDataset._pretokenize_range

    def _slow_background(self, start_idx, end_idx, texts, worker_count, start_time):
        if start_idx >= 1:
            time.sleep(0.2)
        return original_range(self, start_idx, end_idx, texts, worker_count, start_time)

    monkeypatch.setattr(
        LanguageModelingDataset,
        "_pretokenize_range",
        _slow_background,
    )

    dataset = LanguageModelingDataset(
        texts=texts,
        tokenizer=tokenizer,
        max_length=32,
        pretokenize=True,
        pretokenize_workers=2,
        initial_pretokenize_items=1,
        background_pretokenize=True,
    )

    # Ensure an unready index exists
    ready_mask = dataset._ensure_ready_buffer()  # type: ignore[attr-defined]
    assert ready_mask[1] == 0  # type: ignore[index]

    dataset._ready_wait_timeout = 2.0  # type: ignore[attr-defined]

    def _guard_encode(text: str, idx: int):  # pragma: no cover - should not run
        raise AssertionError("不应在等待后台预编码时回退到同步编码")

    original_encode = dataset._encode_to_tensor
    dataset._encode_to_tensor = _guard_encode  # type: ignore[assignment]

    try:
        start = time.time()
        item = dataset[1]
        elapsed = time.time() - start
        assert isinstance(item, torch.Tensor)
        assert elapsed >= 0.15
    finally:
        dataset._encode_to_tensor = original_encode  # type: ignore[assignment]
        if dataset._background_thread is not None:  # type: ignore[attr-defined]
            dataset._background_thread.join(timeout=5)  # pragma: no cover - cleanup

    # Background thread should eventually mark the item as ready
    ready_mask = dataset._ensure_ready_buffer()  # type: ignore[attr-defined]
    assert ready_mask[1] == 1  # type: ignore[index]


def test_wait_for_ready_times_out_without_background() -> None:
    tokenizer = _build_tokenizer()
    texts = ["超时测试一", "超时测试二"]

    dataset = LanguageModelingDataset(
        texts=texts,
        tokenizer=tokenizer,
        max_length=16,
        pretokenize=True,
        pretokenize_workers=1,
        initial_pretokenize_items=len(texts),
        background_pretokenize=False,
    )

    ready = dataset._ensure_ready_buffer()  # type: ignore[attr-defined]
    assert ready is not None

    dataset._background_in_flight = True  # type: ignore[attr-defined]
    dataset._ready_wait_timeout = 0.05  # type: ignore[attr-defined]

    # Clear readiness for index 0 to force timeout behaviour
    ready[0] = 0

    result = dataset._wait_for_ready(0, ready)  # type: ignore[attr-defined]
    assert result is False
