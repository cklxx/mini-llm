"""Tests for LanguageModelingDataset pretokenization parallelism."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

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
    )

    serial_dataset = LanguageModelingDataset(
        texts=texts,
        tokenizer=tokenizer_serial,
        max_length=32,
        pretokenize=True,
        pretokenize_workers=1,
    )

    assert len(parallel_dataset) == len(serial_dataset)

    for idx in range(len(parallel_dataset)):
        parallel_item = parallel_dataset[idx].tolist()
        serial_item = serial_dataset[idx].tolist()
        assert parallel_item == serial_item
