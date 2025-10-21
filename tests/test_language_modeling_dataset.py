"""Unit tests for the simplified LanguageModelingDataset."""

from __future__ import annotations

import sys
from pathlib import Path

try:  # pragma: no cover - dependency guard for minimal environments
    import torch
except ModuleNotFoundError:  # pragma: no cover - skip when torch absent
    import pytest

    pytest.skip("torch is required for dataset tests", allow_module_level=True)

ROOT_DIR = Path(__file__).resolve().parents[1]

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.tokenizer.bpe_tokenizer import BPETokenizer
from src.training.datasets.language_modeling import LanguageModelingDataset


def _build_tokenizer() -> BPETokenizer:
    tokenizer = BPETokenizer(vocab_size=256)
    tokenizer.train([
        "MiniMind 数据处理",
        "让预训练准备更轻量",
        "通过简化数据集加载来提升速度",
    ])
    return tokenizer


def test_dataset_pads_and_truncates() -> None:
    tokenizer = _build_tokenizer()
    texts = ["欢迎体验 MiniMind 风格的数据集"]

    dataset = LanguageModelingDataset(texts=texts, tokenizer=tokenizer, max_length=16)

    sample = dataset[0]
    assert set(sample.keys()) == {"input_ids", "labels", "attention_mask"}

    input_ids = sample["input_ids"]
    labels = sample["labels"]
    attention_mask = sample["attention_mask"]

    assert input_ids.shape == (16,)
    assert torch.equal(input_ids, labels)
    assert attention_mask.dtype == torch.long
    assert attention_mask.sum() == (input_ids != tokenizer.pad_id).sum()


def test_dataset_handles_empty_text() -> None:
    tokenizer = _build_tokenizer()
    dataset = LanguageModelingDataset(texts=[""], tokenizer=tokenizer, max_length=8)

    sample = dataset[0]
    input_ids = sample["input_ids"]

    # Should at least contain BOS/EOS tokens followed by padding.
    assert input_ids[0].item() == tokenizer.bos_id
    assert input_ids[1].item() == tokenizer.eos_id
    assert input_ids.shape[0] == 8


def test_dataset_truncation_resets_eos() -> None:
    tokenizer = _build_tokenizer()
    long_text = "这是一个很长的文本，用于测试截断后末尾的EOS是否保留。" * 10

    dataset = LanguageModelingDataset(texts=[long_text], tokenizer=tokenizer, max_length=12)
    sample = dataset[0]

    input_ids = sample["input_ids"]
    assert input_ids[-1].item() == tokenizer.eos_id
