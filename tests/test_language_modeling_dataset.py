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

    inputs, targets, loss_mask = dataset[0]
    expected_length = 16 - 1

    assert inputs.shape == (expected_length,)
    assert targets.shape == (expected_length,)
    assert loss_mask.shape == (expected_length,)
    assert inputs.dtype == torch.long
    assert targets.dtype == torch.long
    assert loss_mask.dtype == torch.long

    # Targets should be the next token of inputs.
    assert torch.equal(inputs[1:], targets[:-1])

    non_pad_targets = (targets != tokenizer.pad_id).long()
    assert torch.equal(loss_mask, non_pad_targets)


def test_dataset_handles_empty_text() -> None:
    tokenizer = _build_tokenizer()
    dataset = LanguageModelingDataset(texts=[""], tokenizer=tokenizer, max_length=8)

    inputs, targets, loss_mask = dataset[0]
    assert inputs[0].item() == tokenizer.bos_id
    assert targets[0].item() == tokenizer.eos_id
    assert loss_mask[0].item() == 1
    assert inputs.shape[0] == 7


def test_dataset_truncation_resets_eos() -> None:
    tokenizer = _build_tokenizer()
    long_text = "这是一个很长的文本，用于测试截断后末尾的EOS是否保留。" * 10

    dataset = LanguageModelingDataset(texts=[long_text], tokenizer=tokenizer, max_length=12)
    _, targets, _ = dataset[0]

    assert targets[-1].item() == tokenizer.eos_id
