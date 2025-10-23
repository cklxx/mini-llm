#!/usr/bin/env python3
"""Runtime smoke tests for the MiniGPT transformer."""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

try:  # pragma: no cover - optional dependency guard
    import pytest
except ModuleNotFoundError:  # pragma: no cover
    pytest = None

try:  # pragma: no cover - optional dependency guard
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None

from src.model.config import MiniGPTConfig
from src.model.transformer import MiniGPT

if pytest is not None:
    pytestmark = pytest.mark.skipif(torch is None, reason="PyTorch not available")
else:  # pragma: no cover
    pytestmark = []


def _build_test_model(*, use_rope: bool) -> MiniGPT:
    """Construct a tiny MiniGPT instance that runs quickly on CPU."""

    config = MiniGPTConfig(
        vocab_size=128,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=128,
        max_position_embeddings=64,
        use_rope=use_rope,
        dropout=0.0,
        attention_dropout=0.0,
        gradient_checkpointing=False,
        tie_word_embeddings=False,
    )
    return MiniGPT(config)


def _run_forward(model: MiniGPT) -> torch.Tensor:
    batch_size, seq_len = 2, 16
    input_ids = torch.randint(0, model.config.vocab_size, (batch_size, seq_len))
    attention_mask = torch.ones_like(input_ids)
    logits = model(input_ids, attention_mask=attention_mask)
    assert logits.shape == (batch_size, seq_len, model.config.vocab_size)
    return logits


def test_forward_pass_with_rope():
    """The transformer should execute a forward pass when RoPE is enabled."""

    model = _build_test_model(use_rope=True)
    logits = _run_forward(model)
    assert torch.isfinite(logits).all()


def test_forward_pass_without_rope():
    """The transformer should also work when falling back to sinusoidal positions."""

    model = _build_test_model(use_rope=False)
    logits = _run_forward(model)
    assert torch.isfinite(logits).all()
