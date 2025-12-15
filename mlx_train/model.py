from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from .config import MiniLLMConfig


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = mx.ones((dim,), dtype=mx.float32)

    def __call__(self, x: mx.array) -> mx.array:
        x_fp32 = x.astype(mx.float32)
        norm = x_fp32 * mx.rsqrt(mx.mean(x_fp32 * x_fp32, axis=-1, keepdims=True) + self.eps)
        return (norm.astype(x.dtype) * self.weight.astype(x.dtype))


class Attention(nn.Module):
    def __init__(self, config: MiniLLMConfig):
        super().__init__()
        self.n_heads = config.num_attention_heads
        self.n_kv_heads = config.num_attention_heads if config.num_key_value_heads is None else config.num_key_value_heads
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads")
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(config.hidden_size, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, config.hidden_size, bias=False)

        self.resid_dropout = nn.Dropout(config.dropout)
        self.rope = nn.RoPE(self.head_dim, traditional=False, base=config.rope_theta)

    def __call__(self, x: mx.array, *, start_pos: int = 0, attention_mask: Optional[mx.array] = None) -> mx.array:
        bsz, seq_len, _ = x.shape
        q = self.q_proj(x).reshape(bsz, seq_len, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.k_proj(x).reshape(bsz, seq_len, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(bsz, seq_len, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)

        q = self.rope(q, offset=start_pos)
        k = self.rope(k, offset=start_pos)

        if attention_mask is None:
            mask: Optional[mx.array | str] = "causal"
        else:
            # Accept additive/boolean masks broadcastable to [B, N, T, T].
            # For strict alignment with the PyTorch training scripts, you can pass None.
            mask = attention_mask

        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask=mask)
        out = out.transpose(0, 2, 1, 3).reshape(bsz, seq_len, self.n_heads * self.head_dim)
        out = self.o_proj(out)
        return self.resid_dropout(out)


class FeedForward(nn.Module):
    def __init__(self, config: MiniLLMConfig):
        super().__init__()
        config = config.finalize()
        intermediate_size = config.intermediate_size
        if intermediate_size is None:
            raise ValueError("intermediate_size is None after finalize()")
        self.gate_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, config.hidden_size, bias=False)
        self.dropout = nn.Dropout(config.dropout)

        if config.hidden_act != "silu":
            raise ValueError(f"Only silu is supported right now, got: {config.hidden_act}")

    def __call__(self, x: mx.array) -> mx.array:
        return self.dropout(self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x)))


class MiniLLMBlock(nn.Module):
    def __init__(self, layer_id: int, config: MiniLLMConfig):
        super().__init__()
        self.layer_id = layer_id
        self.self_attn = Attention(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        if config.use_moe:
            raise NotImplementedError("MoE is not implemented for the MLX path yet.")
        self.mlp = FeedForward(config)

    def __call__(self, x: mx.array, *, start_pos: int = 0, attention_mask: Optional[mx.array] = None) -> mx.array:
        h = self.self_attn(self.input_layernorm(x), start_pos=start_pos, attention_mask=attention_mask)
        x = x + h
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class MiniLLMModel(nn.Module):
    def __init__(self, config: MiniLLMConfig):
        super().__init__()
        self.config = config.finalize()
        self.embed_tokens = nn.Embedding(self.config.vocab_size, self.config.hidden_size)
        self.dropout = nn.Dropout(self.config.dropout)
        self.layers = [MiniLLMBlock(i, self.config) for i in range(self.config.num_hidden_layers)]
        self.norm = RMSNorm(self.config.hidden_size, eps=self.config.rms_norm_eps)

    def __call__(self, input_ids: mx.array, *, attention_mask: Optional[mx.array] = None) -> mx.array:
        # input_ids: [B, T]
        h = self.embed_tokens(input_ids)
        h = self.dropout(h)
        start_pos = 0
        for layer in self.layers:
            h = layer(h, start_pos=start_pos, attention_mask=attention_mask)
        return self.norm(h)


class MiniLLMForCausalLM(nn.Module):
    def __init__(self, config: Optional[MiniLLMConfig] = None):
        super().__init__()
        self.config = (config or MiniLLMConfig()).finalize()
        self.model = MiniLLMModel(self.config)

    def __call__(self, input_ids: mx.array, *, attention_mask: Optional[mx.array] = None) -> mx.array:
        h = self.model(input_ids, attention_mask=attention_mask)  # [B, T, H]
        # Weight tying: lm_head.weight == embed_tokens.weight
        logits = h @ self.model.embed_tokens.weight.transpose()  # [B, T, V]
        return logits


def count_parameters(params: Dict[str, Any]) -> int:
    def _count(obj: Any) -> int:
        if isinstance(obj, mx.array):
            return int(obj.size)
        if isinstance(obj, dict):
            return sum(_count(v) for v in obj.values())
        if isinstance(obj, (list, tuple)):
            return sum(_count(v) for v in obj)
        return 0

    return _count(params)


def parameters_bytes(params: Dict[str, Any]) -> int:
    def _bytes(obj: Any) -> int:
        if isinstance(obj, mx.array):
            return int(obj.size) * int(obj.dtype.size)
        if isinstance(obj, dict):
            return sum(_bytes(v) for v in obj.values())
        if isinstance(obj, (list, tuple)):
            return sum(_bytes(v) for v in obj)
        return 0

    return _bytes(params)
