"""MiniMind-aligned Transformer implementation used by MiniGPT."""

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import MiniGPTConfig
from .moe import MiniMindFeedForward, build_moelayer_from_config


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, hidden_size: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        return self.weight * hidden_states.to(input_dtype)


def precompute_freqs_cis(dim: int, end: int, theta: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Pre-compute RoPE cos/sin caches following MiniMind."""

    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(end, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1)
    sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1)
    return cos, sin


def apply_rotary_pos_emb(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embeddings with MiniMind's layout."""

    def rotate_half(x: torch.Tensor) -> torch.Tensor:
        return torch.cat((-x[..., x.shape[-1] // 2 :], x[..., : x.shape[-1] // 2]), dim=-1)

    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Repeat key/value heads to match the number of query heads."""

    if n_rep == 1:
        return x
    bsz, seq_len, n_kv_heads, head_dim = x.shape
    return (
        x[:, :, :, None, :]
        .expand(bsz, seq_len, n_kv_heads, n_rep, head_dim)
        .reshape(bsz, seq_len, n_kv_heads * n_rep, head_dim)
    )


class MiniMindAttention(nn.Module):
    """Self-attention block mirroring MiniMind's grouped-query design."""

    def __init__(self, config: MiniGPTConfig) -> None:
        super().__init__()
        kv_heads = config.num_attention_heads if config.num_key_value_heads is None else config.num_key_value_heads
        assert config.num_attention_heads % kv_heads == 0, "num_attention_heads must be divisible by num_key_value_heads"
        self.n_local_heads = config.num_attention_heads
        self.n_local_kv_heads = kv_heads
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = config.hidden_size // config.num_attention_heads

        self.q_proj = nn.Linear(config.hidden_size, self.n_local_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.n_local_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.n_local_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_local_heads * self.head_dim, config.hidden_size, bias=False)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.flash = bool(getattr(config, "flash_attn", False)) and hasattr(
            F, "scaled_dot_product_attention"
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        past_key_value: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[tuple[torch.Tensor, torch.Tensor]]]:
        bsz, seq_len, _ = hidden_states.shape
        xq = self.q_proj(hidden_states)
        xk = self.k_proj(hidden_states)
        xv = self.v_proj(hidden_states)

        xq = xq.view(bsz, seq_len, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)

        cos, sin = position_embeddings
        cos = cos[:seq_len].to(hidden_states.device, hidden_states.dtype)
        sin = sin[:seq_len].to(hidden_states.device, hidden_states.dtype)
        xq, xk = apply_rotary_pos_emb(xq, xk, cos, sin)

        if past_key_value is not None:
            past_k, past_v = past_key_value
            xk = torch.cat([past_k, xk], dim=1)
            xv = torch.cat([past_v, xv], dim=1)

        present = (xk, xv) if use_cache else None

        xq = xq.transpose(1, 2)
        xk = repeat_kv(xk, self.n_rep).transpose(1, 2)
        xv = repeat_kv(xv, self.n_rep).transpose(1, 2)

        if self.flash and seq_len != 1:
            dropout_p = self.attn_dropout.p if self.training else 0.0
            attn_mask = None
            if attention_mask is not None:
                attn_mask = attention_mask.view(bsz, 1, 1, -1).expand(bsz, self.n_local_heads, seq_len, -1)
                attn_mask = attn_mask.bool()
            output = F.scaled_dot_product_attention(
                xq, xk, xv, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=True
            )
        else:
            scores = torch.matmul(xq, xk.transpose(-2, -1)) / math.sqrt(self.head_dim)
            causal_mask = torch.triu(
                torch.full((seq_len, xk.shape[-2]), float("-inf"), device=scores.device), diagonal=1
            )
            scores = scores + causal_mask
            if attention_mask is not None:
                extended_mask = (1.0 - attention_mask.float()).unsqueeze(1).unsqueeze(2) * -1e9
                scores = scores + extended_mask
            scores = F.softmax(scores.float(), dim=-1).type_as(xq)
            scores = self.attn_dropout(scores)
            output = torch.matmul(scores, xv)

        output = output.transpose(1, 2).reshape(bsz, seq_len, -1)
        output = self.resid_dropout(self.o_proj(output))
        return output, present


class MiniMindBlock(nn.Module):
    """Transformer block aligned with MiniMind's architecture."""

    def __init__(self, layer_id: int, config: MiniGPTConfig) -> None:
        super().__init__()
        self.layer_id = layer_id
        self.self_attn = MiniMindAttention(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        if getattr(config, "use_moe", False):
            self.mlp = build_moelayer_from_config(config)
            self.is_moe = True
        else:
            self.mlp = MiniMindFeedForward(
                config.hidden_size,
                config.intermediate_size,
                getattr(config, "hidden_act", "silu"),
                config.dropout,
            )
            self.is_moe = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        past_key_value: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[tuple[torch.Tensor, torch.Tensor]], torch.Tensor]:
        residual = hidden_states
        attn_out, present = self.self_attn(
            self.input_layernorm(hidden_states),
            position_embeddings,
            past_key_value=past_key_value,
            use_cache=use_cache,
            attention_mask=attention_mask,
        )
        hidden_states = residual + attn_out

        mlp_input = self.post_attention_layernorm(hidden_states)
        if self.is_moe:
            mlp_out, aux_loss = self.mlp(mlp_input)
        else:
            mlp_out = self.mlp(mlp_input)
            aux_loss = hidden_states.new_zeros(())
        hidden_states = hidden_states + mlp_out
        return hidden_states, present, aux_loss


class MiniMindModel(nn.Module):
    """Decoder-only Transformer stack aligned with MiniMind."""

    def __init__(self, config: MiniGPTConfig) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList([MiniMindBlock(i, config) for i in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        freqs_cos, freqs_sin = precompute_freqs_cis(
            dim=config.hidden_size // config.num_attention_heads,
            end=config.max_position_embeddings,
            theta=getattr(config, "rope_theta", 1_000_000.0),
        )
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Sequence[Optional[tuple[torch.Tensor, torch.Tensor]]]] = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, List[Optional[tuple[torch.Tensor, torch.Tensor]]], torch.Tensor]:
        batch_size, seq_length = input_ids.shape
        if attention_mask is None:
            attention_mask = torch.ones(batch_size, seq_length, device=input_ids.device, dtype=torch.bool)
        else:
            attention_mask = attention_mask.to(device=input_ids.device, dtype=torch.bool)

        if past_key_values is None:
            past_key_values = [None] * len(self.layers)
        start_pos = past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0

        hidden_states = self.dropout(self.embed_tokens(input_ids))
        cos = self.freqs_cos[start_pos : start_pos + seq_length]
        sin = self.freqs_sin[start_pos : start_pos + seq_length]

        position_embeddings = (cos, sin)
        presents: List[Optional[tuple[torch.Tensor, torch.Tensor]]] = []
        aux_losses: list[torch.Tensor] = []

        for layer, past_key_value in zip(self.layers, past_key_values):
            hidden_states, present, aux_loss = layer(
                hidden_states,
                position_embeddings,
                past_key_value=past_key_value,
                use_cache=use_cache,
                attention_mask=attention_mask,
            )
            presents.append(present)
            if aux_loss is not None:
                aux_losses.append(aux_loss.to(hidden_states.dtype))

        hidden_states = self.norm(hidden_states)
        aux_total = hidden_states.new_zeros(())
        if aux_losses:
            aux_total = torch.stack(aux_losses).sum()

        return hidden_states, presents, aux_total


class MiniGPT(nn.Module):
    """MiniGPT language model backed by a MiniMind-style Transformer."""

    def __init__(self, config: MiniGPTConfig) -> None:
        super().__init__()
        self.config = config
        self.model = MiniMindModel(config)
        self.lm_head: Optional[nn.Linear]
        if getattr(config, "tie_word_embeddings", True):
            self.lm_head = None
        else:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Sequence[Optional[tuple[torch.Tensor, torch.Tensor]]]] = None,
        use_cache: bool = False,
        return_aux_loss: bool = False,
    ):
        hidden_states, presents, aux_loss = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )

        if self.lm_head is None:
            logits = F.linear(hidden_states, self.model.embed_tokens.weight)
        else:
            logits = self.lm_head(hidden_states)

        outputs: list = [logits]
        if use_cache:
            outputs.append(presents)
        if return_aux_loss and getattr(self.config, "use_moe", False):
            outputs.append(aux_loss)

        if len(outputs) == 1:
            return outputs[0]
        if len(outputs) == 2:
            return tuple(outputs)
        return tuple(outputs)

    def generate(
        self,
        input_ids: torch.Tensor,
        max_length: Optional[int] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
    ) -> torch.Tensor:
        max_length = max_length or self.config.max_generate_length
        temperature = temperature or self.config.temperature
        top_k = top_k or self.config.top_k

        self.eval()
        generated = input_ids
        past_key_values = None

        with torch.no_grad():
            for _ in range(max_length):
                model_inputs = generated if past_key_values is None else generated[:, -1:]
                outputs = self.forward(
                    model_inputs,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
                logits, past_key_values = outputs[:2]
                next_token_logits = logits[:, -1, :] / temperature
                if top_k and top_k > 0:
                    topk_values, topk_indices = torch.topk(next_token_logits, top_k)
                    filtered = next_token_logits.new_full(next_token_logits.shape, float("-inf"))
                    filtered.scatter_(1, topk_indices, topk_values)
                    next_token_logits = filtered
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                generated = torch.cat([generated, next_token], dim=1)
                if next_token.item() == self.config.eos_token_id:
                    break

        return generated

    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def create_model(
    vocab_size: int | None = None, model_size: str = "small", config: MiniGPTConfig | None = None
) -> MiniGPT:
    """Factory helper retained for backwards compatibility."""

    if config is not None:
        model_config = config
        if vocab_size is not None:
            model_config.vocab_size = vocab_size
    else:
        from .config import get_config

        model_config = get_config(model_size)
        if vocab_size is not None:
            model_config.vocab_size = vocab_size

    model = MiniGPT(model_config)
    print(f"创建 {model_size} 模型，参数量: {model.get_num_params():,}")
    print(
        f"配置详情: hidden_size={model_config.hidden_size}, layers={model_config.num_hidden_layers}, heads={model_config.num_attention_heads}"
    )
    return model


if __name__ == "__main__":
    vocab_size = 10000
    model = create_model(vocab_size, "small")
    batch_size = 2
    seq_len = 20
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    with torch.no_grad():
        logits = model(input_ids)
        print(f"输入形状: {input_ids.shape}")
        print(f"输出形状: {logits.shape}")
        generated = model.generate(input_ids[:1], max_length=10)
        print(f"生成序列长度: {generated.shape[1]}")
