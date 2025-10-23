"""MiniMind-aligned Mixture-of-Experts layers.

This module mirrors the gating, expert routing and auxiliary loss logic
used by MiniMind so the Transformer stack can reproduce its behavior
faithfully.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _apply_gate_activation(gate: torch.Tensor, activation: str) -> torch.Tensor:
    """Map MiniMind's activation choices to the corresponding torch ops."""

    act = activation.lower()
    if act in {"silu", "swish", "swiglu"}:
        return F.silu(gate)
    if act in {"gelu", "geglu"}:
        return F.gelu(gate)
    if act in {"relu", "reglu"}:
        return F.relu(gate)
    if act in {"sigmoid", "glu"}:
        return torch.sigmoid(gate)
    if act == "tanh":
        return torch.tanh(gate)
    raise ValueError(f"Unsupported activation for gated MLP: {activation}")


class MiniMindFeedForward(nn.Module):
    """Dense feed-forward block configured like MiniMind's FFN."""

    def __init__(self, hidden_size: int, intermediate_size: int, activation: str, dropout: float) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        activated = _apply_gate_activation(gate, self.activation) * up
        return self.dropout(self.down_proj(activated))


class MiniMindMoEGate(nn.Module):
    """Token router matching MiniMind's top-k gating strategy."""

    def __init__(self, *, hidden_size: int, n_routed_experts: int, num_experts_per_tok: int, scoring_func: str,
                 aux_loss_alpha: float, seq_aux: bool, norm_topk_prob: bool) -> None:
        super().__init__()
        self.top_k = num_experts_per_tok
        self.n_routed_experts = n_routed_experts
        self.scoring_func = scoring_func
        self.alpha = aux_loss_alpha
        self.seq_aux = seq_aux
        self.norm_topk_prob = norm_topk_prob
        self.weight = nn.Parameter(torch.empty((self.n_routed_experts, hidden_size)))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, seq_len, hidden_dim = hidden_states.shape
        flat_states = hidden_states.view(-1, hidden_dim)
        logits = F.linear(flat_states, self.weight, None)

        if self.scoring_func != "softmax":
            raise NotImplementedError(
                f"Unsupported scoring function for MoE gating: {self.scoring_func}"
            )
        scores = logits.softmax(dim=-1)
        topk_weight, topk_idx = torch.topk(scores, k=self.top_k, dim=-1, sorted=False)

        if self.top_k > 1 and self.norm_topk_prob:
            denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
            topk_weight = topk_weight / denominator

        if self.training and self.alpha > 0.0:
            aux_loss = self._compute_aux_loss(
                scores, topk_idx.view(batch, -1), seq_len=seq_len
            )
        else:
            aux_loss = topk_weight.new_zeros(())

        return topk_idx, topk_weight, aux_loss

    def _compute_aux_loss(self, scores: torch.Tensor, topk_idx: torch.Tensor, *, seq_len: int) -> torch.Tensor:
        batch = topk_idx.shape[0]
        aux_topk = self.top_k
        if self.seq_aux:
            scores_for_seq = scores.view(batch, seq_len, -1)
            counts = torch.zeros(batch, self.n_routed_experts, device=scores.device)
            counts.scatter_add_(
                1,
                topk_idx,
                torch.ones(batch, seq_len * aux_topk, device=scores.device),
            )
            counts = counts.div_(seq_len * aux_topk / self.n_routed_experts)
            loss = (counts * scores_for_seq.mean(dim=1)).sum(dim=1).mean()
        else:
            mask = F.one_hot(topk_idx.view(-1), num_classes=self.n_routed_experts).float()
            load = mask.mean(0)
            importance = scores.mean(0)
            loss = (importance * load * self.n_routed_experts).sum()
        return loss * self.alpha


class MiniMindMOEFeedForward(nn.Module):
    """MoE feed-forward module matching MiniMind's routed/shared experts."""

    def __init__(self, *, hidden_size: int, intermediate_size: int, activation: str, dropout: float,
                 n_routed_experts: int, n_shared_experts: int, num_experts_per_tok: int, scoring_func: str,
                 aux_loss_alpha: float, seq_aux: bool, norm_topk_prob: bool) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts_per_tok = num_experts_per_tok
        self.experts = nn.ModuleList(
            MiniMindFeedForward(hidden_size, intermediate_size, activation, dropout)
            for _ in range(n_routed_experts)
        )
        self.gate = MiniMindMoEGate(
            hidden_size=hidden_size,
            n_routed_experts=n_routed_experts,
            num_experts_per_tok=num_experts_per_tok,
            scoring_func=scoring_func,
            aux_loss_alpha=aux_loss_alpha,
            seq_aux=seq_aux,
            norm_topk_prob=norm_topk_prob,
        )
        self.shared_experts = nn.ModuleList(
            MiniMindFeedForward(hidden_size, intermediate_size, activation, dropout)
            for _ in range(n_shared_experts)
        )
        self.aux_loss: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch, seq_len, hidden_dim = x.shape
        topk_idx, topk_weight, aux_loss = self.gate(x)
        flat_states = x.view(-1, hidden_dim)
        flat_topk_idx = topk_idx.view(-1)
        if self.training and self.experts:
            expanded_states = flat_states.repeat_interleave(self.num_experts_per_tok, dim=0)
            expert_outputs = torch.zeros_like(expanded_states)
            for expert_id, expert in enumerate(self.experts):
                mask = flat_topk_idx == expert_id
                if mask.any():
                    expert_outputs[mask] = expert(expanded_states[mask]).to(expert_outputs.dtype)
            combined = expert_outputs.view(batch * seq_len, self.num_experts_per_tok, -1)
            combined = (combined * topk_weight.unsqueeze(-1)).sum(dim=1)
        else:
            combined = self._inference_routing(flat_states, flat_topk_idx, topk_weight.view(-1, 1))
        combined = combined.view(batch, seq_len, hidden_dim)

        if self.shared_experts:
            shared = sum(expert(x) for expert in self.shared_experts)
            combined = combined + shared

        self.aux_loss = aux_loss if isinstance(aux_loss, torch.Tensor) else x.new_tensor(aux_loss)
        return combined, self.aux_loss

    @torch.no_grad()
    def _inference_routing(
        self, flat_states: torch.Tensor, flat_expert_indices: torch.Tensor, flat_expert_weights: torch.Tensor
    ) -> torch.Tensor:
        cache = torch.zeros_like(flat_states)
        if flat_expert_indices.numel() == 0 or not self.experts:
            return cache
        idxs = flat_expert_indices.argsort()
        tokens_per_expert = flat_expert_indices.bincount(minlength=len(self.experts)).cumsum(0)
        token_idxs = idxs // self.num_experts_per_tok
        for expert_id, end_idx in enumerate(tokens_per_expert):
            start_idx = 0 if expert_id == 0 else tokens_per_expert[expert_id - 1]
            if start_idx == end_idx:
                continue
            expert = self.experts[expert_id]
            exp_token_idx = token_idxs[start_idx:end_idx]
            expert_tokens = flat_states[exp_token_idx]
            expert_out = expert(expert_tokens).to(cache.dtype)
            expert_out.mul_(flat_expert_weights[idxs[start_idx:end_idx]])
            cache.scatter_add_(0, exp_token_idx.view(-1, 1).repeat(1, flat_states.shape[-1]), expert_out)
        return cache


def build_moelayer_from_config(config) -> MiniMindMOEFeedForward:
    """Factory helper that instantiates a MiniMind-compatible MoE layer."""

    return MiniMindMOEFeedForward(
        hidden_size=config.hidden_size,
        intermediate_size=config.intermediate_size,
        activation=getattr(config, "hidden_act", "silu"),
        dropout=config.dropout,
        n_routed_experts=getattr(config, "n_routed_experts", 0),
        n_shared_experts=getattr(config, "n_shared_experts", 0),
        num_experts_per_tok=getattr(config, "num_experts_per_tok", 1),
        scoring_func=getattr(config, "scoring_func", "softmax"),
        aux_loss_alpha=getattr(config, "aux_loss_alpha", 0.0),
        seq_aux=getattr(config, "seq_aux", True),
        norm_topk_prob=getattr(config, "norm_topk_prob", True),
    )
