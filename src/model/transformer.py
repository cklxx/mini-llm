"""
手写实现Transformer模型架构
包含所有核心组件：注意力机制、前馈网络、位置编码等
用于新手理解Transformer原理
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from .config import MiniGPTConfig
from .moe import SharedExpertMoE, SparseMoE
from .rope import apply_rotary_pos_emb


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization

    RMSNorm 比传统的 LayerNorm 更简单高效，去除了均值中心化操作，
    只保留方差归一化，计算公式为：

    y = x / sqrt(mean(x^2) + eps) * g

    其中 g 是可学习的缩放参数
    """

    def __init__(self, hidden_size: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


def _apply_gate_activation(gate: torch.Tensor, activation: str) -> torch.Tensor:
    """Apply the configured activation to the GLU gate tensor."""

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


class MultiHeadAttention(nn.Module):
    """多头注意力机制（兼容性保留）

    注意力机制的核心公式：
    Attention(Q, K, V) = softmax(QK^T / √d_k)V

    多头注意力将输入投影到多个子空间，分别计算注意力，然后拼接
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model必须能被n_heads整除"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads  # 每个头的维度

        # 线性投影层：将输入投影为Q, K, V
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)

        # 输出投影层
        self.w_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        前向传播

        Args:
            query: (batch_size, seq_len, d_model)
            key: (batch_size, seq_len, d_model)
            value: (batch_size, seq_len, d_model)
            mask: (batch_size, seq_len, seq_len) 或 None

        Returns:
            output: (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, d_model = query.size()

        # 1. 线性投影得到Q, K, V
        Q = self.w_q(query)  # (batch_size, seq_len, d_model)
        K = self.w_k(key)  # (batch_size, seq_len, d_model)
        V = self.w_v(value)  # (batch_size, seq_len, d_model)

        # 2. 重塑为多头格式
        Q = Q.reshape(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = K.reshape(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = V.reshape(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        # 现在形状为: (batch_size, n_heads, seq_len, d_k)

        # 3. 计算注意力
        dropout_p = self.dropout.p if self.training else 0.0
        attention_output = F.scaled_dot_product_attention(
            Q,
            K,
            V,
            attn_mask=mask,
            dropout_p=dropout_p,
        )
        # 形状: (batch_size, n_heads, seq_len, d_k)

        # 4. 拼接多头结果
        attention_output = attention_output.transpose(1, 2).contiguous()
        attention_output = attention_output.reshape(batch_size, seq_len, d_model)

        # 5. 输出投影
        output = self.w_o(attention_output)

        return self.dropout(output)


class SwiGLUFeedForward(nn.Module):
    """
    这是一个实现了 SwiGLU 的标准前馈网络层。
    """

    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        # 通常 hidden_dim 是 dim 的倍数，例如 4 * dim
        # SwiGLU 论文建议使用 2/3 的倍数，如 8/3 * dim

        self.w_gate = nn.Linear(dim, hidden_dim, bias=False)  # 对应公式中的 W
        self.w_up = nn.Linear(dim, hidden_dim, bias=False)  # 对应公式中的 V
        self.w_down = nn.Linear(hidden_dim, dim, bias=False)  # 最后输出的线性层
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. 计算门和数据通路
        gate = self.w_gate(x)  # xW
        up = self.w_up(x)  # xV

        # 2. 应用 Swish 激活函数到门上，并执行逐元素相乘
        # F.silu 是 PyTorch 中 Swish (或 SiLU) 函数的官方实现
        gated_output = F.silu(gate) * up  # Swish(xW) ⊙ (xV)

        # 3. 通过最后一个线性层，将维度映射回原始维度
        output = self.w_down(self.dropout(gated_output))

        return output


class GatedFeedForward(nn.Module):
    """Feed-forward module using gated activations with dropout support."""

    def __init__(self, dim: int, hidden_dim: int, activation: str, dropout: float):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        gated = _apply_gate_activation(gate, self.activation) * up
        return self.down_proj(self.dropout(gated))


class GroupedQueryAttention(nn.Module):
    """Self-attention module with shared KV heads and RoPE support."""

    def __init__(
        self,
        config: MiniGPTConfig,
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.use_gqa = getattr(config, "use_gqa", True)
        requested_kv_heads = getattr(config, "num_key_value_heads", None)
        if self.use_gqa:
            self.num_key_value_heads = MiniGPTConfig._normalize_num_key_value_heads(
                num_attention_heads=self.num_heads,
                requested_kv_heads=requested_kv_heads,
                use_gqa=True,
            ) or self.num_heads
        else:
            self.num_key_value_heads = self.num_heads
        assert (
            self.hidden_size % self.num_heads == 0
        ), "hidden_size必须能被num_attention_heads整除"
        assert (
            self.num_heads % self.num_key_value_heads == 0
        ), "num_attention_heads必须能被num_key_value_heads整除"

        self.head_dim = self.hidden_size // self.num_heads
        self.num_queries_per_kv = self.num_heads // self.num_key_value_heads

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False
        )
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        self.dropout = nn.Dropout(config.attention_dropout)
        self.use_rope = getattr(config, "use_rope", True)
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = getattr(config, "rope_theta", 10000.0)
        self.flash_attn = getattr(config, "flash_attn", False) and hasattr(
            F, "scaled_dot_product_attention"
        )
        self._rope = None

    def _maybe_init_rope(self):
        if self._rope is None:
            from .rope import RotaryPositionEmbedding

            self._rope = RotaryPositionEmbedding(
                self.head_dim, self.max_position_embeddings, self.rope_theta
            )

    def _repeat_kv(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch, num_key_value_heads, slen, head_dim = hidden_states.shape
        if self.num_queries_per_kv == 1:
            return hidden_states
        hidden_states = hidden_states[:, :, None, :, :].expand(
            batch, num_key_value_heads, self.num_queries_per_kv, slen, head_dim
        )
        return hidden_states.reshape(batch, self.num_heads, slen, head_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        cos: torch.Tensor | None = None,
        sin: torch.Tensor | None = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        batch_size, seq_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(
            batch_size, seq_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        value_states = value_states.view(
            batch_size, seq_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)

        if self.use_rope:
            if cos is None or sin is None:
                self._maybe_init_rope()
                cos, sin = self._rope(hidden_states, seq_len)
            query_states, key_states = apply_rotary_pos_emb(
                query_states, key_states, cos, sin, position_ids
            )

        if past_key_value is not None:
            past_key, past_value = past_key_value
            key_states = torch.cat([past_key, key_states], dim=2)
            value_states = torch.cat([past_value, value_states], dim=2)

        present = (key_states, value_states) if use_cache else None

        key_states = self._repeat_kv(key_states)
        value_states = self._repeat_kv(value_states)

        if attention_mask is not None and attention_mask.dtype == torch.bool:
            attention_mask = attention_mask.masked_fill(attention_mask, float("-inf")).to(
                dtype=query_states.dtype
            )
        elif attention_mask is not None and attention_mask.dtype != query_states.dtype:
            attention_mask = attention_mask.to(dtype=query_states.dtype)

        dropout_p = self.dropout.p if self.training else 0.0
        attn_output = F.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=attention_mask,
            dropout_p=dropout_p,
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)
        attn_output = self.dropout(attn_output)

        return attn_output, present

class PositionalEncoding(nn.Module):
    """传统位置编码（兼容性保留）

    为序列中的每个位置添加位置信息
    使用正弦和余弦函数生成位置编码

    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    注意：新模型建议使用RoPE位置编码以获得更好的长序列外推能力
    """

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()

        # 创建位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        # 计算除数项
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        # 计算正弦和余弦
        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数位置
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数位置

        # 添加批次维度并注册为buffer
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, d_model)
        Returns:
            位置编码后的张量
        """
        seq_len = x.size(1)
        return x + self.pe[:seq_len, :].transpose(0, 1)


class TransformerBlock(nn.Module):
    """优化的Transformer块

    包含优化的注意力机制和前馈网络，支持：
    - RoPE位置编码
    - Grouped-Query Attention
    - SwiGLU激活函数
    - RMSNorm归一化
    """

    def __init__(self, config: MiniGPTConfig):
        super().__init__()

        self.config = config

        # 选择注意力机制类型（分组查询注意力实现）
        self.attention = GroupedQueryAttention(config)

        self.has_moe = getattr(config, "use_moe", False)
        self.aux_loss_weight = getattr(config, "aux_loss_alpha", 0.0)

        if self.has_moe:
            total_experts = max(getattr(config, "n_routed_experts", 0), 0)
            shared_experts = max(getattr(config, "n_shared_experts", 0), 0)
            shared_experts = (
                min(shared_experts, total_experts) if total_experts > 0 else shared_experts
            )
            top_k = max(1, getattr(config, "num_experts_per_tok", 1))

            if total_experts <= 0 and shared_experts <= 0:
                # 没有有效的专家，回退为稠密前馈
                self.has_moe = False
            elif shared_experts > 0:
                self.feed_forward = SharedExpertMoE(
                    d_model=config.hidden_size,
                    d_ff=config.intermediate_size,
                    num_shared_experts=shared_experts,
                    num_routed_experts=max(total_experts - shared_experts, 0),
                    top_k=top_k,
                    activation=config.hidden_act,
                    dropout=config.dropout,
                    load_balancing_weight=self.aux_loss_weight,
                )
            else:
                self.feed_forward = SparseMoE(
                    d_model=config.hidden_size,
                    d_ff=config.intermediate_size,
                    num_experts=total_experts,
                    top_k=top_k,
                    activation=config.hidden_act,
                    dropout=config.dropout,
                    load_balancing_weight=self.aux_loss_weight,
                )

        if not getattr(self, "feed_forward", None):
            self.feed_forward = GatedFeedForward(
                config.hidden_size,
                config.intermediate_size,
                getattr(config, "hidden_act", "silu"),
                config.dropout,
            )
            self.has_moe = False

        # 层归一化
        self.norm1 = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.norm2 = RMSNorm(config.hidden_size, config.rms_norm_eps)

        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        cos: torch.Tensor | None = None,
        sin: torch.Tensor | None = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        collect_moe_loss: bool = False,
    ):
        """
        Args:
            x: (batch_size, seq_len, hidden_size)
            mask: 注意力掩码
            position_ids: 位置ID
            collect_moe_loss: 若为True，则返回MoE负载均衡损失
        """
        # Pre-norm架构：先归一化再计算
        normalized_x = self.norm1(x)

        # 多头注意力
        attn_output, present = self.attention(
            hidden_states=normalized_x,
            attention_mask=attention_mask,
            position_ids=position_ids,
            cos=cos,
            sin=sin,
            past_key_value=past_key_value,
            use_cache=use_cache,
        )

        # 残差连接
        x = x + self.dropout(attn_output)

        # 前馈网络
        normalized_x = self.norm2(x)
        moe_loss = None
        if self.has_moe:
            ff_output, moe_loss = self.feed_forward(normalized_x)
            if moe_loss is not None and self.aux_loss_weight:
                moe_loss = moe_loss * self.aux_loss_weight
        else:
            ff_output = self.feed_forward(normalized_x)
            moe_loss = None
        x = x + self.dropout(ff_output)

        if collect_moe_loss and use_cache:
            return x, present, moe_loss
        if collect_moe_loss:
            return x, moe_loss
        if use_cache:
            return x, present

        return x


class MiniGPT(nn.Module):
    """小型GPT模型

    基于Transformer Decoder的自回归语言模型
    """

    def __init__(self, config: MiniGPTConfig):
        super().__init__()

        self.config = config
        self.vocab_size = config.vocab_size
        self.d_model = config.hidden_size
        self.max_len = config.max_position_embeddings

        # 词嵌入层
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)

        # 位置编码（根据配置选择）
        self.use_rope = getattr(config, "use_rope", True)
        if self.use_rope:
            from .rope import RotaryPositionEmbedding

            self.rotary_emb = RotaryPositionEmbedding(
                config.head_dim,
                config.max_position_embeddings,
                getattr(config, "rope_theta", 10000.0),
            )
            self.register_buffer(
                "rotary_cos_cached",
                self.rotary_emb.cos_cached.clone(),
                persistent=False,
            )
            self.register_buffer(
                "rotary_sin_cached",
                self.rotary_emb.sin_cached.clone(),
                persistent=False,
            )
        else:
            self.rotary_emb = None
            self.positional_encoding = PositionalEncoding(
                config.hidden_size, config.max_position_embeddings
            )

        # Transformer层
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.num_hidden_layers)]
        )

        # 输出层
        self.layer_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)

        # 嵌入权重共享（可选）
        if getattr(config, "tie_word_embeddings", False):
            self.lm_head = None  # 将在forward中使用token_embedding的权重
        else:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.dropout = nn.Dropout(config.dropout)

        # 初始化参数
        self.init_weights()

    def init_weights(self):
        """初始化模型参数"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.LayerNorm | RMSNorm):
                nn.init.ones_(module.weight)
                if hasattr(module, "bias") and module.bias is not None:
                    nn.init.zeros_(module.bias)

    def create_causal_mask(self, seq_len: int) -> torch.Tensor:
        """创建用于 ``scaled_dot_product_attention`` 的因果掩码。

        PyTorch 期望布尔掩码为 *True* 表示禁止访问，或浮点掩码为
        ``-inf`` 表示禁止访问、``0`` 表示允许访问。因此我们显式构造一个
        仅屏蔽未来位置的矩阵，确保训练时与自回归推理保持一致。
        """

        return self._make_causal_mask((1, seq_len), device="cpu") [0]

    def _make_causal_mask(
        self,
        input_shape: tuple[int, int],
        device: torch.device | str,
        past_key_values_length: int = 0,
    ) -> torch.Tensor:
        batch_size, target_length = input_shape
        mask = torch.full(
            (target_length, target_length + past_key_values_length),
            fill_value=False,
            device=device,
            dtype=torch.bool,
        )
        mask[:, past_key_values_length:] = torch.triu(
            torch.ones((target_length, target_length), dtype=torch.bool, device=device),
            diagonal=1,
        )
        return mask.unsqueeze(0).unsqueeze(0).expand(
            batch_size, 1, target_length, target_length + past_key_values_length
        )

    def _expand_attention_mask(
        self, attention_mask: torch.Tensor | None, target_length: int
    ) -> torch.Tensor | None:
        if attention_mask is None:
            return None
        if attention_mask.dim() == 2:
            return (~attention_mask.bool()).unsqueeze(1).unsqueeze(1)
        if attention_mask.dim() == 4:
            return attention_mask.bool()
        raise ValueError("Unsupported attention mask rank")

    def _create_position_ids(
        self,
        attention_mask: torch.Tensor | None,
        batch_size: int,
        seq_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        if attention_mask is None:
            return torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        cumulative = attention_mask.long().cumsum(dim=-1) - 1
        cumulative = cumulative.clamp_min(0)
        return cumulative * attention_mask.long()

    def _get_rope_cos_sin(
        self, seq_len: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if not self.use_rope or self.rotary_emb is None:
            return None, None
        if (
            seq_len > getattr(self.rotary_emb, "max_seq_len_cached", 0)
            or self.rotary_emb.cos_cached.device != device
        ):
            self.rotary_emb._set_cos_sin_cache(seq_len, device)
        cos = self.rotary_emb.cos_cached[:seq_len].to(dtype=dtype)
        sin = self.rotary_emb.sin_cached[:seq_len].to(dtype=dtype)
        self.rotary_cos_cached = cos
        self.rotary_sin_cached = sin
        return cos, sin

    def _strip_padding(self, input_ids: torch.Tensor) -> torch.Tensor:
        pad_id = getattr(self.config, "pad_token_id", None)
        if pad_id is None or input_ids.dim() != 2:
            return input_ids
        sequences: list[torch.Tensor] = []
        for row in input_ids:
            non_pad = torch.nonzero(row.ne(pad_id), as_tuple=False).flatten()
            if non_pad.numel() == 0:
                fill_id = (
                    self.config.bos_token_id
                    if self.config.bos_token_id is not None
                    else pad_id
                )
                sequences.append(row.new_tensor([fill_id]))
            else:
                sequences.append(row[: non_pad[-1].item() + 1])
        return pad_sequence(sequences, batch_first=True, padding_value=pad_id)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        return_aux_loss: bool = False,
    ):
        """
        前向传播

        Args:
            input_ids: (batch_size, seq_len) token ID序列
            attention_mask: (batch_size, seq_len) 可选的注意力掩码
            position_ids: (batch_size, seq_len) 位置ID，用于RoPE
            return_aux_loss: 启用MoE时是否返回负载均衡损失

        Returns:
            logits: (batch_size, seq_len, vocab_size) 预测概率
            aux_loss: 若 `return_aux_loss=True` 且启用MoE，则返回负载均衡损失
        """
        batch_size, seq_len = input_ids.size()

        if attention_mask is None:
            if self.config.pad_token_id is not None:
                attention_mask = input_ids.ne(self.config.pad_token_id)
            else:
                attention_mask = torch.ones(
                    (batch_size, seq_len), device=input_ids.device, dtype=torch.bool
                )
        else:
            attention_mask = attention_mask.to(device=input_ids.device, dtype=torch.bool)

        # 1. 词嵌入
        x = self.token_embedding(input_ids)

        # 2. 位置编码（仅在不使用RoPE时）
        if not self.use_rope and hasattr(self, "positional_encoding"):
            x = self.positional_encoding(x)

        x = self.dropout(x)

        # 3. 生成位置ID和注意力掩码
        if position_ids is None:
            position_ids = self._create_position_ids(
                attention_mask, batch_size, seq_len, input_ids.device
            )

        causal_mask = self._make_causal_mask((batch_size, seq_len), input_ids.device)
        key_padding_mask = self._expand_attention_mask(attention_mask, seq_len)
        if key_padding_mask is not None:
            attention_bias = causal_mask | key_padding_mask
        else:
            attention_bias = causal_mask

        cos, sin = self._get_rope_cos_sin(seq_len, x.device, x.dtype)

        # 4. 通过Transformer块
        total_moe_loss = None
        for transformer_block in self.transformer_blocks:
            if return_aux_loss and getattr(self.config, "use_moe", False):
                outputs = transformer_block(
                    x,
                    attention_mask=attention_bias,
                    position_ids=position_ids,
                    cos=cos,
                    sin=sin,
                    collect_moe_loss=True,
                )
                x, block_moe_loss = outputs
                if block_moe_loss is not None:
                    total_moe_loss = (
                        block_moe_loss
                        if total_moe_loss is None
                        else total_moe_loss + block_moe_loss
                    )
            else:
                x = transformer_block(
                    x,
                    attention_mask=attention_bias,
                    position_ids=position_ids,
                    cos=cos,
                    sin=sin,
                )

        # 5. 层归一化
        x = self.layer_norm(x)
        if attention_mask is not None:
            x = x * attention_mask.unsqueeze(-1).to(dtype=x.dtype)

        # 7. 输出投影（支持权重共享）
        if self.lm_head is not None:
            logits = self.lm_head(x)
        else:
            # 使用嵌入权重共享
            logits = F.linear(x, self.token_embedding.weight)

        if return_aux_loss and getattr(self.config, "use_moe", False):
            if total_moe_loss is None:
                total_moe_loss = torch.tensor(0.0, device=logits.device)
            return logits, total_moe_loss

        return logits

    def generate(
        self,
        input_ids: torch.Tensor,
        max_length: int = None,
        temperature: float = None,
        top_k: int = None,
    ) -> torch.Tensor:
        """文本生成

        Args:
            input_ids: (batch_size, seq_len) 输入token序列
            max_length: 最大生成长度，默认使用config中的值
            temperature: 采样温度，越高越随机，默认使用config中的值
            top_k: top-k采样，只从概率最高的k个token中采样，默认使用config中的值

        Returns:
            生成的token序列
        """
        # 使用配置中的默认值
        max_length = max_length or self.config.max_generate_length
        temperature = temperature or self.config.temperature
        top_k = top_k or self.config.top_k
        self.eval()

        with torch.no_grad():
            input_ids = self._strip_padding(input_ids)
            for _ in range(max_length):
                # 前向传播
                logits = self.forward(input_ids)

                # 取最后一个位置的logits
                next_token_logits = logits[:, -1, :] / temperature

                # Top-k采样
                if top_k > 0:
                    top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)
                    next_token_logits = torch.full_like(next_token_logits, -1e9)
                    next_token_logits.scatter_(1, top_k_indices, top_k_logits)

                # 采样下一个token
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

                # 拼接到输入序列
                input_ids = torch.cat([input_ids, next_token], dim=1)

                # 如果生成了结束符，停止生成
                if next_token.item() == self.config.eos_token_id:
                    break

        return input_ids

    def get_num_params(self) -> int:
        """获取模型参数数量"""
        return sum(p.numel() for p in self.parameters())


def create_model(
    vocab_size: int = None, model_size: str = "small", config: MiniGPTConfig = None
) -> MiniGPT:
    """创建不同大小的模型"""

    if config is not None:
        # 如果提供了配置，直接使用
        model_config = config
        if vocab_size is not None:
            model_config.vocab_size = vocab_size
    else:
        # 使用预定义配置
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
    # 测试模型
    vocab_size = 10000
    model = create_model(vocab_size, "small")

    # 创建测试输入
    batch_size = 2
    seq_len = 20
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))

    # 前向传播
    with torch.no_grad():
        logits = model(input_ids)
        print(f"输入形状: {input_ids.shape}")
        print(f"输出形状: {logits.shape}")

        # 测试生成
        generated = model.generate(input_ids[:1], max_length=10)
        print(f"生成序列长度: {generated.shape[1]}")
