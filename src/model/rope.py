"""
RoPE (Rotary Position Embedding) Implementation
Rotary Position Embedding是2024年主流LLM架构的标准选择
相比传统位置编码具有更好的长序列外推能力
"""

import math

import torch
import torch.nn as nn


class RotaryPositionEmbedding(nn.Module):
    """RoPE (Rotary Position Embedding) 实现

    RoPE通过旋转变换将位置信息编码到查询和键向量中
    具有更好的长序列外推性能，被LLaMA、Qwen2等模型采用

    核心思想：
    - 将每个注意力头的维度分为d/2对
    - 对每对维度应用旋转矩阵
    - 旋转角度随位置呈指数衰减
    """

    def __init__(
        self, dim: int, max_position_embeddings: int = 2048, base: float = 10000.0, device=None
    ):
        """
        Args:
            dim: 注意力头的维度 (head_dim)
            max_position_embeddings: 最大位置长度
            base: 频率基数，默认10000
            device: 设备
        """
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base

        # 计算频率向量
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # 预计算cos和sin值以提高效率
        self._set_cos_sin_cache(max_position_embeddings, device)

    def _set_cos_sin_cache(self, seq_len: int, device):
        """预计算并缓存cos和sin值"""
        self.max_seq_len_cached = seq_len

        # 生成位置序列
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)

        # 计算位置与频率的外积
        freqs = torch.outer(t, self.inv_freq)

        # 拼接freqs以匹配完整的head_dim
        emb = torch.cat((freqs, freqs), dim=-1)

        # 计算cos和sin
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, x: torch.Tensor, seq_len: int = None) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: 输入张量，形状为 (..., seq_len, dim)
            seq_len: 序列长度，如果为None则从x推断

        Returns:
            cos, sin: 旋转矩阵的cos和sin部分
        """
        if seq_len is None:
            seq_len = x.shape[-2]

        # 如果序列长度超过缓存，重新计算
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len, x.device)

        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """旋转输入张量的一半维度

    将 [x1, x2, x3, x4, ...] 转换为 [-x2, x1, -x4, x3, ...]
    这是RoPE旋转变换的核心操作
    """
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.Tensor = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """应用RoPE位置编码到查询和键张量

    Args:
        q: 查询张量，形状 (batch_size, num_heads, seq_len, head_dim)
        k: 键张量，形状 (batch_size, num_heads, seq_len, head_dim)
        cos: cosine位置编码，形状 (seq_len, head_dim)
        sin: sine位置编码，形状 (seq_len, head_dim)
        position_ids: 位置ID，如果为None则使用默认序列

    Returns:
        旋转后的查询和键张量
    """
    # 如果提供了position_ids，则根据位置选择对应的cos和sin
    if position_ids is not None:
        # 根据提供的位置索引采样cos/sin，保持batch对齐
        cos = cos[position_ids]  # (batch_size, seq_len, head_dim)
        sin = sin[position_ids]

        # 插入注意力头维度，使其与查询/键张量广播兼容
        cos = cos.unsqueeze(1)  # (batch_size, 1, seq_len, head_dim)
        sin = sin.unsqueeze(1)
    else:
        # 默认情况下广播到(batch, heads, seq_len, head_dim)
        cos = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, head_dim)
        sin = sin.unsqueeze(0).unsqueeze(0)

    # 应用旋转变换
    # RoPE公式: R * x = x * cos + rotate_half(x) * sin
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)

    return q_embed, k_embed


class RoPEAttention(nn.Module):
    """集成RoPE的多头注意力机制"""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.1,
        max_position_embeddings: int = 2048,
        rope_base: float = 10000.0,
    ):
        super().__init__()
        assert d_model % n_heads == 0, "d_model必须能被n_heads整除"

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        # 线性投影层
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)

        # RoPE位置编码
        self.rope = RotaryPositionEmbedding(self.head_dim, max_position_embeddings, rope_base)

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor = None,
        position_ids: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: (batch_size, seq_len, d_model)
            attention_mask: (batch_size, seq_len, seq_len) 或 None
            position_ids: (batch_size, seq_len) 或 None

        Returns:
            output: (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, _ = hidden_states.size()

        # 线性投影
        query = self.w_q(hidden_states)
        key = self.w_k(hidden_states)
        value = self.w_v(hidden_states)

        # 重塑为多头格式
        query = query.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        # 获取RoPE编码
        cos, sin = self.rope(hidden_states, seq_len)

        # 应用RoPE到查询和键
        query, key = apply_rotary_pos_emb(query, key, cos, sin, position_ids)

        # 计算注意力分数
        scores = torch.matmul(query, key.transpose(-2, -1)) / self.scale

        # 应用注意力掩码
        if attention_mask is not None:
            # 将掩码中的0转换为大负数，1保持不变
            scores = scores.masked_fill(attention_mask == 0, -1e9)

        # 计算注意力权重
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 应用注意力权重到值
        attn_output = torch.matmul(attn_weights, value)

        # 重塑回原始形状
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.d_model)

        # 输出投影
        output = self.w_o(attn_output)

        return output


if __name__ == "__main__":
    # 测试RoPE实现
    print("Testing RoPE Implementation...")

    # 参数设置
    batch_size = 2
    seq_len = 128
    d_model = 512
    n_heads = 8
    head_dim = d_model // n_heads

    # 创建测试数据
    hidden_states = torch.randn(batch_size, seq_len, d_model)

    # 测试RoPE位置编码
    rope = RotaryPositionEmbedding(head_dim)
    cos, sin = rope(hidden_states)
    print(f"RoPE cos shape: {cos.shape}")
    print(f"RoPE sin shape: {sin.shape}")

    # 测试RoPE注意力
    rope_attn = RoPEAttention(d_model, n_heads)
    output = rope_attn(hidden_states)
    print(f"RoPE Attention output shape: {output.shape}")

    # 测试长序列外推能力
    long_seq_len = 256
    long_hidden_states = torch.randn(1, long_seq_len, d_model)
    long_output = rope_attn(long_hidden_states)
    print(f"Long sequence output shape: {long_output.shape}")

    print("RoPE implementation test completed successfully!")
