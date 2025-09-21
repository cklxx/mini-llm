"""
Grouped-Query Attention (GQA) Implementation
GQA是提升推理效率的关键技术，被Llama2、Code Llama等模型采用
通过共享键值头降低KV缓存内存开销50-70%
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
from .rope import apply_rotary_pos_emb


class GroupedQueryAttention(nn.Module):
    """分组查询注意力(Grouped-Query Attention)实现

    GQA是Multi-Head Attention和Multi-Query Attention的折中方案：
    - MHA: 每个头都有独立的Q、K、V
    - MQA: 所有头共享同一个K、V
    - GQA: 将头分组，组内共享K、V

    优势：
    - 显著降低KV缓存内存消耗
    - 保持接近MHA的模型质量
    - 提升推理速度，特别是长序列
    """

    def __init__(self,
                 d_model: int,
                 num_heads: int,
                 num_key_value_heads: int = None,
                 dropout: float = 0.1,
                 use_rope: bool = True,
                 max_position_embeddings: int = 2048,
                 rope_base: float = 10000.0):
        """
        Args:
            d_model: 模型维度
            num_heads: 查询头数量
            num_key_value_heads: 键值头数量，默认为num_heads//4
            dropout: dropout率
            use_rope: 是否使用RoPE位置编码
            max_position_embeddings: 最大位置长度
            rope_base: RoPE基数
        """
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads

        # 如果未指定KV头数，默认为查询头数的1/4
        if num_key_value_heads is None:
            num_key_value_heads = max(1, num_heads // 4)
        self.num_key_value_heads = num_key_value_heads

        # 确保查询头数能被KV头数整除
        assert num_heads % num_key_value_heads == 0, \
            f"num_heads ({num_heads}) 必须能被 num_key_value_heads ({num_key_value_heads}) 整除"

        self.num_queries_per_kv = num_heads // num_key_value_heads
        self.head_dim = d_model // num_heads

        # 线性投影层
        self.q_proj = nn.Linear(d_model, num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * self.head_dim, d_model, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)

        # RoPE配置
        self.use_rope = use_rope
        if use_rope:
            from .rope import RotaryPositionEmbedding
            self.rope = RotaryPositionEmbedding(
                self.head_dim,
                max_position_embeddings,
                rope_base
            )

    def _repeat_kv(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """重复键值张量以匹配查询头数

        Args:
            hidden_states: (batch, num_key_value_heads, seq_len, head_dim)

        Returns:
            (batch, num_heads, seq_len, head_dim)
        """
        batch, num_key_value_heads, slen, head_dim = hidden_states.shape
        if self.num_queries_per_kv == 1:
            return hidden_states

        # 扩展KV头以匹配查询头数
        hidden_states = hidden_states[:, :, None, :, :].expand(
            batch, num_key_value_heads, self.num_queries_per_kv, slen, head_dim
        )
        return hidden_states.reshape(batch, num_key_value_heads * self.num_queries_per_kv, slen, head_dim)

    def forward(self,
                hidden_states: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.Tensor] = None,
                past_key_value: Optional[Tuple[torch.Tensor]] = None,
                use_cache: bool = False) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor]]]:
        """
        Args:
            hidden_states: (batch_size, seq_len, d_model)
            attention_mask: (batch_size, seq_len, seq_len) 或 None
            position_ids: (batch_size, seq_len) 或 None
            past_key_value: 过去的键值缓存
            use_cache: 是否使用缓存

        Returns:
            attn_output: (batch_size, seq_len, d_model)
            present_key_value: 当前键值缓存（如果use_cache=True）
        """
        batch_size, seq_len, _ = hidden_states.size()

        # 投影到查询、键、值
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # 重塑为多头格式
        query_states = query_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # 应用RoPE位置编码
        if self.use_rope:
            cos, sin = self.rope(hidden_states, seq_len)
            query_states, key_states = apply_rotary_pos_emb(
                query_states, key_states, cos, sin, position_ids
            )

        # 处理KV缓存
        if past_key_value is not None:
            past_key, past_value = past_key_value
            key_states = torch.cat([past_key, key_states], dim=2)
            value_states = torch.cat([past_value, value_states], dim=2)

        present_key_value = (key_states, value_states) if use_cache else None

        # 重复键值以匹配查询头数
        key_states = self._repeat_kv(key_states)
        value_states = self._repeat_kv(value_states)

        # 计算注意力分数
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / self.scale

        # 应用注意力掩码
        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :seq_len, :key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        # 计算注意力权重
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = self.dropout(attn_weights)

        # 应用注意力权重
        attn_output = torch.matmul(attn_weights, value_states)

        # 重塑回原始形状
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, seq_len, self.d_model)

        # 输出投影
        attn_output = self.o_proj(attn_output)

        return attn_output, present_key_value


class OptimizedMultiHeadAttention(nn.Module):
    """优化的多头注意力，支持GQA和传统MHA"""

    def __init__(self,
                 d_model: int,
                 n_heads: int,
                 dropout: float = 0.1,
                 use_gqa: bool = True,
                 num_key_value_heads: int = None,
                 use_rope: bool = True,
                 max_position_embeddings: int = 2048):
        super().__init__()

        self.use_gqa = use_gqa

        if use_gqa:
            self.attention = GroupedQueryAttention(
                d_model=d_model,
                num_heads=n_heads,
                num_key_value_heads=num_key_value_heads,
                dropout=dropout,
                use_rope=use_rope,
                max_position_embeddings=max_position_embeddings
            )
        else:
            # 传统多头注意力实现
            from .rope import RoPEAttention
            if use_rope:
                self.attention = RoPEAttention(d_model, n_heads, dropout, max_position_embeddings)
            else:
                # 回退到传统注意力
                from .transformer import MultiHeadAttention
                self.attention = MultiHeadAttention(d_model, n_heads, dropout)

    def forward(self,
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                mask: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.Tensor] = None,
                past_key_value: Optional[Tuple[torch.Tensor]] = None,
                use_cache: bool = False) -> torch.Tensor:

        if self.use_gqa:
            # 对于GQA，输入应该是相同的hidden_states
            assert torch.equal(query, key) and torch.equal(key, value), \
                "GQA要求query、key、value为同一个hidden_states"

            output, present_kv = self.attention(
                hidden_states=query,
                attention_mask=mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                use_cache=use_cache
            )

            if use_cache:
                return output, present_kv
            return output
        else:
            # 传统注意力调用
            if hasattr(self.attention, 'forward'):
                # RoPEAttention的情况
                if hasattr(self.attention, 'rope'):
                    return self.attention(query, mask, position_ids)
                else:
                    # 传统MultiHeadAttention
                    return self.attention(query, key, value, mask)


if __name__ == "__main__":
    # 测试GQA实现
    print("Testing Grouped-Query Attention...")

    # 参数设置
    batch_size = 2
    seq_len = 128
    d_model = 512
    num_heads = 8
    num_kv_heads = 2  # 8头查询，2头键值

    # 创建测试数据
    hidden_states = torch.randn(batch_size, seq_len, d_model)

    # 测试GQA
    gqa = GroupedQueryAttention(
        d_model=d_model,
        num_heads=num_heads,
        num_key_value_heads=num_kv_heads
    )

    output, _ = gqa(hidden_states)
    print(f"GQA output shape: {output.shape}")

    # 计算参数量对比
    mha_params = 4 * d_model * d_model  # Q, K, V, O 投影
    gqa_q_params = d_model * d_model
    gqa_kv_params = 2 * d_model * (d_model * num_kv_heads // num_heads)
    gqa_o_params = d_model * d_model
    gqa_total = gqa_q_params + gqa_kv_params + gqa_o_params

    print(f"MHA parameters: {mha_params:,}")
    print(f"GQA parameters: {gqa_total:,}")
    print(f"Parameter reduction: {(1 - gqa_total/mha_params)*100:.1f}%")

    # 测试KV缓存内存节省
    kv_cache_mha = 2 * batch_size * num_heads * seq_len * (d_model // num_heads)
    kv_cache_gqa = 2 * batch_size * num_kv_heads * seq_len * (d_model // num_heads)

    print(f"KV cache MHA: {kv_cache_mha:,} elements")
    print(f"KV cache GQA: {kv_cache_gqa:,} elements")
    print(f"KV cache reduction: {(1 - kv_cache_gqa/kv_cache_mha)*100:.1f}%")

    print("GQA implementation test completed successfully!")