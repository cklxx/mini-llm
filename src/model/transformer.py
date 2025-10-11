"""
手写实现Transformer模型架构
包含所有核心组件：注意力机制、前馈网络、位置编码等
用于新手理解Transformer原理
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import MiniGPTConfig
from .gqa import GroupedQueryAttention
from .moe import SharedExpertMoE, SparseMoE


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization

    RMSNorm 比传统的 LayerNorm 更简单高效，去除了均值中心化操作，
    只保留方差归一化，计算公式为：

    y = x / sqrt(mean(x^2) + eps) * g

    其中 g 是可学习的缩放参数
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


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
        attention_output = F.scaled_dot_product_attention(Q, K, V, mask)
        # 形状: (batch_size, n_heads, seq_len, d_k)

        # 4. 拼接多头结果
        attention_output = attention_output.transpose(1, 2).contiguous()
        attention_output = attention_output.reshape(batch_size, seq_len, d_model)

        # 5. 输出投影
        output = self.w_o(attention_output)

        return output


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

        # 选择注意力机制类型
        if hasattr(config, "use_gqa") and config.use_gqa:
            self.attention = GroupedQueryAttention(
                d_model=config.hidden_size,
                num_heads=config.num_attention_heads,
                num_key_value_heads=getattr(config, "num_key_value_heads", None),
                dropout=config.attention_dropout,
                use_rope=getattr(config, "use_rope", True),
                max_position_embeddings=config.max_position_embeddings,
                rope_base=getattr(config, "rope_theta", 10000.0),
                flash_attn=getattr(config, "flash_attn", False),
            )
        else:
            # 兼容传统注意力
            self.attention = MultiHeadAttention(
                config.hidden_size, config.num_attention_heads, config.attention_dropout
            )

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
            self.feed_forward = SwiGLUFeedForward(
                config.hidden_size, config.intermediate_size, config.dropout
            )
            self.has_moe = False

        # 层归一化
        self.norm1 = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.norm2 = RMSNorm(config.hidden_size, config.rms_norm_eps)

        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
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
        if hasattr(self.config, "use_gqa") and self.config.use_gqa:
            attn_output, _ = self.attention(
                hidden_states=normalized_x, attention_mask=mask, position_ids=position_ids
            )
        else:
            attn_output = self.attention(normalized_x, normalized_x, normalized_x, mask)

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

        if collect_moe_loss:
            return x, moe_loss

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
        if not self.use_rope:
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
        """创建因果掩码（下三角矩阵）

        防止模型在预测时看到未来的token
        """
        mask = torch.tril(torch.ones(seq_len, seq_len))
        return mask  # 1表示可见，0表示掩码

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

        # 1. 词嵌入
        x = self.token_embedding(input_ids)  # (batch_size, seq_len, d_model)

        # 2. 位置编码（仅在不使用RoPE时）
        if not self.use_rope:
            x = self.positional_encoding(x)

        x = self.dropout(x)

        # 3. 创建因果掩码
        causal_mask = self.create_causal_mask(seq_len).to(input_ids.device)

        # 4. 生成position_ids（如果未提供）
        if position_ids is None and self.use_rope:
            position_ids = (
                torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
            )

        # 5. 通过Transformer块
        total_moe_loss = None
        for transformer_block in self.transformer_blocks:
            if return_aux_loss and getattr(self.config, "use_moe", False):
                x, block_moe_loss = transformer_block(
                    x, mask=causal_mask, position_ids=position_ids, collect_moe_loss=True
                )
                if block_moe_loss is not None:
                    if total_moe_loss is None:
                        total_moe_loss = block_moe_loss
                    else:
                        total_moe_loss = total_moe_loss + block_moe_loss
            else:
                x = transformer_block(x, causal_mask, position_ids)

        # 6. 层归一化
        x = self.layer_norm(x)

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
