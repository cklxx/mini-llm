"""
Mixture of Experts (MoE) 架构实现
支持稀疏专家系统，提高模型容量同时保持计算效率
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List
from .activation_functions import SwiGLUFeedForward, get_feedforward_layer


class Router(nn.Module):
    """
    MoE路由器 - 负责将输入token分配给不同的专家

    使用top-k选择策略，只激活最相关的k个专家
    """
    def __init__(
        self,
        d_model: int,
        num_experts: int,
        top_k: int = 2,
        capacity_factor: float = 1.25,
        drop_tokens: bool = True,
        use_bias: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.top_k = top_k
        self.capacity_factor = capacity_factor
        self.drop_tokens = drop_tokens

        # 路由网络
        self.gate = nn.Linear(d_model, num_experts, bias=use_bias)

        # 噪声系数，用于训练期间的负载均衡
        self.noise_epsilon = 1e-2

    def _gates_to_load(self, gates: torch.Tensor) -> torch.Tensor:
        """计算每个专家的负载（用于负载均衡损失）"""
        return (gates > 0).sum(0)

    def _prob_in_top_k(
        self, clean_values: torch.Tensor, noisy_values: torch.Tensor, noise_stddev: torch.Tensor, k: int
    ) -> torch.Tensor:
        """计算每个token在top-k中的概率"""
        batch = clean_values.size(0)
        m = noisy_values.size(-1)
        top_values, _ = torch.topk(noisy_values, k, dim=-1)
        threshold_positions_if_in = torch.arange(batch, device=clean_values.device).unsqueeze(-1) * m + k - 1
        threshold_if_in = torch.gather(
            noisy_values.flatten(), 0, threshold_positions_if_in.flatten()
        ).reshape(batch, 1)
        is_in = torch.gt(noisy_values, threshold_if_in)
        threshold_positions_if_out = threshold_positions_if_in - 1
        threshold_if_out = torch.gather(
            noisy_values.flatten(), 0, threshold_positions_if_out.flatten()
        ).reshape(batch, 1)
        # 防止除零
        normal = torch.distributions.normal.Normal(clean_values, noise_stddev)
        prob_if_in = normal.cdf((threshold_if_in - clean_values) / noise_stddev)
        prob_if_out = normal.cdf((threshold_if_out - clean_values) / noise_stddev)
        prob = torch.where(is_in, prob_if_in, prob_if_out)
        return prob

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        前向传播

        Args:
            x: 输入张量 (batch_size * seq_len, d_model)

        Returns:
            top_k_gates: top-k专家的门控权重 (batch_size * seq_len, top_k)
            top_k_indices: top-k专家的索引 (batch_size * seq_len, top_k)
            load_balancing_loss: 负载均衡损失
        """
        # 计算门控分数
        gates = self.gate(x)  # (batch_size * seq_len, num_experts)

        # 添加噪声（仅在训练时）
        if self.training:
            noise = torch.randn_like(gates) * self.noise_epsilon
            gates = gates + noise

        # 应用softmax
        gates = F.softmax(gates, dim=-1)

        # 选择top-k专家
        top_k_gates, top_k_indices = torch.topk(gates, self.top_k, dim=-1)

        # 重新归一化top-k门控权重
        top_k_gates = top_k_gates / top_k_gates.sum(dim=-1, keepdim=True)

        # 计算负载均衡损失
        if self.training:
            # 计算每个专家的重要性（门控权重的平均值）
            importance = gates.mean(dim=0)
            # 计算每个专家的负载（分配给它的token数量的比例）
            load = torch.zeros(self.num_experts, device=x.device)
            for i in range(self.num_experts):
                load[i] = (top_k_indices == i).float().mean()
            # 负载均衡损失 = importance * load 的变异系数
            aux_loss = (importance * load).sum()
            load_balancing_loss = aux_loss * self.num_experts
        else:
            load_balancing_loss = torch.tensor(0.0, device=x.device)

        return top_k_gates, top_k_indices, load_balancing_loss


class Expert(nn.Module):
    """
    MoE专家网络 - 每个专家是一个前馈网络
    """
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        activation: str = "swiglu",
        dropout: float = 0.1,
        bias: bool = False,
    ):
        super().__init__()
        if activation == "swiglu":
            self.ffn = SwiGLUFeedForward(d_model, d_ff, dropout, bias)
        else:
            self.ffn = get_feedforward_layer(
                d_model, d_ff, "standard", activation, dropout, bias
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ffn(x)


class SparseMoE(nn.Module):
    """
    稀疏MoE层 - 核心的MoE实现

    只激活top-k个专家，大大减少计算量
    """
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        num_experts: int,
        top_k: int = 2,
        capacity_factor: float = 1.25,
        activation: str = "swiglu",
        dropout: float = 0.1,
        bias: bool = False,
        load_balancing_weight: float = 0.01,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.num_experts = num_experts
        self.top_k = top_k
        self.load_balancing_weight = load_balancing_weight

        # 路由器
        self.router = Router(
            d_model=d_model,
            num_experts=num_experts,
            top_k=top_k,
            capacity_factor=capacity_factor,
        )

        # 专家网络
        self.experts = nn.ModuleList([
            Expert(d_model, d_ff, activation, dropout, bias)
            for _ in range(num_experts)
        ])

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播

        Args:
            x: 输入张量 (batch_size, seq_len, d_model)

        Returns:
            output: 输出张量 (batch_size, seq_len, d_model)
            load_balancing_loss: 负载均衡损失
        """
        batch_size, seq_len, d_model = x.shape

        # 重塑为 (batch_size * seq_len, d_model)
        x_flat = x.view(-1, d_model)

        # 路由
        top_k_gates, top_k_indices, load_balancing_loss = self.router(x_flat)

        # 初始化输出
        output = torch.zeros_like(x_flat)

        # 计算专家输出
        # 使用更高效的批处理方法
        for k in range(self.top_k):
            # 获取第k个专家的索引和权重
            expert_indices = top_k_indices[:, k]  # (batch_size * seq_len,)
            gates = top_k_gates[:, k].unsqueeze(-1)  # (batch_size * seq_len, 1)

            # 为每个专家收集输入
            for expert_id in range(self.num_experts):
                expert_mask = (expert_indices == expert_id)
                if expert_mask.any():
                    expert_input = x_flat[expert_mask]
                    expert_output = self.experts[expert_id](expert_input)
                    # 应用门控权重并累加到输出
                    output[expert_mask] += gates[expert_mask] * expert_output

        # 重塑回原始形状
        output = output.view(batch_size, seq_len, d_model)

        return output, load_balancing_loss


class SharedExpertMoE(nn.Module):
    """
    共享专家MoE - 包含共享专家和路由专家

    共享专家始终被激活，路由专家按需激活
    这种设计在某些任务上表现更好
    """
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        num_shared_experts: int,
        num_routed_experts: int,
        top_k: int = 2,
        activation: str = "swiglu",
        dropout: float = 0.1,
        bias: bool = False,
        load_balancing_weight: float = 0.01,
    ):
        super().__init__()
        self.num_shared_experts = num_shared_experts
        self.num_routed_experts = num_routed_experts

        # 共享专家（始终激活）
        self.shared_experts = nn.ModuleList([
            Expert(d_model, d_ff, activation, dropout, bias)
            for _ in range(num_shared_experts)
        ])

        # 路由MoE层
        if num_routed_experts > 0:
            self.routed_moe = SparseMoE(
                d_model=d_model,
                d_ff=d_ff,
                num_experts=num_routed_experts,
                top_k=top_k,
                activation=activation,
                dropout=dropout,
                bias=bias,
                load_balancing_weight=load_balancing_weight,
            )
        else:
            self.routed_moe = None

        # 输出组合权重
        self.shared_weight = nn.Parameter(torch.ones(1))
        if self.routed_moe is not None:
            self.routed_weight = nn.Parameter(torch.ones(1))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播

        Args:
            x: 输入张量 (batch_size, seq_len, d_model)

        Returns:
            output: 输出张量 (batch_size, seq_len, d_model)
            load_balancing_loss: 负载均衡损失
        """
        # 共享专家输出
        shared_output = torch.zeros_like(x)
        for expert in self.shared_experts:
            shared_output += expert(x)
        shared_output = shared_output * self.shared_weight / len(self.shared_experts)

        # 路由专家输出
        if self.routed_moe is not None:
            routed_output, load_balancing_loss = self.routed_moe(x)
            routed_output = routed_output * self.routed_weight
            output = shared_output + routed_output
        else:
            output = shared_output
            load_balancing_loss = torch.tensor(0.0, device=x.device)

        return output, load_balancing_loss


class MoETransformerBlock(nn.Module):
    """
    MoE Transformer块 - 将MoE层集成到Transformer架构中
    """
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        num_experts: int,
        top_k: int = 2,
        moe_type: str = "sparse",  # "sparse" or "shared"
        num_shared_experts: int = 2,
        activation: str = "swiglu",
        dropout: float = 0.1,
        bias: bool = False,
        load_balancing_weight: float = 0.01,
    ):
        super().__init__()
        from .transformer import MultiHeadAttention  # 避免循环导入

        self.attention = MultiHeadAttention(d_model, n_heads, dropout)

        # 选择MoE类型
        if moe_type == "sparse":
            self.feed_forward = SparseMoE(
                d_model=d_model,
                d_ff=d_ff,
                num_experts=num_experts,
                top_k=top_k,
                activation=activation,
                dropout=dropout,
                bias=bias,
                load_balancing_weight=load_balancing_weight,
            )
        elif moe_type == "shared":
            self.feed_forward = SharedExpertMoE(
                d_model=d_model,
                d_ff=d_ff,
                num_shared_experts=num_shared_experts,
                num_routed_experts=num_experts - num_shared_experts,
                top_k=top_k,
                activation=activation,
                dropout=dropout,
                bias=bias,
                load_balancing_weight=load_balancing_weight,
            )
        else:
            raise ValueError(f"Unsupported MoE type: {moe_type}")

        # 层归一化
        self.norm1 = nn.RMSNorm(d_model)
        self.norm2 = nn.RMSNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播

        Args:
            x: 输入张量 (batch_size, seq_len, d_model)
            mask: 注意力掩码

        Returns:
            output: 输出张量 (batch_size, seq_len, d_model)
            load_balancing_loss: 负载均衡损失
        """
        # 多头注意力 + 残差连接 + 层归一化
        attn_output = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))

        # MoE前馈网络 + 残差连接 + 层归一化
        ff_output, load_balancing_loss = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))

        return x, load_balancing_loss


def create_moe_model(
    vocab_size: int,
    d_model: int = 512,
    n_heads: int = 8,
    n_layers: int = 6,
    d_ff: int = 2048,
    num_experts: int = 8,
    top_k: int = 2,
    moe_type: str = "sparse",
    num_shared_experts: int = 2,
    max_len: int = 1024,
    dropout: float = 0.1,
    load_balancing_weight: float = 0.01,
) -> nn.Module:
    """
    创建MoE模型

    Args:
        vocab_size: 词汇表大小
        d_model: 模型维度
        n_heads: 注意力头数
        n_layers: 层数
        d_ff: 前馈网络隐藏维度
        num_experts: 专家数量
        top_k: 激活的专家数量
        moe_type: MoE类型 ("sparse" or "shared")
        num_shared_experts: 共享专家数量（仅用于shared类型）
        max_len: 最大序列长度
        dropout: dropout率
        load_balancing_weight: 负载均衡损失权重

    Returns:
        MoE模型
    """
    from .transformer import PositionalEncoding  # 避免循环导入

    class MoEGPT(nn.Module):
        def __init__(self):
            super().__init__()
            self.vocab_size = vocab_size
            self.d_model = d_model
            self.max_len = max_len
            self.load_balancing_weight = load_balancing_weight

            # 词嵌入层
            self.token_embedding = nn.Embedding(vocab_size, d_model)

            # 位置编码
            self.positional_encoding = PositionalEncoding(d_model, max_len)

            # MoE Transformer层
            self.transformer_blocks = nn.ModuleList([
                MoETransformerBlock(
                    d_model=d_model,
                    n_heads=n_heads,
                    d_ff=d_ff,
                    num_experts=num_experts,
                    top_k=top_k,
                    moe_type=moe_type,
                    num_shared_experts=num_shared_experts,
                    dropout=dropout,
                    load_balancing_weight=load_balancing_weight,
                )
                for _ in range(n_layers)
            ])

            # 输出层
            self.layer_norm = nn.LayerNorm(d_model)
            self.lm_head = nn.Linear(d_model, vocab_size)

            self.dropout = nn.Dropout(dropout)

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
                elif isinstance(module, (nn.LayerNorm, nn.RMSNorm)):
                    nn.init.ones_(module.weight)
                    if hasattr(module, 'bias') and module.bias is not None:
                        nn.init.zeros_(module.bias)

        def create_causal_mask(self, seq_len: int) -> torch.Tensor:
            """创建因果掩码"""
            mask = torch.tril(torch.ones(seq_len, seq_len))
            return mask

        def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> dict:
            """
            前向传播

            Returns:
                dict: 包含logits和load_balancing_loss的字典
            """
            batch_size, seq_len = input_ids.size()

            # 1. 词嵌入
            x = self.token_embedding(input_ids)

            # 2. 位置编码
            x = self.positional_encoding(x)
            x = self.dropout(x)

            # 3. 创建因果掩码
            causal_mask = self.create_causal_mask(seq_len).to(input_ids.device)

            # 4. 通过MoE Transformer块
            total_load_balancing_loss = 0.0
            for transformer_block in self.transformer_blocks:
                x, load_balancing_loss = transformer_block(x, causal_mask)
                total_load_balancing_loss += load_balancing_loss

            # 5. 层归一化
            x = self.layer_norm(x)

            # 6. 输出投影
            logits = self.lm_head(x)

            return {
                "logits": logits,
                "load_balancing_loss": total_load_balancing_loss * self.load_balancing_weight
            }

        def get_num_params(self) -> int:
            """获取模型参数数量"""
            return sum(p.numel() for p in self.parameters())

    return MoEGPT()