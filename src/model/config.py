"""
MiniGPT模型配置类
提供统一的配置管理，支持不同模型大小和训练模式
"""

import math
import warnings


class MiniGPTConfig:
    """MiniGPT模型配置类

    包含模型架构、训练参数和优化选项的完整配置
    """

    model_type = "minigpt"

    def __init__(
        self,
        # 基础模型参数
        vocab_size: int = 6400,
        hidden_size: int = 512,  # d_model
        num_hidden_layers: int = 8,  # n_layers
        num_attention_heads: int = 8,  # n_heads
        intermediate_size: int | None = None,
        max_position_embeddings: int = 32768,  # max_len
        # 归一化和激活
        rms_norm_eps: float = 1e-5,
        hidden_act: str = "silu",  # 激活函数类型（MiniMind 默认）
        # 位置编码
        rope_theta: float = 1_000_000.0,  # RoPE theta参数
        use_rope: bool = True,  # 是否使用RoPE位置编码（推荐）
        rope_scaling: dict | None = None,  # 可选的RoPE扩展配置（如YaRN）
        # 训练参数
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        # 注意力机制优化
        use_gqa: bool = True,  # 是否使用分组查询注意力
        num_key_value_heads: int | None = 2,
        # 权重共享
        tie_word_embeddings: bool = True,  # 是否共享输入输出嵌入权重
        # 特殊token
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        pad_token_id: int = 0,
        # 性能优化
        flash_attn: bool = True,  # 是否使用Flash Attention
        gradient_checkpointing: bool = False,  # 梯度检查点
        # MOE配置 (Mixture of Experts)
        use_moe: bool = False,
        num_experts_per_tok: int = 2,  # 每个token选择的专家数量
        n_routed_experts: int = 4,  # 总的专家数量
        n_shared_experts: int = 1,  # 共享专家数量
        scoring_func: str = "softmax",  # 专家选择评分函数
        aux_loss_alpha: float = 0.1,  # 辅助损失权重
        seq_aux: bool = True,  # 序列级辅助损失
        norm_topk_prob: bool = True,  # 是否归一化top-k概率
        # 生成参数
        max_generate_length: int = 100,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        **kwargs,
    ):
        # 基础参数
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = self._resolve_intermediate_size(
            hidden_size, intermediate_size
        )
        self.max_position_embeddings = max_position_embeddings

        # 归一化和激活
        self.rms_norm_eps = rms_norm_eps
        self.hidden_act = hidden_act

        # 位置编码
        self.rope_theta = rope_theta
        self.use_rope = use_rope
        self.rope_scaling = rope_scaling

        # 训练参数
        self.dropout = dropout
        self.attention_dropout = attention_dropout

        # 注意力机制优化
        self.use_gqa = use_gqa
        self.num_key_value_heads = self._normalize_num_key_value_heads(
            num_attention_heads=num_attention_heads,
            requested_kv_heads=num_key_value_heads,
            use_gqa=use_gqa,
        )

        # 权重共享
        self.tie_word_embeddings = tie_word_embeddings

        # 特殊token
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id

        # 性能优化
        self.flash_attn = flash_attn
        self.gradient_checkpointing = gradient_checkpointing

        # MOE配置
        self.use_moe = use_moe
        self.num_experts_per_tok = num_experts_per_tok
        self.n_routed_experts = n_routed_experts
        self.n_shared_experts = n_shared_experts
        self.scoring_func = scoring_func
        self.aux_loss_alpha = aux_loss_alpha
        self.seq_aux = seq_aux
        self.norm_topk_prob = norm_topk_prob

        # 生成参数
        self.max_generate_length = max_generate_length
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p

        # 验证配置
        self._validate_config()

    @staticmethod
    def _resolve_intermediate_size(
        hidden_size: int, requested_size: int | None
    ) -> int:
        """Resolve FFN hidden size following MiniMind's 8/3 rounding strategy."""

        if requested_size is not None:
            return requested_size

        raw = int(hidden_size * 8 / 3)
        multiple = 64
        return multiple * ((raw + multiple - 1) // multiple)

    @staticmethod
    def _normalize_num_key_value_heads(
        *,
        num_attention_heads: int,
        requested_kv_heads: int | None,
        use_gqa: bool,
    ) -> int | None:
        """Derive a valid number of key/value heads for the configuration.

        When grouped-query attention is enabled we ensure the returned value
        divides the number of attention heads, falling back to sensible
        defaults otherwise.  For plain multi-head attention we keep the
        requested value so downstream modules can detect the original choice.
        """

        if not use_gqa:
            return requested_kv_heads or num_attention_heads

        if requested_kv_heads is None:
            return max(1, num_attention_heads // 4)

        normalized = requested_kv_heads
        if requested_kv_heads <= 0:
            normalized = max(1, num_attention_heads // 4)
            warnings.warn(
                "num_key_value_heads must be positive when using GQA; falling back to"
                f" {normalized}.",
                RuntimeWarning,
                stacklevel=3,
            )
        elif requested_kv_heads > num_attention_heads:
            normalized = num_attention_heads
            warnings.warn(
                "num_key_value_heads cannot exceed num_attention_heads; using"
                f" {normalized}.",
                RuntimeWarning,
                stacklevel=3,
            )
        elif num_attention_heads % requested_kv_heads != 0:
            normalized = math.gcd(num_attention_heads, requested_kv_heads) or 1
            warnings.warn(
                "num_key_value_heads does not evenly divide num_attention_heads;"
                f" using {normalized} instead of {requested_kv_heads}.",
                RuntimeWarning,
                stacklevel=3,
            )

        return normalized

    def _validate_config(self):
        """验证配置参数的有效性"""
        assert (
            self.hidden_size % self.num_attention_heads == 0
        ), f"hidden_size ({self.hidden_size}) 必须能被 num_attention_heads ({self.num_attention_heads}) 整除"

        assert self.vocab_size > 0, "vocab_size 必须大于 0"
        assert self.num_hidden_layers > 0, "num_hidden_layers 必须大于 0"
        assert self.num_attention_heads > 0, "num_attention_heads 必须大于 0"

        if self.use_moe:
            assert (
                self.n_routed_experts >= self.num_experts_per_tok
            ), "n_routed_experts 必须大于等于 num_experts_per_tok"

        if hasattr(self, "use_gqa") and self.use_gqa:
            if self.num_key_value_heads is not None:
                assert (
                    self.num_attention_heads % self.num_key_value_heads == 0
                ), f"num_attention_heads ({self.num_attention_heads}) 必须能被 num_key_value_heads ({self.num_key_value_heads}) 整除"

    @property
    def head_dim(self) -> int:
        """每个注意力头的维度"""
        return self.hidden_size // self.num_attention_heads

    def to_dict(self) -> dict:
        """将配置转换为字典"""
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}

    @classmethod
    def from_dict(cls, config_dict: dict) -> "MiniGPTConfig":
        """从字典创建配置"""
        return cls(**config_dict)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.to_dict()})"


def get_dense_26m_config() -> MiniGPTConfig:
    """26M 级别的 512×8 稠密骨干。"""

    return MiniGPTConfig(
        hidden_size=512,
        num_hidden_layers=8,
        num_attention_heads=8,
        num_key_value_heads=2,
        use_moe=False,
    )


def get_dense_104m_config() -> MiniGPTConfig:
    """104M 级别的 768×16 稠密骨干。"""

    return MiniGPTConfig(
        hidden_size=768,
        num_hidden_layers=16,
        num_attention_heads=8,
        num_key_value_heads=2,
        use_moe=False,
    )


def get_moe_145m_config() -> MiniGPTConfig:
    """145M 级别的 640×8 稀疏专家骨干。"""

    return MiniGPTConfig(
        hidden_size=640,
        num_hidden_layers=8,
        num_attention_heads=8,
        num_key_value_heads=2,
        use_moe=True,
        n_routed_experts=5,
        n_shared_experts=1,
        num_experts_per_tok=2,
    )


def get_tiny_config() -> MiniGPTConfig:
    """向后兼容：返回 26M 稠密配置。"""

    return get_dense_26m_config()


def get_small_config() -> MiniGPTConfig:
    """向后兼容：返回 104M 稠密配置。"""

    return get_dense_104m_config()


def get_small_30m_config() -> MiniGPTConfig:
    """向后兼容：返回 26M 稠密配置。"""

    return get_dense_26m_config()


def get_medium_config() -> MiniGPTConfig:
    """向后兼容：返回 145M 稀疏专家配置。"""

    return get_moe_145m_config()


def get_large_config() -> MiniGPTConfig:
    """向后兼容：返回 104M 稠密配置。"""

    return get_dense_104m_config()


def get_foundation_config() -> MiniGPTConfig:
    """向后兼容：返回 104M 稠密配置。"""

    return get_dense_104m_config()


def get_moe_config() -> MiniGPTConfig:
    """向后兼容：返回 145M 稀疏专家配置。"""

    return get_moe_145m_config()


# 预定义配置映射（统一使用模型规模命名）
CONFIG_MAPPING = {
    "dense_26m": get_dense_26m_config,
    "dense_104m": get_dense_104m_config,
    "moe_145m": get_moe_145m_config,
    # 历史别名
    "tiny": get_tiny_config,
    "small": get_small_config,
    "small_30m": get_small_30m_config,
    "medium": get_medium_config,
    "large": get_large_config,
    "foundation": get_foundation_config,
    "moe": get_moe_config,
}


def get_config(config_name: str) -> MiniGPTConfig:
    """根据名称获取预定义配置"""
    if config_name not in CONFIG_MAPPING:
        raise ValueError(f"未知的配置名称: {config_name}. 可用配置: {list(CONFIG_MAPPING.keys())}")

    return CONFIG_MAPPING[config_name]()


def estimate_params(config: MiniGPTConfig) -> int:
    """估算模型参数量（考虑GQA和权重共享优化）"""
    # 词嵌入
    embedding_params = config.vocab_size * config.hidden_size

    # Transformer层参数计算
    if getattr(config, "use_gqa", False) and getattr(config, "num_key_value_heads", None):
        # GQA情况下的注意力参数
        q_params = config.hidden_size * config.hidden_size  # Q投影
        kv_params = (
            2
            * config.hidden_size
            * (config.hidden_size * config.num_key_value_heads // config.num_attention_heads)
        )  # K,V投影
        o_params = config.hidden_size * config.hidden_size  # O投影
        attention_params = q_params + kv_params + o_params
    else:
        # 传统MHA参数
        attention_params = 4 * config.hidden_size * config.hidden_size  # Q, K, V, O projections

    # 前馈网络：根据是否启用 MoE 调整线性层数量
    if getattr(config, "use_moe", False):
        total_experts = max(getattr(config, "n_routed_experts", 0), 0)
        shared_experts = max(getattr(config, "n_shared_experts", 0), 0)
        if total_experts > 0:
            shared_experts = min(shared_experts, total_experts)
        routed_experts = max(total_experts - shared_experts, 0)
        expert_count = shared_experts + routed_experts
        expert_params = expert_count * 3 * config.hidden_size * config.intermediate_size
        router_params = config.hidden_size * routed_experts
        mixing_params = shared_experts + (1 if routed_experts > 0 else 0)
        ffn_params = expert_params + router_params + mixing_params
    else:
        ffn_params = 3 * config.hidden_size * config.intermediate_size

    # RMSNorm: 每层两个norm
    norm_params = 2 * config.hidden_size

    layer_params = attention_params + ffn_params + norm_params
    transformer_params = config.num_hidden_layers * layer_params

    # 输出层
    output_norm_params = config.hidden_size  # 最终norm层

    # 输出投影（考虑权重共享）
    if getattr(config, "tie_word_embeddings", False):
        output_projection_params = 0  # 共享嵌入权重
    else:
        output_projection_params = config.vocab_size * config.hidden_size

    output_params = output_norm_params + output_projection_params

    total_params = embedding_params + transformer_params + output_params

    return total_params


if __name__ == "__main__":
    # 测试配置
    configs = [
        "dense_26m",
        "dense_104m",
        "moe_145m",
        "tiny",
        "small",
        "small_30m",
        "medium",
        "foundation",
        "large",
        "moe",
    ]

    for config_name in configs:
        config = get_config(config_name)
        print(f"\n{config_name.upper()} 配置:")
        print(f"  参数量估算: ~{estimate_params(config):,}")
        print(f"  hidden_size: {config.hidden_size}")
        print(f"  num_layers: {config.num_hidden_layers}")
        print(f"  num_heads: {config.num_attention_heads}")
        if hasattr(config, "num_key_value_heads") and config.num_key_value_heads:
            print(f"  KV heads: {config.num_key_value_heads} (GQA)")
        print(f"  vocab_size: {config.vocab_size}")
        print(f"  使用RoPE: {getattr(config, 'use_rope', False)}")
        print(f"  使用GQA: {getattr(config, 'use_gqa', False)}")
        print(f"  权重共享: {getattr(config, 'tie_word_embeddings', False)}")
        if config.use_moe:
            print(f"  MOE专家数: {config.n_routed_experts}")
