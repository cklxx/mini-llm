"""
MiniGPT模型配置类
提供统一的配置管理，支持不同模型大小和训练模式
"""


class MiniGPTConfig:
    """MiniGPT模型配置类

    包含模型架构、训练参数和优化选项的完整配置
    """

    model_type = "minigpt"

    def __init__(
        self,
        # 基础模型参数
        vocab_size: int = 10000,
        hidden_size: int = 512,  # d_model
        num_hidden_layers: int = 6,  # n_layers
        num_attention_heads: int = 8,  # n_heads
        intermediate_size: int | None = None,  # d_ff, 默认为 hidden_size * 4
        max_position_embeddings: int = 1024,  # max_len
        # 归一化和激活
        rms_norm_eps: float = 1e-6,
        hidden_act: str = "swiglu",  # 激活函数类型
        # 位置编码
        rope_theta: float = 10000.0,  # RoPE theta参数
        use_rope: bool = True,  # 是否使用RoPE位置编码（推荐）
        # 训练参数
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        # 注意力机制优化
        use_gqa: bool = True,  # 是否使用分组查询注意力
        num_key_value_heads: int | None = None,  # KV头数量（默认为num_attention_heads//4）
        # 权重共享
        tie_word_embeddings: bool = True,  # 是否共享输入输出嵌入权重
        # 特殊token
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        pad_token_id: int = 0,
        # 性能优化
        flash_attn: bool = False,  # 是否使用Flash Attention
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
        self.intermediate_size = intermediate_size or hidden_size * 4
        self.max_position_embeddings = max_position_embeddings

        # 归一化和激活
        self.rms_norm_eps = rms_norm_eps
        self.hidden_act = hidden_act

        # 位置编码
        self.rope_theta = rope_theta
        self.use_rope = use_rope

        # 训练参数
        self.dropout = dropout
        self.attention_dropout = attention_dropout

        # 注意力机制优化
        self.use_gqa = use_gqa
        if use_gqa and num_key_value_heads is None:
            # 默认GQA比例为4:1
            self.num_key_value_heads = max(1, num_attention_heads // 4)
        else:
            self.num_key_value_heads = num_key_value_heads

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


def get_tiny_config() -> MiniGPTConfig:
    """获取tiny模型配置 (~1M参数)
    采用深而窄的架构设计，提升参数效率
    """
    return MiniGPTConfig(
        vocab_size=10000,
        hidden_size=128,
        num_hidden_layers=8,  # 增加深度
        num_attention_heads=4,
        num_key_value_heads=1,  # GQA优化
        intermediate_size=384,  # 调整FFN大小
        max_position_embeddings=512,
        dropout=0.1,
        use_rope=True,
        use_gqa=True,
        tie_word_embeddings=True,
    )


def get_small_config() -> MiniGPTConfig:
    """获取small模型配置 (~25M参数)
    瘦长架构优化：更窄但更深，降低内存峰值
    - 减少宽度(d_model)降低激活值内存占用
    - 增加深度提升模型表达能力
    - 保持总参数量基本不变
    """
    return MiniGPTConfig(
        vocab_size=10000,
        hidden_size=288,  # 384 -> 288 (减少25%)，降低内存峰值
        num_hidden_layers=18,  # 12 -> 18 (增加50%)，提升表达能力
        num_attention_heads=9,  # 保持 hidden_size/n_heads = 32
        num_key_value_heads=3,  # GQA优化：3:1比例 (9/3=3)
        intermediate_size=1152,  # 4倍hidden_size
        max_position_embeddings=1024,
        dropout=0.1,
        use_rope=True,  # ✅ RoPE位置编码
        use_gqa=True,  # ✅ 分组查询注意力
        tie_word_embeddings=True,  # ✅ 权重共享优化
    )


def get_small_30m_config() -> MiniGPTConfig:
    """获取约 30M 参数量的小型模型配置

    相比 `small` 预设略微加深网络并提升上下文长度，适合作为轻量级对话/推理模型的起点。
    """

    return MiniGPTConfig(
        vocab_size=12000,  # 略扩的词表覆盖范围
        hidden_size=384,
        num_hidden_layers=13,  # 比 small 略深增强表示能力
        num_attention_heads=12,
        num_key_value_heads=3,  # 维持 4:1 的 GQA 比例
        intermediate_size=1408,  # ≈3.67× hidden，兼顾算力与表达力
        max_position_embeddings=2048,
        dropout=0.1,
        attention_dropout=0.1,
        use_rope=True,
        use_gqa=True,
        flash_attn=True,
        tie_word_embeddings=True,
    )


def get_medium_config() -> MiniGPTConfig:
    """获取 medium 模型配置 (≈75M 参数，瘦长架构)

    最新版本采用“瘦长”（narrow & deep）的结构策略：
    - 隐藏维度压缩至 384，使单层计算与显存开销更低
    - 层数提升至 20 层，以深度弥补宽度带来的表达能力下降
    - 12 个注意力头 + 3 个 KV 头保持 4:1 的 GQA 比例，head_dim=32 确保稳定性
    - FFN 宽度 1536 (= 4 × hidden)，兼顾收敛速度与实现简洁

    在保持 2K 上下文的同时，Flash Attention 与权重共享仍默认启用，用于保证推理吞吐与显存友好性。
    """

    return MiniGPTConfig(
        vocab_size=20000,
        hidden_size=384,
        num_hidden_layers=20,
        num_attention_heads=12,
        num_key_value_heads=3,  # GQA 4:1
        intermediate_size=1536,
        max_position_embeddings=2048,
        dropout=0.1,
        use_rope=True,
        use_gqa=True,
        flash_attn=True,
        tie_word_embeddings=True,
    )


def get_large_config() -> MiniGPTConfig:
    """获取large模型配置 (~350M参数)
    全面优化的现代架构
    """
    return MiniGPTConfig(
        vocab_size=32000,
        hidden_size=768,  # 优化宽度
        num_hidden_layers=32,  # 显著增加深度
        num_attention_heads=24,
        num_key_value_heads=6,  # GQA优化
        intermediate_size=3072,
        max_position_embeddings=4096,
        dropout=0.1,
        use_rope=True,
        use_gqa=True,
        tie_word_embeddings=True,
    )


def get_foundation_config() -> MiniGPTConfig:
    """获取foundation模型配置 (~200M参数)

    为训练具备基础智能能力的中型模型量身定制：
    - 24层 Transformer，配合更宽的隐藏维度 768，平衡深度与计算吞吐
    - 16 头注意力 + 4 个KV头（GQA 4:1）保证稳定性与高效显存利用
    - FFN 宽度 2688 (= 3.5 × hidden_size) 在计算和表达力之间折衷
    - 约 2.09 亿参数，适配 32GB 级别单卡或两卡训练
    """
    return MiniGPTConfig(
        vocab_size=32000,
        hidden_size=768,
        num_hidden_layers=24,
        num_attention_heads=16,
        num_key_value_heads=4,
        intermediate_size=2688,
        max_position_embeddings=4096,
        dropout=0.1,
        attention_dropout=0.1,
        use_rope=True,
        use_gqa=True,
        flash_attn=True,
        gradient_checkpointing=True,
        tie_word_embeddings=True,
    )


def get_moe_config() -> MiniGPTConfig:
    """获取MOE模型配置
    结合现代架构优化的专家混合模型
    """
    return MiniGPTConfig(
        vocab_size=10000,
        hidden_size=384,
        num_hidden_layers=12,  # 深度优化
        num_attention_heads=12,
        num_key_value_heads=3,  # GQA优化
        intermediate_size=1536,
        max_position_embeddings=1024,
        dropout=0.1,
        use_rope=True,
        use_gqa=True,
        tie_word_embeddings=True,
        use_moe=True,
        num_experts_per_tok=2,
        n_routed_experts=4,
        n_shared_experts=1,
    )


# 预定义配置映射
CONFIG_MAPPING = {
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

    # 前馈网络: SwiGLU需要3个线性层
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
    configs = ["tiny", "small", "small_30m", "medium", "foundation", "large", "moe"]

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
