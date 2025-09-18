"""
MiniGPT模型配置类
提供统一的配置管理，支持不同模型大小和训练模式
"""
from typing import Optional


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
        intermediate_size: Optional[int] = None,  # d_ff, 默认为 hidden_size * 4
        max_position_embeddings: int = 1024,  # max_len

        # 归一化和激活
        rms_norm_eps: float = 1e-6,
        hidden_act: str = 'swiglu',  # 激活函数类型

        # 位置编码
        rope_theta: float = 10000.0,  # RoPE theta参数
        use_rope: bool = False,  # 是否使用RoPE位置编码

        # 训练参数
        dropout: float = 0.1,
        attention_dropout: float = 0.1,

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
        scoring_func: str = 'softmax',  # 专家选择评分函数
        aux_loss_alpha: float = 0.1,  # 辅助损失权重
        seq_aux: bool = True,  # 序列级辅助损失
        norm_topk_prob: bool = True,  # 是否归一化top-k概率

        # 生成参数
        max_generate_length: int = 100,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,

        **kwargs
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
        assert self.hidden_size % self.num_attention_heads == 0, \
            f"hidden_size ({self.hidden_size}) 必须能被 num_attention_heads ({self.num_attention_heads}) 整除"

        assert self.vocab_size > 0, "vocab_size 必须大于 0"
        assert self.num_hidden_layers > 0, "num_hidden_layers 必须大于 0"
        assert self.num_attention_heads > 0, "num_attention_heads 必须大于 0"

        if self.use_moe:
            assert self.n_routed_experts >= self.num_experts_per_tok, \
                "n_routed_experts 必须大于等于 num_experts_per_tok"

    @property
    def head_dim(self) -> int:
        """每个注意力头的维度"""
        return self.hidden_size // self.num_attention_heads

    def to_dict(self) -> dict:
        """将配置转换为字典"""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}

    @classmethod
    def from_dict(cls, config_dict: dict) -> 'MiniGPTConfig':
        """从字典创建配置"""
        return cls(**config_dict)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.to_dict()})"


def get_tiny_config() -> MiniGPTConfig:
    """获取tiny模型配置 (~1M参数)"""
    return MiniGPTConfig(
        vocab_size=10000,
        hidden_size=128,
        num_hidden_layers=4,
        num_attention_heads=2,
        intermediate_size=512,
        max_position_embeddings=256,
        dropout=0.1
    )


def get_small_config() -> MiniGPTConfig:
    """获取small模型配置 (~25M参数)"""
    return MiniGPTConfig(
        vocab_size=10000,
        hidden_size=512,
        num_hidden_layers=6,
        num_attention_heads=8,
        intermediate_size=2048,
        max_position_embeddings=512,
        dropout=0.1
    )


def get_medium_config() -> MiniGPTConfig:
    """获取medium模型配置 (~100M参数)"""
    return MiniGPTConfig(
        vocab_size=10000,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        max_position_embeddings=1024,
        dropout=0.1
    )


def get_large_config() -> MiniGPTConfig:
    """获取large模型配置 (~350M参数)"""
    return MiniGPTConfig(
        vocab_size=32000,
        hidden_size=1024,
        num_hidden_layers=24,
        num_attention_heads=16,
        intermediate_size=4096,
        max_position_embeddings=2048,
        dropout=0.1
    )


def get_moe_config() -> MiniGPTConfig:
    """获取MOE模型配置"""
    return MiniGPTConfig(
        vocab_size=10000,
        hidden_size=512,
        num_hidden_layers=6,
        num_attention_heads=8,
        intermediate_size=2048,
        max_position_embeddings=512,
        dropout=0.1,
        use_moe=True,
        num_experts_per_tok=2,
        n_routed_experts=4,
        n_shared_experts=1
    )


# 预定义配置映射
CONFIG_MAPPING = {
    "tiny": get_tiny_config,
    "small": get_small_config,
    "medium": get_medium_config,
    "large": get_large_config,
    "moe": get_moe_config,
}


def get_config(config_name: str) -> MiniGPTConfig:
    """根据名称获取预定义配置"""
    if config_name not in CONFIG_MAPPING:
        raise ValueError(f"未知的配置名称: {config_name}. 可用配置: {list(CONFIG_MAPPING.keys())}")

    return CONFIG_MAPPING[config_name]()


if __name__ == "__main__":
    # 测试配置
    configs = ["tiny", "small", "medium", "large", "moe"]

    for config_name in configs:
        config = get_config(config_name)
        print(f"\n{config_name.upper()} 配置:")
        print(f"  参数量估算: ~{estimate_params(config):,}")
        print(f"  hidden_size: {config.hidden_size}")
        print(f"  num_layers: {config.num_hidden_layers}")
        print(f"  num_heads: {config.num_attention_heads}")
        print(f"  vocab_size: {config.vocab_size}")
        if config.use_moe:
            print(f"  MOE专家数: {config.n_routed_experts}")


def estimate_params(config: MiniGPTConfig) -> int:
    """估算模型参数量"""
    # 词嵌入
    embedding_params = config.vocab_size * config.hidden_size

    # Transformer层
    # 注意力: 4 * hidden_size^2 (Q, K, V, O projections)
    # 前馈: 2 * hidden_size * intermediate_size
    # RMSNorm: 2 * hidden_size (每层两个norm)
    layer_params = (
        4 * config.hidden_size * config.hidden_size +  # 注意力
        2 * config.hidden_size * config.intermediate_size +  # 前馈
        2 * config.hidden_size  # RMSNorm
    )

    transformer_params = config.num_hidden_layers * layer_params

    # 输出层
    output_params = config.hidden_size + config.vocab_size * config.hidden_size  # norm + lm_head

    total_params = embedding_params + transformer_params + output_params

    return total_params