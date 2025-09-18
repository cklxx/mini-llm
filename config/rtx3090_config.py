"""
RTX 3090 优化配置
针对RTX 3090 24GB显存和CUDA架构进行优化
"""
import torch
from src.model.config import MiniGPTConfig


def get_rtx3090_tiny_config() -> MiniGPTConfig:
    """RTX 3090 优化的 tiny 配置 (~1M参数)
    适合快速测试和原型验证
    """
    return MiniGPTConfig(
        vocab_size=10000,
        hidden_size=256,  # 增大到256以更好利用GPU
        num_hidden_layers=4,
        num_attention_heads=4,  # 保持整除关系
        intermediate_size=1024,  # 4倍关系
        max_position_embeddings=512,
        dropout=0.1,
        attention_dropout=0.1,
        rms_norm_eps=1e-6,
        flash_attn=True,  # 启用Flash Attention
        gradient_checkpointing=False,  # tiny模型不需要
        max_generate_length=256,
        temperature=0.8,
        top_k=50,
        top_p=0.9
    )


def get_rtx3090_small_config() -> MiniGPTConfig:
    """RTX 3090 优化的 small 配置 (~50M参数)
    适合中等规模训练，充分利用24GB显存
    """
    return MiniGPTConfig(
        vocab_size=32000,  # 增大词汇表
        hidden_size=768,   # 增大隐藏层
        num_hidden_layers=8,  # 增加层数
        num_attention_heads=12,
        intermediate_size=3072,  # 4倍关系
        max_position_embeddings=1024,
        dropout=0.1,
        attention_dropout=0.1,
        rms_norm_eps=1e-6,
        flash_attn=True,
        gradient_checkpointing=True,  # 节省显存
        max_generate_length=512,
        temperature=0.8,
        top_k=50,
        top_p=0.9
    )


def get_rtx3090_medium_config() -> MiniGPTConfig:
    """RTX 3090 优化的 medium 配置 (~200M参数)
    最大化利用RTX 3090的24GB显存
    """
    return MiniGPTConfig(
        vocab_size=32000,
        hidden_size=1024,
        num_hidden_layers=16,
        num_attention_heads=16,
        intermediate_size=4096,
        max_position_embeddings=2048,
        dropout=0.1,
        attention_dropout=0.1,
        rms_norm_eps=1e-6,
        flash_attn=True,
        gradient_checkpointing=True,
        max_generate_length=1024,
        temperature=0.8,
        top_k=50,
        top_p=0.9
    )


def get_rtx3090_large_config() -> MiniGPTConfig:
    """RTX 3090 优化的 large 配置 (~500M参数)
    需要使用梯度检查点和混合精度训练
    """
    return MiniGPTConfig(
        vocab_size=50000,
        hidden_size=1280,
        num_hidden_layers=20,
        num_attention_heads=20,
        intermediate_size=5120,
        max_position_embeddings=2048,
        dropout=0.1,
        attention_dropout=0.1,
        rms_norm_eps=1e-6,
        flash_attn=True,
        gradient_checkpointing=True,
        max_generate_length=1024,
        temperature=0.8,
        top_k=50,
        top_p=0.9
    )


def get_rtx3090_moe_config() -> MiniGPTConfig:
    """RTX 3090 优化的 MOE 配置
    使用专家混合模型提高参数效率
    """
    return MiniGPTConfig(
        vocab_size=32000,
        hidden_size=768,
        num_hidden_layers=8,
        num_attention_heads=12,
        intermediate_size=2048,  # MOE中单个专家较小
        max_position_embeddings=1024,
        dropout=0.1,
        attention_dropout=0.1,
        rms_norm_eps=1e-6,
        flash_attn=True,
        gradient_checkpointing=True,
        # MOE 特定配置
        use_moe=True,
        num_experts_per_tok=2,
        n_routed_experts=8,  # 增加专家数量
        n_shared_experts=2,
        scoring_func='softmax',
        aux_loss_alpha=0.1,
        seq_aux=True,
        norm_topk_prob=True,
        max_generate_length=512,
        temperature=0.8,
        top_k=50,
        top_p=0.9
    )


class RTX3090TrainingConfig:
    """RTX 3090 训练配置"""

    def __init__(self, model_config: MiniGPTConfig):
        self.model_config = model_config

        # 自动检测设备
        self.device = self._get_optimal_device()

        # 训练参数 - 针对RTX 3090优化
        self.batch_size = self._get_optimal_batch_size()
        self.learning_rate = 5e-4
        self.weight_decay = 0.01
        self.warmup_steps = 1000
        self.max_steps = 100000

        # 优化器配置
        self.optimizer_type = "adamw"
        self.beta1 = 0.9
        self.beta2 = 0.95
        self.eps = 1e-8

        # 混合精度训练 (RTX 3090 支持)
        self.use_mixed_precision = True
        self.fp16 = True

        # 梯度相关
        self.gradient_accumulation_steps = self._get_gradient_accumulation_steps()
        self.max_grad_norm = 1.0

        # 保存和日志
        self.save_steps = 5000
        self.logging_steps = 100
        self.eval_steps = 2000

        # 数据加载
        self.num_workers = 4  # RTX 3090 配套的CPU核心数
        self.pin_memory = True
        self.persistent_workers = True

        # CUDA优化
        self.compile_model = True  # PyTorch 2.0+ 编译优化
        self.channels_last = False  # 对于Transformer不建议

    def _get_optimal_device(self) -> str:
        """获取最优设备"""
        if torch.cuda.is_available():
            # 验证是否为RTX 3090
            gpu_name = torch.cuda.get_device_name(0)
            if "RTX 3090" in gpu_name or "3090" in gpu_name:
                print(f"检测到 RTX 3090: {gpu_name}")
                return "cuda"
            else:
                print(f"检测到GPU: {gpu_name}")
                return "cuda"
        else:
            print("未检测到CUDA，使用CPU")
            return "cpu"

    def _get_optimal_batch_size(self) -> int:
        """根据模型大小和显存获取最优批量大小"""
        hidden_size = self.model_config.hidden_size

        if hidden_size <= 256:
            return 64  # tiny模型
        elif hidden_size <= 768:
            return 32  # small模型
        elif hidden_size <= 1024:
            return 16  # medium模型
        else:
            return 8   # large模型

    def _get_gradient_accumulation_steps(self) -> int:
        """计算梯度累积步数以达到有效批量大小"""
        target_batch_size = 128  # 目标有效批量大小
        return max(1, target_batch_size // self.batch_size)

    def setup_cuda_optimizations(self):
        """设置CUDA优化"""
        if self.device == "cuda":
            # 启用TensorFloat-32 (RTX 3090支持)
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

            # 启用CuDNN benchmark (固定输入尺寸时有效)
            torch.backends.cudnn.benchmark = True

            # 优化CUDA内存管理
            torch.cuda.empty_cache()

            print("已启用RTX 3090 CUDA优化:")
            print(f"  - TensorFloat-32: {torch.backends.cuda.matmul.allow_tf32}")
            print(f"  - CuDNN Benchmark: {torch.backends.cudnn.benchmark}")
            print(f"  - 混合精度训练: {self.use_mixed_precision}")

    def get_memory_info(self) -> dict:
        """获取显存信息"""
        if self.device == "cuda":
            allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            cached = torch.cuda.memory_reserved() / 1024**3      # GB
            total = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB

            return {
                "allocated_gb": allocated,
                "cached_gb": cached,
                "total_gb": total,
                "free_gb": total - cached,
                "utilization": cached / total * 100
            }
        else:
            return {"device": "cpu", "message": "CPU模式，无显存信息"}


# 配置映射
RTX3090_CONFIG_MAPPING = {
    "tiny": get_rtx3090_tiny_config,
    "small": get_rtx3090_small_config,
    "medium": get_rtx3090_medium_config,
    "large": get_rtx3090_large_config,
    "moe": get_rtx3090_moe_config,
}


def get_rtx3090_config(config_name: str) -> MiniGPTConfig:
    """获取RTX 3090优化配置"""
    if config_name not in RTX3090_CONFIG_MAPPING:
        raise ValueError(f"未知的RTX 3090配置: {config_name}. 可用配置: {list(RTX3090_CONFIG_MAPPING.keys())}")

    return RTX3090_CONFIG_MAPPING[config_name]()


if __name__ == "__main__":
    # 测试RTX 3090配置
    print("RTX 3090 优化配置测试:")

    for config_name in RTX3090_CONFIG_MAPPING.keys():
        print(f"\n=== {config_name.upper()} ===")
        config = get_rtx3090_config(config_name)
        training_config = RTX3090TrainingConfig(config)

        print(f"模型参数: hidden_size={config.hidden_size}, layers={config.num_hidden_layers}")
        print(f"批量大小: {training_config.batch_size}")
        print(f"梯度累积: {training_config.gradient_accumulation_steps}")
        print(f"有效批量: {training_config.batch_size * training_config.gradient_accumulation_steps}")
        print(f"Flash Attention: {config.flash_attn}")
        print(f"梯度检查点: {config.gradient_checkpointing}")