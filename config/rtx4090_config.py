# -*- coding: utf-8 -*-
"""
RTX 4090 优化配置
针对RTX 4090 24GB显存和Ada Lovelace架构进行优化
相比RTX 3090，Ada架构具有更强的计算能力和更高效的内存带宽
"""
import os
import sys

# 添加项目根目录到 Python 路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import torch
from src.model.config import MiniGPTConfig


def get_rtx4090_tiny_config() -> MiniGPTConfig:
    """RTX 4090 优化的 tiny 配置 (~1M参数)
    适合快速测试和原型验证
    """
    return MiniGPTConfig(
        vocab_size=10000,
        hidden_size=256,
        num_hidden_layers=4,
        num_attention_heads=4,
        intermediate_size=1024,
        max_position_embeddings=512,
        dropout=0.1,
        attention_dropout=0.1,
        rms_norm_eps=1e-6,
        flash_attn=False,  # 使用自定义注意力实现
        gradient_checkpointing=False,  # tiny模型不需要
        max_generate_length=256,
        temperature=0.8,
        top_k=50,
        top_p=0.9
    )


def get_rtx4090_small_config() -> MiniGPTConfig:
    """RTX 4090 优化的 small 配置 (~25-30M参数)
    充分利用RTX 4090的24GB显存和Ada架构性能
    
    优化策略：
    - 相比RTX 3090配置，增加序列长度以充分利用显存
    - 使用梯度检查点节省显存
    - 启用混合精度训练
    - 使用GQA (分组查询注意力) 优化内存
    """
    return MiniGPTConfig(
        vocab_size=10000,  # 与training_config.py的SmallConfig保持一致
        hidden_size=288,   # 瘦长架构：较窄的宽度
        num_hidden_layers=18,  # 较深的层数
        num_attention_heads=9,  # 保持 hidden_size/num_heads = 32
        num_key_value_heads=3,  # GQA: 9个query头共享3个KV头
        intermediate_size=1152,  # 4倍hidden_size
        max_position_embeddings=1024,
        dropout=0.1,
        attention_dropout=0.1,
        rms_norm_eps=1e-6,
        use_gqa=True,  # 启用分组查询注意力
        flash_attn=False,  # 使用自定义注意力实现
        gradient_checkpointing=True,  # 节省显存
        max_generate_length=1024,
        temperature=0.8,
        top_k=50,
        top_p=0.9
    )


def get_rtx4090_small_30m_config() -> MiniGPTConfig:
    """RTX 4090 优化的 30M 参数小型模型配置
    针对较大的小型模型优化
    """
    return MiniGPTConfig(
        vocab_size=12000,
        hidden_size=384,
        num_hidden_layers=13,
        num_attention_heads=12,
        intermediate_size=1408,
        max_position_embeddings=2048,
        dropout=0.1,
        attention_dropout=0.1,
        rms_norm_eps=1e-6,
        flash_attn=False,
        gradient_checkpointing=True,
        max_generate_length=1024,
        temperature=0.8,
        top_k=50,
        top_p=0.9
    )


def get_rtx4090_medium_config() -> MiniGPTConfig:
    """RTX 4090 优化的 medium 配置 (~80M参数)
    最大化利用RTX 4090的Ada架构优势
    """
    return MiniGPTConfig(
        vocab_size=20000,
        hidden_size=512,
        num_hidden_layers=16,
        num_attention_heads=16,
        intermediate_size=1536,
        max_position_embeddings=2048,
        dropout=0.1,
        attention_dropout=0.1,
        rms_norm_eps=1e-6,
        flash_attn=False,
        gradient_checkpointing=True,
        max_generate_length=1024,
        temperature=0.8,
        top_k=50,
        top_p=0.9
    )


def get_rtx4090_foundation_config() -> MiniGPTConfig:
    """RTX 4090 优化的 foundation 配置 (~200M参数)
    中型规模训练，充分利用24GB显存
    """
    return MiniGPTConfig(
        vocab_size=32000,
        hidden_size=768,
        num_hidden_layers=24,
        num_attention_heads=16,
        intermediate_size=2688,
        max_position_embeddings=4096,
        dropout=0.1,
        attention_dropout=0.1,
        rms_norm_eps=1e-6,
        flash_attn=False,
        gradient_checkpointing=True,
        max_generate_length=2048,
        temperature=0.8,
        top_k=50,
        top_p=0.9
    )


def get_rtx4090_large_config() -> MiniGPTConfig:
    """RTX 4090 优化的 large 配置 (~350M参数)
    需要精细的显存管理和优化
    """
    return MiniGPTConfig(
        vocab_size=32000,
        hidden_size=768,
        num_hidden_layers=32,
        num_attention_heads=24,
        intermediate_size=3072,
        max_position_embeddings=4096,
        dropout=0.1,
        attention_dropout=0.1,
        rms_norm_eps=1e-6,
        flash_attn=False,
        gradient_checkpointing=True,
        max_generate_length=2048,
        temperature=0.8,
        top_k=50,
        top_p=0.9
    )


def get_rtx4090_moe_config() -> MiniGPTConfig:
    """RTX 4090 优化的 MOE 配置
    使用专家混合模型提高参数效率
    """
    return MiniGPTConfig(
        vocab_size=10000,
        hidden_size=384,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=1536,
        max_position_embeddings=1024,
        dropout=0.1,
        attention_dropout=0.1,
        rms_norm_eps=1e-6,
        flash_attn=False,
        gradient_checkpointing=True,
        # MOE 特定配置
        use_moe=True,
        num_experts_per_tok=2,
        n_routed_experts=4,
        n_shared_experts=1,
        scoring_func='softmax',
        aux_loss_alpha=0.1,
        seq_aux=True,
        norm_topk_prob=True,
        max_generate_length=1024,
        temperature=0.8,
        top_k=50,
        top_p=0.9
    )


class RTX4090TrainingConfig:
    """RTX 4090 训练配置
    
    RTX 4090 特点：
    - 24GB GDDR6X 显存
    - Ada Lovelace 架构
    - 更高的计算吞吐量
    - 更好的能效比
    - 支持 TensorFloat-32, 混合精度训练
    """

    def __init__(self, model_config: MiniGPTConfig):
        self.model_config = model_config

        # 自动检测设备
        self.device = self._get_optimal_device()

        # 训练参数 - 针对RTX 4090优化
        # RTX 4090 有更强的计算能力，可以使用稍大的batch size
        self.batch_size = self._get_optimal_batch_size()
        self.learning_rate = 3e-4
        self.weight_decay = 0.01
        self.warmup_steps = 2000
        self.max_steps = 50000

        # 优化器配置
        self.optimizer_type = "adamw"
        self.beta1 = 0.9
        self.beta2 = 0.95
        self.eps = 1e-8

        # 混合精度训练 (RTX 4090 支持且性能更好)
        self.use_mixed_precision = True
        self.fp16 = True

        # 梯度相关
        self.gradient_accumulation_steps = self._get_gradient_accumulation_steps()
        self.max_grad_norm = 1.0

        # 保存和日志
        self.save_steps = 2000
        self.logging_steps = 100
        self.eval_steps = 1000

        # 数据加载 - RTX 4090 通常配备更强的CPU
        self.num_workers = 8  # 增加worker数量
        self.prefetch_factor = 4  # 预取更多batch
        self.pin_memory = True
        self.persistent_workers = True

        # CUDA优化
        self.compile_model = True  # PyTorch 2.0+ 编译优化
        self.channels_last = False  # 对于Transformer不建议

    def _get_optimal_device(self) -> str:
        """获取最优设备"""
        if torch.cuda.is_available():
            # 验证是否为RTX 4090
            gpu_name = torch.cuda.get_device_name(0)
            print(f"检测到GPU: {gpu_name}")
            
            if "RTX 4090" in gpu_name or "4090" in gpu_name:
                print("✓ RTX 4090 检测成功，已启用Ada架构优化")
            elif "40" in gpu_name and "90" in gpu_name:
                print("✓ 可能是RTX 4090，已启用优化")
            else:
                print(f"⚠ 配置针对RTX 4090优化，当前GPU: {gpu_name}")
                
            return "cuda"
        else:
            print("未检测到CUDA，使用CPU")
            return "cpu"

    def _get_optimal_batch_size(self) -> int:
        """根据模型大小和RTX 4090显存获取最优批量大小
        
        RTX 4090 相比 RTX 3090 有相似的显存(24GB)，
        但Ada架构的内存控制器更高效，可以略微提高batch size
        """
        hidden_size = self.model_config.hidden_size
        num_layers = self.model_config.num_hidden_layers

        # 根据模型大小动态调整
        if hidden_size <= 256:
            return 64  # tiny模型
        elif hidden_size <= 384:
            # small模型 (288-384维，18层左右)
            if num_layers >= 18:
                return 16  # 深层网络，降低batch size
            else:
                return 24  # 浅层网络，可以增加batch size
        elif hidden_size <= 512:
            return 12  # medium模型
        elif hidden_size <= 768:
            return 8   # foundation模型
        else:
            return 4   # large模型

    def _get_gradient_accumulation_steps(self) -> int:
        """计算梯度累积步数以达到有效批量大小
        
        目标有效批量大小：
        - small模型: 128-192
        - medium模型: 128
        - large模型: 128
        """
        hidden_size = self.model_config.hidden_size
        
        if hidden_size <= 384:
            # small模型，目标有效batch size = 144
            target_batch_size = 144
        else:
            # medium/large模型，目标有效batch size = 128
            target_batch_size = 128
            
        return max(1, target_batch_size // self.batch_size)

    def setup_cuda_optimizations(self):
        """设置CUDA优化 - RTX 4090 Ada架构优化"""
        if self.device == "cuda":
            # 启用TensorFloat-32 (Ada架构性能更好)
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

            # 启用CuDNN benchmark (固定输入尺寸时有效)
            torch.backends.cudnn.benchmark = True
            
            # CuDNN deterministic (可选，用于可重复性)
            # torch.backends.cudnn.deterministic = False  # 保持非确定性以获得更好性能

            # 优化CUDA内存管理
            torch.cuda.empty_cache()

            print("\n" + "="*60)
            print("RTX 4090 CUDA 优化已启用:")
            print("="*60)
            print(f"  ✓ TensorFloat-32 (matmul):  {torch.backends.cuda.matmul.allow_tf32}")
            print(f"  ✓ TensorFloat-32 (cudnn):   {torch.backends.cudnn.allow_tf32}")
            print(f"  ✓ CuDNN Benchmark:          {torch.backends.cudnn.benchmark}")
            print(f"  ✓ 混合精度训练 (FP16):      {self.use_mixed_precision}")
            print(f"  ✓ 模型编译优化:              {self.compile_model}")
            print(f"  ✓ 梯度检查点:                {self.model_config.gradient_checkpointing}")
            print("="*60 + "\n")

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

    def print_config_summary(self):
        """打印配置摘要"""
        print("\n" + "="*60)
        print("RTX 4090 训练配置摘要")
        print("="*60)
        print(f"模型配置:")
        print(f"  - Hidden Size:        {self.model_config.hidden_size}")
        print(f"  - Layers:             {self.model_config.num_hidden_layers}")
        print(f"  - Attention Heads:    {self.model_config.num_attention_heads}")
        print(f"  - FFN Size:           {self.model_config.intermediate_size}")
        print(f"  - Max Seq Length:     {self.model_config.max_position_embeddings}")
        print(f"  - Vocab Size:         {self.model_config.vocab_size}")
        
        print(f"\n训练参数:")
        print(f"  - Batch Size:         {self.batch_size}")
        print(f"  - Grad Accumulation:  {self.gradient_accumulation_steps}")
        print(f"  - Effective Batch:    {self.batch_size * self.gradient_accumulation_steps}")
        print(f"  - Learning Rate:      {self.learning_rate}")
        print(f"  - Warmup Steps:       {self.warmup_steps}")
        print(f"  - Max Steps:          {self.max_steps}")
        
        print(f"\n优化设置:")
        print(f"  - Mixed Precision:    {self.use_mixed_precision}")
        print(f"  - Gradient Checkpoint: {self.model_config.gradient_checkpointing}")
        print(f"  - Model Compile:      {self.compile_model}")
        print(f"  - Num Workers:        {self.num_workers}")
        print(f"  - Prefetch Factor:    {self.prefetch_factor}")
        
        if self.device == "cuda":
            mem_info = self.get_memory_info()
            print(f"\n显存信息:")
            print(f"  - Total:              {mem_info['total_gb']:.1f} GB")
            print(f"  - Allocated:          {mem_info['allocated_gb']:.2f} GB")
            print(f"  - Cached:             {mem_info['cached_gb']:.2f} GB")
            print(f"  - Free:               {mem_info['free_gb']:.2f} GB")
        
        print("="*60 + "\n")


# 配置映射
RTX4090_CONFIG_MAPPING = {
    "tiny": get_rtx4090_tiny_config,
    "small": get_rtx4090_small_config,
    "small_30m": get_rtx4090_small_30m_config,
    "medium": get_rtx4090_medium_config,
    "foundation": get_rtx4090_foundation_config,
    "large": get_rtx4090_large_config,
    "moe": get_rtx4090_moe_config,
}


def get_rtx4090_config(config_name: str) -> MiniGPTConfig:
    """获取RTX 4090优化配置
    
    Args:
        config_name: 配置名称 (tiny, small, small_30m, medium, foundation, large, moe)
        
    Returns:
        MiniGPTConfig: 优化后的模型配置
        
    Example:
        >>> config = get_rtx4090_config("small")
        >>> training_config = RTX4090TrainingConfig(config)
        >>> training_config.setup_cuda_optimizations()
        >>> training_config.print_config_summary()
    """
    if config_name not in RTX4090_CONFIG_MAPPING:
        raise ValueError(
            f"未知的RTX 4090配置: {config_name}. "
            f"可用配置: {list(RTX4090_CONFIG_MAPPING.keys())}"
        )

    return RTX4090_CONFIG_MAPPING[config_name]()


if __name__ == "__main__":
    # 测试RTX 4090配置
    print("\n" + "="*60)
    print("RTX 4090 优化配置测试")
    print("="*60)

    for config_name in RTX4090_CONFIG_MAPPING.keys():
        print(f"\n{'='*60}")
        print(f"{config_name.upper()} 配置")
        print(f"{'='*60}")
        
        config = get_rtx4090_config(config_name)
        training_config = RTX4090TrainingConfig(config)
        training_config.setup_cuda_optimizations()
        training_config.print_config_summary()

