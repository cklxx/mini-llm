"""
训练配置文件 - NVIDIA GPU 优化版本
支持 PyTorch 2.4 和 NVIDIA GPU 自动检测优化
"""
import os
import torch
import subprocess


def get_gpu_info():
    """获取 GPU 详细信息"""
    if not torch.cuda.is_available():
        return None

    gpu_info = {
        'count': torch.cuda.device_count(),
        'current_device': torch.cuda.current_device(),
        'devices': []
    }

    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        device_info = {
            'id': i,
            'name': props.name,
            'memory_total': props.total_memory / 1024**3,  # GB
            'memory_allocated': torch.cuda.memory_allocated(i) / 1024**3,  # GB
            'memory_reserved': torch.cuda.memory_reserved(i) / 1024**3,  # GB
            'compute_capability': f"{props.major}.{props.minor}",
            'multi_processor_count': props.multi_processor_count
        }
        gpu_info['devices'].append(device_info)

    return gpu_info


def setup_cuda_optimizations():
    """设置 CUDA 优化"""
    if torch.cuda.is_available():
        # 启用 TensorFloat-32 (适用于 Ampere 架构)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        # 启用 CuDNN benchmark
        torch.backends.cudnn.benchmark = True

        # 设置 CUDA 内存分配策略
        torch.cuda.empty_cache()

        print("已启用 CUDA 优化:")
        print(f"  - TensorFloat-32: {torch.backends.cuda.matmul.allow_tf32}")
        print(f"  - CuDNN Benchmark: {torch.backends.cudnn.benchmark}")
        return True
    return False


def get_device():
    """自动检测最佳设备并优化配置"""
    if torch.cuda.is_available():
        gpu_info = get_gpu_info()
        setup_cuda_optimizations()

        primary_gpu = gpu_info['devices'][0]
        print(f"检测到 NVIDIA GPU: {primary_gpu['name']}")
        print(f"显存总量: {primary_gpu['memory_total']:.1f} GB")
        print(f"计算能力: {primary_gpu['compute_capability']}")
        print(f"多处理器数量: {primary_gpu['multi_processor_count']}")

        return "cuda", gpu_info
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        print("检测到 Apple Silicon GPU (MPS)")
        return "mps", None
    else:
        print("使用 CPU")
        return "cpu", None


class BaseConfig:
    """基础配置类 - NVIDIA GPU 优化"""
    def __init__(self):
        # 基础路径
        self.project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.data_dir = os.path.join(self.project_root, "data", "dataset", "minimind_dataset")
        self.checkpoint_dir = os.path.join(self.project_root, "checkpoints")
        self.log_dir = os.path.join(self.project_root, "logs")

        # 设备配置
        self.device, self.gpu_info = get_device()

        # 数据集路径
        self.pretrain_data_path = os.path.join(self.data_dir, "pretrain_hq.jsonl")
        self.sft_data_path = os.path.join(self.data_dir, "sft_mini_512.jsonl")
        self.dpo_data_path = os.path.join(self.data_dir, "dpo.jsonl")

        # 身份认同和能力增强数据集路径
        self.alex_identity_data_path = os.path.join(self.data_dir, "alex_identity.jsonl")
        self.ultra_think_data_path = os.path.join(self.data_dir, "ultra_think.jsonl")
        self.lora_identity_data_path = os.path.join(self.data_dir, "lora_identity.jsonl")

        # NVIDIA GPU 优化设置
        self.mixed_precision = self.device == "cuda"
        self.compile_model = self.device == "cuda"  # PyTorch 2.4 编译优化
        self.gradient_checkpointing = True
        self.flash_attention = self.device == "cuda"

        # 数据加载优化
        self.num_workers = 4 if self.device == "cuda" else 2
        self.pin_memory = self.device == "cuda"
        self.persistent_workers = True

        # 创建必要目录
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)


class TinyConfig(BaseConfig):
    """超小模型配置 (~1M参数) - GPU 优化"""
    def __init__(self):
        super().__init__()

        # 模型参数
        self.vocab_size = 10000
        self.d_model = 256
        self.n_heads = 4
        self.n_layers = 4
        self.d_ff = 1024
        self.max_seq_len = 512
        self.dropout = 0.1

        # GPU 优化训练参数
        if self.device == "cuda":
            gpu_memory = self.gpu_info['devices'][0]['memory_total'] if self.gpu_info else 8
            self.batch_size = 64 if gpu_memory >= 12 else 32
        else:
            self.batch_size = 16

        self.learning_rate = 1e-3
        self.weight_decay = 0.01
        self.warmup_steps = 500
        self.max_steps = 10000
        self.eval_steps = 1000
        self.save_steps = 2000

        # 梯度累积
        self.gradient_accumulation_steps = max(1, 128 // self.batch_size)

        # 优化器
        self.optimizer = "adamw"
        self.beta1 = 0.9
        self.beta2 = 0.95
        self.eps = 1e-8

        # 生成参数
        self.max_generate_length = 256
        self.temperature = 0.8
        self.top_k = 50
        self.top_p = 0.9


class SmallConfig(BaseConfig):
    """小模型配置 (~50M参数) - GPU 优化"""
    def __init__(self):
        super().__init__()

        # 模型参数
        self.vocab_size = 32000
        self.d_model = 768
        self.n_heads = 12
        self.n_layers = 8
        self.d_ff = 3072
        self.max_seq_len = 1024
        self.dropout = 0.1

        # GPU 优化训练参数
        if self.device == "cuda":
            gpu_memory = self.gpu_info['devices'][0]['memory_total'] if self.gpu_info else 8
            if gpu_memory >= 24:  # RTX 3090/4090
                self.batch_size = 32
            elif gpu_memory >= 12:  # RTX 3060Ti/4060Ti
                self.batch_size = 16
            else:
                self.batch_size = 8
        else:
            self.batch_size = 8

        self.learning_rate = 5e-4
        self.weight_decay = 0.01
        self.warmup_steps = 2000
        self.max_steps = 50000
        self.eval_steps = 2000
        self.save_steps = 5000

        # 梯度累积
        self.gradient_accumulation_steps = max(1, 128 // self.batch_size)

        # 优化器
        self.optimizer = "adamw"
        self.beta1 = 0.9
        self.beta2 = 0.95
        self.eps = 1e-8

        # 生成参数
        self.max_generate_length = 512
        self.temperature = 0.8
        self.top_k = 50
        self.top_p = 0.9


class MediumConfig(BaseConfig):
    """中等模型配置 (~200M参数) - GPU 优化"""
    def __init__(self):
        super().__init__()

        # 模型参数
        self.vocab_size = 32000
        self.d_model = 1024
        self.n_heads = 16
        self.n_layers = 16
        self.d_ff = 4096
        self.max_seq_len = 2048
        self.dropout = 0.1

        # GPU 优化训练参数
        if self.device == "cuda":
            gpu_memory = self.gpu_info['devices'][0]['memory_total'] if self.gpu_info else 8
            if gpu_memory >= 24:  # RTX 3090/4090
                self.batch_size = 16
            elif gpu_memory >= 12:  # RTX 3060Ti/4060Ti
                self.batch_size = 8
            else:
                self.batch_size = 4
        else:
            self.batch_size = 4

        self.learning_rate = 3e-4
        self.weight_decay = 0.01
        self.warmup_steps = 4000
        self.max_steps = 100000
        self.eval_steps = 2000
        self.save_steps = 5000

        # 梯度累积
        self.gradient_accumulation_steps = max(1, 128 // self.batch_size)

        # 优化器
        self.optimizer = "adamw"
        self.beta1 = 0.9
        self.beta2 = 0.95
        self.eps = 1e-8

        # 生成参数
        self.max_generate_length = 1024
        self.temperature = 0.8
        self.top_k = 50
        self.top_p = 0.9


class LargeConfig(BaseConfig):
    """大模型配置 (~500M参数) - 高端 GPU 优化"""
    def __init__(self):
        super().__init__()

        # 模型参数
        self.vocab_size = 50000
        self.d_model = 1280
        self.n_heads = 20
        self.n_layers = 24
        self.d_ff = 5120
        self.max_seq_len = 2048
        self.dropout = 0.1

        # 只在高端 GPU 上运行
        if self.device == "cuda":
            gpu_memory = self.gpu_info['devices'][0]['memory_total'] if self.gpu_info else 8
            if gpu_memory >= 24:  # RTX 3090/4090
                self.batch_size = 8
            elif gpu_memory >= 16:  # RTX 4080
                self.batch_size = 4
            else:
                self.batch_size = 2
                print("警告: 显存不足，建议使用更小的模型配置")
        else:
            self.batch_size = 1
            print("警告: 大模型建议使用 CUDA GPU")

        self.learning_rate = 2e-4
        self.weight_decay = 0.01
        self.warmup_steps = 8000
        self.max_steps = 200000
        self.eval_steps = 5000
        self.save_steps = 10000

        # 梯度累积
        self.gradient_accumulation_steps = max(1, 128 // self.batch_size)

        # 强制启用内存优化
        self.gradient_checkpointing = True

        # 优化器
        self.optimizer = "adamw"
        self.beta1 = 0.9
        self.beta2 = 0.95
        self.eps = 1e-8

        # 生成参数
        self.max_generate_length = 1024
        self.temperature = 0.8
        self.top_k = 50
        self.top_p = 0.9


def get_config(model_size="small"):
    """获取指定大小的配置"""
    configs = {
        "tiny": TinyConfig,
        "small": SmallConfig,
        "medium": MediumConfig,
        "large": LargeConfig
    }

    if model_size not in configs:
        raise ValueError(f"不支持的模型大小: {model_size}. 支持的大小: {list(configs.keys())}")

    config = configs[model_size]()

    # 打印配置信息
    print(f"\n=== {model_size.upper()} 模型配置 ===")
    print(f"设备: {config.device}")
    if config.gpu_info:
        gpu = config.gpu_info['devices'][0]
        print(f"GPU: {gpu['name']} ({gpu['memory_total']:.1f} GB)")
    print(f"批量大小: {config.batch_size}")
    print(f"梯度累积: {config.gradient_accumulation_steps}")
    print(f"有效批量: {config.batch_size * config.gradient_accumulation_steps}")
    print(f"混合精度: {config.mixed_precision}")
    print(f"模型编译: {config.compile_model}")
    print(f"Flash Attention: {config.flash_attention}")
    print(f"梯度检查点: {config.gradient_checkpointing}")

    return config


def get_tiny_config():
    return get_config("tiny")


def get_small_config():
    return get_config("small")


def get_medium_config():
    return get_config("medium")


def get_large_config():
    return get_config("large")


if __name__ == "__main__":
    # 测试所有配置
    configs = ["tiny", "small", "medium", "large"]

    for config_name in configs:
        print(f"\n{'='*50}")
        config = get_config(config_name)
        print(f"配置测试完成: {config_name}")