"""
训练配置文件
包含所有训练相关的超参数设置
"""
from dataclasses import dataclass
from typing import Optional, List


@dataclass
class ModelConfig:
    """模型配置"""
    vocab_size: int = 30000  # 增加词汇表大小以更好支持中文
    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 6
    d_ff: int = 2048
    max_seq_len: int = 1024
    dropout: float = 0.1
    
    # 模型大小预设
    model_size: str = "small"  # tiny, small, medium, large


@dataclass
class TokenizerConfig:
    """分词器配置"""
    vocab_size: int = 30000  # 增加词汇表大小以更好支持中文
    min_frequency: int = 1  # 降低中文字符的最小频率要求
    special_tokens: List[str] = None
    
    def __post_init__(self):
        if self.special_tokens is None:
            self.special_tokens = ['<PAD>', '<UNK>', '<BOS>', '<EOS>']


@dataclass
class DataConfig:
    """数据配置"""
    data_dir: str = "data/dataset/minimind_dataset"
    train_files: List[str] = None
    val_split: float = 0.1
    max_seq_len: int = 512
    batch_size: int = 32
    num_workers: int = 4
    shuffle: bool = True
    
    def __post_init__(self):
        if self.train_files is None:
            self.train_files = ["sft_mini_512.jsonl"]


@dataclass
class PretrainConfig:
    """预训练配置"""
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    max_steps: int = 50000
    save_steps: int = 5000
    eval_steps: int = 1000
    gradient_accumulation_steps: int = 4
    max_grad_norm: float = 1.0
    
    # 学习率调度
    lr_scheduler: str = "cosine"  # linear, cosine, constant
    
    # 早停
    early_stopping_patience: int = 5
    early_stopping_threshold: float = 0.001


@dataclass
class SFTConfig:
    """监督微调配置"""
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    warmup_steps: int = 500
    max_epochs: int = 10
    save_steps: int = 1000
    eval_steps: int = 500
    gradient_accumulation_steps: int = 2
    max_grad_norm: float = 1.0
    
    # SFT特定参数
    response_only_loss: bool = True  # 只对回复部分计算损失
    
    # LoRA配置
    use_lora: bool = False
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    lora_target_modules: List[str] = None
    
    def __post_init__(self):
        if self.lora_target_modules is None:
            self.lora_target_modules = ["w_q", "w_k", "w_v", "w_o"]


@dataclass
class DPOConfig:
    """DPO训练配置"""
    learning_rate: float = 1e-5
    weight_decay: float = 0.01
    max_epochs: int = 5
    beta: float = 0.1  # DPO温度参数
    
    # 参考模型
    reference_model_path: str = ""
    
    # 损失权重
    sft_loss_weight: float = 0.0  # 是否加入SFT损失


@dataclass
class RLConfig:
    """强化学习配置（PPO等）"""
    learning_rate: float = 1e-5
    ppo_epochs: int = 4
    mini_batch_size: int = 8
    clip_range: float = 0.2
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.01
    
    # 奖励模型
    reward_model_path: str = ""
    
    # KL散度惩罚
    kl_coef: float = 0.1
    target_kl: float = 0.1
    
    # 生成参数
    generation_config: dict = None
    
    def __post_init__(self):
        if self.generation_config is None:
            self.generation_config = {
                "max_length": 200,
                "temperature": 0.7,
                "top_k": 50,
                "top_p": 0.9
            }


@dataclass
class OptimizationConfig:
    """优化器配置"""
    optimizer: str = "adamw"  # adamw, adam, sgd
    
    # AdamW参数
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    
    # 梯度裁剪
    max_grad_norm: float = 1.0
    
    # 混合精度训练
    use_fp16: bool = False
    use_bf16: bool = False


@dataclass
class TrainingConfig:
    """完整训练配置"""
    # 基础配置
    model: ModelConfig = None
    tokenizer: TokenizerConfig = None
    data: DataConfig = None
    optimization: OptimizationConfig = None
    
    # 训练阶段配置
    pretrain: PretrainConfig = None
    sft: SFTConfig = None
    dpo: DPOConfig = None
    rl: RLConfig = None
    
    # 环境配置
    device: str = "cpu"  # cpu, cuda, mps
    seed: int = 42
    output_dir: str = "checkpoints"
    logging_dir: str = "logs"
    
    # 保存和加载
    save_total_limit: int = 3
    save_strategy: str = "steps"  # steps, epoch
    load_best_model_at_end: bool = True
    
    # 评估
    evaluation_strategy: str = "steps"  # steps, epoch, no
    metric_for_best_model: str = "loss"
    greater_is_better: bool = False
    
    # 日志
    logging_steps: int = 100
    report_to: List[str] = None  # tensorboard, wandb
    
    # 分布式训练
    use_ddp: bool = False
    world_size: int = 1
    local_rank: int = -1
    
    def __post_init__(self):
        if self.model is None:
            self.model = ModelConfig()
        if self.tokenizer is None:
            self.tokenizer = TokenizerConfig()
        if self.data is None:
            self.data = DataConfig()
        if self.optimization is None:
            self.optimization = OptimizationConfig()
        if self.pretrain is None:
            self.pretrain = PretrainConfig()
        if self.sft is None:
            self.sft = SFTConfig()
        if self.dpo is None:
            self.dpo = DPOConfig()
        if self.rl is None:
            self.rl = RLConfig()
        if self.report_to is None:
            self.report_to = []


# 预定义配置
def get_tiny_config() -> TrainingConfig:
    """获取超小模型配置（用于快速测试）"""
    config = TrainingConfig()
    
    # 模型配置
    config.model.d_model = 128
    config.model.n_heads = 2
    config.model.n_layers = 4
    config.model.d_ff = 512
    config.model.vocab_size = 5000
    config.model.model_size = "tiny"
    
    # 数据配置
    config.data.batch_size = 16
    config.data.max_seq_len = 256
    
    # 训练配置
    config.pretrain.max_steps = 1000
    config.sft.max_epochs = 3
    
    return config


def get_small_config() -> TrainingConfig:
    """获取小模型配置（推荐用于学习）"""
    config = TrainingConfig()
    
    # 模型配置
    config.model.d_model = 512
    config.model.n_heads = 8
    config.model.n_layers = 6
    config.model.d_ff = 2048
    config.model.vocab_size = 10000
    config.model.model_size = "small"
    
    # 数据配置
    config.data.batch_size = 32
    config.data.max_seq_len = 512
    
    return config


def get_medium_config() -> TrainingConfig:
    """获取中等模型配置"""
    config = TrainingConfig()
    
    # 模型配置
    config.model.d_model = 768
    config.model.n_heads = 12
    config.model.n_layers = 12
    config.model.d_ff = 3072
    config.model.vocab_size = 15000
    config.model.model_size = "medium"
    
    # 数据配置
    config.data.batch_size = 16
    config.data.max_seq_len = 1024
    
    return config


def save_config(config: TrainingConfig, path: str):
    """保存配置到文件"""
    import json
    import dataclasses
    
    def asdict_factory(data):
        def convert_value(obj):
            if isinstance(obj, list):
                return [convert_value(item) for item in obj]
            elif dataclasses.is_dataclass(obj):
                return dataclasses.asdict(obj, dict_factory=asdict_factory)
            else:
                return obj
        return dict(data)
    
    config_dict = dataclasses.asdict(config, dict_factory=asdict_factory)
    
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(config_dict, f, indent=2, ensure_ascii=False)


def load_config(path: str) -> TrainingConfig:
    """从文件加载配置"""
    import json
    
    with open(path, 'r', encoding='utf-8') as f:
        config_dict = json.load(f)
    
    # 这里需要实现从字典重建配置对象的逻辑
    # 由于dataclass不直接支持从字典构建，需要自定义实现
    return TrainingConfig()


if __name__ == "__main__":
    # 测试配置
    config = get_small_config()
    print(f"模型参数量估计: ~{config.model.d_model * config.model.n_layers * 12}M")
    
    # 保存配置示例
    save_config(config, "config/small_model_config.json")
    print("配置文件已保存")