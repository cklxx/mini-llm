#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Mac优化训练配置
针对Mac电脑优化，防止系统卡死，使用最小数据集快速验证智能效果
"""
import os
import psutil
from dataclasses import dataclass
from typing import Optional, List
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config.training_config import TrainingConfig, ModelConfig, TokenizerConfig, DataConfig, PretrainConfig, OptimizationConfig


@dataclass
class MacResourceConfig:
    """Mac资源限制配置"""
    max_cpu_percent: float = 85.0  # 最大CPU使用率
    max_memory_percent: float = 85.0  # 最大内存使用率
    max_threads: int = 4  # 最大线程数
    batch_size_limit: int = 8  # 批次大小限制
    enable_monitoring: bool = True  # 启用资源监控
    monitoring_interval: int = 5  # 监控间隔（秒）
    auto_adjust: bool = True  # 自动调整参数


def get_mac_tiny_config() -> TrainingConfig:
    """获取Mac超小模型配置（最小参数，快速验证）"""
    config = TrainingConfig()

    # 超小模型配置 - 减少参数量
    config.model.d_model = 64          # 从512降到64
    config.model.n_heads = 2           # 从8降到2
    config.model.n_layers = 2          # 从6降到2
    config.model.d_ff = 256            # 从2048降到256
    config.model.vocab_size = 2000     # 从30000降到2000
    config.model.max_seq_len = 128     # 从1024降到128
    config.model.model_size = "tiny"   # 超小型

    # 分词器配置
    config.tokenizer.vocab_size = 2000
    config.tokenizer.min_frequency = 1

    # 数据配置 - 使用200条数据集
    config.data.train_files = ["pretrain_200.jsonl"]  # 使用200条数据
    config.data.batch_size = 4         # 小批次
    config.data.max_seq_len = 128      # 短序列
    config.data.num_workers = 2        # 少线程

    # 预训练配置 - 快速验证
    config.pretrain.learning_rate = 1e-3  # 提高学习率加速收敛
    config.pretrain.max_steps = 500       # 增加到500步以适应200条数据
    config.pretrain.save_steps = 50       # 频繁保存
    config.pretrain.eval_steps = 25       # 频繁评估
    config.pretrain.warmup_steps = 20     # 少量预热
    config.pretrain.gradient_accumulation_steps = 2

    # 优化器配置
    config.optimization.use_fp16 = False  # 避免精度问题
    config.optimization.max_grad_norm = 0.5  # 更严格梯度裁剪

    # 输出配置
    config.output_dir = "checkpoints/mac_tiny"
    config.logging_steps = 10
    config.save_total_limit = 2  # 只保存2个检查点

    # 设备配置
    config.device = "cpu"  # 强制使用CPU，避免GPU过热

    return config


def get_mac_small_config() -> TrainingConfig:
    """获取Mac小模型配置（平衡性能和智能）"""
    config = TrainingConfig()

    # 小模型配置
    config.model.d_model = 128
    config.model.n_heads = 4
    config.model.n_layers = 4
    config.model.d_ff = 512
    config.model.vocab_size = 5000
    config.model.max_seq_len = 256
    config.model.model_size = "micro"

    # 分词器配置
    config.tokenizer.vocab_size = 5000

    # 数据配置 - 使用200条数据集
    config.data.train_files = ["pretrain_200.jsonl"]  # 200条数据
    config.data.batch_size = 8
    config.data.max_seq_len = 256
    config.data.num_workers = 2

    # 预训练配置
    config.pretrain.learning_rate = 5e-4
    config.pretrain.max_steps = 1500  # 增加训练步数
    config.pretrain.save_steps = 200
    config.pretrain.eval_steps = 100
    config.pretrain.warmup_steps = 50
    config.pretrain.gradient_accumulation_steps = 4

    # 优化器配置
    config.optimization.use_fp16 = False

    # 输出配置
    config.output_dir = "checkpoints/mac_small"
    config.logging_steps = 20
    config.save_total_limit = 3

    # 根据系统自动选择设备
    import torch
    if torch.backends.mps.is_available():
        config.device = "mps"
    else:
        config.device = "cpu"

    return config


def get_mac_medium_config() -> TrainingConfig:
    """获取Mac中型模型配置（更大的维度和层数）"""
    config = TrainingConfig()

    # 中型模型配置 - 更大的模型
    config.model.d_model = 640         # 比small(512)更大
    config.model.n_heads = 10          # 10个注意力头
    config.model.n_layers = 10         # 10层transformer（比small的6层多）
    config.model.d_ff = 2560           # 更大的前馈网络
    config.model.vocab_size = 10000    # 更大的词汇表
    config.model.max_seq_len = 512     # 更长序列
    config.model.model_size = "medium" # 中型模型

    # 分词器配置
    config.tokenizer.vocab_size = 10000
    config.tokenizer.min_frequency = 2

    # 数据配置 - 使用更多数据，但调整批次大小应对内存压力
    config.data.train_files = ["pretrain_200.jsonl", "pretrain_hq.jsonl"]
    config.data.batch_size = 2         # 大幅减小批次大小
    config.data.max_seq_len = 512
    config.data.num_workers = 2

    # 预训练配置 - 考虑到更大模型需要更多训练
    config.pretrain.learning_rate = 1e-4  # 降低学习率，大模型需要更小学习率
    config.pretrain.max_steps = 4000      # 增加训练步数
    config.pretrain.save_steps = 400      # 保存频率
    config.pretrain.eval_steps = 200      # 评估频率
    config.pretrain.warmup_steps = 200    # 更多预热步数
    config.pretrain.gradient_accumulation_steps = 12  # 大幅增加梯度累积，补偿小批次

    # 优化器配置
    config.optimization.use_fp16 = False
    config.optimization.max_grad_norm = 0.5  # 更严格的梯度裁剪

    # 输出配置
    config.output_dir = "checkpoints/mac_medium"
    config.logging_steps = 40
    config.save_total_limit = 3  # 减少保存的检查点数量

    # 根据系统自动选择设备
    import torch
    if torch.backends.mps.is_available():
        config.device = "mps"
    else:
        config.device = "cpu"

    return config


class MacResourceMonitor:
    """Mac资源监控器"""

    def __init__(self, config: MacResourceConfig):
        self.config = config
        self.should_stop = False

    def check_resources(self) -> dict:
        """检查当前资源使用情况"""
        cpu_percent = psutil.cpu_percent(interval=1)
        memory_percent = psutil.virtual_memory().percent

        return {
            'cpu_percent': cpu_percent,
            'memory_percent': memory_percent,
            'cpu_limit_exceeded': cpu_percent > self.config.max_cpu_percent,
            'memory_limit_exceeded': memory_percent > self.config.max_memory_percent
        }

    def should_pause_training(self) -> bool:
        """判断是否应该暂停训练"""
        if not self.config.enable_monitoring:
            return False

        resources = self.check_resources()
        return resources['cpu_limit_exceeded'] or resources['memory_limit_exceeded']

    def get_recommended_batch_size(self, current_batch_size: int) -> int:
        """根据资源使用情况推荐批次大小"""
        if not self.config.auto_adjust:
            return current_batch_size

        resources = self.check_resources()

        if resources['memory_percent'] > 80:
            return max(1, current_batch_size // 2)
        elif resources['memory_percent'] < 40:
            return min(self.config.batch_size_limit, current_batch_size + 1)
        else:
            return current_batch_size


def estimate_model_size(config: TrainingConfig) -> dict:
    """估算模型大小和内存需求"""
    # 简单估算模型参数量
    d_model = config.model.d_model
    n_layers = config.model.n_layers
    vocab_size = config.model.vocab_size
    d_ff = config.model.d_ff

    # Transformer参数估算
    embedding_params = vocab_size * d_model
    attention_params = n_layers * 4 * d_model * d_model  # Q,K,V,O
    ffn_params = n_layers * 2 * d_model * d_ff  # 两个线性层

    total_params = embedding_params + attention_params + ffn_params

    # 内存估算（假设float32，4字节）
    model_memory_mb = total_params * 4 / (1024 * 1024)
    training_memory_mb = model_memory_mb * 3  # 模型 + 梯度 + 优化器状态

    return {
        'total_params': total_params,
        'model_memory_mb': model_memory_mb,
        'training_memory_mb': training_memory_mb,
        'recommended_batch_size': max(1, min(32, int(4000 / training_memory_mb)))
    }


def validate_config_for_mac(config: TrainingConfig) -> List[str]:
    """验证配置是否适合Mac环境"""
    warnings = []

    # 检查模型大小
    model_info = estimate_model_size(config)
    if model_info['training_memory_mb'] > 4000:  # 4GB限制
        warnings.append(f"模型训练内存需求过高: {model_info['training_memory_mb']:.1f}MB")

    # 检查批次大小
    if config.data.batch_size > 16:
        warnings.append(f"批次大小过大: {config.data.batch_size}，建议≤16")

    # 检查序列长度
    if config.data.max_seq_len > 512:
        warnings.append(f"序列长度过长: {config.data.max_seq_len}，建议≤512")

    # 检查线程数
    if config.data.num_workers > 4:
        warnings.append(f"工作线程过多: {config.data.num_workers}，建议≤4")

    return warnings


def get_system_info() -> dict:
    """获取系统信息"""
    import platform

    return {
        'platform': platform.platform(),
        'cpu_count': psutil.cpu_count(),
        'memory_gb': psutil.virtual_memory().total / (1024**3),
        'available_memory_gb': psutil.virtual_memory().available / (1024**3)
    }


if __name__ == "__main__":
    # 测试配置
    print("=== Mac优化配置测试 ===")

    # 系统信息
    sys_info = get_system_info()
    print(f"系统信息: {sys_info}")

    # 测试tiny配置
    tiny_config = get_mac_tiny_config()
    tiny_info = estimate_model_size(tiny_config)
    print(f"\nTiny模型信息:")
    print(f"  参数量: {tiny_info['total_params']:,}")
    print(f"  内存需求: {tiny_info['training_memory_mb']:.1f}MB")

    # 验证配置
    warnings = validate_config_for_mac(tiny_config)
    if warnings:
        print(f"\n⚠️  配置警告:")
        for warning in warnings:
            print(f"  - {warning}")
    else:
        print(f"\n✅ 配置验证通过")

    # 测试资源监控
    monitor_config = MacResourceConfig()
    monitor = MacResourceMonitor(monitor_config)
    resources = monitor.check_resources()
    print(f"\n当前资源使用:")
    print(f"  CPU: {resources['cpu_percent']:.1f}%")
    print(f"  内存: {resources['memory_percent']:.1f}%")