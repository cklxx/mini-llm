#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于Scaling Laws研究的智能模型训练计划
"""

from dataclasses import dataclass
from typing import List, Dict
import math

@dataclass
class ScalingConfig:
    """模型缩放配置"""
    name: str
    d_model: int
    n_heads: int
    n_layers: int
    d_ff: int
    vocab_size: int
    max_seq_len: int

    # 训练配置
    learning_rate: float
    batch_size: int
    max_steps: int
    warmup_steps: int
    gradient_accumulation_steps: int

    # 数据配置
    min_training_tokens: int  # 基于Chinchilla比例

    def get_params_estimate(self) -> int:
        """估算参数量"""
        # Transformer参数估算: 12 * n_layers * d_model^2
        transformer_params = 12 * self.n_layers * (self.d_model ** 2)
        # 加上embedding和输出层
        embedding_params = self.vocab_size * self.d_model * 2
        return transformer_params + embedding_params

    def get_flops_estimate(self) -> int:
        """估算训练所需FLOPs"""
        # 6 * N * D (Chinchilla公式)
        return 6 * self.get_params_estimate() * self.min_training_tokens

def calculate_chinchilla_tokens(d_model: int, n_layers: int, vocab_size: int) -> int:
    """根据Chinchilla比例计算所需tokens"""
    # 估算参数量
    transformer_params = 12 * n_layers * (d_model ** 2)
    embedding_params = vocab_size * d_model * 2
    total_params = transformer_params + embedding_params

    # Chinchilla比例: 20 tokens per parameter
    return int(total_params * 20)

def get_intelligence_threshold_configs() -> Dict[str, ScalingConfig]:
    """
    基于研究的智能阈值配置

    根据以下研究设计：
    - Anthropic: 22B参数出现道德推理
    - Chinchilla: 参数:数据 = 1:20最优比例
    - OpenAI: 每8倍参数增长需要约5倍数据增长
    """

    configs = {}

    # Large: 接近智能阈值 (简化版22B)
    configs["large"] = ScalingConfig(
        name="large",
        d_model=2048,          # 大幅增加
        n_heads=32,            # 64个头
        n_layers=24,           # 24层
        d_ff=8192,             # 4x d_model
        vocab_size=50000,      # 扩大词汇表
        max_seq_len=2048,      # 更长序列

        # 训练配置
        learning_rate=1e-4,
        batch_size=1,          # 内存限制下的最小batch
        max_steps=50000,       # 大幅增加训练步数
        warmup_steps=2000,
        gradient_accumulation_steps=64,  # 通过梯度累积模拟大batch

        # 数据配置 (基于Chinchilla 1:20比例)
        min_training_tokens=calculate_chinchilla_tokens(2048, 24, 50000)
    )

    # Extra Large: 更接近智能阈值
    configs["xlarge"] = ScalingConfig(
        name="xlarge",
        d_model=2560,          # 进一步增加
        n_heads=40,            # 40个头
        n_layers=30,           # 30层
        d_ff=10240,            # 4x d_model
        vocab_size=50000,
        max_seq_len=2048,

        # 训练配置
        learning_rate=8e-5,    # 大模型需要更小学习率
        batch_size=1,
        max_steps=100000,      # 更多训练步数
        warmup_steps=5000,
        gradient_accumulation_steps=128,

        # 数据配置
        min_training_tokens=calculate_chinchilla_tokens(2560, 30, 50000)
    )

    # XXL: 向22B参数目标
    configs["xxl"] = ScalingConfig(
        name="xxl",
        d_model=4096,          # 接近GPT规模
        n_heads=64,            # 64个头
        n_layers=32,           # 32层
        d_ff=16384,            # 4x d_model
        vocab_size=50000,
        max_seq_len=2048,

        # 训练配置
        learning_rate=5e-5,    # 更小学习率
        batch_size=1,
        max_steps=200000,      # 大量训练步数
        warmup_steps=10000,
        gradient_accumulation_steps=256,  # 大梯度累积

        # 数据配置
        min_training_tokens=calculate_chinchilla_tokens(4096, 32, 50000)
    )

    return configs

def calculate_scaling_metrics(config: ScalingConfig) -> Dict[str, float]:
    """计算缩放指标"""
    params = config.get_params_estimate()
    flops = config.get_flops_estimate()

    # 相对于medium模型的增长倍数
    medium_params = 59484480  # 59.5M

    return {
        "estimated_params": params,
        "params_vs_medium": params / medium_params,
        "estimated_flops": flops,
        "training_tokens": config.min_training_tokens,
        "tokens_per_param": config.min_training_tokens / params if params > 0 else 0,
        "memory_estimate_gb": params * 4 / (1024**3),  # FP32估算
        "training_days_estimate": flops / (1e15 * 86400),  # 假设1PF/s
    }

def print_scaling_analysis():
    """打印缩放分析"""
    print("🧠 基于Scaling Laws的智能模型训练计划")
    print("=" * 60)

    configs = get_intelligence_threshold_configs()

    print(f"📚 研究依据:")
    print(f"  • Anthropic: 道德推理能力 ≥ 22B 参数")
    print(f"  • Chinchilla: 最优比例 = 1参数:20tokens")
    print(f"  • 当前medium模型: 59M 参数 (距离阈值 370x)")
    print()

    for name, config in configs.items():
        metrics = calculate_scaling_metrics(config)

        print(f"🚀 {name.upper()} 模型配置:")
        print(f"  模型结构: {config.d_model}d×{config.n_layers}层×{config.n_heads}头")
        print(f"  参数量: {metrics['estimated_params']:,} ({metrics['estimated_params']/1e9:.1f}B)")
        print(f"  vs Medium: {metrics['params_vs_medium']:.1f}x 增长")
        print(f"  训练数据: {metrics['training_tokens']:,} tokens")
        tokens_per_param = metrics['tokens_per_param']
        print(f"  数据比例: 1:{tokens_per_param:.1f} (需要{tokens_per_param:.1f}个tokens/参数)")
        print(f"  内存需求: ~{metrics['memory_estimate_gb']:.1f}GB")
        print(f"  训练时间: ~{metrics['training_days_estimate']:.1f} 天")
        print()

def get_recommended_next_step() -> ScalingConfig:
    """推荐下一步配置"""
    configs = get_intelligence_threshold_configs()

    # 基于硬件限制，推荐large配置
    return configs["large"]

if __name__ == "__main__":
    print_scaling_analysis()

    print("💡 推荐方案:")
    print("  1. 先训练Large模型 (2.5B参数)")
    print("  2. 使用更多高质量数据 (50M+ tokens)")
    print("  3. 增加训练步数和时间")
    print("  4. 观察是否出现更复杂的推理能力")