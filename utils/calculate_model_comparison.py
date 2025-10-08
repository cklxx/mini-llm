#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
计算Medium模型与Small模型的资源对比
"""
import os
import sys
import torch
import time

# 添加项目路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from model.transformer import create_model


def calculate_model_params(model):
    """计算模型参数量"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def estimate_memory_usage(model, batch_size, seq_length):
    """估算内存使用量"""
    # 模型参数内存 (假设float32)
    model_memory = sum(p.numel() * 4 for p in model.parameters()) / (1024**2)  # MB
    
    # 前向传播激活值内存估算
    d_model = model.d_model
    n_layers = len(model.transformer_blocks)
    
    # 每层的激活值内存估算
    attention_memory = batch_size * seq_length * d_model * 4 / (1024**2)  # Q,K,V,O
    ffn_memory = batch_size * seq_length * model.transformer_blocks[0].feed_forward.w_1.out_features * 4 / (1024**2)
    
    # 总激活值内存 (乘以层数)
    activation_memory = (attention_memory + ffn_memory) * n_layers
    
    # 梯度内存 (与参数量相同)
    gradient_memory = model_memory
    
    # 优化器状态内存 (Adam需要2倍参数量)
    optimizer_memory = model_memory * 2
    
    total_memory = model_memory + activation_memory + gradient_memory + optimizer_memory
    
    return {
        'model_memory': model_memory,
        'activation_memory': activation_memory,
        'gradient_memory': gradient_memory,
        'optimizer_memory': optimizer_memory,
        'total_memory': total_memory
    }


def estimate_training_time_ratio(small_params, medium_params, small_batch, medium_batch):
    """估算训练时间比例"""
    # 假设训练时间主要由参数量和批次大小决定
    param_ratio = medium_params / small_params
    
    # 考虑批次大小的影响 (小批次需要更多步骤)
    batch_ratio = small_batch / medium_batch
    
    # 综合时间比例估算
    time_ratio = param_ratio * batch_ratio
    
    return time_ratio


def main():
    print("=== Medium vs Small 模型资源对比分析 ===\n")
    
    # 模型配置
    vocab_size = 10000
    
    # 创建模型
    print("创建模型...")
    small_model = create_model(vocab_size, "small")
    medium_model = create_model(vocab_size, "medium")
    
    # 计算参数量
    small_params, small_trainable = calculate_model_params(small_model)
    medium_params, medium_trainable = calculate_model_params(medium_model)
    
    print(f"\n📊 参数量对比:")
    print(f"  Small模型:  {small_params:,} 参数 ({small_params/1e6:.1f}M)")
    print(f"  Medium模型: {medium_params:,} 参数 ({medium_params/1e6:.1f}M)")
    print(f"  参数量比例: {medium_params/small_params:.2f}x")
    
    # 配置信息
    small_config = {
        "batch_size": 8,
        "seq_length": 256,
        "d_model": 512,
        "n_layers": 6
    }
    
    medium_config = {
        "batch_size": 2,
        "seq_length": 512,
        "d_model": 640,
        "n_layers": 10
    }
    
    print(f"\n⚙️  配置对比:")
    print(f"  Small:  batch_size={small_config['batch_size']}, seq_len={small_config['seq_length']}, d_model={small_config['d_model']}, layers={small_config['n_layers']}")
    print(f"  Medium: batch_size={medium_config['batch_size']}, seq_len={medium_config['seq_length']}, d_model={medium_config['d_model']}, layers={medium_config['n_layers']}")
    
    # 内存使用估算
    small_memory = estimate_memory_usage(small_model, small_config["batch_size"], small_config["seq_length"])
    medium_memory = estimate_memory_usage(medium_model, medium_config["batch_size"], medium_config["seq_length"])
    
    print(f"\n💾 内存使用估算:")
    print(f"  Small模型:")
    print(f"    - 模型参数: {small_memory['model_memory']:.1f} MB")
    print(f"    - 激活值:   {small_memory['activation_memory']:.1f} MB")
    print(f"    - 梯度:     {small_memory['gradient_memory']:.1f} MB")
    print(f"    - 优化器:   {small_memory['optimizer_memory']:.1f} MB")
    print(f"    - 总计:     {small_memory['total_memory']:.1f} MB ({small_memory['total_memory']/1024:.2f} GB)")
    
    print(f"\n  Medium模型:")
    print(f"    - 模型参数: {medium_memory['model_memory']:.1f} MB")
    print(f"    - 激活值:   {medium_memory['activation_memory']:.1f} MB")
    print(f"    - 梯度:     {medium_memory['gradient_memory']:.1f} MB")
    print(f"    - 优化器:   {medium_memory['optimizer_memory']:.1f} MB")
    print(f"    - 总计:     {medium_memory['total_memory']:.1f} MB ({medium_memory['total_memory']/1024:.2f} GB)")
    
    memory_ratio = medium_memory['total_memory'] / small_memory['total_memory']
    print(f"\n  内存使用比例: {memory_ratio:.2f}x")
    
    # 训练时间估算
    time_ratio = estimate_training_time_ratio(
        small_params, medium_params,
        small_config["batch_size"], medium_config["batch_size"]
    )
    
    print(f"\n⏱️  训练时间估算:")
    print(f"  参数量影响: {medium_params/small_params:.2f}x")
    print(f"  批次大小影响: {small_config['batch_size']/medium_config['batch_size']:.2f}x")
    print(f"  预估时间比例: {time_ratio:.2f}x")
    
    # 训练步数对比
    small_steps = 1500  # from small config
    medium_steps = 4000  # from medium config
    
    effective_batch_small = small_config["batch_size"] * 4  # gradient_accumulation_steps
    effective_batch_medium = medium_config["batch_size"] * 12  # gradient_accumulation_steps
    
    print(f"\n📈 训练设置对比:")
    print(f"  Small:  {small_steps} 步, 有效批次={effective_batch_small}")
    print(f"  Medium: {medium_steps} 步, 有效批次={effective_batch_medium}")
    print(f"  总步数比例: {medium_steps/small_steps:.2f}x")
    
    # 实际训练时间预估
    if small_memory['total_memory'] < 4000:  # 4GB
        small_estimated_time = 45  # 45分钟
    else:
        small_estimated_time = 60  # 1小时
        
    medium_estimated_time = small_estimated_time * time_ratio * (medium_steps/small_steps)
    
    print(f"\n🕐 实际训练时间预估:")
    print(f"  Small模型:  约 {small_estimated_time} 分钟")
    print(f"  Medium模型: 约 {medium_estimated_time:.0f} 分钟 ({medium_estimated_time/60:.1f} 小时)")
    print(f"  时间增加:   {medium_estimated_time/small_estimated_time:.1f}x")
    
    # 推荐配置
    print(f"\n💡 推荐配置:")
    if medium_memory['total_memory'] > 6000:  # > 6GB
        print("  ⚠️  Medium模型内存需求较高，建议:")
        print("     - 确保Mac有充足内存 (16GB+)")
        print("     - 关闭其他应用程序")
        print("     - 考虑降低batch_size到1")
        print("     - 启用梯度检查点 (如果实现)")
    
    print(f"\n📋 总结:")
    print(f"  Medium模型比Small模型:")
    print(f"  - 参数量增加: {medium_params/small_params:.1f}倍")
    print(f"  - 内存需求增加: {memory_ratio:.1f}倍")
    print(f"  - 训练时间增加: {medium_estimated_time/small_estimated_time:.1f}倍")
    print(f"  - 理论性能提升: 预计更好的生成质量和语言理解能力")


if __name__ == "__main__":
    main() 