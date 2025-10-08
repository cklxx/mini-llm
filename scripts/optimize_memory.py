#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPU内存优化工具
自动分析GPU内存并建议最优配置
"""
import os
import sys
import torch
import argparse

# 添加项目路径
project_root = os.path.dirname(os.path.dirname(__file__))
sys.path.append(project_root)

from config.training_config import get_gpu_info


def analyze_memory():
    """分析当前GPU内存使用情况"""
    if not torch.cuda.is_available():
        print("❌ 未检测到CUDA GPU")
        return None

    gpu_info = get_gpu_info()

    print("=" * 60)
    print("🔍 GPU内存分析报告")
    print("=" * 60)

    for device in gpu_info['devices']:
        print(f"\n📊 GPU {device['id']}: {device['name']}")
        print(f"   计算能力: {device['compute_capability']}")
        print(f"   显存总量: {device['memory_total']:.2f} GB")
        print(f"   已分配: {device['memory_allocated']:.2f} GB")
        print(f"   已保留: {device['memory_reserved']:.2f} GB")

        available = device['memory_total'] - device['memory_allocated']
        print(f"   可用显存: {available:.2f} GB")

        # 计算建议配置
        total_mem = device['memory_total']

        print(f"\n💡 针对 {device['name']} 的优化建议:")

        if total_mem >= 40:  # A6000, A100等
            print("   GPU类型: 高端训练卡")
            print("   推荐配置 (Medium模型, 512 hidden, 16 layers):")
            print("   • batch_size: 12-16")
            print("   • gradient_accumulation_steps: 8-12")
            print("   • max_seq_len: 2048")
            print("   • 有效批量: 128-192")
            print("\n   保守配置 (避免OOM):")
            print("   • batch_size: 8")
            print("   • gradient_accumulation_steps: 16")
            print("   • max_seq_len: 1024")
        elif total_mem >= 24:  # RTX 3090/4090
            print("   GPU类型: 消费级高端卡")
            print("   推荐配置 (Small模型):")
            print("   • batch_size: 8")
            print("   • gradient_accumulation_steps: 16")
            print("   • max_seq_len: 1024")
        elif total_mem >= 12:  # RTX 3060Ti/4060Ti
            print("   GPU类型: 消费级中端卡")
            print("   推荐配置 (Tiny模型):")
            print("   • batch_size: 4")
            print("   • gradient_accumulation_steps: 32")
            print("   • max_seq_len: 512")
        else:
            print("   GPU类型: 入门级")
            print("   建议使用CPU训练或选择更小的模型")

        print("\n🛠️  额外优化技巧:")
        print("   1. 启用混合精度训练 (FP16/BF16)")
        print("   2. 使用梯度检查点 (gradient checkpointing)")
        print("   3. 设置环境变量:")
        print("      export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True")
        print("   4. 减小模型维度 (d_model, n_layers)")
        print("   5. 使用更小的词汇表大小")

    return gpu_info


def suggest_config(model_size="medium"):
    """根据GPU建议配置"""
    if not torch.cuda.is_available():
        print("❌ 未检测到CUDA GPU")
        return

    gpu_info = get_gpu_info()
    device = gpu_info['devices'][0]
    total_mem = device['memory_total']

    print("\n" + "=" * 60)
    print(f"📋 {model_size.upper()}模型推荐训练命令")
    print("=" * 60)

    if model_size == "medium":
        if total_mem >= 40:
            cmd = """
# A6000/A100 - 保守配置
python scripts/train.py \\
    --mode pretrain \\
    --config medium \\
    --batch-size 12 \\
    --max-steps 50000

# 如果仍然OOM，进一步降低batch_size:
python scripts/train.py \\
    --mode pretrain \\
    --config medium \\
    --batch-size 8 \\
    --max-steps 50000
"""
        elif total_mem >= 24:
            cmd = """
# RTX 3090/4090
python scripts/train.py \\
    --mode pretrain \\
    --config small \\
    --batch-size 8 \\
    --max-steps 50000
"""
        else:
            cmd = """
# 显存不足，建议使用small或tiny配置
python scripts/train.py \\
    --mode pretrain \\
    --config small \\
    --batch-size 4 \\
    --max-steps 50000
"""

    print(cmd)

    print("\n🔧 环境变量优化:")
    print("export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True")
    print("export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512")


def check_oom_solutions():
    """显示OOM问题的解决方案"""
    print("\n" + "=" * 60)
    print("🚨 CUDA OOM 常见解决方案")
    print("=" * 60)

    solutions = [
        ("降低batch size", "从32 → 16 → 8 → 4 逐步降低"),
        ("增加梯度累积", "保持有效batch size，降低内存峰值"),
        ("减小序列长度", "从2048 → 1024 → 512"),
        ("使用更小的模型", "减少层数或隐藏维度"),
        ("启用梯度检查点", "牺牲速度换内存"),
        ("使用混合精度", "FP16/BF16训练"),
        ("清理GPU缓存", "torch.cuda.empty_cache()"),
        ("设置内存分配策略", "PYTORCH_CUDA_ALLOC_CONF"),
    ]

    for i, (solution, detail) in enumerate(solutions, 1):
        print(f"\n{i}. {solution}")
        print(f"   → {detail}")

    print("\n" + "=" * 60)
    print("📝 实际操作步骤:")
    print("=" * 60)

    print("""
1. 立即尝试 (无需重启):
   python scripts/train.py --mode pretrain --config medium --batch-size 8

2. 如果仍然OOM:
   python scripts/train.py --mode pretrain --config medium --batch-size 4

3. 设置环境变量后重试:
   export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
   python scripts/train.py --mode pretrain --config medium --batch-size 8

4. 终极方案 - 使用small配置:
   python scripts/train.py --mode pretrain --config small --batch-size 8
""")


def main():
    parser = argparse.ArgumentParser(description="GPU内存优化工具")
    parser.add_argument("--analyze", action="store_true", help="分析当前GPU内存")
    parser.add_argument("--suggest", type=str, choices=["tiny", "small", "medium"],
                       help="建议指定模型的配置")
    parser.add_argument("--oom-help", action="store_true", help="显示OOM解决方案")

    args = parser.parse_args()

    if args.analyze:
        analyze_memory()
    elif args.suggest:
        analyze_memory()
        suggest_config(args.suggest)
    elif args.oom_help:
        check_oom_solutions()
    else:
        # 默认显示所有信息
        analyze_memory()
        check_oom_solutions()


if __name__ == "__main__":
    main()
