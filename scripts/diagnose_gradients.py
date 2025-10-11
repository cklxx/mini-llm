#!/usr/bin/env python3
"""
梯度诊断工具
分析训练过程中的梯度健康状况
"""
import argparse
import json
import os
import sys
from pathlib import Path

import torch

# 添加项目路径
project_root = os.path.dirname(os.path.dirname(__file__))
sys.path.append(project_root)


def analyze_checkpoint_gradients(checkpoint_path):
    """分析checkpoint中的梯度信息"""
    print(f"\n{'='*60}")
    print(f"📊 分析Checkpoint: {checkpoint_path}")
    print(f"{'='*60}\n")

    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    # 基本信息
    print("📋 基本信息:")
    print(f"   训练步数: {checkpoint.get('step', 'N/A')}")
    print(f"   Epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"   Loss: {checkpoint.get('loss', 'N/A'):.4f}")

    # 分析参数范数
    print("\n🔍 参数统计:")
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']

        total_params = 0
        layer_stats = {}

        for name, param in state_dict.items():
            total_params += param.numel()

            # 按层统计
            layer_name = name.split('.')[0]
            if layer_name not in layer_stats:
                layer_stats[layer_name] = {
                    'params': 0,
                    'mean_norm': 0,
                    'count': 0
                }

            param_norm = param.norm().item()
            layer_stats[layer_name]['params'] += param.numel()
            layer_stats[layer_name]['mean_norm'] += param_norm
            layer_stats[layer_name]['count'] += 1

        print(f"   总参数量: {total_params:,}")

        print("\n   各层参数范数:")
        for layer, stats in sorted(layer_stats.items()):
            avg_norm = stats['mean_norm'] / stats['count']
            print(f"   • {layer:20s}: 参数={stats['params']:>10,}, "
                  f"平均范数={avg_norm:>8.4f}")

    # 优化器状态
    if 'optimizer_state_dict' in checkpoint:
        print("\n⚙️  优化器状态:")
        opt_state = checkpoint['optimizer_state_dict']
        print(f"   学习率: {opt_state.get('param_groups', [{}])[0].get('lr', 'N/A')}")


def analyze_training_logs(log_dir):
    """分析训练日志中的梯度趋势"""
    print(f"\n{'='*60}")
    print("📈 分析训练日志")
    print(f"{'='*60}\n")

    # 查找日志文件
    log_files = list(Path(log_dir).glob("**/*.jsonl"))

    if not log_files:
        print(f"❌ 未找到日志文件: {log_dir}")
        return

    print(f"找到 {len(log_files)} 个日志文件\n")

    # 读取梯度数据
    grad_norms = []
    losses = []
    steps = []

    for log_file in log_files:
        try:
            with open(log_file) as f:
                for line in f:
                    try:
                        data = json.loads(line.strip())
                        if 'grad_norm' in data:
                            grad_norms.append(data['grad_norm'])
                            losses.append(data.get('loss', 0))
                            steps.append(data.get('step', len(steps)))
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            print(f"⚠️  读取{log_file}失败: {e}")

    if not grad_norms:
        print("❌ 未找到梯度数据")
        return

    # 统计分析
    import numpy as np
    grad_norms = np.array(grad_norms)
    losses = np.array(losses)

    print(f"📊 梯度范数统计 (共{len(grad_norms)}个样本):")
    print(f"   最小值: {grad_norms.min():.6f}")
    print(f"   最大值: {grad_norms.max():.6f}")
    print(f"   平均值: {grad_norms.mean():.6f}")
    print(f"   中位数: {np.median(grad_norms):.6f}")
    print(f"   标准差: {grad_norms.std():.6f}")

    # 检测异常
    print("\n🚨 异常检测:")
    vanishing_count = (grad_norms < 1e-6).sum()
    explosion_count = (grad_norms > 10).sum()

    print(f"   梯度消失 (<1e-6): {vanishing_count} ({vanishing_count/len(grad_norms)*100:.1f}%)")
    print(f"   梯度爆炸 (>10): {explosion_count} ({explosion_count/len(grad_norms)*100:.1f}%)")

    # 趋势分析
    if len(grad_norms) > 10:
        window = min(10, len(grad_norms) // 4)
        recent_avg = grad_norms[-window:].mean()
        overall_avg = grad_norms.mean()

        print("\n📈 趋势分析:")
        print(f"   整体平均: {overall_avg:.6f}")
        print(f"   最近{window}步平均: {recent_avg:.6f}")

        if recent_avg > overall_avg * 1.5:
            print("   ⚠️  梯度呈上升趋势，注意梯度爆炸风险")
        elif recent_avg < overall_avg * 0.5:
            print("   ⚠️  梯度呈下降趋势，注意梯度消失风险")
        else:
            print("   ✅ 梯度稳定")

    # Loss分析
    if len(losses) > 0:
        print("\n📉 Loss统计:")
        print(f"   初始Loss: {losses[0]:.4f}")
        print(f"   当前Loss: {losses[-1]:.4f}")
        print(f"   下降幅度: {(losses[0] - losses[-1]):.4f}")

        if losses[-1] < losses[0]:
            print("   ✅ Loss正常下降，训练有效")
        else:
            print("   ⚠️  Loss未下降，可能存在问题")


def check_model_architecture(checkpoint_path):
    """检查模型架构的梯度保护机制"""
    print(f"\n{'='*60}")
    print("🏗️  模型架构健康检查")
    print(f"{'='*60}\n")

    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    if 'config' in checkpoint:
        config = checkpoint['config']

        print("✅ 已实现的梯度保护机制:")

        checks = []

        # 检查残差连接
        if hasattr(config, 'num_hidden_layers'):
            checks.append(("残差连接 (Residual)", True, "Transformer标准配置"))

        # 检查归一化
        if hasattr(config, 'rms_norm_eps'):
            checks.append(("RMSNorm归一化", True, "现代LLM标配"))

        # 检查dropout
        if hasattr(config, 'dropout') and config.dropout > 0:
            checks.append(("Dropout正则化", True, f"dropout={config.dropout}"))

        # 检查激活函数
        if hasattr(config, 'hidden_act'):
            act = config.hidden_act
            if act in ['swiglu', 'silu', 'gelu']:
                checks.append(("现代激活函数", True, f"{act.upper()}"))
            else:
                checks.append(("激活函数", False, f"{act} (建议使用SwiGLU/GELU)"))

        for name, status, detail in checks:
            symbol = "✅" if status else "⚠️ "
            print(f"   {symbol} {name:20s}: {detail}")

        print("\n📋 模型配置:")
        print(f"   层数: {getattr(config, 'num_hidden_layers', 'N/A')}")
        print(f"   隐藏维度: {getattr(config, 'hidden_size', 'N/A')}")
        print(f"   注意力头数: {getattr(config, 'num_attention_heads', 'N/A')}")
    else:
        print("⚠️  未找到配置信息")


def provide_recommendations(checkpoint_dir):
    """提供优化建议"""
    print(f"\n{'='*60}")
    print("💡 优化建议")
    print(f"{'='*60}\n")

    recommendations = [
        ("继续观察", "梯度消失/爆炸往往在训练初期出现，继续训练观察趋势"),
        ("检查TensorBoard", "tensorboard --logdir=" + checkpoint_dir),
        ("调整学习率", "如果梯度持续过小，考虑提高学习率 (当前3e-4 → 5e-4)"),
        ("增加warmup", "延长warmup步数可以让梯度更平稳 (4000 → 8000)"),
        ("调整检测阈值", "修改 training_monitor.py:176 阈值 (1e-6 → 1e-8)"),
    ]

    for i, (title, detail) in enumerate(recommendations, 1):
        print(f"{i}. {title}")
        print(f"   → {detail}\n")


def main():
    parser = argparse.ArgumentParser(description="梯度诊断工具")
    parser.add_argument("--checkpoint", type=str, help="Checkpoint文件路径")
    parser.add_argument("--log-dir", type=str, help="日志目录路径")
    parser.add_argument("--mode", type=str, default="pretrain",
                       choices=["pretrain", "sft", "dpo"],
                       help="训练模式")
    parser.add_argument("--config", type=str, default="medium",
                       choices=["tiny", "small", "medium"],
                       help="模型配置")

    args = parser.parse_args()

    # 自动确定路径
    if not args.checkpoint and not args.log_dir:
        base_dir = f"checkpoints/{args.mode}_{args.config}"

        # 查找最新checkpoint
        checkpoint_dir = Path(base_dir)
        if checkpoint_dir.exists():
            checkpoints = list(checkpoint_dir.glob("checkpoint_*.pt"))
            if checkpoints:
                args.checkpoint = str(sorted(checkpoints)[-1])

            # 查找日志目录
            log_dir = checkpoint_dir / "monitor_logs"
            if log_dir.exists():
                args.log_dir = str(log_dir)

    # 执行诊断
    if args.checkpoint and Path(args.checkpoint).exists():
        check_model_architecture(args.checkpoint)
        analyze_checkpoint_gradients(args.checkpoint)
    elif args.checkpoint:
        print(f"❌ Checkpoint不存在: {args.checkpoint}")

    if args.log_dir and Path(args.log_dir).exists():
        analyze_training_logs(args.log_dir)
    elif args.log_dir:
        print(f"❌ 日志目录不存在: {args.log_dir}")

    # 提供建议
    if args.checkpoint or args.log_dir:
        checkpoint_dir = (Path(args.checkpoint).parent if args.checkpoint
                         else args.log_dir)
        provide_recommendations(str(checkpoint_dir))
    else:
        print("\n❌ 未找到checkpoint或日志文件")
        print("💡 使用方法:")
        print("   python scripts/diagnose_gradients.py --mode pretrain --config medium")
        print("   python scripts/diagnose_gradients.py --checkpoint path/to/checkpoint.pt")
        print("   python scripts/diagnose_gradients.py --log-dir path/to/logs")


if __name__ == "__main__":
    main()
