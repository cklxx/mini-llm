#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Checkpoint分析工具
=================

功能：
1. 分析PyTorch checkpoint文件的大小和参数量
2. 对比不同保存格式的体积差异
3. 提供模型压缩和优化建议
4. 支持多种checkpoint格式分析

作者: alex-ckl.com AI研发团队
"""

import os
import sys
import json
import pickle
import argparse
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass

# 添加项目根目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("⚠️  PyTorch未安装，部分功能将不可用")

from src.model.config import get_config, MiniGPTConfig


@dataclass
class CheckpointInfo:
    """Checkpoint信息数据类"""
    file_path: str
    file_size_bytes: int
    file_size_mb: float
    file_size_gb: float
    format_type: str

    # 模型信息 (如果可解析)
    total_params: Optional[int] = None
    trainable_params: Optional[int] = None
    model_config: Optional[Dict] = None

    # 内容信息
    contains_model: bool = False
    contains_optimizer: bool = False
    contains_scheduler: bool = False
    contains_metadata: bool = False

    # 压缩信息
    compression_ratio: Optional[float] = None
    estimated_fp16_size: Optional[float] = None
    estimated_int8_size: Optional[float] = None


def get_file_size(file_path: str) -> Tuple[int, float, float]:
    """获取文件大小 (bytes, MB, GB)"""
    size_bytes = os.path.getsize(file_path)
    size_mb = size_bytes / (1024 * 1024)
    size_gb = size_bytes / (1024 * 1024 * 1024)
    return size_bytes, size_mb, size_gb


def analyze_pytorch_checkpoint(checkpoint_path: str) -> CheckpointInfo:
    """分析PyTorch checkpoint文件"""
    print(f"🔍 分析checkpoint: {os.path.basename(checkpoint_path)}")

    # 基本文件信息
    size_bytes, size_mb, size_gb = get_file_size(checkpoint_path)

    info = CheckpointInfo(
        file_path=checkpoint_path,
        file_size_bytes=size_bytes,
        file_size_mb=size_mb,
        file_size_gb=size_gb,
        format_type="PyTorch (.pt/.pth)"
    )

    if not TORCH_AVAILABLE:
        print("⚠️  PyTorch未安装，无法解析checkpoint内容")
        return info

    try:
        # 加载checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        # 分析checkpoint内容
        if isinstance(checkpoint, dict):
            # 检查包含的组件
            info.contains_model = 'model' in checkpoint or 'model_state_dict' in checkpoint
            info.contains_optimizer = 'optimizer' in checkpoint or 'optimizer_state_dict' in checkpoint
            info.contains_scheduler = 'scheduler' in checkpoint or 'lr_scheduler' in checkpoint
            info.contains_metadata = any(key in checkpoint for key in ['epoch', 'step', 'config', 'args'])

            # 尝试获取模型状态字典
            model_state = None
            if 'model_state_dict' in checkpoint:
                model_state = checkpoint['model_state_dict']
            elif 'model' in checkpoint:
                model_state = checkpoint['model']
            elif isinstance(checkpoint, dict) and 'transformer' in str(checkpoint.keys()):
                model_state = checkpoint

            # 分析模型参数
            if model_state is not None:
                total_params = 0
                trainable_params = 0

                for name, param in model_state.items():
                    if isinstance(param, torch.Tensor):
                        param_count = param.numel()
                        total_params += param_count
                        # 假设所有参数都是可训练的
                        trainable_params += param_count

                info.total_params = total_params
                info.trainable_params = trainable_params

                # 估算不同精度下的大小
                info.estimated_fp16_size = total_params * 2 / (1024 * 1024)  # FP16: 2 bytes per param
                info.estimated_int8_size = total_params * 1 / (1024 * 1024)  # INT8: 1 byte per param

            # 尝试获取配置信息
            if 'config' in checkpoint:
                info.model_config = checkpoint['config']
            elif 'args' in checkpoint:
                info.model_config = checkpoint['args']

        elif isinstance(checkpoint, torch.nn.Module):
            # 直接是模型对象
            info.contains_model = True

            total_params = sum(p.numel() for p in checkpoint.parameters())
            trainable_params = sum(p.numel() for p in checkpoint.parameters() if p.requires_grad)

            info.total_params = total_params
            info.trainable_params = trainable_params
            info.estimated_fp16_size = total_params * 2 / (1024 * 1024)
            info.estimated_int8_size = total_params * 1 / (1024 * 1024)

    except Exception as e:
        print(f"❌ 解析checkpoint失败: {e}")

    return info


def analyze_safetensors_checkpoint(checkpoint_path: str) -> CheckpointInfo:
    """分析SafeTensors格式checkpoint"""
    print(f"🔍 分析SafeTensors checkpoint: {os.path.basename(checkpoint_path)}")

    size_bytes, size_mb, size_gb = get_file_size(checkpoint_path)

    info = CheckpointInfo(
        file_path=checkpoint_path,
        file_size_bytes=size_bytes,
        file_size_mb=size_mb,
        file_size_gb=size_gb,
        format_type="SafeTensors (.safetensors)"
    )

    try:
        # 尝试导入safetensors
        from safetensors import safe_open

        total_params = 0
        with safe_open(checkpoint_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                tensor = f.get_tensor(key)
                total_params += tensor.numel()

        info.total_params = total_params
        info.trainable_params = total_params  # 假设都是可训练的
        info.contains_model = True

        # 估算不同精度大小
        info.estimated_fp16_size = total_params * 2 / (1024 * 1024)
        info.estimated_int8_size = total_params * 1 / (1024 * 1024)

    except ImportError:
        print("⚠️  safetensors库未安装，无法解析内容")
    except Exception as e:
        print(f"❌ 解析SafeTensors失败: {e}")

    return info


def create_demo_checkpoint(config_name: str = "medium") -> str:
    """创建演示用的checkpoint文件"""
    if not TORCH_AVAILABLE:
        print("⚠️  PyTorch未安装，无法创建演示checkpoint")
        return None

    print(f"🔧 创建 {config_name} 配置的演示checkpoint...")

    try:
        from src.model.transformer import MiniGPT

        # 获取配置
        config = get_config(config_name)

        # 创建模型
        model = MiniGPT(config)

        # 创建checkpoint目录
        checkpoint_dir = Path("checkpoints/demo")
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # 保存不同格式的checkpoint
        base_name = f"demo_{config_name}_model"

        # 1. 完整checkpoint (包含优化器等)
        full_checkpoint_path = checkpoint_dir / f"{base_name}_full.pt"
        full_checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': torch.optim.AdamW(model.parameters()).state_dict(),
            'scheduler_state_dict': {},
            'epoch': 10,
            'step': 1000,
            'loss': 2.5,
            'config': config.to_dict(),
            'model_config': {
                'total_params': sum(p.numel() for p in model.parameters()),
                'trainable_params': sum(p.numel() for p in model.parameters() if p.requires_grad)
            }
        }
        torch.save(full_checkpoint, full_checkpoint_path)

        # 2. 仅模型权重checkpoint
        model_only_path = checkpoint_dir / f"{base_name}_weights.pt"
        torch.save(model.state_dict(), model_only_path)

        # 3. 压缩checkpoint (如果支持)
        compressed_path = checkpoint_dir / f"{base_name}_compressed.pt"
        torch.save(full_checkpoint, compressed_path, _use_new_zipfile_serialization=True)

        print(f"✅ 演示checkpoint已创建在: {checkpoint_dir}")
        return str(checkpoint_dir)

    except Exception as e:
        print(f"❌ 创建演示checkpoint失败: {e}")
        return None


def compare_checkpoint_formats(checkpoint_dir: str) -> Dict[str, CheckpointInfo]:
    """对比不同格式checkpoint的大小"""
    print(f"\n📊 对比checkpoint格式...")

    results = {}

    # 查找所有checkpoint文件
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        print(f"❌ 目录不存在: {checkpoint_dir}")
        return results

    # 支持的文件扩展名
    extensions = ['.pt', '.pth', '.safetensors', '.ckpt']

    for ext in extensions:
        files = list(checkpoint_dir.glob(f"*{ext}"))
        for file_path in files:
            print(f"\n分析文件: {file_path.name}")

            if ext in ['.pt', '.pth', '.ckpt']:
                info = analyze_pytorch_checkpoint(str(file_path))
            elif ext == '.safetensors':
                info = analyze_safetensors_checkpoint(str(file_path))

            results[file_path.name] = info

    return results


def print_checkpoint_analysis(info: CheckpointInfo):
    """打印checkpoint分析结果"""
    print(f"\n{'='*60}")
    print(f"📁 文件: {os.path.basename(info.file_path)}")
    print(f"{'='*60}")

    # 基本信息
    print(f"📏 文件大小:")
    print(f"  • {info.file_size_bytes:,} bytes")
    print(f"  • {info.file_size_mb:.2f} MB")
    if info.file_size_gb >= 0.1:
        print(f"  • {info.file_size_gb:.3f} GB")

    print(f"📂 格式类型: {info.format_type}")

    # 内容信息
    print(f"\n📦 包含内容:")
    print(f"  • 模型权重: {'✅' if info.contains_model else '❌'}")
    print(f"  • 优化器状态: {'✅' if info.contains_optimizer else '❌'}")
    print(f"  • 调度器状态: {'✅' if info.contains_scheduler else '❌'}")
    print(f"  • 元数据: {'✅' if info.contains_metadata else '❌'}")

    # 参数信息
    if info.total_params:
        print(f"\n🔢 参数统计:")
        print(f"  • 总参数量: {info.total_params:,}")
        if info.trainable_params:
            print(f"  • 可训练参数: {info.trainable_params:,}")

        # 理论大小估算
        print(f"\n💾 理论模型大小 (仅权重):")
        if info.estimated_fp16_size:
            print(f"  • FP16: {info.estimated_fp16_size:.1f} MB")
        if info.estimated_int8_size:
            print(f"  • INT8: {info.estimated_int8_size:.1f} MB")

        # 实际文件大小 vs 理论大小
        if info.estimated_fp16_size:
            overhead = info.file_size_mb / info.estimated_fp16_size
            print(f"\n📈 存储效率:")
            print(f"  • 文件大小 / 理论FP16大小: {overhead:.2f}x")
            if overhead > 2.0:
                print(f"  • ⚠️  文件包含额外信息 (优化器状态、元数据等)")
            elif overhead < 1.1:
                print(f"  • ✅ 高效压缩存储")

    # 配置信息
    if info.model_config:
        print(f"\n⚙️  模型配置:")
        for key, value in info.model_config.items():
            if isinstance(value, (int, float, str, bool)):
                print(f"  • {key}: {value}")


def print_comparison_table(results: Dict[str, CheckpointInfo]):
    """打印对比表格"""
    if not results:
        print("❌ 没有找到checkpoint文件")
        return

    print(f"\n{'='*80}")
    print(f"📊 CHECKPOINT格式对比表")
    print(f"{'='*80}")

    # 表头
    print(f"{'文件名':<30} {'大小(MB)':<10} {'格式':<15} {'参数量':<12} {'效率':<8}")
    print(f"{'-'*30} {'-'*10} {'-'*15} {'-'*12} {'-'*8}")

    # 按文件大小排序
    sorted_results = sorted(results.items(), key=lambda x: x[1].file_size_mb)

    for filename, info in sorted_results:
        params_str = f"{info.total_params/1e6:.1f}M" if info.total_params else "N/A"

        # 计算效率比
        efficiency = ""
        if info.total_params and info.estimated_fp16_size:
            ratio = info.file_size_mb / info.estimated_fp16_size
            if ratio <= 1.2:
                efficiency = "优秀"
            elif ratio <= 2.0:
                efficiency = "良好"
            else:
                efficiency = "一般"

        print(f"{filename:<30} {info.file_size_mb:<10.1f} {info.format_type.split()[0]:<15} {params_str:<12} {efficiency:<8}")

    # 总结
    print(f"\n💡 总结:")
    min_size = min(info.file_size_mb for info in results.values())
    max_size = max(info.file_size_mb for info in results.values())
    print(f"  • 最小文件: {min_size:.1f}MB")
    print(f"  • 最大文件: {max_size:.1f}MB")
    print(f"  • 大小差异: {max_size/min_size:.1f}x")


def provide_optimization_suggestions(results: Dict[str, CheckpointInfo]):
    """提供checkpoint优化建议"""
    print(f"\n{'='*60}")
    print(f"🚀 CHECKPOINT优化建议")
    print(f"{'='*60}")

    suggestions = []

    # 分析结果
    has_full_checkpoint = any(info.contains_optimizer for info in results.values())
    has_model_only = any(not info.contains_optimizer and info.contains_model for info in results.values())

    # 基本建议
    print(f"📝 通用优化建议:")

    if has_full_checkpoint:
        print(f"  1. 💾 分离存储策略:")
        print(f"     • 模型权重单独保存 (用于推理)")
        print(f"     • 训练状态单独保存 (用于恢复训练)")
        print(f"     • 可节省50-80%的推理部署体积")

    print(f"  2. 🗜️  精度优化:")
    print(f"     • FP16: 减少50%存储，保持精度")
    print(f"     • INT8量化: 减少75%存储，轻微精度损失")
    print(f"     • INT4量化: 减少87.5%存储，适合边缘部署")

    print(f"  3. 📦 格式选择:")
    print(f"     • SafeTensors: 更安全，加载更快")
    print(f"     • ONNX: 跨平台部署优化")
    print(f"     • TensorRT: NVIDIA GPU部署加速")

    # 针对性建议
    if results:
        largest_file = max(results.values(), key=lambda x: x.file_size_mb)

        if largest_file.file_size_mb > 500:  # 大于500MB
            print(f"\n⚠️  大文件优化建议:")
            print(f"     • 当前最大文件: {largest_file.file_size_mb:.1f}MB")
            print(f"     • 建议启用模型并行或分片存储")
            print(f"     • 考虑使用梯度检查点减少训练内存")

        if any(info.file_size_mb / info.estimated_fp16_size > 3.0 for info in results.values() if info.estimated_fp16_size):
            print(f"\n🔍 存储效率建议:")
            print(f"     • 检测到低效存储文件")
            print(f"     • 建议清理不必要的优化器状态")
            print(f"     • 使用torch.save(..., _use_new_zipfile_serialization=True)")

    print(f"\n🛠️  实用工具推荐:")
    print(f"     • 模型量化: torch.quantization")
    print(f"     • 模型剪枝: torch.nn.utils.prune")
    print(f"     • 知识蒸馏: 减少模型大小同时保持性能")
    print(f"     • 动态量化: 运行时量化，无需重新训练")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Checkpoint分析工具")
    parser.add_argument("--checkpoint", "-c", type=str, help="checkpoint文件路径")
    parser.add_argument("--directory", "-d", type=str, help="checkpoint目录路径")
    parser.add_argument("--create-demo", action="store_true", help="创建演示checkpoint")
    parser.add_argument("--config", default="medium", help="演示checkpoint的配置 (tiny/small/medium)")
    parser.add_argument("--compare", action="store_true", help="对比不同格式")

    args = parser.parse_args()

    print("🔍 MiniGPT Checkpoint分析工具")
    print("alex-ckl.com AI研发团队")
    print("="*60)

    if args.create_demo:
        demo_dir = create_demo_checkpoint(args.config)
        if demo_dir:
            args.directory = demo_dir
            args.compare = True

    if args.checkpoint:
        # 分析单个checkpoint文件
        if not os.path.exists(args.checkpoint):
            print(f"❌ 文件不存在: {args.checkpoint}")
            return

        if args.checkpoint.endswith('.safetensors'):
            info = analyze_safetensors_checkpoint(args.checkpoint)
        else:
            info = analyze_pytorch_checkpoint(args.checkpoint)

        print_checkpoint_analysis(info)

    elif args.directory or args.compare:
        # 分析目录中的所有checkpoint
        directory = args.directory or "checkpoints"

        if not os.path.exists(directory):
            print(f"❌ 目录不存在: {directory}")
            print(f"💡 使用 --create-demo 创建演示文件")
            return

        results = compare_checkpoint_formats(directory)

        if results:
            # 打印每个文件的详细分析
            for filename, info in results.items():
                print_checkpoint_analysis(info)

            # 打印对比表格
            print_comparison_table(results)

            # 提供优化建议
            provide_optimization_suggestions(results)
        else:
            print(f"❌ 在目录 {directory} 中没有找到checkpoint文件")

    else:
        print("🆘 使用帮助:")
        print("  python checkpoint_analyzer.py --checkpoint model.pt")
        print("  python checkpoint_analyzer.py --directory checkpoints/")
        print("  python checkpoint_analyzer.py --create-demo --config medium")
        print("  python checkpoint_analyzer.py --compare --directory checkpoints/")


if __name__ == "__main__":
    main()