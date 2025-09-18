"""
内存优化系统
包含混合精度训练、梯度累积、内存管理和优化策略
"""
import os
import gc
import psutil
import warnings
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
from contextlib import contextmanager

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
import torch.distributed as dist


@dataclass
class MemoryConfig:
    """内存优化配置"""
    # 混合精度配置
    enable_amp: bool = True
    amp_dtype: torch.dtype = torch.float16
    loss_scale_window: int = 2000
    init_scale: float = 65536.0

    # 梯度累积配置
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    adaptive_grad_clip: bool = True

    # 内存管理配置
    enable_memory_efficient_attention: bool = True
    enable_gradient_checkpointing: bool = False
    clear_cache_frequency: int = 100

    # 动态批处理配置
    adaptive_batch_size: bool = False
    min_batch_size: int = 1
    max_batch_size: int = 64
    oom_batch_size_reduction: float = 0.75

    # 内存监控配置
    memory_fraction_threshold: float = 0.9
    enable_memory_profiling: bool = False


class MemoryMonitor:
    """内存监控器"""

    def __init__(self, device: torch.device):
        self.device = device
        self.memory_history = []

    def get_memory_info(self) -> Dict[str, float]:
        """获取详细内存信息"""
        info = {}

        if self.device.type == 'cuda':
            # CUDA内存信息
            info.update({
                'gpu_allocated_gb': torch.cuda.memory_allocated(self.device) / 1024**3,
                'gpu_reserved_gb': torch.cuda.memory_reserved(self.device) / 1024**3,
                'gpu_max_allocated_gb': torch.cuda.max_memory_allocated(self.device) / 1024**3,
                'gpu_max_reserved_gb': torch.cuda.max_memory_reserved(self.device) / 1024**3,
            })

            # 计算GPU内存使用率
            if torch.cuda.is_available():
                total_memory = torch.cuda.get_device_properties(self.device).total_memory
                info['gpu_total_gb'] = total_memory / 1024**3
                info['gpu_usage_percent'] = info['gpu_allocated_gb'] / info['gpu_total_gb'] * 100

        elif self.device.type == 'mps':
            # MPS内存信息（近似）
            current_alloc = torch.mps.current_allocated_memory() / 1024**3 if hasattr(torch.mps, 'current_allocated_memory') else 0
            info.update({
                'mps_allocated_gb': current_alloc,
                'mps_usage_percent': min(current_alloc / 16 * 100, 100)  # 假设16GB统一内存
            })

        # CPU内存信息
        memory = psutil.virtual_memory()
        info.update({
            'ram_used_gb': memory.used / 1024**3,
            'ram_total_gb': memory.total / 1024**3,
            'ram_usage_percent': memory.percent,
            'ram_available_gb': memory.available / 1024**3
        })

        self.memory_history.append(info)
        return info

    def check_memory_pressure(self, threshold: float = 0.9) -> bool:
        """检查内存压力"""
        info = self.get_memory_info()

        # 检查GPU内存
        if 'gpu_usage_percent' in info:
            if info['gpu_usage_percent'] / 100 > threshold:
                return True

        # 检查MPS内存
        if 'mps_usage_percent' in info:
            if info['mps_usage_percent'] / 100 > threshold:
                return True

        # 检查RAM
        if info['ram_usage_percent'] / 100 > threshold:
            return True

        return False

    def force_cleanup(self):
        """强制清理内存"""
        # Python垃圾回收
        gc.collect()

        # PyTorch内存清理
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(self.device)
        elif self.device.type == 'mps':
            if hasattr(torch.mps, 'empty_cache'):
                torch.mps.empty_cache()

    def get_memory_summary(self) -> str:
        """获取内存使用摘要"""
        info = self.get_memory_info()
        lines = ["Memory Usage Summary:"]

        if 'gpu_allocated_gb' in info:
            lines.append(f"  GPU: {info['gpu_allocated_gb']:.2f}GB/{info.get('gpu_total_gb', 0):.2f}GB "
                        f"({info.get('gpu_usage_percent', 0):.1f}%)")

        if 'mps_allocated_gb' in info:
            lines.append(f"  MPS: {info['mps_allocated_gb']:.2f}GB "
                        f"({info.get('mps_usage_percent', 0):.1f}%)")

        lines.append(f"  RAM: {info['ram_used_gb']:.2f}GB/{info['ram_total_gb']:.2f}GB "
                    f"({info['ram_usage_percent']:.1f}%)")

        return "\n".join(lines)


class MixedPrecisionManager:
    """混合精度训练管理器"""

    def __init__(self, config: MemoryConfig, device: torch.device):
        self.config = config
        self.device = device
        self.enabled = config.enable_amp and self._check_amp_support()

        if self.enabled:
            self.scaler = GradScaler(
                init_scale=config.init_scale,
                growth_factor=2.0,
                backoff_factor=0.5,
                growth_interval=config.loss_scale_window
            )
            print(f"✅ Mixed precision training enabled (dtype: {config.amp_dtype})")
        else:
            self.scaler = None
            print("❌ Mixed precision training disabled or not supported")

    def _check_amp_support(self) -> bool:
        """检查设备是否支持自动混合精度"""
        if self.device.type == 'cuda':
            # 检查CUDA版本和设备支持
            return torch.cuda.is_available() and torch.cuda.amp.autocast
        elif self.device.type == 'cpu':
            # CPU支持AMP（PyTorch 1.10+）
            return hasattr(torch.cpu.amp, 'autocast')
        else:
            # MPS暂不支持AMP
            return False

    @contextmanager
    def autocast_context(self):
        """自动混合精度上下文管理器"""
        if self.enabled:
            if self.device.type == 'cuda':
                with autocast(dtype=self.config.amp_dtype):
                    yield
            elif self.device.type == 'cpu':
                with torch.cpu.amp.autocast(dtype=self.config.amp_dtype):
                    yield
            else:
                yield
        else:
            yield

    def scale_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """缩放损失"""
        if self.enabled and self.scaler:
            return self.scaler.scale(loss)
        return loss

    def step_optimizer(self, optimizer: optim.Optimizer) -> bool:
        """执行优化器步骤"""
        if self.enabled and self.scaler:
            self.scaler.step(optimizer)
            self.scaler.update()
            return True
        else:
            optimizer.step()
            return True

    def backward(self, loss: torch.Tensor):
        """反向传播"""
        if self.enabled and self.scaler:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

    def get_scale(self) -> float:
        """获取当前缩放因子"""
        if self.enabled and self.scaler:
            return self.scaler.get_scale()
        return 1.0


class GradientAccumulator:
    """梯度累积管理器"""

    def __init__(self, config: MemoryConfig, model: nn.Module):
        self.config = config
        self.model = model
        self.accumulation_steps = config.gradient_accumulation_steps
        self.max_grad_norm = config.max_grad_norm
        self.adaptive_clip = config.adaptive_grad_clip

        self.step_count = 0
        self.gradient_norms = []

        print(f"🔄 Gradient accumulation: {self.accumulation_steps} steps")

    def should_update(self) -> bool:
        """检查是否应该更新参数"""
        return (self.step_count + 1) % self.accumulation_steps == 0

    def accumulate_gradients(self, loss: torch.Tensor) -> torch.Tensor:
        """累积梯度"""
        # 按累积步数归一化损失
        normalized_loss = loss / self.accumulation_steps
        return normalized_loss

    def clip_gradients(self) -> float:
        """梯度裁剪"""
        if self.max_grad_norm <= 0:
            return 0.0

        # 计算梯度范数
        if self.adaptive_clip and len(self.gradient_norms) > 10:
            # 自适应梯度裁剪
            recent_norms = self.gradient_norms[-10:]
            adaptive_norm = np.mean(recent_norms) + 2 * np.std(recent_norms)
            clip_norm = min(self.max_grad_norm, adaptive_norm)
        else:
            clip_norm = self.max_grad_norm

        # 执行梯度裁剪
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            max_norm=clip_norm
        ).item()

        self.gradient_norms.append(grad_norm)
        if len(self.gradient_norms) > 100:
            self.gradient_norms = self.gradient_norms[-100:]

        return grad_norm

    def reset_gradients(self):
        """重置梯度"""
        self.model.zero_grad()

    def step(self):
        """执行一步累积"""
        self.step_count += 1


class DynamicBatchSizer:
    """动态批处理大小管理器"""

    def __init__(self, config: MemoryConfig, memory_monitor: MemoryMonitor):
        self.config = config
        self.memory_monitor = memory_monitor
        self.enabled = config.adaptive_batch_size

        self.current_batch_size = config.max_batch_size if config.adaptive_batch_size else None
        self.min_batch_size = config.min_batch_size
        self.max_batch_size = config.max_batch_size
        self.reduction_factor = config.oom_batch_size_reduction

        self.oom_count = 0
        self.last_successful_batch_size = None

        if self.enabled:
            print(f"🔄 Dynamic batch sizing enabled: {self.min_batch_size}-{self.max_batch_size}")

    def handle_oom(self) -> Optional[int]:
        """处理OOM错误，返回新的批处理大小"""
        if not self.enabled or self.current_batch_size is None:
            return None

        self.oom_count += 1
        new_batch_size = max(
            self.min_batch_size,
            int(self.current_batch_size * self.reduction_factor)
        )

        print(f"⚠️  OOM detected! Reducing batch size: {self.current_batch_size} -> {new_batch_size}")

        # 强制清理内存
        self.memory_monitor.force_cleanup()

        self.current_batch_size = new_batch_size
        return new_batch_size

    def try_increase_batch_size(self) -> Optional[int]:
        """尝试增加批处理大小"""
        if not self.enabled or self.current_batch_size is None:
            return None

        # 检查内存压力
        if self.memory_monitor.check_memory_pressure(0.7):
            return None

        # 尝试增加批处理大小
        new_batch_size = min(
            self.max_batch_size,
            int(self.current_batch_size * 1.25)
        )

        if new_batch_size > self.current_batch_size:
            print(f"📈 Increasing batch size: {self.current_batch_size} -> {new_batch_size}")
            self.current_batch_size = new_batch_size
            return new_batch_size

        return None

    def get_current_batch_size(self) -> Optional[int]:
        """获取当前批处理大小"""
        return self.current_batch_size


class MemoryOptimizer:
    """综合内存优化管理器"""

    def __init__(self, model: nn.Module, config: MemoryConfig, device: torch.device):
        self.model = model
        self.config = config
        self.device = device

        # 初始化各个组件
        self.memory_monitor = MemoryMonitor(device)
        self.mixed_precision = MixedPrecisionManager(config, device)
        self.gradient_accumulator = GradientAccumulator(config, model)
        self.batch_sizer = DynamicBatchSizer(config, self.memory_monitor)

        # 启用梯度检查点
        if config.enable_gradient_checkpointing:
            self._enable_gradient_checkpointing()

        # 内存清理计数器
        self.step_count = 0

        print("🚀 MemoryOptimizer initialized successfully")
        print(self.memory_monitor.get_memory_summary())

    def _enable_gradient_checkpointing(self):
        """启用梯度检查点"""
        def apply_gradient_checkpointing(module):
            if hasattr(module, 'gradient_checkpointing'):
                module.gradient_checkpointing = True
            for child in module.children():
                apply_gradient_checkpointing(child)

        apply_gradient_checkpointing(self.model)
        print("✅ Gradient checkpointing enabled")

    @contextmanager
    def optimize_step_context(self, optimizer: optim.Optimizer):
        """优化步骤上下文管理器"""
        try:
            # 内存监控
            if self.step_count % 50 == 0:
                print(self.memory_monitor.get_memory_summary())

            # 检查内存压力
            if self.memory_monitor.check_memory_pressure(self.config.memory_fraction_threshold):
                print("⚠️  High memory pressure detected, forcing cleanup")
                self.memory_monitor.force_cleanup()

            # 混合精度上下文
            with self.mixed_precision.autocast_context():
                yield {
                    'should_update': self.gradient_accumulator.should_update(),
                    'current_batch_size': self.batch_sizer.get_current_batch_size()
                }

            # 梯度处理
            if self.gradient_accumulator.should_update():
                # 梯度裁剪
                grad_norm = self.gradient_accumulator.clip_gradients()

                # 优化器步骤
                self.mixed_precision.step_optimizer(optimizer)

                # 重置梯度
                self.gradient_accumulator.reset_gradients()

                return grad_norm

            # 累积步骤
            self.gradient_accumulator.step()

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                # 处理OOM
                new_batch_size = self.batch_sizer.handle_oom()
                if new_batch_size:
                    raise MemoryError(f"OOM handled, try batch_size={new_batch_size}")
                else:
                    raise e
            else:
                raise e

        finally:
            # 定期清理
            self.step_count += 1
            if self.step_count % self.config.clear_cache_frequency == 0:
                self.memory_monitor.force_cleanup()

    def compute_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """计算损失（带梯度累积）"""
        return self.gradient_accumulator.accumulate_gradients(loss)

    def backward(self, loss: torch.Tensor):
        """反向传播"""
        self.mixed_precision.backward(loss)

    def get_memory_stats(self) -> Dict[str, Any]:
        """获取内存统计信息"""
        return {
            'memory_info': self.memory_monitor.get_memory_info(),
            'amp_scale': self.mixed_precision.get_scale(),
            'gradient_accumulation_steps': self.gradient_accumulator.accumulation_steps,
            'current_batch_size': self.batch_sizer.get_current_batch_size(),
            'oom_count': self.batch_sizer.oom_count
        }

    def optimize_model_for_inference(self):
        """为推理优化模型"""
        print("🔧 Optimizing model for inference...")

        # 切换到评估模式
        self.model.eval()

        # 禁用梯度计算
        for param in self.model.parameters():
            param.requires_grad = False

        # 清理内存
        self.memory_monitor.force_cleanup()

        # 可选：模型量化（如果支持）
        if hasattr(torch.quantization, 'quantize_dynamic'):
            # 动态量化
            try:
                quantized_model = torch.quantization.quantize_dynamic(
                    self.model, {nn.Linear}, dtype=torch.qint8
                )
                print("✅ Model quantization applied")
                return quantized_model
            except Exception as e:
                print(f"⚠️  Quantization failed: {e}")
                return self.model

        return self.model

    def save_memory_profile(self, filepath: str):
        """保存内存使用分析"""
        if not self.config.enable_memory_profiling:
            return

        profile_data = {
            'config': {
                'enable_amp': self.config.enable_amp,
                'gradient_accumulation_steps': self.config.gradient_accumulation_steps,
                'enable_gradient_checkpointing': self.config.enable_gradient_checkpointing
            },
            'memory_history': self.memory_monitor.memory_history,
            'final_stats': self.get_memory_stats()
        }

        import json
        with open(filepath, 'w') as f:
            json.dump(profile_data, f, indent=2)

        print(f"💾 Memory profile saved to: {filepath}")


# 使用示例
if __name__ == "__main__":
    # 创建测试模型
    test_model = nn.Sequential(
        nn.Linear(1000, 500),
        nn.ReLU(),
        nn.Linear(500, 100),
        nn.ReLU(),
        nn.Linear(100, 10)
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_model.to(device)

    # 配置内存优化
    config = MemoryConfig(
        enable_amp=True,
        gradient_accumulation_steps=4,
        adaptive_batch_size=True,
        min_batch_size=8,
        max_batch_size=64,
        enable_gradient_checkpointing=True
    )

    # 初始化内存优化器
    optimizer_params = optim.AdamW(test_model.parameters(), lr=1e-3)
    memory_optimizer = MemoryOptimizer(test_model, config, device)

    print("🧪 Testing memory optimization...")

    # 模拟训练循环
    for step in range(20):
        try:
            with memory_optimizer.optimize_step_context(optimizer_params) as ctx:
                # 模拟批处理数据
                batch_size = ctx.get('current_batch_size', 32)
                x = torch.randn(batch_size, 1000, device=device)

                # 前向传播
                y = test_model(x)
                loss = y.sum()

                # 计算优化后的损失
                optimized_loss = memory_optimizer.compute_loss(loss)

                # 反向传播
                memory_optimizer.backward(optimized_loss)

                if ctx['should_update']:
                    print(f"Step {step}: Updated parameters, loss = {loss.item():.4f}")

        except MemoryError as e:
            print(f"Handled memory error: {e}")
            continue

    # 获取最终统计
    stats = memory_optimizer.get_memory_stats()
    print(f"\n📊 Final memory stats: {stats}")

    print("✅ Memory optimization test completed!")