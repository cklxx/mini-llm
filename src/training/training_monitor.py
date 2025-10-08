"""
综合训练监控和可视化系统
提供实时训练指标监控、性能分析、异常检测和可视化仪表板
"""
import os
import time
import json
import psutil
import threading
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.figure import Figure
import seaborn as sns

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter


@dataclass
class TrainingMetrics:
    """训练指标数据结构"""
    step: int
    epoch: int
    loss: float
    learning_rate: float
    grad_norm: float
    param_norm: float
    timestamp: float

    # 性能指标
    samples_per_sec: float = 0.0
    gpu_memory_used: float = 0.0
    cpu_usage: float = 0.0
    ram_usage: float = 0.0

    # 模型健康指标
    weight_update_ratio: float = 0.0
    activation_mean: float = 0.0
    activation_std: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class SystemMonitor:
    """系统性能监控器"""

    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available()
                                 else 'mps' if torch.backends.mps.is_available()
                                 else 'cpu')

    def get_gpu_memory_info(self) -> Tuple[float, float]:
        """获取GPU内存信息 (used_gb, total_gb)"""
        if self.device.type == 'cuda':
            return (
                torch.cuda.memory_allocated() / 1024**3,
                torch.cuda.max_memory_allocated() / 1024**3
            )
        elif self.device.type == 'mps':
            # MPS内存监控（近似）
            return (psutil.virtual_memory().used / 1024**3 * 0.3,
                   psutil.virtual_memory().total / 1024**3 * 0.3)
        else:
            return 0.0, 0.0

    def get_cpu_info(self) -> Tuple[float, float]:
        """获取CPU信息 (usage_percent, frequency_ghz)"""
        return (
            psutil.cpu_percent(interval=0.1),
            psutil.cpu_freq().current / 1000 if psutil.cpu_freq() else 0.0
        )

    def get_memory_info(self) -> Tuple[float, float]:
        """获取RAM信息 (used_gb, total_gb)"""
        mem = psutil.virtual_memory()
        return (mem.used / 1024**3, mem.total / 1024**3)

    def get_disk_info(self) -> Tuple[float, float]:
        """获取磁盘信息 (used_gb, total_gb)"""
        disk = psutil.disk_usage('/')
        return (disk.used / 1024**3, disk.total / 1024**3)


class ModelHealthMonitor:
    """模型健康监控器"""

    def __init__(self, model: nn.Module, lightweight_mode: bool = False):
        self.model = model
        self.lightweight_mode = lightweight_mode
        self.prev_params = None
        self.grad_history = deque(maxlen=100)
        self.param_history = deque(maxlen=100)

    def compute_gradient_norm(self) -> float:
        """计算梯度范数"""
        total_norm = 0.0
        param_count = 0

        for p in self.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
                param_count += 1

        return np.sqrt(total_norm) if param_count > 0 else 0.0

    def compute_parameter_norm(self) -> float:
        """计算参数范数"""
        total_norm = 0.0

        for p in self.model.parameters():
            param_norm = p.data.norm(2)
            total_norm += param_norm.item() ** 2

        return np.sqrt(total_norm)

    def compute_weight_update_ratio(self) -> float:
        """计算权重更新比例"""
        # 轻量级模式下跳过这个耗时操作
        if self.lightweight_mode:
            return 0.0
            
        if self.prev_params is None:
            self.prev_params = {name: param.clone() for name, param in self.model.named_parameters()}
            return 0.0

        update_norms = []
        param_norms = []

        for name, param in self.model.named_parameters():
            if name in self.prev_params:
                update = param.data - self.prev_params[name]
                update_norm = update.norm().item()
                param_norm = param.data.norm().item()

                if param_norm > 0:
                    update_norms.append(update_norm)
                    param_norms.append(param_norm)

                # 更新历史参数
                self.prev_params[name] = param.clone()

        if len(update_norms) > 0:
            return np.mean(update_norms) / np.mean(param_norms)
        else:
            return 0.0

    def detect_gradient_anomaly(self, grad_norm: float, step: int = 0) -> Dict[str, Any]:
        """检测梯度异常

        Args:
            grad_norm: 当前梯度范数
            step: 当前训练步数（用于动态调整阈值）
        """
        self.grad_history.append(grad_norm)

        if len(self.grad_history) < 10:
            return {'status': 'normal', 'reason': 'insufficient_data'}

        recent_grads = list(self.grad_history)[-10:]
        mean_grad = np.mean(recent_grads)
        std_grad = np.std(recent_grads)

        anomaly_info = {'status': 'normal', 'mean': mean_grad, 'std': std_grad}

        # 梯度爆炸检测
        if grad_norm > mean_grad + 3 * std_grad and grad_norm > 10.0:
            anomaly_info.update({
                'status': 'gradient_explosion',
                'current': grad_norm,
                'threshold': mean_grad + 3 * std_grad
            })

        # 梯度消失检测 - 动态阈值，训练初期更宽松
        # Step 100之前: 1e-10 (几乎不触发)
        # Step 100-1000: 1e-6 (宽松，适应阶段)
        # Step 1000之后: 1e-7 (正常训练中合理的梯度范围)
        if step < 100:
            vanishing_threshold = 1e-10
        elif step < 1000:
            vanishing_threshold = 1e-6
        else:
            vanishing_threshold = 1e-7
            
        if grad_norm < vanishing_threshold:
            anomaly_info.update({
                'status': 'gradient_vanishing',
                'current': grad_norm,
                'threshold': vanishing_threshold
            })

        return anomaly_info

    def get_activation_stats(self, activations: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """获取激活值统计"""
        stats = {}

        for name, tensor in activations.items():
            if tensor.numel() > 0:
                stats[f'{name}_mean'] = tensor.mean().item()
                stats[f'{name}_std'] = tensor.std().item()
                stats[f'{name}_max'] = tensor.max().item()
                stats[f'{name}_min'] = tensor.min().item()

        return stats


class RealTimeVisualizer:
    """实时可视化器"""

    def __init__(self, save_dir: str = "training_plots", max_points: int = 1000):
        self.save_dir = save_dir
        self.max_points = max_points
        os.makedirs(save_dir, exist_ok=True)

        # 数据存储
        self.metrics_history = deque(maxlen=max_points)

        # 设置绘图样式
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")

        # 创建图形
        self.fig = None
        self.axes = None
        self.animation = None

    def start_real_time_plot(self):
        """启动实时绘图"""
        self.fig, self.axes = plt.subplots(2, 3, figsize=(15, 10))
        self.fig.suptitle('Training Monitor Dashboard', fontsize=16)

        # 设置子图标题
        titles = [
            'Training Loss', 'Learning Rate', 'Gradient Norm',
            'GPU Memory Usage', 'CPU Usage', 'Training Speed'
        ]

        for ax, title in zip(self.axes.flat, titles):
            ax.set_title(title)
            ax.grid(True, alpha=0.3)

        # 启动动画
        self.animation = animation.FuncAnimation(
            self.fig, self._update_plots, interval=1000, blit=False
        )

        plt.tight_layout()
        return self.fig

    def _update_plots(self, frame):
        """更新绘图"""
        if len(self.metrics_history) < 2:
            return

        # 提取数据
        steps = [m.step for m in self.metrics_history]
        losses = [m.loss for m in self.metrics_history]
        lrs = [m.learning_rate for m in self.metrics_history]
        grad_norms = [m.grad_norm for m in self.metrics_history]
        gpu_memory = [m.gpu_memory_used for m in self.metrics_history]
        cpu_usage = [m.cpu_usage for m in self.metrics_history]
        speeds = [m.samples_per_sec for m in self.metrics_history]

        # 清除并重绘
        for ax in self.axes.flat:
            ax.clear()

        # 绘制各个指标
        plots_data = [
            (losses, 'Loss', 'red'),
            (lrs, 'Learning Rate', 'blue'),
            (grad_norms, 'Gradient Norm', 'green'),
            (gpu_memory, 'GPU Memory (GB)', 'orange'),
            (cpu_usage, 'CPU Usage (%)', 'purple'),
            (speeds, 'Samples/sec', 'brown')
        ]

        for ax, (data, ylabel, color) in zip(self.axes.flat, plots_data):
            if len(data) > 1:
                ax.plot(steps, data, color=color, linewidth=2)
                ax.set_ylabel(ylabel)
                ax.set_xlabel('Step')
                ax.grid(True, alpha=0.3)

                # 添加最新值标注
                if data:
                    ax.annotate(f'{data[-1]:.3f}',
                              xy=(steps[-1], data[-1]),
                              xytext=(10, 10), textcoords='offset points',
                              bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.3))

        plt.tight_layout()

    def add_metrics(self, metrics: TrainingMetrics):
        """添加新的指标数据"""
        self.metrics_history.append(metrics)

    def save_plots(self, prefix: str = "training_plots"):
        """保存图片"""
        if self.fig is not None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filepath = os.path.join(self.save_dir, f"{prefix}_{timestamp}.png")
            self.fig.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"📊 Plots saved to: {filepath}")

    def generate_summary_report(self) -> Dict[str, Any]:
        """生成训练总结报告"""
        if len(self.metrics_history) < 10:
            return {}

        metrics_list = list(self.metrics_history)

        # 计算统计信息
        losses = [m.loss for m in metrics_list]
        speeds = [m.samples_per_sec for m in metrics_list if m.samples_per_sec > 0]
        grad_norms = [m.grad_norm for m in metrics_list]

        report = {
            'training_summary': {
                'total_steps': len(metrics_list),
                'training_time_hours': (metrics_list[-1].timestamp - metrics_list[0].timestamp) / 3600,
                'final_loss': losses[-1],
                'min_loss': min(losses),
                'loss_reduction': (losses[0] - losses[-1]) / losses[0] * 100,
            },
            'performance_summary': {
                'avg_speed_samples_per_sec': np.mean(speeds) if speeds else 0,
                'max_speed_samples_per_sec': max(speeds) if speeds else 0,
                'avg_gradient_norm': np.mean(grad_norms),
                'max_gradient_norm': max(grad_norms),
            },
            'system_summary': {
                'avg_gpu_memory_gb': np.mean([m.gpu_memory_used for m in metrics_list]),
                'max_gpu_memory_gb': max([m.gpu_memory_used for m in metrics_list]),
                'avg_cpu_usage_percent': np.mean([m.cpu_usage for m in metrics_list]),
            }
        }

        return report


class TrainingMonitor:
    """综合训练监控器"""

    def __init__(self, model: nn.Module, log_dir: str = "training_logs",
                 enable_tensorboard: bool = True, enable_real_time_plots: bool = False,
                 lightweight_mode: bool = False, log_interval: int = 1):
        self.model = model
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

        # 轻量级模式配置
        self.lightweight_mode = lightweight_mode
        self.log_interval = log_interval if lightweight_mode else 1
        
        # 初始化各个监控组件
        self.system_monitor = SystemMonitor()
        self.health_monitor = ModelHealthMonitor(model, lightweight_mode=lightweight_mode)
        self.visualizer = RealTimeVisualizer(os.path.join(log_dir, "plots"))

        # TensorBoard - 支持自定义flush间隔
        if enable_tensorboard:
            flush_secs = 30
            if hasattr(model, 'config'):
                flush_secs = getattr(model.config, 'tensorboard_flush_secs', 30)
            self.tensorboard_writer = SummaryWriter(log_dir, flush_secs=flush_secs)
        else:
            self.tensorboard_writer = None

        # 实时绘图（轻量级模式下禁用）
        self.enable_real_time_plots = enable_real_time_plots and not lightweight_mode
        if self.enable_real_time_plots:
            self.plot_thread = None
            self._start_real_time_plotting()

        # 性能统计
        self.step_start_time = time.time()
        self.samples_processed = 0

        # 异常检测历史
        self.anomaly_history = []

        print(f"🔍 TrainingMonitor initialized:")
        print(f"   Log directory: {log_dir}")
        print(f"   TensorBoard: {'enabled' if enable_tensorboard else 'disabled'}")
        print(f"   Real-time plots: {'enabled' if self.enable_real_time_plots else 'disabled'}")
        print(f"   Lightweight mode: {'enabled' if lightweight_mode else 'disabled'}")
        if lightweight_mode:
            print(f"   Log interval: every {log_interval} steps")

    def _start_real_time_plotting(self):
        """启动实时绘图线程"""
        if self.enable_real_time_plots:
            def plot_worker():
                fig = self.visualizer.start_real_time_plot()
                plt.show()

            self.plot_thread = threading.Thread(target=plot_worker, daemon=True)
            self.plot_thread.start()

    def log_step(self, step: int, epoch: int, loss: float, learning_rate: float,
                batch_size: int = 1) -> Optional[TrainingMetrics]:
        """记录训练步骤"""
        # 轻量级模式下，只在指定间隔记录详细指标
        should_log_full = (step % self.log_interval == 0)
        
        current_time = time.time()

        # 计算性能指标
        step_duration = current_time - self.step_start_time
        samples_per_sec = batch_size / step_duration if step_duration > 0 else 0

        # 获取系统信息（轻量级：降低频率）
        if should_log_full:
            gpu_memory_used, _ = self.system_monitor.get_gpu_memory_info()
            cpu_usage, _ = self.system_monitor.get_cpu_info()
            ram_used, _ = self.system_monitor.get_memory_info()
        else:
            gpu_memory_used = cpu_usage = ram_used = 0.0

        # 获取模型健康指标
        grad_norm = self.health_monitor.compute_gradient_norm()
        param_norm = self.health_monitor.compute_parameter_norm()
        
        # 权重更新比例（最耗时，轻量级模式下跳过）
        if should_log_full:
            weight_update_ratio = self.health_monitor.compute_weight_update_ratio()
        else:
            weight_update_ratio = 0.0

        # 检测异常（始终检查，因为很重要）
        anomaly_info = self.health_monitor.detect_gradient_anomaly(grad_norm, step)
        if anomaly_info['status'] != 'normal':
            self.anomaly_history.append({
                'step': step,
                'timestamp': current_time,
                'anomaly': anomaly_info
            })
            print(f"⚠️  Anomaly detected at step {step}: {anomaly_info['status']}")

        # 创建指标对象
        metrics = TrainingMetrics(
            step=step,
            epoch=epoch,
            loss=loss,
            learning_rate=learning_rate,
            grad_norm=grad_norm,
            param_norm=param_norm,
            timestamp=current_time,
            samples_per_sec=samples_per_sec,
            gpu_memory_used=gpu_memory_used,
            cpu_usage=cpu_usage,
            ram_usage=ram_used,
            weight_update_ratio=weight_update_ratio
        )

        # 记录到各个系统（只在完整记录时写入详细信息）
        if should_log_full:
            self._log_to_tensorboard(metrics)
            self._log_to_console(metrics, step % 100 == 0)  # 每100步打印一次详细信息
            self.visualizer.add_metrics(metrics)
        else:
            # 轻量级：只记录关键指标到 TensorBoard
            if self.tensorboard_writer:
                self.tensorboard_writer.add_scalar('Training/Loss', loss, step)
                self.tensorboard_writer.add_scalar('Training/LearningRate', learning_rate, step)

        # 重置计时器
        self.step_start_time = current_time

        return metrics if should_log_full else None

    def _log_to_tensorboard(self, metrics: TrainingMetrics):
        """记录到TensorBoard"""
        if self.tensorboard_writer is None:
            return

        writer = self.tensorboard_writer
        step = metrics.step

        # 训练指标
        writer.add_scalar('Training/Loss', metrics.loss, step)
        writer.add_scalar('Training/LearningRate', metrics.learning_rate, step)
        writer.add_scalar('Training/GradientNorm', metrics.grad_norm, step)
        writer.add_scalar('Training/ParameterNorm', metrics.param_norm, step)
        writer.add_scalar('Training/WeightUpdateRatio', metrics.weight_update_ratio, step)

        # 性能指标
        writer.add_scalar('Performance/SamplesPerSec', metrics.samples_per_sec, step)
        writer.add_scalar('Performance/GPUMemoryGB', metrics.gpu_memory_used, step)
        writer.add_scalar('Performance/CPUUsagePercent', metrics.cpu_usage, step)
        writer.add_scalar('Performance/RAMUsageGB', metrics.ram_usage, step)

    def _log_to_console(self, metrics: TrainingMetrics, verbose: bool = False):
        """记录到控制台"""
        if verbose:
            print(f"\n📊 Step {metrics.step} (Epoch {metrics.epoch}) Metrics:")
            print(f"   Loss: {metrics.loss:.4f}")
            print(f"   LR: {metrics.learning_rate:.2e}")
            print(f"   Grad Norm: {metrics.grad_norm:.4f}")
            print(f"   Speed: {metrics.samples_per_sec:.1f} samples/sec")
            print(f"   GPU Memory: {metrics.gpu_memory_used:.1f}GB")
            print(f"   CPU: {metrics.cpu_usage:.1f}%")

            if metrics.weight_update_ratio > 0:
                print(f"   Weight Update Ratio: {metrics.weight_update_ratio:.2e}")

    def save_training_summary(self):
        """保存训练总结"""
        summary = self.visualizer.generate_summary_report()

        if summary:
            summary_file = os.path.join(self.log_dir, "training_summary.json")
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)

            print(f"📋 Training summary saved to: {summary_file}")

            # 打印关键统计信息
            print("\n📈 Training Summary:")
            training_summary = summary.get('training_summary', {})
            performance_summary = summary.get('performance_summary', {})

            print(f"   Total Steps: {training_summary.get('total_steps', 0)}")
            print(f"   Training Time: {training_summary.get('training_time_hours', 0):.2f} hours")
            print(f"   Final Loss: {training_summary.get('final_loss', 0):.4f}")
            print(f"   Loss Reduction: {training_summary.get('loss_reduction', 0):.1f}%")
            print(f"   Avg Speed: {performance_summary.get('avg_speed_samples_per_sec', 0):.1f} samples/sec")

    def close(self):
        """关闭监控器"""
        if self.tensorboard_writer:
            self.tensorboard_writer.close()

        self.visualizer.save_plots()
        self.save_training_summary()

        print("🔍 TrainingMonitor closed successfully")


# 使用示例和测试代码
if __name__ == "__main__":
    # 创建一个简单的测试模型
    test_model = nn.Sequential(
        nn.Linear(100, 50),
        nn.ReLU(),
        nn.Linear(50, 10)
    )

    # 初始化监控器
    monitor = TrainingMonitor(
        test_model,
        log_dir="test_logs",
        enable_tensorboard=True,
        enable_real_time_plots=False  # 在测试中禁用实时绘图
    )

    # 模拟训练过程
    print("🧪 Simulating training process...")

    for step in range(100):
        # 模拟一次前向传播
        x = torch.randn(32, 100)
        y = test_model(x)
        loss = y.sum()

        # 模拟反向传播
        loss.backward()

        # 记录指标
        metrics = monitor.log_step(
            step=step,
            epoch=step // 20,
            loss=loss.item(),
            learning_rate=0.001 * (0.95 ** (step // 10)),
            batch_size=32
        )

        # 清除梯度
        test_model.zero_grad()

        time.sleep(0.01)  # 模拟训练时间

    # 关闭监控器
    monitor.close()
    print("✅ Test completed successfully!")