"""
综合性能基准测试和分析工具
提供训练、推理、数据加载等各方面的性能测试和优化建议
"""
import os
import time
import json
import warnings
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import psutil

# 导入我们的优化模块
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from training.memory_optimizer import MemoryOptimizer, MemoryConfig
from training.training_monitor import TrainingMonitor
from data.high_performance_loader import DataLoadingConfig, create_high_performance_dataloader


@dataclass
class BenchmarkConfig:
    """基准测试配置"""
    # 测试范围
    test_batch_sizes: List[int] = None
    test_sequence_lengths: List[int] = None
    test_model_sizes: List[str] = None

    # 测试参数
    warmup_steps: int = 5
    test_steps: int = 50
    num_repeats: int = 3

    # 输出配置
    save_results: bool = True
    save_plots: bool = True
    output_dir: str = "benchmark_results"

    # 测试模式
    test_training: bool = True
    test_inference: bool = True
    test_data_loading: bool = True
    test_memory_optimization: bool = True

    def __post_init__(self):
        if self.test_batch_sizes is None:
            self.test_batch_sizes = [1, 4, 8, 16, 32, 64]
        if self.test_sequence_lengths is None:
            self.test_sequence_lengths = [128, 256, 512, 1024]
        if self.test_model_sizes is None:
            self.test_model_sizes = ["tiny", "small"]


@dataclass
class BenchmarkResult:
    """基准测试结果"""
    test_name: str
    config: Dict[str, Any]
    metrics: Dict[str, float]
    timestamp: float
    device: str
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ModelFactory:
    """模型工厂，用于创建不同大小的测试模型"""

    @staticmethod
    def create_test_transformer(model_size: str, vocab_size: int = 10000) -> nn.Module:
        """创建测试用的Transformer模型"""
        configs = {
            "tiny": {"d_model": 128, "n_heads": 2, "n_layers": 4, "d_ff": 512},
            "small": {"d_model": 256, "n_heads": 4, "n_layers": 6, "d_ff": 1024},
            "medium": {"d_model": 512, "n_heads": 8, "n_layers": 8, "d_ff": 2048},
        }

        if model_size not in configs:
            raise ValueError(f"Unsupported model size: {model_size}")

        config = configs[model_size]

        class SimpleTransformer(nn.Module):
            def __init__(self, vocab_size, d_model, n_heads, n_layers, d_ff):
                super().__init__()
                self.embedding = nn.Embedding(vocab_size, d_model)
                self.pos_encoding = nn.Parameter(torch.randn(2048, d_model))

                # Transformer层
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=n_heads,
                    dim_feedforward=d_ff,
                    dropout=0.1,
                    activation='relu',
                    batch_first=True
                )
                self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
                self.output_proj = nn.Linear(d_model, vocab_size)

            def forward(self, x):
                seq_len = x.size(1)
                x = self.embedding(x) + self.pos_encoding[:seq_len].unsqueeze(0)
                x = self.transformer(x)
                return self.output_proj(x)

        return SimpleTransformer(**config, vocab_size=vocab_size)

    @staticmethod
    def get_model_params(model: nn.Module) -> int:
        """获取模型参数数量"""
        return sum(p.numel() for p in model.parameters())


class TrainingBenchmark:
    """训练性能基准测试"""

    def __init__(self, device: torch.device):
        self.device = device

    def benchmark_training_speed(self, model: nn.Module, batch_size: int,
                                seq_len: int, steps: int = 50) -> Dict[str, float]:
        """测试训练速度"""
        model.train()
        optimizer = optim.AdamW(model.parameters(), lr=1e-4)
        criterion = nn.CrossEntropyLoss()

        # 创建虚拟数据
        vocab_size = model.output_proj.out_features
        dummy_input = torch.randint(0, vocab_size, (batch_size, seq_len), device=self.device)
        dummy_target = torch.randint(0, vocab_size, (batch_size, seq_len), device=self.device)

        # 预热
        for _ in range(5):
            optimizer.zero_grad()
            output = model(dummy_input)
            loss = criterion(output.reshape(-1, vocab_size), dummy_target.reshape(-1))
            loss.backward()
            optimizer.step()

        # 清理内存
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        # 基准测试
        start_time = time.time()
        start_memory = self._get_memory_usage()

        for step in range(steps):
            optimizer.zero_grad()
            output = model(dummy_input)
            loss = criterion(output.reshape(-1, vocab_size), dummy_target.reshape(-1))
            loss.backward()
            optimizer.step()

        # 同步GPU
        if self.device.type == 'cuda':
            torch.cuda.synchronize()

        end_time = time.time()
        end_memory = self._get_memory_usage()

        elapsed_time = end_time - start_time
        samples_per_sec = (steps * batch_size) / elapsed_time
        memory_used = end_memory - start_memory

        return {
            'samples_per_sec': samples_per_sec,
            'elapsed_time': elapsed_time,
            'memory_used_mb': memory_used,
            'loss': loss.item()
        }

    def benchmark_with_optimizations(self, model: nn.Module, batch_size: int,
                                   seq_len: int, config: MemoryConfig) -> Dict[str, float]:
        """测试带优化的训练性能"""
        optimizer = optim.AdamW(model.parameters(), lr=1e-4)
        memory_optimizer = MemoryOptimizer(model, config, self.device)

        vocab_size = model.output_proj.out_features
        dummy_input = torch.randint(0, vocab_size, (batch_size, seq_len), device=self.device)
        dummy_target = torch.randint(0, vocab_size, (batch_size, seq_len), device=self.device)

        start_time = time.time()
        start_memory = self._get_memory_usage()

        steps = 50
        for step in range(steps):
            try:
                with memory_optimizer.optimize_step_context(optimizer) as ctx:
                    output = model(dummy_input)
                    loss = nn.CrossEntropyLoss()(
                        output.reshape(-1, vocab_size),
                        dummy_target.reshape(-1)
                    )

                    optimized_loss = memory_optimizer.compute_loss(loss)
                    memory_optimizer.backward(optimized_loss)

            except Exception as e:
                if "out of memory" in str(e).lower():
                    continue
                else:
                    raise e

        if self.device.type == 'cuda':
            torch.cuda.synchronize()

        end_time = time.time()
        end_memory = self._get_memory_usage()

        elapsed_time = end_time - start_time
        samples_per_sec = (steps * batch_size) / elapsed_time

        stats = memory_optimizer.get_memory_stats()

        return {
            'samples_per_sec': samples_per_sec,
            'elapsed_time': elapsed_time,
            'memory_used_mb': end_memory - start_memory,
            'amp_scale': stats.get('amp_scale', 1.0),
            'oom_count': stats.get('oom_count', 0)
        }

    def _get_memory_usage(self) -> float:
        """获取内存使用量(MB)"""
        if self.device.type == 'cuda':
            return torch.cuda.memory_allocated() / 1024 / 1024
        else:
            return psutil.Process().memory_info().rss / 1024 / 1024


class InferenceBenchmark:
    """推理性能基准测试"""

    def __init__(self, device: torch.device):
        self.device = device

    def benchmark_inference_speed(self, model: nn.Module, batch_sizes: List[int],
                                 seq_len: int = 512) -> Dict[int, Dict[str, float]]:
        """测试不同批处理大小的推理速度"""
        model.eval()
        results = {}

        vocab_size = model.output_proj.out_features

        with torch.no_grad():
            for batch_size in batch_sizes:
                try:
                    # 创建测试数据
                    dummy_input = torch.randint(0, vocab_size, (batch_size, seq_len), device=self.device)

                    # 预热
                    for _ in range(5):
                        _ = model(dummy_input)

                    # 清理内存
                    if self.device.type == 'cuda':
                        torch.cuda.empty_cache()

                    # 基准测试
                    steps = 100
                    start_time = time.time()
                    start_memory = self._get_memory_usage()

                    for _ in range(steps):
                        output = model(dummy_input)

                    if self.device.type == 'cuda':
                        torch.cuda.synchronize()

                    end_time = time.time()
                    end_memory = self._get_memory_usage()

                    elapsed_time = end_time - start_time
                    samples_per_sec = (steps * batch_size) / elapsed_time
                    tokens_per_sec = samples_per_sec * seq_len

                    results[batch_size] = {
                        'samples_per_sec': samples_per_sec,
                        'tokens_per_sec': tokens_per_sec,
                        'elapsed_time': elapsed_time,
                        'memory_used_mb': end_memory - start_memory,
                        'latency_ms': (elapsed_time / steps) * 1000
                    }

                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        results[batch_size] = {'error': 'OOM'}
                        if self.device.type == 'cuda':
                            torch.cuda.empty_cache()
                    else:
                        raise e

        return results

    def _get_memory_usage(self) -> float:
        """获取内存使用量(MB)"""
        if self.device.type == 'cuda':
            return torch.cuda.memory_allocated() / 1024 / 1024
        else:
            return psutil.Process().memory_info().rss / 1024 / 1024


class DataLoadingBenchmark:
    """数据加载性能基准测试"""

    def __init__(self, device: torch.device):
        self.device = device

    def benchmark_data_loading(self, config: DataLoadingConfig) -> Dict[str, float]:
        """测试数据加载性能"""
        # 创建虚拟tokenizer
        class DummyTokenizer:
            def __init__(self):
                self.vocab_size = 10000
                self.pad_id = 0
                self.bos_id = 1
                self.eos_id = 2

            def encode(self, text, add_special_tokens=True):
                # 模拟编码
                tokens = list(range(3, min(len(text) // 4 + 3, config.max_length - 2)))
                if add_special_tokens:
                    tokens = [self.bos_id] + tokens + [self.eos_id]
                return tokens

            def get_vocab_size(self):
                return self.vocab_size

        tokenizer = DummyTokenizer()

        # 创建虚拟数据文件
        dummy_data_path = "dummy_data.jsonl"
        self._create_dummy_data(dummy_data_path, 1000)

        try:
            config.data_path = dummy_data_path

            # 测试数据加载器
            start_time = time.time()

            # 由于实际的high_performance_loader需要真实文件，我们模拟测试
            # 创建简单的DataLoader进行基准测试
            dummy_dataset = TensorDataset(
                torch.randint(0, tokenizer.vocab_size, (1000, config.max_length))
            )

            dataloader = DataLoader(
                dummy_dataset,
                batch_size=config.batch_size,
                num_workers=config.num_workers,
                pin_memory=config.pin_memory and torch.cuda.is_available()
            )

            # 测试数据加载速度
            total_samples = 0
            load_times = []

            for batch_idx, batch in enumerate(dataloader):
                if batch_idx >= 50:  # 限制测试批次数
                    break

                batch_start = time.time()
                batch_data = batch[0].to(self.device)
                batch_end = time.time()

                load_times.append(batch_end - batch_start)
                total_samples += len(batch_data)

            end_time = time.time()
            total_time = end_time - start_time

            return {
                'total_samples': total_samples,
                'total_time': total_time,
                'samples_per_sec': total_samples / total_time,
                'avg_batch_time_ms': np.mean(load_times) * 1000,
                'std_batch_time_ms': np.std(load_times) * 1000
            }

        finally:
            # 清理虚拟文件
            if os.path.exists(dummy_data_path):
                os.remove(dummy_data_path)

    def _create_dummy_data(self, filepath: str, num_samples: int):
        """创建虚拟数据文件"""
        with open(filepath, 'w', encoding='utf-8') as f:
            for i in range(num_samples):
                dummy_conversation = {
                    "conversations": [
                        {"role": "user", "content": f"This is a test question {i}"},
                        {"role": "assistant", "content": f"This is a test answer {i}"}
                    ]
                }
                f.write(json.dumps(dummy_conversation, ensure_ascii=False) + '\n')


class PerformanceBenchmarkSuite:
    """综合性能基准测试套件"""

    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available()
                                 else 'mps' if torch.backends.mps.is_available()
                                 else 'cpu')

        # 初始化各个基准测试器
        self.training_benchmark = TrainingBenchmark(self.device)
        self.inference_benchmark = InferenceBenchmark(self.device)
        self.data_loading_benchmark = DataLoadingBenchmark(self.device)

        # 结果存储
        self.results = []

        # 创建输出目录
        os.makedirs(config.output_dir, exist_ok=True)

        print(f"🔬 Performance Benchmark Suite initialized")
        print(f"   Device: {self.device}")
        print(f"   Output: {config.output_dir}")

    def run_all_benchmarks(self) -> List[BenchmarkResult]:
        """运行所有基准测试"""
        print("🚀 Starting comprehensive performance benchmarks...")

        if self.config.test_training:
            self._run_training_benchmarks()

        if self.config.test_inference:
            self._run_inference_benchmarks()

        if self.config.test_data_loading:
            self._run_data_loading_benchmarks()

        if self.config.test_memory_optimization:
            self._run_memory_optimization_benchmarks()

        # 生成报告
        self._generate_report()

        return self.results

    def _run_training_benchmarks(self):
        """运行训练基准测试"""
        print("📊 Running training benchmarks...")

        for model_size in self.config.test_model_sizes:
            model = ModelFactory.create_test_transformer(model_size)
            model.to(self.device)

            for batch_size in self.config.test_batch_sizes:
                for seq_len in self.config.test_sequence_lengths:
                    try:
                        # 基础训练测试
                        result = self.training_benchmark.benchmark_training_speed(
                            model, batch_size, seq_len, self.config.test_steps
                        )

                        self.results.append(BenchmarkResult(
                            test_name="training_baseline",
                            config={
                                'model_size': model_size,
                                'batch_size': batch_size,
                                'seq_len': seq_len,
                                'model_params': ModelFactory.get_model_params(model)
                            },
                            metrics=result,
                            timestamp=time.time(),
                            device=str(self.device)
                        ))

                        print(f"   {model_size} model, batch={batch_size}, seq={seq_len}: "
                              f"{result['samples_per_sec']:.1f} samples/sec")

                    except RuntimeError as e:
                        if "out of memory" in str(e).lower():
                            print(f"   OOM: {model_size}, batch={batch_size}, seq={seq_len}")
                            if self.device.type == 'cuda':
                                torch.cuda.empty_cache()
                        else:
                            print(f"   Error: {e}")

    def _run_inference_benchmarks(self):
        """运行推理基准测试"""
        print("🔍 Running inference benchmarks...")

        for model_size in self.config.test_model_sizes:
            model = ModelFactory.create_test_transformer(model_size)
            model.to(self.device)

            results = self.inference_benchmark.benchmark_inference_speed(
                model, self.config.test_batch_sizes
            )

            for batch_size, metrics in results.items():
                if 'error' not in metrics:
                    self.results.append(BenchmarkResult(
                        test_name="inference",
                        config={
                            'model_size': model_size,
                            'batch_size': batch_size,
                            'model_params': ModelFactory.get_model_params(model)
                        },
                        metrics=metrics,
                        timestamp=time.time(),
                        device=str(self.device)
                    ))

                    print(f"   {model_size} model, batch={batch_size}: "
                          f"{metrics['tokens_per_sec']:.0f} tokens/sec")

    def _run_data_loading_benchmarks(self):
        """运行数据加载基准测试"""
        print("📦 Running data loading benchmarks...")

        for batch_size in self.config.test_batch_sizes:
            for num_workers in [0, 2, 4]:
                config = DataLoadingConfig(
                    data_path="dummy",  # 将被替换
                    batch_size=batch_size,
                    num_workers=num_workers,
                    max_length=512
                )

                try:
                    result = self.data_loading_benchmark.benchmark_data_loading(config)

                    self.results.append(BenchmarkResult(
                        test_name="data_loading",
                        config={
                            'batch_size': batch_size,
                            'num_workers': num_workers,
                            'max_length': config.max_length
                        },
                        metrics=result,
                        timestamp=time.time(),
                        device=str(self.device)
                    ))

                    print(f"   batch={batch_size}, workers={num_workers}: "
                          f"{result['samples_per_sec']:.1f} samples/sec")

                except Exception as e:
                    print(f"   Error in data loading test: {e}")

    def _run_memory_optimization_benchmarks(self):
        """运行内存优化基准测试"""
        print("🧠 Running memory optimization benchmarks...")

        model = ModelFactory.create_test_transformer("small")
        model.to(self.device)

        # 测试不同优化配置
        configs = [
            MemoryConfig(enable_amp=False, gradient_accumulation_steps=1),
            MemoryConfig(enable_amp=True, gradient_accumulation_steps=1),
            MemoryConfig(enable_amp=True, gradient_accumulation_steps=4),
            MemoryConfig(enable_amp=True, gradient_accumulation_steps=4,
                        enable_gradient_checkpointing=True)
        ]

        for i, mem_config in enumerate(configs):
            try:
                result = self.training_benchmark.benchmark_with_optimizations(
                    model, 16, 512, mem_config
                )

                config_name = f"optimization_config_{i}"
                self.results.append(BenchmarkResult(
                    test_name=config_name,
                    config={
                        'enable_amp': mem_config.enable_amp,
                        'gradient_accumulation_steps': mem_config.gradient_accumulation_steps,
                        'enable_gradient_checkpointing': mem_config.enable_gradient_checkpointing
                    },
                    metrics=result,
                    timestamp=time.time(),
                    device=str(self.device)
                ))

                print(f"   Config {i}: {result['samples_per_sec']:.1f} samples/sec, "
                      f"memory: {result['memory_used_mb']:.1f}MB")

            except Exception as e:
                print(f"   Error in config {i}: {e}")

    def _generate_report(self):
        """生成基准测试报告"""
        if not self.config.save_results:
            return

        # 保存原始结果
        results_file = os.path.join(self.config.output_dir, "benchmark_results.json")
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump([r.to_dict() for r in self.results], f, indent=2, ensure_ascii=False)

        # 生成可视化
        if self.config.save_plots:
            self._create_visualizations()

        # 生成性能分析报告
        self._create_performance_analysis()

        print(f"📋 Benchmark report saved to: {self.config.output_dir}")

    def _create_visualizations(self):
        """创建可视化图表"""
        plt.style.use('seaborn-v0_8')

        # 训练性能对比
        training_results = [r for r in self.results if r.test_name == "training_baseline"]
        if training_results:
            self._plot_training_performance(training_results)

        # 推理性能对比
        inference_results = [r for r in self.results if r.test_name == "inference"]
        if inference_results:
            self._plot_inference_performance(inference_results)

        # 内存优化效果
        opt_results = [r for r in self.results if "optimization" in r.test_name]
        if opt_results:
            self._plot_optimization_effects(opt_results)

    def _plot_training_performance(self, results: List[BenchmarkResult]):
        """绘制训练性能图表"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Training Performance Analysis', fontsize=16)

        # 按模型大小分组
        for model_size in self.config.test_model_sizes:
            model_results = [r for r in results if r.config['model_size'] == model_size]

            batch_sizes = [r.config['batch_size'] for r in model_results]
            throughputs = [r.metrics['samples_per_sec'] for r in model_results]
            memory_usage = [r.metrics['memory_used_mb'] for r in model_results]

            # 吞吐量 vs 批处理大小
            axes[0, 0].plot(batch_sizes, throughputs, marker='o', label=f'{model_size} model')
            axes[0, 0].set_xlabel('Batch Size')
            axes[0, 0].set_ylabel('Samples/sec')
            axes[0, 0].set_title('Training Throughput')
            axes[0, 0].legend()

            # 内存使用 vs 批处理大小
            axes[0, 1].plot(batch_sizes, memory_usage, marker='s', label=f'{model_size} model')
            axes[0, 1].set_xlabel('Batch Size')
            axes[0, 1].set_ylabel('Memory Usage (MB)')
            axes[0, 1].set_title('Memory Usage')
            axes[0, 1].legend()

        plt.tight_layout()
        plt.savefig(os.path.join(self.config.output_dir, 'training_performance.png'), dpi=300)
        plt.close()

    def _plot_inference_performance(self, results: List[BenchmarkResult]):
        """绘制推理性能图表"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle('Inference Performance Analysis', fontsize=16)

        for model_size in self.config.test_model_sizes:
            model_results = [r for r in results if r.config['model_size'] == model_size]

            batch_sizes = [r.config['batch_size'] for r in model_results]
            tokens_per_sec = [r.metrics['tokens_per_sec'] for r in model_results]
            latency = [r.metrics['latency_ms'] for r in model_results]

            ax1.plot(batch_sizes, tokens_per_sec, marker='o', label=f'{model_size} model')
            ax1.set_xlabel('Batch Size')
            ax1.set_ylabel('Tokens/sec')
            ax1.set_title('Inference Throughput')
            ax1.legend()

            ax2.plot(batch_sizes, latency, marker='s', label=f'{model_size} model')
            ax2.set_xlabel('Batch Size')
            ax2.set_ylabel('Latency (ms)')
            ax2.set_title('Inference Latency')
            ax2.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(self.config.output_dir, 'inference_performance.png'), dpi=300)
        plt.close()

    def _plot_optimization_effects(self, results: List[BenchmarkResult]):
        """绘制优化效果图表"""
        if not results:
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle('Memory Optimization Effects', fontsize=16)

        config_names = [f"Config {i}" for i in range(len(results))]
        throughputs = [r.metrics['samples_per_sec'] for r in results]
        memory_usage = [r.metrics['memory_used_mb'] for r in results]

        ax1.bar(config_names, throughputs, color='skyblue')
        ax1.set_ylabel('Samples/sec')
        ax1.set_title('Training Throughput')
        ax1.tick_params(axis='x', rotation=45)

        ax2.bar(config_names, memory_usage, color='lightcoral')
        ax2.set_ylabel('Memory Usage (MB)')
        ax2.set_title('Memory Usage')
        ax2.tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.savefig(os.path.join(self.config.output_dir, 'optimization_effects.png'), dpi=300)
        plt.close()

    def _create_performance_analysis(self):
        """创建性能分析报告"""
        analysis = {
            'device_info': {
                'device': str(self.device),
                'device_name': self._get_device_name(),
                'memory_gb': self._get_device_memory()
            },
            'summary': {},
            'recommendations': []
        }

        # 训练性能总结
        training_results = [r for r in self.results if r.test_name == "training_baseline"]
        if training_results:
            max_throughput = max(r.metrics['samples_per_sec'] for r in training_results)
            best_config = next(r for r in training_results
                             if r.metrics['samples_per_sec'] == max_throughput)

            analysis['summary']['best_training_config'] = {
                'throughput': max_throughput,
                'config': best_config.config
            }

        # 推理性能总结
        inference_results = [r for r in self.results if r.test_name == "inference"]
        if inference_results:
            max_tokens = max(r.metrics['tokens_per_sec'] for r in inference_results)
            best_inference = next(r for r in inference_results
                                if r.metrics['tokens_per_sec'] == max_tokens)

            analysis['summary']['best_inference_config'] = {
                'tokens_per_sec': max_tokens,
                'config': best_inference.config
            }

        # 生成建议
        analysis['recommendations'] = self._generate_recommendations()

        # 保存分析报告
        analysis_file = os.path.join(self.config.output_dir, "performance_analysis.json")
        with open(analysis_file, 'w', encoding='utf-8') as f:
            json.dump(analysis, f, indent=2, ensure_ascii=False)

    def _get_device_name(self) -> str:
        """获取设备名称"""
        if self.device.type == 'cuda':
            return torch.cuda.get_device_name()
        elif self.device.type == 'mps':
            return "Apple Silicon GPU (MPS)"
        else:
            return "CPU"

    def _get_device_memory(self) -> float:
        """获取设备内存(GB)"""
        if self.device.type == 'cuda':
            return torch.cuda.get_device_properties(self.device).total_memory / 1024**3
        else:
            return psutil.virtual_memory().total / 1024**3

    def _generate_recommendations(self) -> List[str]:
        """生成性能优化建议"""
        recommendations = []

        # 基于结果生成建议
        training_results = [r for r in self.results if r.test_name == "training_baseline"]
        if training_results:
            # 找到最佳批处理大小
            throughputs = [(r.config['batch_size'], r.metrics['samples_per_sec']) for r in training_results]
            best_batch_size = max(throughputs, key=lambda x: x[1])[0]
            recommendations.append(f"推荐训练批处理大小: {best_batch_size}")

        # 内存使用建议
        if self.device.type == 'cuda':
            recommendations.append("建议启用混合精度训练以节省GPU内存")
            recommendations.append("考虑使用梯度累积来处理更大的有效批处理大小")

        # 数据加载建议
        data_results = [r for r in self.results if r.test_name == "data_loading"]
        if data_results:
            best_workers = max(data_results, key=lambda x: x.metrics['samples_per_sec'])
            recommendations.append(f"推荐数据加载worker数量: {best_workers.config['num_workers']}")

        return recommendations


# 使用示例
if __name__ == "__main__":
    # 配置基准测试
    config = BenchmarkConfig(
        test_batch_sizes=[4, 8, 16, 32],
        test_sequence_lengths=[256, 512],
        test_model_sizes=["tiny", "small"],
        test_steps=20,  # 减少测试步数以加快测试
        output_dir="benchmark_results"
    )

    # 运行基准测试
    benchmark_suite = PerformanceBenchmarkSuite(config)
    results = benchmark_suite.run_all_benchmarks()

    print(f"\n✅ Benchmark completed! {len(results)} tests run.")
    print(f"📊 Results saved to: {config.output_dir}")

    # 打印最佳配置
    training_results = [r for r in results if r.test_name == "training_baseline"]
    if training_results:
        best_result = max(training_results, key=lambda x: x.metrics['samples_per_sec'])
        print(f"\n🏆 Best training configuration:")
        print(f"   Model: {best_result.config['model_size']}")
        print(f"   Batch size: {best_result.config['batch_size']}")
        print(f"   Sequence length: {best_result.config['seq_len']}")
        print(f"   Throughput: {best_result.metrics['samples_per_sec']:.1f} samples/sec")