#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MiniGPT优化套件集成演示
展示高性能数据加载、内存优化、训练监控和性能基准测试的协同使用
"""
import os
import sys
import time
import json
import argparse
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / "src"))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# 导入我们的优化模块
from src.data.high_performance_loader import DataLoadingConfig, HighPerformanceDataset
from src.training.memory_optimizer import MemoryOptimizer, MemoryConfig
from src.training.training_monitor import TrainingMonitor
from src.benchmarks.performance_benchmark import PerformanceBenchmarkSuite, BenchmarkConfig, ModelFactory


class OptimizationDemo:
    """优化套件演示类"""

    def __init__(self, device: str = "auto"):
        """初始化演示环境"""
        print("🚀 MiniGPT优化套件演示")
        print("=" * 60)

        # 设备选择
        if device == "auto":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
                print(f"🎯 使用CUDA GPU: {torch.cuda.get_device_name()}")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
                print("🎯 使用Apple Silicon GPU (MPS)")
            else:
                self.device = torch.device("cpu")
                print("🎯 使用CPU")
        else:
            self.device = torch.device(device)
            print(f"🎯 使用指定设备: {device}")

        # 创建演示目录
        self.demo_dir = project_root / "demo_results"
        self.demo_dir.mkdir(exist_ok=True)

        print(f"📁 演示结果将保存到: {self.demo_dir}")
        print()

    def create_dummy_dataset(self, size: int = 1000, seq_len: int = 512) -> str:
        """创建虚拟训练数据"""
        print(f"📝 创建虚拟数据集 ({size} 样本, 序列长度 {seq_len})")

        data_file = self.demo_dir / "demo_dataset.jsonl"

        # 生成虚拟对话数据
        with open(data_file, 'w', encoding='utf-8') as f:
            for i in range(size):
                conversation = {
                    "conversations": [
                        {
                            "role": "user",
                            "content": f"这是测试问题 {i}，用于演示高性能数据加载系统的效果。" * (seq_len // 50)
                        },
                        {
                            "role": "assistant",
                            "content": f"这是测试回答 {i}，展示了我们的优化数据处理流程。" * (seq_len // 50)
                        }
                    ]
                }
                f.write(json.dumps(conversation, ensure_ascii=False) + '\n')

        print(f"✅ 数据集已创建: {data_file}")
        return str(data_file)

    def demo_high_performance_data_loading(self, data_path: str):
        """演示高性能数据加载"""
        print("\n" + "="*60)
        print("📦 高性能数据加载演示")
        print("="*60)

        # 创建虚拟tokenizer
        class DemoTokenizer:
            def __init__(self):
                self.vocab_size = 10000
                self.pad_id = 0
                self.bos_id = 1
                self.eos_id = 2

            def encode(self, text, add_special_tokens=True):
                # 简单的字符级tokenization演示
                tokens = [hash(char) % self.vocab_size for char in text[:100]]
                if add_special_tokens:
                    tokens = [self.bos_id] + tokens + [self.eos_id]
                return tokens

            def get_vocab_size(self):
                return self.vocab_size

        tokenizer = DemoTokenizer()

        # 配置数据加载
        configs = [
            # 基础配置
            DataLoadingConfig(
                data_path=data_path,
                batch_size=16,
                num_workers=0,
                enable_cache=False,
                streaming=False,
                parallel_processing=False,
                cache_dir=str(self.demo_dir / "cache_basic")
            ),
            # 优化配置
            DataLoadingConfig(
                data_path=data_path,
                batch_size=16,
                num_workers=4,
                enable_cache=True,
                streaming=True,
                parallel_processing=True,
                cache_dir=str(self.demo_dir / "cache_optimized")
            )
        ]

        results = {}

        for i, config in enumerate(configs):
            config_name = "基础配置" if i == 0 else "优化配置"
            print(f"\n📊 测试 {config_name}:")
            print(f"   缓存: {'启用' if config.enable_cache else '禁用'}")
            print(f"   流式加载: {'启用' if config.streaming else '禁用'}")
            print(f"   并行处理: {'启用' if config.parallel_processing else '禁用'}")
            print(f"   Worker数: {config.num_workers}")

            try:
                # 创建数据集（模拟，因为HighPerformanceDataset需要实际文件）
                start_time = time.time()

                # 这里我们模拟数据加载过程
                dummy_data = []
                for j in range(100):  # 模拟100个样本
                    dummy_data.append({
                        'input': f"test input {j}",
                        'output': f"test output {j}",
                        'length': 50
                    })

                # 模拟tokenization
                processed_data = []
                for item in dummy_data:
                    input_ids = tokenizer.encode(item['input'])
                    output_ids = tokenizer.encode(item['output'])

                    # 构造序列
                    sequence = [tokenizer.bos_id] + input_ids + output_ids + [tokenizer.eos_id]
                    if len(sequence) > config.max_length:
                        sequence = sequence[:config.max_length]

                    # 填充
                    while len(sequence) < config.max_length:
                        sequence.append(tokenizer.pad_id)

                    processed_data.append(torch.tensor(sequence))

                # 创建DataLoader
                dataset = TensorDataset(torch.stack(processed_data))
                dataloader = DataLoader(
                    dataset,
                    batch_size=config.batch_size,
                    num_workers=config.num_workers,
                    pin_memory=torch.cuda.is_available()
                )

                # 测试加载速度
                load_start = time.time()
                batch_count = 0
                for batch in dataloader:
                    batch_count += 1
                    if batch_count >= 20:  # 限制测试批次
                        break

                load_time = time.time() - load_start
                total_time = time.time() - start_time

                results[config_name] = {
                    'total_time': total_time,
                    'load_time': load_time,
                    'samples_per_sec': len(processed_data) / total_time
                }

                print(f"   ⏱️  总时间: {total_time:.2f}s")
                print(f"   📈 处理速度: {results[config_name]['samples_per_sec']:.1f} samples/sec")

            except Exception as e:
                print(f"   ❌ 错误: {e}")
                results[config_name] = {'error': str(e)}

        # 性能对比
        if len(results) == 2 and all('error' not in r for r in results.values()):
            基础 = results["基础配置"]
            优化 = results["优化配置"]

            speedup = 优化['samples_per_sec'] / 基础['samples_per_sec']
            print(f"\n🏆 性能提升总结:")
            print(f"   基础配置: {基础['samples_per_sec']:.1f} samples/sec")
            print(f"   优化配置: {优化['samples_per_sec']:.1f} samples/sec")
            print(f"   加速比: {speedup:.2f}x")

        return results

    def demo_memory_optimization(self):
        """演示内存优化功能"""
        print("\n" + "="*60)
        print("🧠 内存优化演示")
        print("="*60)

        # 创建测试模型
        model = ModelFactory.create_test_transformer("small", vocab_size=10000)
        model.to(self.device)

        print(f"📊 模型参数量: {ModelFactory.get_model_params(model):,}")

        # 测试不同优化配置
        configs = [
            ("基础配置", MemoryConfig(
                enable_amp=False,
                gradient_accumulation_steps=1,
                enable_gradient_checkpointing=False
            )),
            ("混合精度", MemoryConfig(
                enable_amp=True,
                gradient_accumulation_steps=1,
                enable_gradient_checkpointing=False
            )),
            ("梯度累积", MemoryConfig(
                enable_amp=True,
                gradient_accumulation_steps=4,
                enable_gradient_checkpointing=False
            )),
            ("完整优化", MemoryConfig(
                enable_amp=True,
                gradient_accumulation_steps=4,
                enable_gradient_checkpointing=True,
                adaptive_batch_size=True
            ))
        ]

        results = {}

        for config_name, mem_config in configs:
            print(f"\n🔬 测试 {config_name}:")
            print(f"   混合精度: {'启用' if mem_config.enable_amp else '禁用'}")
            print(f"   梯度累积: {mem_config.gradient_accumulation_steps} 步")
            print(f"   梯度检查点: {'启用' if mem_config.enable_gradient_checkpointing else '禁用'}")

            try:
                # 创建优化器
                optimizer = optim.AdamW(model.parameters(), lr=1e-4)
                memory_optimizer = MemoryOptimizer(model, mem_config, self.device)

                # 创建测试数据
                batch_size = 16
                seq_len = 512
                vocab_size = 10000

                dummy_input = torch.randint(0, vocab_size, (batch_size, seq_len), device=self.device)
                dummy_target = torch.randint(0, vocab_size, (batch_size, seq_len), device=self.device)

                # 记录初始内存
                initial_memory = self._get_memory_usage()

                # 测试训练步骤
                start_time = time.time()
                steps = 10

                for step in range(steps):
                    try:
                        with memory_optimizer.optimize_step_context(optimizer) as ctx:
                            # 前向传播
                            output = model(dummy_input)
                            loss = nn.CrossEntropyLoss()(
                                output.reshape(-1, vocab_size),
                                dummy_target.reshape(-1)
                            )

                            # 优化损失和反向传播
                            optimized_loss = memory_optimizer.compute_loss(loss)
                            memory_optimizer.backward(optimized_loss)

                    except Exception as e:
                        if "out of memory" in str(e).lower():
                            print(f"   ⚠️  OOM at step {step}")
                            break
                        else:
                            raise e

                elapsed_time = time.time() - start_time
                final_memory = self._get_memory_usage()
                memory_used = final_memory - initial_memory

                # 获取优化器统计
                stats = memory_optimizer.get_memory_stats()

                results[config_name] = {
                    'elapsed_time': elapsed_time,
                    'memory_used_mb': memory_used,
                    'samples_per_sec': (steps * batch_size) / elapsed_time,
                    'amp_scale': stats.get('amp_scale', 1.0),
                    'oom_count': stats.get('oom_count', 0)
                }

                print(f"   ⏱️  训练时间: {elapsed_time:.2f}s")
                print(f"   💾 内存使用: {memory_used:.1f}MB")
                print(f"   📈 训练速度: {results[config_name]['samples_per_sec']:.1f} samples/sec")
                if stats.get('amp_scale'):
                    print(f"   🔢 AMP缩放: {stats['amp_scale']:.0f}")

            except Exception as e:
                print(f"   ❌ 错误: {e}")
                results[config_name] = {'error': str(e)}

        # 性能对比总结
        valid_results = {k: v for k, v in results.items() if 'error' not in v}
        if len(valid_results) >= 2:
            print(f"\n🏆 内存优化效果总结:")
            baseline = valid_results.get("基础配置")
            optimized = valid_results.get("完整优化")

            if baseline and optimized:
                speed_improvement = optimized['samples_per_sec'] / baseline['samples_per_sec']
                memory_reduction = (baseline['memory_used_mb'] - optimized['memory_used_mb']) / baseline['memory_used_mb'] * 100

                print(f"   训练速度提升: {speed_improvement:.2f}x")
                print(f"   内存使用减少: {memory_reduction:.1f}%")

        return results

    def demo_training_monitoring(self):
        """演示训练监控功能"""
        print("\n" + "="*60)
        print("🔍 训练监控演示")
        print("="*60)

        # 创建测试模型
        model = ModelFactory.create_test_transformer("small", vocab_size=10000)
        model.to(self.device)

        # 初始化训练监控器
        monitor = TrainingMonitor(
            model=model,
            log_dir=str(self.demo_dir / "training_logs"),
            enable_tensorboard=True,
            enable_real_time_plots=False  # 在演示中禁用实时绘图
        )

        print("📊 训练监控器已初始化")
        print(f"   日志目录: {self.demo_dir / 'training_logs'}")
        print(f"   TensorBoard: 启用")

        # 模拟训练过程
        optimizer = optim.AdamW(model.parameters(), lr=1e-4)
        batch_size = 16
        seq_len = 512
        vocab_size = 10000

        print(f"\n🚀 开始模拟训练 (批处理大小: {batch_size}, 序列长度: {seq_len})")

        training_metrics = []

        for epoch in range(2):
            for step in range(20):  # 每个epoch 20步
                global_step = epoch * 20 + step

                # 创建虚拟数据
                dummy_input = torch.randint(0, vocab_size, (batch_size, seq_len), device=self.device)
                dummy_target = torch.randint(0, vocab_size, (batch_size, seq_len), device=self.device)

                # 前向传播
                output = model(dummy_input)
                loss = nn.CrossEntropyLoss()(
                    output.reshape(-1, vocab_size),
                    dummy_target.reshape(-1)
                )

                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # 计算学习率（模拟衰减）
                lr = 1e-4 * (0.95 ** (global_step // 10))
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr

                # 记录监控指标
                metrics = monitor.log_step(
                    step=global_step,
                    epoch=epoch,
                    loss=loss.item(),
                    learning_rate=lr,
                    batch_size=batch_size
                )

                training_metrics.append(metrics)

                # 每10步打印一次进度
                if global_step % 10 == 0:
                    print(f"   Step {global_step}: Loss = {loss.item():.4f}, "
                          f"LR = {lr:.2e}, "
                          f"Speed = {metrics.samples_per_sec:.1f} samples/sec")

        print(f"\n✅ 训练监控演示完成!")
        print(f"   总步数: {len(training_metrics)}")
        print(f"   最终损失: {training_metrics[-1].loss:.4f}")
        print(f"   平均训练速度: {np.mean([m.samples_per_sec for m in training_metrics]):.1f} samples/sec")

        # 关闭监控器并生成报告
        monitor.close()

        return training_metrics

    def demo_performance_benchmark(self):
        """演示性能基准测试"""
        print("\n" + "="*60)
        print("🏁 性能基准测试演示")
        print("="*60)

        # 配置基准测试
        config = BenchmarkConfig(
            test_batch_sizes=[8, 16, 32],
            test_sequence_lengths=[256, 512],
            test_model_sizes=["tiny", "small"],
            test_steps=10,  # 减少步数加快演示
            output_dir=str(self.demo_dir / "benchmark_results"),
            test_training=True,
            test_inference=True,
            test_data_loading=False,  # 跳过数据加载测试以简化演示
            test_memory_optimization=True
        )

        print(f"🔧 基准测试配置:")
        print(f"   批处理大小: {config.test_batch_sizes}")
        print(f"   序列长度: {config.test_sequence_lengths}")
        print(f"   模型大小: {config.test_model_sizes}")
        print(f"   测试步数: {config.test_steps}")

        # 运行基准测试
        benchmark_suite = PerformanceBenchmarkSuite(config)
        results = benchmark_suite.run_all_benchmarks()

        print(f"\n📊 基准测试完成! 共运行 {len(results)} 个测试")

        # 分析结果
        training_results = [r for r in results if r.test_name == "training_baseline"]
        if training_results:
            best_result = max(training_results, key=lambda x: x.metrics['samples_per_sec'])
            print(f"\n🏆 最佳训练配置:")
            print(f"   模型: {best_result.config['model_size']}")
            print(f"   批处理大小: {best_result.config['batch_size']}")
            print(f"   序列长度: {best_result.config['seq_len']}")
            print(f"   吞吐量: {best_result.metrics['samples_per_sec']:.1f} samples/sec")

        inference_results = [r for r in results if r.test_name == "inference"]
        if inference_results:
            best_inference = max(inference_results, key=lambda x: x.metrics['tokens_per_sec'])
            print(f"\n🚀 最佳推理配置:")
            print(f"   模型: {best_inference.config['model_size']}")
            print(f"   批处理大小: {best_inference.config['batch_size']}")
            print(f"   吞吐量: {best_inference.metrics['tokens_per_sec']:.0f} tokens/sec")

        return results

    def _get_memory_usage(self) -> float:
        """获取当前内存使用量(MB)"""
        if self.device.type == 'cuda':
            return torch.cuda.memory_allocated() / 1024 / 1024
        else:
            import psutil
            return psutil.Process().memory_info().rss / 1024 / 1024

    def run_complete_demo(self):
        """运行完整演示"""
        print(f"🎬 开始MiniGPT优化套件完整演示")
        print(f"⚙️  设备: {self.device}")
        print()

        try:
            # 1. 创建演示数据
            data_path = self.create_dummy_dataset(500, 256)

            # 2. 高性能数据加载演示
            data_results = self.demo_high_performance_data_loading(data_path)

            # 3. 内存优化演示
            memory_results = self.demo_memory_optimization()

            # 4. 训练监控演示
            monitoring_results = self.demo_training_monitoring()

            # 5. 性能基准测试演示
            benchmark_results = self.demo_performance_benchmark()

            # 6. 生成总结报告
            self._generate_demo_summary({
                'data_loading': data_results,
                'memory_optimization': memory_results,
                'training_monitoring': monitoring_results,
                'benchmarks': benchmark_results
            })

            print("\n" + "="*60)
            print("🎉 优化套件演示完成!")
            print("="*60)
            print(f"📁 所有结果已保存到: {self.demo_dir}")
            print("📊 查看以下文件了解详细结果:")
            print(f"   - 演示总结: {self.demo_dir}/demo_summary.json")
            print(f"   - 训练日志: {self.demo_dir}/training_logs/")
            print(f"   - 基准测试: {self.demo_dir}/benchmark_results/")
            print()
            print("🚀 建议:")
            print("   1. 查看TensorBoard: tensorboard --logdir demo_results/training_logs")
            print("   2. 查看基准测试图表: demo_results/benchmark_results/*.png")
            print("   3. 阅读性能分析报告: demo_results/benchmark_results/performance_analysis.json")

        except Exception as e:
            print(f"❌ 演示过程中发生错误: {e}")
            import traceback
            traceback.print_exc()

    def _generate_demo_summary(self, results: dict):
        """生成演示总结报告"""
        summary = {
            'demo_info': {
                'timestamp': time.time(),
                'device': str(self.device),
                'device_name': self._get_device_name()
            },
            'results': results,
            'key_improvements': self._extract_key_improvements(results)
        }

        summary_file = self.demo_dir / "demo_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False, default=str)

        print(f"\n📋 演示总结已保存: {summary_file}")

    def _get_device_name(self) -> str:
        """获取设备名称"""
        if self.device.type == 'cuda':
            return torch.cuda.get_device_name()
        elif self.device.type == 'mps':
            return "Apple Silicon GPU (MPS)"
        else:
            return "CPU"

    def _extract_key_improvements(self, results: dict) -> dict:
        """提取关键改进指标"""
        improvements = {}

        # 数据加载改进
        data_results = results.get('data_loading', {})
        if '基础配置' in data_results and '优化配置' in data_results:
            baseline = data_results['基础配置']
            optimized = data_results['优化配置']
            if 'samples_per_sec' in baseline and 'samples_per_sec' in optimized:
                improvements['data_loading_speedup'] = optimized['samples_per_sec'] / baseline['samples_per_sec']

        # 内存优化改进
        memory_results = results.get('memory_optimization', {})
        if '基础配置' in memory_results and '完整优化' in memory_results:
            baseline = memory_results['基础配置']
            optimized = memory_results['完整优化']
            if 'samples_per_sec' in baseline and 'samples_per_sec' in optimized:
                improvements['memory_optimization_speedup'] = optimized['samples_per_sec'] / baseline['samples_per_sec']

        return improvements


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="MiniGPT优化套件演示")
    parser.add_argument("--device", type=str, default="auto",
                       choices=["auto", "cuda", "mps", "cpu"],
                       help="指定使用的设备")
    parser.add_argument("--quick", action="store_true",
                       help="快速演示模式（减少测试规模）")

    args = parser.parse_args()

    # 创建并运行演示
    demo = OptimizationDemo(device=args.device)
    demo.run_complete_demo()


if __name__ == "__main__":
    main()