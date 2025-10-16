#!/usr/bin/env python3
"""
MiniGPT简化优化演示
专注于核心优化功能，避免复杂依赖
"""
import json
import time

import torch
import torch.nn as nn
import torch.optim as optim


class SimpleTransformer(nn.Module):
    """简化的Transformer模型用于演示"""

    def __init__(self, vocab_size=10000, d_model=256, n_heads=4, n_layers=6, d_ff=1024):
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


class SimpleMemoryOptimizer:
    """简化的内存优化器"""

    def __init__(self, model, enable_amp=True, gradient_accumulation_steps=1):
        self.model = model
        self.enable_amp = enable_amp and torch.cuda.is_available()
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.step_count = 0

        if self.enable_amp:
            from torch.cuda.amp import GradScaler
            self.scaler = GradScaler()
            print("✅ 混合精度训练已启用")
        else:
            self.scaler = None
            print("❌ 混合精度训练未启用")

    def should_update(self):
        """是否应该更新参数"""
        return (self.step_count + 1) % self.gradient_accumulation_steps == 0

    def scale_loss(self, loss):
        """缩放损失"""
        return loss / self.gradient_accumulation_steps

    def backward(self, loss):
        """反向传播"""
        scaled_loss = self.scale_loss(loss)
        if self.enable_amp and self.scaler:
            self.scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()

    def step_optimizer(self, optimizer):
        """优化器步骤"""
        if self.should_update():
            if self.enable_amp and self.scaler:
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                optimizer.step()

            optimizer.zero_grad()
            return True

        self.step_count += 1
        return False


def get_memory_usage(device):
    """获取内存使用量(MB)"""
    if device.type == 'cuda':
        return torch.cuda.memory_allocated() / 1024 / 1024
    else:
        import psutil
        return psutil.Process().memory_info().rss / 1024 / 1024


def benchmark_training(model, device, config_name, enable_amp=False, gradient_accumulation_steps=1):
    """基准测试训练性能"""
    print(f"\n🔬 测试配置: {config_name}")
    print(f"   混合精度: {'启用' if enable_amp else '禁用'}")
    print(f"   梯度累积: {gradient_accumulation_steps} 步")

    # 重置模型
    model.train()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)

    # 创建内存优化器
    memory_optimizer = SimpleMemoryOptimizer(
        model, enable_amp, gradient_accumulation_steps
    )

    # 测试参数
    batch_size = 16
    seq_len = 512
    vocab_size = 10000
    test_steps = 20

    # 创建测试数据
    dummy_input = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    dummy_target = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

    # 记录初始内存
    initial_memory = get_memory_usage(device)

    # 清理缓存
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    try:
        # 预热
        for _ in range(3):
            optimizer.zero_grad()
            if enable_amp and device.type == 'cuda':
                from torch.cuda.amp import autocast
                with autocast():
                    output = model(dummy_input)
                    loss = nn.CrossEntropyLoss()(
                        output.reshape(-1, vocab_size),
                        dummy_target.reshape(-1)
                    )
            else:
                output = model(dummy_input)
                loss = nn.CrossEntropyLoss()(
                    output.reshape(-1, vocab_size),
                    dummy_target.reshape(-1)
                )

            memory_optimizer.backward(loss)
            memory_optimizer.step_optimizer(optimizer)

        # 实际测试
        start_time = time.time()

        for _ in range(test_steps):
            if enable_amp and device.type == 'cuda':
                from torch.cuda.amp import autocast
                with autocast():
                    output = model(dummy_input)
                    loss = nn.CrossEntropyLoss()(
                        output.reshape(-1, vocab_size),
                        dummy_target.reshape(-1)
                    )
            else:
                output = model(dummy_input)
                loss = nn.CrossEntropyLoss()(
                    output.reshape(-1, vocab_size),
                    dummy_target.reshape(-1)
                )

            memory_optimizer.backward(loss)
            memory_optimizer.step_optimizer(optimizer)

        # 同步GPU
        if device.type == 'cuda':
            torch.cuda.synchronize()

        end_time = time.time()
        final_memory = get_memory_usage(device)

        # 计算指标
        elapsed_time = end_time - start_time
        total_samples = test_steps * batch_size
        samples_per_sec = total_samples / elapsed_time
        memory_used = final_memory - initial_memory

        results = {
            'samples_per_sec': samples_per_sec,
            'elapsed_time': elapsed_time,
            'memory_used_mb': memory_used,
            'final_loss': loss.item()
        }

        print(f"   ⏱️  训练时间: {elapsed_time:.2f}s")
        print(f"   📈 训练速度: {samples_per_sec:.1f} samples/sec")
        print(f"   💾 内存使用: {memory_used:.1f}MB")
        print(f"   📉 最终损失: {loss.item():.4f}")

        return results

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"   ❌ 内存不足: {e}")
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            return {'error': 'OOM'}
        else:
            raise e


def benchmark_inference(model, device):
    """基准测试推理性能"""
    print("\n🚀 推理性能测试")

    model.eval()
    vocab_size = 10000
    seq_len = 512
    test_batches = [1, 4, 8, 16, 32]

    results = {}

    with torch.no_grad():
        for batch_size in test_batches:
            try:
                # 创建测试数据
                dummy_input = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

                # 预热
                for _ in range(5):
                    _ = model(dummy_input)

                # 清理内存
                if device.type == 'cuda':
                    torch.cuda.empty_cache()

                # 基准测试
                steps = 50
                start_time = time.time()

                for _ in range(steps):
                    _ = model(dummy_input)

                if device.type == 'cuda':
                    torch.cuda.synchronize()

                end_time = time.time()
                elapsed_time = end_time - start_time

                samples_per_sec = (steps * batch_size) / elapsed_time
                tokens_per_sec = samples_per_sec * seq_len
                latency_ms = (elapsed_time / steps) * 1000

                results[batch_size] = {
                    'samples_per_sec': samples_per_sec,
                    'tokens_per_sec': tokens_per_sec,
                    'latency_ms': latency_ms
                }

                print(f"   批处理={batch_size}: {tokens_per_sec:.0f} tokens/sec, 延迟={latency_ms:.1f}ms")

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"   批处理={batch_size}: OOM")
                    results[batch_size] = {'error': 'OOM'}
                    if device.type == 'cuda':
                        torch.cuda.empty_cache()
                else:
                    raise e

    return results


def main():
    """主演示函数"""
    print("🚀 MiniGPT简化优化演示")
    print("=" * 60)

    # 设备选择
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"🎯 使用CUDA GPU: {torch.cuda.get_device_name()}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("🎯 使用Apple Silicon GPU (MPS)")
    else:
        device = torch.device("cpu")
        print("🎯 使用CPU")

    # 创建模型
    print("\n📊 创建测试模型...")
    model = SimpleTransformer(vocab_size=10000, d_model=256, n_heads=4, n_layers=6)
    model.to(device)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"   模型参数量: {param_count:,}")
    print(f"   模型大小: ~{param_count * 4 / 1024 / 1024:.1f}MB")

    # 训练性能测试
    print("\n" + "="*60)
    print("🧠 训练性能对比测试")
    print("="*60)

    training_results = {}

    # 测试配置
    configs = [
        ("基础配置", False, 1),
        ("混合精度", True, 1),
        ("梯度累积", True, 4),
    ]

    for config_name, enable_amp, grad_accum in configs:
        result = benchmark_training(
            model, device, config_name, enable_amp, grad_accum
        )
        training_results[config_name] = result

    # 推理性能测试
    print("\n" + "="*60)
    print("🚀 推理性能测试")
    print("="*60)

    inference_results = benchmark_inference(model, device)

    # 性能总结
    print("\n" + "="*60)
    print("📊 性能总结")
    print("="*60)

    # 训练性能对比
    valid_training = {k: v for k, v in training_results.items() if 'error' not in v}
    if len(valid_training) >= 2:
        print("\n🏆 训练性能对比:")
        baseline = None
        for name, result in valid_training.items():
            speed = result['samples_per_sec']
            memory = result['memory_used_mb']
            print(f"   {name}: {speed:.1f} samples/sec, {memory:.1f}MB")

            if baseline is None:
                baseline = speed
            else:
                speedup = speed / baseline
                print(f"      相对提升: {speedup:.2f}x")

    # 推理性能总结
    valid_inference = {k: v for k, v in inference_results.items() if 'error' not in v}
    if valid_inference:
        print("\n🚀 推理性能总结:")
        best_batch = max(valid_inference.keys(), key=lambda k: valid_inference[k]['tokens_per_sec'])
        best_result = valid_inference[best_batch]
        print(f"   最佳配置: 批处理大小={best_batch}")
        print(f"   最大吞吐量: {best_result['tokens_per_sec']:.0f} tokens/sec")
        print(f"   最低延迟: {min(r['latency_ms'] for r in valid_inference.values()):.1f}ms")

    # 保存结果
    results = {
        'device': str(device),
        'model_params': param_count,
        'training_results': training_results,
        'inference_results': inference_results,
        'timestamp': time.time()
    }

    results_file = "simple_demo_results.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)

    print(f"\n📁 结果已保存到: {results_file}")
    print("\n✅ 演示完成!")


if __name__ == "__main__":
    main()
