#!/usr/bin/env python3
"""
测试轻量级监控的性能影响
对比完整监控 vs 轻量级监控 vs 无监控的训练速度
"""
import os
import sys
import time
import torch
import torch.nn as nn

# 添加项目路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(project_root, 'src'))

from training.training_monitor import TrainingMonitor


def create_test_model(size="small"):
    """创建测试模型"""
    if size == "small":
        return nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512)
        )
    else:  # medium - 类似真实训练
        return nn.Sequential(
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Linear(2048, 4096),
            nn.ReLU(),
            nn.Linear(4096, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024)
        )


def benchmark_training(model, num_steps=500, monitor_mode="none"):
    """
    基准测试训练速度

    Args:
        model: PyTorch模型
        num_steps: 训练步数
        monitor_mode: 监控模式 ("none", "lightweight", "full")
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()

    # 初始化监控器
    monitor = None
    if monitor_mode == "lightweight":
        monitor = TrainingMonitor(
            model=model,
            log_dir=f"benchmark_logs/lightweight_{int(time.time())}",
            enable_tensorboard=True,
            enable_real_time_plots=False,
            lightweight_mode=True,
            log_interval=10
        )
    elif monitor_mode == "full":
        monitor = TrainingMonitor(
            model=model,
            log_dir=f"benchmark_logs/full_{int(time.time())}",
            enable_tensorboard=True,
            enable_real_time_plots=False,
            lightweight_mode=False,
            log_interval=1
        )

    # 训练循环
    model.train()
    start_time = time.time()
    step_times = []

    for step in range(num_steps):
        step_start = time.time()

        # 模拟训练
        batch_size = 32
        x = torch.randn(batch_size, 1024 if "2048" in str(model) else 512, device=device)
        y = torch.randn(batch_size, 1024 if "2048" in str(model) else 512, device=device)

        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()

        # 使用监控器
        if monitor:
            monitor.log_step(
                step=step,
                epoch=0,
                loss=loss.item(),
                learning_rate=1e-4,
                batch_size=batch_size
            )

        step_time = time.time() - step_start
        step_times.append(step_time)

        if step % 100 == 0:
            avg_step_time = sum(step_times[-100:]) / len(step_times[-100:])
            print(f"[{monitor_mode.upper()}] Step {step:4d} | "
                  f"Loss: {loss.item():.4f} | "
                  f"Step time: {avg_step_time*1000:.2f}ms")

    total_time = time.time() - start_time
    avg_step_time = sum(step_times) / len(step_times)

    # 关闭监控器
    if monitor:
        monitor.close()

    return {
        'mode': monitor_mode,
        'total_time': total_time,
        'avg_step_time': avg_step_time,
        'steps_per_sec': 1.0 / avg_step_time,
        'overhead_ms': avg_step_time * 1000
    }


def main():
    print("=" * 70)
    print("🧪 轻量级监控性能测试")
    print("=" * 70)

    # 测试配置
    num_steps = 500
    model_size = "medium"  # 使用中等大小模型，更接近实际情况

    print(f"\n配置:")
    print(f"  模型大小: {model_size}")
    print(f"  训练步数: {num_steps}")
    print(f"  设备: {'CUDA' if torch.cuda.is_available() else 'CPU'}")

    # 测试三种模式
    results = []

    print(f"\n{'='*70}")
    print("1️⃣  测试：无监控")
    print("="*70)
    model1 = create_test_model(model_size)
    result1 = benchmark_training(model1, num_steps, "none")
    results.append(result1)
    del model1
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    time.sleep(2)

    print(f"\n{'='*70}")
    print("2️⃣  测试：轻量级监控 (每10步完整记录)")
    print("="*70)
    model2 = create_test_model(model_size)
    result2 = benchmark_training(model2, num_steps, "lightweight")
    results.append(result2)
    del model2
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    time.sleep(2)

    print(f"\n{'='*70}")
    print("3️⃣  测试：完整监控 (每步完整记录)")
    print("="*70)
    model3 = create_test_model(model_size)
    result3 = benchmark_training(model3, num_steps, "full")
    results.append(result3)
    del model3
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # 打印对比结果
    print(f"\n{'='*70}")
    print("📊 性能对比结果")
    print("="*70)

    baseline = results[0]['avg_step_time']

    print(f"\n{'模式':<20} {'总时间(s)':<15} {'平均步时(ms)':<18} {'步数/秒':<15} {'性能损失':<15}")
    print("-" * 85)

    for result in results:
        overhead_pct = ((result['avg_step_time'] - baseline) / baseline * 100) if baseline > 0 else 0
        print(f"{result['mode']:<20} "
              f"{result['total_time']:<15.2f} "
              f"{result['overhead_ms']:<18.2f} "
              f"{result['steps_per_sec']:<15.2f} "
              f"{overhead_pct:>6.2f}%")

    # 总结建议
    print(f"\n{'='*70}")
    print("💡 性能分析与建议")
    print("="*70)

    lightweight_overhead = ((results[1]['avg_step_time'] - baseline) / baseline * 100)
    full_overhead = ((results[2]['avg_step_time'] - baseline) / baseline * 100)

    print(f"\n1. 轻量级监控性能影响: {lightweight_overhead:.2f}%")
    print(f"2. 完整监控性能影响: {full_overhead:.2f}%")
    print(f"3. 轻量级相比完整监控节省: {full_overhead - lightweight_overhead:.2f}%")

    print("\n推荐使用场景:")
    if lightweight_overhead < 2:
        print("  ✅ 轻量级监控开销极小 (<2%)，推荐在所有训练中使用")
    elif lightweight_overhead < 5:
        print("  ✅ 轻量级监控开销可接受 (<5%)，推荐在长时间训练中使用")
    else:
        print("  ⚠️  轻量级监控开销较大 (>5%)，建议仅在调试时使用")

    if full_overhead > 10:
        print("  ⚠️  完整监控开销显著 (>10%)，仅建议在短期调试时使用")

    print(f"\n{'='*70}")
    print("✅ 测试完成！")
    print("="*70)


if __name__ == "__main__":
    main()

