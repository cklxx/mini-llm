#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
性能测试脚本
对比不同优化配置下的训练性能
"""
import os
import sys
import time
import argparse
import subprocess
import psutil
from datetime import datetime

# 添加项目路径
project_root = os.path.dirname(os.path.dirname(__file__))
sys.path.append(project_root)

def run_performance_test(config_name, enable_optimizations=True, test_steps=100):
    """运行性能测试"""
    print(f"\n{'='*60}")
    print(f"🧪 性能测试: {config_name}")
    print(f"优化: {'启用' if enable_optimizations else '禁用'}")
    print(f"测试步数: {test_steps}")
    print(f"{'='*60}")
    
    # 构建命令
    cmd = [
        'python', 'scripts/train_optimized.py',
        '--config', 'small',  # 使用小模型快速测试
        '--data-files', 'pretrain_200.jsonl',
        '--max-data-size', '5000',  # 使用少量数据
        '--max-steps', str(test_steps),
        '--batch-size', '4',
        '--save-steps', str(test_steps + 1),  # 不保存中间检查点
        '--output-dir', f'checkpoints/perf_test_{config_name}',
        '--disable-monitoring',  # 禁用监控避免干扰
    ]
    
    if enable_optimizations:
        cmd.extend([
            '--num-threads', '4',
            '--dataloader-workers', '2',
            '--enable-compile'
        ])
    else:
        cmd.extend([
            '--disable-optimizations',
            '--num-threads', '1',
            '--dataloader-workers', '0'
        ])
    
    # 记录开始时间和资源
    start_time = time.time()
    start_cpu_percent = psutil.cpu_percent(interval=1)
    start_memory = psutil.virtual_memory()
    
    print(f"开始时间: {datetime.now().strftime('%H:%M:%S')}")
    print(f"初始CPU: {start_cpu_percent:.1f}%")
    print(f"初始内存: {start_memory.percent:.1f}%")
    
    try:
        # 运行训练
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        
        # 记录结束时间和资源
        end_time = time.time()
        end_cpu_percent = psutil.cpu_percent(interval=1)
        end_memory = psutil.virtual_memory()
        
        training_time = end_time - start_time
        
        print(f"\n📊 测试结果:")
        print(f"训练时间: {training_time:.2f} 秒")
        print(f"平均步速: {test_steps/training_time:.2f} steps/sec")
        print(f"结束CPU: {end_cpu_percent:.1f}%")
        print(f"结束内存: {end_memory.percent:.1f}%")
        
        # 分析输出中的性能信息
        if result.returncode == 0:
            output_lines = result.stdout.split('\n')
            for line in output_lines:
                if 'steps/s' in line:
                    print(f"训练输出: {line.strip()}")
            print("✅ 测试完成")
        else:
            print("❌ 测试失败")
            print(f"错误: {result.stderr}")
        
        return {
            'config': config_name,
            'optimizations': enable_optimizations,
            'training_time': training_time,
            'steps_per_sec': test_steps/training_time,
            'success': result.returncode == 0
        }
        
    except subprocess.TimeoutExpired:
        print("❌ 测试超时")
        return {
            'config': config_name,
            'optimizations': enable_optimizations,
            'training_time': float('inf'),
            'steps_per_sec': 0,
            'success': False
        }
    except Exception as e:
        print(f"❌ 测试异常: {e}")
        return {
            'config': config_name,
            'optimizations': enable_optimizations,
            'training_time': float('inf'),
            'steps_per_sec': 0,
            'success': False
        }
    finally:
        # 清理测试输出
        test_dir = f'checkpoints/perf_test_{config_name}'
        if os.path.exists(test_dir):
            import shutil
            shutil.rmtree(test_dir)

def main():
    parser = argparse.ArgumentParser(description='性能测试脚本')
    parser.add_argument('--test-steps', type=int, default=100,
                        help='测试训练步数')
    parser.add_argument('--skip-baseline', action='store_true',
                        help='跳过基准测试')
    
    args = parser.parse_args()
    
    print("🚀 MiniGPT 性能测试")
    print(f"测试步数: {args.test_steps}")
    print(f"系统信息:")
    print(f"  CPU核心: {psutil.cpu_count()}")
    print(f"  物理核心: {psutil.cpu_count(logical=False)}")
    print(f"  总内存: {psutil.virtual_memory().total / (1024**3):.1f}GB")
    
    results = []
    
    if not args.skip_baseline:
        # 基准测试 (无优化)
        print("\n🔴 运行基准测试 (无优化)")
        baseline_result = run_performance_test(
            'baseline', 
            enable_optimizations=False, 
            test_steps=args.test_steps
        )
        results.append(baseline_result)
    
    # 优化测试
    print("\n🟢 运行优化测试")
    optimized_result = run_performance_test(
        'optimized', 
        enable_optimizations=True, 
        test_steps=args.test_steps
    )
    results.append(optimized_result)
    
    # 总结结果
    print(f"\n{'='*60}")
    print("📊 性能测试总结")
    print(f"{'='*60}")
    
    for result in results:
        if result['success']:
            print(f"{result['config']:12s}: {result['training_time']:6.2f}s | {result['steps_per_sec']:6.2f} steps/s")
        else:
            print(f"{result['config']:12s}: 测试失败")
    
    # 计算性能提升
    if len(results) == 2 and all(r['success'] for r in results):
        baseline = next(r for r in results if r['config'] == 'baseline')
        optimized = next(r for r in results if r['config'] == 'optimized')
        
        speedup = optimized['steps_per_sec'] / baseline['steps_per_sec']
        time_reduction = (baseline['training_time'] - optimized['training_time']) / baseline['training_time'] * 100
        
        print(f"\n🎯 性能提升:")
        print(f"加速倍数: {speedup:.2f}x")
        print(f"时间减少: {time_reduction:.1f}%")
        
        if speedup > 1.2:
            print("🎉 优化效果显著！")
        elif speedup > 1.05:
            print("✅ 优化有效果")
        else:
            print("⚠️  优化效果不明显")
    
    print(f"\n💡 建议:")
    print("- 如果优化效果显著，建议在实际训练中启用所有优化")
    print("- 如果系统资源充足，可以尝试增加线程数和worker数量")
    print("- MPS设备建议启用模型编译以获得最佳性能")

if __name__ == "__main__":
    main()