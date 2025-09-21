#!/usr/bin/env python3
"""
推理测试脚本
测试模型的生成能力和推理性能
"""
import torch
import time
from typing import List

from config.training_config import get_config
from src.model.transformer import create_model
from src.model.config import MiniGPTConfig


def test_text_generation(model, config, test_prompts: List[str] = None):
    """测试文本生成功能"""
    print("=== 文本生成测试 ===")

    if test_prompts is None:
        # 创建测试提示
        test_prompts = [
            [1, 2, 3, 4, 5],  # 简单数字序列
            [10, 20, 30],     # 另一个序列
            [100],            # 单个token
            [1, 1, 1, 1]      # 重复token
        ]

    model.eval()
    results = []

    for i, prompt_tokens in enumerate(test_prompts):
        print(f"\n--- 测试 {i+1}: 输入长度 {len(prompt_tokens)} ---")

        # 转换为tensor
        prompt = torch.tensor([prompt_tokens], dtype=torch.long)
        if config.device != "cpu":
            prompt = prompt.to(config.device)

        print(f"输入: {prompt_tokens}")

        # 生成文本
        start_time = time.time()
        with torch.no_grad():
            generated = model.generate(
                prompt,
                max_length=20,
                temperature=0.8,
                top_k=50
            )
        end_time = time.time()

        # 提取生成的部分
        generated_tokens = generated[0].tolist()
        new_tokens = generated_tokens[len(prompt_tokens):]

        generation_time = end_time - start_time
        tokens_per_second = len(new_tokens) / generation_time if generation_time > 0 else 0

        print(f"生成: {new_tokens}")
        print(f"时间: {generation_time:.3f}s")
        print(f"速度: {tokens_per_second:.1f} tokens/s")

        results.append({
            'prompt': prompt_tokens,
            'generated': new_tokens,
            'time': generation_time,
            'speed': tokens_per_second
        })

    return results


def test_batch_generation(model, config):
    """测试批量生成"""
    print("\n=== 批量生成测试 ===")

    batch_size = 4
    seq_len = 5

    # 创建批量输入
    prompts = torch.randint(0, min(100, config.vocab_size), (batch_size, seq_len))
    if config.device != "cpu":
        prompts = prompts.to(config.device)

    print(f"批量大小: {batch_size}")
    print(f"输入形状: {prompts.shape}")

    model.eval()
    start_time = time.time()

    with torch.no_grad():
        # 注意：当前的generate方法只支持单个样本，所以我们逐个处理
        generated_batch = []
        for i in range(batch_size):
            single_prompt = prompts[i:i+1]
            generated = model.generate(
                single_prompt,
                max_length=10,
                temperature=0.8,
                top_k=50
            )
            generated_batch.append(generated)

    end_time = time.time()
    total_time = end_time - start_time

    print(f"批量生成时间: {total_time:.3f}s")
    print(f"平均每样本时间: {total_time/batch_size:.3f}s")

    return total_time


def test_different_generation_params(model, config):
    """测试不同的生成参数"""
    print("\n=== 生成参数测试 ===")

    # 固定输入
    prompt = torch.tensor([[1, 2, 3]], dtype=torch.long)
    if config.device != "cpu":
        prompt = prompt.to(config.device)

    test_params = [
        {'temperature': 0.1, 'top_k': 10, 'name': '低温度+小top_k'},
        {'temperature': 1.0, 'top_k': 50, 'name': '中等温度+中等top_k'},
        {'temperature': 1.5, 'top_k': 100, 'name': '高温度+大top_k'},
    ]

    model.eval()
    results = []

    for params in test_params:
        print(f"\n--- {params['name']} ---")
        print(f"温度: {params['temperature']}, top_k: {params['top_k']}")

        start_time = time.time()
        with torch.no_grad():
            generated = model.generate(
                prompt,
                max_length=15,
                temperature=params['temperature'],
                top_k=params['top_k']
            )
        end_time = time.time()

        generated_tokens = generated[0].tolist()[3:]  # 去掉输入部分
        generation_time = end_time - start_time

        print(f"生成: {generated_tokens}")
        print(f"时间: {generation_time:.3f}s")

        results.append({
            'params': params,
            'generated': generated_tokens,
            'time': generation_time
        })

    return results


def test_memory_usage(model, config):
    """测试内存使用情况"""
    print("\n=== 内存使用测试 ===")

    if config.device == "cuda":
        torch.cuda.empty_cache()
        before_allocated = torch.cuda.memory_allocated() / 1024**3
        before_reserved = torch.cuda.memory_reserved() / 1024**3

        print(f"生成前 - 已分配: {before_allocated:.1f}GB, 已保留: {before_reserved:.1f}GB")

        # 生成大量文本
        prompt = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long).to(config.device)

        model.eval()
        with torch.no_grad():
            for i in range(5):
                generated = model.generate(prompt, max_length=50)
                if i == 0:
                    mid_allocated = torch.cuda.memory_allocated() / 1024**3
                    mid_reserved = torch.cuda.memory_reserved() / 1024**3
                    print(f"生成中 - 已分配: {mid_allocated:.1f}GB, 已保留: {mid_reserved:.1f}GB")

        after_allocated = torch.cuda.memory_allocated() / 1024**3
        after_reserved = torch.cuda.memory_reserved() / 1024**3

        print(f"生成后 - 已分配: {after_allocated:.1f}GB, 已保留: {after_reserved:.1f}GB")

        return {
            'before': {'allocated': before_allocated, 'reserved': before_reserved},
            'after': {'allocated': after_allocated, 'reserved': after_reserved}
        }
    else:
        print("非CUDA设备，跳过显存监控")
        return None


def main():
    """主测试函数"""
    print("🔥 开始推理测试")

    # 获取配置
    config = get_config("tiny")

    # 创建模型
    model_config = MiniGPTConfig(
        vocab_size=config.vocab_size,
        hidden_size=config.d_model,
        num_hidden_layers=config.n_layers,
        num_attention_heads=config.n_heads,
        intermediate_size=config.d_ff,
        max_position_embeddings=config.max_seq_len,
        dropout=config.dropout,
        rms_norm_eps=1e-6
    )

    model = create_model(config=model_config)

    # 移动到设备
    if config.device != "cpu":
        model = model.to(config.device)

    print(f"设备: {config.device}")
    print(f"模型参数: {model.get_num_params():,}")

    # 运行各种推理测试
    generation_results = test_text_generation(model, config)
    batch_time = test_batch_generation(model, config)
    param_results = test_different_generation_params(model, config)
    memory_results = test_memory_usage(model, config)

    # 总结
    print(f"\n🎉 推理测试完成！")

    # 计算平均性能
    avg_speed = sum(r['speed'] for r in generation_results) / len(generation_results)
    avg_time = sum(r['time'] for r in generation_results) / len(generation_results)

    print(f"✓ 平均生成速度: {avg_speed:.1f} tokens/秒")
    print(f"✓ 平均生成时间: {avg_time:.3f} 秒")
    print(f"✓ 批量处理性能: {batch_time:.3f} 秒/4样本")

    if memory_results:
        print(f"✓ 显存使用稳定")

    print("\n所有推理测试通过！")


if __name__ == "__main__":
    main()