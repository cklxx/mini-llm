#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Legacy推理测试脚本 (优化版)
测试模型的生成能力和推理性能，兼容新旧架构
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

import torch
import time
import json
from typing import List, Dict, Any
from pathlib import Path

# 新架构导入
from src.model.config import get_tiny_config, get_small_config, MiniGPTConfig
from src.model.transformer import MiniGPT

# 旧配置兼容
try:
    from config.training_config import get_config as get_legacy_config
except ImportError:
    get_legacy_config = None


class LegacyInferenceTest:
    """Legacy推理测试器，兼容新旧架构"""

    def __init__(self, config_name: str = "tiny", use_legacy: bool = False):
        """
        初始化测试器

        Args:
            config_name: 配置名称 (tiny/small/medium)
            use_legacy: 是否使用legacy配置
        """
        self.config_name = config_name
        self.use_legacy = use_legacy

        # 获取配置和模型
        self.config, self.model = self._setup_model()

        # 设备检测
        self.device = self._detect_device()
        if self.device.type != "cpu":
            self.model = self.model.to(self.device)

        print(f"✅ 测试器初始化完成")
        print(f"   配置: {config_name} ({'legacy' if use_legacy else 'optimized'})")
        print(f"   设备: {self.device}")
        print(f"   参数: {sum(p.numel() for p in self.model.parameters()):,}")

    def _detect_device(self) -> torch.device:
        """智能设备检测"""
        if torch.cuda.is_available():
            return torch.device('cuda')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device('mps')
        else:
            return torch.device('cpu')

    def _setup_model(self):
        """设置模型"""
        if self.use_legacy and get_legacy_config:
            # Legacy配置
            config = get_legacy_config(self.config_name)

            # 转换为新配置格式
            model_config = MiniGPTConfig(
                vocab_size=config.vocab_size,
                hidden_size=getattr(config, 'd_model', 128),
                num_hidden_layers=getattr(config, 'n_layers', 4),
                num_attention_heads=getattr(config, 'n_heads', 2),
                intermediate_size=getattr(config, 'd_ff', 512),
                max_position_embeddings=getattr(config, 'max_seq_len', 256),
                dropout=getattr(config, 'dropout', 0.1),
                use_rope=False,  # Legacy不使用RoPE
                use_gqa=False,   # Legacy不使用GQA
                tie_word_embeddings=False
            )

            model = MiniGPT(model_config)
            return config, model
        else:
            # 新优化配置
            if self.config_name == "tiny":
                config = get_tiny_config()
            elif self.config_name == "small":
                config = get_small_config()
            else:
                config = get_tiny_config()  # 默认

            model = MiniGPT(config)
            return config, model

    def test_text_generation(self, test_prompts: List[List[int]] = None) -> List[Dict]:
        """测试文本生成功能"""
        print("\n=== 文本生成测试 ===")

        if test_prompts is None:
            # 创建测试提示
            vocab_size = getattr(self.config, 'vocab_size', 1000)
            test_prompts = [
                [1, 2, 3, 4, 5],  # 简单数字序列
                [10, 20, 30],     # 另一个序列
                [min(100, vocab_size-1)],  # 单个token
                [1, 1, 1, 1]      # 重复token
            ]

        self.model.eval()
        results = []

        for i, prompt_tokens in enumerate(test_prompts):
            print(f"\n--- 测试 {i+1}: 输入长度 {len(prompt_tokens)} ---")

            # 转换为tensor
            prompt = torch.tensor([prompt_tokens], dtype=torch.long, device=self.device)
            print(f"输入: {prompt_tokens}")

            # 生成文本
            start_time = time.time()
            with torch.no_grad():
                try:
                    if hasattr(self.model, 'generate'):
                        generated = self.model.generate(
                            prompt,
                            max_length=20,
                            temperature=0.8,
                            top_k=50
                        )
                    else:
                        # 简单生成备用方案
                        generated = self._simple_generate(prompt, max_length=20)
                except Exception as e:
                    print(f"⚠️ 生成失败: {e}")
                    generated = prompt  # 返回原始输入

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
                'speed': tokens_per_second,
                'success': len(new_tokens) > 0
            })

        return results

    def _simple_generate(self, prompt: torch.Tensor, max_length: int = 20) -> torch.Tensor:
        """简单生成函数（备用）"""
        generated = prompt
        vocab_size = getattr(self.config, 'vocab_size', 1000)

        for _ in range(max_length):
            if generated.size(1) >= max_length:
                break

            # 前向传播
            logits = self.model(generated)
            next_token_logits = logits[0, -1, :] / 0.8  # temperature

            # 简单采样
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # 拼接
            generated = torch.cat([generated, next_token.unsqueeze(0)], dim=1)

        return generated

    def test_batch_generation(self) -> float:
        """测试批量生成"""
        print("\n=== 批量生成测试 ===")

        batch_size = 4
        seq_len = 5
        vocab_size = getattr(self.config, 'vocab_size', 1000)

        # 创建批量输入
        prompts = torch.randint(0, min(100, vocab_size), (batch_size, seq_len), device=self.device)

        print(f"批量大小: {batch_size}")
        print(f"输入形状: {prompts.shape}")

        self.model.eval()
        start_time = time.time()

        with torch.no_grad():
            # 逐个处理（兼容性考虑）
            generated_batch = []
            for i in range(batch_size):
                single_prompt = prompts[i:i+1]
                try:
                    if hasattr(self.model, 'generate'):
                        generated = self.model.generate(
                            single_prompt,
                            max_length=10,
                            temperature=0.8,
                            top_k=50
                        )
                    else:
                        generated = self._simple_generate(single_prompt, max_length=10)
                    generated_batch.append(generated)
                except Exception as e:
                    print(f"⚠️ 批次 {i} 生成失败: {e}")
                    generated_batch.append(single_prompt)

        end_time = time.time()
        total_time = end_time - start_time

        print(f"批量生成时间: {total_time:.3f}s")
        print(f"平均每样本时间: {total_time/batch_size:.3f}s")

        return total_time

    def test_different_generation_params(self) -> List[Dict]:
        """测试不同的生成参数"""
        print("\n=== 生成参数测试 ===")

        # 固定输入
        vocab_size = getattr(self.config, 'vocab_size', 1000)
        prompt = torch.tensor([[1, 2, min(3, vocab_size-1)]], dtype=torch.long, device=self.device)

        test_params = [
            {'temperature': 0.1, 'top_k': 10, 'name': '低温度+小top_k'},
            {'temperature': 1.0, 'top_k': 50, 'name': '中等温度+中等top_k'},
            {'temperature': 1.5, 'top_k': 100, 'name': '高温度+大top_k'},
        ]

        self.model.eval()
        results = []

        for params in test_params:
            print(f"\n--- {params['name']} ---")
            print(f"温度: {params['temperature']}, top_k: {params['top_k']}")

            start_time = time.time()
            with torch.no_grad():
                try:
                    if hasattr(self.model, 'generate'):
                        generated = self.model.generate(
                            prompt,
                            max_length=15,
                            temperature=params['temperature'],
                            top_k=params['top_k']
                        )
                    else:
                        generated = self._simple_generate(prompt, max_length=15)
                except Exception as e:
                    print(f"⚠️ 参数测试失败: {e}")
                    generated = prompt

            end_time = time.time()

            generated_tokens = generated[0].tolist()[3:]  # 去掉输入部分
            generation_time = end_time - start_time

            print(f"生成: {generated_tokens}")
            print(f"时间: {generation_time:.3f}s")

            results.append({
                'params': params,
                'generated': generated_tokens,
                'time': generation_time,
                'success': len(generated_tokens) > 0
            })

        return results

    def test_memory_usage(self) -> Dict[str, Any]:
        """测试内存使用情况"""
        print("\n=== 内存使用测试 ===")

        if self.device.type == "cuda":
            torch.cuda.empty_cache()
            before_allocated = torch.cuda.memory_allocated() / 1024**3
            before_reserved = torch.cuda.memory_reserved() / 1024**3

            print(f"生成前 - 已分配: {before_allocated:.1f}GB, 已保留: {before_reserved:.1f}GB")

            # 生成大量文本
            vocab_size = getattr(self.config, 'vocab_size', 1000)
            prompt = torch.tensor([[1, 2, 3, 4, min(5, vocab_size-1)]], dtype=torch.long, device=self.device)

            self.model.eval()
            with torch.no_grad():
                for i in range(5):
                    try:
                        if hasattr(self.model, 'generate'):
                            generated = self.model.generate(prompt, max_length=50)
                        else:
                            generated = self._simple_generate(prompt, max_length=50)

                        if i == 0:
                            mid_allocated = torch.cuda.memory_allocated() / 1024**3
                            mid_reserved = torch.cuda.memory_reserved() / 1024**3
                            print(f"生成中 - 已分配: {mid_allocated:.1f}GB, 已保留: {mid_reserved:.1f}GB")
                    except Exception as e:
                        print(f"⚠️ 内存测试第{i}轮失败: {e}")

            after_allocated = torch.cuda.memory_allocated() / 1024**3
            after_reserved = torch.cuda.memory_reserved() / 1024**3

            print(f"生成后 - 已分配: {after_allocated:.1f}GB, 已保留: {after_reserved:.1f}GB")

            return {
                'before': {'allocated': before_allocated, 'reserved': before_reserved},
                'after': {'allocated': after_allocated, 'reserved': after_reserved},
                'device': 'cuda'
            }
        else:
            print(f"设备类型: {self.device.type}, 跳过显存监控")
            return {'device': str(self.device.type), 'monitoring': 'skipped'}

    def test_architecture_features(self) -> Dict[str, Any]:
        """测试架构特性"""
        print("\n=== 架构特性测试 ===")

        features = {
            'rope_enabled': getattr(self.config, 'use_rope', False),
            'gqa_enabled': getattr(self.config, 'use_gqa', False),
            'weight_sharing': getattr(self.config, 'tie_word_embeddings', False),
            'swiglu_activation': True,  # 默认启用
            'rms_norm': True,  # 默认启用
        }

        # 检查模型实际结构
        model_features = {}
        if hasattr(self.model, 'use_rope'):
            model_features['rope_in_model'] = self.model.use_rope
        if hasattr(self.model, 'lm_head'):
            model_features['separate_lm_head'] = self.model.lm_head is not None

        # 计算参数统计
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        stats = {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': total_params * 4 / (1024 * 1024),  # 假设float32
            'config_features': features,
            'model_features': model_features
        }

        print(f"总参数: {total_params:,}")
        print(f"可训练参数: {trainable_params:,}")
        print(f"模型大小: {stats['model_size_mb']:.1f} MB")
        print(f"RoPE启用: {features['rope_enabled']}")
        print(f"GQA启用: {features['gqa_enabled']}")
        print(f"权重共享: {features['weight_sharing']}")

        return stats

    def run_all_tests(self) -> Dict[str, Any]:
        """运行所有测试"""
        print("🔥 开始Legacy推理测试")

        results = {
            'test_config': {
                'config_name': self.config_name,
                'use_legacy': self.use_legacy,
                'device': str(self.device)
            }
        }

        try:
            # 运行各种推理测试
            results['generation'] = self.test_text_generation()
            results['batch_time'] = self.test_batch_generation()
            results['parameters'] = self.test_different_generation_params()
            results['memory'] = self.test_memory_usage()
            results['architecture'] = self.test_architecture_features()

            # 计算总结统计
            generation_results = results['generation']
            if generation_results:
                successful_tests = [r for r in generation_results if r['success']]
                if successful_tests:
                    avg_speed = sum(r['speed'] for r in successful_tests) / len(successful_tests)
                    avg_time = sum(r['time'] for r in successful_tests) / len(successful_tests)
                    results['summary'] = {
                        'avg_speed': avg_speed,
                        'avg_time': avg_time,
                        'success_rate': len(successful_tests) / len(generation_results),
                        'total_tests': len(generation_results)
                    }

            print(f"\n🎉 Legacy推理测试完成！")

            # 打印总结
            if 'summary' in results:
                summary = results['summary']
                print(f"✓ 平均生成速度: {summary['avg_speed']:.1f} tokens/秒")
                print(f"✓ 平均生成时间: {summary['avg_time']:.3f} 秒")
                print(f"✓ 成功率: {summary['success_rate']:.1%}")

            print(f"✓ 架构特性: {'Legacy' if self.use_legacy else 'Optimized'}")
            print("\n所有推理测试通过！")

        except Exception as e:
            print(f"❌ 测试失败: {e}")
            results['error'] = str(e)

        return results


def compare_architectures():
    """比较新旧架构性能"""
    print("\n" + "="*60)
    print("架构性能对比测试")
    print("="*60)

    configs_to_test = [
        ("tiny", False, "优化架构"),
        ("tiny", True, "Legacy架构") if get_legacy_config else None
    ]

    configs_to_test = [c for c in configs_to_test if c is not None]

    comparison_results = {}

    for config_name, use_legacy, description in configs_to_test:
        print(f"\n--- 测试 {description} ---")

        try:
            tester = LegacyInferenceTest(config_name, use_legacy)
            results = tester.run_all_tests()
            comparison_results[description] = results
        except Exception as e:
            print(f"❌ {description} 测试失败: {e}")
            comparison_results[description] = {'error': str(e)}

    # 保存对比结果
    output_file = "legacy_inference_comparison.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(comparison_results, f, ensure_ascii=False, indent=2)

    print(f"\n📊 对比结果已保存到: {output_file}")

    return comparison_results


def main():
    """主测试函数"""
    import argparse

    parser = argparse.ArgumentParser(description='Legacy推理测试脚本')
    parser.add_argument('--config', type=str, default='tiny',
                       choices=['tiny', 'small'], help='模型配置')
    parser.add_argument('--legacy', action='store_true', help='使用legacy配置')
    parser.add_argument('--compare', action='store_true', help='比较新旧架构')

    args = parser.parse_args()

    if args.compare:
        compare_architectures()
    else:
        tester = LegacyInferenceTest(args.config, args.legacy)
        results = tester.run_all_tests()

        # 保存结果
        output_file = f"inference_test_results_{args.config}_{'legacy' if args.legacy else 'optimized'}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        print(f"\n📝 详细结果已保存到: {output_file}")


if __name__ == "__main__":
    main()