#!/usr/bin/env python3
"""
优化版推理脚本
展示所有新能力：工具调用、ultra think、高效推理等
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

import torch
import torch.nn.functional as F
import json
import argparse
import time
from typing import List, Dict, Optional

from src.model.config import MiniGPTConfig
from src.model.transformer import MiniGPT


class OptimizedInference:
    """优化的推理引擎"""

    def __init__(self, model_path: str, device: str = 'auto'):
        """
        初始化推理引擎

        Args:
            model_path: 模型路径
            device: 设备类型
        """
        self.device = self._setup_device(device)
        self.model, self.config, self.tokenizer = self._load_model(model_path)

        print(f"Loaded model with {sum(p.numel() for p in self.model.parameters()):,} parameters")
        print(f"Using device: {self.device}")
        print(f"Model config: {self.config}")

    def _setup_device(self, device: str) -> torch.device:
        """设置计算设备"""
        if device == 'auto':
            if torch.cuda.is_available():
                return torch.device('cuda')
            elif torch.backends.mps.is_available():
                return torch.device('mps')
            else:
                return torch.device('cpu')
        else:
            return torch.device(device)

    def _load_model(self, model_path: str):
        """加载模型"""
        checkpoint = torch.load(model_path, map_location=self.device)

        # 加载配置
        if 'config' in checkpoint:
            config_dict = checkpoint['config']
            config = MiniGPTConfig.from_dict(config_dict)
        else:
            # 默认配置
            from src.model.config import get_small_config
            config = get_small_config()

        # 创建模型
        model = MiniGPT(config)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(self.device)
        model.eval()

        # 加载tokenizer
        tokenizer = checkpoint.get('vocab', None)
        if tokenizer is None:
            # 创建默认tokenizer
            tokenizer = self._create_default_tokenizer()

        return model, config, tokenizer

    def _create_default_tokenizer(self):
        """创建默认tokenizer"""
        chars = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,!?;:()[]{}"\'-\n你我他的是在有一个这那了和为'
        vocab = ['<pad>', '<unk>', '<bos>', '<eos>', '<|tool_call|>', '<|ultra_think|>'] + list(chars)

        return {
            'vocab': vocab,
            'char_to_id': {char: i for i, char in enumerate(vocab)},
            'id_to_char': {i: char for char, i in enumerate(vocab)}
        }

    def tokenize(self, text: str) -> List[int]:
        """文本分词"""
        tokens = [self.tokenizer['char_to_id'].get('<bos>', 2)]

        for char in text:
            tokens.append(self.tokenizer['char_to_id'].get(char, 1))  # UNK=1

        return tokens

    def detokenize(self, tokens: List[int]) -> str:
        """token解码"""
        text = ""
        for token in tokens:
            if token in self.tokenizer['id_to_char']:
                char = self.tokenizer['id_to_char'][token]
                if char not in ['<pad>', '<bos>', '<eos>']:
                    text += char
        return text

    def generate(self,
                prompt: str,
                max_length: int = 100,
                temperature: float = 0.8,
                top_k: int = 50,
                top_p: float = 0.9,
                do_sample: bool = True) -> str:
        """文本生成"""

        # 分词
        input_tokens = self.tokenize(prompt)
        input_ids = torch.tensor([input_tokens], dtype=torch.long, device=self.device)

        print(f"Prompt tokens: {len(input_tokens)}")

        generated_tokens = input_tokens.copy()

        with torch.no_grad():
            for _ in range(max_length):
                # 限制输入长度
                if input_ids.size(1) > self.config.max_position_embeddings - 1:
                    input_ids = input_ids[:, -self.config.max_position_embeddings + 1:]

                # 前向传播
                logits = self.model(input_ids)
                next_token_logits = logits[0, -1, :] / temperature

                # 应用top-k过滤
                if top_k > 0:
                    top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)
                    next_token_logits = torch.full_like(next_token_logits, -float('inf'))
                    next_token_logits.scatter_(0, top_k_indices, top_k_logits)

                # 应用top-p过滤
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
                    sorted_indices_to_remove[0] = 0

                    indices_to_remove = sorted_indices[sorted_indices_to_remove]
                    next_token_logits[indices_to_remove] = -float('inf')

                # 采样
                if do_sample:
                    probs = F.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

                # 检查结束条件
                if next_token.item() == self.tokenizer['char_to_id'].get('<eos>', 3):
                    break

                # 添加到序列
                generated_tokens.append(next_token.item())
                input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)

        # 解码
        generated_text = self.detokenize(generated_tokens)
        return generated_text

    def tool_calling_inference(self, query: str) -> Dict:
        """工具调用推理"""
        print(f"\n🔧 Tool Calling Inference: {query}")

        # 构造工具调用prompt
        tool_prompt = f"用户: {query}\n助手: 我来帮您"

        # 生成响应
        start_time = time.time()
        response = self.generate(
            tool_prompt,
            max_length=200,
            temperature=0.7,
            top_k=30
        )
        inference_time = time.time() - start_time

        # 解析工具调用（简化版本）
        tools_detected = []
        if "查询" in query or "搜索" in query:
            tools_detected.append("web_search")
        if "计算" in query or "算" in query:
            tools_detected.append("calculator")
        if "天气" in query:
            tools_detected.append("weather_api")
        if "翻译" in query:
            tools_detected.append("translator")

        return {
            "query": query,
            "response": response,
            "tools_detected": tools_detected,
            "inference_time": inference_time
        }

    def ultra_think_inference(self, problem: str) -> Dict:
        """Ultra Think深度推理"""
        print(f"\n🧠 Ultra Think Inference: {problem}")

        # 构造ultra think prompt
        ultra_prompt = f"""作为alex-ckl.com开发的AI助手，请展示ultra think能力分析：{problem}

<ultra_think>
让我深度分析这个问题：
"""

        start_time = time.time()
        response = self.generate(
            ultra_prompt,
            max_length=300,
            temperature=0.8,
            top_k=40
        )
        inference_time = time.time() - start_time

        # 分析推理质量（简化指标）
        thinking_markers = ["分析", "考虑", "因为", "所以", "首先", "其次", "最后"]
        thinking_score = sum(1 for marker in thinking_markers if marker in response)

        return {
            "problem": problem,
            "response": response,
            "thinking_score": thinking_score,
            "inference_time": inference_time
        }

    def benchmark_performance(self, num_samples: int = 10) -> Dict:
        """性能基准测试"""
        print(f"\n📊 Performance Benchmark ({num_samples} samples)")

        test_prompts = [
            "你好，今天天气怎么样？",
            "请帮我计算 123 + 456",
            "什么是人工智能？",
            "How are you today?",
            "解释一下机器学习的原理",
        ]

        times = []
        tokens_per_second = []

        for i in range(num_samples):
            prompt = test_prompts[i % len(test_prompts)]

            start_time = time.time()
            response = self.generate(prompt, max_length=50, temperature=0.7)
            end_time = time.time()

            inference_time = end_time - start_time
            response_tokens = len(self.tokenize(response))
            tps = response_tokens / inference_time if inference_time > 0 else 0

            times.append(inference_time)
            tokens_per_second.append(tps)

            print(f"  Sample {i+1}: {inference_time:.3f}s, {tps:.1f} tokens/s")

        return {
            "num_samples": num_samples,
            "avg_time": sum(times) / len(times),
            "avg_tokens_per_second": sum(tokens_per_second) / len(tokens_per_second),
            "min_time": min(times),
            "max_time": max(times)
        }


def interactive_mode(inference_engine: OptimizedInference):
    """交互式模式"""
    print("\n🎯 进入交互式模式 (输入 'quit' 退出)")
    print("命令:")
    print("  - normal: <text>     # 普通生成")
    print("  - tool: <query>      # 工具调用")
    print("  - think: <problem>   # Ultra Think")
    print("  - benchmark          # 性能测试")
    print("  - quit               # 退出")

    while True:
        try:
            user_input = input("\n> ").strip()

            if user_input.lower() == 'quit':
                break
            elif user_input.lower() == 'benchmark':
                result = inference_engine.benchmark_performance()
                print(f"平均推理时间: {result['avg_time']:.3f}s")
                print(f"平均生成速度: {result['avg_tokens_per_second']:.1f} tokens/s")
            elif user_input.startswith('tool:'):
                query = user_input[5:].strip()
                result = inference_engine.tool_calling_inference(query)
                print(f"工具检测: {result['tools_detected']}")
                print(f"响应: {result['response']}")
                print(f"推理时间: {result['inference_time']:.3f}s")
            elif user_input.startswith('think:'):
                problem = user_input[6:].strip()
                result = inference_engine.ultra_think_inference(problem)
                print(f"响应: {result['response']}")
                print(f"思维深度评分: {result['thinking_score']}")
                print(f"推理时间: {result['inference_time']:.3f}s")
            elif user_input.startswith('normal:'):
                text = user_input[7:].strip()
                response = inference_engine.generate(text, max_length=100)
                print(f"响应: {response}")
            else:
                # 默认普通生成
                response = inference_engine.generate(user_input, max_length=100)
                print(f"响应: {response}")

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"错误: {e}")

    print("退出交互式模式")


def main():
    parser = argparse.ArgumentParser(description='优化版MiniGPT推理脚本')
    parser.add_argument('--model-path', type=str, required=True, help='模型路径')
    parser.add_argument('--device', type=str, default='auto', help='设备类型')
    parser.add_argument('--mode', type=str, default='interactive',
                       choices=['interactive', 'single', 'tool', 'think', 'benchmark'],
                       help='推理模式')
    parser.add_argument('--prompt', type=str, help='输入prompt（single模式）')
    parser.add_argument('--max-length', type=int, default=100, help='最大生成长度')
    parser.add_argument('--temperature', type=float, default=0.8, help='采样温度')

    args = parser.parse_args()

    # 初始化推理引擎
    print("🚀 Initializing Optimized Inference Engine...")
    inference_engine = OptimizedInference(args.model_path, args.device)

    if args.mode == 'interactive':
        interactive_mode(inference_engine)

    elif args.mode == 'single':
        if not args.prompt:
            print("错误: single模式需要提供 --prompt 参数")
            return

        response = inference_engine.generate(
            args.prompt,
            max_length=args.max_length,
            temperature=args.temperature
        )
        print(f"输入: {args.prompt}")
        print(f"输出: {response}")

    elif args.mode == 'tool':
        if not args.prompt:
            print("错误: tool模式需要提供 --prompt 参数")
            return

        result = inference_engine.tool_calling_inference(args.prompt)
        print(json.dumps(result, ensure_ascii=False, indent=2))

    elif args.mode == 'think':
        if not args.prompt:
            print("错误: think模式需要提供 --prompt 参数")
            return

        result = inference_engine.ultra_think_inference(args.prompt)
        print(json.dumps(result, ensure_ascii=False, indent=2))

    elif args.mode == 'benchmark':
        result = inference_engine.benchmark_performance()
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()