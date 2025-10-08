#!/usr/bin/env python3
"""
模型评估脚本
全面评估架构升级后的模型性能：困惑度、生成质量、工具调用、ultra think等
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

import torch
import torch.nn.functional as F
import json
import argparse
import time
import math
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
from collections import defaultdict

from src.model.config import MiniGPTConfig
from src.model.transformer import MiniGPT


class ModelEvaluator:
    """模型评估器"""

    def __init__(self, model_path: str, device: str = 'auto'):
        self.device = self._setup_device(device)
        self.model, self.config, self.tokenizer = self._load_model(model_path)
        self.model.eval()

        print(f"Loaded model with {sum(p.numel() for p in self.model.parameters()):,} parameters")
        print(f"Using device: {self.device}")

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
            from src.model.config import get_small_config
            config = get_small_config()

        # 创建模型
        model = MiniGPT(config)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(self.device)

        # 加载tokenizer
        tokenizer = checkpoint.get('vocab', None)
        if tokenizer is None:
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
            tokens.append(self.tokenizer['char_to_id'].get(char, 1))
        tokens.append(self.tokenizer['char_to_id'].get('<eos>', 3))
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

    def calculate_perplexity(self, test_data: List[str]) -> float:
        """计算困惑度"""
        print("Calculating perplexity...")

        total_log_prob = 0
        total_tokens = 0

        with torch.no_grad():
            for text in test_data:
                tokens = self.tokenize(text)
                if len(tokens) < 2:
                    continue

                input_ids = torch.tensor([tokens[:-1]], dtype=torch.long, device=self.device)
                targets = torch.tensor([tokens[1:]], dtype=torch.long, device=self.device)

                if input_ids.size(1) > self.config.max_position_embeddings:
                    continue

                logits = self.model(input_ids)
                log_probs = F.log_softmax(logits, dim=-1)

                # 计算每个token的log概率
                for i, target in enumerate(targets[0]):
                    if target.item() != 0:  # 忽略padding
                        token_log_prob = log_probs[0, i, target.item()].item()
                        total_log_prob += token_log_prob
                        total_tokens += 1

        if total_tokens == 0:
            return float('inf')

        avg_log_prob = total_log_prob / total_tokens
        perplexity = math.exp(-avg_log_prob)

        print(f"Perplexity: {perplexity:.2f} (lower is better)")
        return perplexity

    def evaluate_generation_quality(self, prompts: List[str]) -> Dict[str, Any]:
        """评估生成质量"""
        print("Evaluating generation quality...")

        results = {
            'num_prompts': len(prompts),
            'avg_generation_time': 0,
            'avg_tokens_per_second': 0,
            'response_lengths': [],
            'diversity_scores': [],
            'coherence_scores': []
        }

        generation_times = []
        tokens_per_second = []

        for prompt in prompts:
            start_time = time.time()

            # 生成响应
            response = self._generate_text(prompt, max_length=100)

            end_time = time.time()
            generation_time = end_time - start_time

            # 计算指标
            response_tokens = len(self.tokenize(response))
            tps = response_tokens / generation_time if generation_time > 0 else 0

            generation_times.append(generation_time)
            tokens_per_second.append(tps)
            results['response_lengths'].append(response_tokens)

            # 简单的多样性评分（不重复n-gram比例）
            diversity = self._calculate_diversity(response)
            results['diversity_scores'].append(diversity)

            # 简单的连贯性评分
            coherence = self._calculate_coherence(prompt, response)
            results['coherence_scores'].append(coherence)

        # 计算平均值
        results['avg_generation_time'] = sum(generation_times) / len(generation_times)
        results['avg_tokens_per_second'] = sum(tokens_per_second) / len(tokens_per_second)
        results['avg_response_length'] = sum(results['response_lengths']) / len(results['response_lengths'])
        results['avg_diversity_score'] = sum(results['diversity_scores']) / len(results['diversity_scores'])
        results['avg_coherence_score'] = sum(results['coherence_scores']) / len(results['coherence_scores'])

        print(f"Average generation time: {results['avg_generation_time']:.3f}s")
        print(f"Average tokens per second: {results['avg_tokens_per_second']:.1f}")
        print(f"Average diversity score: {results['avg_diversity_score']:.2f}")
        print(f"Average coherence score: {results['avg_coherence_score']:.2f}")

        return results

    def _generate_text(self, prompt: str, max_length: int = 100) -> str:
        """生成文本"""
        input_tokens = self.tokenize(prompt)
        input_ids = torch.tensor([input_tokens], dtype=torch.long, device=self.device)

        generated_tokens = input_tokens.copy()

        with torch.no_grad():
            for _ in range(max_length):
                if input_ids.size(1) > self.config.max_position_embeddings - 1:
                    input_ids = input_ids[:, -self.config.max_position_embeddings + 1:]

                logits = self.model(input_ids)
                next_token_logits = logits[0, -1, :] / 0.8  # temperature

                # Top-k采样
                top_k = 50
                top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)
                next_token_logits = torch.full_like(next_token_logits, -float('inf'))
                next_token_logits.scatter_(0, top_k_indices, top_k_logits)

                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

                if next_token.item() == self.tokenizer['char_to_id'].get('<eos>', 3):
                    break

                generated_tokens.append(next_token.item())
                input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)

        return self.detokenize(generated_tokens)

    def _calculate_diversity(self, text: str) -> float:
        """计算文本多样性（基于不重复bigram比例）"""
        words = text.split()
        if len(words) < 2:
            return 0.0

        bigrams = set()
        total_bigrams = 0

        for i in range(len(words) - 1):
            bigram = (words[i], words[i + 1])
            bigrams.add(bigram)
            total_bigrams += 1

        return len(bigrams) / total_bigrams if total_bigrams > 0 else 0.0

    def _calculate_coherence(self, prompt: str, response: str) -> float:
        """计算连贯性（简化版本：基于prompt和response的词汇重叠）"""
        prompt_words = set(prompt.lower().split())
        response_words = set(response.lower().split())

        if len(prompt_words) == 0 or len(response_words) == 0:
            return 0.0

        overlap = len(prompt_words.intersection(response_words))
        total_unique = len(prompt_words.union(response_words))

        return overlap / total_unique if total_unique > 0 else 0.0

    def evaluate_tool_calling(self, tool_queries: List[str]) -> Dict[str, Any]:
        """评估工具调用能力"""
        print("Evaluating tool calling capabilities...")

        results = {
            'num_queries': len(tool_queries),
            'tool_detection_accuracy': 0,
            'response_relevance': 0,
            'detected_tools': defaultdict(int)
        }

        correct_detections = 0

        for query in tool_queries:
            # 生成响应
            response = self._generate_text(f"用户: {query}\n助手:", max_length=150)

            # 检测工具（简化版本）
            detected_tools = self._detect_tools_in_response(response)
            expected_tools = self._get_expected_tools(query)

            # 计算准确性
            if any(tool in detected_tools for tool in expected_tools):
                correct_detections += 1

            # 记录检测到的工具
            for tool in detected_tools:
                results['detected_tools'][tool] += 1

        results['tool_detection_accuracy'] = correct_detections / len(tool_queries) if tool_queries else 0
        results['detected_tools'] = dict(results['detected_tools'])

        print(f"Tool detection accuracy: {results['tool_detection_accuracy']:.2%}")
        print(f"Detected tools distribution: {results['detected_tools']}")

        return results

    def _detect_tools_in_response(self, response: str) -> List[str]:
        """从响应中检测工具调用"""
        tools = []
        response_lower = response.lower()

        # 简单的关键词匹配
        tool_keywords = {
            'search': ['搜索', '查询', 'search', 'query'],
            'calculator': ['计算', '算', 'calculate', 'math'],
            'weather': ['天气', 'weather'],
            'translator': ['翻译', 'translate'],
            'email': ['邮件', 'email'],
            'calendar': ['日历', '提醒', 'calendar', 'reminder']
        }

        for tool, keywords in tool_keywords.items():
            if any(keyword in response_lower for keyword in keywords):
                tools.append(tool)

        return tools

    def _get_expected_tools(self, query: str) -> List[str]:
        """获取查询的预期工具"""
        query_lower = query.lower()
        expected = []

        if any(word in query_lower for word in ['搜索', '查询', 'search']):
            expected.append('search')
        if any(word in query_lower for word in ['计算', '算', 'calculate']):
            expected.append('calculator')
        if any(word in query_lower for word in ['天气', 'weather']):
            expected.append('weather')
        if any(word in query_lower for word in ['翻译', 'translate']):
            expected.append('translator')

        return expected

    def evaluate_ultra_think(self, thinking_problems: List[str]) -> Dict[str, Any]:
        """评估ultra think能力"""
        print("Evaluating ultra think capabilities...")

        results = {
            'num_problems': len(thinking_problems),
            'thinking_depth_scores': [],
            'analysis_quality_scores': [],
            'avg_thinking_depth': 0,
            'avg_analysis_quality': 0
        }

        for problem in thinking_problems:
            prompt = f"作为alex-ckl.com的AI助手，请展示ultra think能力分析：{problem}\n\n<ultra_think>"

            response = self._generate_text(prompt, max_length=300)

            # 评估思维深度
            depth_score = self._evaluate_thinking_depth(response)
            results['thinking_depth_scores'].append(depth_score)

            # 评估分析质量
            quality_score = self._evaluate_analysis_quality(response)
            results['analysis_quality_scores'].append(quality_score)

        # 计算平均值
        if results['thinking_depth_scores']:
            results['avg_thinking_depth'] = sum(results['thinking_depth_scores']) / len(results['thinking_depth_scores'])
        if results['analysis_quality_scores']:
            results['avg_analysis_quality'] = sum(results['analysis_quality_scores']) / len(results['analysis_quality_scores'])

        print(f"Average thinking depth: {results['avg_thinking_depth']:.2f}")
        print(f"Average analysis quality: {results['avg_analysis_quality']:.2f}")

        return results

    def _evaluate_thinking_depth(self, response: str) -> float:
        """评估思维深度"""
        # 基于关键思维词汇的出现频率
        thinking_indicators = [
            '分析', '考虑', '因为', '所以', '首先', '其次', '最后',
            '然而', '但是', '另外', '此外', '综合', '总结', '结论'
        ]

        score = 0
        for indicator in thinking_indicators:
            score += response.count(indicator)

        # 归一化到0-1范围
        return min(score / 10.0, 1.0)

    def _evaluate_analysis_quality(self, response: str) -> float:
        """评估分析质量"""
        # 基于结构化分析的特征
        quality_features = [
            ('多维度', ['维度', '角度', '方面', '层面']),
            ('逻辑性', ['逻辑', '推理', '推论', '演绎']),
            ('深度', ['深入', '深度', '本质', '根本']),
            ('系统性', ['系统', '整体', '全面', '完整'])
        ]

        score = 0
        for feature_name, keywords in quality_features:
            if any(keyword in response for keyword in keywords):
                score += 1

        return score / len(quality_features)

    def benchmark_performance(self) -> Dict[str, Any]:
        """性能基准测试"""
        print("Running performance benchmark...")

        test_sequences = [
            "Hello, how are you?",
            "What is artificial intelligence?",
            "请解释一下机器学习的原理",
            "计算123加456等于多少",
            "今天天气怎么样？"
        ]

        results = {
            'inference_times': [],
            'memory_usage': [],
            'tokens_per_second': [],
            'model_size_mb': 0
        }

        # 计算模型大小
        model_size = sum(p.numel() * p.element_size() for p in self.model.parameters()) / (1024 * 1024)
        results['model_size_mb'] = model_size

        for seq in test_sequences:
            # 多次测试取平均
            times = []
            for _ in range(3):
                start_time = time.time()
                response = self._generate_text(seq, max_length=50)
                end_time = time.time()

                inference_time = end_time - start_time
                times.append(inference_time)

                # 计算tokens/s
                response_tokens = len(self.tokenize(response))
                tps = response_tokens / inference_time if inference_time > 0 else 0
                results['tokens_per_second'].append(tps)

            avg_time = sum(times) / len(times)
            results['inference_times'].append(avg_time)

        # 计算平均指标
        results['avg_inference_time'] = sum(results['inference_times']) / len(results['inference_times'])
        results['avg_tokens_per_second'] = sum(results['tokens_per_second']) / len(results['tokens_per_second'])

        print(f"Model size: {model_size:.1f} MB")
        print(f"Average inference time: {results['avg_inference_time']:.3f}s")
        print(f"Average tokens per second: {results['avg_tokens_per_second']:.1f}")

        return results


def load_test_data(data_dir: str) -> Dict[str, List[str]]:
    """加载测试数据"""
    test_data = {
        'general_text': [],
        'tool_queries': [],
        'ultra_think_problems': []
    }

    # 通用文本（用于困惑度计算）
    general_texts = [
        "人工智能是计算机科学的一个分支。",
        "机器学习通过数据训练模型。",
        "深度学习使用神经网络进行学习。",
        "自然语言处理处理人类语言。",
        "Today is a beautiful day.",
        "Machine learning is fascinating.",
        "Natural language processing is important."
    ]
    test_data['general_text'] = general_texts

    # 工具调用查询
    tool_queries = [
        "请帮我搜索人工智能的最新发展",
        "计算123乘以456等于多少",
        "查询今天北京的天气情况",
        "翻译这句话：Hello world",
        "帮我发送一封邮件给张三",
        "设置明天上午9点的会议提醒"
    ]
    test_data['tool_queries'] = tool_queries

    # Ultra think问题
    ultra_think_problems = [
        "分析当前AI发展的主要趋势和挑战",
        "如何设计一个可持续的商业模式",
        "评估远程工作对企业文化的影响",
        "分析区块链技术的未来应用前景"
    ]
    test_data['ultra_think_problems'] = ultra_think_problems

    return test_data


def main():
    parser = argparse.ArgumentParser(description='模型评估脚本')
    parser.add_argument('--model-path', type=str, required=True, help='模型路径')
    parser.add_argument('--data-dir', type=str, default='data/test', help='测试数据目录')
    parser.add_argument('--output-dir', type=str, default='evaluation_results', help='输出目录')
    parser.add_argument('--device', type=str, default='auto', help='设备类型')

    args = parser.parse_args()

    # 创建输出目录
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # 初始化评估器
    print("🔍 Initializing Model Evaluator...")
    evaluator = ModelEvaluator(args.model_path, args.device)

    # 加载测试数据
    test_data = load_test_data(args.data_dir)

    # 运行评估
    results = {}

    print("\n" + "="*60)
    print("1. PERPLEXITY EVALUATION")
    print("="*60)
    results['perplexity'] = evaluator.calculate_perplexity(test_data['general_text'])

    print("\n" + "="*60)
    print("2. GENERATION QUALITY EVALUATION")
    print("="*60)
    generation_prompts = test_data['general_text'][:5]  # 使用前5个作为prompt
    results['generation_quality'] = evaluator.evaluate_generation_quality(generation_prompts)

    print("\n" + "="*60)
    print("3. TOOL CALLING EVALUATION")
    print("="*60)
    results['tool_calling'] = evaluator.evaluate_tool_calling(test_data['tool_queries'])

    print("\n" + "="*60)
    print("4. ULTRA THINK EVALUATION")
    print("="*60)
    results['ultra_think'] = evaluator.evaluate_ultra_think(test_data['ultra_think_problems'])

    print("\n" + "="*60)
    print("5. PERFORMANCE BENCHMARK")
    print("="*60)
    results['performance'] = evaluator.benchmark_performance()

    # 保存结果
    results['model_path'] = args.model_path
    results['evaluation_time'] = time.time()

    output_file = f"{args.output_dir}/evaluation_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n📊 Evaluation completed! Results saved to: {output_file}")

    # 打印总结
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    print(f"Perplexity: {results['perplexity']:.2f}")
    print(f"Average Generation Time: {results['generation_quality']['avg_generation_time']:.3f}s")
    print(f"Tool Detection Accuracy: {results['tool_calling']['tool_detection_accuracy']:.2%}")
    print(f"Ultra Think Depth Score: {results['ultra_think']['avg_thinking_depth']:.2f}")
    print(f"Model Size: {results['performance']['model_size_mb']:.1f} MB")
    print(f"Inference Speed: {results['performance']['avg_tokens_per_second']:.1f} tokens/s")


if __name__ == "__main__":
    main()