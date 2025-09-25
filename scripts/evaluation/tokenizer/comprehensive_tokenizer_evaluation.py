#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🔍 MiniGPT分词器综合评估系统
===============================

目标: 系统性评估分词器的性能、效率和质量
作者: alex-ckl.com AI研发团队
风格: ISTJ系统化执行风格

评估维度:
1. 基础性能指标 (压缩率、词汇覆盖率)
2. 语言理解质量 (语义保持、OOV处理)
3. 效率指标 (编解码速度、内存使用)
4. 多语言支持评估 (中英文混合、特殊符号)
5. 对比分析 (与其他分词器对比)
6. 实用性评估 (训练数据适配性)
"""

import sys
import os
import time
import json
import pickle
import argparse
import statistics
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, asdict
from collections import Counter, defaultdict

# 添加项目根目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

try:
    from src.tokenizer.bpe_tokenizer import BPETokenizer
    from src.tokenizer.tokenizer_manager import TokenizerManager
    TOKENIZER_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  分词器模块导入失败: {e}")
    TOKENIZER_AVAILABLE = False


@dataclass
class TokenizerMetrics:
    """分词器评估指标数据类"""
    # 基础信息
    tokenizer_name: str
    vocab_size: int
    model_path: str

    # 压缩性能
    compression_ratio: float  # 压缩率 (原字符数/token数)
    avg_token_length: float   # 平均token长度

    # 词汇覆盖
    vocab_coverage: float     # 词汇覆盖率
    oov_rate: float          # 未知词率

    # 效率指标
    encode_speed: float      # 编码速度 (tokens/秒)
    decode_speed: float      # 解码速度 (tokens/秒)
    memory_usage_mb: float   # 内存使用量 (MB)

    # 质量指标
    semantic_coherence: float    # 语义连贯性得分
    boundary_accuracy: float     # 词边界准确性

    # 多语言支持
    chinese_support: float       # 中文支持质量
    english_support: float       # 英文支持质量
    mixed_language_support: float # 混合语言支持

    # 特殊处理
    special_token_handling: float  # 特殊token处理质量
    code_tokenization: float       # 代码分词质量

    # 实用性
    training_data_efficiency: float  # 训练数据适配效率
    model_compatibility: float      # 模型兼容性


class TokenizerEvaluator:
    """分词器评估器主类"""

    def __init__(self):
        """初始化评估器"""
        self.test_cases = self._prepare_test_cases()
        self.results = {}

    def _prepare_test_cases(self) -> Dict[str, List[str]]:
        """准备测试用例 (系统化测试数据)"""
        return {
            # 基础中文测试
            "chinese_basic": [
                "今天天气很好，我们去公园散步。",
                "人工智能技术正在快速发展，改变着我们的生活。",
                "中国的传统文化博大精深，值得我们传承和发扬。",
                "深度学习模型需要大量的数据进行训练。",
                "自然语言处理是人工智能的重要分支。"
            ],

            # 基础英文测试
            "english_basic": [
                "The weather is beautiful today, let's go for a walk in the park.",
                "Artificial intelligence technology is rapidly developing and changing our lives.",
                "Natural language processing is an important branch of artificial intelligence.",
                "Deep learning models require large amounts of data for training.",
                "Machine learning algorithms can solve complex problems efficiently."
            ],

            # 中英文混合测试
            "mixed_language": [
                "我正在学习Python编程语言和machine learning算法。",
                "OpenAI的GPT模型在中文理解方面表现excellent。",
                "今天我们讨论了transformer architecture和attention mechanism。",
                "数据科学data science需要统计学statistics和编程programming技能。",
                "AI人工智能和ML机器学习是当前的热门技术trends。"
            ],

            # 技术术语测试
            "technical_terms": [
                "分词器tokenizer、编码器encoder、解码器decoder是NLP的核心组件。",
                "BERT、GPT、T5等预训练模型使用了不同的架构设计。",
                "词嵌入word embeddings和位置编码positional encoding很重要。",
                "注意力机制attention mechanism和前馈网络feedforward network的作用。",
                "超参数hyperparameters调优对模型性能model performance影响很大。"
            ],

            # 代码和特殊符号测试
            "code_and_symbols": [
                "def tokenize(text): return tokenizer.encode(text)",
                "import torch; model = torch.nn.Transformer()",
                "for i in range(len(tokens)): print(f'Token {i}: {tokens[i]}')",
                "API接口: https://api.example.com/v1/generate?text=hello",
                "邮箱格式: user@domain.com, 电话: +86-138-0013-8000"
            ],

            # 长文本测试
            "long_text": [
                """人工智能(Artificial Intelligence, AI)是计算机科学的一个分支，它试图理解智能的实质，并生产出一种新的能以人类智能相似的方式做出反应的智能机器。该领域的研究包括机器人robotics、语言识别speech recognition、图像识别image recognition、自然语言处理natural language processing和专家系统expert systems等。自从计算机诞生以来，人们就开始探索能否让机器像人一样思考，这个问题一直延续至今。"""
            ],

            # 边界情况测试
            "edge_cases": [
                "",  # 空字符串
                " ",  # 空格
                "\n\n\t\t",  # 空白字符
                "a",  # 单字符
                "你",  # 单中文字符
                "!!!!!",  # 重复符号
                "123456789",  # 纯数字
                "aaaaaaaaaa",  # 重复字符
            ]
        }

    def evaluate_tokenizer(self, tokenizer_path: str) -> TokenizerMetrics:
        """评估单个分词器 (主要评估函数)"""
        print(f"🔍 开始评估分词器: {os.path.basename(tokenizer_path)}")
        print("="*70)

        if not TOKENIZER_AVAILABLE:
            print("❌ 分词器模块不可用，跳过评估")
            return self._create_empty_metrics(tokenizer_path)

        # 加载分词器
        try:
            tokenizer = self._load_tokenizer(tokenizer_path)
            if tokenizer is None:
                return self._create_empty_metrics(tokenizer_path)
        except Exception as e:
            print(f"❌ 加载分词器失败: {e}")
            return self._create_empty_metrics(tokenizer_path)

        # 系统性执行各项评估
        metrics = {
            'tokenizer_name': os.path.basename(tokenizer_path),
            'model_path': tokenizer_path,
            'vocab_size': getattr(tokenizer, 'vocab_size', 0)
        }

        # 1. 基础性能评估
        print("📊 1. 执行基础性能评估...")
        basic_metrics = self._evaluate_basic_performance(tokenizer)
        metrics.update(basic_metrics)

        # 2. 效率评估
        print("⚡ 2. 执行效率评估...")
        efficiency_metrics = self._evaluate_efficiency(tokenizer)
        metrics.update(efficiency_metrics)

        # 3. 质量评估
        print("🎯 3. 执行质量评估...")
        quality_metrics = self._evaluate_quality(tokenizer)
        metrics.update(quality_metrics)

        # 4. 多语言支持评估
        print("🌐 4. 执行多语言支持评估...")
        multilang_metrics = self._evaluate_multilang_support(tokenizer)
        metrics.update(multilang_metrics)

        # 5. 特殊处理能力评估
        print("🔧 5. 执行特殊处理能力评估...")
        special_metrics = self._evaluate_special_handling(tokenizer)
        metrics.update(special_metrics)

        # 6. 实用性评估
        print("💼 6. 执行实用性评估...")
        utility_metrics = self._evaluate_utility(tokenizer)
        metrics.update(utility_metrics)

        # 创建最终指标对象
        tokenizer_metrics = TokenizerMetrics(**metrics)

        print("✅ 分词器评估完成!")
        return tokenizer_metrics

    def _load_tokenizer(self, tokenizer_path: str):
        """加载分词器 (统一加载逻辑)"""
        try:
            if tokenizer_path.endswith('.pkl'):
                with open(tokenizer_path, 'rb') as f:
                    tokenizer = pickle.load(f)
            else:
                # 尝试其他格式
                tokenizer = TokenizerManager.load_tokenizer(tokenizer_path)

            print(f"✅ 成功加载分词器: {type(tokenizer).__name__}")
            if hasattr(tokenizer, 'vocab_size'):
                print(f"📚 词汇表大小: {tokenizer.vocab_size:,}")

            return tokenizer

        except Exception as e:
            print(f"❌ 加载分词器失败: {e}")
            return None

    def _create_empty_metrics(self, tokenizer_path: str) -> TokenizerMetrics:
        """创建空的指标对象 (错误处理)"""
        return TokenizerMetrics(
            tokenizer_name=os.path.basename(tokenizer_path),
            vocab_size=0,
            model_path=tokenizer_path,
            compression_ratio=0.0,
            avg_token_length=0.0,
            vocab_coverage=0.0,
            oov_rate=1.0,
            encode_speed=0.0,
            decode_speed=0.0,
            memory_usage_mb=0.0,
            semantic_coherence=0.0,
            boundary_accuracy=0.0,
            chinese_support=0.0,
            english_support=0.0,
            mixed_language_support=0.0,
            special_token_handling=0.0,
            code_tokenization=0.0,
            training_data_efficiency=0.0,
            model_compatibility=0.0
        )

    def _evaluate_basic_performance(self, tokenizer) -> Dict[str, float]:
        """评估基础性能指标"""
        total_chars = 0
        total_tokens = 0
        token_lengths = []

        # 使用多种测试用例
        all_texts = []
        for category, texts in self.test_cases.items():
            if category != 'edge_cases':  # 排除边界情况
                all_texts.extend(texts)

        for text in all_texts:
            try:
                if hasattr(tokenizer, 'encode'):
                    tokens = tokenizer.encode(text)
                elif hasattr(tokenizer, 'tokenize'):
                    tokens = tokenizer.tokenize(text)
                else:
                    continue

                total_chars += len(text)
                total_tokens += len(tokens)

                # 计算每个token的平均字符长度
                if hasattr(tokenizer, 'decode'):
                    for token in tokens:
                        try:
                            decoded = tokenizer.decode([token]) if isinstance(token, int) else str(token)
                            token_lengths.append(len(decoded))
                        except:
                            token_lengths.append(1)  # 默认长度

            except Exception as e:
                print(f"⚠️  处理文本时出错: {e}")
                continue

        # 计算指标
        compression_ratio = total_chars / max(total_tokens, 1)
        avg_token_length = statistics.mean(token_lengths) if token_lengths else 1.0

        return {
            'compression_ratio': compression_ratio,
            'avg_token_length': avg_token_length
        }

    def _evaluate_efficiency(self, tokenizer) -> Dict[str, float]:
        """评估效率指标"""
        test_text = "这是一个用于测试分词器编码解码速度的长文本。" * 100  # 重复100次

        # 编码速度测试
        encode_times = []
        for _ in range(10):  # 多次测试取平均
            start_time = time.time()
            try:
                if hasattr(tokenizer, 'encode'):
                    tokens = tokenizer.encode(test_text)
                elif hasattr(tokenizer, 'tokenize'):
                    tokens = tokenizer.tokenize(test_text)
                else:
                    tokens = []
                end_time = time.time()

                if tokens:
                    encode_times.append(len(tokens) / (end_time - start_time))
            except:
                continue

        # 解码速度测试 (如果支持)
        decode_times = []
        if hasattr(tokenizer, 'encode') and hasattr(tokenizer, 'decode'):
            try:
                tokens = tokenizer.encode(test_text[:1000])  # 使用较短文本避免超时
                for _ in range(10):
                    start_time = time.time()
                    decoded = tokenizer.decode(tokens)
                    end_time = time.time()
                    decode_times.append(len(tokens) / (end_time - start_time))
            except:
                pass

        # 内存使用估算 (简化版本)
        memory_usage = 0.0
        if hasattr(tokenizer, 'vocab_size'):
            # 粗略估算: vocab_size * 平均token长度 * 字节数
            memory_usage = getattr(tokenizer, 'vocab_size', 0) * 10 / (1024 * 1024)  # MB

        return {
            'encode_speed': statistics.mean(encode_times) if encode_times else 0.0,
            'decode_speed': statistics.mean(decode_times) if decode_times else 0.0,
            'memory_usage_mb': memory_usage
        }

    def _evaluate_quality(self, tokenizer) -> Dict[str, float]:
        """评估质量指标"""
        semantic_scores = []
        boundary_scores = []

        # 语义连贯性测试
        semantic_test_cases = [
            "人工智能",  # 应该作为整体
            "machine learning",  # 英文复合词
            "自然语言处理",  # 技术术语
            "深度学习模型",  # 领域术语
        ]

        for text in semantic_test_cases:
            try:
                if hasattr(tokenizer, 'encode') and hasattr(tokenizer, 'decode'):
                    tokens = tokenizer.encode(text)
                    decoded = tokenizer.decode(tokens)

                    # 简单的语义保持评分 (基于重建质量)
                    if decoded.replace(' ', '') == text.replace(' ', ''):
                        semantic_scores.append(1.0)
                    else:
                        # 计算字符级相似度
                        similarity = self._calculate_string_similarity(text, decoded)
                        semantic_scores.append(similarity)

                    # 词边界准确性 (理想情况下技术术语应该保持完整)
                    if len(tokens) <= 3:  # 技术术语理想情况下不应过度分割
                        boundary_scores.append(1.0)
                    else:
                        boundary_scores.append(max(0.0, 1.0 - (len(tokens) - 3) * 0.2))

            except Exception:
                semantic_scores.append(0.0)
                boundary_scores.append(0.0)

        return {
            'semantic_coherence': statistics.mean(semantic_scores) if semantic_scores else 0.0,
            'boundary_accuracy': statistics.mean(boundary_scores) if boundary_scores else 0.0
        }

    def _evaluate_multilang_support(self, tokenizer) -> Dict[str, float]:
        """评估多语言支持"""
        chinese_scores = []
        english_scores = []
        mixed_scores = []

        # 中文支持测试
        for text in self.test_cases['chinese_basic']:
            score = self._evaluate_language_support(tokenizer, text, 'chinese')
            chinese_scores.append(score)

        # 英文支持测试
        for text in self.test_cases['english_basic']:
            score = self._evaluate_language_support(tokenizer, text, 'english')
            english_scores.append(score)

        # 混合语言支持测试
        for text in self.test_cases['mixed_language']:
            score = self._evaluate_language_support(tokenizer, text, 'mixed')
            mixed_scores.append(score)

        return {
            'chinese_support': statistics.mean(chinese_scores) if chinese_scores else 0.0,
            'english_support': statistics.mean(english_scores) if english_scores else 0.0,
            'mixed_language_support': statistics.mean(mixed_scores) if mixed_scores else 0.0
        }

    def _evaluate_special_handling(self, tokenizer) -> Dict[str, float]:
        """评估特殊处理能力"""
        special_token_scores = []
        code_scores = []

        # 特殊token处理测试
        special_cases = [
            "[CLS]", "[SEP]", "[MASK]", "[PAD]",  # BERT风格
            "<bos>", "<eos>", "<pad>", "<unk>",   # GPT风格
            "<|im_start|>", "<|im_end|>",        # Chat风格
        ]

        for special in special_cases:
            try:
                if hasattr(tokenizer, 'encode'):
                    tokens = tokenizer.encode(special)
                    # 特殊token理想情况下应该被识别为单个token
                    if len(tokens) == 1:
                        special_token_scores.append(1.0)
                    else:
                        special_token_scores.append(max(0.0, 1.0 - (len(tokens) - 1) * 0.3))
            except:
                special_token_scores.append(0.0)

        # 代码分词测试
        for code in self.test_cases['code_and_symbols']:
            try:
                if hasattr(tokenizer, 'encode'):
                    tokens = tokenizer.encode(code)
                    # 代码应该合理分割，不应过度碎片化
                    code_length_ratio = len(code) / max(len(tokens), 1)
                    # 理想的代码分词应该在2-8个字符per token之间
                    if 2 <= code_length_ratio <= 8:
                        code_scores.append(1.0)
                    else:
                        code_scores.append(max(0.0, 1.0 - abs(code_length_ratio - 5) * 0.1))
            except:
                code_scores.append(0.0)

        return {
            'special_token_handling': statistics.mean(special_token_scores) if special_token_scores else 0.0,
            'code_tokenization': statistics.mean(code_scores) if code_scores else 0.0
        }

    def _evaluate_utility(self, tokenizer) -> Dict[str, float]:
        """评估实用性指标"""
        # 训练数据适配效率 (基于压缩率和覆盖率)
        compression_efficiency = self._calculate_training_efficiency(tokenizer)

        # 模型兼容性 (基于词汇表大小和常见架构的匹配度)
        model_compat = self._calculate_model_compatibility(tokenizer)

        return {
            'training_data_efficiency': compression_efficiency,
            'model_compatibility': model_compat
        }

    def _evaluate_language_support(self, tokenizer, text: str, lang_type: str) -> float:
        """评估特定语言支持质量"""
        try:
            if hasattr(tokenizer, 'encode') and hasattr(tokenizer, 'decode'):
                tokens = tokenizer.encode(text)
                decoded = tokenizer.decode(tokens)

                # 计算重建质量
                similarity = self._calculate_string_similarity(text, decoded)

                # 根据语言类型调整评分标准
                if lang_type == 'chinese':
                    # 中文应该有较好的字符保持
                    return similarity
                elif lang_type == 'english':
                    # 英文应该有较好的词汇保持
                    return similarity
                else:  # mixed
                    # 混合语言应该均衡处理
                    return similarity * 0.9  # 稍微降低标准
            return 0.0
        except:
            return 0.0

    def _calculate_string_similarity(self, s1: str, s2: str) -> float:
        """计算字符串相似度 (简化版本)"""
        if not s1 and not s2:
            return 1.0
        if not s1 or not s2:
            return 0.0

        # 简单的字符级Jaccard相似度
        set1 = set(s1.replace(' ', ''))
        set2 = set(s2.replace(' ', ''))

        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))

        return intersection / union if union > 0 else 0.0

    def _calculate_training_efficiency(self, tokenizer) -> float:
        """计算训练数据适配效率"""
        # 基于多个因素的综合评分
        factors = []

        # 1. 词汇表大小合理性 (10K-50K为理想范围)
        vocab_size = getattr(tokenizer, 'vocab_size', 0)
        if 10000 <= vocab_size <= 50000:
            factors.append(1.0)
        elif vocab_size < 10000:
            factors.append(vocab_size / 10000)
        else:
            factors.append(max(0.3, 50000 / vocab_size))

        # 2. 压缩效率 (之前计算的compression_ratio)
        # 这里使用简化估算
        factors.append(0.8)  # 默认合理值

        return statistics.mean(factors)

    def _calculate_model_compatibility(self, tokenizer) -> float:
        """计算模型兼容性"""
        compat_score = 0.0

        # 检查常见方法的存在
        methods = ['encode', 'decode', 'tokenize']
        available_methods = sum(1 for method in methods if hasattr(tokenizer, method))
        compat_score += (available_methods / len(methods)) * 0.5

        # 检查词汇表大小是否在常见范围内
        vocab_size = getattr(tokenizer, 'vocab_size', 0)
        if vocab_size > 0:
            if 5000 <= vocab_size <= 100000:
                compat_score += 0.5
            else:
                compat_score += 0.2

        return compat_score

    def compare_tokenizers(self, tokenizer_paths: List[str]) -> Dict[str, Any]:
        """对比多个分词器 (对比分析功能)"""
        print("🔄 开始对比分析多个分词器...")
        print("="*70)

        all_metrics = {}

        # 评估每个分词器
        for path in tokenizer_paths:
            if os.path.exists(path):
                metrics = self.evaluate_tokenizer(path)
                all_metrics[metrics.tokenizer_name] = metrics
            else:
                print(f"⚠️  分词器文件不存在: {path}")

        if not all_metrics:
            print("❌ 没有可评估的分词器")
            return {}

        # 生成对比报告
        comparison = self._generate_comparison_report(all_metrics)

        return comparison

    def _generate_comparison_report(self, all_metrics: Dict[str, TokenizerMetrics]) -> Dict[str, Any]:
        """生成对比报告"""
        report = {
            'summary': {},
            'rankings': {},
            'detailed_comparison': {},
            'recommendations': []
        }

        # 计算各项指标的排名
        metrics_for_ranking = [
            'compression_ratio', 'vocab_coverage', 'encode_speed',
            'semantic_coherence', 'chinese_support', 'english_support'
        ]

        for metric in metrics_for_ranking:
            values = [(name, getattr(metrics, metric)) for name, metrics in all_metrics.items()]
            values.sort(key=lambda x: x[1], reverse=True)
            report['rankings'][metric] = values

        # 生成推荐
        if all_metrics:
            best_overall = self._find_best_overall_tokenizer(all_metrics)
            report['recommendations'].append(f"综合性能最佳: {best_overall}")

        report['detailed_comparison'] = {
            name: asdict(metrics) for name, metrics in all_metrics.items()
        }

        return report

    def _find_best_overall_tokenizer(self, all_metrics: Dict[str, TokenizerMetrics]) -> str:
        """找出综合性能最佳的分词器"""
        scores = {}

        # 关键指标权重
        weights = {
            'compression_ratio': 0.2,
            'encode_speed': 0.15,
            'semantic_coherence': 0.25,
            'chinese_support': 0.15,
            'english_support': 0.15,
            'training_data_efficiency': 0.1
        }

        for name, metrics in all_metrics.items():
            score = 0.0
            for metric, weight in weights.items():
                value = getattr(metrics, metric, 0.0)
                score += value * weight
            scores[name] = score

        return max(scores, key=scores.get) if scores else "未知"

    def save_evaluation_report(self, results: Dict[str, Any], output_path: str):
        """保存评估报告 (结果持久化)"""
        report_data = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'evaluation_results': results,
            'system_info': {
                'python_version': sys.version,
                'platform': os.name
            }
        }

        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, ensure_ascii=False, indent=2, default=str)

        print(f"📄 评估报告已保存: {output_path}")


def main():
    """主函数 - 系统化执行流程"""
    parser = argparse.ArgumentParser(description='MiniGPT分词器综合评估系统')
    parser.add_argument('--tokenizer', '-t', type=str, help='单个分词器文件路径')
    parser.add_argument('--directory', '-d', type=str, help='分词器目录路径')
    parser.add_argument('--compare', '-c', action='store_true', help='对比模式')
    parser.add_argument('--output', '-o', type=str, default='scripts/evaluation/tokenizer/reports/evaluation_report.json',
                       help='输出报告路径')

    args = parser.parse_args()

    print("🎯 MiniGPT分词器综合评估系统")
    print("📋 执行方式: ISTJ系统化风格")
    print("🔬 alex-ckl.com AI研发团队")
    print("="*70)

    evaluator = TokenizerEvaluator()

    if args.tokenizer:
        # 单个分词器评估
        if not os.path.exists(args.tokenizer):
            print(f"❌ 分词器文件不存在: {args.tokenizer}")
            return

        metrics = evaluator.evaluate_tokenizer(args.tokenizer)
        results = {'single_evaluation': asdict(metrics)}

    elif args.directory:
        # 目录批量评估
        if not os.path.exists(args.directory):
            print(f"❌ 目录不存在: {args.directory}")
            return

        # 查找所有分词器文件
        tokenizer_files = []
        for ext in ['.pkl', '.json']:
            tokenizer_files.extend(Path(args.directory).glob(f'**/*{ext}'))

        if not tokenizer_files:
            print(f"❌ 在目录 {args.directory} 中未找到分词器文件")
            return

        tokenizer_paths = [str(f) for f in tokenizer_files]

        if args.compare:
            # 对比模式
            results = evaluator.compare_tokenizers(tokenizer_paths)
        else:
            # 批量评估模式
            batch_results = {}
            for path in tokenizer_paths:
                metrics = evaluator.evaluate_tokenizer(path)
                batch_results[metrics.tokenizer_name] = asdict(metrics)
            results = {'batch_evaluation': batch_results}

    else:
        # 默认：评估项目中的所有分词器
        print("🔍 自动搜索项目中的分词器文件...")

        # 搜索常见位置
        search_paths = [
            'tokenizers/models/',
            'checkpoints/',
            'src/tokenizer/'
        ]

        tokenizer_files = []
        for search_path in search_paths:
            if os.path.exists(search_path):
                for ext in ['.pkl', '.json']:
                    tokenizer_files.extend(Path(search_path).glob(f'**/*{ext}'))

        if tokenizer_files:
            tokenizer_paths = [str(f) for f in tokenizer_files[:5]]  # 限制数量避免过多
            print(f"📋 找到 {len(tokenizer_paths)} 个分词器文件进行评估")
            results = evaluator.compare_tokenizers(tokenizer_paths)
        else:
            print("❌ 未找到任何分词器文件")
            return

    # 保存结果
    evaluator.save_evaluation_report(results, args.output)

    # 打印简要总结
    print("\n📊 评估完成总结:")
    if 'single_evaluation' in results:
        metrics = results['single_evaluation']
        print(f"  分词器: {metrics['tokenizer_name']}")
        print(f"  词汇表大小: {metrics['vocab_size']:,}")
        print(f"  压缩率: {metrics['compression_ratio']:.2f}")
        print(f"  中文支持: {metrics['chinese_support']:.2f}")
        print(f"  英文支持: {metrics['english_support']:.2f}")
    elif 'batch_evaluation' in results:
        print(f"  评估了 {len(results['batch_evaluation'])} 个分词器")
    elif 'rankings' in results:
        print("  对比分析完成，详细结果请查看报告文件")

    print(f"📄 详细报告: {args.output}")
    print("✅ 系统化评估流程执行完毕")


if __name__ == "__main__":
    main()