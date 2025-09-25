#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
⚡ MiniGPT分词器性能基准测试
==============================

专注于量化性能指标的精确测量
执行风格: ISTJ系统化测量与记录

测试项目:
1. 编码/解码速度基准测试
2. 内存使用量测量
3. 吞吐量测试
4. 并发性能测试
5. 大文本处理能力测试
"""

import sys
import os
import time
import psutil
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Tuple
import statistics
import json

sys.path.append(os.path.join(os.path.dirname(__file__), '../../../..'))


class TokenizerBenchmark:
    """分词器性能基准测试器"""

    def __init__(self):
        self.benchmark_texts = self._prepare_benchmark_texts()

    def _prepare_benchmark_texts(self) -> Dict[str, str]:
        """准备标准化的基准测试文本"""
        return {
            'short': "测试短文本tokenization性能。" * 10,
            'medium': "这是一个中等长度的文本，用于测试分词器的性能表现。" * 100,
            'long': "长文本性能测试，包含中文、English mixed content和各种符号!@#$%。" * 1000,
            'xlarge': "超大文本测试。" * 10000
        }

    def run_speed_benchmark(self, tokenizer, iterations: int = 100) -> Dict[str, float]:
        """运行速度基准测试"""
        results = {}

        for size, text in self.benchmark_texts.items():
            # 编码速度测试
            encode_times = []
            for _ in range(iterations):
                start = time.perf_counter()
                try:
                    tokens = tokenizer.encode(text) if hasattr(tokenizer, 'encode') else tokenizer.tokenize(text)
                    end = time.perf_counter()
                    encode_times.append(end - start)
                except Exception as e:
                    print(f"编码测试失败: {e}")
                    break

            # 解码速度测试 (如果支持)
            decode_times = []
            if hasattr(tokenizer, 'decode') and encode_times:
                try:
                    tokens = tokenizer.encode(text)
                    for _ in range(min(iterations, 20)):  # 解码测试次数较少
                        start = time.perf_counter()
                        decoded = tokenizer.decode(tokens)
                        end = time.perf_counter()
                        decode_times.append(end - start)
                except:
                    pass

            results[size] = {
                'encode_avg_ms': statistics.mean(encode_times) * 1000 if encode_times else 0,
                'encode_std_ms': statistics.stdev(encode_times) * 1000 if len(encode_times) > 1 else 0,
                'decode_avg_ms': statistics.mean(decode_times) * 1000 if decode_times else 0,
                'text_length': len(text),
                'throughput_chars_per_sec': len(text) / statistics.mean(encode_times) if encode_times else 0
            }

        return results

    def run_memory_benchmark(self, tokenizer) -> Dict[str, float]:
        """运行内存使用基准测试"""
        process = psutil.Process()

        # 基线内存
        baseline_memory = process.memory_info().rss / 1024 / 1024  # MB

        memory_results = {}

        for size, text in self.benchmark_texts.items():
            try:
                # 执行分词操作
                tokens = tokenizer.encode(text) if hasattr(tokenizer, 'encode') else tokenizer.tokenize(text)

                # 测量内存使用
                current_memory = process.memory_info().rss / 1024 / 1024
                memory_delta = current_memory - baseline_memory

                memory_results[size] = {
                    'memory_usage_mb': memory_delta,
                    'tokens_count': len(tokens) if tokens else 0,
                    'memory_per_token_kb': (memory_delta * 1024) / max(len(tokens), 1) if tokens else 0
                }

            except Exception as e:
                print(f"内存测试失败 {size}: {e}")

        return memory_results

    def run_concurrent_benchmark(self, tokenizer, num_threads: int = 4) -> Dict[str, float]:
        """运行并发性能测试"""
        test_text = self.benchmark_texts['medium']

        def tokenize_text():
            try:
                start = time.perf_counter()
                tokens = tokenizer.encode(test_text) if hasattr(tokenizer, 'encode') else tokenizer.tokenize(test_text)
                end = time.perf_counter()
                return end - start, len(tokens) if tokens else 0
            except:
                return 0, 0

        # 单线程基准
        single_time, token_count = tokenize_text()

        # 多线程测试
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            start_concurrent = time.perf_counter()
            futures = [executor.submit(tokenize_text) for _ in range(num_threads)]
            results = [f.result() for f in futures]
            end_concurrent = time.perf_counter()

        concurrent_times = [r[0] for r in results if r[0] > 0]

        return {
            'single_thread_ms': single_time * 1000,
            'concurrent_total_ms': (end_concurrent - start_concurrent) * 1000,
            'concurrent_avg_ms': statistics.mean(concurrent_times) * 1000 if concurrent_times else 0,
            'speedup_factor': (single_time * num_threads) / (end_concurrent - start_concurrent) if (end_concurrent - start_concurrent) > 0 else 0,
            'thread_count': num_threads
        }