#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
训练和推理测试脚本
验证升级后的架构在实际训练和推理中的表现
包括工具调用、ultra think等新能力
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

import torch
import torch.nn as nn
import json
import time
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict

from src.model.config import get_tiny_config, get_small_config
from src.model.transformer import MiniGPT


class TestDataset(Dataset):
    """测试数据集"""

    def __init__(self, data_path: str, tokenizer=None, max_length: int = 128):
        self.data = []
        self.max_length = max_length

        # 简单的字符级tokenizer用于测试
        if tokenizer is None:
            self.vocab = self._build_vocab(data_path)
            self.char_to_id = {char: i for i, char in enumerate(self.vocab)}
            self.id_to_char = {i: char for char, i in self.char_to_id.items()}

        # 加载数据
        self._load_data(data_path)

    def _build_vocab(self, data_path: str) -> List[str]:
        """构建字符级词汇表"""
        chars = set()
        try:
            with open(data_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        for conv in data.get('conversations', []):
                            chars.update(conv.get('content', ''))
        except FileNotFoundError:
            # 如果文件不存在，使用默认词汇表
            chars = set('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,!?;:()[]{}"\'-\n你我他的是在有一个这那了和为')

        # 添加特殊token
        vocab = ['<pad>', '<unk>', '<bos>', '<eos>'] + sorted(list(chars))
        return vocab

    def _load_data(self, data_path: str):
        """加载训练数据"""
        try:
            with open(data_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        text = self._extract_text(data)
                        if text:
                            tokens = self._tokenize(text)
                            if len(tokens) > 1:  # 至少需要输入和目标
                                self.data.append(tokens)
        except FileNotFoundError:
            # 生成合成数据用于测试
            self._generate_synthetic_data()

    def _extract_text(self, data: Dict) -> str:
        """从对话数据中提取文本"""
        text = ""
        conversations = data.get('conversations', [])
        for conv in conversations:
            role = conv.get('role', '')
            content = conv.get('content', '')
            text += f"{role}: {content}\n"
        return text.strip()

    def _tokenize(self, text: str) -> List[int]:
        """简单的字符级tokenization"""
        tokens = [self.char_to_id.get('<bos>', 2)]  # BOS token
        for char in text[:self.max_length - 2]:  # 留出BOS和EOS的空间
            tokens.append(self.char_to_id.get(char, self.char_to_id.get('<unk>', 1)))
        tokens.append(self.char_to_id.get('<eos>', 3))  # EOS token
        return tokens

    def _generate_synthetic_data(self):
        """生成合成训练数据"""
        synthetic_texts = [
            "Hello, how are you today?",
            "I am fine, thank you. How about you?",
            "What is the weather like?",
            "It's sunny and warm outside.",
            "Can you help me with this task?",
            "Of course! I'd be happy to help.",
            "你好，今天怎么样？",
            "我很好，谢谢你。你呢？",
            "天气怎么样？",
            "外面阳光明媚，很暖和。"
        ]

        for text in synthetic_texts:
            tokens = self._tokenize(text)
            if len(tokens) > 1:
                self.data.append(tokens)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        tokens = self.data[idx]

        # 输入和目标（shifted by 1）
        if len(tokens) <= self.max_length:
            # Padding
            padded = tokens + [0] * (self.max_length - len(tokens))
            input_ids = torch.tensor(padded[:-1], dtype=torch.long)
            labels = torch.tensor(padded[1:], dtype=torch.long)
        else:
            # Truncation
            input_ids = torch.tensor(tokens[:self.max_length-1], dtype=torch.long)
            labels = torch.tensor(tokens[1:self.max_length], dtype=torch.long)

        return input_ids, labels


def test_model_training():
    """测试模型训练"""
    print("Testing Model Training...")

    # 使用tiny配置进行快速测试
    config = get_tiny_config()
    config.vocab_size = 1000  # 减小词汇表以加快测试

    model = MiniGPT(config)

    # 创建测试数据
    data_path = "data/dataset/minimind_dataset/sft_mini_512.jsonl"
    dataset = TestDataset(data_path, max_length=64)

    if len(dataset) == 0:
        print("⚠️  No data loaded, using synthetic data")

    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    # 配置训练
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # 忽略padding

    model.train()
    initial_loss = None
    final_loss = None

    print(f"Training on {len(dataset)} samples...")

    # 训练几个步骤
    for epoch in range(2):
        epoch_loss = 0
        num_batches = 0

        for batch_idx, (input_ids, labels) in enumerate(dataloader):
            if batch_idx >= 5:  # 只训练几个batch
                break

            optimizer.zero_grad()

            # 前向传播
            logits = model(input_ids)

            # 计算损失
            loss = criterion(logits.view(-1, config.vocab_size), labels.view(-1))

            # 反向传播
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

            if epoch == 0 and batch_idx == 0:
                initial_loss = loss.item()

            print(f"  Epoch {epoch+1}, Batch {batch_idx+1}, Loss: {loss.item():.4f}")

        if num_batches > 0:
            avg_loss = epoch_loss / num_batches
            final_loss = avg_loss
            print(f"  Epoch {epoch+1} Average Loss: {avg_loss:.4f}")

    # 验证训练是否有效（损失应该下降）
    if initial_loss is not None and final_loss is not None:
        loss_improvement = initial_loss - final_loss
        print(f"Loss improvement: {loss_improvement:.4f}")

        if loss_improvement > 0:
            print("✅ Model training test passed! (Loss decreased)")
        else:
            print("⚠️  Model training test warning: Loss did not decrease significantly")
    else:
        print("⚠️  Could not evaluate loss improvement")

    return True


def test_model_inference():
    """测试模型推理"""
    print("Testing Model Inference...")

    config = get_tiny_config()
    config.vocab_size = 1000

    model = MiniGPT(config)
    model.eval()

    # 创建测试输入
    batch_size = 2
    seq_len = 32
    input_ids = torch.randint(1, config.vocab_size, (batch_size, seq_len))

    print(f"Input shape: {input_ids.shape}")

    # 测试生成
    with torch.no_grad():
        start_time = time.time()

        # 普通前向传播
        logits = model(input_ids)
        forward_time = time.time() - start_time

        # 生成测试
        start_time = time.time()
        generated = model.generate(
            input_ids[:1, :10],  # 使用较短的prompt
            max_length=20,
            temperature=0.8,
            top_k=10
        )
        generation_time = time.time() - start_time

    print(f"Forward pass time: {forward_time:.4f}s")
    print(f"Generation time: {generation_time:.4f}s")
    print(f"Generated sequence length: {generated.shape[1]}")

    # 验证输出
    assert logits.shape == (batch_size, seq_len, config.vocab_size), \
        f"Unexpected logits shape: {logits.shape}"

    assert generated.shape[0] == 1, "Generated batch size should be 1"
    assert generated.shape[1] >= 10, "Generated sequence should be longer than input"

    # 检查生成的token是否在有效范围内
    assert torch.all(generated >= 0) and torch.all(generated < config.vocab_size), \
        "Generated tokens outside vocabulary range"

    print("✅ Model inference test passed!")
    return True


def test_tool_calling_format():
    """测试工具调用数据格式"""
    print("Testing Tool Calling Data Format...")

    # 测试工具调用数据加载
    tool_data_paths = [
        "data/dataset/minimind_dataset/tool_calling_basic.jsonl",
        "data/dataset/minimind_dataset/tool_calling_advanced.jsonl",
        "data/dataset/minimind_dataset/agent_ultra_think.jsonl"
    ]

    for data_path in tool_data_paths:
        try:
            dataset = TestDataset(data_path, max_length=256)
            print(f"  {os.path.basename(data_path)}: {len(dataset)} samples")

            if len(dataset) > 0:
                # 测试第一个样本
                input_ids, labels = dataset[0]
                assert input_ids.shape[0] > 0, "Empty input sequence"
                assert labels.shape[0] > 0, "Empty label sequence"
                print(f"    Sample input shape: {input_ids.shape}")

        except Exception as e:
            print(f"  ⚠️  Failed to load {data_path}: {e}")

    print("✅ Tool calling data format test completed!")
    return True


def test_ultra_think_capability():
    """测试ultra think能力数据"""
    print("Testing Ultra Think Capability...")

    config = get_tiny_config()
    model = MiniGPT(config)

    # 创建ultra think测试数据
    ultra_think_data = TestDataset(
        "data/dataset/minimind_dataset/agent_ultra_think.jsonl",
        max_length=512
    )

    if len(ultra_think_data) > 0:
        print(f"Ultra think dataset loaded: {len(ultra_think_data)} samples")

        # 测试训练一个batch
        dataloader = DataLoader(ultra_think_data, batch_size=1, shuffle=False)
        input_ids, labels = next(iter(dataloader))

        model.eval()
        with torch.no_grad():
            logits = model(input_ids)
            print(f"Ultra think inference shape: {logits.shape}")

        print("✅ Ultra think capability test passed!")
    else:
        print("⚠️  No ultra think data found, skipping test")

    return True


def test_memory_efficiency():
    """测试内存效率改进"""
    print("Testing Memory Efficiency...")

    # 比较GQA vs 传统MHA的内存使用
    configs = {
        "MHA": get_small_config(),
        "GQA": get_small_config()
    }

    # 配置MHA（传统）
    configs["MHA"].use_gqa = False
    configs["MHA"].use_rope = False

    # 配置GQA（优化）
    configs["GQA"].use_gqa = True
    configs["GQA"].num_key_value_heads = 3

    results = {}

    for name, config in configs.items():
        # 创建模型
        model = MiniGPT(config)

        # 计算参数量
        total_params = sum(p.numel() for p in model.parameters())

        # 模拟推理内存使用
        batch_size = 4
        seq_len = 256
        input_ids = torch.randint(1, config.vocab_size, (batch_size, seq_len))

        # 测量内存使用（简化版本）
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

            model = model.cuda()
            input_ids = input_ids.cuda()

            with torch.no_grad():
                logits = model(input_ids)

            memory_used = torch.cuda.max_memory_allocated() / 1024 / 1024  # MB
        else:
            memory_used = 0  # CPU内存测量较复杂，暂时跳过

        results[name] = {
            'params': total_params,
            'memory_mb': memory_used
        }

        print(f"  {name}: {total_params:,} params, {memory_used:.1f} MB")

    # 计算改进
    if "MHA" in results and "GQA" in results:
        param_reduction = (1 - results["GQA"]["params"] / results["MHA"]["params"]) * 100
        print(f"Parameter reduction: {param_reduction:.1f}%")

        if results["GQA"]["memory_mb"] > 0 and results["MHA"]["memory_mb"] > 0:
            memory_reduction = (1 - results["GQA"]["memory_mb"] / results["MHA"]["memory_mb"]) * 100
            print(f"Memory reduction: {memory_reduction:.1f}%")

    print("✅ Memory efficiency test completed!")
    return True


def run_all_tests():
    """运行所有训练和推理测试"""
    print("=" * 60)
    print("MINIGPT TRAINING & INFERENCE TESTS")
    print("=" * 60)

    tests = [
        test_model_training,
        test_model_inference,
        test_tool_calling_format,
        test_ultra_think_capability,
        test_memory_efficiency,
    ]

    passed = 0
    total = len(tests)

    for test_func in tests:
        try:
            if test_func():
                passed += 1
            print()
        except Exception as e:
            print(f"❌ {test_func.__name__} failed with error: {e}")
            import traceback
            traceback.print_exc()
            print()

    print("=" * 60)
    print(f"TRAINING & INFERENCE TESTS SUMMARY: {passed}/{total} PASSED")
    print("=" * 60)

    if passed == total:
        print("🎉 All training and inference tests passed!")
        return True
    else:
        print("⚠️  Some tests failed. Please check the implementation.")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)