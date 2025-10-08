"""
偏好数据处理模块

处理人类偏好数据，为奖励模型训练做准备。
偏好数据通常包含提示(prompt)和多个回复，以及人类的偏好排序。

数据格式：
1. 简单格式：{prompt, chosen, rejected}
2. 复杂格式：{prompt, responses: [r1, r2, ...], rankings: [1, 2, ...]}
3. 对话格式：多轮对话的偏好数据

处理步骤：
1. 数据加载和验证
2. 文本预处理和分词
3. 数据增强和平衡
4. 批次构建和填充
"""

import json
import random
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np


@dataclass
class PreferenceExample:
    """偏好数据样例"""
    prompt: str
    chosen: str
    rejected: str
    chosen_score: Optional[float] = None
    rejected_score: Optional[float] = None
    metadata: Optional[Dict] = None


@dataclass
class MultiPreferenceExample:
    """多候选偏好数据样例"""
    prompt: str
    responses: List[str]
    rankings: List[int]  # 排序，越小越好
    scores: Optional[List[float]] = None
    metadata: Optional[Dict] = None


class PreferenceDataProcessor:
    """偏好数据处理器"""
    
    def __init__(self, tokenizer, max_length: int = 512):
        """
        初始化数据处理器
        
        Args:
            tokenizer: 分词器
            max_length: 最大序列长度
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def load_preference_data(self, file_path: str) -> List[PreferenceExample]:
        """
        加载偏好数据
        
        Args:
            file_path: 数据文件路径
            
        Returns:
            偏好数据列表
        """
        examples = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line.strip())
                
                # 处理不同的数据格式
                if 'chosen' in data and 'rejected' in data:
                    # 简单格式
                    example = PreferenceExample(
                        prompt=data['prompt'],
                        chosen=data['chosen'],
                        rejected=data['rejected'],
                        chosen_score=data.get('chosen_score'),
                        rejected_score=data.get('rejected_score'),
                        metadata=data.get('metadata', {})
                    )
                    examples.append(example)
                
                elif 'responses' in data and 'rankings' in data:
                    # 复杂格式：转换为多个简单样例
                    multi_example = MultiPreferenceExample(
                        prompt=data['prompt'],
                        responses=data['responses'],
                        rankings=data['rankings'],
                        scores=data.get('scores'),
                        metadata=data.get('metadata', {})
                    )
                    
                    # 转换为简单格式
                    simple_examples = self._convert_multi_to_simple(multi_example)
                    examples.extend(simple_examples)
        
        return examples
    
    def _convert_multi_to_simple(self, multi_example: MultiPreferenceExample) -> List[PreferenceExample]:
        """
        将多候选样例转换为简单样例
        
        Args:
            multi_example: 多候选样例
            
        Returns:
            简单样例列表
        """
        examples = []
        responses = multi_example.responses
        rankings = multi_example.rankings
        scores = multi_example.scores or [0] * len(responses)
        
        # 生成所有可能的配对
        for i in range(len(responses)):
            for j in range(i + 1, len(responses)):
                # 确定哪个更好
                if rankings[i] < rankings[j]:
                    chosen, rejected = responses[i], responses[j]
                    chosen_score, rejected_score = scores[i], scores[j]
                elif rankings[i] > rankings[j]:
                    chosen, rejected = responses[j], responses[i]
                    chosen_score, rejected_score = scores[j], scores[i]
                else:
                    # 相等排名，随机选择或跳过
                    continue
                
                example = PreferenceExample(
                    prompt=multi_example.prompt,
                    chosen=chosen,
                    rejected=rejected,
                    chosen_score=chosen_score,
                    rejected_score=rejected_score,
                    metadata=multi_example.metadata
                )
                examples.append(example)
        
        return examples
    
    def preprocess_text(self, text: str) -> str:
        """
        预处理文本
        
        Args:
            text: 原始文本
            
        Returns:
            预处理后的文本
        """
        # 基本清理
        text = text.strip()
        
        # 可以添加更多预处理步骤
        # - 去除特殊字符
        # - 统一大小写
        # - 处理HTML标签等
        
        return text
    
    def tokenize_pair(self, prompt: str, response: str) -> Dict[str, torch.Tensor]:
        """
        对提示和回复进行分词
        
        Args:
            prompt: 提示文本
            response: 回复文本
            
        Returns:
            分词结果
        """
        # 预处理
        prompt = self.preprocess_text(prompt)
        response = self.preprocess_text(response)
        
        # 构建完整文本
        full_text = f"{prompt} {response}"
        
        # 分词
        tokens = self.tokenizer.encode(full_text, add_special_tokens=True)
        
        # 截断或填充
        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]
        
        # 创建注意力掩码
        attention_mask = [1] * len(tokens)
        
        # 填充到最大长度
        while len(tokens) < self.max_length:
            tokens.append(self.tokenizer.pad_id)
            attention_mask.append(0)
        
        return {
            'input_ids': torch.tensor(tokens, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long)
        }
    
    def augment_data(self, examples: List[PreferenceExample], 
                    augment_ratio: float = 0.2) -> List[PreferenceExample]:
        """
        数据增强
        
        Args:
            examples: 原始样例
            augment_ratio: 增强比例
            
        Returns:
            增强后的样例
        """
        augmented = []
        
        for example in examples:
            # 原始样例
            augmented.append(example)
            
            # 随机增强
            if random.random() < augment_ratio:
                # 交换chosen和rejected（生成负样例）
                swapped = PreferenceExample(
                    prompt=example.prompt,
                    chosen=example.rejected,
                    rejected=example.chosen,
                    chosen_score=example.rejected_score,
                    rejected_score=example.chosen_score,
                    metadata=example.metadata
                )
                augmented.append(swapped)
        
        return augmented
    
    def balance_data(self, examples: List[PreferenceExample]) -> List[PreferenceExample]:
        """
        平衡数据
        
        Args:
            examples: 原始样例
            
        Returns:
            平衡后的样例
        """
        # 统计不同类型的样例
        high_quality = []
        medium_quality = []
        low_quality = []
        
        for example in examples:
            if example.chosen_score and example.rejected_score:
                score_diff = example.chosen_score - example.rejected_score
                if score_diff > 1.0:
                    high_quality.append(example)
                elif score_diff > 0.5:
                    medium_quality.append(example)
                else:
                    low_quality.append(example)
            else:
                medium_quality.append(example)
        
        # 平衡不同质量的样例
        min_count = min(len(high_quality), len(medium_quality), len(low_quality))
        if min_count > 0:
            balanced = []
            balanced.extend(random.sample(high_quality, min(len(high_quality), min_count * 2)))
            balanced.extend(random.sample(medium_quality, min(len(medium_quality), min_count * 2)))
            balanced.extend(random.sample(low_quality, min(len(low_quality), min_count)))
            return balanced
        else:
            return examples


class PreferenceDataset(Dataset):
    """偏好数据集"""
    
    def __init__(self, examples: List[PreferenceExample], processor: PreferenceDataProcessor):
        """
        初始化数据集
        
        Args:
            examples: 偏好样例列表
            processor: 数据处理器
        """
        self.examples = examples
        self.processor = processor
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        
        # 分词chosen和rejected
        chosen_tokens = self.processor.tokenize_pair(example.prompt, example.chosen)
        rejected_tokens = self.processor.tokenize_pair(example.prompt, example.rejected)
        
        return {
            'chosen_input_ids': chosen_tokens['input_ids'],
            'chosen_attention_mask': chosen_tokens['attention_mask'],
            'rejected_input_ids': rejected_tokens['input_ids'],
            'rejected_attention_mask': rejected_tokens['attention_mask'],
            'chosen_score': example.chosen_score or 0.0,
            'rejected_score': example.rejected_score or 0.0
        }


class PreferenceCollator:
    """偏好数据批次整理器"""
    
    def __init__(self, pad_token_id: int = 0):
        """
        初始化整理器
        
        Args:
            pad_token_id: 填充token ID
        """
        self.pad_token_id = pad_token_id
    
    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        整理批次数据
        
        Args:
            batch: 批次样例列表
            
        Returns:
            整理后的批次数据
        """
        # 收集所有字段
        chosen_input_ids = [item['chosen_input_ids'] for item in batch]
        chosen_attention_mask = [item['chosen_attention_mask'] for item in batch]
        rejected_input_ids = [item['rejected_input_ids'] for item in batch]
        rejected_attention_mask = [item['rejected_attention_mask'] for item in batch]
        chosen_scores = [item['chosen_score'] for item in batch]
        rejected_scores = [item['rejected_score'] for item in batch]
        
        return {
            'chosen_input_ids': torch.stack(chosen_input_ids),
            'chosen_attention_mask': torch.stack(chosen_attention_mask),
            'rejected_input_ids': torch.stack(rejected_input_ids),
            'rejected_attention_mask': torch.stack(rejected_attention_mask),
            'chosen_scores': torch.tensor(chosen_scores, dtype=torch.float),
            'rejected_scores': torch.tensor(rejected_scores, dtype=torch.float)
        }


def create_preference_dataloader(data_file: str,
                               tokenizer,
                               batch_size: int = 32,
                               max_length: int = 512,
                               shuffle: bool = True,
                               augment_ratio: float = 0.0,
                               balance_data: bool = False) -> DataLoader:
    """
    创建偏好数据加载器
    
    Args:
        data_file: 数据文件路径
        tokenizer: 分词器
        batch_size: 批次大小
        max_length: 最大序列长度
        shuffle: 是否打乱
        augment_ratio: 数据增强比例
        balance_data: 是否平衡数据
        
    Returns:
        数据加载器
    """
    # 创建处理器
    processor = PreferenceDataProcessor(tokenizer, max_length)
    
    # 加载数据
    examples = processor.load_preference_data(data_file)
    
    # 数据增强
    if augment_ratio > 0:
        examples = processor.augment_data(examples, augment_ratio)
    
    # 数据平衡
    if balance_data:
        examples = processor.balance_data(examples)
    
    # 创建数据集
    dataset = PreferenceDataset(examples, processor)
    
    # 创建整理器
    collator = PreferenceCollator(tokenizer.pad_id)
    
    # 创建数据加载器
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collator,
        num_workers=0  # 可以根据需要调整
    )


def validate_preference_data(data_file: str) -> Dict[str, any]:
    """
    验证偏好数据质量
    
    Args:
        data_file: 数据文件路径
        
    Returns:
        验证结果统计
    """
    stats = {
        'total_examples': 0,
        'valid_examples': 0,
        'avg_prompt_length': 0,
        'avg_chosen_length': 0,
        'avg_rejected_length': 0,
        'score_distribution': {'with_scores': 0, 'without_scores': 0}
    }
    
    prompt_lengths = []
    chosen_lengths = []
    rejected_lengths = []
    
    with open(data_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                stats['total_examples'] += 1
                
                # 检查必要字段
                if 'prompt' in data and ('chosen' in data or 'responses' in data):
                    stats['valid_examples'] += 1
                    
                    # 统计长度
                    prompt_lengths.append(len(data['prompt']))
                    
                    if 'chosen' in data:
                        chosen_lengths.append(len(data['chosen']))
                        rejected_lengths.append(len(data['rejected']))
                        
                        # 统计分数
                        if 'chosen_score' in data and 'rejected_score' in data:
                            stats['score_distribution']['with_scores'] += 1
                        else:
                            stats['score_distribution']['without_scores'] += 1
                    
            except json.JSONDecodeError:
                continue
    
    # 计算平均长度
    if prompt_lengths:
        stats['avg_prompt_length'] = np.mean(prompt_lengths)
    if chosen_lengths:
        stats['avg_chosen_length'] = np.mean(chosen_lengths)
    if rejected_lengths:
        stats['avg_rejected_length'] = np.mean(rejected_lengths)
    
    return stats


if __name__ == "__main__":
    # 简单测试
    print("偏好数据处理模块实现完成")
    print("主要组件：")
    print("- PreferenceExample: 偏好数据样例")
    print("- PreferenceDataProcessor: 数据处理器")
    print("- PreferenceDataset: 偏好数据集")
    print("- PreferenceCollator: 批次整理器")
    print("- create_preference_dataloader: 数据加载器工厂函数")
    print("- validate_preference_data: 数据验证函数")