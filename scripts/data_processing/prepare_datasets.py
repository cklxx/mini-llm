#!/usr/bin/env python3
"""
数据预处理脚本
准备和优化训练数据集，包括工具调用、ultra think等数据
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

import json
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime
import random


def setup_logging():
    """设置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


class DatasetProcessor:
    """数据集处理器"""

    def __init__(self, logger):
        self.logger = logger

    def load_jsonl(self, file_path: str) -> List[Dict]:
        """加载JSONL文件"""
        data = []
        if not os.path.exists(file_path):
            self.logger.warning(f"File not found: {file_path}")
            return data

        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                if line.strip():
                    try:
                        data.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        self.logger.warning(f"Error parsing line {line_num} in {file_path}: {e}")

        self.logger.info(f"Loaded {len(data)} samples from {file_path}")
        return data

    def save_jsonl(self, data: List[Dict], file_path: str):
        """保存JSONL文件"""
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)

        with open(file_path, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')

        self.logger.info(f"Saved {len(data)} samples to {file_path}")

    def validate_conversation_format(self, data: Dict) -> bool:
        """验证对话格式"""
        if 'conversations' not in data:
            return False

        conversations = data['conversations']
        if not isinstance(conversations, list) or len(conversations) == 0:
            return False

        for conv in conversations:
            if not isinstance(conv, dict):
                return False
            if 'role' not in conv or 'content' not in conv:
                return False
            if conv['role'] not in ['user', 'assistant', 'system']:
                return False

        return True

    def filter_by_length(self, data: List[Dict], min_length: int = 10, max_length: int = 2048) -> List[Dict]:
        """按长度过滤数据"""
        filtered = []

        for item in data:
            if not self.validate_conversation_format(item):
                continue

            total_length = 0
            for conv in item['conversations']:
                total_length += len(conv['content'])

            if min_length <= total_length <= max_length:
                filtered.append(item)

        self.logger.info(f"Filtered by length: {len(data)} -> {len(filtered)} samples")
        return filtered

    def deduplicate(self, data: List[Dict]) -> List[Dict]:
        """去重"""
        seen = set()
        deduplicated = []

        for item in data:
            if not self.validate_conversation_format(item):
                continue

            # 创建内容hash
            content_hash = hash(str(item['conversations']))

            if content_hash not in seen:
                seen.add(content_hash)
                deduplicated.append(item)

        self.logger.info(f"Deduplicated: {len(data)} -> {len(deduplicated)} samples")
        return deduplicated

    def enhance_tool_calling_data(self, data: List[Dict]) -> List[Dict]:
        """增强工具调用数据"""
        enhanced = []

        for item in data:
            if not self.validate_conversation_format(item):
                continue

            # 检查是否包含工具调用
            has_tool_calls = any(
                'tool_calls' in conv or 'tools' in conv
                for conv in item['conversations']
            )

            if has_tool_calls:
                # 增强工具调用格式
                enhanced_item = self._enhance_tool_calling_format(item)
                enhanced.append(enhanced_item)
            else:
                enhanced.append(item)

        self.logger.info(f"Enhanced tool calling data: {len(enhanced)} samples")
        return enhanced

    def _enhance_tool_calling_format(self, item: Dict) -> Dict:
        """增强工具调用格式"""
        enhanced_conversations = []

        for conv in item['conversations']:
            enhanced_conv = conv.copy()

            # 确保工具调用格式正确
            if 'tool_calls' in conv:
                tool_calls = conv['tool_calls']
                if isinstance(tool_calls, list):
                    for tool_call in tool_calls:
                        if 'function' in tool_call:
                            # 确保function有name和arguments
                            func = tool_call['function']
                            if 'name' not in func:
                                func['name'] = 'unknown_function'
                            if 'arguments' not in func:
                                func['arguments'] = '{}'

            enhanced_conversations.append(enhanced_conv)

        return {'conversations': enhanced_conversations}

    def create_mixed_dataset(self, datasets: Dict[str, List[Dict]],
                           output_path: str, target_size: int = 10000) -> List[Dict]:
        """创建混合数据集"""
        mixed_data = []

        # 计算每个数据集的采样比例
        total_available = sum(len(data) for data in datasets.values())

        for name, data in datasets.items():
            if len(data) == 0:
                continue

            # 计算采样数量
            ratio = len(data) / total_available
            sample_size = min(int(target_size * ratio), len(data))

            # 随机采样
            sampled = random.sample(data, sample_size)
            mixed_data.extend(sampled)

            self.logger.info(f"Sampled {sample_size} from {name} ({len(data)} total)")

        # 随机打乱
        random.shuffle(mixed_data)

        # 保存
        self.save_jsonl(mixed_data, output_path)

        return mixed_data

    def split_dataset(self, data: List[Dict], train_ratio: float = 0.8,
                     val_ratio: float = 0.1, test_ratio: float = 0.1) -> Dict[str, List[Dict]]:
        """分割数据集"""
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "比例之和必须为1"

        # 随机打乱
        data = data.copy()
        random.shuffle(data)

        total_size = len(data)
        train_size = int(total_size * train_ratio)
        val_size = int(total_size * val_ratio)

        splits = {
            'train': data[:train_size],
            'val': data[train_size:train_size + val_size],
            'test': data[train_size + val_size:]
        }

        for split_name, split_data in splits.items():
            self.logger.info(f"{split_name}: {len(split_data)} samples")

        return splits

    def analyze_dataset(self, data: List[Dict]) -> Dict[str, Any]:
        """分析数据集统计信息"""
        if not data:
            return {}

        stats = {
            'total_samples': len(data),
            'conversation_lengths': [],
            'content_lengths': [],
            'tool_calling_samples': 0,
            'ultra_think_samples': 0,
            'role_distribution': {'user': 0, 'assistant': 0, 'system': 0}
        }

        for item in data:
            if not self.validate_conversation_format(item):
                continue

            conversations = item['conversations']
            stats['conversation_lengths'].append(len(conversations))

            has_tool_calls = False
            has_ultra_think = False

            for conv in conversations:
                content = conv['content']
                stats['content_lengths'].append(len(content))

                role = conv['role']
                if role in stats['role_distribution']:
                    stats['role_distribution'][role] += 1

                # 检查特殊内容
                if 'tool_calls' in conv or 'tools' in conv:
                    has_tool_calls = True

                if 'ultra_think' in content.lower() or '<ultra_think>' in content:
                    has_ultra_think = True

            if has_tool_calls:
                stats['tool_calling_samples'] += 1
            if has_ultra_think:
                stats['ultra_think_samples'] += 1

        # 计算统计量
        if stats['conversation_lengths']:
            stats['avg_conversation_length'] = sum(stats['conversation_lengths']) / len(stats['conversation_lengths'])
            stats['max_conversation_length'] = max(stats['conversation_lengths'])

        if stats['content_lengths']:
            stats['avg_content_length'] = sum(stats['content_lengths']) / len(stats['content_lengths'])
            stats['max_content_length'] = max(stats['content_lengths'])

        return stats


def main():
    parser = argparse.ArgumentParser(description='数据预处理脚本')
    parser.add_argument('--input-dir', type=str,
                       default='data/dataset/minimind_dataset',
                       help='输入数据目录')
    parser.add_argument('--output-dir', type=str,
                       default='data/processed',
                       help='输出数据目录')
    parser.add_argument('--target-size', type=int, default=10000,
                       help='目标数据集大小')
    parser.add_argument('--max-length', type=int, default=1024,
                       help='最大内容长度')
    parser.add_argument('--train-ratio', type=float, default=0.8,
                       help='训练集比例')
    parser.add_argument('--val-ratio', type=float, default=0.1,
                       help='验证集比例')
    parser.add_argument('--test-ratio', type=float, default=0.1,
                       help='测试集比例')

    args = parser.parse_args()

    logger = setup_logging()
    processor = DatasetProcessor(logger)

    logger.info("Starting dataset preparation...")

    # 创建输出目录
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # 定义数据文件
    dataset_files = {
        'sft_base': f"{args.input_dir}/sft_mini_512.jsonl",
        'tool_calling_basic': f"{args.input_dir}/tool_calling_basic.jsonl",
        'tool_calling_advanced': f"{args.input_dir}/tool_calling_advanced.jsonl",
        'ultra_think': f"{args.input_dir}/agent_ultra_think.jsonl",
        'alex_identity': f"{args.input_dir}/alex_identity.jsonl"
    }

    # 加载和处理各个数据集
    datasets = {}
    for name, file_path in dataset_files.items():
        logger.info(f"Processing {name}...")

        # 加载数据
        data = processor.load_jsonl(file_path)

        if data:
            # 过滤和清理
            data = processor.filter_by_length(data, max_length=args.max_length)
            data = processor.deduplicate(data)

            # 特殊处理
            if 'tool_calling' in name:
                data = processor.enhance_tool_calling_data(data)

            # 分析数据
            stats = processor.analyze_dataset(data)
            logger.info(f"{name} stats: {stats}")

            datasets[name] = data

    # 创建混合数据集
    logger.info("Creating mixed dataset...")
    mixed_data = processor.create_mixed_dataset(
        datasets,
        f"{args.output_dir}/mixed_dataset.jsonl",
        args.target_size
    )

    # 分割数据集
    logger.info("Splitting dataset...")
    splits = processor.split_dataset(
        mixed_data,
        args.train_ratio,
        args.val_ratio,
        args.test_ratio
    )

    # 保存分割后的数据
    for split_name, split_data in splits.items():
        output_path = f"{args.output_dir}/{split_name}.jsonl"
        processor.save_jsonl(split_data, output_path)

    # 生成数据集报告
    total_stats = processor.analyze_dataset(mixed_data)

    report = {
        "processing_time": datetime.now().isoformat(),
        "total_samples": len(mixed_data),
        "splits": {name: len(data) for name, data in splits.items()},
        "source_datasets": {name: len(data) for name, data in datasets.items()},
        "statistics": total_stats,
        "parameters": vars(args)
    }

    report_path = f"{args.output_dir}/dataset_report.json"
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    logger.info(f"Dataset preparation completed!")
    logger.info(f"Report saved to: {report_path}")


if __name__ == "__main__":
    main()