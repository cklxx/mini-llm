"""
数据加载模块
支持多种数据格式的加载和预处理
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Optional

from . import load_tokenizer
from src.tokenizer.bpe_tokenizer import BPETokenizer


@dataclass
class DatasetConfig:
    """数据集配置类"""
    data_path: str
    max_length: int = 512
    train_split: float = 0.9
    shuffle: bool = True
    tokenizer_path: Optional[str] = None


class ConversationDataLoader:
    """对话数据加载器

    支持加载SFT格式的对话数据，包括：
    - sft_mini_512.jsonl：极简SFT数据
    - sft_512.jsonl：匠数科技SFT数据
    - sft_1024.jsonl：Qwen2.5蒸馏数据
    - sft_2048.jsonl：完整Qwen2.5蒸馏数据
    """

    def __init__(
        self,
        config: DatasetConfig,
        tokenizer: Optional[BPETokenizer] = None,
    ):
        self.config = config
        self.tokenizer = tokenizer or self._load_default_tokenizer()
        self.data = []

    def load_jsonl(self, file_path: str) -> list[dict]:
        """加载JSONL格式文件"""
        data = []
        with open(file_path, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        data.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        print(f"解析JSON行失败: {e}")
                        continue
        return data

    def load_conversations(self) -> list[dict]:
        """加载对话数据"""
        if not os.path.exists(self.config.data_path):
            raise FileNotFoundError(f"数据文件不存在: {self.config.data_path}")

        raw_data = self.load_jsonl(self.config.data_path)
        processed_data = []

        for item in raw_data:
            if 'conversations' in item:
                # 处理对话格式数据
                conversations = item['conversations']
                if len(conversations) >= 2:
                    # 提取用户输入和助手回复
                    user_input = ""
                    assistant_output = ""

                    for conv in conversations:
                        if conv['role'] == 'user':
                            user_input = conv['content']
                        elif conv['role'] == 'assistant':
                            assistant_output = conv['content']

                    if user_input and assistant_output:
                        char_length = len(user_input) + len(assistant_output)
                        token_length = self._conversation_token_length(
                            user_input, assistant_output
                        )
                        length_value = token_length if token_length is not None else char_length

                        processed_data.append({
                            'input': user_input,
                            'output': assistant_output,
                            'length': length_value,
                            'char_length': char_length,
                            'token_length': token_length
                        })

        # 按长度过滤
        filtered_data = [
            item for item in processed_data
            if item['length'] <= self.config.max_length
        ]

        removed = len(processed_data) - len(filtered_data)
        metric = "token" if self.tokenizer else "字符"
        if removed > 0:
            print(
                f"加载了 {len(filtered_data)} 条对话数据 (按 {metric} 长度过滤，"
                f"丢弃 {removed} 条超出 {self.config.max_length})"
            )
        else:
            print(
                f"加载了 {len(filtered_data)} 条对话数据 (按 {metric} 长度过滤)"
            )
        return filtered_data

    def get_train_test_split(self, data: list[dict]):
        """划分训练集和测试集"""
        if self.config.shuffle:
            import random
            random.shuffle(data)

        split_idx = int(len(data) * self.config.train_split)
        train_data = data[:split_idx]
        test_data = data[split_idx:]

        return train_data, test_data


class PretrainDataLoader:
    """预训练数据加载器

    支持加载预训练格式的数据：
    - pretrain_hq.jsonl：高质量预训练数据
    """

    def __init__(
        self,
        config: DatasetConfig,
        tokenizer: Optional[BPETokenizer] = None,
    ):
        self.config = config
        self.tokenizer = tokenizer or self._load_default_tokenizer()

    def load_pretrain_data(self) -> list[str]:
        """加载预训练数据"""
        if not os.path.exists(self.config.data_path):
            raise FileNotFoundError(f"数据文件不存在: {self.config.data_path}")

        texts = []
        with open(self.config.data_path, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        data = json.loads(line)
                        if 'text' in data:
                            text = data['text']
                            if self._within_length(text):
                                texts.append(text)
                    except json.JSONDecodeError:
                        continue

        metric = "token" if self.tokenizer else "字符"
        print(f"加载了 {len(texts)} 条预训练文本 (按 {metric} 长度过滤)")
        return texts

    def _load_default_tokenizer(self) -> Optional[BPETokenizer]:
        try:
            return load_tokenizer(self.config.tokenizer_path)
        except Exception as exc:
            print(f"⚠️  无法加载分词器资源，回退到字符长度过滤: {exc}")
            return None

    def _within_length(self, text: str) -> bool:
        if not self.tokenizer:
            return len(text) <= self.config.max_length

        try:
            token_ids = self.tokenizer.encode(text, add_special_tokens=True)
        except Exception as exc:  # pragma: no cover - 安全回退
            print(f"⚠️  分词器编码失败，使用字符长度: {exc}")
            return len(text) <= self.config.max_length

        return len(token_ids) <= self.config.max_length


class DPODataLoader:
    """DPO数据加载器

    支持加载DPO格式的数据：
    - dpo.jsonl：RLHF阶段数据
    """

    def __init__(self, config: DatasetConfig):
        self.config = config

    def load_dpo_data(self) -> list[dict]:
        """加载DPO数据"""
        if not os.path.exists(self.config.data_path):
            raise FileNotFoundError(f"数据文件不存在: {self.config.data_path}")

        dpo_data = []
        with open(self.config.data_path, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        data = json.loads(line)
                        # DPO数据通常包含prompt, chosen, rejected字段
                        if all(k in data for k in ['prompt', 'chosen', 'rejected']):
                            dpo_data.append(data)
                    except json.JSONDecodeError:
                        continue

        print(f"加载了 {len(dpo_data)} 条DPO数据")
        return dpo_data


def create_data_loader(data_type: str, config: DatasetConfig):
    """创建数据加载器工厂函数"""
    if data_type == 'sft':
        return ConversationDataLoader(config)
    elif data_type == 'pretrain':
        return PretrainDataLoader(config)
    elif data_type == 'dpo':
        return DPODataLoader(config)
    else:
        raise ValueError(f"不支持的数据类型: {data_type}")


if __name__ == "__main__":
    # 测试数据加载
    config = DatasetConfig(
        data_path="data/dataset/minimind_dataset/sft_mini_512.jsonl",
        max_length=512
    )

    loader = ConversationDataLoader(config)
    data = loader.load_conversations()
    train_data, test_data = loader.get_train_test_split(data)

    print(f"训练集大小: {len(train_data)}")
    print(f"测试集大小: {len(test_data)}")
    print(f"样本示例: {train_data[0]}")
    def _load_default_tokenizer(self) -> Optional[BPETokenizer]:
        try:
            return load_tokenizer(self.config.tokenizer_path)
        except Exception as exc:
            print(f"⚠️  无法加载分词器资源，回退到字符长度过滤: {exc}")
            return None

    def _conversation_token_length(self, user_input: str, assistant_output: str) -> Optional[int]:
        if not self.tokenizer:
            return None

        try:
            input_ids = self.tokenizer.encode(user_input, add_special_tokens=False)
            output_ids = self.tokenizer.encode(assistant_output, add_special_tokens=False)
        except Exception as exc:  # pragma: no cover - 安全回退
            print(f"⚠️  分词器编码失败，使用字符长度: {exc}")
            return None

        # 训练时会附加 BOS/EOS，因此额外加 2
        return len(input_ids) + len(output_ids) + 2
