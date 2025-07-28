"""
数据加载模块
支持多种数据格式的加载和预处理
"""
import json
import os
from typing import List, Dict, Any, Optional, Iterator
from dataclasses import dataclass


@dataclass
class DatasetConfig:
    """数据集配置类"""
    data_path: str
    max_length: int = 512
    train_split: float = 0.9
    shuffle: bool = True
    

class ConversationDataLoader:
    """对话数据加载器
    
    支持加载SFT格式的对话数据，包括：
    - sft_mini_512.jsonl：极简SFT数据
    - sft_512.jsonl：匠数科技SFT数据
    - sft_1024.jsonl：Qwen2.5蒸馏数据
    - sft_2048.jsonl：完整Qwen2.5蒸馏数据
    """
    
    def __init__(self, config: DatasetConfig):
        self.config = config
        self.data = []
        
    def load_jsonl(self, file_path: str) -> List[Dict]:
        """加载JSONL格式文件"""
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        data.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        print(f"解析JSON行失败: {e}")
                        continue
        return data
    
    def load_conversations(self) -> List[Dict]:
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
                        processed_data.append({
                            'input': user_input,
                            'output': assistant_output,
                            'length': len(user_input) + len(assistant_output)
                        })
        
        # 按长度过滤
        filtered_data = [
            item for item in processed_data 
            if item['length'] <= self.config.max_length
        ]
        
        print(f"加载了 {len(filtered_data)} 条对话数据")
        return filtered_data
    
    def get_train_test_split(self, data: List[Dict]):
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
    
    def __init__(self, config: DatasetConfig):
        self.config = config
        
    def load_pretrain_data(self) -> List[str]:
        """加载预训练数据"""
        if not os.path.exists(self.config.data_path):
            raise FileNotFoundError(f"数据文件不存在: {self.config.data_path}")
        
        texts = []
        with open(self.config.data_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        data = json.loads(line)
                        if 'text' in data:
                            text = data['text']
                            if len(text) <= self.config.max_length:
                                texts.append(text)
                    except json.JSONDecodeError:
                        continue
        
        print(f"加载了 {len(texts)} 条预训练文本")
        return texts


class DPODataLoader:
    """DPO数据加载器
    
    支持加载DPO格式的数据：
    - dpo.jsonl：RLHF阶段数据
    """
    
    def __init__(self, config: DatasetConfig):
        self.config = config
        
    def load_dpo_data(self) -> List[Dict]:
        """加载DPO数据"""
        if not os.path.exists(self.config.data_path):
            raise FileNotFoundError(f"数据文件不存在: {self.config.data_path}")
        
        dpo_data = []
        with open(self.config.data_path, 'r', encoding='utf-8') as f:
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