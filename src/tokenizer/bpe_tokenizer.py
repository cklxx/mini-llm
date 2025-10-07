"""
手写实现BPE (Byte Pair Encoding) Tokenizer
用于新手教学理解分词原理，优化了中文处理
"""
import json
import re
import time
import sys
import random
import os
from typing import List, Dict, Tuple, Optional
from collections import defaultdict, Counter
import pickle
from multiprocessing import Pool, cpu_count
from functools import partial


def is_chinese_char(char):
    """判断是否为中文字符"""
    return '\u4e00' <= char <= '\u9fff'


def print_progress_bar(current: int, total: int, prefix: str = '', suffix: str = '', 
                      decimals: int = 1, length: int = 40, fill: str = '█', 
                      empty: str = '░'):
    """打印动态进度条，会覆盖上一行"""
    percent = ("{0:." + str(decimals) + "f}").format(100 * (current / float(total)))
    filled_length = int(length * current // total)
    bar = fill * filled_length + empty * (length - filled_length)
    
    # 使用 \r 回到行首，覆盖上一行内容
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end='', flush=True)
    
    # 如果完成，换行
    if current == total:
        print()


def show_progress_with_stats(current: int, total: int, start_time: float, 
                           prefix: str = '', extra_info: str = ''):
    """统一的进度显示函数，包含速度和预计完成时间"""
    elapsed = time.time() - start_time
    speed = current / elapsed if elapsed > 0 else 0
    eta = (total - current) / speed if speed > 0 else 0
    
    suffix = f"({current:,}/{total:,}) {speed:.0f}/秒"
    if eta > 0:
        suffix += f" ETA: {eta:.0f}秒"
    if extra_info:
        suffix += f" | {extra_info}"
    
    print_progress_bar(current, total, prefix=prefix, suffix=suffix)


class BPETokenizer:
    """BPE分词器实现 - 优化了中文处理
    
    BPE算法核心思想：
    1. 将文本分割成字符
    2. 统计相邻字符对的频率
    3. 合并频率最高的字符对
    4. 重复步骤2-3直到达到词汇表大小
    """
    
    def __init__(self, vocab_size: int = 30000):  # 增加默认词汇表大小
        self.vocab_size = vocab_size
        self.word_freqs = {}  # 词频统计
        self.splits = {}      # 词的分割结果
        self.merges = {}      # 合并规则
        self.vocab = {}       # 词汇表
        
        # 特殊token
        self.pad_token = '<PAD>'
        self.unk_token = '<UNK>'
        self.bos_token = '<BOS>'  # Begin of sequence
        self.eos_token = '<EOS>'  # End of sequence
        
        # 特殊token的ID
        self.pad_id = 0
        self.unk_id = 1
        self.bos_id = 2
        self.eos_id = 3
        
    def pre_tokenize(self, text: str) -> List[str]:
        """预分词：改进中文处理"""
        # 针对中文优化的预分词规则
        words = []
        current_word = ""
        
        for char in text:
            if is_chinese_char(char):
                # 中文字符作为独立的词
                if current_word:
                    words.append(current_word.lower())
                    current_word = ""
                words.append(char)
            elif char.isalnum():
                # 英文/数字字符累积
                current_word += char
            elif char.isspace():
                # 空格分隔
                if current_word:
                    words.append(current_word.lower())
                    current_word = ""
            else:
                # 标点符号
                if current_word:
                    words.append(current_word.lower())
                    current_word = ""
                if char.strip():  # 非空白标点
                    words.append(char)
        
        # 处理最后的词
        if current_word:
            words.append(current_word.lower())
        
        return [w for w in words if w.strip()]  # 过滤空词
    
    def compute_word_frequencies(self, texts: List[str]):
        """计算词频"""
        word_freqs = defaultdict(int)
        
        print("正在计算词频...")
        start_time = time.time()
        total_texts = len(texts)
        
        for i, text in enumerate(texts):
            words = self.pre_tokenize(text)
            for word in words:
                word_freqs[word] += 1
            
            # 显示动态进度
            if (i + 1) % 1000 == 0 or i == total_texts - 1:
                show_progress_with_stats(i + 1, total_texts, start_time, prefix='词频计算')
        
        # 针对中文优化的过滤策略
        original_vocab_size = len(word_freqs)
        
        # 中文字符保留更低的频率阈值
        chinese_chars = {word: freq for word, freq in word_freqs.items() 
                        if len(word) == 1 and is_chinese_char(word)}
        other_words = {word: freq for word, freq in word_freqs.items() 
                      if not (len(word) == 1 and is_chinese_char(word))}
        
        # 中文字符：频率>=1
        # 其他词：根据总词汇量动态调整
        min_freq_chinese = 1
        min_freq_other = 2 if len(other_words) <= 30000 else max(2, len(other_words) // 15000)
        
        filtered_word_freqs = {}
        filtered_word_freqs.update({word: freq for word, freq in chinese_chars.items() 
                                  if freq >= min_freq_chinese})
        filtered_word_freqs.update({word: freq for word, freq in other_words.items() 
                                  if freq >= min_freq_other})
        
        self.word_freqs = filtered_word_freqs
        elapsed = time.time() - start_time
        
        chinese_kept = len([w for w in filtered_word_freqs if len(w) == 1 and is_chinese_char(w)])
        print(f"✅ 词频计算完成! {len(filtered_word_freqs):,}词汇 (中文:{chinese_kept:,}) 耗时:{elapsed:.1f}秒")
    
    def initialize_splits(self):
        """初始化分割：将每个词分割成字符"""
        splits = {}
        for word in self.word_freqs:
            # 对中文字符，不需要进一步分割
            if len(word) == 1 and is_chinese_char(word):
                splits[word] = [word, '</w>']
            else:
                # 对其他词，分割成字符
                splits[word] = [c for c in word] + ['</w>']
        self.splits = splits
        print(f"✅ 初始化分割完成! {len(splits):,}个词")
    
    def compute_pair_frequencies(self) -> Dict[Tuple[str, str], int]:
        """计算相邻字符对的频率"""
        pair_freqs = defaultdict(int)
        
        for word, freq in self.word_freqs.items():
            split = self.splits[word]
            
            # 计算相邻字符对
            for i in range(len(split) - 1):
                pair = (split[i], split[i + 1])
                pair_freqs[pair] += freq
        
        return dict(pair_freqs)
    
    def merge_vocab(self, pair: Tuple[str, str]):
        """合并词汇表中的字符对"""
        new_splits = {}
        merge_count = 0  # 统计实际合并次数
        
        for word in self.word_freqs:
            split = self.splits[word]
            new_split = []
            i = 0
            
            while i < len(split):
                # 查找要合并的字符对
                if i < len(split) - 1 and (split[i], split[i + 1]) == pair:
                    # 合并字符对
                    new_split.append(split[i] + split[i + 1])
                    merge_count += 1
                    i += 2
                else:
                    new_split.append(split[i])
                    i += 1
            
            new_splits[word] = new_split
        
        self.splits = new_splits
        return merge_count  # 返回实际合并次数
    
    def train(self, texts: List[str]):
        """训练BPE分词器"""
        print("🚀 开始训练BPE分词器（中文优化版）...")
        
        # 检查数据量并进行采样
        original_count = len(texts)
        max_texts = 150000  # 增加最大文本数量限制
        
        if original_count > max_texts:
            print(f"⚠️  数据量过大({original_count:,})，采样到{max_texts:,}条")
            random.seed(42)  # 固定随机种子确保可重现
            texts = random.sample(texts, max_texts)
        
        total_start_time = time.time()
        
        # 步骤1：计算词频
        self.compute_word_frequencies(texts)
        
        # 步骤2：初始化分割
        self.initialize_splits()
        
        # 步骤3：迭代合并
        merges = {}
        vocab_size = self.vocab_size - 4  # 减去4个特殊token
        
        chinese_words = len([w for w in self.word_freqs if len(w) == 1 and is_chinese_char(w)])
        print(f"🔄 开始BPE训练: 目标{vocab_size:,}次合并, {len(self.word_freqs):,}词汇(中文:{chinese_words:,})")
        
        merge_start_time = time.time()
        last_best_pair = None  # 上一次最佳字符对
        repeated_pair_count = 0  # 相同字符对重复次数
        no_progress_count = 0  # 无进展次数（没有实际合并）
        
        for i in range(vocab_size):
            # 计算字符对频率
            pair_freqs = self.compute_pair_frequencies()
            
            if not pair_freqs:
                print(f"\n⚠️  没有更多的字符对可以合并，提前结束于第 {i + 1} 次合并")
                break
            
            # 找到频率最高的字符对
            best_pair = max(pair_freqs, key=pair_freqs.get)
            best_freq = pair_freqs[best_pair]
            
            # 死循环检测逻辑
            if best_pair == last_best_pair:
                repeated_pair_count += 1
                if repeated_pair_count >= 3:  # 连续3次相同字符对
                    break
            else:
                repeated_pair_count = 0
            
            if best_pair in merges:
                no_progress_count += 1
                if no_progress_count >= 10:
                    break
                continue
            else:
                no_progress_count = 0
            
            # 合并最频繁的字符对
            merge_count = self.merge_vocab(best_pair)
            
            # 如果没有实际合并任何内容，跳过
            if merge_count == 0:
                no_progress_count += 1
                if no_progress_count >= 10:
                    break
                continue
            else:
                no_progress_count = 0
            
            # 记录合并规则
            merges[best_pair] = i
            last_best_pair = best_pair
            
            # 显示进度（每100次或重要节点）
            if (i + 1) % 100 == 0 or i == vocab_size - 1:
                show_progress_with_stats(i + 1, vocab_size, merge_start_time, 
                                       prefix='BPE训练', 
                                       extra_info=f"最新: '{best_pair[0]}'+'{best_pair[1]}'({best_freq:,})")
        
        self.merges = merges
        
        merge_elapsed = time.time() - merge_start_time
        print(f"\n✅ BPE合并完成！实际合并次数: {len(merges):,} (耗时: {merge_elapsed:.2f}秒)")
        
        # 构建词汇表
        self.build_vocab()
        
        total_elapsed = time.time() - total_start_time
        
        # 统计中文字符覆盖率（包括带</w>的token）
        chinese_tokens = len([token for token in self.vocab 
                            if (len(token) == 1 and is_chinese_char(token)) or 
                               (token.endswith('</w>') and len(token) == 5 and is_chinese_char(token[0]))])
        
        print(f"🎉 训练完成! 词汇表:{len(self.vocab):,} 中文:{chinese_tokens:,} 总耗时:{total_elapsed:.1f}秒")
    
    def build_vocab(self):
        """构建词汇表"""
        vocab = {}
        
        # 添加特殊token
        vocab[self.pad_token] = self.pad_id
        vocab[self.unk_token] = self.unk_id
        vocab[self.bos_token] = self.bos_id
        vocab[self.eos_token] = self.eos_id
        
        # 添加所有子词
        for word in self.word_freqs:
            for token in self.splits[word]:
                if token not in vocab:
                    vocab[token] = len(vocab)
        
        self.vocab = vocab
    
    def encode_word(self, word: str) -> List[str]:
        """编码单个词"""
        # 检查是否是单个中文字符，直接返回带</w>的版本
        if len(word) == 1 and is_chinese_char(word):
            return [word + '</w>']
        
        # 初始化为字符列表
        tokens = [c for c in word] + ['</w>']
        
        # 应用所有合并规则
        for pair, merge_order in sorted(self.merges.items(), key=lambda x: x[1]):
            new_tokens = []
            i = 0
            
            while i < len(tokens):
                if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == pair:
                    new_tokens.append(tokens[i] + tokens[i + 1])
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            
            tokens = new_tokens
        
        return tokens
    
    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """编码文本为token ID列表"""
        # 预分词
        words = self.pre_tokenize(text)
        
        token_ids = []
        
        if add_special_tokens:
            token_ids.append(self.bos_id)
        
        # 编码每个词
        for word in words:
            try:
                tokens = self.encode_word(word)
                for token in tokens:
                    token_id = self.vocab.get(token, self.unk_id)
                    token_ids.append(token_id)
            except Exception as e:
                # 出错时使用未知token
                token_ids.append(self.unk_id)
        
        if add_special_tokens:
            token_ids.append(self.eos_id)
        
        return token_ids
    
    def decode(self, token_ids: List[int]) -> str:
        """解码token ID列表为文本"""
        # 创建ID到token的映射
        id_to_token = {v: k for k, v in self.vocab.items()}
        
        tokens = []
        for token_id in token_ids:
            if token_id in id_to_token:
                token = id_to_token[token_id]
                # 跳过特殊token
                if token not in [self.pad_token, self.bos_token, self.eos_token]:
                    tokens.append(token)
        
        # 合并tokens并处理词结束标记
        text = ''.join(tokens)
        text = text.replace('</w>', ' ')
        
        return text.strip()
    
    def save(self, path: str):
        """保存分词器"""
        data = {
            'vocab': self.vocab,
            'merges': self.merges,
            'vocab_size': self.vocab_size
        }
        
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"分词器已保存到: {path}")
    
    def load(self, path: str):
        """加载分词器"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        self.vocab = data['vocab']
        self.merges = data['merges']
        self.vocab_size = data['vocab_size']
        
        print(f"分词器已加载: {path}")
    
    def get_vocab_size(self) -> int:
        """获取词汇表大小"""
        return len(self.vocab)


def train_tokenizer_from_data(data_path: str, vocab_size: int = 30000) -> BPETokenizer:  # 增加默认词汇表大小
    """从数据文件训练分词器"""
    print(f"📁 加载数据: {data_path}, 目标词汇表: {vocab_size:,}")
    
    texts = []
    start_time = time.time()
    line_count = 0
    valid_count = 0
    
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            line_count += 1
            line = line.strip()
            if line:
                try:
                    data = json.loads(line)
                    if 'conversations' in data:
                        for conv in data['conversations']:
                            texts.append(conv['content'])
                            valid_count += 1
                    elif 'text' in data:
                        texts.append(data['text'])
                        valid_count += 1
                except json.JSONDecodeError:
                    continue
    
    elapsed = time.time() - start_time
    
    # 统计中文字符
    chinese_char_count = sum(len([c for c in text if is_chinese_char(c)]) for text in texts)
    total_char_count = sum(len(text) for text in texts)
    chinese_ratio = chinese_char_count / total_char_count if total_char_count > 0 else 0
    
    print(f"✅ 读取 {len(texts):,} 条文本，中文占比 {chinese_ratio:.1%} (耗时: {elapsed:.2f}秒)")
    
    # 训练分词器
    tokenizer = BPETokenizer(vocab_size=vocab_size)
    tokenizer.train(texts)
    
    return tokenizer


if __name__ == "__main__":
    # 测试分词器
    sample_texts = [
        "Hello world! How are you?",
        "I am learning about BPE tokenization.",
        "This is a sample text for training.",
        "BPE stands for Byte Pair Encoding."
    ]
    
    # 训练分词器
    tokenizer = BPETokenizer(vocab_size=1000)
    tokenizer.train(sample_texts)
    
    # 测试编码和解码
    test_text = "Hello! How are you doing?"
    token_ids = tokenizer.encode(test_text)
    decoded_text = tokenizer.decode(token_ids)
    
    print(f"原文: {test_text}")
    print(f"编码: {token_ids}")
    print(f"解码: {decoded_text}")
    print(f"词汇表大小: {tokenizer.get_vocab_size()}")