# 03 分词策略与信息压缩

> **从字符到子词：语言的离散化与信息论优化**

## 核心思想

分词(Tokenization)是语言模型的第一步，它将连续的文本转换为离散的符号序列。这个看似简单的预处理步骤，实际上深刻影响着模型的学习能力、计算效率和最终性能。

**关键洞察**：
- **信息压缩**：好的分词策略能够用更少的token表达更多的语义信息
- **词汇平衡**：在词汇表大小与序列长度之间找到最优平衡点
- **语言统计**：分词算法实际上是在学习语言的统计结构
- **OOV问题**：子词分解彻底解决了未登录词的问题

从信息论角度看，分词是在寻找**最优的编码方案**，使得文本的平均编码长度最短，同时保持语义的完整性。

## 3.1 从字符到词汇的编码演进

### 编码策略的信息论分析

**字符级编码**：
- 词汇表大小：~100（ASCII）或~65,000（Unicode）
- 序列长度：很长（每个字符一个token）
- OOV率：0%（任何字符都能表示）

**词级编码**：
- 词汇表大小：~50,000-100,000
- 序列长度：较短（每个词一个token）
- OOV率：高（新词、变形、错误等）

**子词编码**：
- 词汇表大小：可配置（通常32K-64K）
- 序列长度：适中
- OOV率：0%（任何词都能分解）

```python
def compare_encoding_strategies(text_corpus, strategies):
    """比较不同编码策略的效果"""
    
    results = {}
    
    for strategy_name, tokenizer in strategies.items():
        total_tokens = 0
        total_chars = 0
        vocab_usage = set()
        oov_count = 0
        
        for text in text_corpus:
            # 编码文本
            if strategy_name == 'char':
                tokens = list(text)
            elif strategy_name == 'word':
                tokens = text.split()
                # 检查OOV
                for token in tokens:
                    if token not in tokenizer.vocab:
                        oov_count += 1
            elif strategy_name == 'subword':
                tokens = tokenizer.encode(text)
            
            total_tokens += len(tokens)
            total_chars += len(text)
            vocab_usage.update(tokens if strategy_name == 'char' else 
                             [str(t) for t in tokens])
        
        # 计算统计指标
        compression_ratio = total_chars / total_tokens if total_tokens > 0 else 0
        vocab_coverage = len(vocab_usage)
        oov_rate = oov_count / total_tokens if total_tokens > 0 else 0
        
        results[strategy_name] = {
            'total_tokens': total_tokens,
            'total_chars': total_chars,
            'compression_ratio': compression_ratio,
            'vocab_coverage': vocab_coverage,
            'oov_rate': oov_rate
        }
        
        print(f"{strategy_name:10s}: tokens={total_tokens:,}, 压缩比={compression_ratio:.2f}, "
              f"词汇覆盖={vocab_coverage:,}, OOV率={oov_rate:.2%}")
    
    return results

def analyze_information_content(text, tokenizer):
    """分析文本的信息含量"""
    
    # 编码文本
    tokens = tokenizer.encode(text)
    
    # 计算token频率
    token_counts = {}
    for token in tokens:
        token_counts[token] = token_counts.get(token, 0) + 1
    
    # 计算概率分布
    total_tokens = len(tokens)
    token_probs = {token: count/total_tokens for token, count in token_counts.items()}
    
    # 计算熵
    entropy = -sum(p * math.log2(p) for p in token_probs.values())
    
    # 计算理论最小编码长度
    min_bits = entropy * total_tokens
    actual_bits = total_tokens * math.log2(len(tokenizer.vocab))
    
    print(f"信息论分析:")
    print(f"  文本长度: {len(text)} 字符")
    print(f"  Token数量: {total_tokens}")
    print(f"  唯一token: {len(token_counts)}")
    print(f"  熵: {entropy:.4f} bits/token")
    print(f"  理论最小编码: {min_bits:.0f} bits")
    print(f"  实际编码: {actual_bits:.0f} bits")
    print(f"  编码效率: {min_bits/actual_bits:.2%}")
    
    return {
        'entropy': entropy,
        'compression_efficiency': min_bits/actual_bits,
        'vocab_utilization': len(token_counts)/len(tokenizer.vocab)
    }
```

## 3.2 BPE算法的数学原理

### 字节对编码的统计学习

**BPE算法核心思想**：迭代地合并最频繁的字符对，直到达到目标词汇表大小。

**算法步骤**：
1. 初始化：每个字符作为基本单元
2. 统计：计算所有相邻字符对的频率
3. 合并：将最频繁的字符对合并为新的子词
4. 迭代：重复步骤2-3，直到达到目标词汇表大小

**数学表达**：
设当前词汇表为$V$，文本语料为$C$，则每次迭代选择的合并对为：
$$(s_1, s_2) = \arg\max_{(s_i, s_j)} \text{count}(s_i, s_j \text{ in } C)$$

```python
# MiniGPT中BPE分词器的核心实现分析
class BPETokenizer:
    """BPE分词器的数学实现与分析"""
    
    def __init__(self, vocab_size=50000):
        self.vocab_size = vocab_size
        self.vocab = {}
        self.merges = {}
        self.word_freqs = {}
    
    def train(self, corpus):
        """训练BPE分词器"""
        
        print("=== BPE训练过程 ===")
        
        # 1. 预处理：统计词频
        print("步骤1: 统计词频")
        for text in corpus:
            words = text.lower().split()
            for word in words:
                word_with_boundary = word + '</w>'  # 添加词边界标记
                self.word_freqs[word_with_boundary] = self.word_freqs.get(word_with_boundary, 0) + 1
        
        print(f"  总词汇数: {len(self.word_freqs)}")
        print(f"  总token数: {sum(self.word_freqs.values())}")
        
        # 2. 初始化：字符级词汇表
        print("\\n步骤2: 初始化字符级词汇表")
        vocab = set()
        for word in self.word_freqs:
            for char in word:
                vocab.add(char)
        
        # 转换为字符序列表示
        word_splits = {}
        for word, freq in self.word_freqs.items():
            word_splits[word] = list(word)
        
        initial_vocab_size = len(vocab)
        print(f"  初始词汇表大小: {initial_vocab_size}")
        
        # 3. 迭代合并
        print("\\n步骤3: 迭代合并最频繁字符对")
        merge_count = 0
        
        while len(vocab) < self.vocab_size:
            # 统计字符对频率
            pair_freqs = self._count_pairs(word_splits, self.word_freqs)
            
            if not pair_freqs:
                print("  没有更多字符对可以合并")
                break
            
            # 找到最频繁的字符对
            best_pair = max(pair_freqs, key=pair_freqs.get)
            best_freq = pair_freqs[best_pair]
            
            # 执行合并
            word_splits = self._merge_pair(best_pair, word_splits)
            self.merges[best_pair] = len(self.merges)
            
            # 更新词汇表
            new_subword = ''.join(best_pair)
            vocab.add(new_subword)
            
            merge_count += 1
            if merge_count % 1000 == 0 or merge_count <= 10:
                print(f"  合并 {merge_count:4d}: {best_pair} → '{new_subword}' (频率: {best_freq})")
        
        # 4. 构建最终词汇表
        self.vocab = {token: i for i, token in enumerate(sorted(vocab))}
        
        print(f"\\n训练完成:")
        print(f"  最终词汇表大小: {len(self.vocab)}")
        print(f"  总合并操作: {len(self.merges)}")
        
        return self
    
    def _count_pairs(self, word_splits, word_freqs):
        """统计字符对频率"""
        pair_counts = {}
        
        for word, splits in word_splits.items():
            word_freq = word_freqs[word]
            
            # 统计该词中的所有相邻字符对
            for i in range(len(splits) - 1):
                pair = (splits[i], splits[i + 1])
                pair_counts[pair] = pair_counts.get(pair, 0) + word_freq
        
        return pair_counts
    
    def _merge_pair(self, pair, word_splits):
        """合并指定字符对"""
        new_word_splits = {}
        
        for word, splits in word_splits.items():
            new_splits = []
            i = 0
            
            while i < len(splits):
                # 检查是否匹配要合并的字符对
                if (i < len(splits) - 1 and 
                    splits[i] == pair[0] and splits[i + 1] == pair[1]):
                    # 合并字符对
                    new_splits.append(pair[0] + pair[1])
                    i += 2
                else:
                    new_splits.append(splits[i])
                    i += 1
            
            new_word_splits[word] = new_splits
        
        return new_word_splits
    
    def encode(self, text):
        """编码文本为token序列"""
        words = text.lower().split()
        encoded = []
        
        for word in words:
            word_with_boundary = word + '</w>'
            
            # 初始化为字符序列
            word_tokens = list(word_with_boundary)
            
            # 应用学到的合并规则
            while len(word_tokens) > 1:
                # 找到可以合并的字符对
                pairs = [(word_tokens[i], word_tokens[i + 1]) 
                        for i in range(len(word_tokens) - 1)]
                
                # 按合并优先级排序
                valid_pairs = [(pair, self.merges[pair]) for pair in pairs 
                              if pair in self.merges]
                
                if not valid_pairs:
                    break
                
                # 选择最先学到的合并（优先级最高）
                best_pair = min(valid_pairs, key=lambda x: x[1])[0]
                
                # 执行合并
                new_tokens = []
                i = 0
                while i < len(word_tokens):
                    if (i < len(word_tokens) - 1 and 
                        word_tokens[i] == best_pair[0] and 
                        word_tokens[i + 1] == best_pair[1]):
                        new_tokens.append(best_pair[0] + best_pair[1])
                        i += 2
                    else:
                        new_tokens.append(word_tokens[i])
                        i += 1
                
                word_tokens = new_tokens
            
            # 转换为词汇表ID
            for token in word_tokens:
                if token in self.vocab:
                    encoded.append(self.vocab[token])
                else:
                    # 处理OOV（理论上BPE不应该有OOV）
                    encoded.append(self.vocab.get('<UNK>', 0))
        
        return encoded

def analyze_bpe_compression_efficiency():
    """分析BPE的压缩效率"""
    
    # 创建测试语料
    test_corpus = [
        "the quick brown fox jumps over the lazy dog",
        "the dog is lazy and the fox is quick",
        "quick brown dogs and lazy foxes are jumping",
        "foxes jump quickly over lazy brown dogs"
    ]
    
    # 训练BPE分词器
    bpe = BPETokenizer(vocab_size=100)
    bpe.train(test_corpus)
    
    # 分析压缩效果
    print("\\n=== 压缩效率分析 ===")
    
    original_chars = sum(len(text) for text in test_corpus)
    total_tokens = 0
    
    for text in test_corpus:
        tokens = bpe.encode(text)
        total_tokens += len(tokens)
        
        print(f"原文: '{text}'")
        print(f"编码: {tokens}")
        print(f"长度: {len(text)} 字符 → {len(tokens)} tokens")
        print()
    
    compression_ratio = original_chars / total_tokens
    print(f"总体压缩比: {compression_ratio:.2f} 字符/token")
    print(f"压缩效率: {(1 - total_tokens/original_chars)*100:.1f}%")
    
    return bpe, compression_ratio
```

### 信息论视角下的BPE优化

**压缩效率**：BPE试图最小化平均编码长度
$$L = \sum_{w \in V} P(w) \cdot |encode(w)|$$

其中$P(w)$是词$w$的概率，$|encode(w)|$是编码后的长度。

**贪心策略的局限性**：BPE采用贪心策略，每次选择频率最高的字符对，但这不一定是全局最优的。

```python
def theoretical_vs_actual_compression():
    """理论最优压缩与BPE实际压缩的比较"""
    
    # 构造简单例子
    words = ['hello', 'world', 'hell', 'word', 'help', 'work']
    freqs = [100, 80, 60, 50, 40, 30]
    
    print("=== 理论最优 vs BPE实际压缩 ===")
    
    # 1. 计算理论最优编码（Huffman编码思想）
    total_freq = sum(freqs)
    word_probs = [f/total_freq for f in freqs]
    
    # 计算熵（理论最优编码长度下界）
    entropy = -sum(p * math.log2(p) for p in word_probs)
    theoretical_bits = entropy * total_freq
    
    print(f"理论分析:")
    print(f"  总词频: {total_freq}")
    print(f"  熵: {entropy:.4f} bits/word")
    print(f"  理论最优编码: {theoretical_bits:.0f} bits")
    
    # 2. BPE实际编码
    # 简化的BPE模拟
    char_vocab = set(''.join(words))
    print(f"\\nBPE分析:")
    print(f"  初始字符词汇: {sorted(char_vocab)}")
    
    # 统计字符对频率
    pair_freqs = {}
    for word, freq in zip(words, freqs):
        for i in range(len(word) - 1):
            pair = (word[i], word[i+1])
            pair_freqs[pair] = pair_freqs.get(pair, 0) + freq
    
    print(f"  字符对频率: {sorted(pair_freqs.items(), key=lambda x: x[1], reverse=True)}")
    
    # 模拟几次合并
    merges = []
    current_splits = {word: list(word) for word in words}
    
    for merge_step in range(3):  # 进行3次合并
        if not pair_freqs:
            break
            
        # 选择最频繁的字符对
        best_pair = max(pair_freqs, key=pair_freqs.get)
        best_freq = pair_freqs[best_pair]
        merges.append((best_pair, best_freq))
        
        print(f"  合并步骤 {merge_step + 1}: {best_pair} (频率: {best_freq})")
        
        # 更新分词结果
        for word in current_splits:
            splits = current_splits[word]
            new_splits = []
            i = 0
            while i < len(splits):
                if (i < len(splits) - 1 and 
                    splits[i] == best_pair[0] and splits[i + 1] == best_pair[1]):
                    new_splits.append(best_pair[0] + best_pair[1])
                    i += 2
                else:
                    new_splits.append(splits[i])
                    i += 1
            current_splits[word] = new_splits
        
        # 重新计算字符对频率
        pair_freqs = {}
        for word, freq in zip(words, freqs):
            splits = current_splits[word]
            for i in range(len(splits) - 1):
                pair = (splits[i], splits[i + 1])
                pair_freqs[pair] = pair_freqs.get(pair, 0) + freq
    
    # 计算BPE编码长度
    bpe_tokens = sum(len(current_splits[word]) * freq for word, freq in zip(words, freqs))
    
    print(f"\\n最终分词结果:")
    for word, freq in zip(words, freqs):
        splits = current_splits[word]
        print(f"  '{word}' → {splits} ({len(splits)} tokens, 频率: {freq})")
    
    print(f"\\nBPE总token数: {bpe_tokens}")
    print(f"平均每词token数: {bpe_tokens/total_freq:.2f}")
    
    # 效率比较
    if theoretical_bits > 0:
        bpe_efficiency = theoretical_bits / (bpe_tokens * math.log2(len(char_vocab) + len(merges)))
        print(f"BPE编码效率: {bpe_efficiency:.2%}")
    
    return current_splits, merges
```

## 3.3 词汇表大小的权衡分析

### 计算复杂度与表达能力的平衡

**词汇表大小对性能的影响**：

1. **计算复杂度**：
   - 嵌入层参数：$O(|V| \times d_{model})$
   - 输出层参数：$O(|V| \times d_{model})$
   - Softmax计算：$O(|V|)$

2. **序列长度**：
   - 大词汇表 → 短序列 → 少的注意力计算
   - 小词汇表 → 长序列 → 多的注意力计算

3. **表达能力**：
   - 大词汇表 → 更精确的语义表达
   - 小词汇表 → 更多的组合表达

```python
def analyze_vocab_size_tradeoffs(text_corpus, vocab_sizes=[1000, 5000, 10000, 30000, 50000]):
    """分析不同词汇表大小的权衡"""
    
    results = {}
    
    for vocab_size in vocab_sizes:
        print(f"\\n=== 词汇表大小: {vocab_size} ===")
        
        # 训练BPE分词器
        bpe = BPETokenizer(vocab_size=vocab_size)
        bpe.train(text_corpus)
        
        # 编码语料库
        total_tokens = 0
        total_chars = 0
        unique_tokens = set()
        
        for text in text_corpus:
            tokens = bpe.encode(text)
            total_tokens += len(tokens)
            total_chars += len(text)
            unique_tokens.update(tokens)
        
        # 计算各项指标
        compression_ratio = total_chars / total_tokens
        vocab_utilization = len(unique_tokens) / vocab_size
        
        # 估算计算成本
        embedding_params = vocab_size * 512  # 假设d_model=512
        output_params = vocab_size * 512
        total_vocab_params = embedding_params + output_params
        
        # 估算注意力计算成本（与序列长度平方成正比）
        avg_seq_len = total_tokens / len(text_corpus)
        attention_cost = avg_seq_len ** 2
        
        results[vocab_size] = {
            'compression_ratio': compression_ratio,
            'vocab_utilization': vocab_utilization,
            'avg_seq_len': avg_seq_len,
            'vocab_params': total_vocab_params,
            'attention_cost': attention_cost,
            'total_cost': total_vocab_params + attention_cost  # 简化的总成本
        }
        
        print(f"  压缩比: {compression_ratio:.2f} 字符/token")
        print(f"  词汇利用率: {vocab_utilization:.2%}")
        print(f"  平均序列长度: {avg_seq_len:.2f}")
        print(f"  词汇参数量: {total_vocab_params:,}")
        print(f"  注意力成本: {attention_cost:.0f}")
    
    # 寻找最优词汇表大小
    print("\\n=== 权衡分析 ===")
    optimal_vocab = min(results.keys(), key=lambda v: results[v]['total_cost'])
    print(f"最优词汇表大小（基于简化成本模型）: {optimal_vocab}")
    
    # 绘制权衡曲线
    import matplotlib.pyplot as plt
    
    vocab_sizes_list = list(results.keys())
    compression_ratios = [results[v]['compression_ratio'] for v in vocab_sizes_list]
    vocab_utilizations = [results[v]['vocab_utilization'] for v in vocab_sizes_list]
    avg_seq_lens = [results[v]['avg_seq_len'] for v in vocab_sizes_list]
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
    
    # 压缩比
    ax1.plot(vocab_sizes_list, compression_ratios, 'b-o')
    ax1.set_xlabel('词汇表大小')
    ax1.set_ylabel('压缩比 (字符/token)')
    ax1.set_title('压缩效率')
    ax1.grid(True)
    
    # 词汇利用率
    ax2.plot(vocab_sizes_list, vocab_utilizations, 'r-o')
    ax2.set_xlabel('词汇表大小')
    ax2.set_ylabel('词汇利用率')
    ax2.set_title('词汇利用效率')
    ax2.grid(True)
    
    # 序列长度
    ax3.plot(vocab_sizes_list, avg_seq_lens, 'g-o')
    ax3.set_xlabel('词汇表大小')
    ax3.set_ylabel('平均序列长度')
    ax3.set_title('序列长度影响')
    ax3.grid(True)
    
    # 成本对比
    vocab_params = [results[v]['vocab_params'] for v in vocab_sizes_list]
    attention_costs = [results[v]['attention_cost'] for v in vocab_sizes_list]
    
    ax4.plot(vocab_sizes_list, vocab_params, 'purple', label='词汇参数成本')
    ax4.plot(vocab_sizes_list, attention_costs, 'orange', label='注意力计算成本')
    ax4.set_xlabel('词汇表大小')
    ax4.set_ylabel('计算成本')
    ax4.set_title('计算成本权衡')
    ax4.legend()
    ax4.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    return results

def memory_and_compute_analysis(d_model=512, seq_len=1024, batch_size=32):
    """详细的内存和计算分析"""
    
    vocab_sizes = [8000, 16000, 32000, 64000]
    
    print("=== 内存和计算详细分析 ===")
    print(f"模型配置: d_model={d_model}, seq_len={seq_len}, batch_size={batch_size}")
    print()
    
    for vocab_size in vocab_sizes:
        print(f"词汇表大小: {vocab_size:,}")
        
        # 1. 参数内存
        embedding_params = vocab_size * d_model * 4  # 4 bytes per float32
        output_params = vocab_size * d_model * 4
        total_vocab_memory = embedding_params + output_params
        
        # 2. 激活内存
        embedding_activations = batch_size * seq_len * d_model * 4
        output_activations = batch_size * seq_len * vocab_size * 4
        
        # 3. 计算量（FLOPs）
        embedding_flops = batch_size * seq_len * d_model
        output_flops = batch_size * seq_len * d_model * vocab_size
        softmax_flops = batch_size * seq_len * vocab_size * 3  # exp + sum + div
        
        print(f"  参数内存:")
        print(f"    嵌入层: {embedding_params/1e6:.1f} MB")
        print(f"    输出层: {output_params/1e6:.1f} MB")
        print(f"    总计: {total_vocab_memory/1e6:.1f} MB")
        
        print(f"  激活内存:")
        print(f"    嵌入激活: {embedding_activations/1e6:.1f} MB")
        print(f"    输出激活: {output_activations/1e6:.1f} MB")
        
        print(f"  计算量:")
        print(f"    嵌入层: {embedding_flops/1e9:.2f} GFLOPs")
        print(f"    输出层: {output_flops/1e9:.2f} GFLOPs")
        print(f"    Softmax: {softmax_flops/1e9:.2f} GFLOPs")
        print(f"    词汇相关总计: {(output_flops + softmax_flops)/1e9:.2f} GFLOPs")
        print()
    
    # 注意力计算成本（与词汇表大小无关）
    attention_memory = batch_size * seq_len * seq_len * 4  # attention weights
    attention_flops = batch_size * seq_len * seq_len * d_model * 4  # Q@K, softmax, @V, proj
    
    print("注意力机制（与词汇表大小无关）:")
    print(f"  内存: {attention_memory/1e6:.1f} MB")
    print(f"  计算量: {attention_flops/1e9:.2f} GFLOPs")
```

## 3.4 多语言与特殊token处理

### 字符集与编码方案

**Unicode支持**：现代分词器需要处理多种语言和字符集

**特殊token**：
- `<PAD>`：填充token，用于批量处理
- `<UNK>`：未知token，处理OOV词
- `<BOS>`/`<EOS>`：序列开始/结束标记
- `<SEP>`：分隔符token，用于多文档任务

```python
class MultilingualBPETokenizer:
    """支持多语言的BPE分词器"""
    
    def __init__(self, vocab_size=50000, special_tokens=None):
        self.vocab_size = vocab_size
        
        # 定义特殊token
        if special_tokens is None:
            special_tokens = ['<PAD>', '<UNK>', '<BOS>', '<EOS>', '<SEP>']
        
        self.special_tokens = special_tokens
        self.special_token_ids = {token: i for i, token in enumerate(special_tokens)}
        
        # 保留词汇表空间给特殊token
        self.regular_vocab_size = vocab_size - len(special_tokens)
        
    def preprocess_multilingual(self, text):
        """多语言文本预处理"""
        
        # 1. Unicode标准化
        text = unicodedata.normalize('NFKC', text)
        
        # 2. 语言检测和处理
        languages = self._detect_languages(text)
        
        # 3. 语言特定的预处理
        processed_segments = []
        
        for segment, lang in self._segment_by_language(text, languages):
            if lang == 'zh':  # 中文
                # 中文分词预处理
                segment = self._preprocess_chinese(segment)
            elif lang == 'ar':  # 阿拉伯语
                # 阿拉伯语预处理（从右到左）
                segment = self._preprocess_arabic(segment)
            elif lang == 'ja':  # 日语
                # 日语预处理（混合字符集）
                segment = self._preprocess_japanese(segment)
            
            processed_segments.append(segment)
        
        return ' '.join(processed_segments)
    
    def _detect_languages(self, text):
        """简化的语言检测"""
        languages = set()
        
        for char in text:
            # 中文字符
            if '\u4e00' <= char <= '\u9fff':
                languages.add('zh')
            # 阿拉伯字符
            elif '\u0600' <= char <= '\u06ff':
                languages.add('ar')
            # 日文平假名/片假名
            elif '\u3040' <= char <= '\u309f' or '\u30a0' <= char <= '\u30ff':
                languages.add('ja')
            # 默认拉丁字符
            else:
                languages.add('en')
        
        return list(languages)
    
    def _segment_by_language(self, text, languages):
        """按语言分割文本"""
        # 简化实现：返回整个文本和主要语言
        main_lang = languages[0] if languages else 'en'
        return [(text, main_lang)]
    
    def _preprocess_chinese(self, text):
        """中文预处理"""
        # 处理中文特定的字符和标点
        # 这里简化处理
        return text
    
    def _preprocess_arabic(self, text):
        """阿拉伯语预处理"""
        # 处理阿拉伯语的从右到左书写
        return text
    
    def _preprocess_japanese(self, text):
        """日语预处理"""
        # 处理平假名、片假名、汉字混合
        return text
    
    def handle_special_cases(self, text):
        """处理特殊情况"""
        
        # 1. 数字处理
        # 将连续数字替换为特殊标记
        import re
        text = re.sub(r'\\d+', '<NUM>', text)
        
        # 2. URL和邮箱处理
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '<URL>', text)
        text = re.sub(r'\\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Z|a-z]{2,}\\b', '<EMAIL>', text)
        
        # 3. 长重复字符处理
        # 将长重复字符压缩
        text = re.sub(r'(.)\\1{3,}', r'\\1\\1\\1', text)
        
        return text

def analyze_tokenization_quality(tokenizer, test_texts, languages):
    """分析分词质量"""
    
    results = {}
    
    for lang, texts in zip(languages, test_texts):
        print(f"\\n=== {lang.upper()} 语言分词质量分析 ===")
        
        total_chars = 0
        total_tokens = 0
        subword_ratios = []
        
        for text in texts:
            # 预处理
            processed_text = tokenizer.preprocess_multilingual(text)
            processed_text = tokenizer.handle_special_cases(processed_text)
            
            # 分词
            tokens = tokenizer.encode(processed_text)
            
            total_chars += len(text)
            total_tokens += len(tokens)
            
            # 分析子词比例
            word_count = len(text.split())
            subword_ratio = len(tokens) / word_count if word_count > 0 else 0
            subword_ratios.append(subword_ratio)
            
            print(f"原文: {text[:50]}...")
            print(f"Token数: {len(tokens)}, 字符数: {len(text)}, 子词比: {subword_ratio:.2f}")
        
        # 统计结果
        avg_compression = total_chars / total_tokens
        avg_subword_ratio = np.mean(subword_ratios)
        
        results[lang] = {
            'compression_ratio': avg_compression,
            'subword_ratio': avg_subword_ratio,
            'total_tokens': total_tokens,
            'total_chars': total_chars
        }
        
        print(f"\\n{lang} 语言统计:")
        print(f"  平均压缩比: {avg_compression:.2f} 字符/token")
        print(f"  平均子词比: {avg_subword_ratio:.2f} token/词")
    
    return results

def cross_lingual_evaluation():
    """跨语言评估"""
    
    # 多语言测试数据
    multilingual_texts = {
        'en': [
            "The quick brown fox jumps over the lazy dog.",
            "Natural language processing is fascinating.",
            "Machine learning models require lots of data."
        ],
        'zh': [
            "自然语言处理是人工智能的重要分支。",
            "机器学习模型需要大量的训练数据。",
            "深度学习在近年来取得了重大突破。"
        ],
        'ja': [
            "自然言語処理は人工知能の重要な分野です。",
            "機械学習モデルは大量のデータが必要です。",
            "深層学習は近年大きな進歩を遂げています。"
        ],
        'ar': [
            "معالجة اللغة الطبيعية هي فرع مهم من الذكاء الاصطناعي.",
            "نماذج التعلم الآلي تتطلب كميات كبيرة من البيانات.",
            "التعلم العميق حقق اختراقات كبيرة في السنوات الأخيرة."
        ]
    }
    
    # 创建多语言分词器
    multilingual_tokenizer = MultilingualBPETokenizer(vocab_size=50000)
    
    # 合并所有语言的文本进行训练
    all_texts = []
    for lang_texts in multilingual_texts.values():
        all_texts.extend(lang_texts)
    
    # 训练分词器
    multilingual_tokenizer.train(all_texts)
    
    # 评估各语言的分词质量
    results = analyze_tokenization_quality(
        multilingual_tokenizer,
        list(multilingual_texts.values()),
        list(multilingual_texts.keys())
    )
    
    # 跨语言一致性分析
    print("\\n=== 跨语言一致性分析 ===")
    compression_ratios = [results[lang]['compression_ratio'] for lang in results]
    consistency_score = 1 - (max(compression_ratios) - min(compression_ratios)) / np.mean(compression_ratios)
    
    print(f"压缩比一致性评分: {consistency_score:.3f}")
    print("(1.0 = 完全一致, 0.0 = 完全不一致)")
    
    return results, multilingual_tokenizer
```

## 3.5 实践：MiniGPT中的分词实现

### 与MiniGPT代码的对应分析

```python
# MiniGPT分词器实现解析 (src/tokenizer/bpe_tokenizer.py)
def analyze_minigpt_tokenizer():
    """分析MiniGPT中的分词器实现"""
    
    print("=== MiniGPT分词器代码解析 ===")
    
    # 从MiniGPT源码中分析关键组件
    tokenizer_components = {
        'vocab_construction': {
            'description': '词汇表构建过程',
            'key_methods': ['build_vocab', 'add_special_tokens'],
            'optimization': '频率统计和合并优化'
        },
        'encoding_process': {
            'description': '文本编码过程',
            'key_methods': ['encode', 'encode_batch'],
            'optimization': '批量处理和缓存机制'
        },
        'decoding_process': {
            'description': '序列解码过程', 
            'key_methods': ['decode', 'decode_batch'],
            'optimization': '特殊token处理和边界检测'
        },
        'serialization': {
            'description': '模型序列化',
            'key_methods': ['save', 'load'],
            'optimization': '词汇表和合并规则的压缩存储'
        }
    }
    
    for component, details in tokenizer_components.items():
        print(f"\\n{component.upper()}:")
        print(f"  描述: {details['description']}")
        print(f"  关键方法: {', '.join(details['key_methods'])}")
        print(f"  优化策略: {details['optimization']}")
    
    # 性能基准测试
    print("\\n=== 性能基准测试 ===")
    
    # 模拟不同大小的文本处理
    text_sizes = [1000, 10000, 100000]  # 字符数
    vocab_sizes = [8000, 16000, 32000]
    
    for vocab_size in vocab_sizes:
        print(f"\\n词汇表大小: {vocab_size}")
        
        for text_size in text_sizes:
            # 生成测试文本
            test_text = "hello world " * (text_size // 12)
            
            # 模拟编码时间
            start_time = time.time()
            
            # 这里使用简化的编码模拟
            # 实际MiniGPT实现会更复杂
            tokens = simple_bpe_encode(test_text, vocab_size)
            
            encoding_time = time.time() - start_time
            
            # 计算吞吐量
            throughput = len(test_text) / encoding_time if encoding_time > 0 else float('inf')
            
            print(f"  文本大小 {text_size:6d}: "
                  f"编码时间 {encoding_time*1000:.2f}ms, "
                  f"吞吐量 {throughput:.0f} 字符/秒")

def simple_bpe_encode(text, vocab_size):
    """简化的BPE编码实现（用于性能测试）"""
    # 这是一个简化版本，实际实现会更复杂
    words = text.split()
    tokens = []
    
    for word in words:
        # 简单的子词分解
        if len(word) <= 4:
            tokens.append(word)
        else:
            # 分解为4字符的子词
            for i in range(0, len(word), 4):
                tokens.append(word[i:i+4])
    
    return tokens

def integration_test_with_model():
    """与模型集成测试"""
    
    print("=== 分词器与模型集成测试 ===")
    
    # 创建测试分词器
    tokenizer = BPETokenizer(vocab_size=1000)
    
    # 训练数据
    train_texts = [
        "机器学习是人工智能的核心技术。",
        "深度学习模型需要大量数据训练。",
        "自然语言处理应用广泛。"
    ]
    
    tokenizer.train(train_texts)
    
    # 测试不同长度的文本
    test_cases = [
        "短文本",
        "这是一个中等长度的测试文本，包含了多种词汇和表达方式。",
        "这是一个很长的测试文本，" * 20 + "用来测试分词器在处理长文本时的性能和稳定性。"
    ]
    
    for i, text in enumerate(test_cases):
        print(f"\\n测试用例 {i+1}:")
        print(f"原文长度: {len(text)} 字符")
        
        # 编码
        tokens = tokenizer.encode(text)
        print(f"Token数量: {len(tokens)}")
        print(f"压缩比: {len(text)/len(tokens):.2f}")
        
        # 解码验证
        decoded_text = tokenizer.decode(tokens)
        print(f"解码一致性: {'✓' if decoded_text.replace(' ', '') == text.replace(' ', '') else '✗'}")
        
        # 内存使用估计
        token_memory = len(tokens) * 4  # 假设每个token ID 4字节
        original_memory = len(text.encode('utf-8'))
        memory_efficiency = original_memory / token_memory
        
        print(f"内存效率: {memory_efficiency:.2f} (原文/token内存)")
```

## 小结与思考

本节深入探讨了分词策略与信息压缩：

1. **编码演进**：从字符级到子词级编码的信息论优化
2. **BPE算法**：通过统计学习实现最优的字符对合并
3. **词汇表权衡**：在计算成本和表达能力之间找到平衡
4. **多语言处理**：跨语言分词的挑战和解决方案

**关键洞察**：
- 分词是语言的信息压缩问题
- BPE通过贪心策略近似最优编码
- 词汇表大小需要综合考虑多个因素
- 特殊token和多语言支持是实用系统的必要功能

**思考题**：
1. 为什么BPE比词级分词更适合现代语言模型？
2. 如何设计更好的子词分解算法？
3. 不同语言的分词策略应该如何差异化？
4. 分词粒度对模型性能的影响机制是什么？

**下一节预告**：我们将学习优化算法深度解析，理解如何高效地训练大规模语言模型。

---

*分词不仅是技术问题，更是语言理解的哲学问题——如何在离散的符号中保持连续的语义。* 🔤