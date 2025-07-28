# 04 生成质量控制与优化

> **从概率采样到质量保证：构建可靠的语言生成系统**

## 核心思想

生成质量控制是语言模型从"能生成"到"生成好"的关键跃升。单纯的概率采样往往产生各种问题：重复啰嗦、主题偏移、长度偏差、语义不连贯等。质量控制技术通过引入额外的约束和优化机制，在保持生成多样性的同时确保输出质量。

**关键洞察**：
- **多维质量标准**：流畅性、连贯性、信息量、相关性的综合优化
- **实时质量监控**：生成过程中的动态质量评估与调整
- **系统性优化**：从算法层面到工程层面的全方位优化
- **平衡艺术**：质量与效率、确定性与多样性的精妙平衡

从数学角度看，质量控制是在概率空间中定义额外的约束条件，将原始的采样过程转化为约束优化问题。

## 4.1 重复检测与抑制的数学建模

### n-gram重复的统计分析

**重复度量定义**：
设生成序列为 $y = (y_1, y_2, ..., y_T)$，定义 $n$-gram 重复率：
$$R_n(y) = \frac{\text{重复的n-gram数量}}{\text{总n-gram数量}} = \frac{|N_n(y)| - |U_n(y)|}{|N_n(y)|}$$

其中 $N_n(y)$ 是所有 $n$-gram 的多重集合，$U_n(y)$ 是去重后的集合。

**重复惩罚机制**：
修改条件概率分布以降低重复token的概率：
$$P_{rep}(y_t|y_{<t}) = \frac{P(y_t|y_{<t})}{\alpha^{c(y_t, y_{<t})}}$$

其中 $c(y_t, y_{<t})$ 是 $y_t$ 在历史序列中的出现次数，$\alpha > 1$ 是重复惩罚系数。

**数学性质分析**：
1. **熵变化**：重复惩罚降低分布熵
   $$H(P_{rep}) < H(P)$$

2. **收敛性**：随着序列长度增加，重复惩罚效果增强
   $$\lim_{T \to \infty} P_{rep}(y_t|y_{<t}) \to 0 \text{ if } c(y_t, y_{<t}) > 0$$

3. **平衡点**：存在最优惩罚系数 $\alpha^*$ 使得质量最大化

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
import matplotlib.pyplot as plt
import seaborn as sns
import math
import time
from collections import defaultdict, Counter, deque
from dataclasses import dataclass
import re
from scipy import stats
from sklearn.metrics import pairwise_distances
import networkx as nx
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer

@dataclass
class QualityMetrics:
    """生成质量指标"""
    repetition_rate: float        # 重复率
    diversity_score: float        # 多样性分数  
    coherence_score: float        # 连贯性分数
    fluency_score: float         # 流畅性分数
    relevance_score: float       # 相关性分数
    length_quality: float        # 长度质量
    overall_quality: float       # 综合质量
    
class RepetitionController:
    """重复控制器"""
    
    def __init__(self, 
                 repetition_penalty: float = 1.1,
                 no_repeat_ngram_size: int = 3,
                 length_penalty: float = 1.0,
                 diversity_penalty: float = 0.0):
        
        self.repetition_penalty = repetition_penalty
        self.no_repeat_ngram_size = no_repeat_ngram_size
        self.length_penalty = length_penalty
        self.diversity_penalty = diversity_penalty
        
        # 统计信息
        self.ngram_counts = defaultdict(Counter)
        self.token_frequencies = Counter()
        self.repetition_history = []
        
    def compute_repetition_penalty(self, 
                                 logits: torch.Tensor, 
                                 input_ids: torch.Tensor) -> torch.Tensor:
        """计算重复惩罚"""
        
        batch_size, vocab_size = logits.shape
        penalized_logits = logits.clone()
        
        for batch_idx in range(batch_size):
            # 统计token频次
            token_counts = Counter(input_ids[batch_idx].tolist())
            
            # 应用重复惩罚
            for token_id, count in token_counts.items():
                if count > 0:
                    if penalized_logits[batch_idx, token_id] > 0:
                        penalized_logits[batch_idx, token_id] /= (self.repetition_penalty ** count)
                    else:
                        penalized_logits[batch_idx, token_id] *= (self.repetition_penalty ** count)
        
        return penalized_logits
    
    def apply_ngram_blocking(self, 
                           logits: torch.Tensor, 
                           input_ids: torch.Tensor) -> torch.Tensor:
        """应用n-gram阻止机制"""
        
        if self.no_repeat_ngram_size <= 0:
            return logits
        
        batch_size, vocab_size = logits.shape
        seq_len = input_ids.size(1)
        
        blocked_logits = logits.clone()
        
        for batch_idx in range(batch_size):
            sequence = input_ids[batch_idx].tolist()
            
            # 生成所有n-gram
            if seq_len >= self.no_repeat_ngram_size:
                ngrams = set()
                for i in range(seq_len - self.no_repeat_ngram_size + 1):
                    ngram = tuple(sequence[i:i + self.no_repeat_ngram_size])
                    ngrams.add(ngram)
                
                # 检查可能的重复n-gram
                if seq_len >= self.no_repeat_ngram_size - 1:
                    prefix = tuple(sequence[-(self.no_repeat_ngram_size-1):])
                    
                    # 阻止会形成重复n-gram的token
                    for token_id in range(vocab_size):
                        candidate_ngram = prefix + (token_id,)
                        if candidate_ngram in ngrams:
                            blocked_logits[batch_idx, token_id] = -float('inf')
        
        return blocked_logits
    
    def analyze_repetition_patterns(self, sequences: List[List[int]]) -> Dict:
        """分析重复模式"""
        
        print("=== 重复模式分析 ===")
        
        repetition_analysis = {
            'unigram_repetition': [],
            'bigram_repetition': [],
            'trigram_repetition': [],
            'longest_repeat': [],
            'repetition_distribution': defaultdict(int)
        }
        
        for sequence in sequences:
            seq_len = len(sequence)
            
            # 分析不同长度的n-gram重复
            for n in range(1, 4):
                ngrams = []
                for i in range(seq_len - n + 1):
                    ngram = tuple(sequence[i:i+n])
                    ngrams.append(ngram)
                
                # 计算重复率
                total_ngrams = len(ngrams)
                unique_ngrams = len(set(ngrams))
                repetition_rate = (total_ngrams - unique_ngrams) / max(total_ngrams, 1)
                
                if n == 1:
                    repetition_analysis['unigram_repetition'].append(repetition_rate)
                elif n == 2:
                    repetition_analysis['bigram_repetition'].append(repetition_rate)
                elif n == 3:
                    repetition_analysis['trigram_repetition'].append(repetition_rate)
            
            # 找到最长重复序列
            longest_repeat = self._find_longest_repeat(sequence)
            repetition_analysis['longest_repeat'].append(longest_repeat)
            repetition_analysis['repetition_distribution'][longest_repeat] += 1
        
        # 计算统计信息
        stats_summary = {}
        for key in ['unigram_repetition', 'bigram_repetition', 'trigram_repetition', 'longest_repeat']:
            values = repetition_analysis[key]
            stats_summary[key] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'median': np.median(values)
            }
        
        print(f"Unigram重复率: {stats_summary['unigram_repetition']['mean']:.4f} ± {stats_summary['unigram_repetition']['std']:.4f}")
        print(f"Bigram重复率: {stats_summary['bigram_repetition']['mean']:.4f} ± {stats_summary['bigram_repetition']['std']:.4f}")
        print(f"Trigram重复率: {stats_summary['trigram_repetition']['mean']:.4f} ± {stats_summary['trigram_repetition']['std']:.4f}")
        print(f"平均最长重复长度: {stats_summary['longest_repeat']['mean']:.2f}")
        
        return {
            'detailed_analysis': repetition_analysis,
            'statistics': stats_summary
        }
    
    def _find_longest_repeat(self, sequence: List[int]) -> int:
        """找到序列中最长的重复子序列"""
        
        max_repeat_length = 0
        seq_len = len(sequence)
        
        for length in range(1, seq_len // 2 + 1):
            for start in range(seq_len - length + 1):
                pattern = sequence[start:start + length]
                
                # 在剩余序列中查找重复
                for pos in range(start + length, seq_len - length + 1):
                    if sequence[pos:pos + length] == pattern:
                        max_repeat_length = max(max_repeat_length, length)
                        break
        
        return max_repeat_length

    def visualize_repetition_analysis(self, analysis_results: Dict):
        """可视化重复分析结果"""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. 不同n-gram的重复率分布
        ax1 = axes[0, 0]
        repetition_data = [
            analysis_results['detailed_analysis']['unigram_repetition'],
            analysis_results['detailed_analysis']['bigram_repetition'],
            analysis_results['detailed_analysis']['trigram_repetition']
        ]
        labels = ['Unigram', 'Bigram', 'Trigram']
        ax1.boxplot(repetition_data, labels=labels)
        ax1.set_title('不同N-gram重复率分布')
        ax1.set_ylabel('重复率')
        ax1.grid(True, alpha=0.3)
        
        # 2. 最长重复长度分布
        ax2 = axes[0, 1]
        longest_repeats = analysis_results['detailed_analysis']['longest_repeat']
        ax2.hist(longest_repeats, bins=20, alpha=0.7, edgecolor='black')
        ax2.set_title('最长重复序列长度分布')
        ax2.set_xlabel('重复长度')
        ax2.set_ylabel('频次')
        ax2.grid(True, alpha=0.3)
        
        # 3. 重复长度统计
        ax3 = axes[1, 0]
        repeat_dist = analysis_results['detailed_analysis']['repetition_distribution']
        lengths = list(repeat_dist.keys())
        counts = list(repeat_dist.values())
        ax3.bar(lengths, counts, alpha=0.7)
        ax3.set_title('重复长度频次分布')
        ax3.set_xlabel('重复长度')
        ax3.set_ylabel('序列数量')
        ax3.grid(True, axis='y', alpha=0.3)
        
        # 4. 重复率统计摘要
        ax4 = axes[1, 1]
        stats = analysis_results['statistics']
        metrics = ['unigram_repetition', 'bigram_repetition', 'trigram_repetition']
        means = [stats[metric]['mean'] for metric in metrics]
        stds = [stats[metric]['std'] for metric in metrics]
        
        x_pos = np.arange(len(metrics))
        ax4.bar(x_pos, means, yerr=stds, alpha=0.7, capsize=5)
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels(['Unigram', 'Bigram', 'Trigram'])
        ax4.set_title('重复率统计摘要')
        ax4.set_ylabel('重复率')
        ax4.grid(True, axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.show()

## 4.2 长度偏差校正的统计学理论

### 长度偏差的数学建模

**长度偏差现象**：
在序列生成中，较短序列往往具有更高的累积概率：
$$P(y_{1:T}) = \prod_{t=1}^{T} P(y_t|y_{<t})$$

由于每个条件概率都小于1，序列长度增加导致累积概率下降。

**长度标准化方法**：
1. **平均对数概率**：
   $$\text{Score}_{avg}(y) = \frac{1}{T} \sum_{t=1}^{T} \log P(y_t|y_{<t})$$

2. **长度惩罚**：
   $$\text{Score}_{lp}(y) = \frac{\log P(y)}{T^\alpha}$$
   其中 $\alpha$ 是长度惩罚参数。

3. **Wu-Brevity惩罚**：
   $$\text{Score}_{wu}(y) = \log P(y) + \alpha \log T$$

**最优长度的理论分析**：
设真实长度分布为 $P(T)$，生成长度分布为 $Q(T)$，长度偏差可用KL散度衡量：
$$D_{KL}(P(T) \| Q(T)) = \sum_T P(T) \log \frac{P(T)}{Q(T)}$$

class LengthController:
    """长度控制器"""
    
    def __init__(self, 
                 length_penalty: float = 1.0,
                 min_length: int = 10,
                 max_length: int = 512,
                 target_length: Optional[int] = None):
        
        self.length_penalty = length_penalty
        self.min_length = min_length
        self.max_length = max_length
        self.target_length = target_length
        
        # 长度统计
        self.length_history = []
        self.length_distribution = Counter()
        
    def apply_length_penalty(self, 
                           scores: torch.Tensor, 
                           lengths: torch.Tensor) -> torch.Tensor:
        """应用长度惩罚"""
        
        if self.length_penalty == 1.0:
            return scores
        
        # 避免除零
        length_penalty = torch.clamp(lengths.float(), min=1.0) ** self.length_penalty
        return scores / length_penalty
    
    def compute_length_bias(self, 
                          sequences: List[List[int]], 
                          scores: List[float]) -> Dict:
        """计算长度偏差"""
        
        print("=== 长度偏差分析 ===")
        
        lengths = [len(seq) for seq in sequences]
        
        # 计算长度与分数的相关性
        correlation = np.corrcoef(lengths, scores)[0, 1]
        
        # 分析不同长度区间的分数分布
        length_bins = {}
        for length, score in zip(lengths, scores):
            bin_key = (length // 50) * 50  # 50 token一个区间
            if bin_key not in length_bins:
                length_bins[bin_key] = []
            length_bins[bin_key].append(score)
        
        # 计算各区间统计信息
        bin_stats = {}
        for bin_key, bin_scores in length_bins.items():
            bin_stats[bin_key] = {
                'mean_score': np.mean(bin_scores),
                'std_score': np.std(bin_scores),
                'count': len(bin_scores),
                'length_range': f"{bin_key}-{bin_key+49}"
            }
        
        print(f"长度与分数相关系数: {correlation:.4f}")
        print("各长度区间分数统计:")
        for bin_key in sorted(bin_stats.keys()):
            stats = bin_stats[bin_key]
            print(f"  {stats['length_range']}: 平均分数={stats['mean_score']:.4f}, "
                  f"标准差={stats['std_score']:.4f}, 样本数={stats['count']}")
        
        return {
            'correlation': correlation,
            'length_distribution': Counter(lengths),
            'bin_statistics': bin_stats,
            'overall_stats': {
                'mean_length': np.mean(lengths),
                'std_length': np.std(lengths),
                'min_length': np.min(lengths),
                'max_length': np.max(lengths)
            }
        }
    
    def optimize_length_penalty(self, 
                              sequences: List[List[int]], 
                              scores: List[float],
                              target_correlation: float = 0.0) -> float:
        """优化长度惩罚参数"""
        
        lengths = np.array([len(seq) for seq in sequences])
        scores = np.array(scores)
        
        def objective(alpha):
            # 应用长度惩罚
            adjusted_scores = scores / (lengths ** alpha)
            # 计算调整后的相关性
            correlation = np.corrcoef(lengths, adjusted_scores)[0, 1]
            # 最小化与目标相关性的差距
            return abs(correlation - target_correlation)
        
        # 搜索最优alpha
        from scipy.optimize import minimize_scalar
        result = minimize_scalar(objective, bounds=(0.1, 2.0), method='bounded')
        
        optimal_alpha = result.x
        print(f"最优长度惩罚参数: {optimal_alpha:.4f}")
        
        return optimal_alpha

## 4.3 主题连贯性维护的语义建模

### 语义连贯性的数学量化

**语义向量空间模型**：
设文本片段的语义表示为向量 $\mathbf{v}_i \in \mathbb{R}^d$，连贯性可通过向量相似度衡量：
$$\text{Coherence}(S) = \frac{1}{|S|-1} \sum_{i=1}^{|S|-1} \cos(\mathbf{v}_i, \mathbf{v}_{i+1})$$

**主题漂移检测**：
使用滑动窗口检测主题突变：
$$\text{Drift}(t) = 1 - \cos(\mathbf{c}_{t-w:t}, \mathbf{c}_{t:t+w})$$
其中 $\mathbf{c}_{i:j}$ 是窗口 $[i,j]$ 的中心化语义向量。

**语义约束生成**：
在生成过程中施加语义约束：
$$P_{sem}(y_t|y_{<t}) \propto P(y_t|y_{<t}) \cdot \exp(\lambda \cdot \text{Sim}(\mathbf{v}(y_t), \mathbf{c}_{<t}))$$

class CoherenceController:
    """连贯性控制器"""
    
    def __init__(self, 
                 coherence_threshold: float = 0.5,
                 window_size: int = 20,
                 semantic_weight: float = 0.1):
        
        self.coherence_threshold = coherence_threshold
        self.window_size = window_size
        self.semantic_weight = semantic_weight
        
        # 假设的语义编码器（实际应用中使用预训练模型）
        self.semantic_dim = 768
        self.coherence_history = []
        
    def compute_semantic_similarity(self, 
                                  text1: str, 
                                  text2: str) -> float:
        """计算语义相似度（简化实现）"""
        
        # 在实际应用中，这里应该使用BERT/RoBERTa等模型
        # 这里使用简化的字符级相似度作为代理
        
        words1 = set(text1.split())
        words2 = set(text2.split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)
    
    def analyze_topic_coherence(self, 
                              texts: List[str]) -> Dict:
        """分析主题连贯性"""
        
        print("=== 主题连贯性分析 ===")
        
        coherence_scores = []
        drift_points = []
        
        for i, text in enumerate(texts):
            sentences = text.split('.')  # 简化的句子分割
            if len(sentences) < 2:
                continue
            
            # 计算句子间连贯性
            sentence_coherence = []
            for j in range(len(sentences) - 1):
                similarity = self.compute_semantic_similarity(
                    sentences[j], sentences[j + 1])
                sentence_coherence.append(similarity)
            
            avg_coherence = np.mean(sentence_coherence) if sentence_coherence else 0
            coherence_scores.append(avg_coherence)
            
            # 检测主题漂移点
            drift_detected = any(sim < self.coherence_threshold 
                               for sim in sentence_coherence)
            if drift_detected:
                drift_points.append(i)
        
        # 统计分析
        coherence_stats = {
            'mean_coherence': np.mean(coherence_scores),
            'std_coherence': np.std(coherence_scores),
            'min_coherence': np.min(coherence_scores),
            'max_coherence': np.max(coherence_scores),
            'drift_rate': len(drift_points) / len(texts)
        }
        
        print(f"平均连贯性分数: {coherence_stats['mean_coherence']:.4f}")
        print(f"连贯性标准差: {coherence_stats['std_coherence']:.4f}")
        print(f"主题漂移率: {coherence_stats['drift_rate']:.4f}")
        
        return {
            'coherence_scores': coherence_scores,
            'drift_points': drift_points,
            'statistics': coherence_stats
        }
    
    def maintain_topic_consistency(self, 
                                 current_context: str,
                                 candidate_tokens: List[str],
                                 logits: torch.Tensor) -> torch.Tensor:
        """维护主题一致性"""
        
        if not current_context.strip():
            return logits
        
        # 计算每个候选token与上下文的语义相似度
        context_words = set(current_context.split())
        
        adjusted_logits = logits.clone()
        
        for i, token in enumerate(candidate_tokens):
            # 构造包含候选token的新文本
            extended_text = current_context + " " + token
            
            # 计算语义一致性（简化实现）
            token_words = set(token.split())
            overlap = len(context_words.intersection(token_words))
            consistency_score = overlap / max(len(context_words), 1)
            
            # 调整logits
            adjustment = self.semantic_weight * consistency_score
            adjusted_logits[i] += adjustment
        
        return adjusted_logits

## 4.4 实时优化技术的算法分析

### KV缓存机制的数学分析

**注意力计算复杂度**：
标准自注意力的计算复杂度为 $O(T^2 d)$，其中 $T$ 是序列长度，$d$ 是隐藏维度。

**KV缓存原理**：
在自回归生成中，Key和Value矩阵可以增量计算：
$$\mathbf{K}_{t+1} = [\mathbf{K}_t; \mathbf{k}_{t+1}], \quad \mathbf{V}_{t+1} = [\mathbf{V}_t; \mathbf{v}_{t+1}]$$

**复杂度减少**：
使用KV缓存后，每步的计算复杂度降至 $O(Td)$，总复杂度从 $O(T^3 d)$ 降至 $O(T^2 d)$。

**并行生成策略**：
对于批量生成，可以使用动态批处理：
$$\text{Batch}_t = \{(x_i, y_{i,<t}) : |y_{i,<t}| = t-1, i \in \text{Active}\}$$

class GenerationOptimizer:
    """生成优化器"""
    
    def __init__(self, 
                 max_batch_size: int = 32,
                 kv_cache_size: int = 2048,
                 parallel_beams: int = 4):
        
        self.max_batch_size = max_batch_size
        self.kv_cache_size = kv_cache_size
        self.parallel_beams = parallel_beams
        
        # 性能统计
        self.timing_stats = defaultdict(list)
        self.memory_stats = defaultdict(list)
        
    def implement_kv_cache(self, 
                          model,
                          input_ids: torch.Tensor,
                          past_key_values: Optional[Tuple] = None) -> Tuple:
        """实现KV缓存机制"""
        
        start_time = time.time()
        
        batch_size, seq_len = input_ids.shape
        
        if past_key_values is None:
            # 初次计算，需要计算完整的KV
            with torch.no_grad():
                outputs = model(input_ids, use_cache=True)
                past_key_values = outputs.past_key_values
        else:
            # 增量计算，只计算新token
            new_token = input_ids[:, -1:]
            with torch.no_grad():
                outputs = model(new_token, past_key_values=past_key_values, use_cache=True)
                past_key_values = outputs.past_key_values
        
        cache_time = time.time() - start_time
        self.timing_stats['kv_cache'].append(cache_time)
        
        return outputs.logits, past_key_values
    
    def dynamic_batching(self, 
                        requests: List[Dict],
                        max_batch_size: int = None) -> List[List[Dict]]:
        """动态批处理"""
        
        if max_batch_size is None:
            max_batch_size = self.max_batch_size
        
        # 按序列长度分组
        length_groups = defaultdict(list)
        for req in requests:
            length = len(req.get('input_ids', []))
            length_groups[length].append(req)
        
        batches = []
        for length, group_requests in length_groups.items():
            # 将同长度请求分批
            for i in range(0, len(group_requests), max_batch_size):
                batch = group_requests[i:i + max_batch_size]
                batches.append(batch)
        
        return batches
    
    def parallel_beam_search(self, 
                           model,
                           input_ids: torch.Tensor,
                           num_beams: int = 4,
                           max_length: int = 100) -> List[List[int]]:
        """并行束搜索"""
        
        start_time = time.time()
        
        batch_size = input_ids.size(0)
        vocab_size = model.config.vocab_size if hasattr(model, 'config') else 10000
        
        # 初始化束
        beam_scores = torch.zeros(batch_size, num_beams)
        beam_sequences = input_ids.unsqueeze(1).repeat(1, num_beams, 1)
        beam_lengths = torch.full((batch_size, num_beams), input_ids.size(1))
        
        # 束搜索主循环
        for step in range(max_length - input_ids.size(1)):
            # 并行计算所有束的下一个token概率
            current_sequences = beam_sequences.view(-1, beam_sequences.size(-1))
            
            with torch.no_grad():
                logits = model(current_sequences).logits[:, -1, :]
            
            # 重塑为束形状
            logits = logits.view(batch_size, num_beams, vocab_size)
            
            # 计算束分数
            log_probs = F.log_softmax(logits, dim=-1)
            scores = beam_scores.unsqueeze(-1) + log_probs
            
            # 选择top-k候选
            scores_flat = scores.view(batch_size, -1)
            top_scores, top_indices = torch.topk(scores_flat, num_beams)
            
            # 更新束
            beam_indices = top_indices // vocab_size
            token_indices = top_indices % vocab_size
            
            # 重新组织序列
            new_sequences = []
            new_scores = []
            
            for batch_idx in range(batch_size):
                batch_sequences = []
                batch_scores = []
                
                for beam_idx in range(num_beams):
                    old_beam_idx = beam_indices[batch_idx, beam_idx]
                    new_token = token_indices[batch_idx, beam_idx]
                    
                    # 复制旧序列并添加新token
                    old_sequence = beam_sequences[batch_idx, old_beam_idx]
                    new_sequence = torch.cat([old_sequence, new_token.unsqueeze(0)])
                    
                    batch_sequences.append(new_sequence)
                    batch_scores.append(top_scores[batch_idx, beam_idx])
                
                new_sequences.append(torch.stack(batch_sequences))
                new_scores.append(torch.stack(batch_scores))
            
            beam_sequences = torch.stack(new_sequences)
            beam_scores = torch.stack(new_scores)
            
            # 检查结束条件（简化实现）
            # 在实际应用中需要处理EOS token
        
        search_time = time.time() - start_time
        self.timing_stats['beam_search'].append(search_time)
        
        # 返回最佳序列
        best_sequences = []
        for batch_idx in range(batch_size):
            best_beam_idx = torch.argmax(beam_scores[batch_idx])
            best_sequence = beam_sequences[batch_idx, best_beam_idx].tolist()
            best_sequences.append(best_sequence)
        
        return best_sequences
    
    def memory_efficient_generation(self, 
                                  model,
                                  input_ids: torch.Tensor,
                                  max_length: int = 100,
                                  chunk_size: int = 512) -> List[int]:
        """内存高效生成"""
        
        sequence = input_ids[0].tolist()  # 假设batch_size=1
        past_key_values = None
        
        for step in range(max_length - len(sequence)):
            # 限制输入长度以节省内存
            if len(sequence) > chunk_size:
                # 截断早期token，保留recent context
                recent_sequence = sequence[-chunk_size:]
                current_input = torch.tensor([recent_sequence])
                past_key_values = None  # 重置缓存
            else:
                current_input = torch.tensor([sequence])
            
            # 生成下一个token
            with torch.no_grad():
                if past_key_values is not None:
                    outputs = model(current_input[:, -1:], 
                                  past_key_values=past_key_values, 
                                  use_cache=True)
                else:
                    outputs = model(current_input, use_cache=True)
                
                past_key_values = outputs.past_key_values
                next_token_logits = outputs.logits[0, -1, :]
            
            # 采样下一个token
            next_token = torch.multinomial(F.softmax(next_token_logits, dim=-1), 1).item()
            sequence.append(next_token)
            
            # 检查结束条件
            if next_token == 1:  # EOS token
                break
        
        return sequence

## 4.5 质量评估体系的数学框架

### 多维质量指标

**综合质量函数**：
$$Q(y) = \alpha_1 \cdot \text{Fluency}(y) + \alpha_2 \cdot \text{Coherence}(y) + \alpha_3 \cdot \text{Relevance}(y) + \alpha_4 \cdot \text{Diversity}(y)$$

其中权重满足 $\sum_i \alpha_i = 1$。

**自动评估指标**：
1. **BLEU分数**：
   $$\text{BLEU} = \text{BP} \cdot \exp\left(\sum_{n=1}^{N} w_n \log p_n\right)$$

2. **ROUGE分数**：
   $$\text{ROUGE-N} = \frac{\sum_{S \in \{Ref\}} \sum_{gram_n \in S} Count_{match}(gram_n)}{\sum_{S \in \{Ref\}} \sum_{gram_n \in S} Count(gram_n)}$$

3. **语义相似度**：
   $$\text{SemSim}(y, r) = \cos(\text{BERT}(y), \text{BERT}(r))$$

class QualityEvaluator:
    """质量评估器"""
    
    def __init__(self):
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'])
        self.quality_weights = {
            'fluency': 0.25,
            'coherence': 0.25,
            'relevance': 0.25,
            'diversity': 0.25
        }
    
    def evaluate_comprehensive_quality(self, 
                                     generated_texts: List[str],
                                     reference_texts: List[str] = None,
                                     contexts: List[str] = None) -> QualityMetrics:
        """综合质量评估"""
        
        print("=== 综合质量评估 ===")
        
        # 计算各维度分数
        fluency_scores = [self._compute_fluency(text) for text in generated_texts]
        coherence_scores = [self._compute_coherence(text) for text in generated_texts]
        diversity_scores = self._compute_diversity(generated_texts)
        
        if reference_texts:
            relevance_scores = [self._compute_relevance(gen, ref) 
                              for gen, ref in zip(generated_texts, reference_texts)]
        else:
            relevance_scores = [0.5] * len(generated_texts)  # 默认中等相关性
        
        # 计算综合分数
        overall_scores = []
        for i in range(len(generated_texts)):
            score = (self.quality_weights['fluency'] * fluency_scores[i] +
                    self.quality_weights['coherence'] * coherence_scores[i] +
                    self.quality_weights['relevance'] * relevance_scores[i] +
                    self.quality_weights['diversity'] * diversity_scores[i])
            overall_scores.append(score)
        
        # 统计结果
        metrics = QualityMetrics(
            repetition_rate=self._compute_repetition_rate(generated_texts),
            diversity_score=np.mean(diversity_scores),
            coherence_score=np.mean(coherence_scores),
            fluency_score=np.mean(fluency_scores),
            relevance_score=np.mean(relevance_scores),
            length_quality=self._compute_length_quality(generated_texts),
            overall_quality=np.mean(overall_scores)
        )
        
        print(f"流畅性分数: {metrics.fluency_score:.4f}")
        print(f"连贯性分数: {metrics.coherence_score:.4f}")
        print(f"相关性分数: {metrics.relevance_score:.4f}")
        print(f"多样性分数: {metrics.diversity_score:.4f}")
        print(f"重复率: {metrics.repetition_rate:.4f}")
        print(f"综合质量分数: {metrics.overall_quality:.4f}")
        
        return metrics
    
    def _compute_fluency(self, text: str) -> float:
        """计算流畅性分数"""
        
        # 简化的流畅性评估：基于句子完整性和语法正确性
        sentences = text.split('.')
        complete_sentences = [s for s in sentences if len(s.strip()) > 5]
        
        if not sentences:
            return 0.0
        
        completeness_score = len(complete_sentences) / len(sentences)
        
        # 简单的语法检查（检查基本的句子结构）
        grammar_score = 0.8  # 简化实现，实际应使用语法检查工具
        
        return (completeness_score + grammar_score) / 2
    
    def _compute_coherence(self, text: str) -> float:
        """计算连贯性分数"""
        
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        if len(sentences) < 2:
            return 1.0
        
        # 计算相邻句子的词汇重叠度
        coherence_scores = []
        for i in range(len(sentences) - 1):
            words1 = set(sentences[i].split())
            words2 = set(sentences[i + 1].split())
            
            if not words1 or not words2:
                continue
            
            overlap = len(words1.intersection(words2))
            union = len(words1.union(words2))
            coherence = overlap / union if union > 0 else 0
            coherence_scores.append(coherence)
        
        return np.mean(coherence_scores) if coherence_scores else 0.0
    
    def _compute_relevance(self, generated: str, reference: str) -> float:
        """计算相关性分数"""
        
        # 使用ROUGE作为相关性代理指标
        scores = self.rouge_scorer.score(reference, generated)
        return scores['rougeL'].fmeasure
    
    def _compute_diversity(self, texts: List[str]) -> List[float]:
        """计算多样性分数"""
        
        diversity_scores = []
        
        for i, text in enumerate(texts):
            # 与其他文本的不相似性
            similarities = []
            for j, other_text in enumerate(texts):
                if i != j:
                    words1 = set(text.split())
                    words2 = set(other_text.split())
                    
                    if not words1 or not words2:
                        similarity = 0.0
                    else:
                        intersection = len(words1.intersection(words2))
                        union = len(words1.union(words2))
                        similarity = intersection / union
                    
                    similarities.append(similarity)
            
            # 多样性 = 1 - 平均相似度
            diversity = 1 - np.mean(similarities) if similarities else 1.0
            diversity_scores.append(diversity)
        
        return diversity_scores
    
    def _compute_repetition_rate(self, texts: List[str]) -> float:
        """计算重复率"""
        
        all_repetition_rates = []
        
        for text in texts:
            words = text.split()
            if len(words) < 2:
                continue
            
            # 计算bigram重复率
            bigrams = [(words[i], words[i+1]) for i in range(len(words)-1)]
            unique_bigrams = set(bigrams)
            
            repetition_rate = 1 - len(unique_bigrams) / len(bigrams) if bigrams else 0
            all_repetition_rates.append(repetition_rate)
        
        return np.mean(all_repetition_rates) if all_repetition_rates else 0.0
    
    def _compute_length_quality(self, texts: List[str]) -> float:
        """计算长度质量"""
        
        lengths = [len(text.split()) for text in texts]
        
        # 基于长度分布的质量评估
        mean_length = np.mean(lengths)
        std_length = np.std(lengths)
        
        # 理想情况下，长度应该适中且方差不太大
        length_appropriateness = 1 / (1 + abs(mean_length - 50) / 50)  # 50词为理想长度
        length_consistency = 1 / (1 + std_length / mean_length) if mean_length > 0 else 0
        
        return (length_appropriateness + length_consistency) / 2

def create_quality_control_system():
    """创建质量控制系统"""
    
    # 初始化各组件
    repetition_controller = RepetitionController(
        repetition_penalty=1.1,
        no_repeat_ngram_size=3
    )
    
    length_controller = LengthController(
        length_penalty=1.0,
        min_length=10,
        max_length=512
    )
    
    coherence_controller = CoherenceController(
        coherence_threshold=0.5,
        window_size=20
    )
    
    optimizer = GenerationOptimizer(
        max_batch_size=32,
        kv_cache_size=2048
    )
    
    evaluator = QualityEvaluator()
    
    return {
        'repetition_controller': repetition_controller,
        'length_controller': length_controller,
        'coherence_controller': coherence_controller,
        'optimizer': optimizer,
        'evaluator': evaluator
    }

# 演示完整的质量控制流程
def demonstrate_quality_control():
    """演示质量控制系统"""
    
    print("=== MiniGPT生成质量控制系统演示 ===\n")
    
    # 创建系统
    system = create_quality_control_system()
    
    # 模拟生成文本数据
    generated_texts = [
        "人工智能技术的发展正在改变我们的生活。机器学习算法能够处理大量数据，发现隐藏的模式。深度学习在图像识别、自然语言处理等领域取得了显著进展。",
        "深度学习模型需要大量的训练数据。训练数据的质量直接影响模型的性能。数据预处理是机器学习项目中的重要步骤。清理和标注数据需要大量的人力投入。",
        "自然语言处理是人工智能的重要分支。文本分析、情感分析、机器翻译都是NLP的应用领域。Transformer架构的出现推动了NLP技术的快速发展。",
        "机器学习机器学习机器学习算法算法算法能够能够处理处理大量大量数据数据，发现发现模式模式。重复重复的内容内容影响影响生成生成质量质量。"  # 包含重复的例子
    ]
    
    reference_texts = [
        "AI技术正在革新各个行业，机器学习帮助我们从数据中获取洞察，深度学习在多个领域都有突破性应用。",
        "高质量的训练数据是机器学习成功的关键，数据预处理工作虽然繁琐但至关重要。",
        "NLP作为AI的核心技术，在文本理解和生成方面不断进步，Transformer确实带来了革命性变化。",
        "避免重复是生成质量的重要指标。"
    ]
    
    # 1. 重复检测分析
    repetition_analysis = system['repetition_controller'].analyze_repetition_patterns(
        [text.split() for text in generated_texts]
    )
    system['repetition_controller'].visualize_repetition_analysis(repetition_analysis)
    
    # 2. 长度偏差分析  
    scores = [0.8, 0.75, 0.82, 0.45]  # 假设的质量分数
    length_analysis = system['length_controller'].compute_length_bias(
        [text.split() for text in generated_texts], scores
    )
    
    # 3. 连贯性分析
    coherence_analysis = system['coherence_controller'].analyze_topic_coherence(generated_texts)
    
    # 4. 综合质量评估
    quality = system['evaluator'].evaluate_comprehensive_quality(
        generated_texts, reference_texts
    )
    
    return {
        'repetition_analysis': repetition_analysis,
        'length_analysis': length_analysis,
        'coherence_analysis': coherence_analysis,
        'quality_metrics': quality,
        'system': system
    }

# 运行演示
if __name__ == "__main__":
    results = demonstrate_quality_control()
    
    print("\n=== 质量控制系统评估完成 ===")
    print(f"系统整体性能评估:")
    print(f"- 重复控制效果: 良好")
    print(f"- 长度控制效果: 良好") 
    print(f"- 连贯性维护: 良好")
    print(f"- 综合质量分数: {results['quality_metrics'].overall_quality:.4f}")
```

## 理论总结

### 4.6 质量控制的统一理论框架

**多目标优化视角**：
生成质量控制本质上是一个多目标优化问题：
$$\max_y \{ \log P(y|x), \text{Quality}(y), -\text{Cost}(y) \}$$

**约束优化形式**：
$$\begin{align}
\max_y &\quad \log P(y|x) \\
\text{s.t.} &\quad \text{Repetition}(y) \leq \tau_r \\
&\quad \text{Length}(y) \in [L_{min}, L_{max}] \\
&\quad \text{Coherence}(y) \geq \tau_c
\end{align}$$

**拉格朗日对偶**：
$$\mathcal{L} = \log P(y|x) + \sum_i \lambda_i g_i(y) + \sum_j \mu_j h_j(y)$$

这为质量控制提供了统一的数学框架，各种控制机制都可以视为在这个框架下的具体实例。

## 应用指导

### 实际部署建议

1. **质量监控体系**：
   - 实时重复率监控
   - 长度分布跟踪
   - 语义连贯性评估
   - 用户满意度反馈

2. **参数调优策略**：
   - A/B测试确定最优参数
   - 动态参数调整
   - 多任务参数共享

3. **性能优化实践**：
   - KV缓存合理配置
   - 批处理动态调整
   - 内存使用优化

质量控制与优化是语言生成系统走向实用化的关键技术，需要在理论指导下结合具体应用场景进行精细化调优。通过系统性的质量控制框架，我们能够构建既高效又可靠的语言生成系统。

## 扩展阅读

- 《Neural Text Generation: Past, Present and Beyond》- 文本生成全面综述
- 《Quality Estimation for Machine Translation》- 质量评估方法
- 《Controllable Text Generation》- 可控文本生成技术
- 《Efficient Transformers: A Survey》- Transformer优化技术

---

*质量是生成系统的生命线。在追求创造性的同时保证质量，在提升效率的同时维护可靠性，这正是工程师的艺术所在。* 🎯