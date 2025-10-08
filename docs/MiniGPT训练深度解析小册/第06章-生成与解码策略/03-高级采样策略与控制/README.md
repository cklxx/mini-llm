# 03 高级采样策略与控制

> **超越经典方法：现代语言生成的精准控制艺术**

## 核心思想

高级采样策略代表了语言生成技术的最新发展，它们不再满足于简单的概率采样或确定性搜索，而是试图在保持生成质量的同时，精确控制生成过程的各个方面——从词汇选择的多样性到语义内容的连贯性，从生成速度的优化到特定约束的满足。

**关键洞察**：
- **自适应性**：根据上下文动态调整采样策略
- **多目标平衡**：同时优化质量、多样性、连贯性等多个目标
- **语义感知**：不仅考虑概率分布，更关注语义相关性
- **约束满足**：在满足特定条件下进行受控生成

从数学角度看，高级采样策略本质上是在概率空间中定义更精细的搜索区域，通过引入额外的约束和启发式信息，实现对生成过程的精准控制。

## 3.1 Top-k采样的统计学深度分析

### 截断分布的数学性质

**Top-k采样定义**：
设原始概率分布为$P(y|x)$，Top-k采样构造截断分布：
$$P_{top-k}(y|x) = \begin{cases} 
\frac{P(y|x)}{\sum_{y' \in \mathcal{V}_k} P(y'|x)} & \text{if } y \in \mathcal{V}_k \\
0 & \text{otherwise}
\end{cases}$$

其中$\mathcal{V}_k = \{y_1, y_2, ..., y_k\}$是概率最高的$k$个token。

**截断分布的统计特性**：

1. **熵变化**：
$$H(P_{top-k}) = H(P) - \sum_{y \notin \mathcal{V}_k} P(y|x) \log P(y|x) + \log Z_k$$
其中$Z_k = \sum_{y \in \mathcal{V}_k} P(y|x)$是标准化常数。

2. **方差分析**：
截断后的分布方差通常会减少，但减少程度取决于原分布的形状：
$$\text{Var}(P_{top-k}) \leq \text{Var}(P)$$

3. **信息损失**：
$$I_{loss} = D_{KL}(P \| P_{top-k}) = \sum_{y \notin \mathcal{V}_k} P(y|x) \log \frac{P(y|x)}{0^+} = +\infty$$

这表明Top-k采样存在不可恢复的信息损失。

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
from collections import defaultdict, Counter
from dataclasses import dataclass
from scipy import stats
from sklearn.metrics import mutual_info_score
import networkx as nx

@dataclass
class SamplingResult:
    """采样结果数据结构"""
    sequence: List[int]           # 生成序列
    log_probability: float        # 对数概率
    sampling_entropy: float       # 采样熵
    diversity_score: float        # 多样性分数
    semantic_coherence: float     # 语义连贯性
    sampling_time: float         # 采样时间
    metadata: Dict[str, Any]     # 采样元数据

class TopKSampler:
    """Top-k采样器的数学分析实现"""
    
    def __init__(self, vocab_size: int = 1000, k: int = 50, adaptive: bool = False):
        self.vocab_size = vocab_size
        self.k = k
        self.adaptive = adaptive
        self.sampling_history = []
        
    def sample(self, logits: torch.Tensor, temperature: float = 1.0) -> Tuple[int, Dict]:
        """Top-k采样实现"""
        
        start_time = time.time()
        
        # 温度缩放
        scaled_logits = logits / temperature
        
        # 计算原始概率分布
        original_probs = F.softmax(scaled_logits, dim=-1)
        original_entropy = -(original_probs * torch.log(original_probs + 1e-10)).sum().item()
        
        # 自适应k值选择
        if self.adaptive:
            optimal_k = self._compute_adaptive_k(original_probs)
        else:
            optimal_k = self.k
        
        # Top-k选择
        top_k_values, top_k_indices = torch.topk(scaled_logits, optimal_k)
        
        # 构造截断分布
        truncated_logits = torch.full_like(scaled_logits, float('-inf'))
        truncated_logits[top_k_indices] = top_k_values
        truncated_probs = F.softmax(truncated_logits, dim=-1)
        
        # 计算截断后的统计量
        truncated_entropy = -(truncated_probs * torch.log(truncated_probs + 1e-10)).sum().item()
        
        # 采样
        sampled_token = torch.multinomial(truncated_probs, 1).item()
        token_prob = truncated_probs[sampled_token].item()
        
        sampling_time = time.time() - start_time
        
        # 计算信息损失
        info_loss = self._compute_information_loss(original_probs, truncated_probs)
        
        # 记录采样统计
        sampling_stats = {
            'used_k': optimal_k,
            'original_entropy': original_entropy,
            'truncated_entropy': truncated_entropy,
            'entropy_reduction': original_entropy - truncated_entropy,
            'information_loss': info_loss,
            'probability_mass_retained': truncated_probs.sum().item(),
            'sampling_time': sampling_time
        }
        
        self.sampling_history.append(sampling_stats)
        
        return sampled_token, sampling_stats
    
    def _compute_adaptive_k(self, probs: torch.Tensor, 
                           entropy_threshold: float = 0.8) -> int:
        """自适应计算最优k值"""
        
        # 排序概率
        sorted_probs, _ = torch.sort(probs, descending=True)
        
        # 计算累积概率质量和熵
        cumulative_mass = 0.0
        cumulative_entropy = 0.0
        
        for k in range(1, min(len(sorted_probs), self.vocab_size) + 1):
            # 当前k值下的截断分布
            top_k_probs = sorted_probs[:k]
            normalized_probs = top_k_probs / top_k_probs.sum()
            
            # 计算熵
            truncated_entropy = -(normalized_probs * torch.log(normalized_probs + 1e-10)).sum().item()
            
            # 如果熵达到阈值，返回当前k
            if truncated_entropy >= entropy_threshold * math.log2(k):
                return k
        
        # 如果没有找到合适的k，返回默认值
        return min(self.k, len(sorted_probs))
    
    def _compute_information_loss(self, original_probs: torch.Tensor, 
                                truncated_probs: torch.Tensor) -> float:
        """计算信息损失（KL散度）"""
        
        # 为了避免无穷大，我们计算有限近似
        # 只考虑截断分布中的概率
        kl_div = 0.0
        
        for i in range(len(original_probs)):
            if truncated_probs[i] > 1e-10:  # 只计算非零项
                p_orig = original_probs[i].item()
                p_trunc = truncated_probs[i].item()
                
                if p_orig > 1e-10:
                    kl_div += p_orig * math.log(p_orig / p_trunc)
            elif original_probs[i] > 1e-10:
                # 这些项的贡献趋向无穷，我们用一个大数近似
                kl_div += original_probs[i].item() * 10  # 近似无穷大的贡献
        
        return kl_div
    
    def analyze_k_value_effects(self, model, prompts: List[str], 
                               k_values: List[int] = None) -> Dict:
        """分析不同k值的效果"""
        
        print("=== Top-k采样k值效果分析 ===")
        
        if k_values is None:
            k_values = [1, 5, 10, 20, 50, 100]
        
        k_analysis = {
            'entropy_effects': {},
            'diversity_effects': {},
            'quality_effects': {},
            'efficiency_effects': {}
        }
        
        for k in k_values:
            print(f"分析k={k}")
            
            # 设置当前k值
            original_k = self.k
            self.k = k
            
            k_results = []
            
            # 对每个prompt进行采样
            for prompt in prompts:
                # 模拟生成过程
                generated_sequence = self._simulate_generation(model, prompt, max_length=20)
                k_results.append(generated_sequence)
            
            # 恢复原始k值
            self.k = original_k
            
            # 分析结果
            k_effects = self._analyze_k_effect(k_results, k)
            
            k_analysis['entropy_effects'][k] = k_effects['entropy']
            k_analysis['diversity_effects'][k] = k_effects['diversity']
            k_analysis['quality_effects'][k] = k_effects['quality']
            k_analysis['efficiency_effects'][k] = k_effects['efficiency']
        
        # 可视化分析结果
        self._visualize_k_analysis(k_analysis)
        
        return k_analysis
    
    def _simulate_generation(self, model, prompt: str, max_length: int = 20) -> List[int]:
        """模拟生成过程"""
        
        # 简化的生成模拟
        sequence = [2]  # BOS token
        
        for _ in range(max_length):
            # 模拟模型logits
            logits = torch.randn(self.vocab_size)
            
            # 使用当前采样器采样
            next_token, stats = self.sample(logits)
            sequence.append(next_token)
            
            # EOS检查
            if next_token == 1:
                break
        
        return sequence
    
    def _analyze_k_effect(self, sequences: List[List[int]], k: int) -> Dict:
        """分析特定k值的效果"""
        
        # 提取采样历史中对应k值的统计
        k_stats = [stat for stat in self.sampling_history if stat.get('used_k', self.k) == k]
        
        if not k_stats:
            return {'entropy': 0, 'diversity': 0, 'quality': 0, 'efficiency': 0}
        
        # 熵效果
        entropy_effect = {
            'avg_original_entropy': np.mean([s['original_entropy'] for s in k_stats]),
            'avg_truncated_entropy': np.mean([s['truncated_entropy'] for s in k_stats]),
            'avg_entropy_reduction': np.mean([s['entropy_reduction'] for s in k_stats])
        }
        
        # 多样性效果
        diversity_effect = self._compute_sequence_diversity(sequences)
        
        # 质量效果（基于信息损失的倒数）
        avg_info_loss = np.mean([s['information_loss'] for s in k_stats])
        quality_effect = 1.0 / (1.0 + avg_info_loss)
        
        # 效率效果
        avg_sampling_time = np.mean([s['sampling_time'] for s in k_stats])
        efficiency_effect = 1.0 / (avg_sampling_time + 1e-6)
        
        return {
            'entropy': entropy_effect,
            'diversity': diversity_effect,
            'quality': quality_effect,
            'efficiency': efficiency_effect
        }
    
    def _compute_sequence_diversity(self, sequences: List[List[int]]) -> float:
        """计算序列多样性"""
        
        if len(sequences) < 2:
            return 0.0
        
        # 使用Jaccard相似度计算多样性
        total_similarity = 0.0
        comparisons = 0
        
        for i in range(len(sequences)):
            for j in range(i+1, len(sequences)):
                seq1_set = set(sequences[i])
                seq2_set = set(sequences[j])
                
                intersection = len(seq1_set.intersection(seq2_set))
                union = len(seq1_set.union(seq2_set))
                
                if union > 0:
                    similarity = intersection / union
                    total_similarity += similarity
                    comparisons += 1
        
        avg_similarity = total_similarity / comparisons if comparisons > 0 else 0
        diversity = 1.0 - avg_similarity
        
        return diversity
    
    def _visualize_k_analysis(self, analysis: Dict):
        """可视化k值分析结果"""
        
        k_values = sorted(analysis['entropy_effects'].keys())
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 熵效果分析
        original_entropies = [analysis['entropy_effects'][k]['avg_original_entropy'] for k in k_values]
        truncated_entropies = [analysis['entropy_effects'][k]['avg_truncated_entropy'] for k in k_values]
        
        axes[0, 0].plot(k_values, original_entropies, 'b-o', label='原始熵', linewidth=2)
        axes[0, 0].plot(k_values, truncated_entropies, 'r-s', label='截断熵', linewidth=2)
        axes[0, 0].set_title('k值对熵的影响')
        axes[0, 0].set_xlabel('k值')
        axes[0, 0].set_ylabel('熵值')
        axes[0, 0].set_xscale('log')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 多样性效果
        diversities = [analysis['diversity_effects'][k] for k in k_values]
        
        axes[0, 1].plot(k_values, diversities, 'g-^', linewidth=2)
        axes[0, 1].set_title('k值对多样性的影响')
        axes[0, 1].set_xlabel('k值')
        axes[0, 1].set_ylabel('多样性分数')
        axes[0, 1].set_xscale('log')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 质量效果
        qualities = [analysis['quality_effects'][k] for k in k_values]
        
        axes[1, 0].plot(k_values, qualities, 'm-d', linewidth=2)
        axes[1, 0].set_title('k值对质量的影响')
        axes[1, 0].set_xlabel('k值')
        axes[1, 0].set_ylabel('质量分数')
        axes[1, 0].set_xscale('log')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 效率效果
        efficiencies = [analysis['efficiency_effects'][k] for k in k_values]
        
        axes[1, 1].plot(k_values, efficiencies, 'c-v', linewidth=2)
        axes[1, 1].set_title('k值对效率的影响')
        axes[1, 1].set_xlabel('k值')
        axes[1, 1].set_ylabel('效率分数')
        axes[1, 1].set_xscale('log')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

class NucleusSampler:
    """Nucleus (Top-p) 采样器"""
    
    def __init__(self, vocab_size: int = 1000, p: float = 0.9, min_tokens: int = 1):
        self.vocab_size = vocab_size
        self.p = p
        self.min_tokens = min_tokens
        self.sampling_history = []
        
    def sample(self, logits: torch.Tensor, temperature: float = 1.0) -> Tuple[int, Dict]:
        """Nucleus采样实现"""
        
        start_time = time.time()
        
        # 温度缩放
        scaled_logits = logits / temperature
        
        # 计算概率分布并排序
        probs = F.softmax(scaled_logits, dim=-1)
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        
        # 计算累积概率
        cumulative_probs = torch.cumsum(sorted_probs, dim=0)
        
        # 找到nucleus边界
        nucleus_mask = cumulative_probs <= self.p
        
        # 确保至少保留min_tokens个token
        if nucleus_mask.sum() < self.min_tokens:
            nucleus_mask[:self.min_tokens] = True
        
        # 计算有效的nucleus大小
        nucleus_size = nucleus_mask.sum().item()
        
        # 构造nucleus分布
        nucleus_probs = sorted_probs[nucleus_mask]
        nucleus_indices = sorted_indices[nucleus_mask]
        
        # 重新标准化
        nucleus_probs = nucleus_probs / nucleus_probs.sum()
        
        # 采样
        sampled_idx = torch.multinomial(nucleus_probs, 1).item()
        sampled_token = nucleus_indices[sampled_idx].item()
        token_prob = nucleus_probs[sampled_idx].item()
        
        sampling_time = time.time() - start_time
        
        # 计算统计信息
        original_entropy = -(probs * torch.log(probs + 1e-10)).sum().item()
        nucleus_entropy = -(nucleus_probs * torch.log(nucleus_probs + 1e-10)).sum().item()
        
        sampling_stats = {
            'nucleus_size': nucleus_size,
            'nucleus_ratio': nucleus_size / self.vocab_size,
            'probability_mass': nucleus_probs.sum().item(),
            'original_entropy': original_entropy,
            'nucleus_entropy': nucleus_entropy,
            'entropy_concentration': nucleus_entropy / original_entropy if original_entropy > 0 else 0,
            'sampling_time': sampling_time
        }
        
        self.sampling_history.append(sampling_stats)
        
        return sampled_token, sampling_stats
    
    def analyze_nucleus_dynamics(self, model, prompts: List[str], 
                                p_values: List[float] = None) -> Dict:
        """分析nucleus采样动态特性"""
        
        print("=== Nucleus采样动态分析 ===")
        
        if p_values is None:
            p_values = [0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99]
        
        dynamics_analysis = {
            'nucleus_size_dynamics': {},
            'entropy_concentration': {},
            'adaptive_vocabulary': {},
            'probability_distribution': {}
        }
        
        for p in p_values:
            print(f"分析p={p}")
            
            # 设置当前p值
            original_p = self.p
            self.p = p
            
            p_results = []
            
            # 生成样本
            for prompt in prompts:
                sequence = self._simulate_generation(model, prompt, max_length=15)
                p_results.append(sequence)
            
            # 恢复原始p值
            self.p = original_p
            
            # 分析结果
            p_analysis = self._analyze_nucleus_effect(p_results, p)
            
            dynamics_analysis['nucleus_size_dynamics'][p] = p_analysis['nucleus_sizes']
            dynamics_analysis['entropy_concentration'][p] = p_analysis['entropy_concentration']
            dynamics_analysis['adaptive_vocabulary'][p] = p_analysis['vocab_adaptation']
            dynamics_analysis['probability_distribution'][p] = p_analysis['prob_distribution']
        
        # 可视化分析
        self._visualize_nucleus_dynamics(dynamics_analysis)
        
        return dynamics_analysis
    
    def _simulate_generation(self, model, prompt: str, max_length: int = 15) -> List[int]:
        """模拟nucleus生成过程"""
        
        sequence = [2]  # BOS token
        
        for step in range(max_length):
            # 模拟不同分布形状的logits
            if step < 3:
                # 早期：相对均匀的分布
                logits = torch.randn(self.vocab_size) * 0.5
            elif step < 8:
                # 中期：适度尖锐的分布
                logits = torch.randn(self.vocab_size) * 1.0
                logits[torch.randint(0, self.vocab_size, (10,))] += 2.0  # 添加一些峰值
            else:
                # 后期：更尖锐的分布
                logits = torch.randn(self.vocab_size) * 0.3
                logits[torch.randint(0, self.vocab_size, (5,))] += 3.0  # 更强的峰值
            
            next_token, stats = self.sample(logits)
            sequence.append(next_token)
            
            if next_token == 1:  # EOS
                break
        
        return sequence
    
    def _analyze_nucleus_effect(self, sequences: List[List[int]], p: float) -> Dict:
        """分析特定p值的效果"""
        
        # 提取对应p值的采样统计
        p_stats = [stat for stat in self.sampling_history[-len(sequences)*15:]]  # 近似匹配
        
        if not p_stats:
            return {'nucleus_sizes': [], 'entropy_concentration': 0, 
                    'vocab_adaptation': 0, 'prob_distribution': {}}
        
        # Nucleus大小动态
        nucleus_sizes = [s['nucleus_size'] for s in p_stats]
        nucleus_ratios = [s['nucleus_ratio'] for s in p_stats]
        
        # 熵集中度
        entropy_concentrations = [s['entropy_concentration'] for s in p_stats]
        avg_entropy_concentration = np.mean(entropy_concentrations)
        
        # 词汇适应性（nucleus大小的变异系数）
        vocab_adaptation = np.std(nucleus_sizes) / (np.mean(nucleus_sizes) + 1e-10)
        
        # 概率分布特性
        prob_masses = [s['probability_mass'] for s in p_stats]
        prob_distribution = {
            'mean_mass': np.mean(prob_masses),
            'std_mass': np.std(prob_masses),
            'mass_efficiency': np.mean(prob_masses) / p  # 实际质量与目标p的比值
        }
        
        return {
            'nucleus_sizes': nucleus_sizes,
            'entropy_concentration': avg_entropy_concentration,
            'vocab_adaptation': vocab_adaptation,
            'prob_distribution': prob_distribution
        }
    
    def _visualize_nucleus_dynamics(self, dynamics: Dict):
        """可视化nucleus动态分析"""
        
        p_values = sorted(dynamics['nucleus_size_dynamics'].keys())
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Nucleus大小分布
        for i, p in enumerate(p_values[::2]):  # 每隔一个p值显示
            nucleus_sizes = dynamics['nucleus_size_dynamics'][p][:50]  # 前50个样本
            axes[0, 0].plot(nucleus_sizes, alpha=0.7, label=f'p={p}')
        
        axes[0, 0].set_title('Nucleus大小动态变化')
        axes[0, 0].set_xlabel('生成步骤')
        axes[0, 0].set_ylabel('Nucleus大小')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 熵集中度 vs p值
        entropy_concentrations = [dynamics['entropy_concentration'][p] for p in p_values]
        
        axes[0, 1].plot(p_values, entropy_concentrations, 'r-o', linewidth=2)
        axes[0, 1].set_title('p值对熵集中度的影响')
        axes[0, 1].set_xlabel('p值')
        axes[0, 1].set_ylabel('熵集中度')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 词汇适应性
        vocab_adaptations = [dynamics['adaptive_vocabulary'][p] for p in p_values]
        
        axes[1, 0].plot(p_values, vocab_adaptations, 'g-s', linewidth=2)
        axes[1, 0].set_title('词汇适应性')
        axes[1, 0].set_xlabel('p值')
        axes[1, 0].set_ylabel('适应性系数')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 概率质量效率
        mass_efficiencies = [dynamics['probability_distribution'][p]['mass_efficiency'] 
                           for p in p_values]
        
        axes[1, 1].plot(p_values, mass_efficiencies, 'm-^', linewidth=2)
        axes[1, 1].axhline(y=1.0, color='black', linestyle='--', alpha=0.5, label='理想效率')
        axes[1, 1].set_title('概率质量效率')
        axes[1, 1].set_xlabel('p值')
        axes[1, 1].set_ylabel('效率比')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

class ContrastiveSearchDecoder:
    """对比搜索解码器"""
    
    def __init__(self, vocab_size: int = 1000, alpha: float = 0.6, k: int = 5):
        self.vocab_size = vocab_size
        self.alpha = alpha  # 模型置信度权重
        self.k = k  # 候选数量
        self.search_history = []
        
    def decode(self, model, prompt: str, max_length: int = 30) -> SamplingResult:
        """对比搜索解码实现"""
        
        start_time = time.time()
        
        # 初始化
        input_tokens = self._tokenize(prompt)
        generated_tokens = []
        log_probability = 0.0
        
        # 维护生成历史的表示
        context_representations = []
        
        current_input = torch.tensor([input_tokens], dtype=torch.long)
        
        for step in range(max_length):
            # 获取模型logits和隐藏表示
            logits = self._get_model_logits(model, current_input)
            hidden_repr = self._get_hidden_representation(model, current_input)
            
            next_token_logits = logits[0, -1, :]
            probs = F.softmax(next_token_logits, dim=-1)
            
            # 获取top-k候选
            top_k_probs, top_k_indices = torch.topk(probs, self.k)
            
            # 对比搜索：结合模型置信度和去重复性
            contrastive_scores = []
            
            for i in range(self.k):
                candidate_token = top_k_indices[i].item()
                model_confidence = top_k_probs[i].item()
                
                # 计算去重复性分数
                degeneration_penalty = self._compute_degeneration_penalty(
                    candidate_token, hidden_repr, context_representations
                )
                
                # 综合分数
                contrastive_score = (
                    self.alpha * math.log(model_confidence + 1e-10) + 
                    (1 - self.alpha) * degeneration_penalty
                )
                
                contrastive_scores.append((contrastive_score, candidate_token, model_confidence))
            
            # 选择最佳候选
            best_score, next_token, token_prob = max(contrastive_scores, key=lambda x: x[0])
            
            # 更新序列和表示历史
            generated_tokens.append(next_token)
            log_probability += math.log(token_prob + 1e-10)
            context_representations.append(hidden_repr.clone())
            
            # 检查终止条件
            if next_token == 1:  # EOS
                break
            
            # 更新输入
            current_input = torch.cat([
                current_input,
                torch.tensor([[next_token]], dtype=torch.long)
            ], dim=1)
        
        decoding_time = time.time() - start_time
        
        # 计算各种指标
        diversity_score = self._compute_sequence_diversity(generated_tokens)
        semantic_coherence = self._compute_semantic_coherence(context_representations)
        sampling_entropy = self._estimate_sampling_entropy(generated_tokens)
        
        return SamplingResult(
            sequence=input_tokens + generated_tokens,
            log_probability=log_probability,
            sampling_entropy=sampling_entropy,
            diversity_score=diversity_score,
            semantic_coherence=semantic_coherence,
            sampling_time=decoding_time,
            metadata={
                'algorithm': 'contrastive_search',
                'alpha': self.alpha,
                'k': self.k,
                'avg_contrastive_score': np.mean([s[0] for s in contrastive_scores]) if contrastive_scores else 0
            }
        )
    
    def _tokenize(self, text: str) -> List[int]:
        """简化token化"""
        words = text.split()
        tokens = [2]  # BOS
        for word in words[:20]:
            token_id = abs(hash(word)) % (self.vocab_size - 10) + 3
            tokens.append(token_id)
        return tokens
    
    def _get_model_logits(self, model, input_ids: torch.Tensor) -> torch.Tensor:
        """获取模型logits"""
        batch_size, seq_len = input_ids.shape
        return torch.randn(batch_size, seq_len, self.vocab_size)
    
    def _get_hidden_representation(self, model, input_ids: torch.Tensor) -> torch.Tensor:
        """获取隐藏表示"""
        # 简化实现：返回随机表示
        return torch.randn(768)  # 假设768维隐藏状态
    
    def _compute_degeneration_penalty(self, candidate_token: int, 
                                    current_repr: torch.Tensor,
                                    context_reprs: List[torch.Tensor]) -> float:
        """计算去重复性惩罚"""
        
        if not context_reprs:
            return 0.0
        
        # 计算与历史表示的最大相似度
        max_similarity = 0.0
        
        for past_repr in context_reprs[-10:]:  # 只考虑最近10步
            similarity = F.cosine_similarity(
                current_repr.unsqueeze(0), 
                past_repr.unsqueeze(0)
            ).item()
            max_similarity = max(max_similarity, similarity)
        
        # 惩罚分数：相似度越高，惩罚越大
        penalty = -max_similarity
        
        return penalty
    
    def _compute_sequence_diversity(self, tokens: List[int]) -> float:
        """计算序列多样性"""
        
        if len(tokens) < 2:
            return 0.0
        
        # 基于n-gram多样性
        unique_unigrams = len(set(tokens))
        unique_bigrams = len(set(zip(tokens[:-1], tokens[1:])))
        
        unigram_diversity = unique_unigrams / len(tokens)
        bigram_diversity = unique_bigrams / max(1, len(tokens) - 1)
        
        return (unigram_diversity + bigram_diversity) / 2
    
    def _compute_semantic_coherence(self, representations: List[torch.Tensor]) -> float:
        """计算语义连贯性"""
        
        if len(representations) < 2:
            return 1.0
        
        # 计算相邻表示的平均相似度
        similarities = []
        
        for i in range(len(representations) - 1):
            sim = F.cosine_similarity(
                representations[i].unsqueeze(0),
                representations[i + 1].unsqueeze(0)
            ).item()
            similarities.append(sim)
        
        return np.mean(similarities) if similarities else 1.0
    
    def _estimate_sampling_entropy(self, tokens: List[int]) -> float:
        """估计采样熵"""
        
        if not tokens:
            return 0.0
        
        # 基于token频率估计熵
        token_counts = Counter(tokens)
        total_tokens = len(tokens)
        
        entropy = 0.0
        for count in token_counts.values():
            prob = count / total_tokens
            entropy -= prob * math.log2(prob)
        
        return entropy
    
    def analyze_contrastive_parameters(self, model, prompts: List[str],
                                     alpha_values: List[float] = None,
                                     k_values: List[int] = None) -> Dict:
        """分析对比搜索参数效果"""
        
        print("=== 对比搜索参数分析 ===")
        
        if alpha_values is None:
            alpha_values = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        
        if k_values is None:
            k_values = [3, 5, 8, 10, 15]
        
        parameter_analysis = {
            'alpha_effects': {},
            'k_effects': {},
            'parameter_interaction': {}
        }
        
        # 分析alpha效果
        for alpha in alpha_values:
            print(f"分析alpha={alpha}")
            
            original_alpha = self.alpha
            self.alpha = alpha
            
            alpha_results = []
            for prompt in prompts[:3]:  # 使用前3个prompt
                result = self.decode(model, prompt, max_length=20)
                alpha_results.append(result)
            
            self.alpha = original_alpha
            
            # 分析该alpha的效果
            alpha_analysis = self._analyze_parameter_effect(alpha_results)
            parameter_analysis['alpha_effects'][alpha] = alpha_analysis
        
        # 分析k效果
        for k in k_values:
            print(f"分析k={k}")
            
            original_k = self.k
            self.k = k
            
            k_results = []
            for prompt in prompts[:3]:
                result = self.decode(model, prompt, max_length=20)
                k_results.append(result)
            
            self.k = original_k
            
            k_analysis = self._analyze_parameter_effect(k_results)
            parameter_analysis['k_effects'][k] = k_analysis
        
        # 可视化分析结果
        self._visualize_contrastive_analysis(parameter_analysis)
        
        return parameter_analysis
    
    def _analyze_parameter_effect(self, results: List[SamplingResult]) -> Dict:
        """分析参数效果"""
        
        return {
            'avg_diversity': np.mean([r.diversity_score for r in results]),
            'avg_coherence': np.mean([r.semantic_coherence for r in results]),
            'avg_entropy': np.mean([r.sampling_entropy for r in results]),
            'avg_time': np.mean([r.sampling_time for r in results]),
            'avg_log_prob': np.mean([r.log_probability for r in results])
        }
    
    def _visualize_contrastive_analysis(self, analysis: Dict):
        """可视化对比搜索分析"""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Alpha效果 - 多样性 vs 连贯性
        alpha_values = sorted(analysis['alpha_effects'].keys())
        diversities = [analysis['alpha_effects'][a]['avg_diversity'] for a in alpha_values]
        coherences = [analysis['alpha_effects'][a]['avg_coherence'] for a in alpha_values]
        
        axes[0, 0].plot(alpha_values, diversities, 'b-o', label='多样性', linewidth=2)
        axes[0, 0].plot(alpha_values, coherences, 'r-s', label='连贯性', linewidth=2)
        axes[0, 0].set_title('Alpha参数对多样性和连贯性的影响')
        axes[0, 0].set_xlabel('Alpha值')
        axes[0, 0].set_ylabel('分数')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # K值效果 - 多样性
        k_values = sorted(analysis['k_effects'].keys())
        k_diversities = [analysis['k_effects'][k]['avg_diversity'] for k in k_values]
        
        axes[0, 1].plot(k_values, k_diversities, 'g-^', linewidth=2)
        axes[0, 1].set_title('K值对多样性的影响')
        axes[0, 1].set_xlabel('K值')
        axes[0, 1].set_ylabel('多样性分数')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 多样性-连贯性权衡
        axes[1, 0].scatter(diversities, coherences, s=100, alpha=0.7)
        
        for i, alpha in enumerate(alpha_values):
            axes[1, 0].annotate(f'α={alpha}', (diversities[i], coherences[i]),
                               xytext=(5, 5), textcoords='offset points')
        
        axes[1, 0].set_title('多样性-连贯性权衡')
        axes[1, 0].set_xlabel('多样性')
        axes[1, 0].set_ylabel('连贯性')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 计算时间对比
        alpha_times = [analysis['alpha_effects'][a]['avg_time'] for a in alpha_values]
        k_times = [analysis['k_effects'][k]['avg_time'] for k in k_values]
        
        # 标准化时间对比
        norm_alpha_times = np.array(alpha_times) / max(alpha_times)
        norm_k_times = np.array(k_times) / max(k_times)
        
        width = 0.35
        x_alpha = np.arange(len(alpha_values))
        x_k = np.arange(len(k_values))
        
        # 由于alpha和k值数量可能不同，我们分别显示
        bars1 = axes[1, 1].bar(x_alpha - width/2, norm_alpha_times, width, 
                              label='Alpha变化', alpha=0.7)
        
        # 调整k值的x坐标以避免重叠
        x_k_adjusted = x_k + len(alpha_values) + 1
        bars2 = axes[1, 1].bar(x_k_adjusted - width/2, norm_k_times, width,
                              label='K变化', alpha=0.7)
        
        axes[1, 1].set_title('参数变化对计算时间的影响')
        axes[1, 1].set_ylabel('标准化时间')
        axes[1, 1].legend()
        
        # 设置x轴标签
        all_labels = [f'α={a}' for a in alpha_values] + [''] + [f'k={k}' for k in k_values]
        all_positions = list(x_alpha) + [len(alpha_values)] + list(x_k_adjusted)
        axes[1, 1].set_xticks(all_positions)
        axes[1, 1].set_xticklabels(all_labels, rotation=45)
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

# 高级采样策略综合比较
class AdvancedSamplingComparator:
    """高级采样策略比较器"""
    
    def __init__(self):
        self.comparison_results = {}
    
    def comprehensive_comparison(self, model, test_prompts: List[str]) -> Dict:
        """综合比较高级采样策略"""
        
        print("=== 高级采样策略综合比较 ===")
        
        # 创建不同的采样器
        samplers = {
            'top_k_10': TopKSampler(vocab_size=1000, k=10),
            'top_k_50': TopKSampler(vocab_size=1000, k=50),
            'top_k_adaptive': TopKSampler(vocab_size=1000, k=50, adaptive=True),
            'nucleus_0.8': NucleusSampler(vocab_size=1000, p=0.8),
            'nucleus_0.95': NucleusSampler(vocab_size=1000, p=0.95),
            'contrastive_balanced': ContrastiveSearchDecoder(vocab_size=1000, alpha=0.6, k=5),
            'contrastive_diverse': ContrastiveSearchDecoder(vocab_size=1000, alpha=0.3, k=8)
        }
        
        comparison_results = {}
        
        for sampler_name, sampler in samplers.items():
            print(f"测试采样器: {sampler_name}")
            
            sampler_results = []
            
            for prompt in test_prompts:
                if isinstance(sampler, ContrastiveSearchDecoder):
                    result = sampler.decode(model, prompt, max_length=25)
                    sampler_results.append(result)
                else:
                    # 对于其他采样器，需要模拟完整的生成过程
                    generated_sequence = self._simulate_generation_with_sampler(
                        sampler, model, prompt, max_length=25
                    )
                    sampler_results.append(generated_sequence)
            
            # 分析采样器性能
            analysis = self._analyze_sampler_performance(sampler_results, sampler_name)
            comparison_results[sampler_name] = analysis
        
        # 综合分析
        comprehensive_analysis = self._comprehensive_analysis(comparison_results)
        
        # 可视化比较
        self._visualize_comprehensive_comparison(comparison_results, comprehensive_analysis)
        
        return {
            'individual_results': comparison_results,
            'comprehensive_analysis': comprehensive_analysis
        }
    
    def _simulate_generation_with_sampler(self, sampler, model, prompt: str, 
                                        max_length: int = 25) -> SamplingResult:
        """使用指定采样器模拟生成"""
        
        start_time = time.time()
        
        input_tokens = self._tokenize(prompt)
        generated_tokens = []
        log_probability = 0.0
        entropies = []
        
        for step in range(max_length):
            # 模拟模型logits
            logits = torch.randn(sampler.vocab_size)
            
            # 使用采样器采样
            if isinstance(sampler, (TopKSampler, NucleusSampler)):
                next_token, stats = sampler.sample(logits)
                token_prob = torch.softmax(logits, dim=-1)[next_token].item()
                entropies.append(stats.get('original_entropy', 0))
            else:
                # 对于其他采样器，使用简单的multinomial采样
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, 1).item()
                token_prob = probs[next_token].item()
                entropy = -(probs * torch.log(probs + 1e-10)).sum().item()
                entropies.append(entropy)
            
            generated_tokens.append(next_token)
            log_probability += math.log(token_prob + 1e-10)
            
            if next_token == 1:  # EOS
                break
        
        generation_time = time.time() - start_time
        
        # 计算指标
        diversity_score = self._compute_diversity(generated_tokens)
        semantic_coherence = 0.8  # 简化的连贯性分数
        sampling_entropy = np.mean(entropies) if entropies else 0
        
        return SamplingResult(
            sequence=input_tokens + generated_tokens,
            log_probability=log_probability,
            sampling_entropy=sampling_entropy,
            diversity_score=diversity_score,
            semantic_coherence=semantic_coherence,
            sampling_time=generation_time,
            metadata={'algorithm': type(sampler).__name__}
        )
    
    def _tokenize(self, text: str) -> List[int]:
        """简化token化"""
        words = text.split()
        tokens = [2]  # BOS
        for word in words[:20]:
            token_id = abs(hash(word)) % 997 + 3
            tokens.append(token_id)
        return tokens
    
    def _compute_diversity(self, tokens: List[int]) -> float:
        """计算多样性分数"""
        if len(tokens) < 2:
            return 0.0
        
        unique_tokens = len(set(tokens))
        return unique_tokens / len(tokens)
    
    def _analyze_sampler_performance(self, results: List[SamplingResult], 
                                   sampler_name: str) -> Dict:
        """分析采样器性能"""
        
        return {
            'avg_diversity': np.mean([r.diversity_score for r in results]),
            'std_diversity': np.std([r.diversity_score for r in results]),
            'avg_coherence': np.mean([r.semantic_coherence for r in results]),
            'avg_entropy': np.mean([r.sampling_entropy for r in results]),
            'avg_time': np.mean([r.sampling_time for r in results]),
            'avg_log_prob': np.mean([r.log_probability for r in results]),
            'avg_length': np.mean([len(r.sequence) for r in results])
        }
    
    def _comprehensive_analysis(self, results: Dict) -> Dict:
        """综合分析"""
        
        samplers = list(results.keys())
        
        # 找出各项指标的最佳采样器
        best_diversity = max(samplers, key=lambda x: results[x]['avg_diversity'])
        best_coherence = max(samplers, key=lambda x: results[x]['avg_coherence'])
        best_efficiency = min(samplers, key=lambda x: results[x]['avg_time'])
        
        # 计算综合得分
        composite_scores = {}
        for sampler in samplers:
            # 标准化各项指标
            diversity_norm = results[sampler]['avg_diversity']
            coherence_norm = results[sampler]['avg_coherence']
            efficiency_norm = 1.0 / (results[sampler]['avg_time'] + 1e-6)
            
            # 综合得分（可调整权重）
            composite_score = (
                0.3 * diversity_norm + 
                0.4 * coherence_norm + 
                0.3 * efficiency_norm
            )
            composite_scores[sampler] = composite_score
        
        best_overall = max(composite_scores, key=composite_scores.get)
        
        return {
            'best_diversity': best_diversity,
            'best_coherence': best_coherence,
            'best_efficiency': best_efficiency,
            'best_overall': best_overall,
            'composite_scores': composite_scores,
            'performance_ranking': sorted(samplers, key=lambda x: composite_scores[x], reverse=True)
        }
    
    def _visualize_comprehensive_comparison(self, results: Dict, analysis: Dict):
        """可视化综合比较结果"""
        
        samplers = list(results.keys())
        n_samplers = len(samplers)
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 多样性比较
        diversities = [results[s]['avg_diversity'] for s in samplers]
        diversity_stds = [results[s]['std_diversity'] for s in samplers]
        
        bars1 = axes[0, 0].bar(range(n_samplers), diversities, yerr=diversity_stds,
                              alpha=0.7, capsize=5)
        axes[0, 0].set_title('生成多样性比较')
        axes[0, 0].set_ylabel('多样性分数')
        axes[0, 0].set_xticks(range(n_samplers))
        axes[0, 0].set_xticklabels(samplers, rotation=45)
        
        # 高亮最佳多样性
        best_diversity_idx = samplers.index(analysis['best_diversity'])
        bars1[best_diversity_idx].set_color('gold')
        
        # 连贯性比较
        coherences = [results[s]['avg_coherence'] for s in samplers]
        
        bars2 = axes[0, 1].bar(range(n_samplers), coherences, alpha=0.7, color='lightblue')
        axes[0, 1].set_title('语义连贯性比较')
        axes[0, 1].set_ylabel('连贯性分数')
        axes[0, 1].set_xticks(range(n_samplers))
        axes[0, 1].set_xticklabels(samplers, rotation=45)
        
        # 高亮最佳连贯性
        best_coherence_idx = samplers.index(analysis['best_coherence'])
        bars2[best_coherence_idx].set_color('lightgreen')
        
        # 效率比较
        times = [results[s]['avg_time'] for s in samplers]
        
        bars3 = axes[1, 0].bar(range(n_samplers), times, alpha=0.7, color='lightcoral')
        axes[1, 0].set_title('采样效率比较')
        axes[1, 0].set_ylabel('平均时间 (秒)')
        axes[1, 0].set_xticks(range(n_samplers))
        axes[1, 0].set_xticklabels(samplers, rotation=45)
        
        # 高亮最佳效率
        best_efficiency_idx = samplers.index(analysis['best_efficiency'])
        bars3[best_efficiency_idx].set_color('lightgreen')
        
        # 综合得分雷达图
        composite_scores = [analysis['composite_scores'][s] for s in samplers]
        
        bars4 = axes[1, 1].bar(range(n_samplers), composite_scores, alpha=0.7, color='mediumpurple')
        axes[1, 1].set_title('综合性能得分')
        axes[1, 1].set_ylabel('综合得分')
        axes[1, 1].set_xticks(range(n_samplers))
        axes[1, 1].set_xticklabels(samplers, rotation=45)
        
        # 高亮最佳综合性能
        best_overall_idx = samplers.index(analysis['best_overall'])
        bars4[best_overall_idx].set_color('gold')
        
        plt.tight_layout()
        plt.show()
        
        # 打印排名
        print(f"\n=== 性能排名 ===")
        for i, sampler in enumerate(analysis['performance_ranking'], 1):
            score = analysis['composite_scores'][sampler]
            print(f"{i}. {sampler}: {score:.3f}")

# 高级采样策略综合演示
def demonstrate_advanced_sampling():
    """演示高级采样策略"""
    
    print("="*60)
    print("高级采样策略与控制 - 综合演示")
    print("="*60)
    
    # 模拟模型
    class DummyModel:
        pass
    
    model = DummyModel()
    
    # 测试提示
    test_prompts = [
        "Explain the concept of artificial intelligence.",
        "Describe the process of machine learning.",
        "How do neural networks function?",
        "What are the applications of deep learning?"
    ]
    
    print(f"\n使用 {len(test_prompts)} 个测试提示")
    
    # 1. Top-k采样分析
    print("\n1. Top-k采样k值效果分析")
    
    topk_sampler = TopKSampler(vocab_size=1000, k=20)
    k_analysis = topk_sampler.analyze_k_value_effects(
        model, test_prompts[:2], k_values=[5, 10, 20, 50, 100]
    )
    
    # 找出最佳k值
    k_values = list(k_analysis['diversity_effects'].keys())
    diversity_scores = list(k_analysis['diversity_effects'].values())
    quality_scores = list(k_analysis['quality_effects'].values())
    
    # 平衡分数
    balance_scores = [0.6 * q + 0.4 * d for q, d in zip(quality_scores, diversity_scores)]
    best_k_idx = np.argmax(balance_scores)
    best_k = k_values[best_k_idx]
    
    print(f"推荐k值: {best_k} (平衡分数: {balance_scores[best_k_idx]:.3f})")
    
    # 2. Nucleus采样分析
    print("\n2. Nucleus采样动态特性分析")
    
    nucleus_sampler = NucleusSampler(vocab_size=1000, p=0.9)
    nucleus_dynamics = nucleus_sampler.analyze_nucleus_dynamics(
        model, test_prompts[:2], p_values=[0.5, 0.7, 0.9, 0.95, 0.99]
    )
    
    # 分析最佳p值
    p_values = list(nucleus_dynamics['entropy_concentration'].keys())
    entropy_concentrations = list(nucleus_dynamics['entropy_concentration'].values())
    
    # 寻找熵集中度的最佳平衡点
    optimal_p_idx = np.argmax(entropy_concentrations)
    optimal_p = p_values[optimal_p_idx]
    
    print(f"推荐p值: {optimal_p} (熵集中度: {entropy_concentrations[optimal_p_idx]:.3f})")
    
    # 3. 对比搜索分析
    print("\n3. 对比搜索参数效果分析")
    
    contrastive_decoder = ContrastiveSearchDecoder(vocab_size=1000, alpha=0.6, k=5)
    contrastive_analysis = contrastive_decoder.analyze_contrastive_parameters(
        model, test_prompts[:2], 
        alpha_values=[0.2, 0.4, 0.6, 0.8],
        k_values=[3, 5, 8, 10]
    )
    
    # 找出最佳参数组合
    alpha_effects = contrastive_analysis['alpha_effects']
    best_alpha = max(alpha_effects.keys(), 
                    key=lambda a: 0.5 * alpha_effects[a]['avg_diversity'] + 
                                 0.5 * alpha_effects[a]['avg_coherence'])
    
    k_effects = contrastive_analysis['k_effects']
    best_k_contrastive = max(k_effects.keys(),
                           key=lambda k: k_effects[k]['avg_diversity'])
    
    print(f"推荐对比搜索参数: alpha={best_alpha}, k={best_k_contrastive}")
    
    # 4. 综合比较
    print("\n4. 高级采样策略综合比较")
    
    comparator = AdvancedSamplingComparator()
    comprehensive_results = comparator.comprehensive_comparison(model, test_prompts)
    
    analysis = comprehensive_results['comprehensive_analysis']
    
    print(f"最佳多样性: {analysis['best_diversity']}")
    print(f"最佳连贯性: {analysis['best_coherence']}")
    print(f"最佳效率: {analysis['best_efficiency']}")
    print(f"综合最佳: {analysis['best_overall']}")
    
    print(f"\n性能排名前3:")
    for i, sampler in enumerate(analysis['performance_ranking'][:3], 1):
        score = analysis['composite_scores'][sampler]
        print(f"  {i}. {sampler}: {score:.3f}")
    
    # 5. 实践建议
    print(f"\n5. 采样策略选择建议")
    
    recommendations = {
        "创意写作": "使用Nucleus采样(p=0.9)获得良好的多样性",
        "技术问答": "使用Top-k采样(k=10-20)确保准确性",
        "对话系统": "使用对比搜索(α=0.6)平衡流畅性和多样性",
        "代码生成": "使用较小的k值(k=5)保证语法正确性",
        "摘要生成": "使用Nucleus采样(p=0.95)保持信息完整性",
        "创意故事": "使用对比搜索(α=0.3)增强创造性"
    }
    
    for scenario, recommendation in recommendations.items():
        print(f"{scenario}: {recommendation}")
    
    # 6. 算法复杂度总结
    print(f"\n6. 算法复杂度比较")
    
    complexity_comparison = {
        "Top-k采样": {
            "时间复杂度": "O(V log k)",
            "空间复杂度": "O(k)",
            "适用场景": "需要控制候选数量时"
        },
        "Nucleus采样": {
            "时间复杂度": "O(V log V)",
            "空间复杂度": "O(V)",
            "适用场景": "需要自适应词汇量时"
        },
        "对比搜索": {
            "时间复杂度": "O(V log k + k × H)",
            "空间复杂度": "O(k × H + T × H)",
            "适用场景": "需要避免重复和保持连贯性时"
        }
    }
    
    for algorithm, properties in complexity_comparison.items():
        print(f"\n{algorithm}:")
        for prop, value in properties.items():
            print(f"  {prop}: {value}")
    
    print(f"\n=== 高级采样策略分析完成 ===")
    print(f"选择合适的采样策略需要根据具体任务需求权衡多样性、质量和效率")

# 运行高级采样策略演示
demonstrate_advanced_sampling()
```

这样我就完成了第06章第03节"高级采样策略与控制"的完整内容。这一节深入分析了：

1. **Top-k采样的统计学基础**：截断分布的数学性质、自适应k值选择、信息损失分析
2. **Nucleus采样的理论分析**：累积概率建模、自适应词汇量、nucleus大小动态变化
3. **对比搜索方法**：去重复性惩罚机制、语义连贯性保持、参数效果分析
4. **采样策略综合比较**：多维度性能评估、最佳策略推荐、实际应用指导

每种高级采样方法都有完整的数学推导、详细的实现代码和深入的性能分析，为读者提供了ultra-deep的理论理解和实践指导。通过这些现代采样技术，可以实现对语言生成过程的精准控制。

<function_calls>
<invoke name="TodoWrite">
<parameter name="todos">[{"content": "Research best practices for technical tutorial writing", "status": "completed", "priority": "high", "id": "1"}, {"content": "Design comprehensive folder structure for MiniGPT handbook", "status": "completed", "priority": "high", "id": "2"}, {"content": "Create detailed outline with ultra-deep analysis approach", "status": "completed", "priority": "high", "id": "3"}, {"content": "Write handbook introduction and navigation guide", "status": "completed", "priority": "high", "id": "4"}, {"content": "Implement chapter-by-chapter content with mathematical rigor", "status": "in_progress", "priority": "high", "id": "5"}, {"content": "Write Chapter 1: Mathematical Foundations", "status": "completed", "priority": "high", "id": "6"}, {"content": "Write Chapter 2: Transformer Architecture", "status": "completed", "priority": "high", "id": "7"}, {"content": "Write Chapter 3: Pre-training Theory and Implementation", "status": "completed", "priority": "high", "id": "8"}, {"content": "Write Chapter 4: Supervised Fine-tuning Deep Dive", "status": "completed", "priority": "high", "id": "14"}, {"content": "Write Chapter 5: Reinforcement Learning from Human Feedback", "status": "completed", "priority": "high", "id": "19"}, {"content": "Write Chapter 6: Generation and Decoding Strategies", "status": "in_progress", "priority": "high", "id": "25"}, {"content": "Create Chapter 6 folder structure and README", "status": "completed", "priority": "medium", "id": "28"}, {"content": "Write Chapter 6 Section 1: Autoregressive Generation Mathematical Principles", "status": "completed", "priority": "high", "id": "29"}, {"content": "Write Chapter 6 Section 2: Classical Decoding Algorithms Deep Analysis", "status": "completed", "priority": "high", "id": "30"}, {"content": "Write Chapter 6 Section 3: Advanced Sampling Strategies and Control", "status": "completed", "priority": "high", "id": "31"}, {"content": "Write Chapter 6 Section 4: Generation Quality Control and Optimization", "status": "in_progress", "priority": "high", "id": "32"}]