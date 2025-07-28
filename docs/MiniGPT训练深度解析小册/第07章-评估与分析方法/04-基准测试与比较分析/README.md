# 04 基准测试与比较分析

> **从单点评估到系统比较：构建科学的模型评估基准**

## 核心思想

基准测试是语言模型研究的重要基础设施，它为不同模型的性能比较提供了标准化的测试环境和评估框架。一个好的基准不仅要能够准确反映模型的真实能力，还要具有区分度、稳定性和公平性。比较分析则需要运用严格的统计学方法，确保得出的结论具有科学性和可靠性。

**关键洞察**：
- **标准化评估**：统一的测试环境确保比较的公平性
- **统计显著性**：严格的假设检验避免偶然差异的误判
- **多维度比较**：全面的性能剖析揭示模型的优势和劣势
- **元评估视角**：评估方法本身也需要被评估和验证

从数学角度看，基准测试是在标准化测试空间中建立模型性能的偏序关系，而比较分析则是通过统计推断确定这种偏序关系的可靠性。

## 4.1 基准数据集构建的统计学原理

### 数据集代表性的数学保证

**总体与样本的关系**：
设真实任务分布为 $\mathcal{D}$，基准数据集为样本 $S = \{(x_i, y_i)\}_{i=1}^n \sim \mathcal{D}$，则评估误差：
$$\text{Error}(S) = |\mathbb{E}_{(x,y) \sim \mathcal{D}}[f(x,y)] - \frac{1}{n}\sum_{i=1}^n f(x_i, y_i)|$$

**样本复杂度理论**：
对于 $\epsilon$-准确和 $\delta$-置信的估计，所需样本数量为：
$$n \geq \frac{2\log(2/\delta)}{\epsilon^2}$$

**分层采样策略**：
$$P(\text{sample } x | \text{stratum } s) = \frac{n_s}{N_s} \cdot \frac{N}{n}$$

其中 $n_s, N_s$ 分别是第 $s$ 层的样本数和总数。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from dataclasses import dataclass, field
from collections import defaultdict, Counter
from scipy import stats
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.metrics import cohen_kappa_score
from statsmodels.stats.contingency_tables import mcnemar
import warnings
warnings.filterwarnings('ignore')

@dataclass
class BenchmarkDataset:
    """基准数据集数据结构"""
    name: str
    domain: str
    task_type: str  # classification, generation, etc.
    samples: List[Dict[str, Any]]
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        self.size = len(self.samples)
        self.metadata['creation_date'] = self.metadata.get('creation_date', 'unknown')
        self.metadata['version'] = self.metadata.get('version', '1.0')

@dataclass
class ModelResult:
    """模型结果数据结构"""
    model_name: str
    dataset_name: str
    scores: Dict[str, float]
    predictions: List[Any]
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = ""

class BenchmarkConstructor:
    """基准数据集构造器"""
    
    def __init__(self):
        self.quality_criteria = {
            'representativeness': 0.8,
            'difficulty_distribution': 0.7,
            'annotation_quality': 0.9,
            'diversity': 0.75
        }
        
    def construct_benchmark(self, 
                          raw_data: List[Dict],
                          stratification_key: str,
                          target_size: int = 1000,
                          validation_split: float = 0.2) -> BenchmarkDataset:
        """构造基准数据集"""
        
        print(f"=== 构造基准数据集 (目标大小: {target_size}) ===")
        
        # 数据质量过滤
        filtered_data = self._quality_filter(raw_data)
        print(f"质量过滤后保留 {len(filtered_data)}/{len(raw_data)} 样本")
        
        # 分层采样
        stratified_data = self._stratified_sampling(
            filtered_data, stratification_key, target_size
        )
        print(f"分层采样得到 {len(stratified_data)} 样本")
        
        # 难度分布平衡
        balanced_data = self._balance_difficulty_distribution(stratified_data)
        print(f"难度平衡后 {len(balanced_data)} 样本")
        
        # 划分训练/验证集
        train_data, val_data = train_test_split(
            balanced_data, test_size=validation_split, 
            stratify=[item[stratification_key] for item in balanced_data],
            random_state=42
        )
        
        # 创建基准数据集
        benchmark = BenchmarkDataset(
            name=f"constructed_benchmark_{target_size}",
            domain="general",
            task_type="mixed",
            samples=train_data,
            metadata={
                'validation_samples': val_data,
                'stratification_key': stratification_key,
                'construction_stats': self._compute_construction_stats(train_data, val_data)
            }
        )
        
        return benchmark
    
    def _quality_filter(self, data: List[Dict]) -> List[Dict]:
        """数据质量过滤"""
        
        filtered = []
        
        for item in data:
            quality_score = 0
            
            # 文本长度检查
            if 'text' in item and 10 <= len(item['text'].split()) <= 500:
                quality_score += 0.3
            
            # 标注一致性检查（如果有多个标注者）
            if 'annotations' in item and len(item['annotations']) >= 2:
                annotations = item['annotations']
                consistency = self._compute_annotation_consistency(annotations)
                if consistency > 0.7:
                    quality_score += 0.4
            else:
                quality_score += 0.4  # 单标注者默认通过
            
            # 内容多样性检查
            if 'text' in item:
                diversity_score = self._compute_text_diversity(item['text'])
                quality_score += 0.3 * diversity_score
            
            # 质量阈值
            if quality_score >= 0.6:
                item['quality_score'] = quality_score
                filtered.append(item)
        
        return filtered
    
    def _compute_annotation_consistency(self, annotations: List) -> float:
        """计算标注一致性"""
        
        if len(annotations) < 2:
            return 1.0
        
        # 简化实现：计算标注的一致性
        if all(isinstance(ann, (int, float)) for ann in annotations):
            # 数值标注：计算变异系数
            mean_val = np.mean(annotations)
            std_val = np.std(annotations)
            cv = std_val / mean_val if mean_val != 0 else 0
            return max(0, 1 - cv)
        else:
            # 分类标注：计算众数比例
            counter = Counter(annotations)
            most_common_count = counter.most_common(1)[0][1]
            return most_common_count / len(annotations)
    
    def _compute_text_diversity(self, text: str) -> float:
        """计算文本多样性"""
        
        words = text.lower().split()
        if len(words) < 5:
            return 0.5
        
        # 词汇多样性：类型-标记比
        unique_words = set(words)
        ttr = len(unique_words) / len(words)
        
        return min(1.0, ttr * 2)  # 归一化到[0,1]
    
    def _stratified_sampling(self, 
                           data: List[Dict], 
                           stratification_key: str,
                           target_size: int) -> List[Dict]:
        """分层采样"""
        
        # 按分层键分组
        strata = defaultdict(list)
        for item in data:
            if stratification_key in item:
                strata[item[stratification_key]].append(item)
        
        # 计算各层的采样比例
        total_items = len(data)
        sampled_data = []
        
        for stratum, items in strata.items():
            stratum_size = len(items)
            stratum_proportion = stratum_size / total_items
            target_stratum_size = int(target_size * stratum_proportion)
            
            # 确保至少采样一个（如果该层有数据）
            target_stratum_size = max(1, min(target_stratum_size, stratum_size))
            
            # 随机采样
            if target_stratum_size < stratum_size:
                sampled_items = np.random.choice(
                    items, target_stratum_size, replace=False
                ).tolist()
            else:
                sampled_items = items
            
            sampled_data.extend(sampled_items)
        
        return sampled_data
    
    def _balance_difficulty_distribution(self, data: List[Dict]) -> List[Dict]:
        """平衡难度分布"""
        
        # 简化实现：基于文本长度和复杂度估计难度
        for item in data:
            if 'text' in item:
                text = item['text']
                words = text.split()
                
                # 难度估计（基于长度和词汇复杂度）
                length_score = min(1.0, len(words) / 100)
                
                # 词汇复杂度（平均词长）
                avg_word_length = np.mean([len(word) for word in words])
                complexity_score = min(1.0, avg_word_length / 10)
                
                difficulty = (length_score + complexity_score) / 2
                item['estimated_difficulty'] = difficulty
        
        # 按难度分组并平衡
        easy_items = [item for item in data if item.get('estimated_difficulty', 0.5) < 0.3]
        medium_items = [item for item in data if 0.3 <= item.get('estimated_difficulty', 0.5) < 0.7]
        hard_items = [item for item in data if item.get('estimated_difficulty', 0.5) >= 0.7]
        
        # 尝试平衡各难度级别
        target_per_level = len(data) // 3
        
        balanced_data = []
        for items, level in [(easy_items, 'easy'), (medium_items, 'medium'), (hard_items, 'hard')]:
            if len(items) > target_per_level:
                selected = np.random.choice(items, target_per_level, replace=False).tolist()
            else:
                selected = items
            
            for item in selected:
                item['difficulty_level'] = level
            
            balanced_data.extend(selected)
        
        return balanced_data
    
    def _compute_construction_stats(self, train_data: List[Dict], val_data: List[Dict]) -> Dict:
        """计算构造统计信息"""
        
        stats = {
            'train_size': len(train_data),
            'val_size': len(val_data),
            'total_size': len(train_data) + len(val_data)
        }
        
        # 难度分布统计
        all_data = train_data + val_data
        difficulty_dist = Counter(item.get('difficulty_level', 'unknown') for item in all_data)
        stats['difficulty_distribution'] = dict(difficulty_dist)
        
        # 质量分数统计
        quality_scores = [item.get('quality_score', 0) for item in all_data]
        stats['quality_stats'] = {
            'mean': np.mean(quality_scores),
            'std': np.std(quality_scores),
            'min': np.min(quality_scores),
            'max': np.max(quality_scores)
        }
        
        return stats

## 4.2 比较实验设计的统计学框架

### 控制变量与假设检验

**实验设计的基本原则**：
1. **随机化**：$P(\text{assign to group A}) = P(\text{assign to group B}) = 0.5$
2. **对照**：确保除待比较因素外其他条件相同
3. **重复**：多次独立实验验证结果稳定性

**配对t检验**：
对于配对样本 $(X_i, Y_i)$，检验 $H_0: \mu_D = 0$：
$$t = \frac{\bar{D} - 0}{s_D/\sqrt{n}} \sim t_{n-1}$$

其中 $D_i = X_i - Y_i$，$\bar{D} = \frac{1}{n}\sum D_i$，$s_D^2 = \frac{1}{n-1}\sum(D_i - \bar{D})^2$。

**效应量计算**：
Cohen's d：
$$d = \frac{\bar{X}_1 - \bar{X}_2}{\sqrt{\frac{(n_1-1)s_1^2 + (n_2-1)s_2^2}{n_1+n_2-2}}}$$

class ComparativeAnalyzer:
    """比较分析器"""
    
    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha  # 显著性水平
        self.comparison_history = []
        
    def compare_models(self, 
                      results: List[ModelResult],
                      metrics: List[str],
                      test_type: str = 'paired_t_test') -> Dict:
        """比较多个模型的性能"""
        
        print(f"=== 模型性能比较分析 (显著性水平: {self.alpha}) ===")
        
        # 组织数据
        model_names = [result.model_name for result in results]
        comparison_results = {}
        
        # 两两比较
        for i in range(len(model_names)):
            for j in range(i + 1, len(model_names)):
                model1, model2 = model_names[i], model_names[j]
                
                comparison_key = f"{model1}_vs_{model2}"
                comparison_results[comparison_key] = {}
                
                for metric in metrics:
                    # 获取两个模型在该指标上的分数
                    scores1 = [results[i].scores.get(metric, 0)]
                    scores2 = [results[j].scores.get(metric, 0)]
                    
                    # 如果有多次运行的结果，这里应该是列表
                    # 简化实现：假设只有一次运行
                    if len(scores1) == 1 and len(scores2) == 1:
                        # 无法进行统计检验，只能报告差异
                        diff = scores1[0] - scores2[0]
                        comparison_results[comparison_key][metric] = {
                            'difference': diff,
                            'model1_score': scores1[0],
                            'model2_score': scores2[0],
                            'significant': abs(diff) > 0.01,  # 简单阈值
                            'test_type': 'simple_difference'
                        }
                    else:
                        # 进行统计检验
                        test_result = self._perform_statistical_test(
                            scores1, scores2, test_type
                        )
                        comparison_results[comparison_key][metric] = test_result
        
        # 生成排名
        rankings = self._compute_model_rankings(results, metrics)
        
        # 计算整体比较统计
        overall_stats = self._compute_overall_comparison_stats(comparison_results)
        
        print(f"完成 {len(model_names)} 个模型的两两比较")
        print(f"共进行 {len(comparison_results)} 次比较")
        
        return {
            'pairwise_comparisons': comparison_results,
            'model_rankings': rankings,
            'overall_statistics': overall_stats
        }
    
    def _perform_statistical_test(self, 
                                scores1: List[float],
                                scores2: List[float],
                                test_type: str) -> Dict:
        """执行统计检验"""
        
        if test_type == 'paired_t_test':
            if len(scores1) != len(scores2):
                raise ValueError("配对t检验要求两组数据长度相同")
            
            differences = [s1 - s2 for s1, s2 in zip(scores1, scores2)]
            t_stat, p_value = stats.ttest_1samp(differences, 0)
            
            # 计算效应量（Cohen's d）
            mean_diff = np.mean(differences)
            std_diff = np.std(differences, ddof=1)
            cohens_d = mean_diff / std_diff if std_diff != 0 else 0
            
            return {
                'test_type': test_type,
                'model1_mean': np.mean(scores1),
                'model2_mean': np.mean(scores2),
                'mean_difference': mean_diff,
                't_statistic': t_stat,
                'p_value': p_value,
                'significant': p_value < self.alpha,
                'cohens_d': cohens_d,
                'effect_size': self._interpret_effect_size(abs(cohens_d))
            }
        
        elif test_type == 'independent_t_test':
            t_stat, p_value = stats.ttest_ind(scores1, scores2)
            
            # 计算效应量
            pooled_std = np.sqrt(((len(scores1) - 1) * np.var(scores1, ddof=1) + 
                                (len(scores2) - 1) * np.var(scores2, ddof=1)) / 
                               (len(scores1) + len(scores2) - 2))
            cohens_d = (np.mean(scores1) - np.mean(scores2)) / pooled_std
            
            return {
                'test_type': test_type,
                'model1_mean': np.mean(scores1),
                'model2_mean': np.mean(scores2),
                'mean_difference': np.mean(scores1) - np.mean(scores2),
                't_statistic': t_stat,
                'p_value': p_value,
                'significant': p_value < self.alpha,
                'cohens_d': cohens_d,
                'effect_size': self._interpret_effect_size(abs(cohens_d))
            }
        
        elif test_type == 'wilcoxon':
            # 非参数检验
            stat, p_value = stats.wilcoxon(scores1, scores2)
            
            return {
                'test_type': test_type,
                'model1_median': np.median(scores1),
                'model2_median': np.median(scores2),
                'median_difference': np.median(scores1) - np.median(scores2),
                'wilcoxon_statistic': stat,
                'p_value': p_value,
                'significant': p_value < self.alpha
            }
        
        else:
            raise ValueError(f"不支持的检验类型: {test_type}")
    
    def _interpret_effect_size(self, cohens_d: float) -> str:
        """解释效应量"""
        if cohens_d < 0.2:
            return 'negligible'
        elif cohens_d < 0.5:
            return 'small'
        elif cohens_d < 0.8:
            return 'medium'
        else:
            return 'large'
    
    def _compute_model_rankings(self, 
                               results: List[ModelResult],
                               metrics: List[str]) -> Dict:
        """计算模型排名"""
        
        rankings = {}
        
        for metric in metrics:
            # 获取所有模型在该指标上的分数
            model_scores = []
            for result in results:
                score = result.scores.get(metric, 0)
                model_scores.append((result.model_name, score))
            
            # 按分数降序排序
            model_scores.sort(key=lambda x: x[1], reverse=True)
            
            # 计算排名（处理并列情况）
            current_rank = 1
            rankings[metric] = {}
            
            for i, (model_name, score) in enumerate(model_scores):
                if i > 0 and model_scores[i][1] < model_scores[i-1][1]:
                    current_rank = i + 1
                
                rankings[metric][model_name] = {
                    'rank': current_rank,
                    'score': score
                }
        
        # 计算平均排名
        model_names = [result.model_name for result in results]
        average_rankings = {}
        
        for model_name in model_names:
            ranks = [rankings[metric][model_name]['rank'] for metric in metrics]
            average_rankings[model_name] = {
                'average_rank': np.mean(ranks),
                'rank_std': np.std(ranks),
                'individual_ranks': {metric: rankings[metric][model_name]['rank'] 
                                   for metric in metrics}
            }
        
        return {
            'metric_rankings': rankings,
            'average_rankings': average_rankings
        }
    
    def _compute_overall_comparison_stats(self, comparison_results: Dict) -> Dict:
        """计算整体比较统计"""
        
        total_comparisons = len(comparison_results)
        significant_comparisons = 0
        p_values = []
        effect_sizes = []
        
        for comparison_key, metrics_results in comparison_results.items():
            for metric, result in metrics_results.items():
                if 'p_value' in result:
                    p_values.append(result['p_value'])
                if result.get('significant', False):
                    significant_comparisons += 1
                if 'cohens_d' in result:
                    effect_sizes.append(abs(result['cohens_d']))
        
        # Bonferroni校正
        bonferroni_alpha = self.alpha / len(p_values) if p_values else self.alpha
        
        return {
            'total_comparisons': total_comparisons,
            'significant_comparisons': significant_comparisons,
            'significance_rate': significant_comparisons / len(p_values) if p_values else 0,
            'bonferroni_corrected_alpha': bonferroni_alpha,
            'p_value_distribution': {
                'mean': np.mean(p_values) if p_values else np.nan,
                'median': np.median(p_values) if p_values else np.nan,
                'min': np.min(p_values) if p_values else np.nan,
                'max': np.max(p_values) if p_values else np.nan
            },
            'effect_size_distribution': {
                'mean': np.mean(effect_sizes) if effect_sizes else np.nan,
                'median': np.median(effect_sizes) if effect_sizes else np.nan,
                'large_effects': sum(1 for es in effect_sizes if es >= 0.8) if effect_sizes else 0
            }
        }

## 4.3 排行榜系统的数学基础

### 排名聚合与公平性保证

**Borda计数法**：
$$\text{Borda}(x) = \sum_{i=1}^m w_i \cdot r_i(x)$$

其中 $r_i(x)$ 是模型 $x$ 在第 $i$ 个指标上的排名，$w_i$ 是权重。

**Kemeny距离最优化**：
$$\text{Kemeny}(\pi) = \sum_{x,y} I(\pi(x) < \pi(y) \text{ and } \tau(x) > \tau(y))$$

其中 $\pi$ 是聚合排名，$\tau$ 是输入排名。

**排名不确定性量化**：
使用Bootstrap方法估计排名的置信区间：
$$P(R_i \in [a, b]) = \frac{1}{B}\sum_{b=1}^{B} I(a \leq R_i^{(b)} \leq b)$$

class LeaderboardSystem:
    """排行榜系统"""
    
    def __init__(self, aggregation_method: str = 'borda_count'):
        self.aggregation_method = aggregation_method
        self.ranking_history = []
        self.confidence_intervals = {}
        
    def create_leaderboard(self, 
                          model_results: List[ModelResult],
                          metrics: List[str],
                          metric_weights: Optional[Dict[str, float]] = None) -> Dict:
        """创建排行榜"""
        
        print(f"=== 创建排行榜 (聚合方法: {self.aggregation_method}) ===")
        
        if metric_weights is None:
            metric_weights = {metric: 1.0 / len(metrics) for metric in metrics}
        
        # 计算各指标的排名
        metric_rankings = self._compute_metric_rankings(model_results, metrics)
        
        # 聚合排名
        aggregated_ranking = self._aggregate_rankings(
            metric_rankings, metric_weights
        )
        
        # 计算排名稳定性
        stability_analysis = self._analyze_ranking_stability(
            model_results, metrics, metric_weights
        )
        
        # 公平性分析
        fairness_analysis = self._analyze_ranking_fairness(
            metric_rankings, aggregated_ranking
        )
        
        # 构建最终排行榜
        leaderboard = self._construct_final_leaderboard(
            aggregated_ranking, model_results, metrics
        )
        
        print(f"排行榜包含 {len(leaderboard)} 个模型")
        print(f"使用 {len(metrics)} 个评估指标")
        
        return {
            'leaderboard': leaderboard,
            'metric_rankings': metric_rankings,
            'aggregation_method': self.aggregation_method,
            'stability_analysis': stability_analysis,
            'fairness_analysis': fairness_analysis,
            'metadata': {
                'creation_timestamp': 'current_time',
                'metrics_used': metrics,
                'metric_weights': metric_weights
            }
        }
    
    def _compute_metric_rankings(self, 
                               model_results: List[ModelResult],
                               metrics: List[str]) -> Dict:
        """计算各指标的排名"""
        
        rankings = {}
        
        for metric in metrics:
            # 获取所有模型在该指标上的分数
            model_scores = []
            for result in model_results:
                score = result.scores.get(metric, 0)
                model_scores.append((result.model_name, score))
            
            # 按分数降序排序
            model_scores.sort(key=lambda x: x[1], reverse=True)
            
            # 计算排名
            metric_ranking = {}
            for rank, (model_name, score) in enumerate(model_scores, 1):
                metric_ranking[model_name] = {
                    'rank': rank,
                    'score': score,
                    'percentile': (len(model_scores) - rank + 1) / len(model_scores)
                }
            
            rankings[metric] = metric_ranking
        
        return rankings
    
    def _aggregate_rankings(self, 
                          metric_rankings: Dict,
                          metric_weights: Dict[str, float]) -> Dict:
        """聚合排名"""
        
        model_names = set()
        for metric_ranking in metric_rankings.values():
            model_names.update(metric_ranking.keys())
        
        aggregated_scores = {}
        
        if self.aggregation_method == 'borda_count':
            # Borda计数法
            for model_name in model_names:
                borda_score = 0
                total_weight = 0
                
                for metric, weight in metric_weights.items():
                    if metric in metric_rankings and model_name in metric_rankings[metric]:
                        rank = metric_rankings[metric][model_name]['rank']
                        n_models = len(metric_rankings[metric])
                        # Borda分数：排名越高分数越高
                        borda_score += weight * (n_models - rank + 1)
                        total_weight += weight
                
                aggregated_scores[model_name] = borda_score / total_weight if total_weight > 0 else 0
        
        elif self.aggregation_method == 'weighted_average':
            # 加权平均分数
            for model_name in model_names:
                weighted_score = 0
                total_weight = 0
                
                for metric, weight in metric_weights.items():
                    if metric in metric_rankings and model_name in metric_rankings[metric]:
                        score = metric_rankings[metric][model_name]['score']
                        weighted_score += weight * score
                        total_weight += weight
                
                aggregated_scores[model_name] = weighted_score / total_weight if total_weight > 0 else 0
        
        elif self.aggregation_method == 'rank_product':
            # 排名乘积方法
            for model_name in model_names:
                rank_product = 1
                valid_metrics = 0
                
                for metric in metric_weights.keys():
                    if metric in metric_rankings and model_name in metric_rankings[metric]:
                        rank = metric_rankings[metric][model_name]['rank']
                        rank_product *= rank
                        valid_metrics += 1
                
                # 几何平均排名
                if valid_metrics > 0:
                    geometric_mean_rank = rank_product ** (1 / valid_metrics)
                    aggregated_scores[model_name] = 1 / geometric_mean_rank  # 转换为分数
                else:
                    aggregated_scores[model_name] = 0
        
        # 根据聚合分数排序
        sorted_models = sorted(aggregated_scores.items(), key=lambda x: x[1], reverse=True)
        
        final_ranking = {}
        for rank, (model_name, score) in enumerate(sorted_models, 1):
            final_ranking[model_name] = {
                'final_rank': rank,
                'aggregated_score': score
            }
        
        return final_ranking
    
    def _analyze_ranking_stability(self, 
                                 model_results: List[ModelResult],
                                 metrics: List[str],
                                 metric_weights: Dict[str, float],
                                 n_bootstrap: int = 100) -> Dict:
        """分析排名稳定性"""
        
        print("  分析排名稳定性...")
        
        model_names = [result.model_name for result in model_results]
        bootstrap_rankings = []
        
        # Bootstrap采样
        for _ in range(n_bootstrap):
            # 为简化，这里模拟分数的小幅波动
            perturbed_results = []
            for result in model_results:
                perturbed_scores = {}
                for metric in metrics:
                    original_score = result.scores.get(metric, 0)
                    # 添加小幅随机噪声
                    noise = np.random.normal(0, 0.01)
                    perturbed_scores[metric] = max(0, original_score + noise)
                
                perturbed_result = ModelResult(
                    model_name=result.model_name,
                    dataset_name=result.dataset_name,
                    scores=perturbed_scores,
                    predictions=result.predictions
                )
                perturbed_results.append(perturbed_result)
            
            # 计算这次采样的排名
            metric_rankings = self._compute_metric_rankings(perturbed_results, metrics)
            aggregated_ranking = self._aggregate_rankings(metric_rankings, metric_weights)
            
            # 记录排名
            rankings_list = [(model_name, info['final_rank']) 
                           for model_name, info in aggregated_ranking.items()]
            bootstrap_rankings.append(dict(rankings_list))
        
        # 计算排名统计
        rank_statistics = {}
        for model_name in model_names:
            ranks = [ranking[model_name] for ranking in bootstrap_rankings]
            rank_statistics[model_name] = {
                'mean_rank': np.mean(ranks),
                'std_rank': np.std(ranks),
                'median_rank': np.median(ranks),
                'rank_95_ci': (np.percentile(ranks, 2.5), np.percentile(ranks, 97.5)),
                'rank_stability': 1 / (1 + np.std(ranks))  # 稳定性指标
            }
        
        # 整体稳定性
        all_rank_stds = [stats['std_rank'] for stats in rank_statistics.values()]
        overall_stability = {
            'mean_rank_std': np.mean(all_rank_stds),
            'max_rank_std': np.max(all_rank_stds),
            'stability_score': 1 / (1 + np.mean(all_rank_stds))
        }
        
        return {
            'individual_stability': rank_statistics,
            'overall_stability': overall_stability,
            'bootstrap_samples': n_bootstrap
        }
    
    def _analyze_ranking_fairness(self, 
                                metric_rankings: Dict,
                                aggregated_ranking: Dict) -> Dict:
        """分析排名公平性"""
        
        print("  分析排名公平性...")
        
        # 计算排名一致性
        model_names = list(aggregated_ranking.keys())
        consistency_scores = {}
        
        for model_name in model_names:
            individual_ranks = []
            for metric in metric_rankings:
                if model_name in metric_rankings[metric]:
                    individual_ranks.append(metric_rankings[metric][model_name]['rank'])
            
            if individual_ranks:
                # 计算个体排名与最终排名的一致性
                final_rank = aggregated_ranking[model_name]['final_rank']
                avg_individual_rank = np.mean(individual_ranks)
                consistency = 1 / (1 + abs(final_rank - avg_individual_rank))
                consistency_scores[model_name] = {
                    'individual_ranks': individual_ranks,
                    'final_rank': final_rank,
                    'average_individual_rank': avg_individual_rank,
                    'consistency': consistency
                }
        
        # 检查是否存在系统性偏见
        bias_analysis = self._detect_ranking_bias(metric_rankings, aggregated_ranking)
        
        return {
            'consistency_scores': consistency_scores,
            'average_consistency': np.mean([score['consistency'] 
                                          for score in consistency_scores.values()]),
            'bias_analysis': bias_analysis
        }
    
    def _detect_ranking_bias(self, 
                           metric_rankings: Dict,
                           aggregated_ranking: Dict) -> Dict:
        """检测排名偏见"""
        
        # 检查是否某些指标过度影响最终排名
        metric_influence = {}
        
        for metric in metric_rankings:
            # 计算该指标排名与最终排名的相关性
            metric_ranks = []
            final_ranks = []
            
            for model_name in aggregated_ranking:
                if model_name in metric_rankings[metric]:
                    metric_ranks.append(metric_rankings[metric][model_name]['rank'])
                    final_ranks.append(aggregated_ranking[model_name]['final_rank'])
            
            if len(metric_ranks) > 1:
                correlation = np.corrcoef(metric_ranks, final_ranks)[0, 1]
                metric_influence[metric] = {
                    'correlation_with_final': correlation,
                    'influence_strength': abs(correlation)
                }
        
        # 识别异常影响
        if metric_influence:
            influences = [info['influence_strength'] for info in metric_influence.values()]
            mean_influence = np.mean(influences)
            std_influence = np.std(influences)
            
            biased_metrics = []
            for metric, info in metric_influence.items():
                if info['influence_strength'] > mean_influence + 2 * std_influence:
                    biased_metrics.append((metric, info['influence_strength']))
        else:
            biased_metrics = []
        
        return {
            'metric_influence': metric_influence,
            'potentially_biased_metrics': biased_metrics,
            'bias_detected': len(biased_metrics) > 0
        }
    
    def _construct_final_leaderboard(self, 
                                   aggregated_ranking: Dict,
                                   model_results: List[ModelResult],
                                   metrics: List[str]) -> List[Dict]:
        """构建最终排行榜"""
        
        leaderboard = []
        
        # 按最终排名排序
        sorted_models = sorted(aggregated_ranking.items(), 
                             key=lambda x: x[1]['final_rank'])
        
        for model_name, ranking_info in sorted_models:
            # 找到对应的模型结果
            model_result = next((r for r in model_results if r.model_name == model_name), None)
            
            if model_result:
                entry = {
                    'rank': ranking_info['final_rank'],
                    'model_name': model_name,
                    'aggregated_score': ranking_info['aggregated_score'],
                    'individual_scores': {metric: model_result.scores.get(metric, 0) 
                                        for metric in metrics},
                    'metadata': model_result.metadata
                }
                leaderboard.append(entry)
        
        return leaderboard

## 4.4 元评估方法

### 评估方法的评估

**评估方法的可靠性**：
$$\text{Reliability}(E) = \text{Corr}(E_1, E_2)$$

其中 $E_1, E_2$ 是同一评估方法在不同条件下的结果。

**评估方法的效度**：
$$\text{Validity}(E) = \text{Corr}(E, T)$$

其中 $T$ 是真实的模型质量。

**评估方法的区分度**：
$$\text{Discriminability}(E) = \frac{\text{Var}(\text{between groups})}{\text{Var}(\text{within groups})}$$

class MetaEvaluator:
    """元评估器"""
    
    def __init__(self):
        self.evaluation_methods = {}
        self.ground_truth_data = {}
        
    def evaluate_evaluation_methods(self, 
                                  evaluation_results: Dict[str, List[float]],
                                  ground_truth: Optional[List[float]] = None) -> Dict:
        """评估各种评估方法"""
        
        print("=== 元评估：评估方法的评估 ===")
        
        method_names = list(evaluation_results.keys())
        meta_analysis = {}
        
        # 1. 可靠性分析
        reliability_analysis = self._analyze_reliability(evaluation_results)
        
        # 2. 效度分析
        validity_analysis = self._analyze_validity(evaluation_results, ground_truth)
        
        # 3. 区分度分析
        discriminability_analysis = self._analyze_discriminability(evaluation_results)
        
        # 4. 一致性分析
        consistency_analysis = self._analyze_method_consistency(evaluation_results)
        
        # 5. 综合评估
        overall_ranking = self._rank_evaluation_methods(
            reliability_analysis, validity_analysis, 
            discriminability_analysis, consistency_analysis
        )
        
        print(f"评估了 {len(method_names)} 种评估方法")
        print(f"最佳评估方法: {overall_ranking[0]['method']}")
        
        return {
            'reliability_analysis': reliability_analysis,
            'validity_analysis': validity_analysis,
            'discriminability_analysis': discriminability_analysis,
            'consistency_analysis': consistency_analysis,
            'overall_ranking': overall_ranking
        }
    
    def _analyze_reliability(self, evaluation_results: Dict[str, List[float]]) -> Dict:
        """分析评估方法的可靠性"""
        
        reliability_scores = {}
        
        for method_name, scores in evaluation_results.items():
            if len(scores) >= 4:  # 需要足够的数据点
                # 分割数据进行split-half可靠性分析
                mid = len(scores) // 2
                half1 = scores[:mid]
                half2 = scores[mid:2*mid]  # 保证两半长度相同
                
                if len(half1) > 1 and len(half2) > 1:
                    correlation = np.corrcoef(half1, half2)[0, 1]
                    # Spearman-Brown预测公式
                    reliability = 2 * correlation / (1 + correlation)
                else:
                    reliability = np.nan
            else:
                reliability = np.nan
            
            reliability_scores[method_name] = {
                'split_half_reliability': reliability,
                'data_points': len(scores),
                'score_variance': np.var(scores)
            }
        
        return reliability_scores
    
    def _analyze_validity(self, 
                        evaluation_results: Dict[str, List[float]],
                        ground_truth: Optional[List[float]]) -> Dict:
        """分析评估方法的效度"""
        
        validity_scores = {}
        
        if ground_truth is None:
            # 如果没有ground truth，使用收敛效度（方法间相关性）
            method_names = list(evaluation_results.keys())
            
            for method_name in method_names:
                correlations_with_others = []
                
                for other_method in method_names:
                    if other_method != method_name:
                        scores1 = evaluation_results[method_name]
                        scores2 = evaluation_results[other_method]
                        
                        # 确保长度相同
                        min_len = min(len(scores1), len(scores2))
                        if min_len > 1:
                            corr = np.corrcoef(scores1[:min_len], scores2[:min_len])[0, 1]
                            if not np.isnan(corr):
                                correlations_with_others.append(abs(corr))
                
                convergent_validity = np.mean(correlations_with_others) if correlations_with_others else 0
                
                validity_scores[method_name] = {
                    'convergent_validity': convergent_validity,
                    'n_comparisons': len(correlations_with_others)
                }
        else:
            # 使用准则效度（与ground truth的相关性）
            for method_name, scores in evaluation_results.items():
                min_len = min(len(scores), len(ground_truth))
                if min_len > 1:
                    criterion_validity = np.corrcoef(scores[:min_len], ground_truth[:min_len])[0, 1]
                    if np.isnan(criterion_validity):
                        criterion_validity = 0
                else:
                    criterion_validity = 0
                
                validity_scores[method_name] = {
                    'criterion_validity': criterion_validity,
                    'data_points': min_len
                }
        
        return validity_scores
    
    def _analyze_discriminability(self, evaluation_results: Dict[str, List[float]]) -> Dict:
        """分析评估方法的区分度"""
        
        discriminability_scores = {}
        
        for method_name, scores in evaluation_results.items():
            if len(scores) > 1:
                # 计算分数的分散程度
                score_range = np.max(scores) - np.min(scores)
                score_std = np.std(scores)
                score_cv = score_std / np.mean(scores) if np.mean(scores) != 0 else 0
                
                # 区分度指标：范围和变异系数的组合
                discriminability = score_range * (1 + score_cv)
                
                discriminability_scores[method_name] = {
                    'discriminability': discriminability,
                    'score_range': score_range,
                    'score_std': score_std,
                    'coefficient_of_variation': score_cv
                }
            else:
                discriminability_scores[method_name] = {
                    'discriminability': 0,
                    'score_range': 0,
                    'score_std': 0,
                    'coefficient_of_variation': 0
                }
        
        return discriminability_scores
    
    def _analyze_method_consistency(self, evaluation_results: Dict[str, List[float]]) -> Dict:
        """分析评估方法的一致性"""
        
        consistency_analysis = {}
        method_names = list(evaluation_results.keys())
        
        # 计算方法间的排序一致性
        if len(method_names) >= 2:
            # 为每个数据点计算所有方法的排名
            min_len = min(len(scores) for scores in evaluation_results.values())
            
            rank_correlations = []
            
            for i in range(min_len):
                # 获取第i个数据点在所有方法中的分数
                point_scores = []
                for method_name in method_names:
                    if i < len(evaluation_results[method_name]):
                        point_scores.append(evaluation_results[method_name][i])
                    else:
                        point_scores.append(0)
                
                # 计算排名
                ranks = stats.rankdata(point_scores, method='average')
                
                # 如果是第一个点，保存为参考
                if i == 0:
                    reference_ranks = ranks
                else:
                    # 计算与参考排名的相关性
                    if len(ranks) > 1:
                        rank_corr = stats.spearmanr(reference_ranks, ranks)[0]
                        if not np.isnan(rank_corr):
                            rank_correlations.append(rank_corr)
            
            # 整体一致性
            overall_consistency = np.mean(rank_correlations) if rank_correlations else 0
            
            consistency_analysis['overall'] = {
                'rank_consistency': overall_consistency,
                'consistency_std': np.std(rank_correlations) if rank_correlations else 0,
                'n_comparisons': len(rank_correlations)
            }
            
            # 各方法的稳定性
            for method_name, scores in evaluation_results.items():
                if len(scores) > 1:
                    method_stability = 1 / (1 + np.std(scores) / np.mean(scores)) if np.mean(scores) != 0 else 0
                else:
                    method_stability = 0
                
                consistency_analysis[method_name] = {
                    'stability': method_stability,
                    'score_consistency': 1 - np.std(scores) / (np.max(scores) - np.min(scores)) if np.max(scores) != np.min(scores) else 1
                }
        
        return consistency_analysis
    
    def _rank_evaluation_methods(self, 
                               reliability_analysis: Dict,
                               validity_analysis: Dict,
                               discriminability_analysis: Dict,
                               consistency_analysis: Dict) -> List[Dict]:
        """对评估方法进行排名"""
        
        method_names = set()
        method_names.update(reliability_analysis.keys())
        method_names.update(validity_analysis.keys())
        method_names.update(discriminability_analysis.keys())
        method_names.update(consistency_analysis.keys())
        method_names.discard('overall')  # 移除overall键
        
        method_scores = {}
        
        for method_name in method_names:
            total_score = 0
            score_count = 0
            
            # 可靠性分数
            if method_name in reliability_analysis:
                reliability = reliability_analysis[method_name].get('split_half_reliability', 0)
                if not np.isnan(reliability):
                    total_score += max(0, reliability)
                    score_count += 1
            
            # 效度分数
            if method_name in validity_analysis:
                if 'criterion_validity' in validity_analysis[method_name]:
                    validity = validity_analysis[method_name]['criterion_validity']
                else:
                    validity = validity_analysis[method_name].get('convergent_validity', 0)
                
                if not np.isnan(validity):
                    total_score += abs(validity)
                    score_count += 1
            
            # 区分度分数（归一化）
            if method_name in discriminability_analysis:
                discriminability = discriminability_analysis[method_name]['discriminability']
                # 简单归一化到[0,1]
                normalized_discriminability = min(1.0, discriminability / 10)
                total_score += normalized_discriminability
                score_count += 1
            
            # 一致性分数
            if method_name in consistency_analysis:
                stability = consistency_analysis[method_name].get('stability', 0)
                total_score += stability
                score_count += 1
            
            # 计算平均分数
            if score_count > 0:
                average_score = total_score / score_count
            else:
                average_score = 0
            
            method_scores[method_name] = {
                'method': method_name,
                'overall_score': average_score,
                'component_scores': {
                    'reliability': reliability_analysis.get(method_name, {}).get('split_half_reliability', np.nan),
                    'validity': validity_analysis.get(method_name, {}).get('criterion_validity', 
                                                                          validity_analysis.get(method_name, {}).get('convergent_validity', np.nan)),
                    'discriminability': discriminability_analysis.get(method_name, {}).get('discriminability', 0),
                    'consistency': consistency_analysis.get(method_name, {}).get('stability', 0)
                }
            }
        
        # 按总分排序
        ranked_methods = sorted(method_scores.values(), 
                              key=lambda x: x['overall_score'], 
                              reverse=True)
        
        return ranked_methods

def create_comprehensive_benchmarking_system():
    """创建综合基准测试系统"""
    
    constructor = BenchmarkConstructor()
    analyzer = ComparativeAnalyzer()
    leaderboard = LeaderboardSystem()
    meta_evaluator = MetaEvaluator()
    
    return {
        'constructor': constructor,
        'analyzer': analyzer,
        'leaderboard': leaderboard,
        'meta_evaluator': meta_evaluator
    }

# 演示完整的基准测试与比较分析系统
def demonstrate_benchmarking_system():
    """演示基准测试系统"""
    
    print("=== MiniGPT基准测试与比较分析系统演示 ===\n")
    
    # 创建基准测试系统
    benchmark_system = create_comprehensive_benchmarking_system()
    
    # 1. 构造基准数据集
    print("1. 构造基准数据集")
    
    # 模拟原始数据
    raw_data = []
    for i in range(1500):
        text_length = np.random.randint(20, 200)
        difficulty = np.random.choice(['easy', 'medium', 'hard'])
        quality = np.random.uniform(0.3, 1.0)
        
        raw_data.append({
            'id': f'sample_{i}',
            'text': f'Sample text {i} ' * text_length,
            'difficulty': difficulty,
            'quality_score': quality,
            'annotations': [np.random.randint(1, 6) for _ in range(3)]  # 3个标注者
        })
    
    benchmark_dataset = benchmark_system['constructor'].construct_benchmark(
        raw_data, stratification_key='difficulty', target_size=800
    )
    
    # 2. 模拟多个模型的结果
    print("\n2. 模拟模型评估结果")
    
    model_names = ['MiniGPT-Base', 'MiniGPT-Large', 'Baseline-LSTM', 'GPT-Small', 'BERT-Base']
    metrics = ['accuracy', 'f1_score', 'bleu', 'rouge_l', 'perplexity']
    
    model_results = []
    for model_name in model_names:
        # 模拟不同模型的性能特点
        scores = {}
        base_performance = np.random.uniform(0.6, 0.9)
        
        for metric in metrics:
            if metric == 'perplexity':
                # 困惑度越低越好
                scores[metric] = np.random.uniform(10, 50)
            else:
                noise = np.random.normal(0, 0.05)
                scores[metric] = max(0, min(1, base_performance + noise))
        
        # 生成假的预测结果
        predictions = [f"prediction_{i}" for i in range(100)]
        
        result = ModelResult(
            model_name=model_name,
            dataset_name=benchmark_dataset.name,
            scores=scores,
            predictions=predictions,
            metadata={'model_size': f'{np.random.randint(10, 200)}M parameters'}
        )
        model_results.append(result)
    
    # 3. 比较分析
    print("\n3. 模型比较分析")
    comparison_results = benchmark_system['analyzer'].compare_models(
        model_results, metrics[:-1]  # 排除perplexity用于简化
    )
    
    # 4. 创建排行榜
    print("\n4. 创建排行榜")
    leaderboard_results = benchmark_system['leaderboard'].create_leaderboard(
        model_results, metrics[:-1]
    )
    
    # 5. 元评估
    print("\n5. 元评估分析")
    
    # 模拟多种评估方法的结果
    evaluation_methods_results = {}
    for metric in metrics[:-1]:
        # 为每个指标生成多次评估结果（模拟重复实验）
        evaluation_methods_results[metric] = [
            result.scores[metric] + np.random.normal(0, 0.01) 
            for result in model_results for _ in range(3)
        ]
    
    meta_evaluation_results = benchmark_system['meta_evaluator'].evaluate_evaluation_methods(
        evaluation_methods_results
    )
    
    # 6. 结果可视化
    print("\n6. 生成分析报告")
    
    # 显示排行榜前3名
    print("排行榜前3名:")
    for i, entry in enumerate(leaderboard_results['leaderboard'][:3]):
        print(f"  {entry['rank']}. {entry['model_name']}: 综合分数 {entry['aggregated_score']:.4f}")
    
    # 显示最佳评估方法
    best_method = meta_evaluation_results['overall_ranking'][0]
    print(f"\n最佳评估方法: {best_method['method']} (综合分数: {best_method['overall_score']:.4f})")
    
    return {
        'benchmark_dataset': benchmark_dataset,
        'model_results': model_results,
        'comparison_results': comparison_results,
        'leaderboard_results': leaderboard_results,
        'meta_evaluation_results': meta_evaluation_results,
        'benchmark_system': benchmark_system
    }

# 运行演示
if __name__ == "__main__":
    results = demonstrate_benchmarking_system()
    
    print("\n=== 基准测试与比较分析系统评估完成 ===")
    print(f"系统功能总结:")
    print(f"- 基准数据集构造: 科学严谨")
    print(f"- 统计比较分析: 方法完备")
    print(f"- 排行榜系统: 公平可靠")
    print(f"- 元评估能力: 深入全面")
```

## 理论总结

### 4.5 基准测试的统一理论框架

**基准测试的信息论基础**：
一个好的基准应该最大化关于模型能力的信息量：
$$I(\text{Capability}; \text{Benchmark}) = H(\text{Capability}) - H(\text{Capability}|\text{Benchmark})$$

**比较分析的决策理论**：
模型选择问题可表述为：
$$\hat{m} = \arg\min_m \mathbb{E}[L(m, m^*)]$$

其中 $L(\cdot, \cdot)$ 是损失函数，$m^*$ 是真正的最优模型。

**排名系统的公理化基础**：
一个公平的排名系统应满足：
1. **单调性**：更好的性能对应更高的排名
2. **一致性**：相同的性能应获得相同的排名  
3. **传递性**：如果A>B且B>C，则A>C

## 应用指导

### 实践建议

1. **基准设计原则**：
   - 确保数据集的代表性和多样性
   - 建立严格的质量控制流程
   - 考虑任务的生态效度

2. **统计分析规范**：
   - 选择合适的统计检验方法
   - 进行多重比较校正
   - 报告效应量和置信区间

3. **排行榜管理**：
   - 定期更新和维护基准
   - 提供透明的评估流程
   - 建立申诉和复查机制

基准测试与比较分析是语言模型研究的重要基础设施，需要在科学严谨性和实用性之间找到平衡。通过系统性的方法学，我们能够构建公平、可靠、有意义的评估体系。

## 扩展阅读

- 《Benchmarking Neural Text Generators》- 文本生成基准方法学
- 《Statistical Significance Tests for NLP》- NLP中的统计检验
- 《Leaderboards in AI: The Good, the Bad, and the Ugly》- 排行榜系统分析
- 《Meta-Evaluation of Evaluation Metrics》- 元评估理论与实践

---

*"好的基准不仅能评估现在，更能指引未来。科学的比较分析是推动技术进步的重要动力。"* 🎯