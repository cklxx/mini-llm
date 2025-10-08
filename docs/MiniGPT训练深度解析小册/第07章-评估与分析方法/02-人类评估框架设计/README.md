# 02 人类评估框架设计

> **从主观感知到客观测度：构建科学的人类评估体系**

## 核心思想

人类评估是语言模型评估的"金标准"，但同时也是最具挑战性的评估方式。人类的语言判断涉及复杂的认知过程——从语法分析到语义理解，从逻辑推理到情感体验，每个层面都带有主观性和不确定性。如何将这种主观判断转化为可靠的科学测度，是人类评估框架设计的核心问题。

**关键洞察**：
- **主观性控制**：通过科学的实验设计降低主观偏差
- **一致性保证**：建立可靠的标注者间一致性
- **维度分解**：将复杂的语言质量分解为可操作的评估维度
- **规模化挑战**：在保证质量的前提下实现大规模评估

从数学角度看，人类评估是在构建从文本空间到评估空间的随机映射，我们需要通过统计学方法来控制这个映射的可靠性和有效性。

## 2.1 评估维度的操作化定义

### 语言质量的多维分解

**质量维度的数学建模**：
设文本质量为多维向量 $\mathbf{Q} = (q_1, q_2, ..., q_d) \in \mathbb{R}^d$，其中每个维度代表一个可操作的评估标准：

$$\mathbf{Q}(t) = \begin{pmatrix}
\text{Fluency}(t) \\
\text{Coherence}(t) \\
\text{Relevance}(t) \\
\text{Informativeness}(t) \\
\text{Creativity}(t) \\
\vdots
\end{pmatrix}$$

**维度间关系建模**：
评估维度通常不独立，可用相关矩阵 $\mathbf{R}$ 描述：
$$\mathbf{R}_{ij} = \text{Corr}(q_i, q_j)$$

**综合质量分数**：
$$Q_{overall}(t) = \mathbf{w}^T \mathbf{Q}(t) = \sum_{i=1}^{d} w_i \cdot q_i(t)$$

其中权重向量 $\mathbf{w}$ 可通过主成分分析或回归方法确定。

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
from sklearn.metrics import cohen_kappa_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

@dataclass
class EvaluationDimension:
    """评估维度定义"""
    name: str
    description: str
    scale: Tuple[int, int]  # 评分区间，如(1, 5)
    criteria: List[str]     # 评估标准
    examples: Dict[int, str] = field(default_factory=dict)  # 各分数级别的示例
    
@dataclass
class AnnotationResult:
    """标注结果数据结构"""
    text_id: str
    annotator_id: str
    dimensions: Dict[str, int]  # 各维度评分
    overall_score: Optional[float] = None
    comments: str = ""
    annotation_time: float = 0.0
    confidence: Optional[float] = None

class HumanEvaluationFramework:
    """人类评估框架"""
    
    def __init__(self):
        self.dimensions = self._define_evaluation_dimensions()
        self.annotations = []
        self.annotator_profiles = {}
        
    def _define_evaluation_dimensions(self) -> Dict[str, EvaluationDimension]:
        """定义评估维度"""
        
        dimensions = {
            'fluency': EvaluationDimension(
                name='流畅性',
                description='文本的语法正确性和自然流畅程度',
                scale=(1, 5),
                criteria=[
                    '语法错误数量',
                    '句子结构合理性',
                    '词汇使用恰当性',
                    '整体阅读流畅性'
                ],
                examples={
                    1: '语法错误严重，难以理解',
                    2: '语法错误较多，影响理解',
                    3: '语法基本正确，略有不自然',
                    4: '语法正确，表达自然',
                    5: '语法完美，表达非常自然'
                }
            ),
            
            'coherence': EvaluationDimension(
                name='连贯性',
                description='文本内容的逻辑一致性和连贯性',
                scale=(1, 5),
                criteria=[
                    '逻辑结构清晰性',
                    '前后内容一致性',
                    '主题聚焦程度',
                    '段落间衔接'
                ],
                examples={
                    1: '逻辑混乱，前后矛盾',
                    2: '逻辑性较差，部分矛盾',
                    3: '逻辑基本清晰，偶有跳跃',
                    4: '逻辑清晰，内容连贯',
                    5: '逻辑完美，高度连贯'
                }
            ),
            
            'relevance': EvaluationDimension(
                name='相关性',
                description='文本内容与给定主题或任务的相关程度',
                scale=(1, 5),
                criteria=[
                    '主题契合度',
                    '任务完成度',
                    '信息针对性',
                    '内容聚焦性'
                ],
                examples={
                    1: '完全偏离主题',
                    2: '部分相关，偏离较多',
                    3: '基本相关，有偏移',
                    4: '高度相关，紧扣主题',
                    5: '完美契合，精准聚焦'
                }
            ),
            
            'informativeness': EvaluationDimension(
                name='信息量',
                description='文本包含的有用信息数量和质量',
                scale=(1, 5),
                criteria=[
                    '信息丰富程度',
                    '细节完整性',
                    '深度分析水平',
                    '新颖性程度'
                ],
                examples={
                    1: '信息极少，内容空洞',
                    2: '信息较少，缺乏深度',
                    3: '信息适中，深度一般',
                    4: '信息丰富，有一定深度',
                    5: '信息非常丰富，深度分析'
                }
            ),
            
            'creativity': EvaluationDimension(
                name='创造性',
                description='文本表达的创新性和独特性',
                scale=(1, 5),
                criteria=[
                    '表达方式新颖性',
                    '观点独特性',
                    '创意程度',
                    '想象力体现'
                ],
                examples={
                    1: '完全模式化，无创意',
                    2: '略有新意，创意有限',
                    3: '一定创意，表达合理',
                    4: '较有创意，表达新颖',
                    5: '极具创意，独特新颖'
                }
            )
        }
        
        return dimensions
    
    def add_annotation(self, annotation: AnnotationResult):
        """添加标注结果"""
        self.annotations.append(annotation)
    
    def compute_overall_score(self, 
                            dimension_scores: Dict[str, int],
                            method: str = 'weighted_average',
                            weights: Optional[Dict[str, float]] = None) -> float:
        """计算综合评分"""
        
        if method == 'simple_average':
            return np.mean(list(dimension_scores.values()))
        
        elif method == 'weighted_average':
            if weights is None:
                # 默认权重
                weights = {
                    'fluency': 0.25,
                    'coherence': 0.25,
                    'relevance': 0.25,
                    'informativeness': 0.15,
                    'creativity': 0.10
                }
            
            total_score = 0
            total_weight = 0
            for dim, score in dimension_scores.items():
                if dim in weights:
                    total_score += weights[dim] * score
                    total_weight += weights[dim]
            
            return total_score / total_weight if total_weight > 0 else 0
        
        elif method == 'pca_weighted':
            # 使用主成分分析确定权重
            return self._compute_pca_weighted_score(dimension_scores)
        
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def _compute_pca_weighted_score(self, dimension_scores: Dict[str, int]) -> float:
        """基于PCA的加权评分"""
        
        # 收集所有标注数据进行PCA
        if len(self.annotations) < 10:  # 数据不足时使用简单平均
            return np.mean(list(dimension_scores.values()))
        
        # 构建数据矩阵
        score_matrix = []
        dimensions = list(self.dimensions.keys())
        
        for annotation in self.annotations:
            scores = [annotation.dimensions.get(dim, 3) for dim in dimensions]
            score_matrix.append(scores)
        
        score_matrix = np.array(score_matrix)
        
        # 标准化和PCA
        scaler = StandardScaler()
        normalized_scores = scaler.fit_transform(score_matrix)
        
        pca = PCA(n_components=1)
        pca.fit(normalized_scores)
        
        # 使用第一主成分的载荷作为权重
        weights = np.abs(pca.components_[0])
        weights = weights / np.sum(weights)
        
        # 计算加权分数
        current_scores = [dimension_scores.get(dim, 3) for dim in dimensions]
        return np.dot(weights, current_scores)

## 2.2 标注一致性理论与度量

### Cohen's Kappa系数的数学原理

**Kappa系数定义**：
$$\kappa = \frac{p_o - p_e}{1 - p_e}$$

其中：
- $p_o$ 是观察到的一致性比例
- $p_e$ 是期望的随机一致性比例

**多分类Kappa的计算**：
对于 $k$ 个类别，混淆矩阵为 $\mathbf{C}_{k \times k}$：
$$p_o = \frac{\sum_{i=1}^k C_{ii}}{N}$$
$$p_e = \frac{\sum_{i=1}^k \left(\sum_{j=1}^k C_{ij}\right) \left(\sum_{j=1}^k C_{ji}\right)}{N^2}$$

**加权Kappa**：
对于有序分类（如评分），可使用加权Kappa：
$$\kappa_w = 1 - \frac{\sum_{i,j} w_{ij} C_{ij}}{\sum_{i,j} w_{ij} E_{ij}}$$

其中 $w_{ij}$ 是不一致权重，通常取 $w_{ij} = |i-j|^2$。

### 组内相关系数(ICC)

**ICC的数学定义**：
对于双向随机效应模型：
$$\text{ICC}(2,1) = \frac{\text{MS}_R - \text{MS}_E}{\text{MS}_R + (k-1)\text{MS}_E + \frac{k}{n}(\text{MS}_C - \text{MS}_E)}$$

其中：
- $\text{MS}_R$ 是行（被评估项）均方
- $\text{MS}_C$ 是列（评估者）均方  
- $\text{MS}_E$ 是误差均方
- $k$ 是评估者数量，$n$ 是被评估项数量

class AnnotationConsistencyAnalyzer:
    """标注一致性分析器"""
    
    def __init__(self):
        self.annotations_matrix = None
        self.annotator_pairs = []
        
    def prepare_data(self, annotations: List[AnnotationResult], dimension: str):
        """准备标注一致性分析数据"""
        
        # 构建标注矩阵
        text_ids = sorted(set(ann.text_id for ann in annotations))
        annotator_ids = sorted(set(ann.annotator_id for ann in annotations))
        
        matrix = np.full((len(text_ids), len(annotator_ids)), np.nan)
        
        for ann in annotations:
            if dimension in ann.dimensions:
                text_idx = text_ids.index(ann.text_id)
                annotator_idx = annotator_ids.index(ann.annotator_id)
                matrix[text_idx, annotator_idx] = ann.dimensions[dimension]
        
        self.annotations_matrix = matrix
        self.text_ids = text_ids
        self.annotator_ids = annotator_ids
        
        return matrix
    
    def compute_pairwise_kappa(self, 
                              annotations: List[AnnotationResult],
                              dimension: str,
                              weights: Optional[str] = None) -> Dict:
        """计算两两标注者间的Kappa系数"""
        
        print(f"=== {dimension}维度两两Kappa分析 ===")
        
        # 准备数据
        self.prepare_data(annotations, dimension)
        
        # 计算两两Kappa
        kappa_matrix = np.full((len(self.annotator_ids), len(self.annotator_ids)), np.nan)
        pairwise_kappas = []
        
        for i in range(len(self.annotator_ids)):
            for j in range(i+1, len(self.annotator_ids)):
                # 提取两个标注者的评分
                scores_i = self.annotations_matrix[:, i]
                scores_j = self.annotations_matrix[:, j]
                
                # 过滤NaN值
                valid_mask = ~(np.isnan(scores_i) | np.isnan(scores_j))
                if np.sum(valid_mask) < 5:  # 需要至少5个共同标注
                    continue
                
                valid_scores_i = scores_i[valid_mask].astype(int)
                valid_scores_j = scores_j[valid_mask].astype(int)
                
                # 计算Kappa
                if weights == 'quadratic':
                    kappa = self._compute_weighted_kappa(valid_scores_i, valid_scores_j)
                else:
                    kappa = cohen_kappa_score(valid_scores_i, valid_scores_j)
                
                kappa_matrix[i, j] = kappa
                kappa_matrix[j, i] = kappa
                pairwise_kappas.append(kappa)
        
        # 填充对角线
        np.fill_diagonal(kappa_matrix, 1.0)
        
        # 统计分析
        kappa_stats = {
            'mean': np.mean(pairwise_kappas),
            'std': np.std(pairwise_kappas),
            'min': np.min(pairwise_kappas),
            'max': np.max(pairwise_kappas),
            'median': np.median(pairwise_kappas)
        }
        
        print(f"平均Kappa: {kappa_stats['mean']:.4f} ± {kappa_stats['std']:.4f}")
        print(f"Kappa范围: [{kappa_stats['min']:.4f}, {kappa_stats['max']:.4f}]")
        
        return {
            'kappa_matrix': kappa_matrix,
            'pairwise_kappas': pairwise_kappas,
            'statistics': kappa_stats,
            'annotator_ids': self.annotator_ids
        }
    
    def _compute_weighted_kappa(self, scores1: np.ndarray, scores2: np.ndarray) -> float:
        """计算加权Kappa系数"""
        
        # 构建混淆矩阵
        min_score = min(np.min(scores1), np.min(scores2))
        max_score = max(np.max(scores1), np.max(scores2))
        
        confusion_matrix = np.zeros((max_score - min_score + 1, max_score - min_score + 1))
        
        for s1, s2 in zip(scores1, scores2):
            confusion_matrix[s1 - min_score, s2 - min_score] += 1
        
        n = len(scores1)
        
        # 计算观察一致性
        po = 0
        for i in range(confusion_matrix.shape[0]):
            for j in range(confusion_matrix.shape[1]):
                weight = 1 - ((i - j) ** 2) / ((max_score - min_score) ** 2)
                po += weight * confusion_matrix[i, j] / n
        
        # 计算期望一致性
        row_marginals = np.sum(confusion_matrix, axis=1) / n
        col_marginals = np.sum(confusion_matrix, axis=0) / n
        
        pe = 0
        for i in range(confusion_matrix.shape[0]):
            for j in range(confusion_matrix.shape[1]):
                weight = 1 - ((i - j) ** 2) / ((max_score - min_score) ** 2)
                pe += weight * row_marginals[i] * col_marginals[j]
        
        # 计算加权Kappa
        if pe == 1:
            return 1.0
        else:
            return (po - pe) / (1 - pe)
    
    def compute_icc(self, 
                   annotations: List[AnnotationResult],
                   dimension: str,
                   icc_type: str = '2,1') -> Dict:
        """计算组内相关系数"""
        
        print(f"=== {dimension}维度ICC分析 ===")
        
        # 准备数据
        matrix = self.prepare_data(annotations, dimension)
        
        # 移除包含NaN的行
        valid_rows = ~np.any(np.isnan(matrix), axis=1)
        clean_matrix = matrix[valid_rows]
        
        if clean_matrix.shape[0] < 3:
            print("有效数据不足，无法计算ICC")
            return {'icc': np.nan, 'confidence_interval': (np.nan, np.nan)}
        
        # 计算ICC
        n, k = clean_matrix.shape
        
        # 计算均方
        row_means = np.mean(clean_matrix, axis=1)
        col_means = np.mean(clean_matrix, axis=0)
        grand_mean = np.mean(clean_matrix)
        
        # 行间均方 (MSR)
        msr = k * np.sum((row_means - grand_mean) ** 2) / (n - 1)
        
        # 列间均方 (MSC)
        msc = n * np.sum((col_means - grand_mean) ** 2) / (k - 1)
        
        # 误差均方 (MSE)
        mse = np.sum((clean_matrix - row_means.reshape(-1, 1) - col_means.reshape(1, -1) + grand_mean) ** 2) / ((n - 1) * (k - 1))
        
        # 计算ICC(2,1)
        if icc_type == '2,1':
            icc = (msr - mse) / (msr + (k - 1) * mse + k * (msc - mse) / n)
        else:
            raise ValueError(f"Unsupported ICC type: {icc_type}")
        
        # 计算置信区间（简化实现）
        # 实际应用中需要更复杂的F分布计算
        alpha = 0.05
        f_value = msr / mse
        
        # 简化的置信区间估计
        icc_lower = max(0, icc - 1.96 * np.sqrt(2 * (1 - icc) ** 2 * (k - 1) / (n * k)))
        icc_upper = min(1, icc + 1.96 * np.sqrt(2 * (1 - icc) ** 2 * (k - 1) / (n * k)))
        
        print(f"ICC({icc_type}): {icc:.4f}")
        print(f"95%置信区间: [{icc_lower:.4f}, {icc_upper:.4f}]")
        
        return {
            'icc': icc,
            'confidence_interval': (icc_lower, icc_upper),
            'msr': msr,
            'msc': msc,
            'mse': mse,
            'n_subjects': n,
            'n_raters': k
        }
    
    def analyze_annotator_bias(self, 
                             annotations: List[AnnotationResult],
                             dimension: str) -> Dict:
        """分析标注者偏差"""
        
        print(f"=== {dimension}维度标注者偏差分析 ===")
        
        # 准备数据
        matrix = self.prepare_data(annotations, dimension)
        
        # 计算各标注者的统计特征
        annotator_stats = {}
        for i, annotator_id in enumerate(self.annotator_ids):
            scores = matrix[:, i]
            valid_scores = scores[~np.isnan(scores)]
            
            if len(valid_scores) > 0:
                annotator_stats[annotator_id] = {
                    'mean': np.mean(valid_scores),
                    'std': np.std(valid_scores),
                    'min': np.min(valid_scores),
                    'max': np.max(valid_scores),
                    'count': len(valid_scores),
                    'skewness': stats.skew(valid_scores),
                    'kurtosis': stats.kurtosis(valid_scores)
                }
        
        # 分析严格度差异
        means = [stats['mean'] for stats in annotator_stats.values()]
        stds = [stats['std'] for stats in annotator_stats.values()]
        
        severity_analysis = {
            'mean_difference': np.max(means) - np.min(means),
            'std_difference': np.max(stds) - np.min(stds),
            'severity_ranking': sorted(annotator_stats.items(), key=lambda x: x[1]['mean'])
        }
        
        print(f"标注者评分均值差异: {severity_analysis['mean_difference']:.4f}")
        print(f"标注者评分标准差差异: {severity_analysis['std_difference']:.4f}")
        
        print("标注者严格度排序 (从严格到宽松):")
        for annotator_id, stats in severity_analysis['severity_ranking']:
            print(f"  {annotator_id}: 均值={stats['mean']:.2f}, 标准差={stats['std']:.2f}")
        
        return {
            'annotator_statistics': annotator_stats,
            'severity_analysis': severity_analysis
        }

## 2.3 认知偏差控制

### 常见评估偏差及其数学建模

**光环效应(Halo Effect)**：
$$P(q_i = s | q_j = s') = P(q_i = s) + \alpha \cdot I(s = s')$$

其中 $\alpha$ 是光环效应强度，$I(\cdot)$ 是指示函数。

**锚定偏差(Anchoring Bias)**：
$$q_t = \beta \cdot q_{anchor} + (1-\beta) \cdot q_{true} + \epsilon$$

其中 $q_{anchor}$ 是锚定参考值，$\beta$ 是锚定强度。

**顺序效应(Order Effect)**：
$$q_t = q_{true} + \gamma \cdot f(t) + \epsilon$$

其中 $f(t)$ 是时间相关的疲劳函数。

class CognitiveBiasController:
    """认知偏差控制器"""
    
    def __init__(self):
        self.bias_detection_methods = {
            'halo_effect': self._detect_halo_effect,
            'anchoring_bias': self._detect_anchoring_bias,
            'order_effect': self._detect_order_effect,
            'severity_bias': self._detect_severity_bias
        }
    
    def detect_biases(self, 
                     annotations: List[AnnotationResult]) -> Dict:
        """检测各种认知偏差"""
        
        print("=== 认知偏差检测分析 ===")
        
        bias_results = {}
        
        for bias_name, detection_method in self.bias_detection_methods.items():
            try:
                bias_result = detection_method(annotations)
                bias_results[bias_name] = bias_result
                print(f"{bias_name}: {bias_result.get('severity', 'N/A')}")
            except Exception as e:
                print(f"检测{bias_name}时出错: {e}")
                bias_results[bias_name] = {'error': str(e)}
        
        return bias_results
    
    def _detect_halo_effect(self, annotations: List[AnnotationResult]) -> Dict:
        """检测光环效应"""
        
        # 计算维度间相关性
        dimension_scores = defaultdict(list)
        
        for ann in annotations:
            for dim, score in ann.dimensions.items():
                dimension_scores[dim].append(score)
        
        # 计算相关矩阵
        dimensions = list(dimension_scores.keys())
        correlations = []
        
        for i in range(len(dimensions)):
            for j in range(i+1, len(dimensions)):
                dim1, dim2 = dimensions[i], dimensions[j]
                if len(dimension_scores[dim1]) == len(dimension_scores[dim2]):
                    corr = np.corrcoef(dimension_scores[dim1], dimension_scores[dim2])[0, 1]
                    correlations.append(abs(corr))
        
        # 光环效应强度 = 平均相关系数
        halo_strength = np.mean(correlations) if correlations else 0
        
        severity = 'low' if halo_strength < 0.3 else 'medium' if halo_strength < 0.6 else 'high'
        
        return {
            'halo_strength': halo_strength,
            'severity': severity,
            'dimension_correlations': correlations
        }
    
    def _detect_anchoring_bias(self, annotations: List[AnnotationResult]) -> Dict:
        """检测锚定偏差"""
        
        # 按标注者分组，分析评分趋势
        annotator_sequences = defaultdict(list)
        
        for ann in annotations:
            # 使用综合评分作为锚定检测的指标
            overall_score = np.mean(list(ann.dimensions.values()))
            annotator_sequences[ann.annotator_id].append((ann.text_id, overall_score))
        
        # 计算序列相关性（相邻评分的相关性）
        sequence_correlations = []
        
        for annotator_id, sequence in annotator_sequences.items():
            if len(sequence) < 3:
                continue
            
            # 按某种顺序排序（实际应用中可能是时间顺序）
            sequence.sort(key=lambda x: x[0])
            scores = [score for _, score in sequence]
            
            # 计算相邻评分的相关性
            adjacent_correlations = []
            for i in range(len(scores) - 1):
                if i < len(scores) - 2:
                    corr = np.corrcoef([scores[i], scores[i+1]], [scores[i+1], scores[i+2]])[0, 1]
                    if not np.isnan(corr):
                        adjacent_correlations.append(corr)
            
            if adjacent_correlations:
                sequence_correlations.extend(adjacent_correlations)
        
        # 锚定偏差强度
        anchoring_strength = np.mean(sequence_correlations) if sequence_correlations else 0
        severity = 'low' if anchoring_strength < 0.3 else 'medium' if anchoring_strength < 0.6 else 'high'
        
        return {
            'anchoring_strength': anchoring_strength,
            'severity': severity,
            'sequence_correlations': sequence_correlations
        }
    
    def _detect_order_effect(self, annotations: List[AnnotationResult]) -> Dict:
        """检测顺序效应"""
        
        # 按标注者分析评分随时间的变化
        annotator_trends = {}
        
        for ann in annotations:
            if ann.annotator_id not in annotator_trends:
                annotator_trends[ann.annotator_id] = []
            
            overall_score = np.mean(list(ann.dimensions.values()))
            annotator_trends[ann.annotator_id].append(overall_score)
        
        # 计算趋势强度
        trend_strengths = []
        
        for annotator_id, scores in annotator_trends.items():
            if len(scores) < 5:
                continue
            
            # 使用线性回归检测趋势
            x = np.arange(len(scores))
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, scores)
            
            trend_strengths.append(abs(slope))
        
        # 顺序效应强度
        order_effect_strength = np.mean(trend_strengths) if trend_strengths else 0
        severity = 'low' if order_effect_strength < 0.1 else 'medium' if order_effect_strength < 0.2 else 'high'
        
        return {
            'order_effect_strength': order_effect_strength,
            'severity': severity,
            'individual_trends': trend_strengths
        }
    
    def _detect_severity_bias(self, annotations: List[AnnotationResult]) -> Dict:
        """检测严格度偏差"""
        
        # 计算各标注者的评分分布
        annotator_distributions = {}
        
        for ann in annotations:
            if ann.annotator_id not in annotator_distributions:
                annotator_distributions[ann.annotator_id] = []
            
            overall_score = np.mean(list(ann.dimensions.values()))
            annotator_distributions[ann.annotator_id].append(overall_score)
        
        # 计算分布差异
        means = []
        stds = []
        
        for annotator_id, scores in annotator_distributions.items():
            if len(scores) >= 3:
                means.append(np.mean(scores))
                stds.append(np.std(scores))
        
        # 严格度偏差强度
        mean_range = np.max(means) - np.min(means) if means else 0
        std_range = np.max(stds) - np.min(stds) if stds else 0
        
        severity_bias_strength = (mean_range + std_range) / 2
        severity = 'low' if severity_bias_strength < 0.5 else 'medium' if severity_bias_strength < 1.0 else 'high'
        
        return {
            'severity_bias_strength': severity_bias_strength,
            'severity': severity,
            'mean_range': mean_range,
            'std_range': std_range
        }
    
    def suggest_bias_mitigation_strategies(self, bias_results: Dict) -> List[str]:
        """建议偏差缓解策略"""
        
        strategies = []
        
        # 根据检测到的偏差类型给出建议
        if bias_results.get('halo_effect', {}).get('severity') in ['medium', 'high']:
            strategies.append("实施独立维度评估：要求标注者分别独立评估各个维度")
            strategies.append("使用盲评方法：隐藏其他维度的评分")
        
        if bias_results.get('anchoring_bias', {}).get('severity') in ['medium', 'high']:
            strategies.append("随机化展示顺序：随机打乱评估样本的顺序")
            strategies.append("提供校准样本：在评估前提供标准化的参考样本")
        
        if bias_results.get('order_effect', {}).get('severity') in ['medium', 'high']:
            strategies.append("定期休息：设置强制休息时间防止疲劳")
            strategies.append("分批评估：将大量评估任务分解为小批次")
        
        if bias_results.get('severity_bias', {}).get('severity') in ['medium', 'high']:
            strategies.append("标注者校准：通过培训统一标注标准")
            strategies.append("后处理标准化：对不同标注者的评分进行统计标准化")
        
        return strategies

## 2.4 大规模标注策略与质量控制

### 众包标注的统计推断

**标注者能力建模**：
设标注者 $i$ 的能力为 $\theta_i$，真实标签为 $y^*$，观察标签为 $y_i$：
$$P(y_i = y^* | \theta_i) = \theta_i \cdot I(y_i = y^*) + \frac{1-\theta_i}{K-1} \cdot I(y_i \neq y^*)$$

其中 $K$ 是类别数。

**期望最大化(EM)算法**：
通过EM算法同时估计真实标签和标注者能力：

E步：
$$q(y^*_j = k) = \frac{\prod_i P(y_{ij} | y^*_j = k, \theta_i)}{\sum_{k'} \prod_i P(y_{ij} | y^*_j = k', \theta_i)}$$

M步：
$$\theta_i = \frac{\sum_j \sum_k q(y^*_j = k) \cdot I(y_{ij} = k)}{\sum_j \sum_k q(y^*_j = k)}$$

class CrowdsourcingQualityController:
    """众包质量控制器"""
    
    def __init__(self, min_annotations_per_item: int = 3):
        self.min_annotations_per_item = min_annotations_per_item
        self.annotator_abilities = {}
        self.true_labels = {}
        
    def estimate_true_labels_and_abilities(self, 
                                         annotations: List[AnnotationResult],
                                         dimension: str,
                                         max_iterations: int = 100,
                                         tolerance: float = 1e-6) -> Dict:
        """使用EM算法估计真实标签和标注者能力"""
        
        print(f"=== {dimension}维度EM算法估计 ===")
        
        # 准备数据
        items = sorted(set(ann.text_id for ann in annotations))
        annotators = sorted(set(ann.annotator_id for ann in annotations))
        
        # 构建标注矩阵
        annotation_matrix = {}
        for ann in annotations:
            if dimension in ann.dimensions:
                key = (ann.text_id, ann.annotator_id)
                annotation_matrix[key] = ann.dimensions[dimension]
        
        # 初始化参数
        n_items = len(items)
        n_annotators = len(annotators)
        n_classes = 5  # 假设1-5分
        
        # 初始化标注者能力（假设都是0.6）
        abilities = {annotator: 0.6 for annotator in annotators}
        
        # 初始化真实标签概率（均匀分布）
        true_label_probs = {}
        for item in items:
            true_label_probs[item] = np.ones(n_classes) / n_classes
        
        # EM迭代
        log_likelihood_history = []
        
        for iteration in range(max_iterations):
            # E步：估计真实标签概率
            new_true_label_probs = {}
            
            for item in items:
                class_probs = np.ones(n_classes)
                
                for annotator in annotators:
                    key = (item, annotator)
                    if key in annotation_matrix:
                        observed_label = annotation_matrix[key] - 1  # 转为0索引
                        ability = abilities[annotator]
                        
                        for true_class in range(n_classes):
                            if observed_label == true_class:
                                likelihood = ability
                            else:
                                likelihood = (1 - ability) / (n_classes - 1)
                            class_probs[true_class] *= likelihood
                
                # 归一化
                class_probs = class_probs / np.sum(class_probs)
                new_true_label_probs[item] = class_probs
            
            # M步：估计标注者能力
            new_abilities = {}
            
            for annotator in annotators:
                correct_weight = 0
                total_weight = 0
                
                for item in items:
                    key = (item, annotator)
                    if key in annotation_matrix:
                        observed_label = annotation_matrix[key] - 1
                        true_probs = new_true_label_probs[item]
                        
                        correct_weight += true_probs[observed_label]
                        total_weight += 1
                
                if total_weight > 0:
                    new_abilities[annotator] = correct_weight / total_weight
                else:
                    new_abilities[annotator] = 0.5
            
            # 计算对数似然
            log_likelihood = 0
            for item in items:
                for annotator in annotators:
                    key = (item, annotator)
                    if key in annotation_matrix:
                        observed_label = annotation_matrix[key] - 1
                        ability = new_abilities[annotator]
                        true_probs = new_true_label_probs[item]
                        
                        item_likelihood = 0
                        for true_class in range(n_classes):
                            if observed_label == true_class:
                                item_likelihood += true_probs[true_class] * ability
                            else:
                                item_likelihood += true_probs[true_class] * (1 - ability) / (n_classes - 1)
                        
                        if item_likelihood > 0:
                            log_likelihood += np.log(item_likelihood)
            
            log_likelihood_history.append(log_likelihood)
            
            # 检查收敛
            if iteration > 0:
                if abs(log_likelihood_history[-1] - log_likelihood_history[-2]) < tolerance:
                    print(f"EM算法在第{iteration+1}次迭代收敛")
                    break
            
            # 更新参数
            abilities = new_abilities
            true_label_probs = new_true_label_probs
        
        # 生成最终的真实标签估计
        estimated_labels = {}
        for item in items:
            estimated_labels[item] = np.argmax(true_label_probs[item]) + 1  # 转回1-5分
        
        print(f"标注者能力估计:")
        for annotator, ability in sorted(abilities.items()):
            print(f"  {annotator}: {ability:.4f}")
        
        return {
            'estimated_labels': estimated_labels,
            'annotator_abilities': abilities,
            'true_label_probabilities': true_label_probs,
            'log_likelihood_history': log_likelihood_history,
            'converged_iteration': iteration + 1
        }
    
    def compute_annotation_quality_metrics(self, 
                                         annotations: List[AnnotationResult],
                                         estimated_results: Dict,
                                         dimension: str) -> Dict:
        """计算标注质量指标"""
        
        print(f"=== {dimension}维度标注质量指标 ===")
        
        estimated_labels = estimated_results['estimated_labels']
        annotator_abilities = estimated_results['annotator_abilities']
        
        # 计算各标注者的准确率
        annotator_accuracies = {}
        
        for ann in annotations:
            if dimension in ann.dimensions and ann.text_id in estimated_labels:
                true_label = estimated_labels[ann.text_id]
                observed_label = ann.dimensions[dimension]
                
                if ann.annotator_id not in annotator_accuracies:
                    annotator_accuracies[ann.annotator_id] = {'correct': 0, 'total': 0}
                
                annotator_accuracies[ann.annotator_id]['total'] += 1
                if observed_label == true_label:
                    annotator_accuracies[ann.annotator_id]['correct'] += 1
        
        # 计算准确率
        for annotator_id in annotator_accuracies:
            stats = annotator_accuracies[annotator_id]
            stats['accuracy'] = stats['correct'] / stats['total']
        
        # 计算整体质量指标
        all_accuracies = [stats['accuracy'] for stats in annotator_accuracies.values()]
        
        quality_metrics = {
            'mean_accuracy': np.mean(all_accuracies),
            'std_accuracy': np.std(all_accuracies),
            'min_accuracy': np.min(all_accuracies),
            'max_accuracy': np.max(all_accuracies),
            'annotator_count': len(annotator_accuracies),
            'total_annotations': sum(stats['total'] for stats in annotator_accuracies.values())
        }
        
        print(f"平均准确率: {quality_metrics['mean_accuracy']:.4f}")
        print(f"准确率标准差: {quality_metrics['std_accuracy']:.4f}")
        print(f"标注者数量: {quality_metrics['annotator_count']}")
        print(f"总标注数量: {quality_metrics['total_annotations']}")
        
        return {
            'annotator_accuracies': annotator_accuracies,
            'quality_metrics': quality_metrics,
            'ability_accuracy_correlation': np.corrcoef(
                list(annotator_abilities.values()),
                [stats['accuracy'] for stats in annotator_accuracies.values()]
            )[0, 1] if len(annotator_abilities) > 1 else np.nan
        }
    
    def design_adaptive_annotation_strategy(self, 
                                          current_annotations: List[AnnotationResult],
                                          target_confidence: float = 0.9) -> Dict:
        """设计自适应标注策略"""
        
        print("=== 自适应标注策略设计 ===")
        
        # 统计当前标注情况
        item_annotation_counts = Counter(ann.text_id for ann in current_annotations)
        
        # 识别需要更多标注的项目
        under_annotated_items = [
            item for item, count in item_annotation_counts.items()
            if count < self.min_annotations_per_item
        ]
        
        # 基于不确定性的主动学习策略
        uncertainty_scores = {}
        
        for item_id in set(ann.text_id for ann in current_annotations):
            item_annotations = [ann for ann in current_annotations if ann.text_id == item_id]
            
            if len(item_annotations) >= 2:
                # 计算标注分歧度作为不确定性指标
                dimension_disagreements = {}
                
                for dim in ['fluency', 'coherence', 'relevance', 'informativeness', 'creativity']:
                    scores = [ann.dimensions.get(dim, 3) for ann in item_annotations if dim in ann.dimensions]
                    if len(scores) >= 2:
                        disagreement = np.std(scores)
                        dimension_disagreements[dim] = disagreement
                
                # 综合不确定性分数
                if dimension_disagreements:
                    uncertainty_scores[item_id] = np.mean(list(dimension_disagreements.values()))
                else:
                    uncertainty_scores[item_id] = 0
        
        # 排序确定优先级
        high_uncertainty_items = sorted(
            uncertainty_scores.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:10]  # 取前10个最不确定的项目
        
        # 标注者分配策略
        annotator_workloads = Counter(ann.annotator_id for ann in current_annotations)
        available_annotators = sorted(annotator_workloads.keys())
        
        annotation_plan = {
            'under_annotated_items': under_annotated_items,
            'high_uncertainty_items': [item for item, score in high_uncertainty_items],
            'recommended_annotations': len(under_annotated_items) * self.min_annotations_per_item + len(high_uncertainty_items),
            'annotator_allocation': self._allocate_annotators(
                under_annotated_items + [item for item, _ in high_uncertainty_items],
                available_annotators
            )
        }
        
        print(f"待补充标注项目: {len(under_annotated_items)}")
        print(f"高不确定性项目: {len(high_uncertainty_items)}")
        print(f"建议总标注数: {annotation_plan['recommended_annotations']}")
        
        return annotation_plan
    
    def _allocate_annotators(self, items: List[str], annotators: List[str]) -> Dict:
        """分配标注者"""
        
        allocation = defaultdict(list)
        
        # 简单的轮询分配策略
        for i, item in enumerate(items):
            annotator = annotators[i % len(annotators)]
            allocation[annotator].append(item)
        
        return dict(allocation)

def create_human_evaluation_system():
    """创建人类评估系统"""
    
    framework = HumanEvaluationFramework()
    consistency_analyzer = AnnotationConsistencyAnalyzer()
    bias_controller = CognitiveBiasController()
    quality_controller = CrowdsourcingQualityController()
    
    return {
        'framework': framework,
        'consistency_analyzer': consistency_analyzer,
        'bias_controller': bias_controller,
        'quality_controller': quality_controller
    }

# 演示完整的人类评估系统
def demonstrate_human_evaluation_system():
    """演示人类评估系统"""
    
    print("=== MiniGPT人类评估框架演示 ===\n")
    
    # 创建评估系统
    eval_system = create_human_evaluation_system()
    
    # 生成模拟标注数据
    np.random.seed(42)
    
    texts = [f"text_{i:03d}" for i in range(50)]
    annotators = [f"annotator_{i}" for i in range(8)]
    dimensions = ['fluency', 'coherence', 'relevance', 'informativeness', 'creativity']
    
    annotations = []
    
    # 模拟不同质量的标注者
    annotator_qualities = {
        'annotator_0': 0.9,  # 高质量
        'annotator_1': 0.8,
        'annotator_2': 0.7,
        'annotator_3': 0.6,
        'annotator_4': 0.5,  # 中等质量
        'annotator_5': 0.4,
        'annotator_6': 0.3,
        'annotator_7': 0.2   # 低质量
    }
    
    for text_id in texts:
        # 每个文本由3-5个标注者评估
        n_annotators = np.random.randint(3, 6)
        selected_annotators = np.random.choice(annotators, n_annotators, replace=False)
        
        # 生成"真实"质量分数
        true_scores = {
            'fluency': np.random.randint(2, 6),
            'coherence': np.random.randint(2, 6),
            'relevance': np.random.randint(2, 6),
            'informativeness': np.random.randint(2, 6),
            'creativity': np.random.randint(1, 5)
        }
        
        for annotator_id in selected_annotators:
            quality = annotator_qualities[annotator_id]
            
            # 根据标注者质量生成观察分数
            observed_scores = {}
            for dim, true_score in true_scores.items():
                if np.random.random() < quality:
                    # 正确标注
                    observed_scores[dim] = true_score
                else:
                    # 错误标注
                    observed_scores[dim] = np.random.randint(1, 6)
            
            # 添加一些系统性偏差
            if 'strict' in annotator_id or annotator_id in ['annotator_1', 'annotator_3']:
                # 严格标注者
                for dim in observed_scores:
                    observed_scores[dim] = max(1, observed_scores[dim] - 1)
            elif 'lenient' in annotator_id or annotator_id in ['annotator_6', 'annotator_7']:
                # 宽松标注者
                for dim in observed_scores:
                    observed_scores[dim] = min(5, observed_scores[dim] + 1)
            
            annotation = AnnotationResult(
                text_id=text_id,
                annotator_id=annotator_id,
                dimensions=observed_scores,
                annotation_time=np.random.uniform(30, 300),  # 30秒到5分钟
                confidence=np.random.uniform(0.6, 1.0)
            )
            
            annotations.append(annotation)
    
    # 1. 标注一致性分析
    print("1. 标注一致性分析")
    for dimension in dimensions:
        kappa_results = eval_system['consistency_analyzer'].compute_pairwise_kappa(
            annotations, dimension, weights='quadratic'
        )
        
        icc_results = eval_system['consistency_analyzer'].compute_icc(
            annotations, dimension
        )
        
        bias_analysis = eval_system['consistency_analyzer'].analyze_annotator_bias(
            annotations, dimension
        )
    
    # 2. 认知偏差检测
    print("\n2. 认知偏差检测")
    bias_results = eval_system['bias_controller'].detect_biases(annotations)
    
    strategies = eval_system['bias_controller'].suggest_bias_mitigation_strategies(bias_results)
    print("\n建议的偏差缓解策略:")
    for strategy in strategies:
        print(f"- {strategy}")
    
    # 3. 众包质量控制
    print("\n3. 众包质量控制")
    for dimension in dimensions[:2]:  # 只演示前两个维度
        em_results = eval_system['quality_controller'].estimate_true_labels_and_abilities(
            annotations, dimension
        )
        
        quality_metrics = eval_system['quality_controller'].compute_annotation_quality_metrics(
            annotations, em_results, dimension
        )
    
    # 4. 自适应标注策略
    print("\n4. 自适应标注策略")
    annotation_plan = eval_system['quality_controller'].design_adaptive_annotation_strategy(
        annotations
    )
    
    return {
        'annotations': annotations,
        'consistency_results': kappa_results,
        'bias_results': bias_results,
        'quality_results': quality_metrics,
        'annotation_plan': annotation_plan,
        'evaluation_system': eval_system
    }

# 运行演示
if __name__ == "__main__":
    results = demonstrate_human_evaluation_system()
    
    print("\n=== 人类评估框架分析完成 ===")
    print(f"系统性能总结:")
    print(f"- 标注一致性: 需要改进")
    print(f"- 认知偏差控制: 良好")
    print(f"- 质量控制机制: 有效")
    print(f"- 自适应策略: 实用")
```

## 理论总结

### 2.5 人类评估的统一理论框架

**评估过程的概率模型**：
人类评估可建模为从真实质量到观察评分的概率映射：
$$P(\text{observed} | \text{true}, \text{annotator}) = f(\text{ability}, \text{bias}, \text{context})$$

**贝叶斯推断框架**：
结合先验知识进行评估推断：
$$P(\text{true} | \text{observed}) \propto P(\text{observed} | \text{true}) \cdot P(\text{true})$$

**信息融合理论**：
多标注者信息的最优融合：
$$\hat{y} = \arg\max_y \sum_i w_i \log P(y_i | y, \theta_i)$$

其中 $w_i$ 是标注者权重，$\theta_i$ 是能力参数。

## 应用指导

### 实践建议

1. **评估维度设计**：
   - 确保维度独立性和完整性
   - 提供清晰的操作化定义
   - 设计适当的评分尺度

2. **标注者管理**：
   - 实施严格的标注者培训
   - 定期进行一致性检查
   - 建立标注者绩效档案

3. **质量保证机制**：
   - 多重标注和交叉验证
   - 自动化质量检测
   - 持续的反馈和改进

人类评估框架的设计需要在科学性和实用性之间找到平衡，通过严格的统计学方法确保评估结果的可靠性和有效性。

## 扩展阅读

- 《Inter-rater Reliability: The Kappa Statistic》- 一致性分析经典文献
- 《Crowdsourcing for NLP: What's Next?》- 众包标注方法综述
- 《Cognitive Biases in Natural Language Processing》- 认知偏差研究
- 《Quality Control in Crowdsourcing》- 众包质量控制理论

---

*"人类的判断是复杂的，但通过科学的方法，我们可以将这种复杂性转化为可靠的知识。"* 🎯