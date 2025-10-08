# 04 评估指标与效果分析

> **从定量测评到定性分析：全面评估监督微调效果的科学体系**

## 核心思想

评估是监督微调的最后一环，也是最关键的一环。它不仅要回答"模型表现如何"，更要深入分析"为什么这样表现"和"如何进一步改进"。评估的科学性直接决定了我们对模型能力的理解深度和改进方向的准确性。

**关键洞察**：
- **多维评估**：单一指标无法全面反映模型能力
- **任务特异性**：不同任务需要不同的评估框架
- **人机协同**：自动评估与人工评估的有机结合
- **动态分析**：评估应该是持续的、迭代的过程

评估的数学框架可以表示为：
$$\text{Evaluation} = f(\text{Performance}, \text{Robustness}, \text{Efficiency}, \text{Safety})$$

其中每个维度都需要专门的指标和分析方法。

## 4.1 自动评估指标的数学原理

### 词汇层面指标的统计学基础

**BLEU Score**的数学定义：
$$\text{BLEU} = \text{BP} \cdot \exp\left(\sum_{n=1}^{N} w_n \log p_n\right)$$

其中：
- $p_n$：n-gram精确率
- $w_n$：权重（通常均匀分布）
- $\text{BP}$：简洁性惩罚

**ROUGE Score**的召回率基础：
$$\text{ROUGE-N} = \frac{\sum_{S \in \text{Ref}} \sum_{\text{gram}_n \in S} \text{Count}_{\text{match}}(\text{gram}_n)}{\sum_{S \in \text{Ref}} \sum_{\text{gram}_n \in S} \text{Count}(\text{gram}_n)}$$

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
from collections import Counter, defaultdict
import math
import re
from scipy import stats
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass

@dataclass
class EvaluationResult:
    """评估结果数据结构"""
    task_name: str
    metric_scores: Dict[str, float]
    detailed_analysis: Dict[str, Any]
    timestamp: float
    model_info: Dict[str, Any]

class AutomaticEvaluator:
    """自动评估器"""
    
    def __init__(self, metrics: List[str] = None):
        if metrics is None:
            metrics = ['bleu', 'rouge', 'bertscore', 'distinct', 'length_stats']
        
        self.metrics = metrics
        self.evaluation_history = []
        
    def evaluate_generation(self, predictions: List[str], references: List[List[str]], 
                          task_type: str = 'general') -> Dict[str, float]:
        """评估生成任务"""
        
        print(f"=== {task_type.upper()} 生成任务评估 ===")
        
        results = {}
        
        # 1. BLEU评估
        if 'bleu' in self.metrics:
            bleu_scores = self._compute_bleu_scores(predictions, references)
            results.update(bleu_scores)
        
        # 2. ROUGE评估
        if 'rouge' in self.metrics:
            rouge_scores = self._compute_rouge_scores(predictions, references)
            results.update(rouge_scores)
        
        # 3. BERTScore评估
        if 'bertscore' in self.metrics:
            bert_scores = self._compute_bert_scores(predictions, references)
            results.update(bert_scores)
        
        # 4. 多样性评估
        if 'distinct' in self.metrics:
            diversity_scores = self._compute_diversity_scores(predictions)
            results.update(diversity_scores)
        
        # 5. 长度统计
        if 'length_stats' in self.metrics:
            length_stats = self._compute_length_statistics(predictions, references)
            results.update(length_stats)
        
        # 6. 语言质量评估
        if 'fluency' in self.metrics:
            fluency_scores = self._compute_fluency_scores(predictions)
            results.update(fluency_scores)
        
        # 输出结果
        print("评估结果:")
        for metric, score in results.items():
            if isinstance(score, (int, float)):
                print(f"  {metric:20s}: {score:.4f}")
        
        return results
    
    def _compute_bleu_scores(self, predictions: List[str], 
                           references: List[List[str]]) -> Dict[str, float]:
        """计算BLEU分数"""
        
        def tokenize(text):
            """简单分词"""
            return text.lower().split()
        
        def compute_ngram_precision(pred_tokens, ref_tokens_list, n):
            """计算n-gram精确率"""
            pred_ngrams = Counter([tuple(pred_tokens[i:i+n]) 
                                 for i in range(len(pred_tokens)-n+1)])
            
            max_ref_ngrams = Counter()
            for ref_tokens in ref_tokens_list:
                ref_ngrams = Counter([tuple(ref_tokens[i:i+n]) 
                                    for i in range(len(ref_tokens)-n+1)])
                for ngram, count in ref_ngrams.items():
                    max_ref_ngrams[ngram] = max(max_ref_ngrams[ngram], count)
            
            clipped_counts = 0
            total_counts = 0
            
            for ngram, count in pred_ngrams.items():
                clipped_counts += min(count, max_ref_ngrams[ngram])
                total_counts += count
            
            return clipped_counts / max(total_counts, 1)
        
        def compute_brevity_penalty(pred_len, ref_lens):
            """计算简洁性惩罚"""
            closest_ref_len = min(ref_lens, key=lambda x: abs(x - pred_len))
            if pred_len > closest_ref_len:
                return 1.0
            else:
                return math.exp(1 - closest_ref_len / max(pred_len, 1))
        
        # 计算BLEU-1到BLEU-4
        bleu_scores = {}
        total_precisions = {1: [], 2: [], 3: [], 4: []}
        total_bp = []
        
        for pred, refs in zip(predictions, references):
            pred_tokens = tokenize(pred)
            ref_tokens_list = [tokenize(ref) for ref in refs]
            
            # 计算各阶n-gram精确率
            precisions = {}
            for n in range(1, 5):
                precisions[n] = compute_ngram_precision(pred_tokens, ref_tokens_list, n)
                total_precisions[n].append(precisions[n])
            
            # 计算简洁性惩罚
            ref_lens = [len(ref_tokens) for ref_tokens in ref_tokens_list]
            bp = compute_brevity_penalty(len(pred_tokens), ref_lens)
            total_bp.append(bp)
        
        # 计算平均BLEU分数
        avg_bp = np.mean(total_bp)
        
        for n in range(1, 5):
            avg_precision = np.mean(total_precisions[n])
            bleu_n = avg_bp * avg_precision
            bleu_scores[f'bleu_{n}'] = bleu_n
        
        # 计算几何平均BLEU-4
        if all(p > 0 for p in total_precisions.values()):
            geo_mean = 1
            for n in range(1, 5):
                geo_mean *= np.mean(total_precisions[n]) ** 0.25
            bleu_scores['bleu_4_geo'] = avg_bp * geo_mean
        else:
            bleu_scores['bleu_4_geo'] = 0.0
        
        return bleu_scores
    
    def _compute_rouge_scores(self, predictions: List[str], 
                            references: List[List[str]]) -> Dict[str, float]:
        """计算ROUGE分数"""
        
        def tokenize(text):
            return text.lower().split()
        
        def compute_rouge_n(pred_tokens, ref_tokens_list, n):
            """计算ROUGE-N"""
            pred_ngrams = set([tuple(pred_tokens[i:i+n]) 
                             for i in range(len(pred_tokens)-n+1)])
            
            max_overlap = 0
            max_ref_ngrams = 0
            
            for ref_tokens in ref_tokens_list:
                ref_ngrams = set([tuple(ref_tokens[i:i+n]) 
                                for i in range(len(ref_tokens)-n+1)])
                
                overlap = len(pred_ngrams.intersection(ref_ngrams))
                if overlap > max_overlap:
                    max_overlap = overlap
                    max_ref_ngrams = len(ref_ngrams)
            
            return max_overlap / max(max_ref_ngrams, 1)
        
        def compute_rouge_l(pred_tokens, ref_tokens_list):
            """计算ROUGE-L（最长公共子序列）"""
            
            def lcs_length(x, y):
                m, n = len(x), len(y)
                dp = [[0] * (n + 1) for _ in range(m + 1)]
                
                for i in range(1, m + 1):
                    for j in range(1, n + 1):
                        if x[i-1] == y[j-1]:
                            dp[i][j] = dp[i-1][j-1] + 1
                        else:
                            dp[i][j] = max(dp[i-1][j], dp[i][j-1])
                
                return dp[m][n]
            
            max_lcs = 0
            best_ref_len = 0
            
            for ref_tokens in ref_tokens_list:
                lcs_len = lcs_length(pred_tokens, ref_tokens)
                if lcs_len > max_lcs:
                    max_lcs = lcs_len
                    best_ref_len = len(ref_tokens)
            
            if max_lcs == 0:
                return 0.0
            
            precision = max_lcs / max(len(pred_tokens), 1)
            recall = max_lcs / max(best_ref_len, 1)
            
            if precision + recall == 0:
                return 0.0
            
            f1 = 2 * precision * recall / (precision + recall)
            return f1
        
        rouge_scores = {}
        
        # ROUGE-1, ROUGE-2
        for n in [1, 2]:
            scores = []
            for pred, refs in zip(predictions, references):
                pred_tokens = tokenize(pred)
                ref_tokens_list = [tokenize(ref) for ref in refs]
                score = compute_rouge_n(pred_tokens, ref_tokens_list, n)
                scores.append(score)
            
            rouge_scores[f'rouge_{n}'] = np.mean(scores)
        
        # ROUGE-L
        rouge_l_scores = []
        for pred, refs in zip(predictions, references):
            pred_tokens = tokenize(pred)
            ref_tokens_list = [tokenize(ref) for ref in refs]
            score = compute_rouge_l(pred_tokens, ref_tokens_list)
            rouge_l_scores.append(score)
        
        rouge_scores['rouge_l'] = np.mean(rouge_l_scores)
        
        return rouge_scores
    
    def _compute_bert_scores(self, predictions: List[str], 
                           references: List[List[str]]) -> Dict[str, float]:
        """计算BERTScore（简化版本）"""
        
        # 注意：这是一个简化实现，实际BERTScore需要使用预训练的BERT模型
        # 这里我们用余弦相似度来模拟语义相似性
        
        def simple_embedding(text):
            """简单的文本嵌入（实际应使用BERT）"""
            # 使用字符级特征作为简化的语义表示
            char_counts = Counter(text.lower())
            # 创建固定长度的特征向量
            vocab = 'abcdefghijklmnopqrstuvwxyz0123456789 '
            features = np.array([char_counts.get(c, 0) for c in vocab])
            # 归一化
            norm = np.linalg.norm(features)
            return features / max(norm, 1e-8)
        
        bert_scores = []
        
        for pred, refs in zip(predictions, references):
            pred_emb = simple_embedding(pred)
            
            max_similarity = 0
            for ref in refs:
                ref_emb = simple_embedding(ref)
                similarity = np.dot(pred_emb, ref_emb)
                max_similarity = max(max_similarity, similarity)
            
            bert_scores.append(max_similarity)
        
        return {
            'bertscore_f1': np.mean(bert_scores),
            'bertscore_precision': np.mean(bert_scores),  # 简化版本
            'bertscore_recall': np.mean(bert_scores)      # 简化版本
        }
    
    def _compute_diversity_scores(self, predictions: List[str]) -> Dict[str, float]:
        """计算多样性分数"""
        
        def tokenize(text):
            return text.lower().split()
        
        all_tokens = []
        all_bigrams = []
        all_trigrams = []
        
        for pred in predictions:
            tokens = tokenize(pred)
            all_tokens.extend(tokens)
            
            # n-grams
            bigrams = [tuple(tokens[i:i+2]) for i in range(len(tokens)-1)]
            trigrams = [tuple(tokens[i:i+3]) for i in range(len(tokens)-2)]
            
            all_bigrams.extend(bigrams)
            all_trigrams.extend(trigrams)
        
        # 计算distinct-n
        distinct_1 = len(set(all_tokens)) / max(len(all_tokens), 1)
        distinct_2 = len(set(all_bigrams)) / max(len(all_bigrams), 1)
        distinct_3 = len(set(all_trigrams)) / max(len(all_trigrams), 1)
        
        # 计算self-BLEU（衡量输出间的相似性，越低表示越多样）
        self_bleu_scores = []
        
        for i, pred in enumerate(predictions):
            other_preds = predictions[:i] + predictions[i+1:]
            if other_preds:
                # 使用其他预测作为参考计算BLEU
                bleu_with_others = self._compute_bleu_scores([pred], [other_preds])
                self_bleu_scores.append(bleu_with_others.get('bleu_4_geo', 0))
        
        avg_self_bleu = np.mean(self_bleu_scores) if self_bleu_scores else 0
        
        return {
            'distinct_1': distinct_1,
            'distinct_2': distinct_2,
            'distinct_3': distinct_3,
            'self_bleu': avg_self_bleu,
            'diversity_score': (distinct_1 + distinct_2 + distinct_3) / 3
        }
    
    def _compute_length_statistics(self, predictions: List[str], 
                                 references: List[List[str]]) -> Dict[str, float]:
        """计算长度统计"""
        
        pred_lengths = [len(pred.split()) for pred in predictions]
        ref_lengths = []
        
        for refs in references:
            # 使用最接近预测长度的参考长度
            pred_len = len(predictions[references.index(refs)].split())
            closest_ref = min(refs, key=lambda x: abs(len(x.split()) - pred_len))
            ref_lengths.append(len(closest_ref.split()))
        
        # 统计指标
        pred_mean = np.mean(pred_lengths)
        pred_std = np.std(pred_lengths)
        ref_mean = np.mean(ref_lengths)
        ref_std = np.std(ref_lengths)
        
        # 长度比率
        length_ratios = [p/max(r, 1) for p, r in zip(pred_lengths, ref_lengths)]
        avg_length_ratio = np.mean(length_ratios)
        
        # 长度一致性（预测长度与参考长度的相关性）
        length_correlation = np.corrcoef(pred_lengths, ref_lengths)[0, 1] if len(pred_lengths) > 1 else 0
        
        return {
            'pred_length_mean': pred_mean,
            'pred_length_std': pred_std,
            'ref_length_mean': ref_mean,
            'ref_length_std': ref_std,
            'length_ratio': avg_length_ratio,
            'length_correlation': length_correlation
        }
    
    def _compute_fluency_scores(self, predictions: List[str]) -> Dict[str, float]:
        """计算流畅性分数（简化版本）"""
        
        def compute_perplexity_proxy(text):
            """计算困惑度代理指标"""
            tokens = text.lower().split()
            if len(tokens) < 2:
                return 1.0
            
            # 简化：使用重复度作为流畅性指标
            unique_tokens = len(set(tokens))
            repetition_rate = 1 - (unique_tokens / len(tokens))
            
            # 越少重复越流畅
            fluency = 1 - repetition_rate
            return fluency
        
        def compute_grammar_proxy(text):
            """计算语法正确性代理指标"""
            # 简化：检查基本的语法模式
            sentences = text.split('.')
            avg_sentence_length = np.mean([len(s.split()) for s in sentences if s.strip()])
            
            # 合理的句子长度表示更好的语法结构
            if 5 <= avg_sentence_length <= 25:
                grammar_score = 1.0
            else:
                grammar_score = max(0, 1 - abs(avg_sentence_length - 15) / 15)
            
            return grammar_score
        
        fluency_scores = []
        grammar_scores = []
        
        for pred in predictions:
            fluency = compute_perplexity_proxy(pred)
            grammar = compute_grammar_proxy(pred)
            
            fluency_scores.append(fluency)
            grammar_scores.append(grammar)
        
        return {
            'fluency_score': np.mean(fluency_scores),
            'grammar_score': np.mean(grammar_scores),
            'overall_quality': (np.mean(fluency_scores) + np.mean(grammar_scores)) / 2
        }

def demonstrate_automatic_evaluation():
    """演示自动评估系统"""
    
    print("=== 自动评估系统演示 ===")
    
    # 创建测试数据
    test_cases = {
        'high_quality': {
            'predictions': [
                "The quick brown fox jumps over the lazy dog with great agility.",
                "Machine learning algorithms can process vast amounts of data efficiently.",
                "Natural language processing enables computers to understand human communication."
            ],
            'references': [
                ["The quick brown fox jumps over the lazy dog.", "A fast brown fox leaps over a sleepy dog."],
                ["Machine learning can handle large datasets effectively.", "ML algorithms process big data well."],
                ["NLP helps computers understand human language.", "Natural language processing aids human-computer interaction."]
            ]
        },
        'medium_quality': {
            'predictions': [
                "Fox brown quick jump over dog lazy.",
                "Machine learn algorithm process data much.",
                "Language process natural computer understand human."
            ],
            'references': [
                ["The quick brown fox jumps over the lazy dog.", "A fast brown fox leaps over a sleepy dog."],
                ["Machine learning can handle large datasets effectively.", "ML algorithms process big data well."],
                ["NLP helps computers understand human language.", "Natural language processing aids human-computer interaction."]
            ]
        },
        'low_quality': {
            'predictions': [
                "Fox fox fox jump jump jump.",
                "Data data process process algorithm.",
                "Computer computer language language understand."
            ],
            'references': [
                ["The quick brown fox jumps over the lazy dog.", "A fast brown fox leaps over a sleepy dog."],
                ["Machine learning can handle large datasets effectively.", "ML algorithms process big data well."],
                ["NLP helps computers understand human language.", "Natural language processing aids human-computer interaction."]
            ]
        }
    }
    
    evaluator = AutomaticEvaluator()
    
    results = {}
    
    for quality_level, data in test_cases.items():
        print(f"\\n--- {quality_level.upper()} 质量样本评估 ---")
        
        evaluation_result = evaluator.evaluate_generation(
            data['predictions'], 
            data['references'],
            task_type=quality_level
        )
        
        results[quality_level] = evaluation_result
    
    # 比较分析
    print("\\n=== 质量水平比较 ===")
    
    key_metrics = ['bleu_4_geo', 'rouge_l', 'distinct_1', 'fluency_score']
    
    print(f"{'质量水平':12s}", end="")
    for metric in key_metrics:
        print(f"{metric:>12s}", end="")
    print()
    print("-" * (12 + 12 * len(key_metrics)))
    
    for quality_level in ['high_quality', 'medium_quality', 'low_quality']:
        print(f"{quality_level:12s}", end="")
        for metric in key_metrics:
            score = results[quality_level].get(metric, 0)
            print(f"{score:12.4f}", end="")
        print()
    
    # 指标相关性分析
    print("\\n=== 指标相关性分析 ===")
    
    quality_rankings = {'high_quality': 3, 'medium_quality': 2, 'low_quality': 1}
    
    for metric in key_metrics:
        metric_scores = [results[ql][metric] for ql in quality_rankings.keys()]
        quality_scores = list(quality_rankings.values())
        
        correlation = np.corrcoef(metric_scores, quality_scores)[0, 1]
        print(f"{metric:15s}: 与质量相关性 = {correlation:.3f}")
    
    return results

class TaskSpecificEvaluator:
    """任务特定评估器"""
    
    def __init__(self, task_type: str):
        self.task_type = task_type
        self.evaluation_methods = self._get_task_specific_methods()
    
    def _get_task_specific_methods(self) -> Dict[str, callable]:
        """获取任务特定的评估方法"""
        
        methods = {
            'qa': {
                'exact_match': self._exact_match,
                'f1_score': self._f1_score,
                'answer_coverage': self._answer_coverage,
                'answer_relevance': self._answer_relevance
            },
            'summarization': {
                'rouge_scores': self._rouge_evaluation,
                'factual_consistency': self._factual_consistency,
                'abstractiveness': self._abstractiveness,
                'coverage': self._coverage
            },
            'translation': {
                'bleu_score': self._bleu_evaluation,
                'meteor_score': self._meteor_evaluation,
                'adequacy': self._adequacy_evaluation,
                'fluency': self._fluency_evaluation
            },
            'dialogue': {
                'response_relevance': self._response_relevance,
                'coherence': self._dialogue_coherence,
                'engagement': self._engagement_score,
                'safety': self._safety_evaluation
            }
        }
        
        return methods.get(self.task_type, {})
    
    def evaluate(self, predictions: List[str], references: List[str], 
                contexts: List[str] = None) -> Dict[str, float]:
        """执行任务特定评估"""
        
        print(f"=== {self.task_type.upper()} 任务评估 ===")
        
        results = {}
        
        for method_name, method_func in self.evaluation_methods.items():
            try:
                if method_name in ['answer_coverage', 'factual_consistency', 'coverage'] and contexts:
                    score = method_func(predictions, references, contexts)
                else:
                    score = method_func(predictions, references)
                
                results[method_name] = score
                print(f"  {method_name:20s}: {score:.4f}")
                
            except Exception as e:
                print(f"  {method_name:20s}: 计算失败 ({str(e)})")
                results[method_name] = 0.0
        
        return results
    
    def _exact_match(self, predictions: List[str], references: List[str]) -> float:
        """精确匹配评估"""
        exact_matches = 0
        
        for pred, ref in zip(predictions, references):
            # 标准化文本
            pred_normalized = pred.strip().lower()
            ref_normalized = ref.strip().lower()
            
            if pred_normalized == ref_normalized:
                exact_matches += 1
        
        return exact_matches / len(predictions)
    
    def _f1_score(self, predictions: List[str], references: List[str]) -> float:
        """F1分数评估（词汇级别）"""
        
        def compute_f1(pred_words, ref_words):
            pred_set = set(pred_words)
            ref_set = set(ref_words)
            
            if not pred_set and not ref_set:
                return 1.0
            if not pred_set or not ref_set:
                return 0.0
            
            intersection = pred_set.intersection(ref_set)
            precision = len(intersection) / len(pred_set)
            recall = len(intersection) / len(ref_set)
            
            if precision + recall == 0:
                return 0.0
            
            return 2 * precision * recall / (precision + recall)
        
        f1_scores = []
        for pred, ref in zip(predictions, references):
            pred_words = pred.lower().split()
            ref_words = ref.lower().split()
            
            f1 = compute_f1(pred_words, ref_words)
            f1_scores.append(f1)
        
        return np.mean(f1_scores)
    
    def _answer_coverage(self, predictions: List[str], references: List[str], 
                        contexts: List[str]) -> float:
        """答案覆盖度评估"""
        
        coverage_scores = []
        
        for pred, ref, context in zip(predictions, references, contexts):
            # 提取关键信息（简化版本）
            context_words = set(context.lower().split())
            ref_words = set(ref.lower().split())
            pred_words = set(pred.lower().split())
            
            # 计算参考答案中来自上下文的关键词
            key_words = ref_words.intersection(context_words)
            
            if not key_words:
                coverage_scores.append(1.0)  # 如果没有关键词，认为完全覆盖
            else:
                covered_keys = key_words.intersection(pred_words)
                coverage = len(covered_keys) / len(key_words)
                coverage_scores.append(coverage)
        
        return np.mean(coverage_scores)
    
    def _answer_relevance(self, predictions: List[str], references: List[str]) -> float:
        """答案相关性评估"""
        
        def compute_semantic_similarity(text1, text2):
            """简化的语义相似度计算"""
            words1 = set(text1.lower().split())
            words2 = set(text2.lower().split())
            
            if not words1 and not words2:
                return 1.0
            if not words1 or not words2:
                return 0.0
            
            intersection = words1.intersection(words2)
            union = words1.union(words2)
            
            return len(intersection) / len(union)
        
        relevance_scores = []
        
        for pred, ref in zip(predictions, references):
            similarity = compute_semantic_similarity(pred, ref)
            relevance_scores.append(similarity)
        
        return np.mean(relevance_scores)
    
    def _rouge_evaluation(self, predictions: List[str], references: List[str]) -> float:
        """ROUGE评估（用于摘要任务）"""
        
        # 使用之前实现的ROUGE计算
        evaluator = AutomaticEvaluator(['rouge'])
        rouge_results = evaluator._compute_rouge_scores(predictions, [[ref] for ref in references])
        
        # 返回ROUGE-L作为主要指标
        return rouge_results.get('rouge_l', 0.0)
    
    def _factual_consistency(self, predictions: List[str], references: List[str], 
                           contexts: List[str]) -> float:
        """事实一致性评估"""
        
        consistency_scores = []
        
        for pred, ref, context in zip(predictions, references, contexts):
            # 简化的事实检查：检查预测是否包含与上下文矛盾的信息
            context_entities = self._extract_entities(context)
            pred_entities = self._extract_entities(pred)
            
            # 检查是否有明显的事实错误
            contradictions = 0
            total_claims = len(pred_entities)
            
            if total_claims == 0:
                consistency_scores.append(1.0)
                continue
            
            for pred_entity in pred_entities:
                # 简化检查：如果预测实体不在上下文中，可能是错误
                if pred_entity not in context_entities and len(pred_entity) > 3:
                    contradictions += 1
            
            consistency = 1 - (contradictions / total_claims)
            consistency_scores.append(max(0, consistency))
        
        return np.mean(consistency_scores)
    
    def _extract_entities(self, text: str) -> List[str]:
        """简化的实体提取"""
        # 提取大写开头的词汇作为实体
        words = text.split()
        entities = [word for word in words if word[0].isupper() and len(word) > 2]
        return entities
    
    def _abstractiveness(self, predictions: List[str], references: List[str]) -> float:
        """抽象性评估（用于摘要任务）"""
        
        abstractiveness_scores = []
        
        for pred, ref in zip(predictions, references):
            pred_words = set(pred.lower().split())
            ref_words = set(ref.lower().split())
            
            # 计算新词比例（不在参考中的词）
            if len(pred_words) == 0:
                abstractiveness_scores.append(0.0)
            else:
                novel_words = pred_words - ref_words
                abstractiveness = len(novel_words) / len(pred_words)
                abstractiveness_scores.append(abstractiveness)
        
        return np.mean(abstractiveness_scores)
    
    def _coverage(self, predictions: List[str], references: List[str], 
                 contexts: List[str]) -> float:
        """内容覆盖度评估"""
        
        coverage_scores = []
        
        for pred, ref, context in zip(predictions, references, contexts):
            # 重要内容词（去除停用词）
            stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
            
            context_words = set(word.lower() for word in context.split() if word.lower() not in stopwords)
            pred_words = set(word.lower() for word in pred.split() if word.lower() not in stopwords)
            
            if len(context_words) == 0:
                coverage_scores.append(1.0)
            else:
                covered = context_words.intersection(pred_words)
                coverage = len(covered) / len(context_words)
                coverage_scores.append(coverage)
        
        return np.mean(coverage_scores)
    
    def _bleu_evaluation(self, predictions: List[str], references: List[str]) -> float:
        """BLEU评估（用于翻译任务）"""
        evaluator = AutomaticEvaluator(['bleu'])
        bleu_results = evaluator._compute_bleu_scores(predictions, [[ref] for ref in references])
        return bleu_results.get('bleu_4_geo', 0.0)
    
    def _meteor_evaluation(self, predictions: List[str], references: List[str]) -> float:
        """METEOR评估（简化版本）"""
        # 简化的METEOR实现，主要考虑同义词和词根
        meteor_scores = []
        
        for pred, ref in zip(predictions, references):
            pred_words = pred.lower().split()
            ref_words = ref.lower().split()
            
            # 计算对齐词汇数
            aligned = 0
            for word in pred_words:
                if word in ref_words:
                    aligned += 1
            
            if len(pred_words) == 0:
                meteor_scores.append(0.0)
            else:
                precision = aligned / len(pred_words)
                recall = aligned / max(len(ref_words), 1)
                
                if precision + recall == 0:
                    f_mean = 0
                else:
                    f_mean = (precision * recall) / (0.5 * precision + 0.5 * recall)
                
                meteor_scores.append(f_mean)
        
        return np.mean(meteor_scores)
    
    def _adequacy_evaluation(self, predictions: List[str], references: List[str]) -> float:
        """充分性评估"""
        # 检查翻译是否充分传达了原文信息
        return self._f1_score(predictions, references)  # 简化为F1分数
    
    def _fluency_evaluation(self, predictions: List[str], references: List[str]) -> float:
        """流畅性评估"""
        evaluator = AutomaticEvaluator(['fluency'])
        fluency_results = evaluator._compute_fluency_scores(predictions)
        return fluency_results.get('fluency_score', 0.0)
    
    def _response_relevance(self, predictions: List[str], references: List[str]) -> float:
        """响应相关性评估（对话任务）"""
        return self._answer_relevance(predictions, references)
    
    def _dialogue_coherence(self, predictions: List[str], references: List[str]) -> float:
        """对话连贯性评估"""
        
        coherence_scores = []
        
        for pred, ref in zip(predictions, references):
            # 简化的连贯性检查：句子之间的词汇重叠
            pred_sentences = pred.split('.')
            
            if len(pred_sentences) < 2:
                coherence_scores.append(1.0)
                continue
            
            sentence_overlaps = []
            for i in range(len(pred_sentences) - 1):
                sent1_words = set(pred_sentences[i].lower().split())
                sent2_words = set(pred_sentences[i+1].lower().split())
                
                if len(sent1_words) == 0 or len(sent2_words) == 0:
                    overlap = 0
                else:
                    intersection = sent1_words.intersection(sent2_words)
                    union = sent1_words.union(sent2_words)
                    overlap = len(intersection) / len(union) if union else 0
                
                sentence_overlaps.append(overlap)
            
            coherence = np.mean(sentence_overlaps) if sentence_overlaps else 0
            coherence_scores.append(coherence)
        
        return np.mean(coherence_scores)
    
    def _engagement_score(self, predictions: List[str], references: List[str]) -> float:
        """参与度评分"""
        
        engagement_keywords = [
            'interesting', 'exciting', 'wonderful', 'amazing', 'great',
            '有趣', '精彩', '好的', '棒', '厉害'
        ]
        
        question_indicators = ['?', '？', 'what', 'how', 'why', '什么', '怎么', '为什么']
        
        engagement_scores = []
        
        for pred in predictions:
            pred_lower = pred.lower()
            
            # 积极词汇得分
            positive_score = sum(1 for keyword in engagement_keywords if keyword in pred_lower)
            
            # 问题引导得分
            question_score = sum(1 for indicator in question_indicators if indicator in pred_lower)
            
            # 长度适中性（不要太短也不要太长）
            length = len(pred.split())
            if 5 <= length <= 50:
                length_score = 1.0
            else:
                length_score = max(0, 1 - abs(length - 25) / 25)
            
            total_engagement = (positive_score * 0.4 + question_score * 0.3 + length_score * 0.3) / 3
            engagement_scores.append(min(total_engagement, 1.0))
        
        return np.mean(engagement_scores)
    
    def _safety_evaluation(self, predictions: List[str], references: List[str]) -> float:
        """安全性评估"""
        
        harmful_patterns = [
            'violence', 'hate', 'discrimination', 'illegal', 'harmful',
            '暴力', '仇恨', '歧视', '违法', '有害'
        ]
        
        safety_scores = []
        
        for pred in predictions:
            pred_lower = pred.lower()
            
            # 检查有害内容
            harmful_count = sum(1 for pattern in harmful_patterns if pattern in pred_lower)
            
            # 安全分数：没有有害内容得1分，否则根据有害内容数量扣分
            safety_score = max(0, 1 - harmful_count * 0.5)
            safety_scores.append(safety_score)
        
        return np.mean(safety_scores)

def comprehensive_task_evaluation():
    """综合任务评估演示"""
    
    print("=== 综合任务评估演示 ===")
    
    # 定义不同任务的测试数据
    task_test_data = {
        'qa': {
            'predictions': [
                "The capital of France is Paris.",
                "Machine learning is a subset of artificial intelligence.",
                "The largest planet in our solar system is Jupiter."
            ],
            'references': [
                "Paris is the capital city of France.",
                "Machine learning is part of AI technology.",
                "Jupiter is the biggest planet in the solar system."
            ],
            'contexts': [
                "France is a country in Europe with Paris as its capital city.",
                "Artificial intelligence includes machine learning as one of its key technologies.",
                "The solar system contains several planets, with Jupiter being the largest."
            ]
        },
        'summarization': {
            'predictions': [
                "The study shows positive results in medical treatment.",
                "Economic growth continues despite global challenges.",
                "New technology improves communication efficiency."
            ],
            'references': [
                "Research demonstrates significant improvement in patient outcomes through novel medical interventions.",
                "Despite worldwide economic uncertainties, the national economy maintains steady growth trajectory.",
                "Advanced communication technologies enhance operational efficiency across multiple sectors."
            ],
            'contexts': [
                "A comprehensive medical study involving 1000 patients demonstrated that the new treatment protocol resulted in 85% success rate, significantly higher than traditional methods which showed only 60% effectiveness. The research was conducted over 18 months.",
                "The national economy recorded 3.2% growth in the last quarter, outperforming expectations despite ongoing global trade tensions and supply chain disruptions affecting multiple industries worldwide.",
                "Implementation of next-generation communication systems has led to 40% improvement in operational efficiency, reducing response times and enhancing collaboration between remote teams across different time zones."
            ]
        },
        'dialogue': {
            'predictions': [
                "That's interesting! Can you tell me more about this topic?",
                "I understand your concern. Let me help you with that.",
                "Thank you for sharing. I appreciate your perspective on this matter."
            ],
            'references': [
                "I find that fascinating! Could you elaborate on this subject?",
                "I see your point. I'd be happy to assist you with this issue.",
                "Thanks for your input. Your viewpoint is valuable and insightful."
            ]
        }
    }
    
    # 对每个任务进行评估
    task_results = {}
    
    for task_type, test_data in task_test_data.items():
        print(f"\\n{'='*20} {task_type.upper()} 任务评估 {'='*20}")
        
        evaluator = TaskSpecificEvaluator(task_type)
        
        if 'contexts' in test_data:
            results = evaluator.evaluate(
                test_data['predictions'],
                test_data['references'],
                test_data['contexts']
            )
        else:
            results = evaluator.evaluate(
                test_data['predictions'],
                test_data['references']
            )
        
        task_results[task_type] = results
    
    # 跨任务比较
    print(f"\\n{'='*20} 跨任务性能比较 {'='*20}")
    
    # 找出共同指标
    common_metrics = set(task_results['qa'].keys())
    for task_results_dict in task_results.values():
        common_metrics = common_metrics.intersection(set(task_results_dict.keys()))
    
    if common_metrics:
        print(f"{'任务':12s}", end="")
        for metric in sorted(common_metrics):
            print(f"{metric:>15s}", end="")
        print()
        print("-" * (12 + 15 * len(common_metrics)))
        
        for task_type, results in task_results.items():
            print(f"{task_type:12s}", end="")
            for metric in sorted(common_metrics):
                score = results.get(metric, 0)
                print(f"{score:15.4f}", end="")
            print()
    
    # 任务难度分析
    print(f"\\n=== 任务难度分析 ===")
    
    for task_type, results in task_results.items():
        avg_score = np.mean(list(results.values()))
        
        if avg_score > 0.8:
            difficulty = "简单"
        elif avg_score > 0.6:
            difficulty = "中等"
        elif avg_score > 0.4:
            difficulty = "困难"
        else:
            difficulty = "极难"
        
        print(f"{task_type:15s}: 平均得分={avg_score:.3f}, 难度={difficulty}")
    
    return task_results
```

## 4.2 人工评估框架设计

### 评估维度的心理学基础

人工评估需要考虑人类认知的多个维度：
1. **信息处理维度**：准确性、完整性、相关性
2. **语言质量维度**：流畅性、连贯性、自然性
3. **交互体验维度**：有用性、满意度、信任度
4. **安全伦理维度**：无害性、公平性、透明性

```python
class HumanEvaluationFramework:
    """人工评估框架"""
    
    def __init__(self, evaluation_dimensions: Dict[str, Dict] = None):
        if evaluation_dimensions is None:
            evaluation_dimensions = self._get_default_dimensions()
        
        self.dimensions = evaluation_dimensions
        self.evaluation_sessions = []
        self.annotator_agreements = {}
        
    def _get_default_dimensions(self) -> Dict[str, Dict]:
        """获取默认评估维度"""
        
        return {
            'accuracy': {
                'description': '信息的准确性和事实正确性',
                'scale': 'likert_5',
                'anchors': {
                    1: '完全错误',
                    2: '大部分错误',
                    3: '部分正确',
                    4: '大部分正确',
                    5: '完全正确'
                },
                'weight': 0.25
            },
            'fluency': {
                'description': '语言的流畅性和自然性',
                'scale': 'likert_5',
                'anchors': {
                    1: '完全不流畅',
                    2: '不太流畅',
                    3: '一般流畅',
                    4: '比较流畅',
                    5: '完全流畅'
                },
                'weight': 0.20
            },
            'relevance': {
                'description': '回答与问题的相关程度',
                'scale': 'likert_5',
                'anchors': {
                    1: '完全不相关',
                    2: '不太相关',
                    3: '部分相关',
                    4: '比较相关',
                    5: '完全相关'
                },
                'weight': 0.25
            },
            'helpfulness': {
                'description': '回答的有用性和实用性',
                'scale': 'likert_5',
                'anchors': {
                    1: '完全无用',
                    2: '不太有用',
                    3: '一般有用',
                    4: '比较有用',
                    5: '非常有用'
                },
                'weight': 0.20
            },
            'safety': {
                'description': '内容的安全性和无害性', 
                'scale': 'binary',
                'anchors': {
                    0: '不安全/有害',
                    1: '安全/无害'
                },
                'weight': 0.10,
                'critical': True  # 关键维度，不安全则整体评分为0
            }
        }
    
    def design_evaluation_interface(self, task_type: str) -> Dict:
        """设计评估界面"""
        
        print(f"=== {task_type.upper()} 任务人工评估界面设计 ===")
        
        interface_config = {
            'task_type': task_type,
            'instructions': self._generate_instructions(task_type),
            'evaluation_form': self._generate_evaluation_form(),
            'quality_controls': self._design_quality_controls(),
            'estimated_time': self._estimate_evaluation_time()
        }
        
        print("界面配置:")
        print(f"  任务类型: {interface_config['task_type']}")
        print(f"  预估时间: {interface_config['estimated_time']} 分钟/样本")
        print(f"  质量控制: {len(interface_config['quality_controls'])} 项措施")
        
        return interface_config
    
    def _generate_instructions(self, task_type: str) -> str:
        """生成评估指导说明"""
        
        base_instructions = """
        请仔细阅读以下评估指导：
        
        1. 评估原则：
           - 保持客观公正，避免个人偏见
           - 基于明确的标准进行评分
           - 如有疑问，选择保守的评分
        
        2. 评分过程：
           - 先整体阅读所有内容
           - 逐个维度进行评分
           - 必要时重新阅读进行确认
        
        3. 注意事项：
           - 关注内容质量，不受长度影响
           - 考虑目标用户的实际需求
           - 标记任何不确定的评分
        """
        
        task_specific = {
            'qa': """
        4. 问答任务特殊要求：
           - 答案必须直接回应问题
           - 检查事实准确性
           - 评估答案的完整性
            """,
            'summarization': """
        4. 摘要任务特殊要求：
           - 检查关键信息是否被涵盖
           - 评估信息的压缩效率
           - 确认没有添加原文没有的信息
            """,
            'dialogue': """
        4. 对话任务特殊要求：
           - 评估回应的自然性
           - 检查上下文的连贯性
           - 考虑对话的参与度
            """
        }
        
        return base_instructions + task_specific.get(task_type, "")
    
    def _generate_evaluation_form(self) -> List[Dict]:
        """生成评估表单"""
        
        form_items = []
        
        for dim_name, dim_config in self.dimensions.items():
            item = {
                'dimension': dim_name,
                'question': f"请评估以下内容的{dim_config['description']}：",
                'scale_type': dim_config['scale'],
                'options': dim_config['anchors'],
                'required': True,
                'weight': dim_config['weight']
            }
            
            if dim_config.get('critical', False):
                item['critical'] = True
                item['validation'] = "如果选择不安全，请说明原因"
            
            form_items.append(item)
        
        # 添加开放性评论
        form_items.append({
            'dimension': 'comments',
            'question': '其他评论或建议（可选）：',
            'scale_type': 'text',
            'options': {},
            'required': False,
            'weight': 0
        })
        
        return form_items
    
    def _design_quality_controls(self) -> List[Dict]:
        """设计质量控制措施"""
        
        return [
            {
                'type': 'attention_check',
                'description': '注意力检查题',
                'implementation': '在评估中插入明确指示的题目',
                'frequency': '每20个样本1次'
            },
            {
                'type': 'consistency_check',
                'description': '一致性检查',
                'implementation': '重复评估部分样本',
                'frequency': '10%样本重复'
            },
            {
                'type': 'inter_annotator_agreement',
                'description': '标注者间一致性',
                'implementation': '多人评估同一批样本',
                'target': 'Kappa > 0.6'
            },
            {
                'type': 'time_validation',
                'description': '时间验证',
                'implementation': '监控评估时间，过快或过慢都需要review',
                'threshold': '30秒-10分钟/样本'
            }
        ]
    
    def _estimate_evaluation_time(self) -> float:
        """估算评估时间"""
        
        # 基础阅读时间
        reading_time = 1.0  # 分钟
        
        # 每个维度的评估时间
        evaluation_time = len(self.dimensions) * 0.5  # 分钟
        
        # 质量控制额外时间
        quality_control_time = 0.5  # 分钟
        
        total_time = reading_time + evaluation_time + quality_control_time
        
        return round(total_time, 1)
    
    def simulate_human_evaluation(self, samples: List[Dict], 
                                num_annotators: int = 3) -> Dict:
        """模拟人工评估过程"""
        
        print(f"=== 模拟人工评评估 (样本数: {len(samples)}, 评估者: {num_annotators}) ===")
        
        # 模拟评估结果
        evaluation_results = []
        annotator_scores = defaultdict(list)
        
        for sample_idx, sample in enumerate(samples):
            sample_results = {
                'sample_id': sample_idx,
                'content': sample,
                'annotations': []
            }
            
            # 每个评估者的评分
            for annotator_id in range(num_annotators):
                annotation = self._simulate_annotator_scores(sample, annotator_id)
                sample_results['annotations'].append(annotation)
                
                # 记录各评估者的分数用于一致性分析
                for dim, score in annotation['scores'].items():
                    annotator_scores[f"{annotator_id}_{dim}"].append(score)
            
            evaluation_results.append(sample_results)
        
        # 计算评估者间一致性
        agreement_analysis = self._compute_inter_annotator_agreement(annotator_scores, num_annotators)
        
        # 计算综合分数
        aggregated_scores = self._aggregate_scores(evaluation_results)
        
        return {
            'detailed_results': evaluation_results,
            'aggregated_scores': aggregated_scores,
            'agreement_analysis': agreement_analysis,
            'quality_metrics': self._compute_evaluation_quality_metrics(evaluation_results)
        }
    
    def _simulate_annotator_scores(self, sample: Dict, annotator_id: int) -> Dict:
        """模拟单个评估者的评分"""
        
        # 模拟评估者的偏好和一致性
        np.random.seed(annotator_id * 100 + hash(str(sample)) % 1000)
        
        # 不同评估者的偏好（严格程度）
        annotator_bias = {
            0: 0.1,   # 较严格
            1: 0.0,   # 中性
            2: -0.1   # 较宽松
        }.get(annotator_id, 0.0)
        
        scores = {}
        
        for dim_name, dim_config in self.dimensions.items():
            if dim_config['scale'] == 'likert_5':
                # 基础分数 + 评估者偏好 + 随机噪声
                base_score = 3.0  # 中性分数
                bias_adjusted = base_score + annotator_bias
                noise = np.random.normal(0, 0.3)  # 添加噪声
                
                raw_score = bias_adjusted + noise
                # 限制在1-5范围内
                final_score = max(1, min(5, round(raw_score)))
                
            elif dim_config['scale'] == 'binary':
                # 二进制评分（安全性）
                prob_safe = 0.9  # 90%概率认为安全
                final_score = 1 if np.random.random() < prob_safe else 0
            
            scores[dim_name] = final_score
        
        return {
            'annotator_id': annotator_id,
            'scores': scores,
            'evaluation_time': np.random.normal(3.0, 1.0),  # 模拟评估时间
            'confidence': np.random.uniform(0.7, 1.0)       # 模拟置信度
        }
    
    def _compute_inter_annotator_agreement(self, annotator_scores: Dict, 
                                         num_annotators: int) -> Dict:
        """计算评估者间一致性"""
        
        agreement_results = {}
        
        # 按维度计算一致性
        for dim_name in self.dimensions.keys():
            dim_scores = []
            
            for annotator_id in range(num_annotators):
                key = f"{annotator_id}_{dim_name}"
                if key in annotator_scores:
                    dim_scores.append(annotator_scores[key])
            
            if len(dim_scores) >= 2:
                # 计算相关系数（简化的一致性指标）
                correlations = []
                for i in range(len(dim_scores)):
                    for j in range(i+1, len(dim_scores)):
                        if len(dim_scores[i]) > 1 and len(dim_scores[j]) > 1:
                            corr = np.corrcoef(dim_scores[i], dim_scores[j])[0, 1]
                            if not np.isnan(corr):
                                correlations.append(corr)
                
                if correlations:
                    avg_correlation = np.mean(correlations)
                    agreement_results[dim_name] = {
                        'correlation': avg_correlation,
                        'agreement_level': self._interpret_agreement(avg_correlation)
                    }
        
        # 计算整体一致性
        if agreement_results:
            overall_correlation = np.mean([result['correlation'] for result in agreement_results.values()])
            agreement_results['overall'] = {
                'correlation': overall_correlation,
                'agreement_level': self._interpret_agreement(overall_correlation)
            }
        
        return agreement_results
    
    def _interpret_agreement(self, correlation: float) -> str:
        """解释一致性水平"""
        
        if correlation >= 0.8:
            return "excellent"
        elif correlation >= 0.6:
            return "good"
        elif correlation >= 0.4:
            return "moderate"
        elif correlation >= 0.2:
            return "fair"
        else:
            return "poor"
    
    def _aggregate_scores(self, evaluation_results: List[Dict]) -> Dict:
        """聚合评分结果"""
        
        aggregated = {}
        
        for dim_name in self.dimensions.keys():
            dim_scores = []
            
            for sample_result in evaluation_results:
                sample_scores = []
                for annotation in sample_result['annotations']:
                    if dim_name in annotation['scores']:
                        sample_scores.append(annotation['scores'][dim_name])
                
                if sample_scores:
                    # 使用平均分作为样本在该维度的得分
                    sample_avg = np.mean(sample_scores)
                    dim_scores.append(sample_avg)
            
            if dim_scores:
                aggregated[dim_name] = {
                    'mean': np.mean(dim_scores),
                    'std': np.std(dim_scores),
                    'median': np.median(dim_scores),
                    'min': np.min(dim_scores),
                    'max': np.max(dim_scores)
                }
        
        # 计算加权综合分数
        if aggregated:
            weighted_score = 0
            total_weight = 0
            
            for dim_name, dim_config in self.dimensions.items():
                if dim_name in aggregated:
                    weight = dim_config['weight']
                    score = aggregated[dim_name]['mean']
                    
                    # 关键维度处理
                    if dim_config.get('critical', False) and score < 0.5:
                        weighted_score = 0
                        break
                    
                    weighted_score += weight * score
                    total_weight += weight
            
            if total_weight > 0:
                weighted_score /= total_weight
            
            aggregated['overall'] = {
                'weighted_score': weighted_score,
                'weights_used': total_weight
            }
        
        return aggregated
    
    def _compute_evaluation_quality_metrics(self, evaluation_results: List[Dict]) -> Dict:
        """计算评估质量指标"""
        
        quality_metrics = {}
        
        # 评估时间分析
        all_times = []
        for sample_result in evaluation_results:
            for annotation in sample_result['annotations']:
                if 'evaluation_time' in annotation:
                    all_times.append(annotation['evaluation_time'])
        
        if all_times:
            quality_metrics['evaluation_time'] = {
                'mean': np.mean(all_times),
                'std': np.std(all_times),
                'outliers': len([t for t in all_times if t < 0.5 or t > 10])  # 异常时间
            }
        
        # 置信度分析
        all_confidences = []
        for sample_result in evaluation_results:
            for annotation in sample_result['annotations']:
                if 'confidence' in annotation:
                    all_confidences.append(annotation['confidence'])
        
        if all_confidences:
            quality_metrics['confidence'] = {
                'mean': np.mean(all_confidences),
                'std': np.std(all_confidences),
                'low_confidence_rate': len([c for c in all_confidences if c < 0.7]) / len(all_confidences)
            }
        
        # 评分分布分析
        score_distributions = {}
        for dim_name in self.dimensions.keys():
            dim_scores = []
            for sample_result in evaluation_results:
                for annotation in sample_result['annotations']:
                    if dim_name in annotation['scores']:
                        dim_scores.append(annotation['scores'][dim_name])
            
            if dim_scores:
                score_distributions[dim_name] = {
                    'distribution': np.bincount(dim_scores) if dim_scores else [],
                    'variance': np.var(dim_scores),
                    'skewness': stats.skew(dim_scores) if len(dim_scores) > 2 else 0
                }
        
        quality_metrics['score_distributions'] = score_distributions
        
        return quality_metrics

def demonstrate_human_evaluation_framework():
    """演示人工评估框架"""
    
    print("=== 人工评估框架演示 ===")
    
    # 创建评估框架
    eval_framework = HumanEvaluationFramework()
    
    # 设计评估界面
    interface_config = eval_framework.design_evaluation_interface('qa')
    
    # 准备测试样本
    test_samples = [
        {
            'question': "什么是机器学习？",
            'response': "机器学习是人工智能的一个分支，它使计算机能够在没有明确编程的情况下学习和改进。",
            'context': "人工智能和机器学习的基本概念"
        },
        {
            'question': "如何优化神经网络？",
            'response': "可以通过调整学习率、使用正则化、增加训练数据等方法来优化神经网络的性能。",
            'context': "神经网络优化技术"
        },
        {
            'question': "深度学习有什么应用？",
            'response': "深度学习广泛应用于图像识别、自然语言处理、语音识别、推荐系统等领域。",
            'context': "深度学习的实际应用"
        }
    ]
    
    # 模拟人工评估
    eval_results = eval_framework.simulate_human_evaluation(test_samples, num_annotators=3)
    
    # 输出结果
    print("\\n=== 评估结果分析 ===")
    
    # 聚合分数
    aggregated = eval_results['aggregated_scores']
    print("各维度平均得分:")
    for dim_name, scores in aggregated.items():
        if dim_name != 'overall' and isinstance(scores, dict):
            mean_score = scores['mean']
            std_score = scores['std']
            print(f"  {dim_name:12s}: {mean_score:.3f} ± {std_score:.3f}")
    
    if 'overall' in aggregated:
        overall_score = aggregated['overall']['weighted_score']
        print(f"  {'overall':12s}: {overall_score:.3f}")
    
    # 一致性分析
    agreement = eval_results['agreement_analysis']
    print("\\n评估者一致性:")
    for dim_name, agreement_info in agreement.items():
        correlation = agreement_info['correlation']
        level = agreement_info['agreement_level']
        print(f"  {dim_name:12s}: r={correlation:.3f} ({level})")
    
    # 质量指标
    quality_metrics = eval_results['quality_metrics']
    if 'evaluation_time' in quality_metrics:
        time_info = quality_metrics['evaluation_time']
        print(f"\\n评估质量:")
        print(f"  平均评估时间: {time_info['mean']:.1f} ± {time_info['std']:.1f} 分钟")
        print(f"  异常时间样本: {time_info['outliers']} 个")
    
    if 'confidence' in quality_metrics:
        conf_info = quality_metrics['confidence']
        print(f"  平均置信度: {conf_info['mean']:.3f}")
        print(f"  低置信度比例: {conf_info['low_confidence_rate']:.1%}")
    
    return eval_results

class EvaluationReportGenerator:
    """评估报告生成器"""
    
    def __init__(self):
        self.report_templates = {
            'executive_summary': self._generate_executive_summary,
            'detailed_analysis': self._generate_detailed_analysis,
            'comparative_analysis': self._generate_comparative_analysis,
            'recommendations': self._generate_recommendations
        }
    
    def generate_comprehensive_report(self, evaluation_data: Dict, 
                                    model_info: Dict = None) -> Dict:
        """生成综合评估报告"""
        
        print("=== 生成综合评估报告 ===")
        
        report = {
            'metadata': {
                'report_generated_at': '2024-01-01 12:00:00',
                'model_info': model_info or {'name': 'MiniGPT', 'version': '1.0'},
                'evaluation_scope': 'Supervised Fine-tuning Performance'
            },
            'sections': {}
        }
        
        # 生成各个部分
        for section_name, generator_func in self.report_templates.items():
            try:
                section_content = generator_func(evaluation_data)
                report['sections'][section_name] = section_content
                print(f"  ✓ {section_name} 部分已生成")
            except Exception as e:
                print(f"  ✗ {section_name} 部分生成失败: {e}")
                report['sections'][section_name] = {'error': str(e)}
        
        return report
    
    def _generate_executive_summary(self, evaluation_data: Dict) -> Dict:
        """生成执行摘要"""
        
        # 提取关键指标
        if 'aggregated_scores' in evaluation_data:
            aggregated = evaluation_data['aggregated_scores']
            
            # 找出最佳和最差维度
            dimension_scores = {}
            for dim_name, scores in aggregated.items():
                if dim_name != 'overall' and isinstance(scores, dict):
                    dimension_scores[dim_name] = scores['mean']
            
            if dimension_scores:
                best_dimension = max(dimension_scores.items(), key=lambda x: x[1])
                worst_dimension = min(dimension_scores.items(), key=lambda x: x[1])
                avg_performance = np.mean(list(dimension_scores.values()))
            else:
                best_dimension = worst_dimension = ("N/A", 0)
                avg_performance = 0
            
            overall_score = aggregated.get('overall', {}).get('weighted_score', 0)
        else:
            best_dimension = worst_dimension = ("N/A", 0)
            avg_performance = overall_score = 0
        
        # 性能等级判定
        if overall_score >= 4.0:
            performance_level = "优秀"
            performance_desc = "模型表现出色，各项指标均达到高标准"
        elif overall_score >= 3.0:
            performance_level = "良好"
            performance_desc = "模型表现良好，大部分指标满足要求"
        elif overall_score >= 2.0:
            performance_level = "一般"
            performance_desc = "模型表现一般，存在改进空间"
        else:
            performance_level = "较差"
            performance_desc = "模型表现不佳，需要显著改进"
        
        summary = {
            'overall_performance': {
                'score': overall_score,
                'level': performance_level,
                'description': performance_desc
            },
            'key_findings': {
                'best_aspect': {
                    'dimension': best_dimension[0],
                    'score': best_dimension[1]
                },
                'worst_aspect': {
                    'dimension': worst_dimension[0],
                    'score': worst_dimension[1]
                },
                'average_performance': avg_performance
            },
            'critical_issues': self._identify_critical_issues(evaluation_data),
            'summary_statistics': self._compute_summary_statistics(evaluation_data)
        }
        
        return summary
    
    def _identify_critical_issues(self, evaluation_data: Dict) -> List[Dict]:
        """识别关键问题"""
        
        issues = []
        
        if 'aggregated_scores' in evaluation_data:
            aggregated = evaluation_data['aggregated_scores']
            
            for dim_name, scores in aggregated.items():
                if dim_name != 'overall' and isinstance(scores, dict):
                    mean_score = scores['mean']
                    std_score = scores['std']
                    
                    # 低分问题
                    if mean_score < 2.0:
                        issues.append({
                            'type': 'low_performance',
                            'dimension': dim_name,
                            'severity': 'high',
                            'description': f"{dim_name}维度得分过低({mean_score:.2f})"
                        })
                    
                    # 高方差问题
                    if std_score > 1.5:
                        issues.append({
                            'type': 'high_variance',
                            'dimension': dim_name,
                            'severity': 'medium',
                            'description': f"{dim_name}维度得分不稳定(标准差{std_score:.2f})"
                        })
        
        # 一致性问题
        if 'agreement_analysis' in evaluation_data:
            agreement = evaluation_data['agreement_analysis']
            
            for dim_name, agreement_info in agreement.items():
                if dim_name != 'overall':
                    correlation = agreement_info['correlation']
                    if correlation < 0.4:
                        issues.append({
                            'type': 'low_agreement',
                            'dimension': dim_name,
                            'severity': 'medium',
                            'description': f"{dim_name}维度评估者一致性低(r={correlation:.3f})"
                        })
        
        return issues
    
    def _compute_summary_statistics(self, evaluation_data: Dict) -> Dict:
        """计算汇总统计"""
        
        stats = {}
        
        if 'detailed_results' in evaluation_data:
            results = evaluation_data['detailed_results']
            
            stats['sample_count'] = len(results)
            
            if results:
                annotator_counts = [len(result['annotations']) for result in results]
                stats['avg_annotators_per_sample'] = np.mean(annotator_counts)
                stats['total_annotations'] = sum(annotator_counts)
        
        if 'quality_metrics' in evaluation_data:
            quality = evaluation_data['quality_metrics']
            
            if 'evaluation_time' in quality:
                time_info = quality['evaluation_time']
                stats['avg_evaluation_time'] = time_info['mean']
                stats['evaluation_efficiency'] = 'high' if time_info['mean'] < 3.0 else 'normal'
            
            if 'confidence' in quality:
                conf_info = quality['confidence']
                stats['avg_confidence'] = conf_info['mean']
                stats['confidence_level'] = 'high' if conf_info['mean'] > 0.8 else 'normal'
        
        return stats
    
    def _generate_detailed_analysis(self, evaluation_data: Dict) -> Dict:
        """生成详细分析"""
        
        analysis = {
            'dimension_analysis': {},
            'distribution_analysis': {},
            'correlation_analysis': {},
            'outlier_analysis': {}
        }
        
        if 'aggregated_scores' in evaluation_data:
            aggregated = evaluation_data['aggregated_scores']
            
            # 维度分析
            for dim_name, scores in aggregated.items():
                if dim_name != 'overall' and isinstance(scores, dict):
                    analysis['dimension_analysis'][dim_name] = {
                        'descriptive_stats': scores,
                        'interpretation': self._interpret_dimension_score(dim_name, scores['mean']),
                        'improvement_potential': self._assess_improvement_potential(scores)
                    }
        
        if 'quality_metrics' in evaluation_data:
            quality = evaluation_data['quality_metrics']
            
            # 分布分析
            if 'score_distributions' in quality:
                for dim_name, dist_info in quality['score_distributions'].items():
                    analysis['distribution_analysis'][dim_name] = {
                        'variance': dist_info['variance'],
                        'skewness': dist_info['skewness'],
                        'distribution_shape': self._describe_distribution(dist_info)
                    }
        
        return analysis
    
    def _interpret_dimension_score(self, dimension: str, score: float) -> str:
        """解释维度得分"""
        
        interpretations = {
            'accuracy': {
                4.0: "信息准确性优秀，事实错误极少",
                3.0: "信息基本准确，偶有小错误",
                2.0: "信息准确性一般，存在明显错误",
                1.0: "信息准确性差，错误较多"
            },
            'fluency': {
                4.0: "语言表达自然流畅，符合语言习惯",
                3.0: "语言表达基本流畅，偶有不自然",
                2.0: "语言表达一般，存在明显问题",
                1.0: "语言表达不流畅，难以理解"
            },
            'relevance': {
                4.0: "内容高度相关，完全切中要点", 
                3.0: "内容基本相关，大部分切题",
                2.0: "内容部分相关，存在偏题",
                1.0: "内容相关性差，明显偏题"
            }
        }
        
        dim_interp = interpretations.get(dimension, {})
        
        # 找到最接近的得分区间
        thresholds = sorted(dim_interp.keys(), reverse=True)
        for threshold in thresholds:
            if score >= threshold:
                return dim_interp[threshold]
        
        return "得分过低，需要重点改进"
    
    def _assess_improvement_potential(self, scores: Dict) -> str:
        """评估改进潜力"""
        
        mean_score = scores['mean']
        std_score = scores['std']
        
        if mean_score >= 4.0:
            return "已达优秀水平，保持即可"
        elif mean_score >= 3.0:
            if std_score < 0.5:
                return "表现稳定，可进行精细化优化"
            else:
                return "平均水平良好，需要提高稳定性"
        else:
            if std_score > 1.0:
                return "表现不稳定，需要系统性改进"
            else:
                return "整体水平偏低，需要重点提升"
    
    def _describe_distribution(self, dist_info: Dict) -> str:
        """描述分布形状"""
        
        variance = dist_info['variance']
        skewness = dist_info['skewness']
        
        # 方差描述
        if variance < 0.5:
            var_desc = "集中"
        elif variance < 1.5:
            var_desc = "适中"
        else:
            var_desc = "分散"
        
        # 偏度描述
        if abs(skewness) < 0.5:
            skew_desc = "对称"
        elif skewness > 0.5:
            skew_desc = "右偏"
        else:
            skew_desc = "左偏"
        
        return f"分布{var_desc}且{skew_desc}"
    
    def _generate_comparative_analysis(self, evaluation_data: Dict) -> Dict:
        """生成比较分析"""
        
        # 这里可以与基线模型或历史版本进行比较
        # 简化实现，主要展示框架结构
        
        comparison = {
            'baseline_comparison': {
                'note': "需要基线数据进行比较",
                'improvement_areas': [],
                'regression_areas': []
            },
            'task_difficulty_analysis': self._analyze_task_difficulty(evaluation_data),
            'performance_consistency': self._analyze_consistency(evaluation_data)
        }
        
        return comparison
    
    def _analyze_task_difficulty(self, evaluation_data: Dict) -> Dict:
        """分析任务难度"""
        
        if 'aggregated_scores' not in evaluation_data:
            return {'note': '缺少评分数据'}
        
        aggregated = evaluation_data['aggregated_scores']
        overall_score = aggregated.get('overall', {}).get('weighted_score', 0)
        
        if overall_score >= 4.0:
            difficulty = "简单"
            explanation = "模型在此任务上表现优秀，任务相对简单"
        elif overall_score >= 3.0:
            difficulty = "中等"
            explanation = "模型表现良好，任务难度适中"
        elif overall_score >= 2.0:
            difficulty = "困难"
            explanation = "模型表现一般，任务具有一定挑战性"
        else:
            difficulty = "极难"
            explanation = "模型表现不佳，任务非常具有挑战性"
        
        return {
            'difficulty_level': difficulty,
            'explanation': explanation,
            'score_based_assessment': overall_score
        }
    
    def _analyze_consistency(self, evaluation_data: Dict) -> Dict:
        """分析性能一致性"""
        
        consistency_analysis = {
            'score_consistency': {},
            'annotator_consistency': {},
            'overall_consistency': 'unknown'
        }
        
        # 分数一致性
        if 'aggregated_scores' in evaluation_data:
            aggregated = evaluation_data['aggregated_scores']
            
            std_scores = []
            for dim_name, scores in aggregated.items():
                if dim_name != 'overall' and isinstance(scores, dict):
                    std_score = scores['std']
                    std_scores.append(std_score)
                    
                    if std_score < 0.5:
                        consistency_level = "高"
                    elif std_score < 1.0:
                        consistency_level = "中"
                    else:
                        consistency_level = "低"
                    
                    consistency_analysis['score_consistency'][dim_name] = {
                        'std': std_score,
                        'level': consistency_level
                    }
            
            if std_scores:
                avg_std = np.mean(std_scores)
                if avg_std < 0.5:
                    overall_consistency = "高"
                elif avg_std < 1.0:
                    overall_consistency = "中"
                else:
                    overall_consistency = "低"
                
                consistency_analysis['overall_consistency'] = overall_consistency
        
        # 评估者一致性
        if 'agreement_analysis' in evaluation_data:
            agreement = evaluation_data['agreement_analysis']
            
            for dim_name, agreement_info in agreement.items():
                consistency_analysis['annotator_consistency'][dim_name] = {
                    'correlation': agreement_info['correlation'],
                    'level': agreement_info['agreement_level']
                }
        
        return consistency_analysis
    
    def _generate_recommendations(self, evaluation_data: Dict) -> Dict:
        """生成改进建议"""
        
        recommendations = {
            'high_priority': [],
            'medium_priority': [],
            'low_priority': [],
            'technical_suggestions': [],
            'evaluation_suggestions': []
        }
        
        # 基于关键问题生成建议
        executive_summary = self._generate_executive_summary(evaluation_data)
        critical_issues = executive_summary.get('critical_issues', [])
        
        for issue in critical_issues:
            if issue['severity'] == 'high':
                priority_list = recommendations['high_priority']
            elif issue['severity'] == 'medium':
                priority_list = recommendations['medium_priority']
            else:
                priority_list = recommendations['low_priority']
            
            suggestion = self._generate_issue_specific_recommendation(issue)
            priority_list.append(suggestion)
        
        # 通用技术建议
        if 'aggregated_scores' in evaluation_data:
            aggregated = evaluation_data['aggregated_scores']
            overall_score = aggregated.get('overall', {}).get('weighted_score', 0)
            
            if overall_score < 3.0:
                recommendations['technical_suggestions'].extend([
                    "考虑增加训练数据量和多样性",
                    "调整模型架构或超参数",
                    "引入更多的正则化技术",
                    "优化损失函数设计"
                ])
        
        # 评估流程建议
        if 'quality_metrics' in evaluation_data:
            quality = evaluation_data['quality_metrics']
            
            if 'confidence' in quality:
                conf_info = quality['confidence']
                if conf_info['low_confidence_rate'] > 0.3:
                    recommendations['evaluation_suggestions'].append(
                        "评估者置信度偏低，建议加强评估指导和培训"
                    )
        
        return recommendations
    
    def _generate_issue_specific_recommendation(self, issue: Dict) -> Dict:
        """生成特定问题的建议"""
        
        issue_type = issue['type']
        dimension = issue['dimension']
        
        recommendation_templates = {
            'low_performance': {
                'accuracy': "加强事实检查和知识更新，引入更多高质量训练数据",
                'fluency': "优化语言模型训练，增加语言流畅性相关的训练目标",
                'relevance': "改进指令理解模块，加强上下文相关性建模",
                'helpfulness': "增加任务特定的微调数据，优化用户意图理解"
            },
            'high_variance': {
                'accuracy': "标准化事实验证流程，减少模型输出的不确定性",
                'fluency': "加强语言一致性训练，减少生成质量波动",
                'relevance': "优化注意力机制，提高响应相关性的稳定性",
                'helpfulness': "建立统一的有用性评估标准，减少评估差异"
            },
            'low_agreement': {
                'default': f"改进{dimension}维度的评估标准，加强评估者培训"
            }
        }
        
        templates = recommendation_templates.get(issue_type, {})
        suggestion_text = templates.get(dimension, templates.get('default', "需要进一步分析具体问题"))
        
        return {
            'issue': issue['description'],
            'suggestion': suggestion_text,
            'expected_impact': self._estimate_impact(issue_type, dimension)
        }
    
    def _estimate_impact(self, issue_type: str, dimension: str) -> str:
        """估算改进影响"""
        
        if issue_type == 'low_performance':
            return "预期显著提升整体性能"
        elif issue_type == 'high_variance':
            return "预期提高性能稳定性"
        elif issue_type == 'low_agreement':
            return "预期提高评估可靠性"
        else:
            return "预期带来正面影响"

def comprehensive_evaluation_demo():
    """综合评估演示"""
    
    print("=== 综合评估系统演示 ===")
    
    # 1. 自动评估
    print("\\n1. 执行自动评估...")
    auto_results = demonstrate_automatic_evaluation()
    
    # 2. 人工评估
    print("\\n2. 执行人工评估...")
    human_results = demonstrate_human_evaluation_framework()
    
    # 3. 生成综合报告
    print("\\n3. 生成评估报告...")
    
    # 合并评估数据
    combined_evaluation_data = {
        'automatic_results': auto_results,
        'human_results': human_results,
        'aggregated_scores': human_results['aggregated_scores'],
        'agreement_analysis': human_results['agreement_analysis'],
        'quality_metrics': human_results['quality_metrics'],
        'detailed_results': human_results['detailed_results']
    }
    
    # 生成报告
    report_generator = EvaluationReportGenerator()
    comprehensive_report = report_generator.generate_comprehensive_report(
        combined_evaluation_data,
        model_info={'name': 'MiniGPT-SFT', 'version': '1.0', 'parameters': '125M'}
    )
    
    # 展示报告摘要
    print("\\n=== 评估报告摘要 ===")
    
    exec_summary = comprehensive_report['sections']['executive_summary']
    
    print("整体性能:")
    overall_perf = exec_summary['overall_performance']
    print(f"  得分: {overall_perf['score']:.3f}")
    print(f"  等级: {overall_perf['level']}")
    print(f"  描述: {overall_perf['description']}")
    
    print("\\n关键发现:")
    key_findings = exec_summary['key_findings']
    print(f"  最佳方面: {key_findings['best_aspect']['dimension']} ({key_findings['best_aspect']['score']:.3f})")
    print(f"  最差方面: {key_findings['worst_aspect']['dimension']} ({key_findings['worst_aspect']['score']:.3f})")
    print(f"  平均性能: {key_findings['average_performance']:.3f}")
    
    print("\\n关键问题:")
    critical_issues = exec_summary['critical_issues']
    for i, issue in enumerate(critical_issues[:3]):  # 显示前3个问题
        print(f"  {i+1}. {issue['description']} (严重程度: {issue['severity']})")
    
    # 显示改进建议
    recommendations = comprehensive_report['sections']['recommendations']
    print("\\n改进建议:")
    
    high_priority = recommendations['high_priority']
    if high_priority:
        print("  高优先级:")
        for rec in high_priority[:2]:  # 显示前2个建议
            print(f"    • {rec['suggestion']}")
    
    medium_priority = recommendations['medium_priority']
    if medium_priority:
        print("  中优先级:")
        for rec in medium_priority[:2]:  # 显示前2个建议
            print(f"    • {rec['suggestion']}")
    
    return comprehensive_report

# 运行综合演示
if __name__ == "__main__":
    comprehensive_evaluation_demo()
```

## 小结与思考

本节全面探讨了监督微调的评估指标与效果分析：

1. **自动评估指标**：从BLEU、ROUGE到BERTScore的数学原理和实现细节
2. **任务特定评估**：针对不同任务设计专门的评估框架和指标
3. **人工评估体系**：基于心理学和认知科学的多维度评估框架
4. **评估报告生成**：将复杂的评估数据转化为可操作的改进建议

**关键洞察**：
- 单一指标无法全面反映模型能力，需要多维度综合评估
- 自动评估与人工评估各有优势，需要有机结合
- 评估的一致性和可靠性是确保结果可信的关键
- 评估应该为模型改进提供明确的方向指导

**思考题**：
1. 如何设计评估指标来衡量模型的创造性和原创性？
2. 在多语言评估中如何确保跨语言的公平性？
3. 如何评估模型在长对话中的表现一致性？
4. 怎样设计评估框架来预测模型在真实应用中的表现？

**第4章总结**：本章完整地介绍了监督微调的理论与实践，从任务适应的数学框架到具体的评估体系，为构建高质量的AI助手提供了全面的技术指导。

---

*评估不仅是对模型能力的测量，更是对AI系统质量的持续监控和改进的科学方法。* 📊