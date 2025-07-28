# 03 损失函数设计与优化

> **从标准交叉熵到任务特化：监督微调中的损失函数数学艺术**

## 核心思想

损失函数是模型学习的指南针，它不仅定义了什么是"好"的预测，更塑造了模型的行为模式。在监督微调中，标准的交叉熵损失往往不足以应对复杂的任务需求，我们需要设计更精细、更有针对性的损失函数。

**关键洞察**：
- **任务特化**：不同任务需要不同的损失函数设计
- **长度偏差**：标准损失函数存在对短序列的偏好
- **多目标平衡**：需要在多个优化目标间找到平衡
- **梯度特性**：损失函数的梯度特性直接影响训练动态

从数学角度看，损失函数设计是一个多约束优化问题：
$$\mathcal{L}_{total} = \sum_{i} \lambda_i \mathcal{L}_i + \sum_{j} \mu_j \mathcal{R}_j$$

其中$\mathcal{L}_i$是任务特定损失，$\mathcal{R}_j$是正则化项，$\lambda_i, \mu_j$是权重系数。

## 3.1 标准交叉熵的局限性分析

### 长度偏差的数学建模

**标准交叉熵损失**：
$$\mathcal{L}_{CE} = -\frac{1}{N}\sum_{i=1}^{N} \sum_{t=1}^{T_i} \log P(y_t^{(i)} | x^{(i)}, y_{<t}^{(i)})$$

**长度偏差问题**：
- 短序列：总损失小，梯度信号弱
- 长序列：总损失大，梯度信号强
- 结果：模型倾向于生成短序列

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union
import math
from scipy import stats
from collections import defaultdict

class LossFunctionAnalyzer:
    """损失函数分析器"""
    
    def __init__(self, vocab_size: int = 50000):
        self.vocab_size = vocab_size
        self.loss_history = []
        
    def analyze_length_bias(self, sequences: List[torch.Tensor], 
                           logits: List[torch.Tensor]) -> Dict:
        """分析长度偏差问题"""
        
        print("=== 长度偏差分析 ===")
        
        # 按长度分组分析
        length_groups = defaultdict(list)
        
        for seq, logit in zip(sequences, logits):
            seq_len = len(seq)
            
            # 计算标准交叉熵损失
            targets = seq[1:]  # 移除第一个token
            predictions = logit[:-1]  # 移除最后一个预测
            
            ce_loss = F.cross_entropy(predictions, targets, reduction='none')
            
            # 不同的损失计算方式
            total_loss = ce_loss.sum().item()           # 总损失
            mean_loss = ce_loss.mean().item()           # 平均损失
            normalized_loss = total_loss / seq_len       # 长度标准化损失
            
            length_groups[seq_len].append({
                'total_loss': total_loss,
                'mean_loss': mean_loss,
                'normalized_loss': normalized_loss,
                'sequence': seq,
                'per_token_loss': ce_loss.tolist()
            })
        
        # 统计分析
        length_stats = {}
        
        for length, group_data in length_groups.items():
            if len(group_data) < 2:  # 需要足够样本
                continue
                
            total_losses = [item['total_loss'] for item in group_data]
            mean_losses = [item['mean_loss'] for item in group_data]
            normalized_losses = [item['normalized_loss'] for item in group_data]
            
            length_stats[length] = {
                'count': len(group_data),
                'total_loss': {
                    'mean': np.mean(total_losses),
                    'std': np.std(total_losses)
                },
                'mean_loss': {
                    'mean': np.mean(mean_losses),
                    'std': np.std(mean_losses)
                },
                'normalized_loss': {
                    'mean': np.mean(normalized_losses),
                    'std': np.std(normalized_losses)
                }
            }
        
        # 分析长度与损失的相关性
        lengths = list(length_stats.keys())
        total_loss_means = [length_stats[l]['total_loss']['mean'] for l in lengths]
        mean_loss_means = [length_stats[l]['mean_loss']['mean'] for l in lengths]
        
        # 计算相关系数
        if len(lengths) > 2:
            total_corr, total_p = stats.pearsonr(lengths, total_loss_means)
            mean_corr, mean_p = stats.pearsonr(lengths, mean_loss_means)
        else:
            total_corr = mean_corr = total_p = mean_p = 0
        
        print(f"长度统计:")
        for length in sorted(lengths)[:10]:  # 显示前10个长度
            stats_data = length_stats[length]
            print(f"  长度 {length:3d}: 样本数={stats_data['count']:3d}, " +
                  f"总损失={stats_data['total_loss']['mean']:.3f}±{stats_data['total_loss']['std']:.3f}, " +
                  f"平均损失={stats_data['mean_loss']['mean']:.3f}±{stats_data['mean_loss']['std']:.3f}")
        
        print(f"\\n相关性分析:")
        print(f"  长度-总损失相关性: r={total_corr:.3f} (p={total_p:.3f})")
        print(f"  长度-平均损失相关性: r={mean_corr:.3f} (p={mean_p:.3f})")
        
        # 偏差程度评估
        if total_corr > 0.5:
            bias_level = "严重偏差"
        elif total_corr > 0.3:
            bias_level = "中等偏差"
        elif total_corr > 0.1:
            bias_level = "轻微偏差"
        else:
            bias_level = "无明显偏差"
        
        print(f"  长度偏差程度: {bias_level}")
        
        return {
            'length_stats': length_stats,
            'correlations': {
                'total_loss': {'r': total_corr, 'p': total_p},
                'mean_loss': {'r': mean_corr, 'p': mean_p}
            },
            'bias_assessment': bias_level
        }
    
    def analyze_gradient_magnitude_bias(self, sequences: List[torch.Tensor], 
                                      model: nn.Module) -> Dict:
        """分析梯度幅度偏差"""
        
        print("\\n=== 梯度幅度偏差分析 ===")
        
        model.train()
        
        gradient_stats = defaultdict(list)
        
        for seq in sequences:
            seq_len = len(seq)
            
            # 前向传播
            model.zero_grad()
            outputs = model(seq.unsqueeze(0))  # 添加batch维度
            
            # 计算损失
            targets = seq[1:]
            logits = outputs[0, :-1]  # 移除batch维度和最后一个预测
            
            loss = F.cross_entropy(logits, targets)
            
            # 反向传播
            loss.backward()
            
            # 收集梯度信息
            total_grad_norm = 0
            param_count = 0
            
            for param in model.parameters():
                if param.grad is not None:
                    param_grad_norm = param.grad.norm().item()
                    total_grad_norm += param_grad_norm ** 2
                    param_count += param.numel()
            
            total_grad_norm = math.sqrt(total_grad_norm)
            avg_grad_norm = total_grad_norm / math.sqrt(param_count)
            
            gradient_stats[seq_len].append({
                'total_grad_norm': total_grad_norm,
                'avg_grad_norm': avg_grad_norm,
                'loss': loss.item()
            })
        
        # 统计分析
        print("梯度幅度统计:")
        grad_bias_data = {}
        
        for length, grad_data in gradient_stats.items():
            if len(grad_data) < 2:
                continue
            
            total_grads = [item['total_grad_norm'] for item in grad_data]
            avg_grads = [item['avg_grad_norm'] for item in grad_data]
            losses = [item['loss'] for item in grad_data]
            
            grad_bias_data[length] = {
                'total_grad_norm': np.mean(total_grads),
                'avg_grad_norm': np.mean(avg_grads),
                'avg_loss': np.mean(losses),
                'grad_loss_ratio': np.mean(total_grads) / max(np.mean(losses), 1e-6)
            }
            
            print(f"  长度 {length:3d}: 梯度范数={np.mean(total_grads):.4f}, " +
                  f"平均梯度={np.mean(avg_grads):.6f}, 损失={np.mean(losses):.4f}")
        
        # 梯度偏差分析
        if len(grad_bias_data) > 2:
            lengths = list(grad_bias_data.keys())
            grad_norms = [grad_bias_data[l]['total_grad_norm'] for l in lengths]
            
            grad_corr, grad_p = stats.pearsonr(lengths, grad_norms)
            print(f"\\n长度-梯度相关性: r={grad_corr:.3f} (p={grad_p:.3f})")
            
            if grad_corr > 0.5:
                print("  存在显著的梯度偏差，长序列获得更强的训练信号")
            elif grad_corr < -0.5:
                print("  存在反向梯度偏差，短序列获得更强的训练信号")
            else:
                print("  梯度偏差较小")
        
        return grad_bias_data
    
    def compare_loss_functions(self, sequences: List[torch.Tensor], 
                              logits: List[torch.Tensor]) -> Dict:
        """比较不同损失函数的特性"""
        
        print("\\n=== 损失函数比较 ===")
        
        loss_functions = {
            'standard_ce': self._compute_standard_ce,
            'length_normalized_ce': self._compute_length_normalized_ce,
            'focal_loss': self._compute_focal_loss,
            'label_smoothing_ce': self._compute_label_smoothing_ce,
            'weighted_ce': self._compute_weighted_ce
        }
        
        comparison_results = {}
        
        for loss_name, loss_func in loss_functions.items():
            total_loss = 0
            loss_values = []
            length_bias_scores = []
            
            for seq, logit in zip(sequences, logits):
                seq_len = len(seq)
                targets = seq[1:]
                predictions = logit[:-1]
                
                loss_value = loss_func(predictions, targets, seq_len)
                total_loss += loss_value
                loss_values.append(loss_value)
                
                # 计算长度偏差评分（loss per token）
                length_bias_scores.append(loss_value / seq_len)
            
            # 分析长度偏差
            seq_lengths = [len(seq) for seq in sequences]
            if len(seq_lengths) > 2:
                bias_corr, bias_p = stats.pearsonr(seq_lengths, loss_values)
                normalized_bias_corr, _ = stats.pearsonr(seq_lengths, length_bias_scores)
            else:
                bias_corr = normalized_bias_corr = 0
                bias_p = 1.0
            
            comparison_results[loss_name] = {
                'total_loss': total_loss,
                'mean_loss': np.mean(loss_values),
                'std_loss': np.std(loss_values),
                'length_bias_correlation': bias_corr,
                'normalized_bias_correlation': normalized_bias_corr,
                'bias_p_value': bias_p
            }
            
            print(f"{loss_name:20s}: 平均损失={np.mean(loss_values):.4f}, " +
                  f"长度偏差相关性={bias_corr:.3f}")
        
        return comparison_results
    
    def _compute_standard_ce(self, logits: torch.Tensor, targets: torch.Tensor, 
                           seq_len: int) -> float:
        """计算标准交叉熵损失"""
        return F.cross_entropy(logits, targets, reduction='sum').item()
    
    def _compute_length_normalized_ce(self, logits: torch.Tensor, targets: torch.Tensor, 
                                    seq_len: int) -> float:
        """计算长度标准化交叉熵损失"""
        ce_loss = F.cross_entropy(logits, targets, reduction='sum').item()
        return ce_loss / seq_len
    
    def _compute_focal_loss(self, logits: torch.Tensor, targets: torch.Tensor, 
                          seq_len: int, alpha: float = 1.0, gamma: float = 2.0) -> float:
        """计算Focal Loss"""
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = alpha * (1 - pt) ** gamma * ce_loss
        return focal_loss.sum().item()
    
    def _compute_label_smoothing_ce(self, logits: torch.Tensor, targets: torch.Tensor, 
                                  seq_len: int, smoothing: float = 0.1) -> float:
        """计算标签平滑交叉熵损失"""
        log_probs = F.log_softmax(logits, dim=-1)
        
        # 创建平滑标签
        num_classes = logits.size(-1)
        smooth_targets = torch.zeros_like(log_probs)
        smooth_targets.fill_(smoothing / (num_classes - 1))
        smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - smoothing)
        
        loss = -(smooth_targets * log_probs).sum(dim=-1)
        return loss.sum().item()
    
    def _compute_weighted_ce(self, logits: torch.Tensor, targets: torch.Tensor, 
                           seq_len: int) -> float:
        """计算权重交叉熵损失（基于位置）"""
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        
        # 位置权重：后面的token权重更高
        weights = torch.linspace(0.5, 1.0, len(targets))
        weighted_loss = ce_loss * weights
        
        return weighted_loss.sum().item()

def generate_test_data(vocab_size: int = 1000, num_samples: int = 100) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """生成测试数据"""
    
    sequences = []
    logits = []
    
    # 生成不同长度的序列
    for _ in range(num_samples):
        # 随机序列长度 (10-100)
        seq_len = torch.randint(10, 101, (1,)).item()
        
        # 生成随机序列
        seq = torch.randint(0, vocab_size, (seq_len,))
        sequences.append(seq)
        
        # 生成对应的logits（模拟模型输出）
        logit = torch.randn(seq_len, vocab_size)
        logits.append(logit)
    
    return sequences, logits

def comprehensive_loss_analysis():
    """综合损失函数分析"""
    
    print("=== 综合损失函数分析 ===")
    
    # 生成测试数据
    sequences, logits = generate_test_data()
    
    # 创建分析器
    analyzer = LossFunctionAnalyzer()
    
    # 1. 长度偏差分析
    length_bias_results = analyzer.analyze_length_bias(sequences, logits)
    
    # 2. 损失函数比较
    loss_comparison = analyzer.compare_loss_functions(sequences, logits)
    
    # 3. 推荐最佳损失函数
    print("\\n=== 损失函数推荐 ===")
    
    # 根据长度偏差相关性排序
    sorted_losses = sorted(loss_comparison.items(), 
                          key=lambda x: abs(x[1]['length_bias_correlation']))
    
    print("按长度偏差程度排序（偏差越小越好）:")
    for i, (loss_name, metrics) in enumerate(sorted_losses):
        bias_level = abs(metrics['length_bias_correlation'])
        if bias_level < 0.1:
            recommendation = "推荐"
        elif bias_level < 0.3:
            recommendation = "可选"
        else:
            recommendation = "不推荐"
        
        print(f"{i+1}. {loss_name:20s}: 偏差={bias_level:.3f} ({recommendation})")
    
    return length_bias_results, loss_comparison
```

## 3.2 长度标准化损失函数

### 数学推导与实现

**问题分析**：标准交叉熵损失$\mathcal{L}_{CE} = \sum_{t=1}^T \log P(y_t|x, y_{<t})$与序列长度$T$成正比。

**解决方案**：长度标准化
$$\mathcal{L}_{LN} = \frac{1}{T} \sum_{t=1}^T \log P(y_t|x, y_{<t})$$

**高级标准化策略**：
$$\mathcal{L}_{AN} = \frac{1}{T^\alpha} \sum_{t=1}^T \log P(y_t|x, y_{<t})$$

其中$\alpha \in [0, 1]$控制标准化强度。

```python
class LengthNormalizedLoss(nn.Module):
    """长度标准化损失函数"""
    
    def __init__(self, alpha: float = 1.0, ignore_index: int = -100, 
                 reduction: str = 'mean', label_smoothing: float = 0.0):
        super().__init__()
        self.alpha = alpha
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.label_smoothing = label_smoothing
        
    def forward(self, logits: torch.Tensor, targets: torch.Tensor, 
                lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            logits: (batch_size, seq_len, vocab_size)
            targets: (batch_size, seq_len)
            lengths: (batch_size,) 实际长度，如果为None则使用全长
        """
        
        batch_size, seq_len, vocab_size = logits.shape
        
        # 展平输入
        flat_logits = logits.view(-1, vocab_size)
        flat_targets = targets.view(-1)
        
        # 计算每个token的损失
        if self.label_smoothing > 0:
            token_losses = self._label_smoothing_loss(flat_logits, flat_targets)
        else:
            token_losses = F.cross_entropy(flat_logits, flat_targets, 
                                         ignore_index=self.ignore_index, 
                                         reduction='none')
        
        # 重塑为原始形状
        token_losses = token_losses.view(batch_size, seq_len)
        
        # 创建掩码
        if lengths is not None:
            # 使用提供的长度
            mask = torch.arange(seq_len, device=logits.device).unsqueeze(0) < lengths.unsqueeze(1)
        else:
            # 使用ignore_index创建掩码
            mask = (targets != self.ignore_index)
        
        # 应用掩码
        masked_losses = token_losses * mask.float()
        
        # 计算每个序列的有效长度
        effective_lengths = mask.sum(dim=1).float()
        
        # 长度标准化
        if self.alpha == 0:
            # 不标准化
            sequence_losses = masked_losses.sum(dim=1)
        else:
            # 标准化
            sequence_losses = masked_losses.sum(dim=1) / (effective_lengths ** self.alpha)
        
        # 应用reduction
        if self.reduction == 'mean':
            return sequence_losses.mean()
        elif self.reduction == 'sum':
            return sequence_losses.sum()
        else:  # 'none'
            return sequence_losses
    
    def _label_smoothing_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """标签平滑损失"""
        
        log_probs = F.log_softmax(logits, dim=-1)
        vocab_size = logits.size(-1)
        
        # 创建平滑标签
        true_dist = torch.zeros_like(log_probs)
        true_dist.fill_(self.label_smoothing / (vocab_size - 1))
        
        # 处理ignore_index
        valid_mask = (targets != self.ignore_index)
        
        if valid_mask.any():
            true_dist[valid_mask] = true_dist[valid_mask].scatter_(
                1, targets[valid_mask].unsqueeze(1), 1.0 - self.label_smoothing
            )
        
        # 计算损失
        loss = -(true_dist * log_probs).sum(dim=-1)
        
        # 对ignore_index位置设为0
        loss = loss * valid_mask.float()
        
        return loss

class AdaptiveLengthLoss(nn.Module):
    """自适应长度损失函数"""
    
    def __init__(self, target_length: float = 50.0, length_penalty: float = 0.1,
                 base_loss: str = 'ce', **kwargs):
        super().__init__()
        self.target_length = target_length
        self.length_penalty = length_penalty
        self.base_loss = base_loss
        
        # 基础损失函数
        if base_loss == 'ce':
            self.base_loss_fn = nn.CrossEntropyLoss(reduction='none', **kwargs)
        elif base_loss == 'focal':
            self.base_loss_fn = FocalLoss(reduction='none', **kwargs)
        else:
            raise ValueError(f"Unsupported base loss: {base_loss}")
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor, 
                lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        自适应长度损失：根据序列长度与目标长度的差异调整损失
        """
        
        batch_size, seq_len, vocab_size = logits.shape
        
        # 计算基础损失
        flat_logits = logits.view(-1, vocab_size)
        flat_targets = targets.view(-1)
        
        if isinstance(self.base_loss_fn, nn.CrossEntropyLoss):
            token_losses = F.cross_entropy(flat_logits, flat_targets, 
                                         ignore_index=self.base_loss_fn.ignore_index,
                                         reduction='none')
        else:
            token_losses = self.base_loss_fn(flat_logits, flat_targets)
        
        token_losses = token_losses.view(batch_size, seq_len)
        
        # 计算有效长度
        if lengths is not None:
            effective_lengths = lengths.float()
        else:
            mask = (targets != getattr(self.base_loss_fn, 'ignore_index', -100))
            effective_lengths = mask.sum(dim=1).float()
        
        # 计算长度偏差
        length_deviation = torch.abs(effective_lengths - self.target_length)
        
        # 长度惩罚因子
        length_penalty_factor = 1.0 + self.length_penalty * (length_deviation / self.target_length)
        
        # 应用长度惩罚
        sequence_losses = token_losses.sum(dim=1) * length_penalty_factor
        
        return sequence_losses.mean()

def analyze_length_normalization_effects():
    """分析长度标准化的效果"""
    
    print("=== 长度标准化效果分析 ===")
    
    # 生成不同长度的测试数据
    vocab_size = 1000
    batch_size = 32
    
    # 创建不同长度的序列
    short_seq_len = 20
    medium_seq_len = 50
    long_seq_len = 100
    
    test_data = {
        'short': {
            'logits': torch.randn(batch_size, short_seq_len, vocab_size),
            'targets': torch.randint(0, vocab_size, (batch_size, short_seq_len)),
            'lengths': torch.full((batch_size,), short_seq_len)
        },
        'medium': {
            'logits': torch.randn(batch_size, medium_seq_len, vocab_size),
            'targets': torch.randint(0, vocab_size, (batch_size, medium_seq_len)),
            'lengths': torch.full((batch_size,), medium_seq_len)
        },
        'long': {
            'logits': torch.randn(batch_size, long_seq_len, vocab_size),
            'targets': torch.randint(0, vocab_size, (batch_size, long_seq_len)),
            'lengths': torch.full((batch_size,), long_seq_len)
        }
    }
    
    # 测试不同的alpha值
    alphas = [0.0, 0.5, 1.0]
    loss_functions = {}
    
    for alpha in alphas:
        loss_functions[f'alpha_{alpha}'] = LengthNormalizedLoss(alpha=alpha)
    
    # 标准交叉熵作为基线
    loss_functions['standard_ce'] = nn.CrossEntropyLoss()
    
    print("不同长度序列的损失值:")
    print(f"{'损失函数':15s} {'短序列':>10s} {'中序列':>10s} {'长序列':>10s} {'偏差评分':>10s}")
    print("-" * 65)
    
    results = {}
    
    for loss_name, loss_fn in loss_functions.items():
        losses = {}
        
        for seq_type, data in test_data.items():
            logits = data['logits']
            targets = data['targets']
            lengths = data['lengths']
            
            if loss_name == 'standard_ce':
                # 标准交叉熵
                loss = loss_fn(logits.view(-1, vocab_size), targets.view(-1))
            else:
                # 长度标准化损失
                loss = loss_fn(logits, targets, lengths)
            
            losses[seq_type] = loss.item()
        
        # 计算偏差评分（长序列损失 / 短序列损失）
        bias_score = losses['long'] / losses['short'] if losses['short'] > 0 else float('inf')
        
        results[loss_name] = {
            'losses': losses,
            'bias_score': bias_score
        }
        
        print(f"{loss_name:15s} {losses['short']:10.4f} {losses['medium']:10.4f} " +
              f"{losses['long']:10.4f} {bias_score:10.4f}")
    
    # 分析结果
    print("\\n分析结果:")
    print("1. alpha=0.0 (无标准化): 保持原始的长度偏差")
    print("2. alpha=0.5 (平方根标准化): 部分缓解长度偏差")
    print("3. alpha=1.0 (完全标准化): 最大程度缓解长度偏差")
    print("4. 偏差评分越接近1.0，表示长度偏差越小")
    
    return results

class FocalLoss(nn.Module):
    """Focal Loss实现"""
    
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, 
                 ignore_index: int = -100, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.reduction = reduction
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (N, C) where N is batch size, C is number of classes
            targets: (N,) target class indices
        """
        
        # 计算交叉熵
        ce_loss = F.cross_entropy(logits, targets, ignore_index=self.ignore_index, 
                                reduction='none')
        
        # 计算概率
        pt = torch.exp(-ce_loss)
        
        # 应用focal weight
        focal_weight = self.alpha * (1 - pt) ** self.gamma
        focal_loss = focal_weight * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

def compare_focal_vs_standard_loss():
    """比较Focal Loss与标准交叉熵"""
    
    print("\\n=== Focal Loss vs 标准交叉熵比较 ===")
    
    vocab_size = 1000
    batch_size = 100
    seq_len = 50
    
    # 生成测试数据
    logits = torch.randn(batch_size * seq_len, vocab_size)
    targets = torch.randint(0, vocab_size, (batch_size * seq_len,))
    
    # 创建不同置信度的预测
    # 高置信度预测：正确类别的logit很大
    high_conf_indices = torch.randperm(len(targets))[:len(targets)//3]
    logits[high_conf_indices, targets[high_conf_indices]] += 3.0
    
    # 低置信度预测：所有logit都比较平均
    low_conf_indices = torch.randperm(len(targets))[len(targets)//3:2*len(targets)//3]
    logits[low_conf_indices] = torch.randn_like(logits[low_conf_indices]) * 0.5
    
    # 计算概率和预测正确性
    probs = F.softmax(logits, dim=-1)
    predicted_probs = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
    
    # 按置信度分组
    confidence_groups = {
        'high_conf': (predicted_probs > 0.7),
        'medium_conf': ((predicted_probs > 0.3) & (predicted_probs <= 0.7)),
        'low_conf': (predicted_probs <= 0.3)
    }
    
    # 比较不同损失函数
    standard_ce = nn.CrossEntropyLoss(reduction='none')
    focal_loss_fn = FocalLoss(alpha=1.0, gamma=2.0, reduction='none')
    
    ce_losses = standard_ce(logits, targets)
    focal_losses = focal_loss_fn(logits, targets)
    
    print("不同置信度样本的损失比较:")
    print(f"{'置信度组':12s} {'样本数':>8s} {'平均CE损失':>12s} {'平均Focal损失':>15s} {'Focal/CE比值':>12s}")
    print("-" * 70)
    
    for group_name, mask in confidence_groups.items():
        if mask.sum() == 0:
            continue
        
        group_ce = ce_losses[mask].mean().item()
        group_focal = focal_losses[mask].mean().item()
        ratio = group_focal / group_ce if group_ce > 0 else 0
        
        print(f"{group_name:12s} {mask.sum():8d} {group_ce:12.4f} " +
              f"{group_focal:15.4f} {ratio:12.4f}")
    
    print("\\n观察:")
    print("1. 高置信度样本: Focal Loss < 标准CE (减少简单样本权重)")
    print("2. 低置信度样本: Focal Loss ≈ 标准CE (保持困难样本权重)")
    print("3. Focal Loss自动平衡简单样本和困难样本的贡献")
    
    return {
        'ce_losses': ce_losses,
        'focal_losses': focal_losses,
        'confidence_groups': confidence_groups
    }
```

## 3.3 多任务损失函数设计

### 任务权重的动态调整

**多任务损失**：
$$\mathcal{L}_{multi} = \sum_{i=1}^{K} \lambda_i(t) \mathcal{L}_i$$

其中$\lambda_i(t)$是第$i$个任务在时间$t$的权重。

**动态权重策略**：
1. **不确定性权重**：$\lambda_i = \frac{1}{2\sigma_i^2}$
2. **困难度自适应**：$\lambda_i = \exp(\mathcal{L}_i / \tau)$
3. **梯度平衡**：调整权重使各任务梯度范数相近

```python
class MultiTaskLoss(nn.Module):
    """多任务损失函数"""
    
    def __init__(self, task_names: List[str], weighting_strategy: str = 'uncertainty',
                 temperature: float = 1.0, update_freq: int = 100):
        super().__init__()
        
        self.task_names = task_names
        self.num_tasks = len(task_names)
        self.weighting_strategy = weighting_strategy
        self.temperature = temperature
        self.update_freq = update_freq
        
        # 初始化任务权重
        if weighting_strategy == 'uncertainty':
            # 不确定性权重：学习每个任务的不确定性参数
            self.log_vars = nn.Parameter(torch.zeros(self.num_tasks))
        else:
            # 固定或自适应权重
            self.register_buffer('task_weights', torch.ones(self.num_tasks))
        
        # 历史损失记录
        self.loss_history = defaultdict(list)
        self.gradient_history = defaultdict(list)
        self.update_counter = 0
        
    def forward(self, task_losses: Dict[str, torch.Tensor], 
                model: Optional[nn.Module] = None) -> Tuple[torch.Tensor, Dict]:
        """
        Args:
            task_losses: 字典，包含各任务的损失
            model: 模型，用于梯度分析（可选）
        
        Returns:
            total_loss: 加权总损失
            loss_info: 损失信息字典
        """
        
        # 确保所有任务都有损失值
        losses = []
        for task_name in self.task_names:
            if task_name in task_losses:
                losses.append(task_losses[task_name])
            else:
                losses.append(torch.tensor(0.0, device=next(iter(task_losses.values())).device))
        
        losses = torch.stack(losses)
        
        # 更新权重
        if self.update_counter % self.update_freq == 0:
            self._update_weights(losses, model)
        
        # 计算加权损失
        if self.weighting_strategy == 'uncertainty':
            # 不确定性加权
            precisions = torch.exp(-self.log_vars)
            weighted_losses = precisions * losses + self.log_vars
            total_loss = weighted_losses.sum()
        else:
            # 其他策略
            weighted_losses = self.task_weights * losses
            total_loss = weighted_losses.sum()
        
        # 记录损失历史
        for i, task_name in enumerate(self.task_names):
            self.loss_history[task_name].append(losses[i].item())
        
        self.update_counter += 1
        
        # 准备返回信息
        loss_info = {
            'individual_losses': {name: losses[i].item() for i, name in enumerate(self.task_names)},
            'task_weights': self._get_current_weights().tolist(),
            'weighted_losses': weighted_losses.tolist(),
            'total_loss': total_loss.item()
        }
        
        return total_loss, loss_info
    
    def _update_weights(self, current_losses: torch.Tensor, model: Optional[nn.Module]):
        """更新任务权重"""
        
        if self.weighting_strategy == 'uncertainty':
            # 不确定性权重通过梯度自动更新
            pass
        
        elif self.weighting_strategy == 'difficulty_adaptive':
            # 基于困难度的自适应权重
            difficulties = F.softmax(current_losses / self.temperature, dim=0)
            self.task_weights = difficulties
        
        elif self.weighting_strategy == 'gradient_balance' and model is not None:
            # 梯度平衡权重
            self._update_gradient_balanced_weights(current_losses, model)
        
        elif self.weighting_strategy == 'performance_based':
            # 基于性能的权重调整
            self._update_performance_based_weights()
    
    def _update_gradient_balanced_weights(self, current_losses: torch.Tensor, model: nn.Module):
        """更新梯度平衡权重"""
        
        gradient_norms = []
        
        for i, loss in enumerate(current_losses):
            if loss.item() > 0:
                # 计算单个任务的梯度范数
                model.zero_grad()
                loss.backward(retain_graph=True)
                
                total_norm = 0
                for param in model.parameters():
                    if param.grad is not None:
                        total_norm += param.grad.norm().item() ** 2
                
                gradient_norms.append(math.sqrt(total_norm))
            else:
                gradient_norms.append(0.0)
        
        gradient_norms = torch.tensor(gradient_norms, device=current_losses.device)
        
        # 记录梯度历史
        for i, task_name in enumerate(self.task_names):
            self.gradient_history[task_name].append(gradient_norms[i].item())
        
        # 计算平衡权重（目标：使各任务梯度范数相近）
        if gradient_norms.sum() > 0:
            avg_gradient = gradient_norms.mean()
            # 梯度小的任务给更大权重，梯度大的任务给更小权重
            balanced_weights = avg_gradient / (gradient_norms + 1e-8)
            # 归一化
            self.task_weights = balanced_weights / balanced_weights.sum() * self.num_tasks
    
    def _update_performance_based_weights(self):
        """基于历史性能更新权重"""
        
        if len(self.loss_history[self.task_names[0]]) < 10:
            return
        
        # 计算各任务的学习速度（损失下降速度）
        learning_rates = []
        
        for task_name in self.task_names:
            recent_losses = self.loss_history[task_name][-10:]
            if len(recent_losses) >= 2:
                # 计算损失下降趋势
                x = np.arange(len(recent_losses))
                slope, _, _, _, _ = stats.linregress(x, recent_losses)
                learning_rate = -slope  # 负斜率表示下降，转为正的学习速度
            else:
                learning_rate = 0
            
            learning_rates.append(max(learning_rate, 0))
        
        learning_rates = torch.tensor(learning_rates)
        
        # 学习慢的任务给更大权重
        if learning_rates.sum() > 0:
            # 反向权重：学习速度慢的任务权重更大
            inverse_weights = 1.0 / (learning_rates + 1e-6)
            self.task_weights = inverse_weights / inverse_weights.sum() * self.num_tasks
    
    def _get_current_weights(self) -> torch.Tensor:
        """获取当前任务权重"""
        
        if self.weighting_strategy == 'uncertainty':
            return torch.exp(-self.log_vars)
        else:
            return self.task_weights
    
    def get_task_statistics(self) -> Dict:
        """获取任务统计信息"""
        
        stats = {}
        
        for task_name in self.task_names:
            if task_name in self.loss_history and self.loss_history[task_name]:
                losses = self.loss_history[task_name]
                
                stats[task_name] = {
                    'current_loss': losses[-1],
                    'average_loss': np.mean(losses),
                    'loss_std': np.std(losses),
                    'loss_trend': self._compute_trend(losses),
                    'num_updates': len(losses)
                }
                
                if task_name in self.gradient_history and self.gradient_history[task_name]:
                    gradients = self.gradient_history[task_name]
                    stats[task_name]['average_gradient'] = np.mean(gradients)
                    stats[task_name]['gradient_std'] = np.std(gradients)
        
        return stats
    
    def _compute_trend(self, values: List[float]) -> str:
        """计算趋势"""
        
        if len(values) < 3:
            return 'insufficient_data'
        
        recent = values[-5:]  # 最近5个值
        x = np.arange(len(recent))
        slope, _, _, p_value, _ = stats.linregress(x, recent)
        
        if p_value > 0.05:
            return 'stable'
        elif slope < -0.01:
            return 'decreasing'
        elif slope > 0.01:
            return 'increasing'
        else:
            return 'stable'

def demonstrate_multitask_loss():
    """演示多任务损失函数"""
    
    print("=== 多任务损失函数演示 ===")
    
    # 定义任务
    task_names = ['qa', 'summarization', 'translation', 'classification']
    
    # 测试不同的权重策略
    strategies = ['uncertainty', 'difficulty_adaptive', 'gradient_balance', 'performance_based']
    
    results = {}
    
    for strategy in strategies:
        print(f"\\n--- 测试策略: {strategy} ---")
        
        # 创建多任务损失函数
        multitask_loss = MultiTaskLoss(task_names, weighting_strategy=strategy)
        
        # 模拟训练过程
        training_history = {
            'total_losses': [],
            'task_weights': [],
            'individual_losses': defaultdict(list)
        }
        
        for step in range(50):
            # 模拟不同任务的损失
            # QA任务：快速收敛
            qa_loss = torch.tensor(2.0 * math.exp(-step * 0.1) + 0.5)
            
            # 摘要任务：中等收敛速度
            sum_loss = torch.tensor(3.0 * math.exp(-step * 0.05) + 1.0)
            
            # 翻译任务：慢收敛
            trans_loss = torch.tensor(4.0 * math.exp(-step * 0.02) + 1.5)
            
            # 分类任务：很快收敛但后期波动
            cls_loss = torch.tensor(1.5 * math.exp(-step * 0.15) + 0.3 + 0.1 * math.sin(step * 0.5))
            
            task_losses = {
                'qa': qa_loss,
                'summarization': sum_loss,
                'translation': trans_loss,
                'classification': cls_loss
            }
            
            # 计算多任务损失
            total_loss, loss_info = multitask_loss(task_losses)
            
            # 记录历史
            training_history['total_losses'].append(total_loss.item())
            training_history['task_weights'].append(loss_info['task_weights'])
            
            for task_name, loss_val in loss_info['individual_losses'].items():
                training_history['individual_losses'][task_name].append(loss_val)
            
            # 每10步输出一次
            if step % 10 == 0:
                weights_str = ', '.join([f"{w:.3f}" for w in loss_info['task_weights']])
                print(f"  步骤 {step:2d}: 总损失={total_loss:.3f}, 权重=[{weights_str}]")
        
        # 最终统计
        final_stats = multitask_loss.get_task_statistics()
        print(f"\\n最终统计:")
        for task_name, stats in final_stats.items():
            print(f"  {task_name:15s}: 损失={stats['current_loss']:.3f}, " +
                  f"趋势={stats['loss_trend']}")
        
        results[strategy] = {
            'training_history': training_history,
            'final_stats': final_stats
        }
    
    # 比较不同策略
    print(f"\\n=== 策略比较 ===")
    print(f"{'策略':20s} {'最终总损失':>12s} {'权重方差':>10s} {'收敛稳定性':>12s}")
    print("-" * 60)
    
    for strategy, result in results.items():
        final_loss = result['training_history']['total_losses'][-1]
        
        # 计算权重方差（衡量权重分布的均匀性）
        final_weights = result['training_history']['task_weights'][-1]
        weight_variance = np.var(final_weights)
        
        # 计算收敛稳定性（最后10步的损失标准差）
        recent_losses = result['training_history']['total_losses'][-10:]
        convergence_stability = 1.0 / (np.std(recent_losses) + 1e-6)
        
        print(f"{strategy:20s} {final_loss:12.3f} {weight_variance:10.3f} {convergence_stability:12.1f}")
    
    return results

class InstructionResponseLoss(nn.Module):
    """指令-响应对特化损失函数"""
    
    def __init__(self, instruction_weight: float = 0.1, response_weight: float = 1.0,
                 length_normalize: bool = True, alpha: float = 1.0):
        super().__init__()
        self.instruction_weight = instruction_weight
        self.response_weight = response_weight
        self.length_normalize = length_normalize
        self.alpha = alpha
        
    def forward(self, logits: torch.Tensor, targets: torch.Tensor, 
                instruction_mask: torch.Tensor, response_mask: torch.Tensor) -> torch.Tensor:
        """
        计算指令-响应对的特化损失
        
        Args:
            logits: (batch_size, seq_len, vocab_size)
            targets: (batch_size, seq_len) 
            instruction_mask: (batch_size, seq_len) 指令部分掩码
            response_mask: (batch_size, seq_len) 响应部分掩码
        """
        
        batch_size, seq_len, vocab_size = logits.shape
        
        # 计算token级别的损失
        flat_logits = logits.view(-1, vocab_size)
        flat_targets = targets.view(-1)
        
        token_losses = F.cross_entropy(flat_logits, flat_targets, reduction='none')
        token_losses = token_losses.view(batch_size, seq_len)
        
        # 分别计算指令和响应的损失
        instruction_losses = token_losses * instruction_mask.float()
        response_losses = token_losses * response_mask.float()
        
        # 长度标准化
        if self.length_normalize:
            instruction_lengths = instruction_mask.sum(dim=1).float()
            response_lengths = response_mask.sum(dim=1).float()
            
            # 避免除零
            instruction_lengths = torch.clamp(instruction_lengths, min=1.0)
            response_lengths = torch.clamp(response_lengths, min=1.0)
            
            normalized_instruction_loss = instruction_losses.sum(dim=1) / (instruction_lengths ** self.alpha)
            normalized_response_loss = response_losses.sum(dim=1) / (response_lengths ** self.alpha)
        else:
            normalized_instruction_loss = instruction_losses.sum(dim=1)
            normalized_response_loss = response_losses.sum(dim=1)
        
        # 加权组合
        total_loss = (self.instruction_weight * normalized_instruction_loss + 
                     self.response_weight * normalized_response_loss)
        
        return total_loss.mean()

def test_instruction_response_loss():
    """测试指令-响应损失函数"""
    
    print("\\n=== 指令-响应损失函数测试 ===")
    
    # 创建测试数据
    batch_size = 8
    vocab_size = 1000
    
    # 不同长度的指令和响应
    test_cases = [
        {'inst_len': 10, 'resp_len': 20, 'desc': '短指令-中响应'},
        {'inst_len': 30, 'resp_len': 10, 'desc': '长指令-短响应'},
        {'inst_len': 20, 'resp_len': 40, 'desc': '中指令-长响应'},
    ]
    
    # 不同的权重配置
    weight_configs = [
        {'inst_w': 0.1, 'resp_w': 1.0, 'desc': '重响应'},
        {'inst_w': 0.5, 'resp_w': 1.0, 'desc': '平衡'},
        {'inst_w': 1.0, 'resp_w': 0.5, 'desc': '重指令'},
    ]
    
    print("测试不同配置的损失值:")
    print(f"{'测试用例':15s} {'权重配置':10s} {'指令损失':>10s} {'响应损失':>10s} {'总损失':>10s}")
    print("-" * 70)
    
    for case in test_cases:
        inst_len = case['inst_len']
        resp_len = case['resp_len']
        seq_len = inst_len + resp_len
        
        # 生成测试数据
        logits = torch.randn(batch_size, seq_len, vocab_size)
        targets = torch.randint(0, vocab_size, (batch_size, seq_len))
        
        # 创建掩码
        instruction_mask = torch.zeros(batch_size, seq_len)
        response_mask = torch.zeros(batch_size, seq_len)
        
        instruction_mask[:, :inst_len] = 1
        response_mask[:, inst_len:] = 1
        
        for weight_config in weight_configs:
            # 创建损失函数
            loss_fn = InstructionResponseLoss(
                instruction_weight=weight_config['inst_w'],
                response_weight=weight_config['resp_w'],
                length_normalize=True
            )
            
            # 计算损失
            total_loss = loss_fn(logits, targets, instruction_mask, response_mask)
            
            # 分别计算指令和响应损失用于显示
            token_losses = F.cross_entropy(
                logits.view(-1, vocab_size), targets.view(-1), reduction='none'
            ).view(batch_size, seq_len)
            
            inst_loss = (token_losses * instruction_mask.float()).sum() / instruction_mask.sum()
            resp_loss = (token_losses * response_mask.float()).sum() / response_mask.sum()
            
            print(f"{case['desc']:15s} {weight_config['desc']:10s} " +
                  f"{inst_loss:10.4f} {resp_loss:10.4f} {total_loss:10.4f}")
    
    print("\\n观察:")
    print("1. 权重配置直接影响指令vs响应的学习重点")
    print("2. 长度标准化避免了长序列的过度惩罚")
    print("3. 可以根据任务特点调整指令和响应的重要性")
```

## 小结与思考

本节深入探讨了监督微调中的损失函数设计与优化：

1. **标准交叉熵的局限性**：长度偏差问题的数学分析和量化方法
2. **长度标准化技术**：多种标准化策略的数学推导和实验对比
3. **多任务损失设计**：动态权重调整的理论框架和实现方法
4. **任务特化损失**：针对指令跟随任务的专门损失函数设计

**关键洞察**：
- 损失函数直接决定了模型的学习目标和行为偏好
- 长度偏差是标准交叉熵的固有问题，需要专门的技术解决
- 多任务学习需要智能的权重平衡策略
- 不同任务特性需要定制化的损失函数设计

**思考题**：
1. 如何设计损失函数来平衡生成质量与多样性？
2. 在多语言微调中如何处理不同语言的损失权重？
3. 强化学习信号如何与监督学习损失结合？
4. 如何设计损失函数来缓解模型的幻觉问题？

<function_calls>
<invoke name="TodoWrite">
<parameter name="todos">[{"content": "Research best practices for technical tutorial writing", "status": "completed", "priority": "high", "id": "1"}, {"content": "Design comprehensive folder structure for MiniGPT handbook", "status": "completed", "priority": "high", "id": "2"}, {"content": "Create detailed outline with ultra-deep analysis approach", "status": "completed", "priority": "high", "id": "3"}, {"content": "Write handbook introduction and navigation guide", "status": "completed", "priority": "high", "id": "4"}, {"content": "Implement chapter-by-chapter content with mathematical rigor", "status": "in_progress", "priority": "high", "id": "5"}, {"content": "Write Chapter 1: Mathematical Foundations", "status": "completed", "priority": "high", "id": "6"}, {"content": "Write Chapter 2: Transformer Architecture", "status": "completed", "priority": "high", "id": "7"}, {"content": "Write Chapter 3: Pre-training Theory and Implementation", "status": "completed", "priority": "high", "id": "8"}, {"content": "Write Chapter 4: Supervised Fine-tuning Deep Dive", "status": "in_progress", "priority": "high", "id": "14"}, {"content": "Create Chapter 4 folder structure and README", "status": "completed", "priority": "medium", "id": "15"}, {"content": "Write Chapter 4 Section 1: Task Adaptation Theory Framework", "status": "completed", "priority": "high", "id": "16"}, {"content": "Write Chapter 4 Section 2: Instruction Following and Dialogue Modeling", "status": "completed", "priority": "high", "id": "17"}, {"content": "Write Chapter 4 Section 3: Loss Function Design and Optimization", "status": "completed", "priority": "high", "id": "18"}, {"content": "Write Chapter 4 Section 4: Evaluation Metrics and Effect Analysis", "status": "in_progress", "priority": "high", "id": "19"}]