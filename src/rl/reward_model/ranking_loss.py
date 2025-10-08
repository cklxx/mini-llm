"""
排序损失函数实现

在RLHF中，奖励模型通过学习人类偏好来评估文本质量。
主要使用排序损失（Ranking Loss）来训练奖励模型。

核心概念：
- 偏好数据：(prompt, chosen_response, rejected_response)
- 排序损失：确保chosen_response的奖励高于rejected_response
- Bradley-Terry模型：P(y1 > y2) = σ(r(y1) - r(y2))

损失函数：
L = -log(σ(r(chosen) - r(rejected)))
其中σ是sigmoid函数，r是奖励模型
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


class RankingLoss(nn.Module):
    """排序损失函数
    
    基于Bradley-Terry模型的排序损失
    """
    
    def __init__(self, margin: float = 0.0, reduction: str = 'mean'):
        """
        初始化排序损失
        
        Args:
            margin: 边际值，增加chosen和rejected之间的最小差距
            reduction: 损失规约方式 ('mean', 'sum', 'none')
        """
        super().__init__()
        self.margin = margin
        self.reduction = reduction
    
    def forward(self, chosen_rewards: torch.Tensor, 
                rejected_rewards: torch.Tensor) -> torch.Tensor:
        """
        计算排序损失
        
        Args:
            chosen_rewards: (batch_size,) 选中回复的奖励
            rejected_rewards: (batch_size,) 拒绝回复的奖励
            
        Returns:
            loss: 排序损失
        """
        # 计算奖励差异
        reward_diff = chosen_rewards - rejected_rewards - self.margin
        
        # Bradley-Terry损失：-log(sigmoid(r_chosen - r_rejected))
        loss = -F.logsigmoid(reward_diff)
        
        # 应用规约
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class ContrastiveLoss(nn.Module):
    """对比损失函数
    
    另一种训练奖励模型的损失函数
    """
    
    def __init__(self, temperature: float = 1.0, reduction: str = 'mean'):
        """
        初始化对比损失
        
        Args:
            temperature: 温度参数，控制分布的尖锐程度
            reduction: 损失规约方式
        """
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction
    
    def forward(self, chosen_rewards: torch.Tensor,
                rejected_rewards: torch.Tensor) -> torch.Tensor:
        """
        计算对比损失
        
        Args:
            chosen_rewards: (batch_size,) 选中回复的奖励
            rejected_rewards: (batch_size,) 拒绝回复的奖励
            
        Returns:
            loss: 对比损失
        """
        # 将奖励堆叠
        rewards = torch.stack([chosen_rewards, rejected_rewards], dim=1)  # (batch_size, 2)
        
        # 应用温度
        scaled_rewards = rewards / self.temperature
        
        # 计算softmax概率
        probs = F.softmax(scaled_rewards, dim=1)
        
        # 目标是chosen回复的概率为1
        targets = torch.zeros(chosen_rewards.size(0), dtype=torch.long, device=chosen_rewards.device)
        
        # 交叉熵损失
        loss = F.cross_entropy(scaled_rewards, targets, reduction='none')
        
        # 应用规约
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class MultiPairRankingLoss(nn.Module):
    """多对排序损失
    
    处理多个候选回复的排序损失
    """
    
    def __init__(self, temperature: float = 1.0, reduction: str = 'mean'):
        """
        初始化多对排序损失
        
        Args:
            temperature: 温度参数
            reduction: 损失规约方式
        """
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction
    
    def forward(self, rewards: torch.Tensor, 
                rankings: torch.Tensor) -> torch.Tensor:
        """
        计算多对排序损失
        
        Args:
            rewards: (batch_size, num_candidates) 候选回复的奖励
            rankings: (batch_size, num_candidates) 排序标签（越小越好）
            
        Returns:
            loss: 多对排序损失
        """
        batch_size, num_candidates = rewards.shape
        
        # 应用温度
        scaled_rewards = rewards / self.temperature
        
        # 计算所有配对的损失
        total_loss = 0
        num_pairs = 0
        
        for i in range(num_candidates):
            for j in range(i + 1, num_candidates):
                # 确定哪个应该更好
                if rankings[:, i] < rankings[:, j]:
                    # i应该比j好
                    better_rewards = scaled_rewards[:, i]
                    worse_rewards = scaled_rewards[:, j]
                elif rankings[:, i] > rankings[:, j]:
                    # j应该比i好
                    better_rewards = scaled_rewards[:, j]
                    worse_rewards = scaled_rewards[:, i]
                else:
                    # 相等，跳过
                    continue
                
                # 计算排序损失
                pair_loss = -F.logsigmoid(better_rewards - worse_rewards)
                total_loss += pair_loss
                num_pairs += 1
        
        if num_pairs == 0:
            return torch.tensor(0.0, device=rewards.device, requires_grad=True)
        
        # 平均损失
        loss = total_loss / num_pairs
        
        # 应用规约
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class PreferenceLoss(nn.Module):
    """偏好损失函数
    
    结合多种损失函数的复合损失
    """
    
    def __init__(self, 
                 ranking_weight: float = 1.0,
                 contrastive_weight: float = 0.0,
                 margin: float = 0.0,
                 temperature: float = 1.0,
                 reduction: str = 'mean'):
        """
        初始化偏好损失
        
        Args:
            ranking_weight: 排序损失权重
            contrastive_weight: 对比损失权重
            margin: 排序损失边际
            temperature: 温度参数
            reduction: 损失规约方式
        """
        super().__init__()
        self.ranking_weight = ranking_weight
        self.contrastive_weight = contrastive_weight
        
        # 子损失函数
        self.ranking_loss = RankingLoss(margin=margin, reduction=reduction)
        self.contrastive_loss = ContrastiveLoss(temperature=temperature, reduction=reduction)
    
    def forward(self, chosen_rewards: torch.Tensor,
                rejected_rewards: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        计算复合偏好损失
        
        Args:
            chosen_rewards: (batch_size,) 选中回复的奖励
            rejected_rewards: (batch_size,) 拒绝回复的奖励
            
        Returns:
            loss_dict: 包含各种损失的字典
        """
        loss_dict = {}
        total_loss = 0
        
        # 排序损失
        if self.ranking_weight > 0:
            ranking_loss = self.ranking_loss(chosen_rewards, rejected_rewards)
            loss_dict['ranking_loss'] = ranking_loss
            total_loss += self.ranking_weight * ranking_loss
        
        # 对比损失
        if self.contrastive_weight > 0:
            contrastive_loss = self.contrastive_loss(chosen_rewards, rejected_rewards)
            loss_dict['contrastive_loss'] = contrastive_loss
            total_loss += self.contrastive_weight * contrastive_loss
        
        loss_dict['total_loss'] = total_loss
        
        # 添加一些有用的统计信息
        loss_dict['reward_diff'] = (chosen_rewards - rejected_rewards).mean()
        loss_dict['chosen_reward_mean'] = chosen_rewards.mean()
        loss_dict['rejected_reward_mean'] = rejected_rewards.mean()
        loss_dict['accuracy'] = (chosen_rewards > rejected_rewards).float().mean()
        
        return loss_dict


class RewardRegularization(nn.Module):
    """奖励正则化
    
    防止奖励模型过拟合和奖励坍缩
    """
    
    def __init__(self, reg_coef: float = 0.01):
        """
        初始化奖励正则化
        
        Args:
            reg_coef: 正则化系数
        """
        super().__init__()
        self.reg_coef = reg_coef
    
    def forward(self, rewards: torch.Tensor) -> torch.Tensor:
        """
        计算奖励正则化损失
        
        Args:
            rewards: (batch_size, ...) 奖励值
            
        Returns:
            reg_loss: 正则化损失
        """
        # L2正则化：防止奖励值过大
        l2_reg = torch.mean(rewards ** 2)
        
        # 方差正则化：鼓励奖励有一定分布
        variance_reg = -torch.var(rewards)
        
        return self.reg_coef * (l2_reg + variance_reg)


def create_preference_loss(ranking_weight: float = 1.0,
                          contrastive_weight: float = 0.0,
                          margin: float = 0.0,
                          temperature: float = 1.0,
                          reduction: str = 'mean') -> PreferenceLoss:
    """
    创建偏好损失函数的工厂函数
    
    Args:
        ranking_weight: 排序损失权重
        contrastive_weight: 对比损失权重
        margin: 排序损失边际
        temperature: 温度参数
        reduction: 损失规约方式
        
    Returns:
        PreferenceLoss: 偏好损失函数实例
    """
    return PreferenceLoss(
        ranking_weight=ranking_weight,
        contrastive_weight=contrastive_weight,
        margin=margin,
        temperature=temperature,
        reduction=reduction
    )


if __name__ == "__main__":
    # 简单测试
    print("排序损失模块实现完成")
    print("主要组件：")
    print("- RankingLoss: 基础排序损失")
    print("- ContrastiveLoss: 对比损失")
    print("- MultiPairRankingLoss: 多对排序损失")
    print("- PreferenceLoss: 复合偏好损失")
    print("- RewardRegularization: 奖励正则化")
    print("- create_preference_loss: 工厂函数")
    
    # 测试示例
    batch_size = 4
    chosen_rewards = torch.randn(batch_size)
    rejected_rewards = torch.randn(batch_size) - 1.0  # 让chosen普遍更好
    
    # 测试排序损失
    ranking_loss = RankingLoss()
    loss = ranking_loss(chosen_rewards, rejected_rewards)
    print(f"\\n排序损失示例: {loss.item():.4f}")
    
    # 测试偏好损失
    preference_loss = create_preference_loss()
    loss_dict = preference_loss(chosen_rewards, rejected_rewards)
    print(f"偏好损失示例: {loss_dict['total_loss'].item():.4f}")
    print(f"准确率: {loss_dict['accuracy'].item():.4f}")