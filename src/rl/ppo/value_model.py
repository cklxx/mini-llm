"""
PPO价值函数模型实现

价值函数V(s)用于估计在状态s下的期望累积奖励。
在语言模型RLHF中，状态通常是token序列的前缀。

核心概念：
- 价值函数评估当前状态的"好坏程度"
- 用于计算优势函数 A(s,a) = Q(s,a) - V(s)
- 减少策略梯度的方差，提高训练稳定性
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class ValueHead(nn.Module):
    """价值函数头部
    
    将transformer的隐藏状态映射为标量价值
    """
    
    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        
        # 价值预测层：hidden_states -> scalar value
        self.value_projection = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 4, 1)  # 输出标量价值
        )
        
        # 初始化参数
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        for layer in self.value_projection:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
    
    def forward(self, hidden_states: torch.Tensor, 
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播
        
        Args:
            hidden_states: (batch_size, seq_len, d_model) transformer隐藏状态
            attention_mask: (batch_size, seq_len) 注意力掩码
            
        Returns:
            values: (batch_size, seq_len) 每个位置的价值估计
        """
        batch_size, seq_len, d_model = hidden_states.shape
        
        # 投影到价值空间
        values = self.value_projection(hidden_states)  # (batch_size, seq_len, 1)
        values = values.squeeze(-1)  # (batch_size, seq_len)
        
        # 应用注意力掩码（将padding位置的价值置为0）
        if attention_mask is not None:
            values = values * attention_mask.float()
        
        return values


class ValueModel(nn.Module):
    """完整的价值模型
    
    结合transformer backbone和价值头部
    可以独立训练，也可以与策略模型共享backbone
    """
    
    def __init__(self, transformer_model, freeze_backbone: bool = False):
        """
        初始化价值模型
        
        Args:
            transformer_model: 预训练的transformer模型
            freeze_backbone: 是否冻结backbone参数
        """
        super().__init__()
        
        # Transformer backbone（共享或独立）
        self.transformer = transformer_model
        
        # 价值预测头部
        self.value_head = ValueHead(transformer_model.d_model)
        
        # 可选：冻结backbone参数
        if freeze_backbone:
            for param in self.transformer.parameters():
                param.requires_grad = False
    
    def forward(self, input_ids: torch.Tensor, 
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播
        
        Args:
            input_ids: (batch_size, seq_len) token IDs
            attention_mask: (batch_size, seq_len) 注意力掩码
            
        Returns:
            values: (batch_size, seq_len) 每个位置的价值估计
        """
        # 获取transformer隐藏状态
        hidden_states = self.transformer(input_ids)  # (batch_size, seq_len, d_model)
        
        # 计算价值
        values = self.value_head(hidden_states, attention_mask)
        
        return values
    
    def get_values_for_sequences(self, input_ids: torch.Tensor, 
                                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        获取序列级别的价值（取最后一个非padding位置的价值）
        
        在RLHF中，通常关心完整序列的价值，而非每个token的价值
        
        Args:
            input_ids: (batch_size, seq_len)
            attention_mask: (batch_size, seq_len)
            
        Returns:
            sequence_values: (batch_size,) 每个序列的价值
        """
        values = self.forward(input_ids, attention_mask)  # (batch_size, seq_len)
        
        if attention_mask is not None:
            # 找到每个序列的最后一个有效位置
            sequence_lengths = attention_mask.sum(dim=1) - 1  # (batch_size,)
            batch_indices = torch.arange(values.size(0), device=values.device)
            sequence_values = values[batch_indices, sequence_lengths]
        else:
            # 如果没有掩码，取最后一个位置
            sequence_values = values[:, -1]
        
        return sequence_values


class ValueLoss(nn.Module):
    """价值函数损失
    
    使用均方误差损失训练价值函数
    """
    
    def __init__(self, clip_value: float = 0.2):
        """
        Args:
            clip_value: 价值函数的裁剪阈值，防止更新过大
        """
        super().__init__()
        self.clip_value = clip_value
        self.mse_loss = nn.MSELoss(reduction='none')
    
    def forward(self, predicted_values: torch.Tensor, 
                target_values: torch.Tensor,
                old_values: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        计算价值函数损失
        
        Args:
            predicted_values: (batch_size, seq_len) 模型预测的价值
            target_values: (batch_size, seq_len) 目标价值（通常来自奖励累积）
            old_values: (batch_size, seq_len) 旧模型的价值（用于裁剪）
            attention_mask: (batch_size, seq_len) 注意力掩码
            
        Returns:
            loss: 标量损失值
        """
        # 基础MSE损失
        value_loss = self.mse_loss(predicted_values, target_values)
        
        # 可选：应用价值裁剪（类似PPO的策略裁剪）
        if old_values is not None:
            clipped_values = old_values + torch.clamp(
                predicted_values - old_values, 
                -self.clip_value, 
                self.clip_value
            )
            clipped_loss = self.mse_loss(clipped_values, target_values)
            value_loss = torch.max(value_loss, clipped_loss)
        
        # 应用注意力掩码
        if attention_mask is not None:
            value_loss = value_loss * attention_mask.float()
            # 计算平均损失（只考虑有效位置）
            loss = value_loss.sum() / attention_mask.sum()
        else:
            loss = value_loss.mean()
        
        return loss


def create_value_model(transformer_model, freeze_backbone: bool = False) -> ValueModel:
    """
    创建价值模型的工厂函数
    
    Args:
        transformer_model: 预训练的transformer模型
        freeze_backbone: 是否冻结backbone
        
    Returns:
        ValueModel: 价值模型实例
    """
    return ValueModel(transformer_model, freeze_backbone)


if __name__ == "__main__":
    # 简单测试
    print("价值模型模块实现完成")
    print("主要组件：")
    print("- ValueHead: 价值函数头部")
    print("- ValueModel: 完整价值模型") 
    print("- ValueLoss: 价值函数损失")
    print("- create_value_model: 工厂函数")