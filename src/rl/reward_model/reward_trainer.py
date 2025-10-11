"""
奖励模型训练器

奖励模型是RLHF的关键组件，用于学习人类偏好并为强化学习提供奖励信号。

核心思想：
1. 基于预训练语言模型添加奖励头部
2. 使用人类偏好数据训练排序损失
3. 学习区分高质量和低质量的回复
4. 为PPO训练提供奖励信号

训练流程：
1. 加载预训练模型和偏好数据
2. 添加奖励预测头部
3. 使用排序损失训练
4. 验证和评估模型性能
"""

import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from .ranking_loss import RewardRegularization, create_preference_loss


class RewardHead(nn.Module):
    """奖励预测头部

    将语言模型的隐藏状态映射为标量奖励
    """

    def __init__(self, d_model: int, dropout: float = 0.1):
        """
        初始化奖励头部

        Args:
            d_model: 隐藏层维度
            dropout: dropout概率
        """
        super().__init__()
        self.d_model = d_model

        # 奖励预测层
        self.reward_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 4, 1)  # 输出标量奖励
        )

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        """初始化权重"""
        for layer in self.reward_head:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, hidden_states: torch.Tensor,
                attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        前向传播

        Args:
            hidden_states: (batch_size, seq_len, d_model) 隐藏状态
            attention_mask: (batch_size, seq_len) 注意力掩码

        Returns:
            reward: (batch_size,) 奖励标量
        """
        batch_size, seq_len, d_model = hidden_states.shape

        # 获取序列的最后一个有效位置
        if attention_mask is not None:
            # 找到每个序列的最后一个有效位置
            sequence_lengths = attention_mask.sum(dim=1) - 1  # (batch_size,)
            batch_indices = torch.arange(batch_size, device=hidden_states.device)
            last_hidden_states = hidden_states[batch_indices, sequence_lengths]
        else:
            # 如果没有掩码，取最后一个位置
            last_hidden_states = hidden_states[:, -1, :]

        # 预测奖励
        reward = self.reward_head(last_hidden_states)  # (batch_size, 1)
        reward = reward.squeeze(-1)  # (batch_size,)

        return reward


class RewardModel(nn.Module):
    """奖励模型

    结合语言模型backbone和奖励头部
    """

    def __init__(self, backbone_model, freeze_backbone: bool = False):
        """
        初始化奖励模型

        Args:
            backbone_model: 预训练的语言模型
            freeze_backbone: 是否冻结backbone参数
        """
        super().__init__()

        # 语言模型backbone
        self.backbone = backbone_model

        # 奖励预测头部
        self.reward_head = RewardHead(backbone_model.d_model)

        # 可选：冻结backbone参数
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(self, input_ids: torch.Tensor,
                attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        前向传播

        Args:
            input_ids: (batch_size, seq_len) token IDs
            attention_mask: (batch_size, seq_len) 注意力掩码

        Returns:
            reward: (batch_size,) 奖励值
        """
        # 获取backbone输出
        hidden_states = self.backbone(input_ids)  # (batch_size, seq_len, d_model)

        # 预测奖励
        reward = self.reward_head(hidden_states, attention_mask)

        return reward


class RewardTrainer:
    """奖励模型训练器"""

    def __init__(self,
                 model: RewardModel,
                 tokenizer,
                 device: str = 'cpu',
                 # 损失函数参数
                 ranking_weight: float = 1.0,
                 contrastive_weight: float = 0.0,
                 margin: float = 0.0,
                 temperature: float = 1.0,
                 reg_coef: float = 0.01,
                 # 训练参数
                 learning_rate: float = 5e-5,
                 weight_decay: float = 0.01,
                 max_grad_norm: float = 1.0,
                 warmup_steps: int = 100):
        """
        初始化奖励训练器

        Args:
            model: 奖励模型
            tokenizer: 分词器
            device: 设备
            ranking_weight: 排序损失权重
            contrastive_weight: 对比损失权重
            margin: 排序损失边际
            temperature: 温度参数
            reg_coef: 正则化系数
            learning_rate: 学习率
            weight_decay: 权重衰减
            max_grad_norm: 梯度裁剪阈值
            warmup_steps: 预热步数
        """
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device

        # 损失函数
        self.preference_loss = create_preference_loss(
            ranking_weight=ranking_weight,
            contrastive_weight=contrastive_weight,
            margin=margin,
            temperature=temperature
        )
        self.regularization = RewardRegularization(reg_coef)

        # 优化器
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        # 学习率调度器
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=1000
        )

        # 训练参数
        self.max_grad_norm = max_grad_norm
        self.warmup_steps = warmup_steps

        # 训练记录
        self.train_stats = defaultdict(list)
        self.val_stats = defaultdict(list)
        self.step = 0

    def train_step(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        """
        单步训练

        Args:
            batch: 训练批次

        Returns:
            损失统计
        """
        self.model.train()

        # 移动数据到设备
        for key in batch:
            if torch.is_tensor(batch[key]):
                batch[key] = batch[key].to(self.device)

        # 前向传播
        chosen_rewards = self.model(
            batch['chosen_input_ids'],
            batch['chosen_attention_mask']
        )
        rejected_rewards = self.model(
            batch['rejected_input_ids'],
            batch['rejected_attention_mask']
        )

        # 计算偏好损失
        loss_dict = self.preference_loss(chosen_rewards, rejected_rewards)

        # 添加正则化
        all_rewards = torch.cat([chosen_rewards, rejected_rewards])
        reg_loss = self.regularization(all_rewards)
        loss_dict['reg_loss'] = reg_loss
        loss_dict['total_loss'] += reg_loss

        # 反向传播
        self.optimizer.zero_grad()
        loss_dict['total_loss'].backward()

        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

        # 更新参数
        self.optimizer.step()

        # 预热调度
        if self.step < self.warmup_steps:
            lr_scale = min(1.0, float(self.step + 1) / self.warmup_steps)
            for pg in self.optimizer.param_groups:
                pg['lr'] = pg['lr'] * lr_scale
        else:
            self.scheduler.step()

        self.step += 1

        # 返回损失统计
        return {key: value.item() if torch.is_tensor(value) else value
                for key, value in loss_dict.items()}

    def validate_step(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        """
        单步验证

        Args:
            batch: 验证批次

        Returns:
            验证统计
        """
        self.model.eval()

        with torch.no_grad():
            # 移动数据到设备
            for key in batch:
                if torch.is_tensor(batch[key]):
                    batch[key] = batch[key].to(self.device)

            # 前向传播
            chosen_rewards = self.model(
                batch['chosen_input_ids'],
                batch['chosen_attention_mask']
            )
            rejected_rewards = self.model(
                batch['rejected_input_ids'],
                batch['rejected_attention_mask']
            )

            # 计算损失
            loss_dict = self.preference_loss(chosen_rewards, rejected_rewards)

            # 添加正则化
            all_rewards = torch.cat([chosen_rewards, rejected_rewards])
            reg_loss = self.regularization(all_rewards)
            loss_dict['reg_loss'] = reg_loss
            loss_dict['total_loss'] += reg_loss

        return {key: value.item() if torch.is_tensor(value) else value
                for key, value in loss_dict.items()}

    def train_epoch(self, train_dataloader: DataLoader) -> dict[str, float]:
        """
        训练一个epoch

        Args:
            train_dataloader: 训练数据加载器

        Returns:
            epoch统计
        """
        epoch_stats = defaultdict(list)

        progress_bar = tqdm(train_dataloader, desc="Training")

        for batch in progress_bar:
            # 训练步骤
            step_stats = self.train_step(batch)

            # 累积统计
            for key, value in step_stats.items():
                epoch_stats[key].append(value)

            # 更新进度条
            progress_bar.set_postfix({
                'loss': f"{step_stats['total_loss']:.4f}",
                'acc': f"{step_stats['accuracy']:.4f}",
                'lr': f"{self.optimizer.param_groups[0]['lr']:.6f}"
            })

        # 计算平均统计
        return {key: np.mean(values) for key, values in epoch_stats.items()}

    def validate_epoch(self, val_dataloader: DataLoader) -> dict[str, float]:
        """
        验证一个epoch

        Args:
            val_dataloader: 验证数据加载器

        Returns:
            验证统计
        """
        epoch_stats = defaultdict(list)

        progress_bar = tqdm(val_dataloader, desc="Validation")

        for batch in progress_bar:
            # 验证步骤
            step_stats = self.validate_step(batch)

            # 累积统计
            for key, value in step_stats.items():
                epoch_stats[key].append(value)

            # 更新进度条
            progress_bar.set_postfix({
                'loss': f"{step_stats['total_loss']:.4f}",
                'acc': f"{step_stats['accuracy']:.4f}"
            })

        # 计算平均统计
        return {key: np.mean(values) for key, values in epoch_stats.items()}

    def train(self,
              train_dataloader: DataLoader,
              val_dataloader: DataLoader | None = None,
              num_epochs: int = 10,
              save_interval: int = 1,
              save_dir: str = "reward_model_checkpoints"):
        """
        完整训练流程

        Args:
            train_dataloader: 训练数据加载器
            val_dataloader: 验证数据加载器
            num_epochs: 训练轮数
            save_interval: 保存间隔
            save_dir: 保存目录
        """
        os.makedirs(save_dir, exist_ok=True)

        best_val_loss = float('inf')

        for epoch in range(num_epochs):
            print(f"\\nEpoch {epoch + 1}/{num_epochs}")

            # 训练
            train_stats = self.train_epoch(train_dataloader)

            # 记录训练统计
            for key, value in train_stats.items():
                self.train_stats[key].append(value)

            print(f"训练损失: {train_stats['total_loss']:.4f}")
            print(f"训练准确率: {train_stats['accuracy']:.4f}")

            # 验证
            if val_dataloader:
                val_stats = self.validate_epoch(val_dataloader)

                # 记录验证统计
                for key, value in val_stats.items():
                    self.val_stats[key].append(value)

                print(f"验证损失: {val_stats['total_loss']:.4f}")
                print(f"验证准确率: {val_stats['accuracy']:.4f}")

                # 保存最佳模型
                if val_stats['total_loss'] < best_val_loss:
                    best_val_loss = val_stats['total_loss']
                    self.save_checkpoint(os.path.join(save_dir, "best_model.pt"))
                    print("保存最佳模型")

            # 定期保存
            if (epoch + 1) % save_interval == 0:
                self.save_checkpoint(os.path.join(save_dir, f"checkpoint_epoch_{epoch + 1}.pt"))

        # 绘制训练曲线
        self.plot_training_curves(save_dir)

    def save_checkpoint(self, path: str):
        """保存模型checkpoint"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_stats': dict(self.train_stats),
            'val_stats': dict(self.val_stats),
            'step': self.step
        }, path)

    def load_checkpoint(self, path: str):
        """加载模型checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.train_stats = defaultdict(list, checkpoint.get('train_stats', {}))
        self.val_stats = defaultdict(list, checkpoint.get('val_stats', {}))
        self.step = checkpoint.get('step', 0)

    def plot_training_curves(self, save_dir: str):
        """绘制训练曲线"""
        if not self.train_stats:
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        epochs = range(1, len(self.train_stats['total_loss']) + 1)

        # 总损失
        axes[0, 0].plot(epochs, self.train_stats['total_loss'], 'b-', label='训练')
        if self.val_stats['total_loss']:
            axes[0, 0].plot(epochs, self.val_stats['total_loss'], 'r-', label='验证')
        axes[0, 0].set_title('Total Loss')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)

        # 准确率
        axes[0, 1].plot(epochs, self.train_stats['accuracy'], 'b-', label='训练')
        if self.val_stats['accuracy']:
            axes[0, 1].plot(epochs, self.val_stats['accuracy'], 'r-', label='验证')
        axes[0, 1].set_title('Accuracy')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True)

        # 奖励差异
        axes[1, 0].plot(epochs, self.train_stats['reward_diff'], 'b-', label='训练')
        if self.val_stats['reward_diff']:
            axes[1, 0].plot(epochs, self.val_stats['reward_diff'], 'r-', label='验证')
        axes[1, 0].set_title('Reward Difference')
        axes[1, 0].set_ylabel('Reward Diff')
        axes[1, 0].legend()
        axes[1, 0].grid(True)

        # 排序损失
        axes[1, 1].plot(epochs, self.train_stats['ranking_loss'], 'b-', label='训练')
        if self.val_stats['ranking_loss']:
            axes[1, 1].plot(epochs, self.val_stats['ranking_loss'], 'r-', label='验证')
        axes[1, 1].set_title('Ranking Loss')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].legend()
        axes[1, 1].grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'reward_training_curves.png'))
        plt.close()


def create_reward_model(backbone_model, freeze_backbone: bool = False) -> RewardModel:
    """
    创建奖励模型

    Args:
        backbone_model: 预训练的语言模型
        freeze_backbone: 是否冻结backbone

    Returns:
        RewardModel: 奖励模型
    """
    return RewardModel(backbone_model, freeze_backbone)


def create_reward_trainer(model: RewardModel,
                         tokenizer,
                         device: str = 'cpu',
                         **kwargs) -> RewardTrainer:
    """
    创建奖励训练器

    Args:
        model: 奖励模型
        tokenizer: 分词器
        device: 设备
        **kwargs: 其他参数

    Returns:
        RewardTrainer: 奖励训练器
    """
    return RewardTrainer(model, tokenizer, device, **kwargs)


if __name__ == "__main__":
    # 简单测试
    print("奖励模型训练器实现完成")
    print("主要组件：")
    print("- RewardHead: 奖励预测头部")
    print("- RewardModel: 奖励模型")
    print("- RewardTrainer: 奖励训练器")
    print("- create_reward_model: 工厂函数")
    print("- create_reward_trainer: 训练器工厂函数")
