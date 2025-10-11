"""
PPO策略梯度实现

策略梯度是强化学习的核心，通过优化策略参数来最大化期望奖励。
PPO (Proximal Policy Optimization) 通过限制策略更新幅度来提高训练稳定性。

核心概念：
- 策略梯度：∇θ J(θ) = E[∇θ log π(a|s) * A(s,a)]
- 优势函数：A(s,a) = Q(s,a) - V(s) = r + γV(s') - V(s)
- PPO目标：L_CLIP = E[min(r_t(θ)A_t, clip(r_t(θ), 1-ε, 1+ε)A_t)]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AdvantageCalculator:
    """优势函数计算器

    优势函数衡量某个动作相比平均水平的好坏程度
    使用GAE (Generalized Advantage Estimation) 方法
    """

    def __init__(self, gamma: float = 0.99, lambda_gae: float = 0.95):
        """
        Args:
            gamma: 折扣因子，控制未来奖励的重要性
            lambda_gae: GAE参数，控制偏差-方差权衡
        """
        self.gamma = gamma
        self.lambda_gae = lambda_gae

    def compute_advantages(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor,
        next_values: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        计算优势函数和目标价值

        GAE公式：
        A_t = δ_t + (γλ)δ_{t+1} + (γλ)²δ_{t+2} + ...
        其中 δ_t = r_t + γV(s_{t+1}) - V(s_t)

        Args:
            rewards: (batch_size, seq_len) 奖励
            values: (batch_size, seq_len) 价值函数预测
            dones: (batch_size, seq_len) 是否终止
            next_values: (batch_size, seq_len) 下一状态价值（可选）

        Returns:
            advantages: (batch_size, seq_len) 优势函数
            returns: (batch_size, seq_len) 目标价值（用于训练价值函数）
        """
        batch_size, seq_len = rewards.shape
        device = rewards.device

        # 如果没有提供next_values，使用values的右移版本
        if next_values is None:
            next_values = torch.cat(
                [values[:, 1:], torch.zeros(batch_size, 1, device=device)], dim=1
            )

        # 计算TD误差：δ_t = r_t + γV(s_{t+1}) - V(s_t)
        td_errors = rewards + self.gamma * next_values * (1 - dones) - values

        # 使用GAE计算优势
        advantages = torch.zeros_like(rewards)
        advantage = torch.zeros(batch_size, device=device)

        # 从后往前计算（动态规划）
        for t in reversed(range(seq_len)):
            advantage = td_errors[:, t] + self.gamma * self.lambda_gae * advantage * (
                1 - dones[:, t]
            )
            advantages[:, t] = advantage

        # 计算目标价值（用于价值函数训练）
        returns = advantages + values

        return advantages, returns

    def normalize_advantages(
        self, advantages: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        标准化优势函数（减少方差）

        Args:
            advantages: (batch_size, seq_len) 优势函数
            mask: (batch_size, seq_len) 有效位置掩码

        Returns:
            normalized_advantages: 标准化后的优势函数
        """
        if mask is not None:
            # 只考虑有效位置
            valid_advantages = advantages[mask.bool()]
            if len(valid_advantages) > 1:
                mean = valid_advantages.mean()
                std = valid_advantages.std() + 1e-8
                normalized = (advantages - mean) / std
                return normalized * mask.float()
            else:
                return advantages * mask.float()
        else:
            mean = advantages.mean()
            std = advantages.std() + 1e-8
            return (advantages - mean) / std


class PPOLoss(nn.Module):
    """PPO损失函数

    结合策略损失、价值损失和熵正则化
    """

    def __init__(
        self, clip_ratio: float = 0.2, value_coef: float = 0.5, entropy_coef: float = 0.01
    ):
        """
        Args:
            clip_ratio: PPO裁剪比率 ε
            value_coef: 价值损失系数
            entropy_coef: 熵正则化系数
        """
        super().__init__()
        self.clip_ratio = clip_ratio
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef

    def forward(
        self,
        log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        values: torch.Tensor,
        returns: torch.Tensor,
        entropy: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        计算PPO损失

        Args:
            log_probs: (batch_size, seq_len) 新策略的log概率
            old_log_probs: (batch_size, seq_len) 旧策略的log概率
            advantages: (batch_size, seq_len) 优势函数
            values: (batch_size, seq_len) 价值预测
            returns: (batch_size, seq_len) 目标价值
            entropy: (batch_size, seq_len) 策略熵
            mask: (batch_size, seq_len) 有效位置掩码

        Returns:
            损失字典
        """
        # 计算概率比率
        ratio = torch.exp(log_probs - old_log_probs)

        # PPO裁剪目标
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages
        policy_loss = -torch.min(surr1, surr2)

        # 价值损失（MSE）
        value_loss = F.mse_loss(values, returns, reduction="none")

        # 应用掩码
        if mask is not None:
            policy_loss = policy_loss * mask.float()
            value_loss = value_loss * mask.float()
            entropy = entropy * mask.float()

            # 计算平均损失
            valid_count = mask.sum()
            if valid_count > 0:
                policy_loss = policy_loss.sum() / valid_count
                value_loss = value_loss.sum() / valid_count
                entropy_loss = entropy.sum() / valid_count
            else:
                policy_loss = policy_loss.mean()
                value_loss = value_loss.mean()
                entropy_loss = entropy.mean()
        else:
            policy_loss = policy_loss.mean()
            value_loss = value_loss.mean()
            entropy_loss = entropy.mean()

        # 总损失
        total_loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy_loss

        return {
            "total_loss": total_loss,
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "entropy_loss": entropy_loss,
            "approx_kl": ((ratio - 1) - (log_probs - old_log_probs)).mean(),
        }


class PolicyGradientComputer:
    """策略梯度计算器

    封装策略梯度相关的所有计算逻辑
    """

    def __init__(
        self,
        gamma: float = 0.99,
        lambda_gae: float = 0.95,
        clip_ratio: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
    ):
        """
        初始化策略梯度计算器

        Args:
            gamma: 折扣因子
            lambda_gae: GAE参数
            clip_ratio: PPO裁剪比率
            value_coef: 价值损失系数
            entropy_coef: 熵正则化系数
        """
        self.advantage_calculator = AdvantageCalculator(gamma, lambda_gae)
        self.ppo_loss = PPOLoss(clip_ratio, value_coef, entropy_coef)

    def compute_log_probs(
        self, logits: torch.Tensor, actions: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        计算动作的log概率

        Args:
            logits: (batch_size, seq_len, vocab_size) 模型输出
            actions: (batch_size, seq_len) 采取的动作
            mask: (batch_size, seq_len) 有效位置掩码

        Returns:
            log_probs: (batch_size, seq_len) log概率
        """
        log_probs = F.log_softmax(logits, dim=-1)
        selected_log_probs = torch.gather(log_probs, dim=-1, index=actions.unsqueeze(-1)).squeeze(
            -1
        )

        if mask is not None:
            selected_log_probs = selected_log_probs * mask.float()

        return selected_log_probs

    def compute_entropy(
        self, logits: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        计算策略熵（鼓励探索）

        Args:
            logits: (batch_size, seq_len, vocab_size) 模型输出
            mask: (batch_size, seq_len) 有效位置掩码

        Returns:
            entropy: (batch_size, seq_len) 熵值
        """
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        entropy = -(probs * log_probs).sum(dim=-1)

        if mask is not None:
            entropy = entropy * mask.float()

        return entropy

    def compute_ppo_loss(self, batch_data: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        计算完整的PPO损失

        Args:
            batch_data: 包含所有必要数据的字典
                - new_logits: 新策略的logits
                - old_log_probs: 旧策略的log概率
                - actions: 采取的动作
                - rewards: 奖励
                - values: 价值预测
                - old_values: 旧价值预测
                - dones: 终止标志
                - mask: 有效位置掩码

        Returns:
            损失字典
        """
        # 计算新策略的log概率
        new_log_probs = self.compute_log_probs(
            batch_data["new_logits"], batch_data["actions"], batch_data.get("mask")
        )

        # 计算熵
        entropy = self.compute_entropy(batch_data["new_logits"], batch_data.get("mask"))

        # 计算优势函数
        advantages, returns = self.advantage_calculator.compute_advantages(
            batch_data["rewards"], batch_data["old_values"], batch_data["dones"]
        )

        # 标准化优势
        advantages = self.advantage_calculator.normalize_advantages(
            advantages, batch_data.get("mask")
        )

        # 计算PPO损失
        loss_dict = self.ppo_loss(
            new_log_probs,
            batch_data["old_log_probs"],
            advantages,
            batch_data["values"],
            returns,
            entropy,
            batch_data.get("mask"),
        )

        # 添加一些有用的统计信息
        loss_dict.update(
            {
                "advantages_mean": advantages.mean(),
                "advantages_std": advantages.std(),
                "returns_mean": returns.mean(),
                "entropy_mean": entropy.mean(),
            }
        )

        return loss_dict


def create_policy_gradient_computer(
    gamma: float = 0.99,
    lambda_gae: float = 0.95,
    clip_ratio: float = 0.2,
    value_coef: float = 0.5,
    entropy_coef: float = 0.01,
) -> PolicyGradientComputer:
    """
    创建策略梯度计算器的工厂函数

    Args:
        gamma: 折扣因子
        lambda_gae: GAE参数
        clip_ratio: PPO裁剪比率
        value_coef: 价值损失系数
        entropy_coef: 熵正则化系数

    Returns:
        PolicyGradientComputer: 策略梯度计算器实例
    """
    return PolicyGradientComputer(gamma, lambda_gae, clip_ratio, value_coef, entropy_coef)


if __name__ == "__main__":
    # 简单测试
    print("策略梯度模块实现完成")
    print("主要组件：")
    print("- AdvantageCalculator: 优势函数计算")
    print("- PPOLoss: PPO损失函数")
    print("- PolicyGradientComputer: 策略梯度计算器")
    print("- create_policy_gradient_computer: 工厂函数")
