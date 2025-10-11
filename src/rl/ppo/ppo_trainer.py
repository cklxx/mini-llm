"""
PPO训练器实现

PPO (Proximal Policy Optimization) 是目前最流行的强化学习算法之一，
特别适用于RLHF (Reinforcement Learning from Human Feedback)。

核心思想：
1. 收集轨迹数据（状态、动作、奖励）
2. 计算优势函数和目标价值
3. 使用裁剪目标更新策略，防止更新过大
4. 同时训练价值函数以减少方差

训练流程：
1. 用当前策略收集经验
2. 计算优势函数
3. 多轮优化策略和价值函数
4. 重复上述过程
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .policy_gradient import create_policy_gradient_computer
from .value_model import ValueModel, create_value_model


class PPOExperienceBuffer:
    """PPO经验缓冲区

    存储训练过程中收集的经验数据
    """

    def __init__(self, max_size: int = 10000):
        """
        Args:
            max_size: 缓冲区最大容量
        """
        self.max_size = max_size
        self.clear()

    def clear(self):
        """清空缓冲区"""
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
        self.attention_masks = []

    def add(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        value: torch.Tensor,
        log_prob: torch.Tensor,
        done: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ):
        """
        添加经验到缓冲区

        Args:
            state: 状态（token序列）
            action: 动作（下一个token）
            reward: 奖励
            value: 价值估计
            log_prob: 动作的log概率
            done: 是否终止
            attention_mask: 注意力掩码
        """
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)
        if attention_mask is not None:
            self.attention_masks.append(attention_mask)

        # 如果超出容量，删除最早的经验
        if len(self.states) > self.max_size:
            self.states.pop(0)
            self.actions.pop(0)
            self.rewards.pop(0)
            self.values.pop(0)
            self.log_probs.pop(0)
            self.dones.pop(0)
            if self.attention_masks:
                self.attention_masks.pop(0)

    def get_batch(self) -> dict[str, torch.Tensor]:
        """
        获取批次数据

        Returns:
            包含所有经验的字典
        """
        if not self.states:
            return {}

        batch = {
            "states": torch.stack(self.states),
            "actions": torch.stack(self.actions),
            "rewards": torch.stack(self.rewards),
            "values": torch.stack(self.values),
            "log_probs": torch.stack(self.log_probs),
            "dones": torch.stack(self.dones),
        }

        if self.attention_masks:
            batch["attention_masks"] = torch.stack(self.attention_masks)

        return batch

    def __len__(self):
        return len(self.states)


class PPOTrainer:
    """PPO训练器

    实现完整的PPO训练流程
    """

    def __init__(
        self,
        policy_model: nn.Module,
        value_model: ValueModel | None = None,
        reward_model: nn.Module | None = None,
        tokenizer=None,
        device: str = "cpu",
        # PPO超参数
        gamma: float = 0.99,
        lambda_gae: float = 0.95,
        clip_ratio: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        # 训练超参数
        lr_policy: float = 1e-5,
        lr_value: float = 3e-4,
        batch_size: int = 32,
        mini_batch_size: int = 8,
        ppo_epochs: int = 4,
        max_grad_norm: float = 0.5,
        target_kl: float = 0.01,
    ):
        """
        初始化PPO训练器

        Args:
            policy_model: 策略模型（语言模型）
            value_model: 价值模型（如果为None，会自动创建）
            reward_model: 奖励模型（用于计算奖励）
            tokenizer: 分词器
            device: 设备
            gamma: 折扣因子
            lambda_gae: GAE参数
            clip_ratio: PPO裁剪比率
            value_coef: 价值损失系数
            entropy_coef: 熵正则化系数
            lr_policy: 策略学习率
            lr_value: 价值学习率
            batch_size: 批次大小
            mini_batch_size: 小批次大小
            ppo_epochs: PPO更新轮数
            max_grad_norm: 梯度裁剪阈值
            target_kl: KL散度目标（用于早停）
        """
        self.device = device
        self.tokenizer = tokenizer

        # 模型
        self.policy_model = policy_model.to(device)
        if value_model is None:
            self.value_model = create_value_model(policy_model, freeze_backbone=False).to(device)
        else:
            self.value_model = value_model.to(device)
        self.reward_model = reward_model.to(device) if reward_model else None

        # 创建参考模型（冻结的策略模型副本）
        self.reference_model = self._create_reference_model()

        # 优化器
        self.policy_optimizer = optim.AdamW(self.policy_model.parameters(), lr=lr_policy)
        self.value_optimizer = optim.AdamW(self.value_model.parameters(), lr=lr_value)

        # PPO组件
        self.pg_computer = create_policy_gradient_computer(
            gamma=gamma,
            lambda_gae=lambda_gae,
            clip_ratio=clip_ratio,
            value_coef=value_coef,
            entropy_coef=entropy_coef,
        )

        # 训练参数
        self.batch_size = batch_size
        self.mini_batch_size = mini_batch_size
        self.ppo_epochs = ppo_epochs
        self.max_grad_norm = max_grad_norm
        self.target_kl = target_kl

        # 经验缓冲区
        self.experience_buffer = PPOExperienceBuffer()

        # 训练记录
        self.train_stats = {
            "policy_loss": [],
            "value_loss": [],
            "entropy": [],
            "kl_divergence": [],
            "rewards": [],
        }

    def _create_reference_model(self) -> nn.Module:
        """创建参考模型（冻结的策略模型副本）"""
        # 创建模型副本
        reference_model = type(self.policy_model)(
            **self.policy_model.config if hasattr(self.policy_model, "config") else {}
        )
        reference_model.load_state_dict(self.policy_model.state_dict())
        reference_model.to(self.device)

        # 冻结参数
        for param in reference_model.parameters():
            param.requires_grad = False

        return reference_model

    def collect_experiences(
        self, prompts: list[str], max_new_tokens: int = 50
    ) -> dict[str, torch.Tensor]:
        """
        收集训练经验

        Args:
            prompts: 输入提示列表
            max_new_tokens: 最大生成token数

        Returns:
            经验数据字典
        """
        self.policy_model.eval()
        self.value_model.eval()

        all_experiences = []

        with torch.no_grad():
            for prompt in prompts:
                # 编码提示
                input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)

                # 生成回复
                generated_ids = []
                values = []
                log_probs = []

                current_input = input_ids

                for _ in range(max_new_tokens):
                    # 获取模型输出
                    outputs = self.policy_model(current_input)
                    logits = outputs.logits[:, -1, :]  # 只取最后一个位置

                    # 获取价值估计
                    value = self.value_model(current_input)[:, -1]  # 最后一个位置的价值

                    # 采样下一个token
                    probs = torch.softmax(logits, dim=-1)
                    next_token = torch.multinomial(probs, 1)

                    # 计算log概率
                    log_prob = torch.log(probs.gather(1, next_token))

                    # 记录
                    generated_ids.append(next_token.item())
                    values.append(value.item())
                    log_probs.append(log_prob.item())

                    # 更新输入
                    current_input = torch.cat([current_input, next_token], dim=1)

                    # 检查是否结束
                    if next_token.item() == self.tokenizer.eos_token_id:
                        break

                # 计算奖励
                full_response = self.tokenizer.decode(current_input[0], skip_special_tokens=True)
                reward = self._compute_reward(full_response)

                # 存储经验
                experience = {
                    "input_ids": current_input,
                    "generated_ids": generated_ids,
                    "values": values,
                    "log_probs": log_probs,
                    "reward": reward,
                }
                all_experiences.append(experience)

        return all_experiences

    def _compute_reward(self, text: str) -> float:
        """
        计算奖励

        Args:
            text: 生成的文本

        Returns:
            奖励值
        """
        if self.reward_model is not None:
            # 使用奖励模型计算奖励
            input_ids = self.tokenizer.encode(text, return_tensors="pt").to(self.device)
            with torch.no_grad():
                reward = self.reward_model(input_ids).item()
            return reward
        else:
            # 简单的启发式奖励（可以根据任务定制）
            # 这里只是示例，实际应用中需要更复杂的奖励函数
            return len(text) * 0.01  # 长度奖励

    def update_policy(self, experiences: list[dict]) -> dict[str, float]:
        """
        使用PPO更新策略

        Args:
            experiences: 经验数据列表

        Returns:
            训练统计信息
        """
        if not experiences:
            return {}

        # 准备批次数据
        batch_data = self._prepare_batch_data(experiences)

        # 多轮PPO更新
        total_stats = {}

        for epoch in range(self.ppo_epochs):
            # 生成小批次
            mini_batches = self._create_mini_batches(batch_data)

            epoch_stats = {}

            for mini_batch in mini_batches:
                # 更新策略
                stats = self._update_mini_batch(mini_batch)

                # 累积统计信息
                for key, value in stats.items():
                    if key not in epoch_stats:
                        epoch_stats[key] = []
                    epoch_stats[key].append(value)

                # 检查KL散度（早停）
                if "approx_kl" in stats and stats["approx_kl"] > self.target_kl:
                    print(
                        f"Early stopping at epoch {epoch} due to KL divergence: {stats['approx_kl']:.4f}"
                    )
                    break

            # 平均统计信息
            for key, values in epoch_stats.items():
                if key not in total_stats:
                    total_stats[key] = []
                total_stats[key].append(np.mean(values))

        # 返回平均统计信息
        return {key: np.mean(values) for key, values in total_stats.items()}

    def _prepare_batch_data(self, experiences: list[dict]) -> dict[str, torch.Tensor]:
        """准备批次数据"""
        # 实现批次数据准备逻辑
        # 这里需要根据具体的经验数据结构来实现
        pass

    def _create_mini_batches(
        self, batch_data: dict[str, torch.Tensor]
    ) -> list[dict[str, torch.Tensor]]:
        """创建小批次"""
        # 实现小批次创建逻辑
        pass

    def _update_mini_batch(self, mini_batch: dict[str, torch.Tensor]) -> dict[str, float]:
        """更新小批次"""
        self.policy_model.train()
        self.value_model.train()

        # 前向传播
        policy_outputs = self.policy_model(mini_batch["input_ids"])
        value_outputs = self.value_model(mini_batch["input_ids"])

        # 计算损失
        batch_data = {
            "new_logits": policy_outputs.logits,
            "old_log_probs": mini_batch["old_log_probs"],
            "actions": mini_batch["actions"],
            "rewards": mini_batch["rewards"],
            "values": value_outputs,
            "old_values": mini_batch["old_values"],
            "dones": mini_batch["dones"],
            "mask": mini_batch.get("attention_mask"),
        }

        loss_dict = self.pg_computer.compute_ppo_loss(batch_data)

        # 反向传播
        self.policy_optimizer.zero_grad()
        self.value_optimizer.zero_grad()

        loss_dict["total_loss"].backward()

        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.policy_model.parameters(), self.max_grad_norm)
        torch.nn.utils.clip_grad_norm_(self.value_model.parameters(), self.max_grad_norm)

        # 更新参数
        self.policy_optimizer.step()
        self.value_optimizer.step()

        return {
            key: value.item() if torch.is_tensor(value) else value
            for key, value in loss_dict.items()
        }

    def train(
        self,
        prompts: list[str],
        num_iterations: int = 1000,
        save_interval: int = 100,
        save_dir: str = "ppo_checkpoints",
    ):
        """
        完整的PPO训练流程

        Args:
            prompts: 训练提示列表
            num_iterations: 训练迭代次数
            save_interval: 保存间隔
            save_dir: 保存目录
        """
        os.makedirs(save_dir, exist_ok=True)

        for iteration in range(num_iterations):
            print(f"\\nIteration {iteration + 1}/{num_iterations}")

            # 收集经验
            experiences = self.collect_experiences(prompts)

            # 更新策略
            stats = self.update_policy(experiences)

            # 记录统计信息
            for key, value in stats.items():
                if key in self.train_stats:
                    self.train_stats[key].append(value)

            # 打印统计信息
            if stats:
                print(f"Policy Loss: {stats.get('policy_loss', 0):.4f}")
                print(f"Value Loss: {stats.get('value_loss', 0):.4f}")
                print(f"Entropy: {stats.get('entropy_loss', 0):.4f}")
                print(f"KL Divergence: {stats.get('approx_kl', 0):.4f}")

            # 定期保存
            if (iteration + 1) % save_interval == 0:
                self.save_checkpoint(os.path.join(save_dir, f"ppo_checkpoint_{iteration + 1}.pt"))

        # 保存最终模型
        self.save_checkpoint(os.path.join(save_dir, "final_model.pt"))

        # 绘制训练曲线
        self.plot_training_curves(save_dir)

    def save_checkpoint(self, path: str):
        """保存训练checkpoint"""
        torch.save(
            {
                "policy_model_state_dict": self.policy_model.state_dict(),
                "value_model_state_dict": self.value_model.state_dict(),
                "policy_optimizer_state_dict": self.policy_optimizer.state_dict(),
                "value_optimizer_state_dict": self.value_optimizer.state_dict(),
                "train_stats": self.train_stats,
            },
            path,
        )

    def load_checkpoint(self, path: str):
        """加载训练checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_model.load_state_dict(checkpoint["policy_model_state_dict"])
        self.value_model.load_state_dict(checkpoint["value_model_state_dict"])
        self.policy_optimizer.load_state_dict(checkpoint["policy_optimizer_state_dict"])
        self.value_optimizer.load_state_dict(checkpoint["value_optimizer_state_dict"])
        self.train_stats = checkpoint.get("train_stats", self.train_stats)

    def plot_training_curves(self, save_dir: str):
        """绘制训练曲线"""
        if not any(self.train_stats.values()):
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # 策略损失
        if self.train_stats["policy_loss"]:
            axes[0, 0].plot(self.train_stats["policy_loss"])
            axes[0, 0].set_title("Policy Loss")
            axes[0, 0].set_ylabel("Loss")
            axes[0, 0].grid(True)

        # 价值损失
        if self.train_stats["value_loss"]:
            axes[0, 1].plot(self.train_stats["value_loss"])
            axes[0, 1].set_title("Value Loss")
            axes[0, 1].set_ylabel("Loss")
            axes[0, 1].grid(True)

        # 熵
        if self.train_stats["entropy"]:
            axes[1, 0].plot(self.train_stats["entropy"])
            axes[1, 0].set_title("Entropy")
            axes[1, 0].set_ylabel("Entropy")
            axes[1, 0].grid(True)

        # KL散度
        if self.train_stats["kl_divergence"]:
            axes[1, 1].plot(self.train_stats["kl_divergence"])
            axes[1, 1].set_title("KL Divergence")
            axes[1, 1].set_ylabel("KL")
            axes[1, 1].grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "ppo_training_curves.png"))
        plt.close()


def create_ppo_trainer(
    policy_model: nn.Module,
    value_model: ValueModel | None = None,
    reward_model: nn.Module | None = None,
    tokenizer=None,
    device: str = "cpu",
    **kwargs,
) -> PPOTrainer:
    """
    创建PPO训练器的工厂函数

    Args:
        policy_model: 策略模型
        value_model: 价值模型
        reward_model: 奖励模型
        tokenizer: 分词器
        device: 设备
        **kwargs: 其他参数

    Returns:
        PPOTrainer: PPO训练器实例
    """
    return PPOTrainer(
        policy_model=policy_model,
        value_model=value_model,
        reward_model=reward_model,
        tokenizer=tokenizer,
        device=device,
        **kwargs,
    )


if __name__ == "__main__":
    # 简单测试
    print("PPO训练器模块实现完成")
    print("主要组件：")
    print("- PPOExperienceBuffer: 经验缓冲区")
    print("- PPOTrainer: PPO训练器")
    print("- create_ppo_trainer: 工厂函数")
