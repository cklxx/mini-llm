# 04 DPO与替代RLHF方法

> **跳出强化学习框架：直接偏好优化的数学革新**

## 核心思想

直接偏好优化(Direct Preference Optimization, DPO)是RLHF领域的重大突破。它绕过了传统RLHF中复杂的奖励建模和强化学习训练过程，直接在偏好数据上优化语言模型，实现了更简单、更稳定、更高效的对齐训练。

**关键洞察**：
- **消除奖励模型**：直接利用偏好比较数据，无需显式奖励函数
- **避开RL复杂性**：跳过PPO等复杂的强化学习算法
- **理论等价性**：数学上等价于隐式奖励模型的最优解
- **训练稳定性**：避免了RL训练的不稳定性问题

DPO的数学精髓在于重新参数化偏好学习问题，将强化学习优化转化为简单的分类损失。

## 4.1 DPO的数学推导与理论基础

### 从奖励模型到直接优化

**传统RLHF的问题分解**：
1. **第一阶段**：训练奖励模型 $R_\phi(x,y)$
2. **第二阶段**：用RL优化策略 $\pi_\theta$

DPO的关键观察：我们可以直接得到最优策略的闭式解！

**Bradley-Terry模型回顾**：
$$P(y_1 \succ y_2 | x) = \frac{\exp(R(x, y_1))}{\exp(R(x, y_1)) + \exp(R(x, y_2))} = \sigma(R(x, y_1) - R(x, y_2))$$

**RL目标函数**：
$$\max_{\pi} \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi(\cdot|x)}[R(x,y)] - \beta \mathbb{E}_{x \sim \mathcal{D}}[D_{KL}(\pi(\cdot|x) \| \pi_{ref}(\cdot|x))]$$

**关键定理**：上述RL优化问题的最优解为：
$$\pi^*(y|x) = \frac{1}{Z(x)} \pi_{ref}(y|x) \exp\left(\frac{1}{\beta} R(x,y)\right)$$

其中 $Z(x) = \sum_y \pi_{ref}(y|x) \exp(\frac{1}{\beta} R(x,y))$ 是配分函数。

**DPO的核心洞察**：我们可以反解出隐式奖励函数：
$$R(x,y) = \beta \log \frac{\pi^*(y|x)}{\pi_{ref}(y|x)} + \beta \log Z(x)$$

由于 $Z(x)$ 在比较中被消除，我们得到：
$$R(x, y_1) - R(x, y_2) = \beta \log \frac{\pi^*(y_1|x)}{\pi_{ref}(y_1|x)} - \beta \log \frac{\pi^*(y_2|x)}{\pi_{ref}(y_2|x)}$$

**DPO损失函数**：
$$\mathcal{L}_{DPO}(\pi_\theta) = -\mathbb{E}_{(x,y_w,y_l) \sim \mathcal{D}}[\log \sigma(\beta \log \frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)})]$$

其中 $y_w$ 是偏好回复，$y_l$ 是非偏好回复。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
import matplotlib.pyplot as plt
import seaborn as sns
import math
import random
from scipy import stats
from collections import defaultdict, Counter

@dataclass
class DPOTrainingData:
    """DPO训练数据结构"""
    prompt: str                    # 输入提示
    chosen_response: str           # 偏好回复
    rejected_response: str         # 非偏好回复
    margin: float = 1.0           # 偏好强度
    metadata: Dict = None         # 元数据
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

class DPOTrainer:
    """直接偏好优化训练器"""
    
    def __init__(self, model, reference_model, tokenizer=None, 
                 beta: float = 0.1, max_length: int = 512):
        
        self.model = model                    # 待训练模型
        self.reference_model = reference_model # 参考模型（冻结）
        self.tokenizer = tokenizer
        self.beta = beta                      # DPO温度参数
        self.max_length = max_length
        
        # 冻结参考模型
        for param in self.reference_model.parameters():
            param.requires_grad = False
        self.reference_model.eval()
        
        # 训练统计
        self.training_stats = {
            'loss': [],
            'accuracy': [],
            'chosen_rewards': [],
            'rejected_rewards': [],
            'reward_margins': [],
            'kl_divergence': []
        }
        
    def compute_log_probabilities(self, model, prompts: List[str], 
                                responses: List[str]) -> torch.Tensor:
        """计算序列的对数概率"""
        
        log_probs = []
        
        for prompt, response in zip(prompts, responses):
            # 简化的token化（实际需要真实tokenizer）
            full_text = prompt + " " + response
            tokens = self._tokenize_text(full_text)
            
            if len(tokens) < 2:
                log_probs.append(torch.tensor(-10.0))  # 很低的概率
                continue
            
            # 分离prompt和response tokens
            prompt_tokens = self._tokenize_text(prompt)
            prompt_length = len(prompt_tokens)
            
            # 将tokens转换为tensor
            input_ids = torch.tensor(tokens[:-1], dtype=torch.long).unsqueeze(0)
            target_ids = torch.tensor(tokens[1:], dtype=torch.long)
            
            with torch.no_grad() if model == self.reference_model else torch.enable_grad():
                # 前向传播获取logits
                logits = self._get_model_logits(model, input_ids)
                
                # 计算对数概率
                log_probs_seq = F.log_softmax(logits, dim=-1)
                
                # 只计算response部分的概率
                response_log_probs = []
                for i in range(prompt_length, len(tokens) - 1):
                    if i < log_probs_seq.size(1):
                        token_log_prob = log_probs_seq[0, i, target_ids[i]]
                        response_log_probs.append(token_log_prob)
                
                if response_log_probs:
                    total_log_prob = torch.stack(response_log_probs).sum()
                else:
                    total_log_prob = torch.tensor(-10.0)
                
                log_probs.append(total_log_prob)
        
        return torch.stack(log_probs)
    
    def _tokenize_text(self, text: str) -> List[int]:
        """简化的token化"""
        if self.tokenizer:
            return self.tokenizer.encode(text)
        else:
            # 简化实现
            words = text.lower().split()
            tokens = [2]  # BOS
            for word in words[:30]:  # 限制长度
                token_id = abs(hash(word)) % 9997 + 3  # 避开特殊tokens
                tokens.append(token_id)
            tokens.append(1)  # EOS
            return tokens
    
    def _get_model_logits(self, model, input_ids: torch.Tensor) -> torch.Tensor:
        """获取模型logits（简化实现）"""
        # 实际应用中这里应该是model(input_ids).logits
        # 这里用随机logits模拟
        batch_size, seq_len = input_ids.shape
        vocab_size = 10000
        return torch.randn(batch_size, seq_len, vocab_size)
    
    def compute_dpo_loss(self, batch_data: List[DPOTrainingData]) -> Dict[str, torch.Tensor]:
        """计算DPO损失"""
        
        if not batch_data:
            return {'loss': torch.tensor(0.0)}
        
        prompts = [data.prompt for data in batch_data]
        chosen_responses = [data.chosen_response for data in batch_data]
        rejected_responses = [data.rejected_response for data in batch_data]
        
        # 计算模型对数概率
        chosen_log_probs = self.compute_log_probabilities(self.model, prompts, chosen_responses)
        rejected_log_probs = self.compute_log_probabilities(self.model, prompts, rejected_responses)
        
        # 计算参考模型对数概率
        chosen_ref_log_probs = self.compute_log_probabilities(self.reference_model, prompts, chosen_responses)
        rejected_ref_log_probs = self.compute_log_probabilities(self.reference_model, prompts, rejected_responses)
        
        # 计算隐式奖励
        chosen_rewards = self.beta * (chosen_log_probs - chosen_ref_log_probs)
        rejected_rewards = self.beta * (rejected_log_probs - rejected_ref_log_probs)
        
        # DPO损失
        logits = chosen_rewards - rejected_rewards
        loss = -F.logsigmoid(logits).mean()
        
        # 计算准确率
        with torch.no_grad():
            accuracy = (logits > 0).float().mean()
            
            # KL散度估计
            chosen_kl = (chosen_log_probs - chosen_ref_log_probs).mean()
            rejected_kl = (rejected_log_probs - rejected_ref_log_probs).mean()
            avg_kl = (chosen_kl + rejected_kl) / 2
        
        return {
            'loss': loss,
            'accuracy': accuracy,
            'chosen_rewards': chosen_rewards.mean(),
            'rejected_rewards': rejected_rewards.mean(),
            'reward_margin': (chosen_rewards - rejected_rewards).mean(),
            'kl_divergence': avg_kl,
            'logits': logits.mean()
        }
    
    def train_step(self, batch_data: List[DPOTrainingData], 
                   optimizer: torch.optim.Optimizer) -> Dict[str, float]:
        """执行一步DPO训练"""
        
        # 前向传播
        loss_dict = self.compute_dpo_loss(batch_data)
        
        # 反向传播
        optimizer.zero_grad()
        loss_dict['loss'].backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # 转换为标量值
        scalar_dict = {}
        for key, value in loss_dict.items():
            if isinstance(value, torch.Tensor):
                scalar_dict[key] = value.item()
            else:
                scalar_dict[key] = value
        
        # 更新统计
        self.training_stats['loss'].append(scalar_dict['loss'])
        self.training_stats['accuracy'].append(scalar_dict['accuracy'])
        self.training_stats['chosen_rewards'].append(scalar_dict['chosen_rewards'])
        self.training_stats['rejected_rewards'].append(scalar_dict['rejected_rewards'])
        self.training_stats['reward_margins'].append(scalar_dict['reward_margin'])
        self.training_stats['kl_divergence'].append(scalar_dict['kl_divergence'])
        
        return scalar_dict
    
    def full_training_loop(self, training_data: List[DPOTrainingData],
                          validation_data: List[DPOTrainingData] = None,
                          num_epochs: int = 3, batch_size: int = 4,
                          learning_rate: float = 1e-6) -> Dict:
        """完整的DPO训练循环"""
        
        print(f"=== DPO训练循环 ===")
        print(f"训练数据: {len(training_data)} 个偏好对")
        print(f"训练轮数: {num_epochs}")
        print(f"批次大小: {batch_size}")
        print(f"学习率: {learning_rate}")
        
        # 创建优化器
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
        
        training_history = []
        
        for epoch in range(num_epochs):
            print(f"\n--- Epoch {epoch + 1}/{num_epochs} ---")
            
            # 打乱训练数据
            shuffled_data = training_data.copy()
            random.shuffle(shuffled_data)
            
            epoch_losses = []
            epoch_accuracies = []
            
            # 分批训练
            for i in range(0, len(shuffled_data), batch_size):
                batch = shuffled_data[i:i + batch_size]
                
                if not batch:
                    continue
                
                # 训练步骤
                step_stats = self.train_step(batch, optimizer)
                
                epoch_losses.append(step_stats['loss'])
                epoch_accuracies.append(step_stats['accuracy'])
                
                if i % (batch_size * 5) == 0:  # 每几个batch打印一次
                    print(f"  Batch {i//batch_size + 1}: "
                          f"Loss={step_stats['loss']:.4f}, "
                          f"Acc={step_stats['accuracy']:.3f}, "
                          f"Margin={step_stats['reward_margin']:.4f}")
            
            # 学习率调度
            scheduler.step()
            
            # Epoch统计
            epoch_stats = {
                'epoch': epoch,
                'avg_loss': np.mean(epoch_losses),
                'avg_accuracy': np.mean(epoch_accuracies),
                'learning_rate': scheduler.get_last_lr()[0]
            }
            
            # 验证集评估
            if validation_data:
                val_stats = self._evaluate(validation_data)
                epoch_stats.update(val_stats)
            
            training_history.append(epoch_stats)
            
            print(f"  Epoch {epoch + 1} 完成: "
                  f"平均损失={epoch_stats['avg_loss']:.4f}, "
                  f"平均准确率={epoch_stats['avg_accuracy']:.3f}")
        
        # 分析训练结果
        self._analyze_training_results(training_history)
        
        return {
            'training_history': training_history,
            'final_stats': training_history[-1] if training_history else {},
            'model_state': self.model.state_dict()
        }
    
    def _evaluate(self, validation_data: List[DPOTrainingData]) -> Dict[str, float]:
        """评估模型性能"""
        
        self.model.eval()
        
        val_losses = []
        val_accuracies = []
        
        with torch.no_grad():
            for i in range(0, len(validation_data), 4):  # 小批次评估
                batch = validation_data[i:i + 4]
                
                if not batch:
                    continue
                
                loss_dict = self.compute_dpo_loss(batch)
                val_losses.append(loss_dict['loss'].item())
                val_accuracies.append(loss_dict['accuracy'].item())
        
        self.model.train()
        
        return {
            'val_loss': np.mean(val_losses) if val_losses else 0.0,
            'val_accuracy': np.mean(val_accuracies) if val_accuracies else 0.0
        }
    
    def _analyze_training_results(self, training_history: List[Dict]):
        """分析训练结果"""
        
        print(f"\n=== DPO训练结果分析 ===")
        
        if not training_history:
            print("无训练历史数据")
            return
        
        final_stats = training_history[-1]
        print(f"最终训练损失: {final_stats['avg_loss']:.4f}")
        print(f"最终训练准确率: {final_stats['avg_accuracy']:.3f}")
        
        if 'val_loss' in final_stats:
            print(f"最终验证损失: {final_stats['val_loss']:.4f}")
            print(f"最终验证准确率: {final_stats['val_accuracy']:.3f}")
        
        # 收敛分析
        losses = [h['avg_loss'] for h in training_history]
        if len(losses) >= 2:
            loss_improvement = losses[0] - losses[-1]
            print(f"损失改进: {loss_improvement:.4f}")
            
            # 检查过拟合
            if 'val_loss' in training_history[-1]:
                val_losses = [h.get('val_loss', 0) for h in training_history]
                if len(val_losses) >= 2 and val_losses[-1] > val_losses[-2]:
                    print("⚠️ 可能存在过拟合现象")
        
        # 可视化训练过程
        self._visualize_training_progress(training_history)
    
    def _visualize_training_progress(self, training_history: List[Dict]):
        """可视化训练进度"""
        
        if len(training_history) < 2:
            return
        
        epochs = [h['epoch'] for h in training_history]
        train_losses = [h['avg_loss'] for h in training_history]
        train_accuracies = [h['avg_accuracy'] for h in training_history]
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # 损失曲线
        axes[0].plot(epochs, train_losses, 'b-', linewidth=2, label='训练损失')
        
        if 'val_loss' in training_history[0]:
            val_losses = [h.get('val_loss', 0) for h in training_history]
            axes[0].plot(epochs, val_losses, 'r--', linewidth=2, label='验证损失')
        
        axes[0].set_title('DPO训练损失')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('损失')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 准确率曲线
        axes[1].plot(epochs, train_accuracies, 'g-', linewidth=2, label='训练准确率')
        
        if 'val_accuracy' in training_history[0]:
            val_accuracies = [h.get('val_accuracy', 0) for h in training_history]
            axes[1].plot(epochs, val_accuracies, 'orange', linestyle='--', linewidth=2, label='验证准确率')
        
        axes[1].set_title('DPO训练准确率')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('准确率')
        axes[1].set_ylim(0, 1)
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def analyze_preference_learning(self, test_data: List[DPOTrainingData]) -> Dict:
        """分析偏好学习效果"""
        
        print(f"=== 偏好学习效果分析 ===")
        
        self.model.eval()
        
        with torch.no_grad():
            all_margins = []
            correct_predictions = 0
            total_predictions = len(test_data)
            
            for data in test_data:
                # 计算隐式奖励
                chosen_log_prob = self.compute_log_probabilities(
                    self.model, [data.prompt], [data.chosen_response]
                )[0]
                rejected_log_prob = self.compute_log_probabilities(
                    self.model, [data.prompt], [data.rejected_response]
                )[0]
                
                chosen_ref_log_prob = self.compute_log_probabilities(
                    self.reference_model, [data.prompt], [data.chosen_response]
                )[0]
                rejected_ref_log_prob = self.compute_log_probabilities(
                    self.reference_model, [data.prompt], [data.rejected_response]
                )[0]
                
                chosen_reward = self.beta * (chosen_log_prob - chosen_ref_log_prob)
                rejected_reward = self.beta * (rejected_log_prob - rejected_ref_log_prob)
                
                margin = chosen_reward - rejected_reward
                all_margins.append(margin.item())
                
                if margin > 0:
                    correct_predictions += 1
        
        accuracy = correct_predictions / total_predictions
        avg_margin = np.mean(all_margins)
        margin_std = np.std(all_margins)
        
        results = {
            'accuracy': accuracy,
            'average_margin': avg_margin,
            'margin_std': margin_std,
            'margins': all_margins
        }
        
        print(f"测试准确率: {accuracy:.3f}")
        print(f"平均奖励差距: {avg_margin:.4f}")
        print(f"差距标准差: {margin_std:.4f}")
        
        # 可视化奖励差距分布
        plt.figure(figsize=(8, 4))
        plt.hist(all_margins, bins=20, alpha=0.7, edgecolor='black')
        plt.axvline(x=0, color='red', linestyle='--', alpha=0.7, label='中性线')
        plt.xlabel('奖励差距 (偏好 - 非偏好)')
        plt.ylabel('频次')
        plt.title('DPO学习的偏好差距分布')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
        
        return results

class DPOVariants:
    """DPO变体方法"""
    
    @staticmethod
    def identity_dpo_loss(chosen_log_probs, rejected_log_probs, chosen_ref_log_probs, 
                         rejected_ref_log_probs, beta=0.1):
        """Identity DPO: 更强的正则化"""
        
        chosen_rewards = beta * (chosen_log_probs - chosen_ref_log_probs)
        rejected_rewards = beta * (rejected_log_probs - rejected_ref_log_probs)
        
        # 添加身份映射项
        identity_term = 0.1 * (chosen_log_probs - chosen_ref_log_probs).pow(2).mean()
        
        logits = chosen_rewards - rejected_rewards
        loss = -F.logsigmoid(logits).mean() + identity_term
        
        return loss
    
    @staticmethod
    def kto_loss(chosen_log_probs, rejected_log_probs, chosen_ref_log_probs, 
                 rejected_ref_log_probs, beta=0.1, desirable_weight=1.0, undesirable_weight=1.0):
        """Kahneman-Tversky Optimization (KTO): 基于前景理论"""
        
        # 计算KL散度
        chosen_kl = chosen_log_probs - chosen_ref_log_probs
        rejected_kl = rejected_log_probs - rejected_ref_log_probs
        
        # KTO损失：分别处理偏好和非偏好
        chosen_loss = -F.logsigmoid(beta * chosen_kl - 0.1).mean()
        rejected_loss = -F.logsigmoid(-beta * rejected_kl - 0.1).mean()
        
        total_loss = desirable_weight * chosen_loss + undesirable_weight * rejected_loss
        
        return total_loss
    
    @staticmethod
    def rso_loss(chosen_log_probs, rejected_log_probs, chosen_ref_log_probs, 
                 rejected_ref_log_probs, beta=0.1, alpha=0.9):
        """Statistical Rejection Sampling Optimization (RSO)"""
        
        chosen_rewards = beta * (chosen_log_probs - chosen_ref_log_probs)
        rejected_rewards = beta * (rejected_log_probs - rejected_ref_log_probs)
        
        # RSO使用重要性权重
        importance_weights = torch.exp(alpha * (chosen_rewards - rejected_rewards))
        weighted_logits = importance_weights * (chosen_rewards - rejected_rewards)
        
        loss = -F.logsigmoid(weighted_logits).mean()
        
        return loss

class DPOvsRLHFComparison:
    """DPO与RLHF对比分析"""
    
    def __init__(self):
        self.comparison_results = {}
    
    def compare_training_efficiency(self, dpo_history: List[Dict], rlhf_history: List[Dict]) -> Dict:
        """比较训练效率"""
        
        print(f"=== DPO vs RLHF 效率对比 ===")
        
        # 收敛速度对比
        dpo_convergence = self._analyze_convergence_speed(dpo_history, 'avg_loss')
        rlhf_convergence = self._analyze_convergence_speed(rlhf_history, 'avg_reward')
        
        # 训练稳定性对比
        dpo_stability = self._analyze_training_stability(dpo_history, 'avg_loss')
        rlhf_stability = self._analyze_training_stability(rlhf_history, 'avg_reward')
        
        # 内存使用估计（简化）
        dpo_memory = self._estimate_memory_usage('dpo')
        rlhf_memory = self._estimate_memory_usage('rlhf')
        
        comparison = {
            'convergence_speed': {
                'dpo': dpo_convergence,
                'rlhf': rlhf_convergence,
                'winner': 'DPO' if dpo_convergence > rlhf_convergence else 'RLHF'
            },
            'training_stability': {
                'dpo': dpo_stability,
                'rlhf': rlhf_stability,
                'winner': 'DPO' if dpo_stability > rlhf_stability else 'RLHF'
            },
            'memory_efficiency': {
                'dpo': dpo_memory,
                'rlhf': rlhf_memory,
                'winner': 'DPO' if dpo_memory < rlhf_memory else 'RLHF'
            }
        }
        
        print(f"收敛速度: {comparison['convergence_speed']['winner']} 胜")
        print(f"训练稳定性: {comparison['training_stability']['winner']} 胜")
        print(f"内存效率: {comparison['memory_efficiency']['winner']} 胜")
        
        # 可视化对比
        self._visualize_comparison(comparison)
        
        return comparison
    
    def _analyze_convergence_speed(self, history: List[Dict], metric_key: str) -> float:
        """分析收敛速度"""
        
        if len(history) < 3:
            return 0.0
        
        values = [h.get(metric_key, 0) for h in history]
        
        # 计算改进率
        initial_value = values[0]
        final_value = values[-1]
        
        if metric_key == 'avg_loss':
            improvement = initial_value - final_value  # 损失降低
        else:
            improvement = final_value - initial_value  # 奖励增加
        
        # 标准化改进率
        convergence_speed = improvement / (abs(initial_value) + 1e-10)
        
        return max(0, convergence_speed)
    
    def _analyze_training_stability(self, history: List[Dict], metric_key: str) -> float:
        """分析训练稳定性"""
        
        if len(history) < 5:
            return 0.0
        
        values = [h.get(metric_key, 0) for h in history]
        
        # 计算后半段的方差（越小越稳定）
        mid_point = len(values) // 2
        late_values = values[mid_point:]
        
        stability = 1.0 / (1.0 + np.std(late_values))
        
        return stability
    
    def _estimate_memory_usage(self, method: str) -> float:
        """估计内存使用（简化模拟）"""
        
        if method == 'dpo':
            # DPO只需要训练模型和参考模型
            return 2.0  # 2x模型大小
        else:
            # RLHF需要策略模型、价值模型、奖励模型
            return 3.5  # 3.5x模型大小
    
    def _visualize_comparison(self, comparison: Dict):
        """可视化对比结果"""
        
        methods = ['DPO', 'RLHF']
        
        # 创建对比图表
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        # 收敛速度
        speeds = [comparison['convergence_speed']['dpo'], comparison['convergence_speed']['rlhf']]
        bars1 = axes[0].bar(methods, speeds, alpha=0.7, color=['blue', 'orange'])
        axes[0].set_title('收敛速度对比')
        axes[0].set_ylabel('收敛速度')
        
        # 标记获胜者
        winner_idx = 0 if comparison['convergence_speed']['winner'] == 'DPO' else 1
        bars1[winner_idx].set_color('green')
        
        # 训练稳定性
        stabilities = [comparison['training_stability']['dpo'], comparison['training_stability']['rlhf']]
        bars2 = axes[1].bar(methods, stabilities, alpha=0.7, color=['blue', 'orange'])
        axes[1].set_title('训练稳定性对比')
        axes[1].set_ylabel('稳定性分数')
        
        winner_idx = 0 if comparison['training_stability']['winner'] == 'DPO' else 1
        bars2[winner_idx].set_color('green')
        
        # 内存效率
        memory_usage = [comparison['memory_efficiency']['dpo'], comparison['memory_efficiency']['rlhf']]
        bars3 = axes[2].bar(methods, memory_usage, alpha=0.7, color=['blue', 'orange'])
        axes[2].set_title('内存使用对比')
        axes[2].set_ylabel('相对内存使用')
        
        # 内存使用越低越好
        winner_idx = 0 if comparison['memory_efficiency']['winner'] == 'DPO' else 1
        bars3[winner_idx].set_color('green')
        
        plt.tight_layout()
        plt.show()

# DPO完整演示
def demonstrate_dpo_training():
    """演示DPO训练完整流程"""
    
    print("="*60)
    print("直接偏好优化(DPO) - 综合演示")
    print("="*60)
    
    # 1. 创建模型（简化版本）
    from第01节的PolicyNetwork import PolicyNetwork
    
    model = PolicyNetwork(vocab_size=1000, hidden_size=256)
    reference_model = PolicyNetwork(vocab_size=1000, hidden_size=256)
    
    # 将参考模型设置为模型的副本
    reference_model.load_state_dict(model.state_dict())
    
    # 2. 创建DPO训练器
    dpo_trainer = DPOTrainer(
        model=model,
        reference_model=reference_model,
        beta=0.1,
        max_length=256
    )
    
    # 3. 准备训练数据
    training_data = [
        DPOTrainingData(
            prompt="What is the capital of France?",
            chosen_response="The capital of France is Paris, a beautiful city known for its rich history and culture.",
            rejected_response="France capital is Paris."
        ),
        DPOTrainingData(
            prompt="Explain machine learning in simple terms.",
            chosen_response="Machine learning is a branch of artificial intelligence where computers learn to recognize patterns from data without being explicitly programmed for every task.",
            rejected_response="ML is when computers learn stuff."
        ),
        DPOTrainingData(
            prompt="How do you make pasta?",
            chosen_response="To make pasta, bring a large pot of salted water to a rolling boil, add pasta, cook according to package directions, then drain and serve with your favorite sauce.",
            rejected_response="Boil water, add pasta, done."
        ),
        DPOTrainingData(
            prompt="What are the benefits of renewable energy?",
            chosen_response="Renewable energy sources like solar and wind power provide clean electricity without greenhouse gas emissions, reduce dependence on fossil fuels, and can lead to long-term cost savings.",
            rejected_response="Clean energy good for environment."
        ),
        DPOTrainingData(
            prompt="Describe the process of photosynthesis.",
            chosen_response="Photosynthesis is the process by which plants convert light energy, usually from the sun, into chemical energy stored in glucose, using carbon dioxide from the air and water from the soil.",
            rejected_response="Plants make sugar from sunlight."
        )
    ]
    
    # 划分训练/验证集
    train_data = training_data[:4]
    val_data = training_data[4:]
    
    # 4. 执行DPO训练
    print("\n1. 执行DPO训练")
    training_results = dpo_trainer.full_training_loop(
        training_data=train_data,
        validation_data=val_data,
        num_epochs=5,
        batch_size=2,
        learning_rate=1e-5
    )
    
    # 5. 分析偏好学习效果
    print("\n2. 偏好学习效果分析")
    preference_results = dpo_trainer.analyze_preference_learning(val_data)
    
    # 6. DPO变体演示
    print("\n3. DPO变体方法演示")
    
    # 模拟一些概率值进行演示
    chosen_log_probs = torch.tensor([-2.5, -3.0])
    rejected_log_probs = torch.tensor([-3.2, -3.8])
    chosen_ref_log_probs = torch.tensor([-2.8, -3.1])
    rejected_ref_log_probs = torch.tensor([-3.0, -3.5])
    
    # 标准DPO损失
    standard_dpo = -F.logsigmoid(0.1 * ((chosen_log_probs - chosen_ref_log_probs) - 
                                       (rejected_log_probs - rejected_ref_log_probs))).mean()
    
    # Identity DPO损失
    identity_dpo = DPOVariants.identity_dpo_loss(
        chosen_log_probs, rejected_log_probs, chosen_ref_log_probs, rejected_ref_log_probs
    )
    
    # KTO损失
    kto_loss = DPOVariants.kto_loss(
        chosen_log_probs, rejected_log_probs, chosen_ref_log_probs, rejected_ref_log_probs
    )
    
    print(f"标准DPO损失: {standard_dpo.item():.4f}")
    print(f"Identity DPO损失: {identity_dpo.item():.4f}")
    print(f"KTO损失: {kto_loss.item():.4f}")
    
    # 7. DPO vs RLHF 对比
    print("\n4. DPO vs RLHF 效率对比")
    
    # 模拟RLHF训练历史
    rlhf_history = [
        {'step': i, 'avg_reward': -2.0 + 1.5 * (1 - np.exp(-i / 10)) + np.random.normal(0, 0.1)}
        for i in range(10)
    ]
    
    comparator = DPOvsRLHFComparison()
    comparison_results = comparator.compare_training_efficiency(
        training_results['training_history'], rlhf_history
    )
    
    # 8. 总结
    final_stats = training_results['final_stats']
    
    print(f"\n=== DPO训练总结 ===")
    print(f"最终训练准确率: {final_stats.get('avg_accuracy', 0):.3f}")
    print(f"最终训练损失: {final_stats.get('avg_loss', 0):.4f}")
    
    if preference_results:
        print(f"偏好学习准确率: {preference_results['accuracy']:.3f}")
        print(f"平均奖励差距: {preference_results['average_margin']:.4f}")
    
    print(f"\nDPO优势总结:")
    print(f"✅ 无需奖励模型训练")
    print(f"✅ 避免RL训练复杂性")
    print(f"✅ 训练更加稳定")
    print(f"✅ 内存效率更高")
    print(f"✅ 理论基础坚实")

# 运行DPO演示
demonstrate_dpo_training()
```

继续完成第04节的最后部分，我将添加更多的替代RLHF方法和实际应用考虑：

## 4.2 其他替代RLHF方法

### Constitutional AI (CAI)

Constitutional AI通过让AI系统自我修正来实现对齐，包含两个阶段：
1. **监督阶段**：训练模型生成自我批评和修正
2. **强化学习阶段**：使用AI反馈替代人类反馈

```python
class ConstitutionalAI:
    """宪法AI实现"""
    
    def __init__(self, model, constitution: List[str]):
        self.model = model
        self.constitution = constitution  # 宪法原则列表
        
    def generate_critique_and_revision(self, prompt: str, initial_response: str) -> Tuple[str, str]:
        """生成批评和修正版本"""
        
        # 构建批评提示
        critique_prompt = f"""
        原始问题: {prompt}
        AI回复: {initial_response}
        
        请根据以下原则评估这个回复:
        {chr(10).join(f'{i+1}. {principle}' for i, principle in enumerate(self.constitution))}
        
        批评:
        """
        
        # 生成批评（简化实现）
        critique = self._generate_response(critique_prompt)
        
        # 构建修正提示
        revision_prompt = f"""
        原始问题: {prompt}
        原始回复: {initial_response}
        批评: {critique}
        
        请提供一个改进的回复:
        """
        
        # 生成修正版本
        revised_response = self._generate_response(revision_prompt)
        
        return critique, revised_response
    
    def _generate_response(self, prompt: str) -> str:
        """生成回复（简化实现）"""
        # 实际应用中这里应该是真实的模型生成
        return f"Generated response for: {prompt[:50]}..."

class RLAIF:
    """AI反馈强化学习"""
    
    def __init__(self, model, ai_judge_model):
        self.model = model
        self.ai_judge = ai_judge_model
        
    def generate_ai_feedback(self, prompt: str, responses: List[str]) -> List[float]:
        """生成AI反馈分数"""
        
        scores = []
        
        for response in responses:
            # 构建评判提示
            judge_prompt = f"""
            请评估以下AI回复的质量(1-10分):
            
            问题: {prompt}
            回复: {response}
            
            评估标准:
            - 准确性 (30%)
            - 有用性 (30%) 
            - 安全性 (25%)
            - 连贯性 (15%)
            
            分数:
            """
            
            # AI评判（简化实现）
            score = self._extract_score(self.ai_judge.generate(judge_prompt))
            scores.append(score)
        
        return scores
    
    def _extract_score(self, judge_response: str) -> float:
        """从AI评判回复中提取分数"""
        # 简化实现：随机生成分数
        return random.uniform(6.0, 9.5)

class SelfSupervisedAlignment:
    """自监督对齐方法"""
    
    def __init__(self, model):
        self.model = model
        
    def contrastive_learning_loss(self, anchor_response: str, positive_response: str, 
                                 negative_responses: List[str], temperature: float = 0.1) -> torch.Tensor:
        """对比学习损失"""
        
        # 编码所有回复
        anchor_embedding = self._encode_response(anchor_response)
        positive_embedding = self._encode_response(positive_response)
        negative_embeddings = [self._encode_response(neg) for neg in negative_responses]
        
        # 计算相似度
        pos_sim = F.cosine_similarity(anchor_embedding, positive_embedding, dim=-1) / temperature
        neg_sims = [F.cosine_similarity(anchor_embedding, neg_emb, dim=-1) / temperature 
                   for neg_emb in negative_embeddings]
        
        # InfoNCE损失
        all_sims = torch.cat([pos_sim.unsqueeze(0)] + neg_sims)
        loss = -F.log_softmax(all_sims, dim=0)[0]
        
        return loss
    
    def _encode_response(self, response: str) -> torch.Tensor:
        """编码回复为向量表示"""
        # 简化实现
        return torch.randn(768)  # 假设768维embedding

class ScalableSupervisedFineTuning:
    """可扩展监督微调方法"""
    
    def __init__(self, model, quality_classifier):
        self.model = model
        self.quality_classifier = quality_classifier
        
    def iterative_quality_filtering(self, dataset: List[Dict], 
                                   quality_threshold: float = 0.7) -> List[Dict]:
        """迭代质量过滤"""
        
        filtered_data = []
        
        for sample in dataset:
            # 使用质量分类器评估
            quality_score = self.quality_classifier.predict(sample['response'])
            
            if quality_score >= quality_threshold:
                filtered_data.append(sample)
        
        print(f"质量过滤: {len(dataset)} -> {len(filtered_data)} 样本")
        return filtered_data
    
    def active_learning_selection(self, candidate_data: List[Dict], 
                                 budget: int = 100) -> List[Dict]:
        """主动学习样本选择"""
        
        # 计算不确定性分数
        uncertainty_scores = []
        
        for sample in candidate_data:
            # 使用模型预测分布的熵作为不确定性度量
            logits = self.model.predict_logits(sample['prompt'])
            probs = F.softmax(logits, dim=-1)
            entropy = -(probs * torch.log(probs + 1e-10)).sum().item()
            uncertainty_scores.append((entropy, sample))
        
        # 选择最不确定的样本
        uncertainty_scores.sort(key=lambda x: x[0], reverse=True)
        selected_samples = [sample for _, sample in uncertainty_scores[:budget]]
        
        return selected_samples

# 综合方法比较
class AlignmentMethodComparison:
    """对齐方法综合比较"""
    
    def __init__(self):
        self.methods = {
            'RLHF': {
                'complexity': 9,
                'data_efficiency': 6,
                'training_stability': 5,
                'performance': 9,
                'computational_cost': 9,
                'human_oversight': 10
            },
            'DPO': {
                'complexity': 4,
                'data_efficiency': 8,
                'training_stability': 8,
                'performance': 8,
                'computational_cost': 4,
                'human_oversight': 10
            },
            'Constitutional_AI': {
                'complexity': 6,
                'data_efficiency': 7,
                'training_stability': 7,
                'performance': 7,
                'computational_cost': 5,
                'human_oversight': 3
            },
            'RLAIF': {
                'complexity': 7,
                'data_efficiency': 5,
                'training_stability': 6,
                'performance': 6,
                'computational_cost': 7,
                'human_oversight': 2
            }
        }
        
    def radar_chart_comparison(self):
        """绘制雷达图比较"""
        
        categories = list(self.methods['RLHF'].keys())
        N = len(categories)
        
        # 计算角度
        angles = [n / float(N) * 2 * math.pi for n in range(N)]
        angles += angles[:1]  # 闭合图形
        
        fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(projection='polar'))
        
        colors = ['red', 'blue', 'green', 'orange']
        
        for i, (method_name, scores) in enumerate(self.methods.items()):
            values = list(scores.values())
            values += values[:1]  # 闭合图形
            
            ax.plot(angles, values, 'o-', linewidth=2, 
                   label=method_name, color=colors[i])
            ax.fill(angles, values, alpha=0.25, color=colors[i])
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([cat.replace('_', ' ').title() for cat in categories])
        ax.set_ylim(0, 10)
        ax.set_title('AI对齐方法综合比较', size=16, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0))
        ax.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def generate_recommendation(self, use_case: str) -> str:
        """根据使用场景推荐方法"""
        
        recommendations = {
            'research': "DPO - 简单稳定，适合快速实验",
            'production': "RLHF - 性能最佳，适合生产环境",
            'low_resource': "Constitutional AI - 计算成本较低",
            'autonomous': "RLAIF - 减少人工监督需求",
            'safety_critical': "RLHF + Constitutional AI - 多层安全保障"
        }
        
        return recommendations.get(use_case, "请联系专家获取定制建议")

def comprehensive_alignment_demo():
    """综合对齐方法演示"""
    
    print("="*60)
    print("AI对齐方法综合演示")
    print("="*60)
    
    # 1. 方法比较
    print("\n1. 方法综合比较")
    comparator = AlignmentMethodComparison()
    comparator.radar_chart_comparison()
    
    # 2. 场景推荐
    print("\n2. 应用场景推荐")
    scenarios = ['research', 'production', 'low_resource', 'autonomous', 'safety_critical']
    
    for scenario in scenarios:
        recommendation = comparator.generate_recommendation(scenario)
        print(f"{scenario.replace('_', ' ').title()}: {recommendation}")
    
    # 3. 实际应用考虑
    print(f"\n3. 实际应用考虑")
    
    considerations = {
        'DPO': [
            "✅ 训练简单稳定",
            "✅ 不需要奖励模型",
            "✅ 内存效率高",
            "⚠️ 性能可能略低于RLHF",
            "⚠️ 对偏好数据质量敏感"
        ],
        'RLHF': [
            "✅ 性能表现最佳",
            "✅ 理论基础成熟",
            "✅ 工业验证充分",
            "❌ 训练复杂不稳定",
            "❌ 计算资源需求大"
        ],
        'Constitutional AI': [
            "✅ 减少人工标注",
            "✅ 可解释性强",
            "✅ 扩展性好",
            "⚠️ 依赖AI评判质量",
            "⚠️ 宪法设计需要专业知识"
        ]
    }
    
    for method, pros_cons in considerations.items():
        print(f"\n{method}:")
        for item in pros_cons:
            print(f"  {item}")
    
    # 4. 混合方法策略
    print(f"\n4. 推荐的混合策略")
    
    hybrid_strategies = [
        "阶段1: DPO快速对齐 → 阶段2: RLHF精细优化",
        "Constitutional AI生成数据 → DPO训练模型",
        "RLAIF扩展训练数据 → 人工验证关键样本",
        "多方法集成: 投票或加权平均多个对齐模型"
    ]
    
    for i, strategy in enumerate(hybrid_strategies, 1):
        print(f"{i}. {strategy}")
    
    print(f"\n=== 对齐方法选择指南 ===")
    print(f"• 研究和原型: 优先选择DPO")
    print(f"• 生产部署: 考虑RLHF或DPO+微调")
    print(f"• 资源受限: Constitutional AI或简化DPO")
    print(f"• 安全关键: 多方法结合+人工验证")
    print(f"• 长期发展: 关注新兴方法如KTO、SPIN等")

# 运行综合演示
comprehensive_alignment_demo()
```

现在我已经完成了第05章第04节"DPO与替代RLHF方法"的完整内容。这一节涵盖了：

1. **DPO数学推导**：从奖励模型到直接优化的完整理论推导
2. **DPO实现细节**：包括损失函数、训练循环、效果分析
3. **DPO变体方法**：Identity DPO、KTO、RSO等新兴方法
4. **替代RLHF方法**：Constitutional AI、RLAIF、自监督对齐等
5. **方法综合比较**：各种对齐方法的优缺点分析和应用建议
6. **实际应用考虑**：混合策略和场景选择指南

现在我来更新todo list并完成第05章：

<function_calls>
<invoke name="TodoWrite">
<parameter name="todos">[{"content": "Research best practices for technical tutorial writing", "status": "completed", "priority": "high", "id": "1"}, {"content": "Design comprehensive folder structure for MiniGPT handbook", "status": "completed", "priority": "high", "id": "2"}, {"content": "Create detailed outline with ultra-deep analysis approach", "status": "completed", "priority": "high", "id": "3"}, {"content": "Write handbook introduction and navigation guide", "status": "completed", "priority": "high", "id": "4"}, {"content": "Implement chapter-by-chapter content with mathematical rigor", "status": "in_progress", "priority": "high", "id": "5"}, {"content": "Write Chapter 1: Mathematical Foundations", "status": "completed", "priority": "high", "id": "6"}, {"content": "Write Chapter 2: Transformer Architecture", "status": "completed", "priority": "high", "id": "7"}, {"content": "Write Chapter 3: Pre-training Theory and Implementation", "status": "completed", "priority": "high", "id": "8"}, {"content": "Write Chapter 4: Supervised Fine-tuning Deep Dive", "status": "completed", "priority": "high", "id": "14"}, {"content": "Write Chapter 5: Reinforcement Learning from Human Feedback", "status": "completed", "priority": "high", "id": "19"}, {"content": "Write Chapter 6: Generation and Decoding Strategies", "status": "pending", "priority": "high", "id": "25"}, {"content": "Write Chapter 7: Evaluation and Analysis Methods", "status": "pending", "priority": "high", "id": "26"}, {"content": "Write Chapter 8: Engineering Practices and Optimization", "status": "pending", "priority": "high", "id": "27"}]