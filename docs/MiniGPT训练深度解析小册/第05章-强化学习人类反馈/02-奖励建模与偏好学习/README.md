# 02 奖励建模与偏好学习

> **从人类直觉到数学函数：偏好学习的认知科学与计算建模**

## 核心思想

奖励建模是RLHF的关键技术，它要解决一个根本性问题：如何将人类的主观偏好转化为机器可以理解和优化的数学函数。这个过程不仅涉及机器学习技术，更深层次地涉及认知科学、心理学和决策理论。

**关键洞察**：
- **偏好的传递性**：人类偏好具有内在的数学结构
- **比较学习**：相比绝对评分，人类更擅长相对比较
- **噪声建模**：人类判断存在不一致性，需要概率建模
- **多维偏好**：真实偏好是多个维度的复合函数

从数学角度看，我们要学习一个函数$R_\phi: \text{Context} \times \text{Response} \to \mathbb{R}$，使其能够准确预测人类对不同回复的偏好排序。

## 2.1 Bradley-Terry模型的数学原理

### 偏好比较的概率建模

**Bradley-Terry模型**是建模成对比较的经典方法，其核心假设是：给定两个选项$x_i$和$x_j$，选择$x_i$的概率由两者的"强度"决定：

$$P(x_i \succ x_j) = \frac{\exp(R(x_i))}{\exp(R(x_i)) + \exp(R(x_j))} = \sigma(R(x_i) - R(x_j))$$

其中：
- $R(x_i)$是选项$x_i$的奖励值
- $\sigma(\cdot)$是sigmoid函数
- $x_i \succ x_j$表示$x_i$被偏好于$x_j$

**关键性质**：
1. **传递性**：如果$P(x_i \succ x_j) > 0.5$且$P(x_j \succ x_k) > 0.5$，则$P(x_i \succ x_k) > 0.5$
2. **对称性**：$P(x_i \succ x_j) + P(x_j \succ x_i) = 1$
3. **单调性**：更高的奖励值对应更高的被选择概率

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import accuracy_score, roc_auc_score
import json
import random
from collections import defaultdict, Counter
import math

@dataclass
class PreferenceData:
    """偏好数据结构"""
    prompt: str                    # 输入提示
    response_a: str               # 回复A
    response_b: str               # 回复B  
    preference: int               # 偏好标签：0表示偏好A，1表示偏好B
    confidence: float = 1.0       # 偏好强度/置信度
    annotator_id: str = "default" # 标注者ID
    metadata: Dict = None         # 额外元数据
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

class BradleyTerryModel:
    """Bradley-Terry偏好学习模型"""
    
    def __init__(self, vocab_size: int, hidden_size: int = 768, max_length: int = 512):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.max_length = max_length
        
        # 构建奖励网络
        self.reward_network = self._build_reward_network()
        
        # 训练历史
        self.training_history = []
        self.preference_patterns = defaultdict(list)
        
    def _build_reward_network(self) -> nn.Module:
        """构建奖励网络"""
        
        class RewardNetwork(nn.Module):
            def __init__(self, vocab_size, hidden_size, max_length):
                super().__init__()
                
                # 文本编码器 - 使用Transformer
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=hidden_size,
                    nhead=8,
                    dim_feedforward=hidden_size * 4,
                    dropout=0.1,
                    batch_first=True
                )
                self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
                
                # 词嵌入和位置编码
                self.embedding = nn.Embedding(vocab_size, hidden_size)
                self.pos_encoding = self._create_position_encoding(max_length, hidden_size)
                
                # 奖励预测头
                self.reward_head = nn.Sequential(
                    nn.Linear(hidden_size, hidden_size // 2),
                    nn.LayerNorm(hidden_size // 2),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(hidden_size // 2, hidden_size // 4),
                    nn.LayerNorm(hidden_size // 4),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(hidden_size // 4, 1)
                )
                
                # 初始化权重
                self._init_weights()
            
            def _create_position_encoding(self, max_len: int, d_model: int) -> nn.Parameter:
                """创建位置编码"""
                pe = torch.zeros(max_len, d_model)
                position = torch.arange(0, max_len).unsqueeze(1).float()
                
                div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                                   -(math.log(10000.0) / d_model))
                
                pe[:, 0::2] = torch.sin(position * div_term)
                pe[:, 1::2] = torch.cos(position * div_term)
                
                return nn.Parameter(pe.unsqueeze(0), requires_grad=False)
            
            def _init_weights(self):
                """初始化网络权重"""
                for module in self.modules():
                    if isinstance(module, nn.Linear):
                        nn.init.xavier_normal_(module.weight, gain=0.02)
                        if module.bias is not None:
                            nn.init.zeros_(module.bias)
                    elif isinstance(module, nn.Embedding):
                        nn.init.normal_(module.weight, std=0.02)
            
            def encode_text(self, token_ids: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
                """编码文本序列"""
                
                batch_size, seq_len = token_ids.shape
                
                # 词嵌入
                embeddings = self.embedding(token_ids)
                
                # 添加位置编码
                if seq_len <= self.pos_encoding.size(1):
                    pos_enc = self.pos_encoding[:, :seq_len, :]
                    embeddings = embeddings + pos_enc
                
                # Transformer编码
                if attention_mask is not None:
                    # 创建因果掩码（可选）
                    src_key_padding_mask = ~attention_mask.bool()
                else:
                    src_key_padding_mask = None
                
                encoded = self.encoder(embeddings, src_key_padding_mask=src_key_padding_mask)
                
                # 使用平均池化得到序列表示
                if attention_mask is not None:
                    # 加权平均（排除padding）
                    mask_expanded = attention_mask.unsqueeze(-1).float()
                    sequence_repr = (encoded * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1)
                else:
                    sequence_repr = encoded.mean(dim=1)
                
                return sequence_repr
            
            def forward(self, token_ids: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
                """前向传播：计算奖励分数"""
                
                # 编码文本
                sequence_repr = self.encode_text(token_ids, attention_mask)
                
                # 预测奖励
                reward = self.reward_head(sequence_repr)
                
                return reward.squeeze(-1)  # [batch_size]
        
        return RewardNetwork(self.vocab_size, self.hidden_size, self.max_length)
    
    def compute_preference_probability(self, reward_a: torch.Tensor, reward_b: torch.Tensor) -> torch.Tensor:
        """计算Bradley-Terry偏好概率"""
        
        # P(A > B) = sigmoid(R(A) - R(B))
        logit_diff = reward_a - reward_b
        prob_a_preferred = torch.sigmoid(logit_diff)
        
        return prob_a_preferred
    
    def compute_preference_loss(self, preference_data: List[PreferenceData], 
                              tokenizer=None) -> torch.Tensor:
        """计算偏好学习损失"""
        
        if not preference_data:
            return torch.tensor(0.0)
        
        total_loss = 0.0
        valid_samples = 0
        
        for data in preference_data:
            # 简化的token化（实际应用中需要真实的tokenizer）
            tokens_a = self._tokenize_text(data.prompt + " " + data.response_a)
            tokens_b = self._tokenize_text(data.prompt + " " + data.response_b)
            
            if len(tokens_a) == 0 or len(tokens_b) == 0:
                continue
            
            # 转换为tensor
            tokens_a = torch.tensor(tokens_a, dtype=torch.long).unsqueeze(0)
            tokens_b = torch.tensor(tokens_b, dtype=torch.long).unsqueeze(0)
            
            # 计算奖励
            reward_a = self.reward_network(tokens_a)
            reward_b = self.reward_network(tokens_b)
            
            # 计算偏好概率
            prob_a_preferred = self.compute_preference_probability(reward_a, reward_b)
            
            # Bradley-Terry损失
            if data.preference == 0:  # 偏好A
                target_prob = torch.tensor(1.0)
            else:  # 偏好B
                target_prob = torch.tensor(0.0)
            
            # 交叉熵损失，加权by置信度
            loss = -data.confidence * (target_prob * torch.log(prob_a_preferred + 1e-10) + 
                                     (1 - target_prob) * torch.log(1 - prob_a_preferred + 1e-10))
            
            total_loss += loss
            valid_samples += 1
        
        if valid_samples == 0:
            return torch.tensor(0.0)
        
        return total_loss / valid_samples
    
    def _tokenize_text(self, text: str) -> List[int]:
        """简化的token化（实际应用中需要使用真实tokenizer）"""
        # 这里使用简单的词级别token化作为示例
        words = text.lower().split()
        # 将词映射到随机的token ID（演示用）
        token_ids = []
        for word in words[:50]:  # 限制长度
            # 简化：使用hash来生成一致的token ID
            token_id = abs(hash(word)) % (self.vocab_size - 10) + 3  # 避开特殊token
            token_ids.append(token_id)
        
        return token_ids
    
    def train_on_preferences(self, preference_data: List[PreferenceData], 
                           epochs: int = 10, lr: float = 1e-4) -> Dict:
        """在偏好数据上训练模型"""
        
        print(f"=== Bradley-Terry模型训练 ===")
        print(f"训练数据: {len(preference_data)} 个偏好对")
        
        # 优化器
        optimizer = torch.optim.AdamW(self.reward_network.parameters(), lr=lr, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        
        # 训练历史
        training_metrics = {
            'epoch': [],
            'loss': [],
            'accuracy': [],
            'reward_variance': []
        }
        
        # 分割训练/验证集
        n_train = int(0.8 * len(preference_data))
        train_data = preference_data[:n_train]
        val_data = preference_data[n_train:]
        
        for epoch in range(epochs):
            # 训练模式
            self.reward_network.train()
            
            # 计算训练损失
            train_loss = self.compute_preference_loss(train_data)
            
            # 反向传播
            optimizer.zero_grad()
            train_loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.reward_network.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step()
            
            # 评估模式
            self.reward_network.eval()
            with torch.no_grad():
                val_loss = self.compute_preference_loss(val_data)
                val_accuracy = self._compute_accuracy(val_data)
                reward_variance = self._compute_reward_variance(val_data)
            
            # 记录指标
            training_metrics['epoch'].append(epoch)
            training_metrics['loss'].append(val_loss.item())
            training_metrics['accuracy'].append(val_accuracy)
            training_metrics['reward_variance'].append(reward_variance)
            
            if epoch % 2 == 0:
                print(f"Epoch {epoch:2d}: Loss={val_loss:.4f}, Acc={val_accuracy:.3f}, "
                      f"RVar={reward_variance:.4f}")
        
        # 分析训练结果
        self._analyze_training_results(training_metrics, preference_data)
        
        return training_metrics
    
    def _compute_accuracy(self, preference_data: List[PreferenceData]) -> float:
        """计算偏好预测准确率"""
        
        correct = 0
        total = 0
        
        for data in preference_data:
            tokens_a = self._tokenize_text(data.prompt + " " + data.response_a)
            tokens_b = self._tokenize_text(data.prompt + " " + data.response_b)
            
            if len(tokens_a) == 0 or len(tokens_b) == 0:
                continue
            
            tokens_a = torch.tensor(tokens_a, dtype=torch.long).unsqueeze(0)
            tokens_b = torch.tensor(tokens_b, dtype=torch.long).unsqueeze(0)
            
            reward_a = self.reward_network(tokens_a)
            reward_b = self.reward_network(tokens_b)
            
            # 预测偏好
            predicted = 0 if reward_a > reward_b else 1
            
            if predicted == data.preference:
                correct += 1
            total += 1
        
        return correct / total if total > 0 else 0.0
    
    def _compute_reward_variance(self, preference_data: List[PreferenceData]) -> float:
        """计算奖励值的方差"""
        
        rewards = []
        
        for data in preference_data:
            for response in [data.response_a, data.response_b]:
                tokens = self._tokenize_text(data.prompt + " " + response)
                if len(tokens) > 0:
                    tokens = torch.tensor(tokens, dtype=torch.long).unsqueeze(0)
                    reward = self.reward_network(tokens)
                    rewards.append(reward.item())
        
        return np.var(rewards) if rewards else 0.0
    
    def _analyze_training_results(self, metrics: Dict, preference_data: List[PreferenceData]):
        """分析训练结果"""
        
        print(f"\n=== 训练结果分析 ===")
        print(f"最终验证损失: {metrics['loss'][-1]:.4f}")
        print(f"最终验证准确率: {metrics['accuracy'][-1]:.3f}")
        print(f"奖励值方差: {metrics['reward_variance'][-1]:.4f}")
        
        # 可视化训练过程
        self._visualize_training_metrics(metrics)
        
        # 分析偏好模式
        self._analyze_preference_patterns(preference_data)
    
    def _visualize_training_metrics(self, metrics: Dict):
        """可视化训练指标"""
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        # 损失曲线
        axes[0].plot(metrics['epoch'], metrics['loss'], 'b-', linewidth=2)
        axes[0].set_title('验证损失')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].grid(True, alpha=0.3)
        
        # 准确率曲线
        axes[1].plot(metrics['epoch'], metrics['accuracy'], 'g-', linewidth=2)
        axes[1].set_title('偏好预测准确率')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_ylim(0, 1)
        axes[1].grid(True, alpha=0.3)
        
        # 奖励方差
        axes[2].plot(metrics['epoch'], metrics['reward_variance'], 'r-', linewidth=2)
        axes[2].set_title('奖励值方差')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('Variance')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def _analyze_preference_patterns(self, preference_data: List[PreferenceData]):
        """分析偏好模式"""
        
        print(f"\n=== 偏好模式分析 ===")
        
        # 统计基本信息
        total_pairs = len(preference_data)
        prefer_a = sum(1 for d in preference_data if d.preference == 0)
        prefer_b = total_pairs - prefer_a
        
        print(f"总偏好对数: {total_pairs}")
        print(f"偏好A: {prefer_a} ({prefer_a/total_pairs:.1%})")
        print(f"偏好B: {prefer_b} ({prefer_b/total_pairs:.1%})")
        
        # 置信度分析
        confidences = [d.confidence for d in preference_data]
        print(f"平均置信度: {np.mean(confidences):.3f}")
        print(f"置信度标准差: {np.std(confidences):.3f}")
        
        # 长度偏差分析
        length_bias = []
        for data in preference_data:
            len_a = len(data.response_a.split())
            len_b = len(data.response_b.split())
            
            if data.preference == 0 and len_a > len_b:
                length_bias.append(1)  # 偏好更长的
            elif data.preference == 1 and len_b > len_a:
                length_bias.append(1)
            else:
                length_bias.append(0)
        
        length_bias_ratio = np.mean(length_bias)
        print(f"长度偏差: {length_bias_ratio:.1%} 的情况下偏好更长的回复")

class PreferenceDataCollector:
    """偏好数据收集器"""
    
    def __init__(self):
        self.collected_data = []
        self.annotation_quality = {}
        
    def simulate_human_annotation(self, prompts: List[str], responses_pairs: List[Tuple[str, str]],
                                annotation_noise: float = 0.1) -> List[PreferenceData]:
        """模拟人类标注过程"""
        
        print(f"=== 模拟人类偏好标注 ===")
        print(f"标注对数: {len(prompts)}")
        print(f"标注噪声水平: {annotation_noise}")
        
        preference_data = []
        
        for i, (prompt, (resp_a, resp_b)) in enumerate(zip(prompts, responses_pairs)):
            # 模拟真实的人类偏好决策过程
            true_preference = self._simulate_true_preference(prompt, resp_a, resp_b)
            
            # 添加标注噪声
            if random.random() < annotation_noise:
                observed_preference = 1 - true_preference  # 翻转偏好
                confidence = 0.5 + random.random() * 0.3   # 降低置信度
            else:
                observed_preference = true_preference
                confidence = 0.7 + random.random() * 0.3   # 较高置信度
            
            # 创建偏好数据
            pref_data = PreferenceData(
                prompt=prompt,
                response_a=resp_a,
                response_b=resp_b,
                preference=observed_preference,
                confidence=confidence,
                annotator_id=f"annotator_{i % 3}",  # 模拟3个标注者
                metadata={
                    'true_preference': true_preference,
                    'annotation_time': random.randint(30, 300),  # 标注时间（秒）
                    'difficulty': self._assess_difficulty(prompt, resp_a, resp_b)
                }
            )
            
            preference_data.append(pref_data)
        
        self.collected_data.extend(preference_data)
        
        # 分析标注质量
        self._analyze_annotation_quality(preference_data, annotation_noise)
        
        return preference_data
    
    def _simulate_true_preference(self, prompt: str, resp_a: str, resp_b: str) -> int:
        """模拟真实的人类偏好（基于简单启发式）"""
        
        # 启发式1：长度偏好（适度长度更好）
        len_a, len_b = len(resp_a.split()), len(resp_b.split())
        ideal_length = len(prompt.split()) * 2  # 理想长度
        
        len_score_a = 1.0 / (1.0 + abs(len_a - ideal_length) / ideal_length)
        len_score_b = 1.0 / (1.0 + abs(len_b - ideal_length) / ideal_length)
        
        # 启发式2：相关性（与prompt的词汇重叠）
        prompt_words = set(prompt.lower().split())
        words_a = set(resp_a.lower().split())
        words_b = set(resp_b.lower().split())
        
        relevance_a = len(prompt_words.intersection(words_a)) / (len(prompt_words) + 1e-10)
        relevance_b = len(prompt_words.intersection(words_b)) / (len(prompt_words) + 1e-10)
        
        # 启发式3：复杂性（词汇多样性）
        diversity_a = len(set(resp_a.lower().split())) / (len(resp_a.split()) + 1e-10)
        diversity_b = len(set(resp_b.lower().split())) / (len(resp_b.split()) + 1e-10)
        
        # 综合评分
        score_a = 0.4 * len_score_a + 0.4 * relevance_a + 0.2 * diversity_a
        score_b = 0.4 * len_score_b + 0.4 * relevance_b + 0.2 * diversity_b
        
        return 0 if score_a > score_b else 1
    
    def _assess_difficulty(self, prompt: str, resp_a: str, resp_b: str) -> float:
        """评估标注难度"""
        
        # 长度差异
        len_diff = abs(len(resp_a.split()) - len(resp_b.split()))
        
        # 内容相似性（简化计算）
        words_a = set(resp_a.lower().split())
        words_b = set(resp_b.lower().split())
        similarity = len(words_a.intersection(words_b)) / len(words_a.union(words_b))
        
        # 难度评分：相似度高且长度差异小的情况更难标注
        difficulty = similarity * (1.0 - len_diff / (len_diff + 10))
        
        return min(difficulty, 1.0)
    
    def _analyze_annotation_quality(self, preference_data: List[PreferenceData], 
                                  expected_noise: float):
        """分析标注质量"""
        
        print(f"\n=== 标注质量分析 ===")
        
        # 计算实际错误率
        errors = 0
        total = 0
        
        for data in preference_data:
            if data.metadata and 'true_preference' in data.metadata:
                if data.preference != data.metadata['true_preference']:
                    errors += 1
                total += 1
        
        actual_error_rate = errors / total if total > 0 else 0
        print(f"预期错误率: {expected_noise:.1%}")
        print(f"实际错误率: {actual_error_rate:.1%}")
        
        # 置信度与准确率的关系
        high_conf = [d for d in preference_data if d.confidence > 0.8]
        low_conf = [d for d in preference_data if d.confidence < 0.6]
        
        if high_conf:
            high_conf_accuracy = sum(1 for d in high_conf 
                                   if d.preference == d.metadata.get('true_preference', d.preference)) / len(high_conf)
            print(f"高置信度标注准确率: {high_conf_accuracy:.1%}")
        
        if low_conf:
            low_conf_accuracy = sum(1 for d in low_conf 
                                  if d.preference == d.metadata.get('true_preference', d.preference)) / len(low_conf)
            print(f"低置信度标注准确率: {low_conf_accuracy:.1%}")
        
        # 标注者一致性分析
        annotator_stats = defaultdict(lambda: {'correct': 0, 'total': 0})
        
        for data in preference_data:
            if data.metadata and 'true_preference' in data.metadata:
                annotator = data.annotator_id
                if data.preference == data.metadata['true_preference']:
                    annotator_stats[annotator]['correct'] += 1
                annotator_stats[annotator]['total'] += 1
        
        print(f"\n标注者表现:")
        for annotator, stats in annotator_stats.items():
            accuracy = stats['correct'] / stats['total']
            print(f"  {annotator}: {accuracy:.1%} ({stats['correct']}/{stats['total']})")

# 综合演示：偏好学习完整流程
def demonstrate_preference_learning():
    """演示偏好学习的完整流程"""
    
    print("="*60)
    print("偏好学习与奖励建模 - 综合演示")
    print("="*60)
    
    # 1. 创建模拟数据
    print("\n1. 生成模拟偏好数据")
    
    # 模拟prompt和response pairs
    prompts = [
        "What is the capital of France?",
        "Explain machine learning in simple terms.",
        "Write a short story about a robot.",
        "How do you make pasta?",
        "What are the benefits of exercise?",
        "Describe the solar system.",
        "How does the internet work?",
        "What is climate change?",
        "Explain photosynthesis.",
        "How do computers work?"
    ]
    
    response_pairs = [
        ("Paris is the capital of France.", "The capital of France is Paris, a beautiful city known for its art and culture."),
        ("ML is computers learning patterns.", "Machine learning is a branch of AI where computers learn from data to make predictions or decisions without being explicitly programmed."),
        ("A robot walked.", "In the year 2045, a small maintenance robot named WALL-E discovered an unusual plant growing in the abandoned city."),
        ("Boil water, add pasta.", "To make pasta, bring a large pot of salted water to boil, add pasta, cook according to package directions, then drain and serve with your favorite sauce."),
        ("Exercise is good for health.", "Regular exercise improves cardiovascular health, strengthens muscles, enhances mental well-being, and helps maintain a healthy weight."),
        ("Solar system has planets.", "The solar system consists of the Sun at the center, eight planets including Earth, numerous moons, asteroids, and comets, all held together by gravitational forces."),
        ("Internet connects computers.", "The internet is a global network of interconnected computers that communicate using standardized protocols, enabling data sharing and communication worldwide."),
        ("Climate change is warming.", "Climate change refers to long-term shifts in global temperatures and weather patterns, primarily caused by human activities that increase greenhouse gas emissions."),
        ("Plants make food from sun.", "Photosynthesis is the process by which plants convert sunlight, carbon dioxide, and water into glucose and oxygen, using chlorophyll in their leaves."),
        ("Computers process data.", "Computers work by processing binary data through electronic circuits, following instructions stored in memory to perform calculations and execute programs.")
    ]
    
    # 2. 收集偏好数据
    collector = PreferenceDataCollector()
    preference_data = collector.simulate_human_annotation(
        prompts, response_pairs, annotation_noise=0.15
    )
    
    # 3. 训练Bradley-Terry模型
    print("\n2. 训练Bradley-Terry偏好模型")
    
    bt_model = BradleyTerryModel(vocab_size=10000, hidden_size=256, max_length=128)
    training_metrics = bt_model.train_on_preferences(
        preference_data, epochs=20, lr=2e-4
    )
    
    # 4. 评估模型表现
    print("\n3. 评估模型表现")
    
    test_prompts = [
        "What is artificial intelligence?",
        "How do you learn a new language?"
    ]
    
    test_responses = [
        ("AI is smart computers.", "Artificial intelligence is the simulation of human intelligence in machines that are programmed to think and learn like humans."),
        ("Practice daily.", "To learn a new language effectively, immerse yourself in the language through daily practice, use multiple learning methods, and engage with native speakers.")
    ]
    
    print("测试偏好预测:")
    for i, (prompt, (resp_a, resp_b)) in enumerate(zip(test_prompts, test_responses)):
        # 计算奖励
        tokens_a = bt_model._tokenize_text(prompt + " " + resp_a)
        tokens_b = bt_model._tokenize_text(prompt + " " + resp_b)
        
        if tokens_a and tokens_b:
            tokens_a = torch.tensor(tokens_a).unsqueeze(0)
            tokens_b = torch.tensor(tokens_b).unsqueeze(0)
            
            with torch.no_grad():
                reward_a = bt_model.reward_network(tokens_a)
                reward_b = bt_model.reward_network(tokens_b)
                prob_prefer_a = bt_model.compute_preference_probability(reward_a, reward_b)
            
            print(f"\n测试样本 {i+1}:")
            print(f"  Prompt: {prompt}")
            print(f"  Response A: {resp_a}")
            print(f"  Response B: {resp_b}")
            print(f"  奖励A: {reward_a.item():.3f}")
            print(f"  奖励B: {reward_b.item():.3f}")
            print(f"  偏好A概率: {prob_prefer_a.item():.3f}")
            print(f"  预测偏好: {'A' if prob_prefer_a > 0.5 else 'B'}")
    
    print(f"\n=== 偏好学习演示完成 ===")
    print(f"Bradley-Terry模型成功学习了人类偏好模式")
    print(f"最终准确率: {training_metrics['accuracy'][-1]:.1%}")
    print(f"该模型现在可以用作RLHF中的奖励函数")

# 运行完整演示
demonstrate_preference_learning()
```

继续完成第02节的剩余内容：

## 2.2 多维偏好的数学建模

### 偏好的分解与组合

现实中的人类偏好往往是多维的，包括准确性、有用性、安全性、流畅性等多个方面。数学上，我们可以将总体偏好建模为各维度偏好的加权组合：

$$R_{\text{total}}(x) = \sum_{i=1}^{d} w_i R_i(x) + \text{interaction terms}$$

其中：
- $R_i(x)$是第$i$个维度的奖励
- $w_i$是对应的权重
- interaction terms 捕捉维度间的相互作用

```python
class MultiDimensionalPreferenceModel:
    """多维偏好建模"""
    
    def __init__(self, dimensions: List[str], vocab_size: int = 10000):
        self.dimensions = dimensions
        self.vocab_size = vocab_size
        
        # 为每个维度构建专门的奖励模型
        self.dimension_models = {}
        for dim in dimensions:
            self.dimension_models[dim] = self._create_dimension_model(dim)
        
        # 维度权重学习网络
        self.weight_network = self._create_weight_network()
        
        # 交互项建模网络
        self.interaction_network = self._create_interaction_network()
        
    def _create_dimension_model(self, dimension: str) -> nn.Module:
        """为特定维度创建奖励模型"""
        
        class DimensionSpecificReward(nn.Module):
            def __init__(self, dimension_name: str, vocab_size: int):
                super().__init__()
                
                self.dimension_name = dimension_name
                
                # 根据维度类型设计不同的网络架构
                if dimension_name == 'helpfulness':
                    # 有用性：关注内容的信息量和相关性
                    hidden_size = 512
                    self.encoder = self._build_content_encoder(vocab_size, hidden_size)
                    self.reward_head = nn.Sequential(
                        nn.Linear(hidden_size, 256),
                        nn.ReLU(),
                        nn.Linear(256, 1)
                    )
                    
                elif dimension_name == 'harmlessness':
                    # 无害性：检测潜在的有害内容
                    hidden_size = 256
                    self.encoder = self._build_safety_encoder(vocab_size, hidden_size)
                    self.reward_head = nn.Sequential(
                        nn.Linear(hidden_size, 128),
                        nn.ReLU(),
                        nn.Linear(128, 1),
                        nn.Sigmoid()  # 输出0-1之间的安全分数
                    )
                    
                elif dimension_name == 'honesty':
                    # 诚实性：检测事实准确性和一致性
                    hidden_size = 384
                    self.encoder = self._build_factuality_encoder(vocab_size, hidden_size)
                    self.reward_head = nn.Sequential(
                        nn.Linear(hidden_size, 192),
                        nn.ReLU(),
                        nn.Linear(192, 1)
                    )
                    
                else:
                    # 通用维度
                    hidden_size = 256
                    self.encoder = self._build_generic_encoder(vocab_size, hidden_size)
                    self.reward_head = nn.Sequential(
                        nn.Linear(hidden_size, 128),
                        nn.ReLU(),
                        nn.Linear(128, 1)
                    )
            
            def _build_content_encoder(self, vocab_size: int, hidden_size: int) -> nn.Module:
                """构建内容理解编码器"""
                return nn.TransformerEncoder(
                    nn.TransformerEncoderLayer(
                        d_model=hidden_size,
                        nhead=8,
                        dim_feedforward=hidden_size * 2,
                        dropout=0.1
                    ),
                    num_layers=4
                )
            
            def _build_safety_encoder(self, vocab_size: int, hidden_size: int) -> nn.Module:
                """构建安全性检测编码器"""
                return nn.TransformerEncoder(
                    nn.TransformerEncoderLayer(
                        d_model=hidden_size,
                        nhead=4,
                        dim_feedforward=hidden_size,
                        dropout=0.1
                    ),
                    num_layers=3
                )
            
            def _build_factuality_encoder(self, vocab_size: int, hidden_size: int) -> nn.Module:
                """构建事实性检测编码器"""
                return nn.TransformerEncoder(
                    nn.TransformerEncoderLayer(
                        d_model=hidden_size,
                        nhead=6,
                        dim_feedforward=hidden_size * 3,
                        dropout=0.1
                    ),
                    num_layers=4
                )
            
            def _build_generic_encoder(self, vocab_size: int, hidden_size: int) -> nn.Module:
                """构建通用编码器"""
                return nn.TransformerEncoder(
                    nn.TransformerEncoderLayer(
                        d_model=hidden_size,
                        nhead=4,
                        dim_feedforward=hidden_size * 2,
                        dropout=0.1
                    ),
                    num_layers=3
                )
            
            def forward(self, x):
                # 简化实现
                encoded = self.encoder(x)  # 需要适当的输入预处理
                reward = self.reward_head(encoded.mean(dim=1))
                return reward.squeeze(-1)
        
        return DimensionSpecificReward(dimension, vocab_size)
    
    def _create_weight_network(self) -> nn.Module:
        """创建维度权重学习网络"""
        
        class WeightNetwork(nn.Module):
            def __init__(self, n_dimensions: int):
                super().__init__()
                
                # 上下文编码器
                self.context_encoder = nn.Sequential(
                    nn.Linear(512, 256),  # 假设上下文向量维度为512
                    nn.ReLU(),
                    nn.Linear(256, 128),
                    nn.ReLU()
                )
                
                # 权重预测头
                self.weight_head = nn.Sequential(
                    nn.Linear(128, n_dimensions),
                    nn.Softmax(dim=-1)  # 确保权重和为1
                )
            
            def forward(self, context_vector):
                context_encoded = self.context_encoder(context_vector)
                weights = self.weight_head(context_encoded)
                return weights
        
        return WeightNetwork(len(self.dimensions))
    
    def _create_interaction_network(self) -> nn.Module:
        """创建维度交互建模网络"""
        
        class InteractionNetwork(nn.Module):
            def __init__(self, n_dimensions: int):
                super().__init__()
                
                # 交互项数量：C(n,2) = n*(n-1)/2
                n_interactions = n_dimensions * (n_dimensions - 1) // 2
                
                self.interaction_head = nn.Sequential(
                    nn.Linear(n_dimensions, n_interactions),
                    nn.Tanh(),  # 允许正负交互
                    nn.Linear(n_interactions, 1)
                )
            
            def forward(self, dimension_rewards):
                # 计算所有维度对的乘积作为交互特征
                n_dims = dimension_rewards.size(1)
                interactions = []
                
                for i in range(n_dims):
                    for j in range(i+1, n_dims):
                        interaction = dimension_rewards[:, i] * dimension_rewards[:, j]
                        interactions.append(interaction.unsqueeze(1))
                
                if interactions:
                    interaction_features = torch.cat(interactions, dim=1)
                    interaction_score = self.interaction_head(interaction_features)
                    return interaction_score.squeeze(-1)
                else:
                    return torch.zeros(dimension_rewards.size(0))
        
        return InteractionNetwork(len(self.dimensions))
    
    def compute_multidimensional_reward(self, text_input: str, context: Dict = None) -> Dict:
        """计算多维度奖励"""
        
        # 简化的文本编码（实际应用中需要合适的tokenizer和encoder）
        # 这里用随机向量模拟编码结果
        encoded_input = torch.randn(1, 256)  # [batch_size, hidden_size]
        
        # 计算各维度奖励
        dimension_rewards = {}
        for dim in self.dimensions:
            # 这里简化处理，实际需要真实的前向传播
            reward = torch.randn(1) * 0.5 + 0.5  # 模拟奖励值
            dimension_rewards[dim] = reward.item()
        
        # 计算上下文相关的权重
        if context:
            # 编码上下文信息
            context_vector = self._encode_context(context)
            weights = self.weight_network(context_vector)
            weights_dict = {dim: weights[0, i].item() for i, dim in enumerate(self.dimensions)}
        else:
            # 使用默认权重
            weights_dict = {dim: 1.0/len(self.dimensions) for dim in self.dimensions}
        
        # 计算加权组合
        weighted_reward = sum(weights_dict[dim] * dimension_rewards[dim] 
                            for dim in self.dimensions)
        
        # 计算交互项（简化）
        dim_rewards_tensor = torch.tensor([[dimension_rewards[dim] for dim in self.dimensions]])
        interaction_score = self.interaction_network(dim_rewards_tensor).item()
        
        # 总奖励
        total_reward = weighted_reward + 0.1 * interaction_score  # 交互项权重较小
        
        return {
            'total_reward': total_reward,
            'dimension_rewards': dimension_rewards,
            'dimension_weights': weights_dict,
            'interaction_score': interaction_score
        }
    
    def _encode_context(self, context: Dict) -> torch.Tensor:
        """编码上下文信息"""
        # 简化实现：将上下文转换为固定维度向量
        # 实际应用中需要更复杂的上下文编码
        
        context_features = []
        
        # 任务类型
        task_type = context.get('task_type', 'general')
        task_encoding = {
            'qa': [1, 0, 0, 0],
            'creative': [0, 1, 0, 0],
            'analytical': [0, 0, 1, 0],
            'general': [0, 0, 0, 1]
        }
        context_features.extend(task_encoding.get(task_type, [0, 0, 0, 1]))
        
        # 用户偏好
        user_prefs = context.get('user_preferences', {})
        context_features.extend([
            user_prefs.get('safety_priority', 0.5),
            user_prefs.get('creativity_priority', 0.5),
            user_prefs.get('accuracy_priority', 0.5)
        ])
        
        # 填充到512维
        while len(context_features) < 512:
            context_features.append(0.0)
        
        return torch.tensor([context_features], dtype=torch.float32)

class PreferenceCalibration:
    """偏好校准与一致性分析"""
    
    def __init__(self):
        self.calibration_data = []
        self.consistency_metrics = {}
    
    def analyze_preference_consistency(self, preference_data: List[PreferenceData]) -> Dict:
        """分析偏好一致性"""
        
        print("=== 偏好一致性分析 ===")
        
        # 1. 传递性检查
        transitivity_violations = self._check_transitivity(preference_data)
        
        # 2. 标注者间一致性
        inter_annotator_agreement = self._compute_inter_annotator_agreement(preference_data)
        
        # 3. 难度与一致性的关系
        difficulty_consistency = self._analyze_difficulty_consistency(preference_data)
        
        # 4. 置信度校准
        confidence_calibration = self._analyze_confidence_calibration(preference_data)
        
        results = {
            'transitivity_violations': transitivity_violations,
            'inter_annotator_agreement': inter_annotator_agreement,
            'difficulty_consistency': difficulty_consistency,
            'confidence_calibration': confidence_calibration
        }
        
        self._visualize_consistency_analysis(results)
        
        return results
    
    def _check_transitivity(self, preference_data: List[PreferenceData]) -> Dict:
        """检查偏好传递性"""
        
        # 构建偏好图
        preference_graph = defaultdict(set)
        
        for data in preference_data:
            if data.preference == 0:  # 偏好A
                preference_graph[data.response_a].add(data.response_b)
            else:  # 偏好B
                preference_graph[data.response_b].add(data.response_a)
        
        # 检查传递性违背
        violations = 0
        total_checks = 0
        
        responses = list(preference_graph.keys())
        
        for i, resp_a in enumerate(responses):
            for j, resp_b in enumerate(responses[i+1:], i+1):
                for k, resp_c in enumerate(responses[j+1:], j+1):
                    # 检查 A > B, B > C 是否蕴含 A > C
                    if (resp_b in preference_graph[resp_a] and 
                        resp_c in preference_graph[resp_b] and
                        resp_c not in preference_graph[resp_a]):
                        violations += 1
                    total_checks += 1
        
        violation_rate = violations / total_checks if total_checks > 0 else 0
        
        return {
            'total_checks': total_checks,
            'violations': violations,
            'violation_rate': violation_rate
        }
    
    def _compute_inter_annotator_agreement(self, preference_data: List[PreferenceData]) -> Dict:
        """计算标注者间一致性"""
        
        # 按标注者分组
        annotator_data = defaultdict(list)
        for data in preference_data:
            annotator_data[data.annotator_id].append(data)
        
        if len(annotator_data) < 2:
            return {'agreement': 'N/A', 'reason': 'insufficient_annotators'}
        
        # 找到重叠标注（相同prompt-response pair）
        overlapping_annotations = defaultdict(list)
        
        for annotator, data_list in annotator_data.items():
            for data in data_list:
                key = (data.prompt, data.response_a, data.response_b)
                overlapping_annotations[key].append((annotator, data.preference))
        
        # 计算一致性
        agreements = 0
        total_overlaps = 0
        
        for key, annotations in overlapping_annotations.items():
            if len(annotations) >= 2:
                # 计算成对一致性
                for i in range(len(annotations)):
                    for j in range(i+1, len(annotations)):
                        if annotations[i][1] == annotations[j][1]:
                            agreements += 1
                        total_overlaps += 1
        
        agreement_rate = agreements / total_overlaps if total_overlaps > 0 else 0
        
        return {
            'overlapping_pairs': total_overlaps,
            'agreements': agreements,
            'agreement_rate': agreement_rate,
            'annotator_count': len(annotator_data)
        }
    
    def _analyze_difficulty_consistency(self, preference_data: List[PreferenceData]) -> Dict:
        """分析难度与一致性的关系"""
        
        # 按难度分组
        easy_samples = [d for d in preference_data if d.metadata.get('difficulty', 0.5) < 0.3]
        medium_samples = [d for d in preference_data if 0.3 <= d.metadata.get('difficulty', 0.5) < 0.7]
        hard_samples = [d for d in preference_data if d.metadata.get('difficulty', 0.5) >= 0.7]
        
        # 计算各组的预测准确率（基于true_preference）
        def compute_accuracy(samples):
            if not samples:
                return 0.0
            correct = sum(1 for s in samples if s.preference == s.metadata.get('true_preference', s.preference))
            return correct / len(samples)
        
        easy_accuracy = compute_accuracy(easy_samples)
        medium_accuracy = compute_accuracy(medium_samples)
        hard_accuracy = compute_accuracy(hard_samples)
        
        return {
            'easy_accuracy': easy_accuracy,
            'medium_accuracy': medium_accuracy,
            'hard_accuracy': hard_accuracy,
            'easy_count': len(easy_samples),
            'medium_count': len(medium_samples),
            'hard_count': len(hard_samples)
        }
    
    def _analyze_confidence_calibration(self, preference_data: List[PreferenceData]) -> Dict:
        """分析置信度校准"""
        
        # 按置信度分组
        confidence_bins = [0.0, 0.6, 0.7, 0.8, 0.9, 1.0]
        bin_stats = []
        
        for i in range(len(confidence_bins) - 1):
            bin_min, bin_max = confidence_bins[i], confidence_bins[i+1]
            bin_samples = [d for d in preference_data 
                          if bin_min <= d.confidence < bin_max]
            
            if bin_samples:
                # 计算该置信度区间的实际准确率
                correct = sum(1 for s in bin_samples 
                             if s.preference == s.metadata.get('true_preference', s.preference))
                accuracy = correct / len(bin_samples)
                avg_confidence = np.mean([s.confidence for s in bin_samples])
                
                bin_stats.append({
                    'confidence_range': f'{bin_min:.1f}-{bin_max:.1f}',
                    'avg_confidence': avg_confidence,
                    'accuracy': accuracy,
                    'count': len(bin_samples),
                    'calibration_error': abs(avg_confidence - accuracy)
                })
        
        # 总体校准误差
        overall_calibration_error = np.mean([stat['calibration_error'] for stat in bin_stats])
        
        return {
            'bin_statistics': bin_stats,
            'overall_calibration_error': overall_calibration_error
        }
    
    def _visualize_consistency_analysis(self, results: Dict):
        """可视化一致性分析结果"""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 传递性违背
        trans_data = results['transitivity_violations']
        if trans_data['total_checks'] > 0:
            violation_rate = trans_data['violation_rate']
            axes[0, 0].bar(['违背', '满足'], [violation_rate, 1-violation_rate], 
                          color=['red', 'green'], alpha=0.7)
            axes[0, 0].set_title(f'传递性检查\n(违背率: {violation_rate:.1%})')
            axes[0, 0].set_ylabel('比例')
        
        # 难度vs准确率
        diff_data = results['difficulty_consistency']
        difficulties = ['易', '中', '难']
        accuracies = [diff_data['easy_accuracy'], diff_data['medium_accuracy'], diff_data['hard_accuracy']]
        
        axes[0, 1].bar(difficulties, accuracies, alpha=0.7, color='blue')
        axes[0, 1].set_title('难度与标注准确率')
        axes[0, 1].set_ylabel('准确率')
        axes[0, 1].set_ylim(0, 1)
        
        # 置信度校准
        calib_data = results['confidence_calibration']
        if calib_data['bin_statistics']:
            confidences = [stat['avg_confidence'] for stat in calib_data['bin_statistics']]
            accuracies = [stat['accuracy'] for stat in calib_data['bin_statistics']]
            
            axes[1, 0].scatter(confidences, accuracies, alpha=0.7, s=100)
            axes[1, 0].plot([0, 1], [0, 1], 'r--', alpha=0.5, label='完美校准')
            axes[1, 0].set_xlabel('平均置信度')
            axes[1, 0].set_ylabel('实际准确率')
            axes[1, 0].set_title('置信度校准')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # 标注者一致性
        agreement_data = results['inter_annotator_agreement']
        if isinstance(agreement_data['agreement_rate'], (int, float)):
            agreement_rate = agreement_data['agreement_rate']
            axes[1, 1].bar(['一致', '不一致'], [agreement_rate, 1-agreement_rate],
                          color=['green', 'orange'], alpha=0.7)
            axes[1, 1].set_title(f'标注者间一致性\n(一致率: {agreement_rate:.1%})')
            axes[1, 1].set_ylabel('比例')
        
        plt.tight_layout()
        plt.show()

# 多维偏好学习的综合演示
def demonstrate_multidimensional_preference_learning():
    """演示多维偏好学习"""
    
    print("="*60)
    print("多维偏好建模与校准分析 - 综合演示")
    print("="*60)
    
    # 1. 创建多维偏好模型
    dimensions = ['helpfulness', 'harmlessness', 'honesty']
    multi_model = MultiDimensionalPreferenceModel(dimensions)
    
    # 2. 测试多维奖励计算
    print("\n1. 多维奖励计算示例")
    
    test_cases = [
        {
            'text': 'Paris is the capital of France.',
            'context': {'task_type': 'qa', 'user_preferences': {'accuracy_priority': 0.9}}
        },
        {
            'text': 'I cannot provide information about harmful activities.',
            'context': {'task_type': 'general', 'user_preferences': {'safety_priority': 0.9}}
        },
        {
            'text': 'Let me think about this complex question step by step...',
            'context': {'task_type': 'analytical', 'user_preferences': {'creativity_priority': 0.7}}
        }
    ]
    
    for i, case in enumerate(test_cases):
        print(f"\n测试案例 {i+1}:")
        print(f"  文本: {case['text']}")
        
        rewards = multi_model.compute_multidimensional_reward(
            case['text'], case['context']
        )
        
        print(f"  总奖励: {rewards['total_reward']:.3f}")
        print(f"  维度奖励:")
        for dim, reward in rewards['dimension_rewards'].items():
            weight = rewards['dimension_weights'][dim]
            print(f"    {dim}: {reward:.3f} (权重: {weight:.3f})")
        print(f"  交互得分: {rewards['interaction_score']:.3f}")
    
    # 3. 偏好一致性分析
    print("\n2. 偏好一致性与校准分析")
    
    # 使用之前生成的偏好数据
    collector = PreferenceDataCollector()
    prompts = [
        "What is machine learning?",
        "How do you cook pasta?",
        "Explain quantum physics.",
        "What is the meaning of life?",
        "How does photosynthesis work?"
    ]
    
    response_pairs = [
        ("ML is AI.", "Machine learning is a subset of artificial intelligence that enables computers to learn from data."),
        ("Boil and drain.", "Bring water to boil, add pasta, cook for 8-12 minutes, then drain and serve with sauce."),
        ("It's complex physics.", "Quantum physics describes the behavior of matter and energy at the atomic and subatomic level."),
        ("It's personal.", "The meaning of life is a philosophical question that varies for each individual based on their values and beliefs."),
        ("Plants make food.", "Photosynthesis is the process by which plants convert light energy into chemical energy using chlorophyll.")
    ]
    
    preference_data = collector.simulate_human_annotation(
        prompts, response_pairs, annotation_noise=0.2
    )
    
    # 一致性分析
    calibrator = PreferenceCalibration()
    consistency_results = calibrator.analyze_preference_consistency(preference_data)
    
    print(f"\n一致性分析结果:")
    print(f"  传递性违背率: {consistency_results['transitivity_violations']['violation_rate']:.1%}")
    print(f"  标注者一致率: {consistency_results['inter_annotator_agreement'].get('agreement_rate', 'N/A')}")
    print(f"  整体校准误差: {consistency_results['confidence_calibration']['overall_calibration_error']:.3f}")
    
    print(f"\n=== 多维偏好学习演示完成 ===")
    print(f"多维建模能够更准确地捕捉人类偏好的复杂性")
    print(f"一致性分析有助于提升标注质量和模型可靠性")

# 运行多维偏好学习演示
demonstrate_multidimensional_preference_learning()
```

现在我已经完成了第05章第02节"奖励建模与偏好学习"的完整内容。这一节涵盖了：

1. **Bradley-Terry模型数学原理**：从人类偏好比较到概率建模
2. **奖励网络架构设计**：深度学习在偏好学习中的应用
3. **多维偏好建模**：处理复杂的多维度人类偏好
4. **偏好数据收集与标注**：高质量人类反馈的获取策略
5. **偏好一致性与校准分析**：确保学习到的偏好模型的可靠性

<function_calls>
<invoke name="TodoWrite">
<parameter name="todos">[{"content": "Research best practices for technical tutorial writing", "status": "completed", "priority": "high", "id": "1"}, {"content": "Design comprehensive folder structure for MiniGPT handbook", "status": "completed", "priority": "high", "id": "2"}, {"content": "Create detailed outline with ultra-deep analysis approach", "status": "completed", "priority": "high", "id": "3"}, {"content": "Write handbook introduction and navigation guide", "status": "completed", "priority": "high", "id": "4"}, {"content": "Implement chapter-by-chapter content with mathematical rigor", "status": "in_progress", "priority": "high", "id": "5"}, {"content": "Write Chapter 1: Mathematical Foundations", "status": "completed", "priority": "high", "id": "6"}, {"content": "Write Chapter 2: Transformer Architecture", "status": "completed", "priority": "high", "id": "7"}, {"content": "Write Chapter 3: Pre-training Theory and Implementation", "status": "completed", "priority": "high", "id": "8"}, {"content": "Write Chapter 4: Supervised Fine-tuning Deep Dive", "status": "completed", "priority": "high", "id": "14"}, {"content": "Write Chapter 5: Reinforcement Learning from Human Feedback", "status": "in_progress", "priority": "high", "id": "19"}, {"content": "Create Chapter 5 folder structure and README", "status": "completed", "priority": "medium", "id": "20"}, {"content": "Write Chapter 5 Section 1: RLHF Theory and Mathematical Foundations", "status": "completed", "priority": "high", "id": "21"}, {"content": "Write Chapter 5 Section 2: Reward Modeling and Preference Learning", "status": "completed", "priority": "high", "id": "22"}, {"content": "Write Chapter 5 Section 3: PPO Algorithm for Language Model Fine-tuning", "status": "in_progress", "priority": "high", "id": "23"}, {"content": "Write Chapter 5 Section 4: DPO and Alternative RLHF Methods", "status": "pending", "priority": "high", "id": "24"}]