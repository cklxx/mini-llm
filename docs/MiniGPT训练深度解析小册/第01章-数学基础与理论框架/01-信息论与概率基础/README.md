# 01 信息论与概率基础

> **理解语言模型的信息论本质**

## 核心思想

语言模型本质上是一个概率分布估计器，它试图学习自然语言的统计规律。信息论为我们提供了量化和理解这一过程的数学工具。

**关键洞察**：
- 语言的不确定性可以用熵来度量
- 模型的预测质量可以用交叉熵来衡量
- 分布间的差异可以用KL散度来量化

## 1.1 香农信息论的核心概念

### 信息量的定义

对于概率为 $p$ 的事件，其信息量（自信息）定义为：

$$I(x) = -\log p(x)$$

**直观理解**：
- 概率越小的事件，信息量越大
- 必然事件($p=1$)的信息量为0
- 不可能事件($p=0$)的信息量为无穷大

**代码验证**：
```python
import torch
import torch.nn.functional as F

def self_information(prob):
    """计算自信息"""
    return -torch.log(prob)

# 示例：不同概率事件的信息量
probs = torch.tensor([0.5, 0.1, 0.01, 0.001])
info = self_information(probs)
print(f"概率: {probs}")
print(f"信息量: {info}")
# 概率越小，信息量越大
```

### 熵：不确定性的度量

对于随机变量 $X$ 的概率分布 $P(X)$，其熵定义为：

$$H(X) = -\sum_{x} P(x) \log P(x) = \mathbb{E}_{x \sim P}[-\log P(x)]$$

**数学性质**：
1. **非负性**：$H(X) \geq 0$
2. **最大熵**：当 $P(x)$ 为均匀分布时，$H(X)$ 达到最大值 $\log |X|$
3. **确定性**：当 $P(x)$ 为点分布时，$H(X) = 0$

**在语言模型中的应用**：
```python
def entropy(probs):
    """计算熵"""
    return -torch.sum(probs * torch.log(probs + 1e-8))

# 词汇表上的概率分布
vocab_probs = F.softmax(torch.randn(10000), dim=0)
lang_entropy = entropy(vocab_probs)
print(f"语言的熵: {lang_entropy:.4f}")

# 对比：均匀分布的熵
uniform_probs = torch.ones(10000) / 10000
uniform_entropy = entropy(uniform_probs)
print(f"均匀分布的熵: {uniform_entropy:.4f}")
```

**MiniGPT 中的对应**：
```python
# src/training/trainer.py 中困惑度的计算
def compute_perplexity(self, loss):
    """困惑度 = exp(交叉熵) = exp(平均负对数似然)"""
    return torch.exp(loss)
```

## 1.2 交叉熵：模型质量的衡量

给定真实分布 $P$ 和模型分布 $Q$，交叉熵定义为：

$$H(P, Q) = -\sum_{x} P(x) \log Q(x) = \mathbb{E}_{x \sim P}[-\log Q(x)]$$

**数学推导**：

在语言模型中，我们有：
- 真实分布：$P(x_{1:T}) = \prod_{t=1}^{T} P(x_t | x_{<t})$
- 模型分布：$Q_\theta(x_{1:T}) = \prod_{t=1}^{T} Q_\theta(x_t | x_{<t})$

交叉熵损失为：
$$\mathcal{L}_{CE} = -\frac{1}{T}\sum_{t=1}^{T} \log Q_\theta(x_t | x_{<t})$$

**关键洞察**：交叉熵 = 熵 + KL散度
$$H(P, Q) = H(P) + D_{KL}(P||Q)$$

由于 $H(P)$ 是常数，最小化交叉熵等价于最小化KL散度。

**代码实现分析**：
```python
# MiniGPT 中的交叉熵损失计算
def compute_loss(self, logits, labels):
    """计算语言模型损失"""
    # logits: (batch_size * seq_len, vocab_size)
    # labels: (batch_size * seq_len,)
    
    # PyTorch的CrossEntropyLoss内部实现：
    # 1. log_softmax(logits)
    # 2. nll_loss(log_probs, labels)
    loss_fn = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_id)
    loss = loss_fn(logits, labels)
    
    return loss
```

**手工实现交叉熵**：
```python
def manual_cross_entropy(logits, labels, ignore_index=-100):
    """手工实现交叉熵，理解计算过程"""
    # 计算log概率
    log_probs = F.log_softmax(logits, dim=-1)
    
    # 选择正确类别的log概率
    selected_log_probs = log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    
    # 应用掩码
    mask = (labels != ignore_index)
    masked_log_probs = selected_log_probs * mask.float()
    
    # 计算平均损失
    loss = -masked_log_probs.sum() / mask.sum()
    
    return loss
```

## 1.3 KL散度：分布差异的度量

KL散度（Kullback-Leibler divergence）衡量两个概率分布的差异：

$$D_{KL}(P||Q) = \sum_{x} P(x) \log \frac{P(x)}{Q(x)} = \mathbb{E}_{x \sim P}\left[\log \frac{P(x)}{Q(x)}\right]$$

**数学性质**：
1. **非负性**：$D_{KL}(P||Q) \geq 0$，当且仅当 $P = Q$ 时等号成立
2. **非对称性**：$D_{KL}(P||Q) \neq D_{KL}(Q||P)$
3. **不满足三角不等式**：不是真正的度量

**在深度学习中的应用**：

### 1. 知识蒸馏
```python
def kl_divergence_loss(student_logits, teacher_logits, temperature=3.0):
    """知识蒸馏中的KL散度损失"""
    # 软化概率分布
    student_probs = F.softmax(student_logits / temperature, dim=-1)
    teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
    
    # 计算KL散度
    kl_loss = F.kl_div(
        F.log_softmax(student_logits / temperature, dim=-1),
        teacher_probs,
        reduction='batchmean'
    ) * (temperature ** 2)
    
    return kl_loss
```

### 2. 正则化项
```python
def kl_regularization(model_probs, prior_probs):
    """KL散度作为正则化项"""
    return F.kl_div(
        torch.log(model_probs + 1e-8),
        prior_probs,
        reduction='batchmean'
    )
```

### 3. RLHF中的KL约束
```python
# 在PPO训练中约束策略不要偏离太远
def ppo_kl_penalty(policy_logprobs, ref_logprobs, beta=0.1):
    """PPO中的KL惩罚项"""
    kl_div = policy_logprobs - ref_logprobs
    return beta * kl_div.mean()
```

## 1.4 互信息：依赖关系的量化

互信息衡量两个随机变量之间的依赖程度：

$$I(X; Y) = \sum_{x,y} P(x,y) \log \frac{P(x,y)}{P(x)P(y)}$$

**等价表达式**：
$$I(X; Y) = H(X) - H(X|Y) = H(Y) - H(Y|X)$$

**直观理解**：
- $I(X; Y) = 0$ 当且仅当 $X$ 和 $Y$ 独立
- $I(X; Y) = H(X)$ 当 $Y$ 完全确定 $X$ 时
- 互信息总是非负的

**在注意力机制中的应用**：

注意力权重可以看作是查询和键之间互信息的一种近似：

```python
def attention_mutual_info(attention_weights):
    """估计注意力权重对应的互信息"""
    # attention_weights: (batch_size, n_heads, seq_len, seq_len)
    
    # 边际分布
    p_query = attention_weights.sum(dim=-1, keepdim=True)  # 对键求和
    p_key = attention_weights.sum(dim=-2, keepdim=True)    # 对查询求和
    
    # 联合分布
    p_joint = attention_weights
    
    # 独立性假设下的分布
    p_independent = p_query * p_key
    
    # 互信息近似
    mi = p_joint * torch.log(p_joint / (p_independent + 1e-8) + 1e-8)
    
    return mi.sum(dim=(-2, -1))
```

## 1.5 困惑度：语言模型的评估指标

困惑度（Perplexity）是语言模型最重要的评估指标：

$$\text{PPL} = \exp(H(P, Q)) = \exp\left(-\frac{1}{T}\sum_{t=1}^{T} \log Q_\theta(x_t | x_{<t})\right)$$

**信息论解释**：
- 困惑度表示模型在每个位置平均"困惑"于多少个选择
- 完美模型：$\text{PPL} = 1$
- 随机模型：$\text{PPL} = |V|$（词汇表大小）

**与压缩的关系**：
$$\text{平均编码长度} = \log_2(\text{PPL})$$

**代码实现**：
```python
def compute_perplexity(model, dataloader, tokenizer):
    """计算模型在数据集上的困惑度"""
    model.eval()
    total_loss = 0
    total_tokens = 0
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            
            # 前向传播
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            
            # 计算损失（只在非padding位置）
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()
            
            loss_fn = nn.CrossEntropyLoss(
                ignore_index=tokenizer.pad_token_id,
                reduction='sum'
            )
            
            loss = loss_fn(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
            
            # 累积
            total_loss += loss.item()
            total_tokens += (shift_labels != tokenizer.pad_token_id).sum().item()
    
    # 计算困惑度
    avg_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss))
    
    return perplexity.item()
```

## 1.6 实践：信息论在MiniGPT中的应用

### 损失函数的信息论解释

```python
# MiniGPT 中的损失计算
class PreTrainer:
    def compute_loss(self, logits, labels):
        """语言模型损失 = 交叉熵 = 负对数似然"""
        # 展平张量进行计算
        logits = logits.reshape(-1, logits.size(-1))
        labels = labels.reshape(-1)
        
        # 交叉熵损失
        loss_fn = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_id)
        loss = loss_fn(logits, labels)
        
        return loss
    
    def compute_perplexity(self, loss):
        """困惑度 = exp(交叉熵)"""
        return torch.exp(loss)
```

### 信息论指标的监控

```python
def log_info_metrics(self, loss, logits, labels):
    """记录信息论相关指标"""
    # 基础指标
    perplexity = torch.exp(loss)
    
    # 预测分布的熵
    probs = F.softmax(logits, dim=-1)
    pred_entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1).mean()
    
    # 预测的置信度（最大概率）
    max_prob = probs.max(dim=-1)[0].mean()
    
    print(f"Loss: {loss:.4f}")
    print(f"Perplexity: {perplexity:.4f}")
    print(f"Prediction Entropy: {pred_entropy:.4f}")
    print(f"Max Probability: {max_prob:.4f}")
```

## 小结与思考

本节介绍了信息论的核心概念及其在语言模型中的应用：

1. **熵**量化了语言的不确定性
2. **交叉熵**衡量了模型预测的质量
3. **KL散度**度量了分布间的差异
4. **互信息**量化了变量间的依赖关系
5. **困惑度**是语言模型的核心评估指标

**思考题**：
1. 为什么说语言模型本质上是在进行数据压缩？
2. 如何从信息论角度理解过拟合现象？
3. 注意力机制如何实现信息的选择性传递？

**下一节预告**：我们将学习线性代数基础，理解Transformer中矩阵运算的几何意义。

---

*信息论为我们提供了理解智能的数学语言，语言模型正是这一理论的完美实践。* 📊