# 01 语言建模概率基础

> **从信息熵到困惑度：语言建模的统计学基石**

## 核心思想

语言建模的本质是**概率建模**：给定一个词汇序列的前缀，预测下一个词的概率分布。这个看似简单的任务，实际上包含了语言理解的全部复杂性——语法、语义、常识、推理等都隐含在这个条件概率分布中。

**关键洞察**：
- **概率链式法则**将复杂的序列建模分解为可处理的条件概率
- **最大似然估计**提供了从数据中学习模型参数的理论框架
- **信息熵**度量了语言的固有不确定性和模型的预测能力
- **困惑度**是评估语言模型质量的核心指标

从数学角度看，语言建模就是在学习真实语言分布$P_{data}$的一个近似$P_\theta$，使得两个分布尽可能接近。

## 1.1 概率链式分解的数学基础

### 从联合概率到条件概率

**基本问题**：给定词汇表$V$和序列长度$n$，语言的联合概率空间大小为$|V|^n$。对于现实的词汇表（~50K词）和序列长度（~2K tokens），这个空间是天文数字，无法直接建模。

**解决方案**：概率链式法则
$$P(x_1, x_2, ..., x_n) = P(x_1) \prod_{i=2}^{n} P(x_i | x_1, x_2, ..., x_{i-1})$$

这将联合概率分解为一系列**条件概率**的乘积，每个条件概率都可以通过神经网络来近似。

**数学表达**：
```python
# MiniGPT中的概率计算实现解析
def compute_sequence_probability(model, input_ids, attention_mask=None):
    """计算序列的对数概率"""
    with torch.no_grad():
        # 前向传播获取logits
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs  # (batch_size, seq_len, vocab_size)
        
        # 计算每个位置的条件概率
        log_probs = F.log_softmax(logits, dim=-1)
        
        # 获取真实token的对数概率
        # 注意：预测位置i使用的是token i+1的概率
        target_ids = input_ids[:, 1:]  # 移除<BOS>或使用下一个token
        target_log_probs = log_probs[:, :-1].gather(dim=-1, 
                                                   index=target_ids.unsqueeze(-1)).squeeze(-1)
        
        # 考虑attention mask（忽略padding）
        if attention_mask is not None:
            mask = attention_mask[:, 1:]  # 对应target位置
            target_log_probs = target_log_probs * mask
            seq_log_prob = target_log_probs.sum(dim=-1)
            seq_length = mask.sum(dim=-1)
        else:
            seq_log_prob = target_log_probs.sum(dim=-1)
            seq_length = torch.tensor(target_ids.size(1))
        
        return seq_log_prob, seq_length
```

### 条件独立性假设的影响

**马尔可夫假设**：在实际实现中，我们通常假设当前词只依赖于有限的历史窗口：
$$P(x_i | x_1, ..., x_{i-1}) \approx P(x_i | x_{i-k}, ..., x_{i-1})$$

**Transformer的优势**：通过注意力机制，Transformer可以建模**全序列的依赖关系**，而不受固定窗口限制：
```python
def analyze_dependency_range(attention_weights):
    """分析模型的实际依赖范围"""
    # attention_weights: (batch_size, n_heads, seq_len, seq_len)
    
    batch_size, n_heads, seq_len, _ = attention_weights.shape
    
    # 计算每个位置的有效注意力范围
    effective_ranges = []
    
    for pos in range(seq_len):
        # 获取位置pos对历史的注意力分布
        attn_dist = attention_weights[:, :, pos, :pos+1].mean(dim=(0,1))  # 平均所有头和batch
        
        # 计算有效范围（累积90%注意力权重的范围）
        sorted_attn, indices = torch.sort(attn_dist, descending=True)
        cumsum_attn = torch.cumsum(sorted_attn, dim=0)
        effective_range = (cumsum_attn <= 0.9).sum().item() + 1
        
        effective_ranges.append(effective_range)
        
        print(f"位置 {pos}: 有效依赖范围 = {effective_range} tokens")
    
    avg_range = sum(effective_ranges) / len(effective_ranges)
    print(f"平均有效依赖范围: {avg_range:.2f} tokens")
    
    return effective_ranges
```

## 1.2 最大似然估计的理论框架

### 似然函数与对数似然

给定训练数据集$\mathcal{D} = \{(x^{(1)}, x^{(2)}, ..., x^{(N)})\}$，模型参数$\theta$的**似然函数**为：

$$L(\theta) = \prod_{i=1}^{N} P(x^{(i)}; \theta) = \prod_{i=1}^{N} \prod_{j=1}^{|x^{(i)}|} P(x^{(i)}_j | x^{(i)}_{<j}; \theta)$$

**对数似然**（更数值稳定）：
$$\ell(\theta) = \sum_{i=1}^{N} \sum_{j=1}^{|x^{(i)}|} \log P(x^{(i)}_j | x^{(i)}_{<j}; \theta)$$

**最大似然估计**：
$$\hat{\theta} = \arg\max_\theta \ell(\theta) = \arg\min_\theta \left(-\frac{1}{N}\ell(\theta)\right)$$

这就是我们熟悉的**交叉熵损失函数**！

```python
# MiniGPT中的损失函数实现 (src/training/trainer.py)
def compute_loss(self, outputs, targets, attention_mask=None):
    """计算语言建模损失（负对数似然）"""
    
    # outputs: (batch_size, seq_len, vocab_size)
    # targets: (batch_size, seq_len)
    
    # 将输出和目标展平
    logits = outputs.view(-1, outputs.size(-1))  # (batch_size * seq_len, vocab_size)
    targets = targets.view(-1)  # (batch_size * seq_len,)
    
    # 计算交叉熵损失
    loss = F.cross_entropy(logits, targets, ignore_index=self.tokenizer.pad_token_id, reduction='none')
    
    # 如果有attention mask，应用mask
    if attention_mask is not None:
        mask = attention_mask.view(-1)
        loss = loss * mask
        # 返回平均损失（只考虑非padding tokens）
        return loss.sum() / mask.sum()
    else:
        return loss.mean()
```

### Fisher信息与参数估计的方差

**Fisher信息矩阵**定义为：
$$I(\theta) = -\mathbb{E}\left[\frac{\partial^2 \ell(\theta)}{\partial \theta^2}\right]$$

**Cramér–Rao下界**告诉我们，参数估计的方差至少为：
$$\text{Var}(\hat{\theta}) \geq I(\theta)^{-1}$$

这解释了为什么**更多的数据**（更大的$N$）能够提供**更精确的参数估计**：

```python
def analyze_parameter_estimation_quality(model, data_loader, num_samples=1000):
    """分析参数估计的质量"""
    
    model.eval()
    log_likelihoods = []
    parameter_gradients = []
    
    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            if i >= num_samples:
                break
                
            input_ids, targets = batch
            
            # 计算对数似然
            outputs = model(input_ids)
            loss = F.cross_entropy(outputs.view(-1, outputs.size(-1)), 
                                 targets.view(-1), reduction='mean')
            log_likelihood = -loss.item()
            log_likelihoods.append(log_likelihood)
            
            # 计算梯度（用于估计Fisher信息）
            model.zero_grad()
            loss.backward()
            
            grad_norm = 0
            for param in model.parameters():
                if param.grad is not None:
                    grad_norm += param.grad.norm().item() ** 2
            parameter_gradients.append(math.sqrt(grad_norm))
    
    # 统计分析
    ll_mean = np.mean(log_likelihoods)
    ll_std = np.std(log_likelihoods)
    grad_mean = np.mean(parameter_gradients)
    grad_std = np.std(parameter_gradients)
    
    print(f"对数似然统计:")
    print(f"  均值: {ll_mean:.4f}")
    print(f"  标准差: {ll_std:.4f}")
    print(f"  置信区间 (95%): [{ll_mean - 1.96*ll_std:.4f}, {ll_mean + 1.96*ll_std:.4f}]")
    
    print(f"\\n梯度范数统计:")
    print(f"  均值: {grad_mean:.4f}")  
    print(f"  标准差: {grad_std:.4f}")
    
    return {
        'log_likelihood': {'mean': ll_mean, 'std': ll_std},
        'gradient_norm': {'mean': grad_mean, 'std': grad_std}
    }
```

## 1.3 信息熵与语言复杂度

### Shannon熵的语言学解释

**语言的熵**衡量了语言的固有不确定性：
$$H(X) = -\sum_{x \in V} P(x) \log P(x)$$

对于条件分布（语言建模的核心）：
$$H(X|Y) = -\sum_{y} P(y) \sum_{x} P(x|y) \log P(x|y)$$

**直觉理解**：
- 高熵 → 语言更随机，更难预测
- 低熵 → 语言更规律，更容易预测

```python
def compute_language_entropy(model, data_loader, vocabulary_size):
    """计算语言的条件熵"""
    
    model.eval()
    total_entropy = 0
    total_tokens = 0
    
    with torch.no_grad():
        for batch in data_loader:
            input_ids, targets = batch
            
            # 获取预测分布
            outputs = model(input_ids)  # (batch_size, seq_len, vocab_size)
            probs = F.softmax(outputs, dim=-1)
            
            # 计算每个位置的熵
            # H = -sum(p * log(p))
            log_probs = torch.log(probs + 1e-8)  # 数值稳定性
            entropy = -(probs * log_probs).sum(dim=-1)  # (batch_size, seq_len)
            
            # 累积统计
            total_entropy += entropy.sum().item()
            total_tokens += entropy.numel()
    
    avg_entropy = total_entropy / total_tokens
    
    # 转换为不同单位
    entropy_nat = avg_entropy  # 自然对数单位
    entropy_bit = avg_entropy / math.log(2)  # bit单位
    entropy_normalized = avg_entropy / math.log(vocabulary_size)  # 归一化到[0,1]
    
    print(f"语言条件熵分析:")
    print(f"  条件熵 (nat): {entropy_nat:.4f}")
    print(f"  条件熵 (bit): {entropy_bit:.4f}")
    print(f"  归一化熵: {entropy_normalized:.4f}")
    print(f"  理论最大熵: {math.log(vocabulary_size):.4f} nat")
    print(f"  熵效率: {entropy_normalized:.2%}")
    
    return {
        'entropy_nat': entropy_nat,
        'entropy_bit': entropy_bit,
        'entropy_normalized': entropy_normalized,
        'max_entropy': math.log(vocabulary_size)
    }
```

### 交叉熵与KL散度

**交叉熵**衡量使用模型分布$Q$来编码真实分布$P$的成本：
$$H(P, Q) = -\sum_{x} P(x) \log Q(x)$$

**KL散度**衡量两个分布的差异：
$$D_{KL}(P||Q) = \sum_{x} P(x) \log \frac{P(x)}{Q(x)} = H(P, Q) - H(P)$$

**关键洞察**：
$$\mathbb{E}_{x \sim P_{data}}[-\log P_\theta(x)] = H(P_{data}) + D_{KL}(P_{data}||P_\theta)$$

最小化交叉熵损失等价于最小化模型分布与数据分布的KL散度！

```python
def analyze_distribution_divergence(model, true_distribution, test_data):
    """分析模型分布与真实分布的差异"""
    
    model.eval()
    kl_divergences = []
    cross_entropies = []
    
    with torch.no_grad():
        for batch in test_data:
            input_ids, targets = batch
            
            # 获取模型预测分布
            outputs = model(input_ids)
            model_probs = F.softmax(outputs, dim=-1)  # Q(x|context)
            
            # 计算交叉熵
            true_probs = true_distribution[targets]  # P(x|context) 
            cross_entropy = -(true_probs * torch.log(model_probs + 1e-8)).sum(dim=-1)
            
            # 计算KL散度  
            kl_div = (true_probs * torch.log(true_probs / (model_probs + 1e-8) + 1e-8)).sum(dim=-1)
            
            cross_entropies.extend(cross_entropy.flatten().tolist())
            kl_divergences.extend(kl_div.flatten().tolist())
    
    # 统计分析
    avg_ce = np.mean(cross_entropies)
    avg_kl = np.mean(kl_divergences)
    
    print(f"分布分析:")
    print(f"  平均交叉熵: {avg_ce:.4f}")
    print(f"  平均KL散度: {avg_kl:.4f}")
    print(f"  数据分布熵估计: {avg_ce - avg_kl:.4f}")
    
    return {
        'cross_entropy': avg_ce,
        'kl_divergence': avg_kl,
        'data_entropy_estimate': avg_ce - avg_kl
    }
```

## 1.4 困惑度：语言模型的标准评估指标

### 困惑度的数学定义

**困惑度(Perplexity)**是交叉熵的指数：
$$\text{PPL} = \exp(H(P_{data}, P_\theta)) = \exp\left(-\frac{1}{N}\sum_{i=1}^{N} \log P_\theta(x_i)\right)$$

**几何解释**：困惑度表示模型在每个位置"困惑"的选择数量。
- PPL = 1：模型完全确定（总是正确预测）
- PPL = |V|：模型完全随机（均匀分布预测）

**信息论解释**：困惑度是**有效词汇量**，即模型认为在当前上下文中等可能出现的词的数量。

```python
def compute_perplexity(model, data_loader, tokenizer):
    """计算模型困惑度"""
    
    model.eval()
    total_loss = 0
    total_tokens = 0
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Computing perplexity"):
            input_ids = batch['input_ids']
            attention_mask = batch.get('attention_mask', None)
            
            # 计算损失
            outputs = model(input_ids)
            
            # 创建targets（向左偏移一位）
            targets = input_ids[:, 1:].contiguous()
            logits = outputs[:, :-1].contiguous()
            
            # 计算交叉熵损失
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=tokenizer.pad_token_id,
                reduction='sum'
            )
            
            # 计算有效token数量
            if attention_mask is not None:
                mask = attention_mask[:, 1:].contiguous()
                num_tokens = mask.sum().item()
            else:
                num_tokens = (targets != tokenizer.pad_token_id).sum().item()
            
            total_loss += loss.item()
            total_tokens += num_tokens
    
    # 计算平均损失和困惑度
    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)
    
    print(f"困惑度评估结果:")
    print(f"  平均损失: {avg_loss:.4f}")
    print(f"  困惑度: {perplexity:.2f}")
    print(f"  总token数: {total_tokens:,}")
    
    return perplexity, avg_loss
```

### 困惑度的分解分析

困惑度可以按不同维度分解，帮助理解模型的行为：

```python
def detailed_perplexity_analysis(model, data_loader, tokenizer):
    """详细的困惑度分析"""
    
    model.eval()
    
    # 按位置分析
    position_losses = defaultdict(list)
    # 按token频率分析  
    token_losses = defaultdict(list)
    # 按token类型分析
    token_type_losses = defaultdict(list)
    
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids']
            
            outputs = model(input_ids)
            targets = input_ids[:, 1:]
            logits = outputs[:, :-1]
            
            # 计算每个位置的损失
            losses = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=tokenizer.pad_token_id,
                reduction='none'
            ).view(targets.shape)
            
            # 按位置统计
            for pos in range(targets.size(1)):
                valid_mask = targets[:, pos] != tokenizer.pad_token_id
                if valid_mask.sum() > 0:
                    pos_loss = losses[:, pos][valid_mask].mean().item()
                    position_losses[pos].append(pos_loss)
            
            # 按token统计  
            for i in range(targets.size(0)):
                for j in range(targets.size(1)):
                    token_id = targets[i, j].item()
                    if token_id != tokenizer.pad_token_id:
                        loss = losses[i, j].item()
                        token_losses[token_id].append(loss)
                        
                        # token类型分析
                        token = tokenizer.decode([token_id])
                        if token.isalpha():
                            token_type = 'alphabetic'
                        elif token.isdigit():
                            token_type = 'numeric'
                        elif token.isspace():
                            token_type = 'whitespace'
                        else:
                            token_type = 'punctuation'
                        
                        token_type_losses[token_type].append(loss)
    
    # 统计分析
    print("=== 困惑度分解分析 ===")
    
    # 1. 位置分析
    print("\\n1. 按位置分析:")
    for pos in sorted(position_losses.keys())[:10]:  # 前10个位置
        losses = position_losses[pos]
        avg_loss = np.mean(losses)
        ppl = math.exp(avg_loss)
        print(f"  位置 {pos:2d}: 损失={avg_loss:.4f}, 困惑度={ppl:.2f}")
    
    # 2. 按token类型分析
    print("\\n2. 按token类型分析:")
    for token_type, losses in token_type_losses.items():
        avg_loss = np.mean(losses)
        ppl = math.exp(avg_loss)
        count = len(losses)
        print(f"  {token_type:12s}: 损失={avg_loss:.4f}, 困惑度={ppl:.2f}, 样本数={count:,}")
    
    # 3. 高频vs低频token分析
    print("\\n3. 按token频率分析:")
    token_freqs = [(token_id, len(losses)) for token_id, losses in token_losses.items()]
    token_freqs.sort(key=lambda x: x[1], reverse=True)
    
    # 高频token (top 10%)
    high_freq_tokens = token_freqs[:len(token_freqs)//10]
    high_freq_losses = []
    for token_id, _ in high_freq_tokens:
        high_freq_losses.extend(token_losses[token_id])
    
    # 低频token (bottom 10%)  
    low_freq_tokens = token_freqs[-len(token_freqs)//10:]
    low_freq_losses = []
    for token_id, _ in low_freq_tokens:
        low_freq_losses.extend(token_losses[token_id])
    
    high_freq_ppl = math.exp(np.mean(high_freq_losses))
    low_freq_ppl = math.exp(np.mean(low_freq_losses))
    
    print(f"  高频token (top 10%): 困惑度={high_freq_ppl:.2f}")
    print(f"  低频token (bottom 10%): 困惑度={low_freq_ppl:.2f}")
    print(f"  困惑度比值: {low_freq_ppl/high_freq_ppl:.2f}")
    
    return {
        'position_losses': dict(position_losses),
        'token_type_losses': dict(token_type_losses),
        'frequency_analysis': {
            'high_freq_ppl': high_freq_ppl,
            'low_freq_ppl': low_freq_ppl
        }
    }
```

### 困惑度与人类语言能力的对比

**经验数据**：
- 随机基线：困惑度 ≈ |V| (词汇表大小)
- 传统n-gram模型：困惑度 ≈ 100-300
- 早期神经语言模型：困惑度 ≈ 50-100  
- 现代Transformer：困惑度 ≈ 10-30
- 人类表现估计：困惑度 ≈ 12

```python
def compare_with_baselines(model, test_data, tokenizer):
    """与基线模型比较困惑度"""
    
    # 1. 计算目标模型困惑度
    model_ppl, _ = compute_perplexity(model, test_data, tokenizer)
    
    # 2. 计算随机基线困惑度
    vocab_size = len(tokenizer.vocab)
    random_ppl = vocab_size
    
    # 3. 计算unigram基线困惑度
    # 统计训练数据的token频率
    token_counts = defaultdict(int)
    total_tokens = 0
    
    for batch in test_data:
        input_ids = batch['input_ids']
        for token_id in input_ids.flatten():
            if token_id != tokenizer.pad_token_id:
                token_counts[token_id.item()] += 1
                total_tokens += 1
    
    # 计算unigram概率和困惑度
    unigram_loss = 0
    for batch in test_data:
        input_ids = batch['input_ids']
        targets = input_ids[:, 1:]
        
        for token_id in targets.flatten():
            if token_id != tokenizer.pad_token_id:
                token_id = token_id.item()
                prob = token_counts[token_id] / total_tokens
                unigram_loss += -math.log(prob)
    
    unigram_ppl = math.exp(unigram_loss / sum(token_counts.values()))
    
    # 4. 结果比较
    print("=== 困惑度基线比较 ===")
    print(f"随机基线:     {random_ppl:.2f}")  
    print(f"Unigram基线:  {unigram_ppl:.2f}")
    print(f"目标模型:     {model_ppl:.2f}")
    print(f"人类估计:     ~12")
    print()
    print("改进分析:")
    print(f"vs 随机基线:   {random_ppl/model_ppl:.1f}x 改进")
    print(f"vs Unigram:   {unigram_ppl/model_ppl:.1f}x 改进")
    print(f"vs 人类估计:   {model_ppl/12:.1f}x 差距")
    
    return {
        'model_ppl': model_ppl,
        'random_ppl': random_ppl,
        'unigram_ppl': unigram_ppl,
        'human_estimate': 12
    }
```

## 1.5 实践：MiniGPT中的概率计算

### 完整的训练损失计算

```python
# MiniGPT训练循环中的损失计算实现
class LanguageModelingLoss:
    """语言建模损失函数的完整实现"""
    
    def __init__(self, vocab_size, ignore_index=-100, label_smoothing=0.0):
        self.vocab_size = vocab_size
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing
    
    def __call__(self, logits, targets, reduction='mean'):
        """
        计算语言建模损失
        
        Args:
            logits: (batch_size, seq_len, vocab_size) 模型输出
            targets: (batch_size, seq_len) 目标token ids
            reduction: 'mean', 'sum', or 'none'
        
        Returns:
            loss: 标量损失值
        """
        
        # 展平输入
        flat_logits = logits.view(-1, self.vocab_size)  # (N, vocab_size)
        flat_targets = targets.view(-1)  # (N,)
        
        if self.label_smoothing > 0:
            # 标签平滑正则化
            loss = self._label_smoothing_loss(flat_logits, flat_targets, reduction)
        else:
            # 标准交叉熵损失
            loss = F.cross_entropy(
                flat_logits, 
                flat_targets, 
                ignore_index=self.ignore_index,
                reduction=reduction
            )
        
        return loss
    
    def _label_smoothing_loss(self, logits, targets, reduction):
        """标签平滑损失实现"""
        
        # 计算对数概率
        log_probs = F.log_softmax(logits, dim=-1)
        
        # 创建平滑标签
        smooth_targets = torch.zeros_like(log_probs)
        
        # 有效目标掩码
        valid_mask = targets != self.ignore_index
        valid_targets = targets[valid_mask]
        
        if valid_targets.numel() > 0:
            # 真实标签概率
            true_prob = 1.0 - self.label_smoothing
            # 其他标签概率
            smooth_prob = self.label_smoothing / (self.vocab_size - 1)
            
            # 填充平滑标签
            smooth_targets[valid_mask] = smooth_prob
            smooth_targets[valid_mask, valid_targets] = true_prob
            
            # 计算KL散度损失
            loss = -torch.sum(smooth_targets * log_probs, dim=-1)
            
            if reduction == 'mean':
                return loss[valid_mask].mean()
            elif reduction == 'sum':
                return loss[valid_mask].sum()
            else:
                return loss
        else:
            return torch.tensor(0.0, device=logits.device)

# 使用示例
def training_step_with_detailed_loss(model, batch, tokenizer):
    """训练步骤中的详细损失计算"""
    
    input_ids = batch['input_ids']
    attention_mask = batch.get('attention_mask', None)
    
    # 前向传播
    outputs = model(input_ids, attention_mask=attention_mask)
    
    # 准备targets（自回归：预测下一个token）
    targets = input_ids[:, 1:].contiguous()
    logits = outputs[:, :-1].contiguous()
    
    # 计算损失
    loss_fn = LanguageModelingLoss(
        vocab_size=len(tokenizer.vocab),
        ignore_index=tokenizer.pad_token_id,
        label_smoothing=0.1
    )
    
    loss = loss_fn(logits, targets)
    
    # 详细统计
    with torch.no_grad():
        # 计算困惑度
        ppl = torch.exp(loss)
        
        # 计算准确率（下一个token预测正确率）
        predictions = logits.argmax(dim=-1)
        valid_mask = targets != tokenizer.pad_token_id
        correct = (predictions == targets) & valid_mask
        accuracy = correct.sum().float() / valid_mask.sum().float()
        
        # 计算top-k准确率
        _, top5_preds = logits.topk(5, dim=-1)
        top5_correct = (top5_preds == targets.unsqueeze(-1)).any(dim=-1) & valid_mask
        top5_accuracy = top5_correct.sum().float() / valid_mask.sum().float()
        
        print(f"训练统计:")
        print(f"  损失: {loss.item():.4f}")
        print(f"  困惑度: {ppl.item():.2f}")
        print(f"  Top-1准确率: {accuracy.item():.2%}")
        print(f"  Top-5准确率: {top5_accuracy.item():.2%}")
    
    return loss, {
        'perplexity': ppl.item(),
        'accuracy': accuracy.item(),
        'top5_accuracy': top5_accuracy.item()
    }
```

## 小结与思考

本节深入探讨了语言建模的概率基础：

1. **概率链式分解**：将复杂的序列建模问题分解为可处理的条件概率预测
2. **最大似然估计**：提供了从数据中学习模型参数的理论框架
3. **信息熵理论**：揭示了语言的固有复杂度和模型的预测能力
4. **困惑度指标**：成为评估语言模型质量的标准工具

**关键洞察**：
- 语言建模本质上是概率分布的学习和近似
- 交叉熵损失等价于最大化数据似然和最小化KL散度
- 困惑度直观地反映了模型的"确信程度"
- 信息论提供了理解和分析语言模型的数学工具

**思考题**：
1. 为什么自回归分解是处理变长序列的有效方法？
2. 标签平滑如何从信息论角度改善模型性能？
3. 困惑度与人类语言理解能力的关系是什么？
4. 如何从概率角度理解模型的泛化能力？

**下一节预告**：我们将学习自回归建模与因果掩码，理解如何在Transformer中实现时间序列的因果性约束。

---

*概率论是语言建模的数学基石，它不仅告诉我们如何训练模型，更帮助我们理解语言的本质和模型的行为。* 📊