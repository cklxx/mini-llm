# 01 注意力机制数学原理

> **模型的"眼睛"：教机器学会"看重点"**

## 核心思想

注意力机制听起来很玄乎，其实就是在模拟人类看文章时的行为——读到某个词的时候，会自动回头看看前面哪些词跟它有关系。比如看到"他"的时候，会联想到前面提到的"小明"。

传统的RNN就像是只有一个小纸条传话，信息传着传着就丢了。注意力机制不一样，它让每个词都能"回头看"整个句子，直接找到跟自己最相关的信息。这就是为什么Transformer能够处理更长的序列，理解更复杂的语言关系。

**关键洞察**：
- 注意力是一种**加权平均**机制，权重由查询和键的相似度决定
- 缩放因子$\sqrt{d_k}$确保了梯度的稳定性
- Softmax归一化保证了权重的概率性质
- 掩码机制实现了不同的注意力模式（因果、双向等）

## 1.1 注意力的信息论基础

### 信息选择的数学建模

从信息论的角度看，注意力机制可以理解为一个**信息选择和聚合过程**。给定查询$q$，我们需要从一组键值对$\{(k_i, v_i)\}_{i=1}^n$中选择最相关的信息。

设查询$q$与键$k_i$的相关性为$s_i = f(q, k_i)$，则注意力权重可以表示为：

$$\alpha_i = \frac{\exp(s_i)}{\sum_{j=1}^{n} \exp(s_j)}$$

这是一个**Boltzmann分布**，其中$s_i$可以理解为"能量"，$\alpha_i$是对应的概率。

**代码实现分析**：
```python
# MiniGPT中的核心注意力计算 (src/model/transformer.py:81-102)
def scaled_dot_product_attention(self, Q, K, V, mask=None):
    """缩放点积注意力的数学实现"""
    # 1. 计算注意力分数: Q @ K^T / √d_k
    scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
    # 形状: (batch_size, n_heads, seq_len, seq_len)
    
    # 2. 应用掩码（如果有）
    if mask is not None:
        # 扩展mask维度以匹配scores
        if mask.dim() == 3:  # (batch_size, seq_len, seq_len)
            mask = mask.unsqueeze(1)  # (batch_size, 1, seq_len, seq_len)
        scores = scores.masked_fill(mask == 0, -1e9)
    
    # 3. 计算注意力权重：Boltzmann分布
    attention_weights = F.softmax(scores, dim=-1)
    attention_weights = self.dropout(attention_weights)
    
    # 4. 应用注意力权重到值：加权平均
    output = torch.matmul(attention_weights, V)
    
    return output
```

### 互信息与注意力权重

从信息论角度，注意力权重$\alpha_{ij}$可以理解为位置$i$与位置$j$之间的**互信息**的某种近似：

$$\alpha_{ij} \propto \exp(I(x_i; x_j))$$

其中$I(x_i; x_j)$是互信息。这意味着注意力机制倾向于关注那些与当前查询信息相关性最高的位置。

**信息熵的视角**：
```python
def analyze_attention_information(attention_weights):
    """分析注意力权重的信息论特性"""
    # attention_weights: (batch_size, n_heads, seq_len, seq_len)
    
    # 1. 计算每个查询位置的注意力熵
    attention_entropy = -(attention_weights * torch.log(attention_weights + 1e-8)).sum(dim=-1)
    
    # 2. 熵的解释：
    # - 高熵：注意力分散，关注多个位置
    # - 低熵：注意力集中，关注少数位置
    
    print(f"平均注意力熵: {attention_entropy.mean():.4f}")
    print(f"注意力熵标准差: {attention_entropy.std():.4f}")
    
    # 3. 有效注意力范围（熵的指数）
    effective_range = torch.exp(attention_entropy)
    print(f"平均有效注意力范围: {effective_range.mean():.2f} 个位置")
    
    return attention_entropy
```

## 1.2 缩放点积注意力的几何解释

### 向量空间中的相似度计算

缩放点积注意力的核心公式为：

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

**几何分析**：

1. **内积计算**：$QK^T$计算查询向量与键向量的内积，衡量它们的相似度
2. **几何意义**：$q_i^T k_j = \|q_i\| \|k_j\| \cos\theta_{ij}$，其中$\theta_{ij}$是两向量的夹角
3. **相似度度量**：夹角越小，内积越大，注意力权重越大

```python
def geometric_analysis_attention(Q, K):
    """注意力机制的几何分析"""
    batch_size, seq_len, d_k = Q.shape
    
    # 1. 计算向量模长
    Q_norm = torch.norm(Q, dim=-1)  # (batch_size, seq_len)
    K_norm = torch.norm(K, dim=-1)  # (batch_size, seq_len)
    
    # 2. 计算余弦相似度
    Q_normalized = F.normalize(Q, dim=-1)
    K_normalized = F.normalize(K, dim=-1)
    cosine_sim = torch.matmul(Q_normalized, K_normalized.transpose(-2, -1))
    
    # 3. 分析角度分布
    angles = torch.acos(torch.clamp(cosine_sim, -1+1e-7, 1-1e-7))
    angles_deg = angles * 180 / math.pi
    
    print(f"查询向量平均模长: {Q_norm.mean():.4f}")
    print(f"键向量平均模长: {K_norm.mean():.4f}")
    print(f"平均夹角: {angles_deg.mean():.2f}°")
    print(f"夹角标准差: {angles_deg.std():.2f}°")
    
    # 4. 注意力分数的分解
    raw_scores = torch.matmul(Q, K.transpose(-2, -1))
    magnitude_effect = Q_norm.unsqueeze(-1) * K_norm.unsqueeze(-2)
    
    return {
        'cosine_similarity': cosine_sim,
        'angles': angles_deg,
        'magnitude_effect': magnitude_effect,
        'raw_scores': raw_scores
    }
```

### 缩放因子的数学推导

**为什么需要缩放因子$\sqrt{d_k}$？**

当$d_k$较大时，内积$q^T k$的方差会增大。假设$q$和$k$的分量是独立的随机变量，均值为0，方差为1，则：

$$\text{Var}(q^T k) = \text{Var}\left(\sum_{i=1}^{d_k} q_i k_i\right) = \sum_{i=1}^{d_k} \text{Var}(q_i k_i) = d_k$$

因此，内积的标准差为$\sqrt{d_k}$。为了保持softmax输入的方差稳定，我们需要除以$\sqrt{d_k}$。

**数值稳定性分析**：
```python
def analyze_scaling_effect(d_k_values, num_samples=1000):
    """分析缩放因子对数值稳定性的影响"""
    results = {}
    
    for d_k in d_k_values:
        # 生成随机查询和键向量
        Q = torch.randn(num_samples, d_k)
        K = torch.randn(num_samples, d_k)
        
        # 计算未缩放的内积
        raw_scores = torch.matmul(Q, K.t())
        
        # 计算缩放后的内积
        scaled_scores = raw_scores / math.sqrt(d_k)
        
        # 分析统计特性
        results[d_k] = {
            'raw_mean': raw_scores.mean().item(),
            'raw_std': raw_scores.std().item(),
            'scaled_mean': scaled_scores.mean().item(),
            'scaled_std': scaled_scores.std().item(),
            'raw_max': raw_scores.max().item(),
            'scaled_max': scaled_scores.max().item()
        }
        
        print(f"d_k={d_k}:")
        print(f"  原始分数 - 均值: {results[d_k]['raw_mean']:.4f}, "
              f"标准差: {results[d_k]['raw_std']:.4f}, 最大值: {results[d_k]['raw_max']:.4f}")
        print(f"  缩放分数 - 均值: {results[d_k]['scaled_mean']:.4f}, "
              f"标准差: {results[d_k]['scaled_std']:.4f}, 最大值: {results[d_k]['scaled_max']:.4f}")
        print()
    
    return results
```

### Softmax的概率解释

Softmax函数将实数向量映射为概率分布：

$$\text{softmax}(x_i) = \frac{\exp(x_i)}{\sum_{j=1}^{n} \exp(x_j)}$$

**性质分析**：
1. **非负性**：$\text{softmax}(x_i) \geq 0$
2. **归一化**：$\sum_{i=1}^{n} \text{softmax}(x_i) = 1$
3. **单调性**：$x_i > x_j \Rightarrow \text{softmax}(x_i) > \text{softmax}(x_j)$
4. **温度参数**：$\text{softmax}(x_i/T)$中，$T$控制分布的"尖锐程度"

```python
def softmax_temperature_analysis(scores, temperatures=[0.1, 0.5, 1.0, 2.0, 5.0]):
    """分析温度参数对softmax的影响"""
    
    for T in temperatures:
        # 应用温度缩放
        scaled_scores = scores / T
        probs = F.softmax(scaled_scores, dim=-1)
        
        # 计算熵（分布的集中程度）
        entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1).mean()
        
        # 计算最大概率（分布的峰值）
        max_prob = probs.max(dim=-1)[0].mean()
        
        print(f"温度 T={T}:")
        print(f"  熵: {entropy:.4f} (越高越分散)")
        print(f"  最大概率: {max_prob:.4f} (越高越集中)")
        print()
```

## 1.3 注意力矩阵的概率性质

### 行随机矩阵的谱理论

注意力权重矩阵$\mathbf{A} \in \mathbb{R}^{n \times n}$具有以下重要性质：

1. **行随机性**：$\sum_{j=1}^{n} A_{ij} = 1, \forall i$
2. **非负性**：$A_{ij} \geq 0, \forall i,j$
3. **最大特征值**：$\lambda_{\max} = 1$
4. **Perron-Frobenius性质**：对应于特征值1的特征向量为正向量

**数学分析**：
```python
def analyze_attention_matrix_properties(attention_weights):
    """分析注意力矩阵的数学性质"""
    # attention_weights: (batch_size, n_heads, seq_len, seq_len)
    
    batch_size, n_heads, seq_len, _ = attention_weights.shape
    
    for head in range(min(n_heads, 3)):  # 分析前3个头
        A = attention_weights[0, head]  # 取第一个样本的第head个头
        
        print(f"\\n=== 注意力头 {head+1} ===")
        
        # 1. 验证行随机性
        row_sums = A.sum(dim=-1)
        print(f"行和检查 (应该全为1): 最小值={row_sums.min():.6f}, 最大值={row_sums.max():.6f}")
        
        # 2. 计算特征值
        eigenvals, eigenvecs = torch.linalg.eig(A)
        eigenvals_real = eigenvals.real
        
        # 3. 最大特征值分析
        max_eigenval = eigenvals_real.max()
        print(f"最大特征值: {max_eigenval:.6f} (理论值: 1.0)")
        
        # 4. 条件数分析
        min_eigenval = eigenvals_real[eigenvals_real > 1e-6].min()
        condition_number = max_eigenval / min_eigenval
        print(f"条件数: {condition_number:.2f}")
        
        # 5. 谱半径
        spectral_radius = eigenvals_real.abs().max()
        print(f"谱半径: {spectral_radius:.6f}")
        
        # 6. 对角线优势度（自注意力强度）
        diag_strength = torch.diag(A).mean()
        print(f"平均对角线强度: {diag_strength:.4f}")
```

### 注意力模式的分类

根据注意力权重的分布特征，我们可以将注意力模式分为几类：

1. **局部注意力**：权重集中在邻近位置
2. **全局注意力**：权重相对均匀分布
3. **稀疏注意力**：权重集中在少数几个位置
4. **对角注意力**：主要关注自身位置

```python
def classify_attention_patterns(attention_weights, threshold=0.1):
    """分类注意力模式"""
    batch_size, n_heads, seq_len, _ = attention_weights.shape
    
    patterns = {}
    
    for head in range(n_heads):
        A = attention_weights[:, head].mean(dim=0)  # 平均所有batch
        
        # 1. 局部性度量：相邻位置的注意力强度
        locality = 0
        for i in range(seq_len - 1):
            locality += A[i, i+1] + A[i+1, i]
        locality /= (2 * (seq_len - 1))
        
        # 2. 对角线强度
        diag_strength = torch.diag(A).mean()
        
        # 3. 稀疏性：高于阈值的权重比例
        sparsity = (A > threshold).float().mean()
        
        # 4. 熵（分布的均匀程度）
        entropy = -(A * torch.log(A + 1e-8)).sum(dim=-1).mean()
        
        # 5. 最大权重
        max_weight = A.max()
        
        patterns[f'head_{head}'] = {
            'locality': locality.item(),
            'diag_strength': diag_strength.item(),
            'sparsity': sparsity.item(),
            'entropy': entropy.item(),
            'max_weight': max_weight.item()
        }
        
        # 模式分类
        if diag_strength > 0.3:
            pattern_type = "对角主导"
        elif locality > 0.2:
            pattern_type = "局部关注"
        elif entropy > math.log(seq_len) * 0.8:
            pattern_type = "全局关注"
        else:
            pattern_type = "稀疏关注"
        
        print(f"头 {head}: {pattern_type}")
        print(f"  局部性: {locality:.3f}, 对角强度: {diag_strength:.3f}")
        print(f"  稀疏性: {sparsity:.3f}, 熵: {entropy:.3f}")
    
    return patterns
```

## 1.4 掩码机制的数学建模

### 因果掩码的矩阵表示

对于自回归语言模型，我们需要确保位置$i$只能看到位置$j \leq i$的信息。这通过因果掩码实现：

$$M_{ij} = \begin{cases}
0 & \text{if } i < j \\
1 & \text{if } i \geq j
\end{cases}$$

在计算注意力时，被掩码的位置会被设置为$-\infty$：

$$\text{scores}_{ij} = \begin{cases}
\frac{q_i^T k_j}{\sqrt{d_k}} & \text{if } M_{ij} = 1 \\
-\infty & \text{if } M_{ij} = 0
\end{cases}$$

**代码实现**：
```python
# MiniGPT中的因果掩码实现 (src/model/transformer.py:243-249)
def create_causal_mask(self, seq_len: int) -> torch.Tensor:
    """创建因果掩码（下三角矩阵）
    
    防止模型在预测时看到未来的token
    """
    mask = torch.tril(torch.ones(seq_len, seq_len))
    return mask  # 1表示可见，0表示掩码
```

### 不同掩码类型的数学分析

```python
def analyze_mask_effects(seq_len=10):
    """分析不同掩码类型的效果"""
    
    # 1. 因果掩码（下三角）
    causal_mask = torch.tril(torch.ones(seq_len, seq_len))
    
    # 2. 双向掩码（全1矩阵）
    bidirectional_mask = torch.ones(seq_len, seq_len)
    
    # 3. 局部掩码（带状矩阵）
    local_mask = torch.zeros(seq_len, seq_len)
    bandwidth = 3
    for i in range(seq_len):
        start = max(0, i - bandwidth)
        end = min(seq_len, i + bandwidth + 1)
        local_mask[i, start:end] = 1
    
    # 4. 稀疏掩码（随机选择）
    sparse_mask = torch.bernoulli(torch.ones(seq_len, seq_len) * 0.3)
    
    masks = {
        'Causal': causal_mask,
        'Bidirectional': bidirectional_mask, 
        'Local': local_mask,
        'Sparse': sparse_mask
    }
    
    for name, mask in masks.items():
        # 计算掩码统计
        total_positions = seq_len * seq_len
        visible_positions = mask.sum().item()
        density = visible_positions / total_positions
        
        # 计算每个位置的可见范围
        avg_visible = mask.sum(dim=-1).float().mean()
        
        print(f"{name} 掩码:")
        print(f"  密度: {density:.2%}")
        print(f"  平均可见位置数: {avg_visible:.1f}")
        print(f"  最大可见位置数: {mask.sum(dim=-1).max()}")
        print()
    
    return masks
```

### 掩码对信息流的影响

```python
def analyze_information_flow_with_mask(attention_weights, mask):
    """分析掩码对信息流的影响"""
    
    # 应用掩码
    masked_attention = attention_weights * mask.unsqueeze(0).unsqueeze(1)
    
    # 重新归一化（因为掩码改变了行和）
    row_sums = masked_attention.sum(dim=-1, keepdim=True)
    normalized_attention = masked_attention / (row_sums + 1e-8)
    
    # 分析信息流特性
    seq_len = attention_weights.size(-1)
    
    # 1. 信息传播距离
    distances = torch.arange(seq_len).unsqueeze(0) - torch.arange(seq_len).unsqueeze(1)
    distances = distances.abs().float()
    
    # 加权平均传播距离
    avg_distance = (normalized_attention.mean(dim=(0,1)) * distances).sum() / normalized_attention.mean(dim=(0,1)).sum()
    
    # 2. 信息集中度
    entropy = -(normalized_attention * torch.log(normalized_attention + 1e-8)).sum(dim=-1)
    avg_entropy = entropy.mean()
    
    # 3. 最远信息传播
    max_distance = distances[normalized_attention.mean(dim=(0,1)) > 0.01].max()
    
    print(f"信息流分析:")
    print(f"  平均传播距离: {avg_distance:.2f} 个位置")
    print(f"  平均注意力熵: {avg_entropy:.4f}")
    print(f"  最远有效传播: {max_distance:.0f} 个位置")
    
    return {
        'avg_distance': avg_distance.item(),
        'avg_entropy': avg_entropy.item(),
        'max_distance': max_distance.item()
    }
```

## 1.5 实践：MiniGPT中的注意力实现

### 完整的注意力模块解析

```python
# MiniGPT中的完整注意力实现解析
class ScaledDotProductAttention:
    """缩放点积注意力的详细实现与分析"""
    
    def __init__(self, d_k, dropout=0.1):
        self.d_k = d_k
        self.scale = math.sqrt(d_k)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, Q, K, V, mask=None, return_attention=False):
        """
        前向传播with详细注释
        
        Args:
            Q: 查询矩阵 (batch_size, n_heads, seq_len, d_k)
            K: 键矩阵 (batch_size, n_heads, seq_len, d_k)  
            V: 值矩阵 (batch_size, n_heads, seq_len, d_v)
            mask: 注意力掩码 (seq_len, seq_len)
            
        Returns:
            output: 注意力输出 (batch_size, n_heads, seq_len, d_v)
            attention_weights: 注意力权重 (可选)
        """
        
        # 1. 计算原始注意力分数
        # 数学公式: S = Q @ K^T
        scores = torch.matmul(Q, K.transpose(-2, -1))
        print(f"原始分数范围: [{scores.min():.4f}, {scores.max():.4f}]")
        
        # 2. 缩放处理
        # 数学公式: S_scaled = S / √d_k
        scaled_scores = scores / self.scale
        print(f"缩放后分数范围: [{scaled_scores.min():.4f}, {scaled_scores.max():.4f}]")
        
        # 3. 应用掩码
        if mask is not None:
            # 将掩码位置设为负无穷，softmax后接近0
            scaled_scores = scaled_scores.masked_fill(mask == 0, -1e9)
            print(f"掩码后分数范围: [{scaled_scores.min():.4f}, {scaled_scores.max():.4f}]")
        
        # 4. 计算注意力权重
        # 数学公式: A = softmax(S_scaled)
        attention_weights = F.softmax(scaled_scores, dim=-1)
        
        # 验证概率性质
        row_sums = attention_weights.sum(dim=-1)
        print(f"行和检查: 最小值={row_sums.min():.6f}, 最大值={row_sums.max():.6f}")
        
        # 5. 应用dropout（训练时）
        attention_weights = self.dropout(attention_weights)
        
        # 6. 计算加权输出
        # 数学公式: O = A @ V
        output = torch.matmul(attention_weights, V)
        
        if return_attention:
            return output, attention_weights
        return output
```

### 注意力可视化工具

```python
def visualize_attention_patterns(attention_weights, tokens, save_path=None):
    """可视化注意力模式"""
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # attention_weights: (n_heads, seq_len, seq_len)
    n_heads, seq_len, _ = attention_weights.shape
    
    # 创建子图
    fig, axes = plt.subplots(2, (n_heads + 1) // 2, figsize=(15, 8))
    if n_heads == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for head in range(n_heads):
        attn_matrix = attention_weights[head].detach().cpu().numpy()
        
        # 绘制热力图
        sns.heatmap(
            attn_matrix,
            xticklabels=tokens if len(tokens) <= 20 else False,
            yticklabels=tokens if len(tokens) <= 20 else False,
            ax=axes[head],
            cmap='Blues',
            cbar=True,
            square=True
        )
        
        axes[head].set_title(f'Head {head + 1}')
        axes[head].set_xlabel('Key Position')
        axes[head].set_ylabel('Query Position')
    
    # 隐藏多余的子图
    for i in range(n_heads, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"注意力可视化已保存到: {save_path}")
    
    plt.show()
    
    # 打印统计信息
    for head in range(n_heads):
        attn_matrix = attention_weights[head]
        
        # 计算各种统计量
        diag_strength = torch.diag(attn_matrix).mean()
        entropy = -(attn_matrix * torch.log(attn_matrix + 1e-8)).sum(dim=-1).mean()
        max_weight = attn_matrix.max()
        sparsity = (attn_matrix > 0.1).float().mean()
        
        print(f"\\nHead {head + 1} 统计:")
        print(f"  对角线强度: {diag_strength:.4f}")
        print(f"  平均熵: {entropy:.4f}")
        print(f"  最大权重: {max_weight:.4f}")
        print(f"  稀疏性 (>0.1): {sparsity:.4f}")
```

## 小结与思考

本节深入分析了注意力机制的数学原理：

1. **信息论基础**：注意力是基于互信息的信息选择机制
2. **几何解释**：缩放点积计算向量间的几何相似度
3. **概率性质**：注意力权重形成行随机矩阵的Boltzmann分布
4. **掩码机制**：通过矩阵操作实现不同的信息流约束
5. **数值稳定性**：缩放因子确保梯度传播的稳定性

**思考题**：
1. 为什么注意力机制比RNN更适合并行计算？
2. 如何从信息论角度理解不同注意力模式的作用？
3. 缩放因子的选择是否还有其他的数学依据？
4. 注意力权重的稀疏性对模型性能有什么影响？

**下一节预告**：我们将学习多头注意力的子空间分解理论，理解并行处理如何提升模型能力。

---

*注意力机制的数学之美在于它用简单的线性代数操作实现了复杂的信息选择，这正是深度学习中"简单而强大"的完美体现。* 🎯