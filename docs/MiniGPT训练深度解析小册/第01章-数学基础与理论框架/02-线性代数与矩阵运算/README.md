# 02 线性代数与矩阵运算

> **理解Transformer计算的几何本质**

## 核心思想

Transformer 的所有计算都可以归结为线性代数运算。理解这些运算的几何意义，有助于我们深入理解注意力机制、前馈网络等核心组件的工作原理。

**关键洞察**：
- 向量表示语义空间中的点
- 矩阵实现空间的线性变换
- 注意力是向量间相似度的几何计算
- 多头注意力是子空间的并行处理

## 2.1 向量空间与语义表示

### 向量空间的基本概念

一个向量空间 $V$ 是一个满足特定公理的集合，配备两种运算：
- **向量加法**：$\mathbf{u} + \mathbf{v} \in V$
- **标量乘法**：$c\mathbf{u} \in V$，其中 $c \in \mathbb{R}$

**在深度学习中的意义**：
- 词向量存在于 $\mathbb{R}^d$ 空间中
- 相似词汇在空间中距离较近
- 语义关系可以用向量运算表示

### 基与维度

给定向量空间 $V$，如果存在线性无关的向量集合 $\{\mathbf{v}_1, \mathbf{v}_2, ..., \mathbf{v}_n\}$ 使得 $V$ 中任意向量都可以表示为它们的线性组合，则称这个集合为 $V$ 的一组**基**。

**数学表达**：
$$\mathbf{v} = c_1\mathbf{v}_1 + c_2\mathbf{v}_2 + ... + c_n\mathbf{v}_n$$

**在Transformer中的应用**：
```python
# 词嵌入矩阵定义了词汇空间的基
class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        # 嵌入矩阵：每一行是一个基向量
        self.embedding = nn.Embedding(vocab_size, d_model)
        
    def forward(self, input_ids):
        # 将离散token映射到连续向量空间
        return self.embedding(input_ids)
```

### 内积与相似度

两个向量的内积定义为：
$$\langle \mathbf{u}, \mathbf{v} \rangle = \mathbf{u}^T \mathbf{v} = \sum_{i=1}^{d} u_i v_i$$

**几何意义**：
$$\langle \mathbf{u}, \mathbf{v} \rangle = \|\mathbf{u}\| \|\mathbf{v}\| \cos \theta$$

其中 $\theta$ 是两向量间的夹角。

**注意力机制的几何解释**：
```python
def scaled_dot_product_attention(Q, K, V):
    """注意力 = 查询与键的相似度加权值"""
    # Q, K, V: (batch_size, seq_len, d_k)
    
    # 1. 计算相似度：内积衡量向量夹角
    scores = torch.matmul(Q, K.transpose(-2, -1))  # (batch_size, seq_len, seq_len)
    
    # 2. 缩放：防止内积过大
    scores = scores / math.sqrt(Q.size(-1))
    
    # 3. 归一化：转为概率分布
    attention_weights = F.softmax(scores, dim=-1)
    
    # 4. 加权求和：按相似度聚合信息
    output = torch.matmul(attention_weights, V)
    
    return output, attention_weights
```

**代码解析**：注意力权重 $A_{ij}$ 的几何意义
```python
def analyze_attention_geometry(Q, K):
    """分析注意力的几何特性"""
    # 归一化查询和键向量
    Q_norm = F.normalize(Q, dim=-1)
    K_norm = F.normalize(K, dim=-1)
    
    # 余弦相似度矩阵
    cosine_sim = torch.matmul(Q_norm, K_norm.transpose(-2, -1))
    
    # 注意力分数（未归一化）
    attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(Q.size(-1))
    
    return cosine_sim, attention_scores
```

## 2.2 矩阵分解与特征值

### 特征值分解

对于方阵 $\mathbf{A} \in \mathbb{R}^{n \times n}$，如果存在非零向量 $\mathbf{v}$ 和标量 $\lambda$ 使得：
$$\mathbf{A}\mathbf{v} = \lambda\mathbf{v}$$

则称 $\lambda$ 为特征值，$\mathbf{v}$ 为对应的特征向量。

**几何意义**：特征向量是矩阵变换下方向不变的向量，特征值是缩放因子。

**谱分解**：
如果 $\mathbf{A}$ 有 $n$ 个线性无关的特征向量，则可以分解为：
$$\mathbf{A} = \mathbf{P}\mathbf{\Lambda}\mathbf{P}^{-1}$$

其中 $\mathbf{P}$ 是特征向量矩阵，$\mathbf{\Lambda}$ 是特征值对角矩阵。

**实际应用**：
```python
def analyze_weight_matrix(weight_matrix):
    """分析权重矩阵的谱特性"""
    # 计算特征值和特征向量
    eigenvals, eigenvecs = torch.linalg.eig(weight_matrix)
    
    # 特征值的分布反映了变换的主要方向
    eigenvals_real = eigenvals.real
    
    # 条件数：最大特征值/最小特征值
    condition_number = eigenvals_real.max() / eigenvals_real.min()
    
    print(f"特征值范围: [{eigenvals_real.min():.4f}, {eigenvals_real.max():.4f}]")
    print(f"条件数: {condition_number:.4f}")
    
    return eigenvals, eigenvecs
```

### 奇异值分解(SVD)

任意矩阵 $\mathbf{A} \in \mathbb{R}^{m \times n}$ 都可以分解为：
$$\mathbf{A} = \mathbf{U}\mathbf{\Sigma}\mathbf{V}^T$$

其中：
- $\mathbf{U} \in \mathbb{R}^{m \times m}$：左奇异向量矩阵
- $\mathbf{\Sigma} \in \mathbb{R}^{m \times n}$：奇异值对角矩阵
- $\mathbf{V} \in \mathbb{R}^{n \times n}$：右奇异向量矩阵

**几何解释**：任意线性变换都可以分解为旋转-缩放-旋转的复合。

**在深度学习中的应用**：

1. **权重初始化**：
```python
def xavier_uniform_init(weight):
    """基于SVD的权重初始化"""
    fan_in = weight.size(1)
    fan_out = weight.size(0)
    
    # Xavier初始化的方差
    std = math.sqrt(2.0 / (fan_in + fan_out))
    
    # 生成随机矩阵并SVD分解
    random_matrix = torch.randn_like(weight)
    U, _, V = torch.svd(random_matrix)
    
    # 构造正交初始化
    if U.shape == weight.shape:
        weight.data = U * std
    else:
        weight.data = V.t() * std
```

2. **低秩近似**：
```python
def low_rank_approximation(matrix, rank):
    """使用SVD进行低秩近似"""
    U, S, V = torch.svd(matrix)
    
    # 保留前k个奇异值
    U_k = U[:, :rank]
    S_k = S[:rank]
    V_k = V[:, :rank]
    
    # 重构矩阵
    approximation = U_k @ torch.diag(S_k) @ V_k.t()
    
    return approximation
```

## 2.3 多头注意力的子空间分解

### 子空间的概念

给定向量空间 $V$，其**子空间** $W$ 是 $V$ 的一个子集，且 $W$ 本身也构成向量空间。

**多头注意力的数学表示**：
$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, ..., \text{head}_h)W^O$$

其中每个头：
$$\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$$

**子空间分解的几何意义**：

每个注意力头在不同的子空间中操作：
- $W_i^Q, W_i^K, W_i^V \in \mathbb{R}^{d_{model} \times d_k}$
- $d_k = d_{model} / h$（通常情况）

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        # 子空间投影矩阵
        self.w_q = nn.Linear(d_model, d_model)  # 投影到查询子空间
        self.w_k = nn.Linear(d_model, d_model)  # 投影到键子空间
        self.w_v = nn.Linear(d_model, d_model)  # 投影到值子空间
        self.w_o = nn.Linear(d_model, d_model)  # 输出投影
    
    def forward(self, query, key, value):
        batch_size = query.size(0)
        
        # 1. 线性投影到各个子空间
        Q = self.w_q(query)  # (batch_size, seq_len, d_model)
        K = self.w_k(key)
        V = self.w_v(value)
        
        # 2. 重塑为多头格式：分割到不同子空间
        Q = Q.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        # 现在：(batch_size, n_heads, seq_len, d_k)
        
        # 3. 在每个子空间中计算注意力
        attention_output = self.scaled_dot_product_attention(Q, K, V)
        
        # 4. 拼接所有子空间的结果
        attention_output = attention_output.transpose(1, 2).contiguous()
        attention_output = attention_output.view(batch_size, -1, self.d_model)
        
        # 5. 最终的线性变换
        output = self.w_o(attention_output)
        
        return output
```

**子空间独立性分析**：
```python
def analyze_subspace_independence(multi_head_attn):
    """分析多头注意力的子空间独立性"""
    # 获取投影矩阵
    W_q = multi_head_attn.w_q.weight  # (d_model, d_model)
    W_k = multi_head_attn.w_k.weight
    W_v = multi_head_attn.w_v.weight
    
    # 重塑为多头格式
    n_heads = multi_head_attn.n_heads
    d_k = multi_head_attn.d_k
    
    W_q_heads = W_q.view(n_heads, d_k, -1)  # (n_heads, d_k, d_model)
    W_k_heads = W_k.view(n_heads, d_k, -1)
    
    # 计算不同头之间的相关性
    correlations = []
    for i in range(n_heads):
        for j in range(i+1, n_heads):
            # 计算投影矩阵的余弦相似度
            qi_flat = W_q_heads[i].flatten()
            qj_flat = W_q_heads[j].flatten()
            
            correlation = F.cosine_similarity(qi_flat, qj_flat, dim=0)
            correlations.append(correlation.item())
    
    avg_correlation = sum(correlations) / len(correlations)
    print(f"平均头间相关性: {avg_correlation:.4f}")
    
    return correlations
```

## 2.4 矩阵微分与梯度计算

### 标量对向量的导数

设 $f: \mathbb{R}^n \rightarrow \mathbb{R}$，则梯度定义为：
$$\nabla_\mathbf{x} f = \frac{\partial f}{\partial \mathbf{x}} = \begin{bmatrix} \frac{\partial f}{\partial x_1} \\ \vdots \\ \frac{\partial f}{\partial x_n} \end{bmatrix}$$

### 标量对矩阵的导数

设 $f: \mathbb{R}^{m \times n} \rightarrow \mathbb{R}$，则：
$$\frac{\partial f}{\partial \mathbf{A}} = \begin{bmatrix} \frac{\partial f}{\partial A_{11}} & \cdots & \frac{\partial f}{\partial A_{1n}} \\ \vdots & \ddots & \vdots \\ \frac{\partial f}{\partial A_{m1}} & \cdots & \frac{\partial f}{\partial A_{mn}} \end{bmatrix}$$

### 重要的矩阵微分公式

1. **线性函数**：$\frac{\partial (\mathbf{a}^T\mathbf{x})}{\partial \mathbf{x}} = \mathbf{a}$

2. **二次型**：$\frac{\partial (\mathbf{x}^T\mathbf{A}\mathbf{x})}{\partial \mathbf{x}} = \mathbf{A}\mathbf{x} + \mathbf{A}^T\mathbf{x}$

3. **矩阵乘法**：$\frac{\partial \text{tr}(\mathbf{A}\mathbf{B})}{\partial \mathbf{A}} = \mathbf{B}^T$

**注意力机制的梯度推导**：

设注意力分数 $S = QK^T$，注意力权重 $A = \text{softmax}(S)$，输出 $O = AV$。

对于损失函数 $L$，我们需要计算：
$$\frac{\partial L}{\partial Q}, \frac{\partial L}{\partial K}, \frac{\partial L}{\partial V}$$

**反向传播的链式法则**：
```python
def attention_backward(grad_output, Q, K, V, attention_weights):
    """注意力机制的反向传播"""
    # grad_output: (batch_size, seq_len, d_v)
    # attention_weights: (batch_size, seq_len, seq_len)
    
    # 1. 对V的梯度
    grad_V = torch.matmul(attention_weights.transpose(-2, -1), grad_output)
    
    # 2. 对attention_weights的梯度
    grad_attn = torch.matmul(grad_output, V.transpose(-2, -1))
    
    # 3. softmax的反向传播
    grad_scores = attention_weights * (grad_attn - 
        (grad_attn * attention_weights).sum(dim=-1, keepdim=True))
    
    # 4. 对Q和K的梯度
    d_k = Q.size(-1)
    grad_Q = torch.matmul(grad_scores, K) / math.sqrt(d_k)
    grad_K = torch.matmul(grad_scores.transpose(-2, -1), Q) / math.sqrt(d_k)
    
    return grad_Q, grad_K, grad_V
```

## 2.5 实践：MiniGPT中的线性代数运算

### 权重矩阵的几何分析

```python
def analyze_transformer_weights(model):
    """分析Transformer权重矩阵的几何特性"""
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            weight = module.weight.data  # (out_features, in_features)
            
            # 1. 奇异值分析
            U, S, V = torch.svd(weight)
            
            # 2. 条件数
            condition_number = S.max() / S.min()
            
            # 3. 有效秩（大于阈值的奇异值个数）
            threshold = 0.01 * S.max()
            effective_rank = (S > threshold).sum().item()
            
            # 4. 谱范数（最大奇异值）
            spectral_norm = S.max()
            
            print(f"Layer: {name}")
            print(f"  Shape: {weight.shape}")
            print(f"  Condition Number: {condition_number:.2f}")
            print(f"  Effective Rank: {effective_rank}/{min(weight.shape)}")
            print(f"  Spectral Norm: {spectral_norm:.4f}")
            print()
```

### 注意力权重的几何可视化

```python
def visualize_attention_geometry(attention_weights, tokens):
    """可视化注意力权重的几何结构"""
    # attention_weights: (n_heads, seq_len, seq_len)
    
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    n_heads = attention_weights.size(0)
    
    fig, axes = plt.subplots(2, n_heads//2, figsize=(15, 6))
    axes = axes.flatten()
    
    for head in range(n_heads):
        attn_matrix = attention_weights[head].cpu().numpy()
        
        # 绘制注意力热力图
        sns.heatmap(attn_matrix, 
                   xticklabels=tokens, 
                   yticklabels=tokens,
                   ax=axes[head],
                   cmap='Blues')
        axes[head].set_title(f'Head {head+1}')
    
    plt.tight_layout()
    plt.show()
    
    # 分析注意力的几何特性
    for head in range(n_heads):
        attn_matrix = attention_weights[head]
        
        # 1. 对角线强度（自注意力）
        diag_strength = torch.diag(attn_matrix).mean()
        
        # 2. 局部性（相邻位置的注意力强度）
        locality = 0
        for i in range(attn_matrix.size(0)-1):
            locality += attn_matrix[i, i+1] + attn_matrix[i+1, i]
        locality /= (2 * (attn_matrix.size(0) - 1))
        
        # 3. 熵（注意力分布的集中程度）
        attn_entropy = -(attn_matrix * torch.log(attn_matrix + 1e-8)).sum(dim=-1).mean()
        
        print(f"Head {head+1}:")
        print(f"  Self-attention strength: {diag_strength:.4f}")
        print(f"  Locality: {locality:.4f}")
        print(f"  Attention entropy: {attn_entropy:.4f}")
```

## 小结与思考

本节介绍了支撑Transformer计算的线性代数基础：

1. **向量空间**为语义表示提供了几何框架
2. **内积**实现了相似度的几何计算
3. **矩阵分解**揭示了权重矩阵的内在结构
4. **子空间分解**是多头注意力的几何本质
5. **矩阵微分**是反向传播算法的数学基础

**思考题**：
1. 为什么多头注意力比单头注意力更有效？从子空间的角度分析。
2. 如何从奇异值分解的角度理解权重矩阵的表达能力？
3. 注意力权重矩阵的几何性质反映了什么语言学现象？

**下一节预告**：我们将学习优化理论，理解训练算法的数学原理。

---

*线性代数为深度学习提供了计算的语言，而几何直觉让我们理解了计算的意义。* 📐