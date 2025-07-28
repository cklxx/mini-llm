# 02 多头注意力子空间分解

> **从单一空间到多维语义的并行建模**

## 核心思想

多头注意力是 Transformer 架构的关键创新，它将单个注意力机制扩展为多个并行的"注意力头"。每个头在不同的表征子空间中独立操作，最后将结果拼接融合。这种设计让模型能够同时关注不同类型的语义关系。

**核心洞察**：
- 每个注意力头关注不同的**语义维度**（语法、语义、位置等）
- 子空间分解提供了**表征多样性**，避免单一视角的局限
- 并行计算实现了**计算效率**的提升
- 参数共享与独立性的平衡实现了**模型容量**的优化

**数学表达**：
$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, ..., \text{head}_h)W^O$$

其中：
$$\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$$

## 2.1 子空间分解的数学框架

### 线性子空间的定义

给定向量空间 $\mathbb{R}^d$，**子空间** $S \subseteq \mathbb{R}^d$ 是一个满足以下条件的集合：
1. **零向量**：$\mathbf{0} \in S$
2. **加法封闭**：$\mathbf{u}, \mathbf{v} \in S \Rightarrow \mathbf{u} + \mathbf{v} \in S$
3. **标量乘法封闭**：$\mathbf{u} \in S, c \in \mathbb{R} \Rightarrow c\mathbf{u} \in S$

在多头注意力中，每个投影矩阵 $W_i^Q, W_i^K, W_i^V \in \mathbb{R}^{d \times d_k}$ 定义了一个 $d_k$ 维子空间。

**子空间投影的数学性质**：
```python
def analyze_subspace_properties(W_q, W_k, W_v):
    """分析子空间投影矩阵的数学性质"""
    
    # W_q, W_k, W_v: (d_model, d_k)
    d_model, d_k = W_q.shape
    
    print(f"原始空间维度: {d_model}")
    print(f"子空间维度: {d_k}")
    print(f"维度压缩比: {d_k/d_model:.2%}")
    
    # 1. 分析投影矩阵的秩
    rank_q = torch.linalg.matrix_rank(W_q)
    rank_k = torch.linalg.matrix_rank(W_k)
    rank_v = torch.linalg.matrix_rank(W_v)
    
    print(f"\\n投影矩阵的秩:")
    print(f"  W^Q: {rank_q}/{d_k} (理论最大: {min(d_model, d_k)})")
    print(f"  W^K: {rank_k}/{d_k}")
    print(f"  W^V: {rank_v}/{d_k}")
    
    # 2. 计算投影矩阵的奇异值
    U_q, S_q, V_q = torch.svd(W_q)
    U_k, S_k, V_k = torch.svd(W_k)
    U_v, S_v, V_v = torch.svd(W_v)
    
    print(f"\\n奇异值分析:")
    print(f"  W^Q最大奇异值: {S_q.max():.4f}, 最小奇异值: {S_q.min():.4f}")
    print(f"  W^K最大奇异值: {S_k.max():.4f}, 最小奇异值: {S_k.min():.4f}")
    print(f"  W^V最大奇异值: {S_v.max():.4f}, 最小奇异值: {S_v.min():.4f}")
    
    # 3. 条件数分析（数值稳定性）
    cond_q = S_q.max() / S_q.min()
    cond_k = S_k.max() / S_k.min()
    cond_v = S_v.max() / S_v.min()
    
    print(f"\\n条件数 (数值稳定性指标):")
    print(f"  W^Q: {cond_q:.2f}")
    print(f"  W^K: {cond_k:.2f}")
    print(f"  W^V: {cond_v:.2f}")
    
    return {
        'ranks': (rank_q, rank_k, rank_v),
        'singular_values': (S_q, S_k, S_v),
        'condition_numbers': (cond_q, cond_k, cond_v)
    }
```

### 直和分解的理论基础

理想情况下，多头注意力的子空间应该形成**直和分解**：

$$\mathbb{R}^d = S_1 \oplus S_2 \oplus ... \oplus S_h$$

其中 $S_i \cap S_j = \{\mathbf{0}\}, \forall i \neq j$。

这意味着不同的头关注完全不同的特征维度，没有信息冗余。

**实际分析**：
```python
def analyze_subspace_orthogonality(multi_head_attention):
    """分析多头注意力子空间的正交性"""
    
    # 获取所有头的投影矩阵
    W_q = multi_head_attention.w_q.weight.data  # (d_model, d_model)
    W_k = multi_head_attention.w_k.weight.data
    W_v = multi_head_attention.w_v.weight.data
    
    n_heads = multi_head_attention.n_heads
    d_k = multi_head_attention.d_k
    
    # 重塑为多头格式: (n_heads, d_k, d_model)
    W_q_heads = W_q.t().view(n_heads, d_k, -1)
    W_k_heads = W_k.t().view(n_heads, d_k, -1)
    W_v_heads = W_v.t().view(n_heads, d_k, -1)
    
    # 计算头之间的相似度矩阵
    similarity_matrix = torch.zeros(n_heads, n_heads)
    
    for i in range(n_heads):
        for j in range(n_heads):
            if i != j:
                # 计算两个子空间的Grassmann距离
                # 这里简化为投影矩阵的Frobenius内积
                sim_q = F.cosine_similarity(
                    W_q_heads[i].flatten(), 
                    W_q_heads[j].flatten(), 
                    dim=0
                )
                similarity_matrix[i, j] = sim_q.abs()
    
    # 分析结果
    avg_similarity = similarity_matrix.sum() / (n_heads * (n_heads - 1))
    max_similarity = similarity_matrix.max()
    
    print(f"子空间相似度分析:")
    print(f"  平均相似度: {avg_similarity:.4f} (越小越正交)")
    print(f"  最大相似度: {max_similarity:.4f}")
    
    # 理想情况下应该接近0（完全正交）
    if avg_similarity < 0.1:
        print("  ✓ 子空间基本正交，信息冗余较少")
    elif avg_similarity < 0.3:
        print("  ⚠ 子空间部分重叠，存在一定冗余")  
    else:
        print("  ❌ 子空间高度重叠，冗余严重")
    
    return similarity_matrix
```

## 2.2 多头并行的几何意义

### 不同语义维度的独立建模

多头注意力的核心思想是让不同的头学习不同类型的依赖关系：

- **句法关系**：主谓宾、定状补等语法结构
- **语义关系**：同义词、反义词、上下位关系
- **位置关系**：相对位置、距离信息
- **话题关系**：主题一致性、话题转换

**代码实现解析**：
```python
# MiniGPT中的多头注意力实现 (src/model/transformer.py:62-79)
def forward(self, query, key, value, mask=None):
    batch_size, seq_len, d_model = query.size()
    
    # 1. 线性投影到各个子空间
    Q = self.w_q(query)  # (batch_size, seq_len, d_model)
    K = self.w_k(key)    # (batch_size, seq_len, d_model)  
    V = self.w_v(value)  # (batch_size, seq_len, d_model)
    
    # 2. 重塑为多头格式：分割到不同子空间
    Q = Q.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
    K = K.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
    V = V.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
    # 现在形状: (batch_size, n_heads, seq_len, d_k)
    
    # 3. 在每个子空间中独立计算注意力
    attention_output = self.scaled_dot_product_attention(Q, K, V, mask)
    
    # 4. 拼接所有子空间的结果
    attention_output = attention_output.transpose(1, 2).contiguous()
    attention_output = attention_output.view(batch_size, seq_len, d_model)
    
    # 5. 最终的线性变换（信息融合）
    output = self.w_o(attention_output)
    
    return output
```

### 维度变换的几何解释

多头注意力涉及一系列维度变换，每一步都有明确的几何意义：

```python
def trace_dimension_transformations(batch_size=2, seq_len=10, d_model=512, n_heads=8):
    """追踪多头注意力的维度变换过程"""
    
    d_k = d_model // n_heads
    print(f"多头注意力维度变换追踪:")
    print(f"  批次大小: {batch_size}")
    print(f"  序列长度: {seq_len}")  
    print(f"  模型维度: {d_model}")
    print(f"  注意力头数: {n_heads}")
    print(f"  每头维度: {d_k}")
    print()
    
    # 模拟输入
    x = torch.randn(batch_size, seq_len, d_model)
    print(f"1. 输入: {x.shape}")
    print(f"   几何意义: {batch_size}个样本，每个包含{seq_len}个{d_model}维向量")
    
    # 线性投影
    W_q = torch.randn(d_model, d_model)
    Q = torch.matmul(x, W_q)
    print(f"\\n2. 查询投影: {Q.shape}")
    print(f"   几何意义: 将输入向量投影到查询空间")
    
    # 重塑为多头格式
    Q_reshaped = Q.view(batch_size, seq_len, n_heads, d_k)
    print(f"\\n3. 多头重塑: {Q_reshaped.shape}")
    print(f"   几何意义: 将{d_model}维向量分割为{n_heads}个{d_k}维子向量")
    
    # 转置以便并行计算
    Q_transposed = Q_reshaped.transpose(1, 2)
    print(f"\\n4. 转置: {Q_transposed.shape}")
    print(f"   几何意义: 重排维度以支持并行的头计算")
    
    # 注意力计算（简化）
    K_transposed = Q_transposed  # 简化，实际中K来自不同投影
    attention_scores = torch.matmul(Q_transposed, K_transposed.transpose(-2, -1))
    print(f"\\n5. 注意力分数: {attention_scores.shape}")
    print(f"   几何意义: 每个头计算{seq_len}×{seq_len}的相似度矩阵")
    
    # 注意力权重和输出
    attention_weights = F.softmax(attention_scores / math.sqrt(d_k), dim=-1)
    V_transposed = Q_transposed  # 简化
    attention_output = torch.matmul(attention_weights, V_transposed)
    print(f"\\n6. 注意力输出: {attention_output.shape}")
    print(f"   几何意义: 每个头产生{seq_len}个{d_k}维加权向量")
    
    # 拼接
    attention_concat = attention_output.transpose(1, 2).contiguous()
    attention_concat = attention_concat.view(batch_size, seq_len, d_model)
    print(f"\\n7. 拼接结果: {attention_concat.shape}")
    print(f"   几何意义: 将{n_heads}个{d_k}维向量拼接回{d_model}维")
    
    # 输出投影
    W_o = torch.randn(d_model, d_model)
    final_output = torch.matmul(attention_concat, W_o)
    print(f"\\n8. 最终输出: {final_output.shape}")
    print(f"   几何意义: 融合多头信息，回到原始空间维度")
```

## 2.3 参数共享与表征多样性

### 投影矩阵的初始化策略

不同头的投影矩阵需要合理初始化以确保表征多样性：

```python
def analyze_initialization_strategies():
    """分析不同初始化策略对多头表征多样性的影响"""
    
    d_model, n_heads = 512, 8
    d_k = d_model // n_heads
    
    strategies = {
        'Xavier Uniform': lambda: nn.init.xavier_uniform_(torch.empty(d_model, d_model)),
        'Xavier Normal': lambda: nn.init.xavier_normal_(torch.empty(d_model, d_model)),
        'Kaiming Uniform': lambda: nn.init.kaiming_uniform_(torch.empty(d_model, d_model)),
        'Orthogonal': lambda: nn.init.orthogonal_(torch.empty(d_model, d_model)),
        'Random Normal': lambda: torch.randn(d_model, d_model) * 0.02
    }
    
    for name, init_fn in strategies.items():
        print(f"\\n=== {name} 初始化 ===")
        
        # 初始化投影矩阵
        W_q = init_fn()
        
        # 分析多头相似度
        W_q_heads = W_q.view(d_model, n_heads, d_k)
        
        similarities = []
        for i in range(n_heads):
            for j in range(i+1, n_heads):
                # 计算两个头的余弦相似度
                head_i = W_q_heads[:, i, :].flatten()
                head_j = W_q_heads[:, j, :].flatten()
                sim = F.cosine_similarity(head_i, head_j, dim=0)
                similarities.append(sim.abs().item())
        
        avg_sim = sum(similarities) / len(similarities)
        max_sim = max(similarities)
        
        print(f"  平均头间相似度: {avg_sim:.4f}")
        print(f"  最大头间相似度: {max_sim:.4f}")
        
        # 分析权重分布
        print(f"  权重均值: {W_q.mean():.6f}")
        print(f"  权重标准差: {W_q.std():.6f}")
        
        # 谱分析
        U, S, V = torch.svd(W_q)
        print(f"  条件数: {S.max()/S.min():.2f}")
```

### 表征多样性的量化指标

```python
def measure_representation_diversity(multi_head_attention, input_data):
    """量化多头注意力的表征多样性"""
    
    model = multi_head_attention
    model.eval()
    
    with torch.no_grad():
        # 获取所有头的输出
        batch_size, seq_len, d_model = input_data.shape
        
        # 前向传播到注意力计算
        Q = model.w_q(input_data)
        K = model.w_k(input_data)
        V = model.w_v(input_data)
        
        # 重塑为多头格式
        Q = Q.view(batch_size, seq_len, model.n_heads, model.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, model.n_heads, model.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, model.n_heads, model.d_k).transpose(1, 2)
        
        # 计算每个头的注意力
        head_outputs = []
        head_attentions = []
        
        for head in range(model.n_heads):
            q_h = Q[:, head:head+1]  # 保持维度
            k_h = K[:, head:head+1]
            v_h = V[:, head:head+1]
            
            # 计算注意力
            scores = torch.matmul(q_h, k_h.transpose(-2, -1)) / math.sqrt(model.d_k)
            attention = F.softmax(scores, dim=-1)
            output = torch.matmul(attention, v_h)
            
            head_outputs.append(output.squeeze(1))  # (batch_size, seq_len, d_k)
            head_attentions.append(attention.squeeze(1))  # (batch_size, seq_len, seq_len)
    
    # 分析表征多样性
    print("=== 表征多样性分析 ===")
    
    # 1. 输出相似度分析
    output_similarities = []
    for i in range(model.n_heads):
        for j in range(i+1, model.n_heads):
            # 计算两个头输出的相似度
            out_i = head_outputs[i].flatten()
            out_j = head_outputs[j].flatten()
            sim = F.cosine_similarity(out_i, out_j, dim=0)
            output_similarities.append(sim.item())
    
    avg_output_sim = sum(output_similarities) / len(output_similarities)
    print(f"平均输出相似度: {avg_output_sim:.4f}")
    
    # 2. 注意力模式相似度
    attention_similarities = []
    for i in range(model.n_heads):
        for j in range(i+1, model.n_heads):
            att_i = head_attentions[i].flatten()
            att_j = head_attentions[j].flatten()
            sim = F.cosine_similarity(att_i, att_j, dim=0)
            attention_similarities.append(sim.item())
    
    avg_attention_sim = sum(attention_similarities) / len(attention_similarities)
    print(f"平均注意力模式相似度: {avg_attention_sim:.4f}")
    
    # 3. 信息熵分析
    entropies = []
    for head in range(model.n_heads):
        attention = head_attentions[head]
        entropy = -(attention * torch.log(attention + 1e-8)).sum(dim=-1).mean()
        entropies.append(entropy.item())
    
    entropy_diversity = torch.tensor(entropies).std().item()
    print(f"注意力熵多样性: {entropy_diversity:.4f}")
    
    return {
        'output_similarity': avg_output_sim,
        'attention_similarity': avg_attention_sim,
        'entropy_diversity': entropy_diversity,
        'head_entropies': entropies
    }
```

## 2.4 头间信息融合的理论

### 输出投影的作用机制

输出投影矩阵 $W^O$ 负责融合来自不同头的信息：

$$\text{Output} = \text{Concat}(\text{head}_1, ..., \text{head}_h) W^O$$

**融合策略分析**：
```python
def analyze_output_projection_fusion(W_o, n_heads, d_k):
    """分析输出投影矩阵的信息融合机制"""
    
    d_model = n_heads * d_k
    
    # 将输出投影矩阵按头分块
    W_o_blocks = W_o.view(d_model, n_heads, d_k)
    
    print(f"输出投影矩阵分析:")
    print(f"  形状: {W_o.shape}")
    print(f"  分块后: {W_o_blocks.shape}")
    
    # 分析每个头的贡献权重
    head_contributions = []
    for head in range(n_heads):
        # 计算该头对应的投影块的Frobenius范数
        head_weight = torch.norm(W_o_blocks[:, head, :], 'fro')
        head_contributions.append(head_weight.item())
    
    head_contributions = torch.tensor(head_contributions)
    
    print(f"\\n各头贡献权重:")
    for head in range(n_heads):
        print(f"  头 {head+1}: {head_contributions[head]:.4f}")
    
    # 分析贡献的均衡性
    contribution_std = head_contributions.std()
    contribution_cv = contribution_std / head_contributions.mean()  # 变异系数
    
    print(f"\\n贡献均衡性:")
    print(f"  标准差: {contribution_std:.4f}")
    print(f"  变异系数: {contribution_cv:.4f}")
    
    if contribution_cv < 0.2:
        print("  ✓ 各头贡献均衡")
    elif contribution_cv < 0.5:
        print("  ⚠ 各头贡献略有差异")
    else:
        print("  ❌ 各头贡献差异较大，可能存在头冗余")
    
    # 分析头间交互
    interaction_matrix = torch.zeros(n_heads, n_heads)
    for i in range(n_heads):
        for j in range(n_heads):
            if i != j:
                # 计算两个头对应投影块的内积
                block_i = W_o_blocks[:, i, :].flatten()
                block_j = W_o_blocks[:, j, :].flatten() 
                interaction = torch.dot(block_i, block_j).abs()
                interaction_matrix[i, j] = interaction
    
    avg_interaction = interaction_matrix.sum() / (n_heads * (n_heads - 1))
    print(f"\\n头间交互强度: {avg_interaction:.4f}")
    
    return {
        'head_contributions': head_contributions,
        'contribution_cv': contribution_cv,
        'interaction_matrix': interaction_matrix,
        'avg_interaction': avg_interaction
    }
```

### 残差连接与多头融合

多头注意力的输出还需要与输入进行残差连接：

$$\text{Output} = \text{LayerNorm}(\text{Input} + \text{MultiHead}(\text{Input}))$$

**残差连接的作用**：
```python
def analyze_residual_contribution(input_tensor, multihead_output):
    """分析残差连接中原输入和多头输出的相对贡献"""
    
    # 计算各部分的范数
    input_norm = torch.norm(input_tensor, dim=-1).mean()
    output_norm = torch.norm(multihead_output, dim=-1).mean()
    
    # 残差连接后的结果
    residual_sum = input_tensor + multihead_output
    residual_norm = torch.norm(residual_sum, dim=-1).mean()
    
    print(f"残差连接分析:")
    print(f"  输入范数: {input_norm:.4f}")
    print(f"  多头输出范数: {output_norm:.4f}")
    print(f"  残差和范数: {residual_norm:.4f}")
    
    # 计算相对贡献
    input_contribution = input_norm / residual_norm
    output_contribution = output_norm / residual_norm
    
    print(f"\\n相对贡献:")
    print(f"  输入贡献: {input_contribution:.4f} ({input_contribution*100:.1f}%)")
    print(f"  多头贡献: {output_contribution:.4f} ({output_contribution*100:.1f}%)")
    
    # 分析方向相似性
    input_flat = input_tensor.flatten()
    output_flat = multihead_output.flatten()
    
    direction_similarity = F.cosine_similarity(input_flat, output_flat, dim=0)
    print(f"\\n方向相似性: {direction_similarity:.4f}")
    
    if direction_similarity > 0.8:
        print("  多头输出与输入方向高度相似，可能存在退化")
    elif direction_similarity > 0.3:
        print("  多头输出对输入进行了适度修改")
    else:
        print("  多头输出显著改变了输入的方向")
    
    return {
        'input_norm': input_norm.item(),
        'output_norm': output_norm.item(),
        'input_contribution': input_contribution.item(),
        'output_contribution': output_contribution.item(),
        'direction_similarity': direction_similarity.item()
    }
```

## 2.5 实践：MiniGPT中的多头注意力优化

### 高效的多头计算实现

```python
class OptimizedMultiHeadAttention(nn.Module):
    """优化的多头注意力实现，包含详细分析"""
    
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        # 使用单个矩阵进行投影，然后分割（更高效）
        self.w_qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)
        
        # 用于分析的缓存
        self.attention_weights = None
        self.head_outputs = None
    
    def forward(self, x, mask=None, return_attention=False):
        batch_size, seq_len, d_model = x.size()
        
        # 1. 一次性计算Q, K, V投影
        qkv = self.w_qkv(x)  # (batch_size, seq_len, 3*d_model)
        qkv = qkv.view(batch_size, seq_len, 3, self.n_heads, self.d_k)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, batch_size, n_heads, seq_len, d_k)
        
        Q, K, V = qkv[0], qkv[1], qkv[2]
        
        # 2. 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # 3. 计算注意力权重
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # 缓存用于分析
        self.attention_weights = attention_weights.detach()
        
        # 4. 应用注意力权重
        attention_output = torch.matmul(attention_weights, V)
        
        # 缓存各头输出用于分析
        self.head_outputs = attention_output.detach()
        
        # 5. 拼接并投影
        attention_output = attention_output.transpose(1, 2).contiguous()
        attention_output = attention_output.view(batch_size, seq_len, d_model)
        
        output = self.w_o(attention_output)
        
        if return_attention:
            return output, attention_weights
        return output
    
    def analyze_heads(self, head_names=None):
        """分析各个注意力头的特性"""
        if self.attention_weights is None:
            print("请先进行一次前向传播")
            return
        
        attention = self.attention_weights  # (batch_size, n_heads, seq_len, seq_len)
        head_outputs = self.head_outputs    # (batch_size, n_heads, seq_len, d_k)
        
        batch_size, n_heads, seq_len, _ = attention.shape
        
        # 平均所有batch的结果
        avg_attention = attention.mean(dim=0)  # (n_heads, seq_len, seq_len)
        avg_outputs = head_outputs.mean(dim=0)  # (n_heads, seq_len, d_k)
        
        print(f"=== 多头注意力分析 ===")
        print(f"批次大小: {batch_size}, 头数: {n_heads}, 序列长度: {seq_len}")
        
        for head in range(n_heads):
            head_name = head_names[head] if head_names else f"Head-{head+1}"
            print(f"\\n{head_name}:")
            
            # 注意力特性
            attn = avg_attention[head]
            
            # 1. 对角线强度（自注意力）
            diag_strength = torch.diag(attn).mean()
            
            # 2. 局部性（相邻位置关注度）
            locality = 0
            if seq_len > 1:
                for i in range(seq_len - 1):
                    locality += attn[i, i+1] + attn[i+1, i]
                locality /= (2 * (seq_len - 1))
            
            # 3. 注意力熵
            entropy = -(attn * torch.log(attn + 1e-8)).sum(dim=-1).mean()
            
            # 4. 最大注意力权重
            max_attention = attn.max()
            
            # 5. 输出激活统计
            output = avg_outputs[head]
            output_mean = output.mean()
            output_std = output.std()
            
            print(f"  自注意力强度: {diag_strength:.4f}")
            print(f"  局部关注度: {locality:.4f}")
            print(f"  注意力熵: {entropy:.4f}")
            print(f"  最大权重: {max_attention:.4f}")
            print(f"  输出统计: 均值={output_mean:.4f}, 标准差={output_std:.4f}")
```

### 多头注意力的可视化分析

```python
def comprehensive_multihead_visualization(model, input_data, tokens, save_dir="./attention_analysis"):
    """多头注意力的综合可视化分析"""
    
    import matplotlib.pyplot as plt
    import seaborn as sns
    import os
    
    os.makedirs(save_dir, exist_ok=True)
    
    model.eval()
    with torch.no_grad():
        # 获取注意力权重
        output, attention_weights = model(input_data, return_attention=True)
    
    # attention_weights: (batch_size, n_heads, seq_len, seq_len)
    batch_size, n_heads, seq_len, _ = attention_weights.shape
    
    # 平均所有batch
    avg_attention = attention_weights.mean(dim=0)  # (n_heads, seq_len, seq_len)
    
    # 1. 绘制所有头的注意力模式
    fig, axes = plt.subplots(2, n_heads//2, figsize=(20, 8))
    axes = axes.flatten()
    
    for head in range(n_heads):
        attn_matrix = avg_attention[head].cpu().numpy()
        
        sns.heatmap(
            attn_matrix,
            ax=axes[head],
            cmap='Blues',
            square=True,
            cbar=True,
            xticklabels=tokens if len(tokens) <= 10 else False,
            yticklabels=tokens if len(tokens) <= 10 else False
        )
        axes[head].set_title(f'Head {head+1}')
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/all_heads_attention.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # 2. 注意力模式聚类分析
    attention_flat = avg_attention.view(n_heads, -1)  # (n_heads, seq_len*seq_len)
    
    # 计算头间相似度矩阵
    similarity_matrix = F.cosine_similarity(
        attention_flat.unsqueeze(1), 
        attention_flat.unsqueeze(0), 
        dim=2
    )
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        similarity_matrix.cpu().numpy(),
        annot=True,
        fmt='.3f',
        cmap='coolwarm',
        center=0,
        square=True,
        xticklabels=[f'H{i+1}' for i in range(n_heads)],
        yticklabels=[f'H{i+1}' for i in range(n_heads)]
    )
    plt.title('头间注意力模式相似度')
    plt.tight_layout()
    plt.savefig(f"{save_dir}/head_similarity.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # 3. 注意力统计分析
    stats = []
    for head in range(n_heads):
        attn = avg_attention[head]
        
        # 计算各种统计量
        diag_strength = torch.diag(attn).mean().item()
        entropy = -(attn * torch.log(attn + 1e-8)).sum(dim=-1).mean().item()
        max_weight = attn.max().item()
        
        # 局部性度量
        locality = 0
        if seq_len > 1:
            for i in range(seq_len - 1):
                locality += attn[i, i+1] + attn[i+1, i]
            locality = locality.item() / (2 * (seq_len - 1))
        
        stats.append({
            'head': head + 1,
            'diag_strength': diag_strength,
            'entropy': entropy,
            'max_weight': max_weight,
            'locality': locality
        })
    
    # 绘制统计图表
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    heads = [s['head'] for s in stats]
    
    # 对角线强度
    axes[0,0].bar(heads, [s['diag_strength'] for s in stats])
    axes[0,0].set_title('自注意力强度')
    axes[0,0].set_xlabel('头编号')
    axes[0,0].set_ylabel('强度')
    
    # 注意力熵
    axes[0,1].bar(heads, [s['entropy'] for s in stats])
    axes[0,1].set_title('注意力熵')
    axes[0,1].set_xlabel('头编号')
    axes[0,1].set_ylabel('熵值')
    
    # 最大权重
    axes[1,0].bar(heads, [s['max_weight'] for s in stats])
    axes[1,0].set_title('最大注意力权重')
    axes[1,0].set_xlabel('头编号')
    axes[1,0].set_ylabel('权重')
    
    # 局部性
    axes[1,1].bar(heads, [s['locality'] for s in stats])
    axes[1,1].set_title('局部关注度')
    axes[1,1].set_xlabel('头编号')
    axes[1,1].set_ylabel('局部性')
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/attention_statistics.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # 打印详细统计
    print("\\n=== 多头注意力详细统计 ===")
    for stat in stats:
        print(f"头 {stat['head']}:")
        print(f"  自注意力: {stat['diag_strength']:.4f}")
        print(f"  注意力熵: {stat['entropy']:.4f}")
        print(f"  最大权重: {stat['max_weight']:.4f}")
        print(f"  局部性: {stat['locality']:.4f}")
        print()
    
    return stats, similarity_matrix
```

## 小结与思考

本节深入分析了多头注意力的子空间分解理论：

1. **子空间分解**：将高维空间分解为多个低维子空间，实现并行语义建模
2. **表征多样性**：不同头关注不同的语义维度，增强模型的表征能力
3. **信息融合**：输出投影矩阵负责融合多头信息，平衡各头贡献
4. **几何意义**：多头机制在几何上实现了多视角的相似度计算
5. **优化策略**：通过合理初始化和融合设计提升多头效果

**思考题**：
1. 多头注意力中，头数增加是否总能提升性能？存在什么权衡？
2. 如何设计更好的头间信息融合机制？
3. 子空间正交性对模型性能有什么影响？
4. 能否设计自适应的头数选择机制？

**下一节预告**：我们将学习位置编码的几何学原理，理解如何在Transformer中引入位置信息。

---

*多头注意力的精妙之处在于用并行的子空间分解实现了认知的多维度建模，这正是人类理解复杂概念时多角度思考的数学体现。* 🧠