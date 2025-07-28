# 第一章：基础架构篇 - Transformer 的数学本质

## 引言：从循环到注意力的范式转变

Transformer 架构的诞生标志着序列建模从循环计算向并行计算的根本性转变。其核心洞察在于：**序列中任意两个位置之间的依赖关系可以通过注意力机制直接建模，无需递归计算**。

## 1.1 注意力机制的数学推导与实现

### 1.1.1 缩放点积注意力的信息论解释

注意力机制的核心公式为：

```math
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
```

**数学本质分析：**

1. **查询-键匹配**：$QK^T$ 计算查询向量与所有键向量的内积，度量语义相似性
2. **温度缩放**：$\sqrt{d_k}$ 缩放因子防止内积过大导致 softmax 饱和
3. **注意力分布**：softmax 将相似性分数转化为概率分布
4. **信息聚合**：概率加权求和提取相关信息

**代码实现分析** (`src/model/transformer.py:81-102`)：

```python
def scaled_dot_product_attention(self, Q, K, V, mask=None):
    # 计算注意力分数: Q @ K^T / √d_k
    scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
    # 形状: (batch_size, n_heads, seq_len, seq_len)
    
    # 应用掩码（如果有）
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    
    # 计算注意力权重
    attention_weights = F.softmax(scores, dim=-1)
    attention_weights = self.dropout(attention_weights)
    
    # 应用注意力权重到V
    output = torch.matmul(attention_weights, V)
    return output
```

**关键技术细节：**

- **缩放因子推导**：当 $d_k$ 较大时，内积的方差为 $d_k$，缩放后方差为 1，保持梯度稳定
- **掩码技术**：将需要屏蔽的位置设为 $-\infty$，经过 softmax 后概率接近 0
- **Dropout 正则化**：在注意力权重上应用 dropout，提高模型泛化能力

### 1.1.2 多头注意力的子空间分解理论

多头注意力将单个注意力机制扩展到多个并行的"注意力头"：

```math
\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, ..., \text{head}_h)W^O
```

其中每个头：
```math
\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)
```

**子空间分解的数学意义：**

1. **表征多样性**：不同头关注不同的语义子空间
2. **并行计算**：多头可并行计算，提高效率
3. **信息融合**：输出投影矩阵 $W^O$ 融合多头信息

**代码实现分析** (`src/model/transformer.py:62-79`)：

```python
# 重塑为多头格式
Q = Q.reshape(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
K = K.reshape(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
V = V.reshape(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
# 现在形状为: (batch_size, n_heads, seq_len, d_k)

# 计算注意力
attention_output = self.scaled_dot_product_attention(Q, K, V, mask)

# 拼接多头结果
attention_output = attention_output.transpose(1, 2).contiguous()
attention_output = attention_output.reshape(batch_size, seq_len, d_model)
```

**维度变换解析：**
- 原始维度：`(batch_size, seq_len, d_model)`
- 多头重塑：`(batch_size, n_heads, seq_len, d_k)` 其中 `d_k = d_model / n_heads`
- 注意力计算后拼接回：`(batch_size, seq_len, d_model)`

### 1.1.3 注意力权重的概率分布特性

注意力权重矩阵 $A \in \mathbb{R}^{n \times n}$ 具有以下重要性质：

1. **概率分布**：$\sum_{j=1}^n A_{ij} = 1, A_{ij} \geq 0$
2. **非对称性**：$A_{ij} \neq A_{ji}$（与相似度矩阵不同）
3. **稀疏性**：实际应用中注意力权重往往集中在少数位置

**数学分析：**
- 每行和为 1，满足概率分布约束
- 对角线元素表示自注意力强度
- 远距离依赖通过非零的远程权重建模

## 1.2 位置编码的几何学原理

### 1.2.1 正弦位置编码的傅里叶变换视角

Transformer 使用固定的正弦位置编码：

```math
PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right)

PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)
```

**傅里叶变换解释：**

1. **频率递减**：从高频到低频的正弦波组合
2. **周期性编码**：不同维度具有不同周期
3. **线性组合性**：相对位置可通过三角恒等式计算

**代码实现分析** (`src/model/transformer.py:134-151`)：

```python
# 创建位置编码矩阵
pe = torch.zeros(max_len, d_model)
position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

# 计算除数项
div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                   (-math.log(10000.0) / d_model))

# 计算正弦和余弦
pe[:, 0::2] = torch.sin(position * div_term)  # 偶数位置
pe[:, 1::2] = torch.cos(position * div_term)  # 奇数位置
```

**关键数学洞察：**
- `div_term` 实现了 $10000^{-2i/d_{model}}$ 的计算
- 偶数维度使用 sin，奇数维度使用 cos
- 不同维度的周期从 $2\pi$ 到 $2\pi \times 10000$

### 1.2.2 相对位置关系的编码机制

位置编码的核心优势在于能够表达相对位置关系：

对于位置 $pos$ 和 $pos + k$，它们的位置编码可以通过线性组合表示彼此：

```math
PE_{pos+k} = T_k \cdot PE_{pos}
```

其中 $T_k$ 是仅依赖于相对距离 $k$ 的变换矩阵。

**几何解释：**
- 位置编码在高维空间中形成螺旋结构
- 相对位置对应螺旋上的固定变换
- 模型可学习到位置不变的模式

## 1.3 残差连接与层归一化的优化理论

### 1.3.1 梯度流动与深度网络训练稳定性

残差连接的数学表达：
```math
H_{l+1} = H_l + F(H_l)
```

**梯度分析：**
```math
\frac{\partial H_{l+1}}{\partial H_l} = I + \frac{\partial F(H_l)}{\partial H_l}
```

这确保了梯度至少有恒等映射的贡献，缓解梯度消失问题。

**代码实现分析** (`src/model/transformer.py:182-191`)：

```python
def forward(self, x, mask=None):
    # 多头注意力 + 残差连接 + 层归一化
    attn_output = self.attention(x, x, x, mask)
    x = self.norm1(x + self.dropout(attn_output))  # 残差连接
    
    # 前馈网络 + 残差连接 + 层归一化
    ff_output = self.feed_forward(x)
    x = self.norm2(x + self.dropout(ff_output))    # 残差连接
    
    return x
```

### 1.3.2 LayerNorm vs BatchNorm 的统计学差异

**LayerNorm 归一化：**
```math
\text{LayerNorm}(x) = \gamma \frac{x - \mu}{\sigma} + \beta
```

其中 $\mu$ 和 $\sigma$ 在特征维度上计算：
```math
\mu = \frac{1}{d}\sum_{i=1}^d x_i, \quad \sigma = \sqrt{\frac{1}{d}\sum_{i=1}^d (x_i - \mu)^2}
```

**关键优势：**
1. **序列长度无关**：不依赖于批次大小和序列长度
2. **推理一致性**：训练和推理行为一致
3. **梯度稳定**：归一化后激活值分布稳定

### 1.3.3 残差连接对损失景观的平滑作用

**理论分析：**
残差连接使损失函数更加平滑，具体表现为：

1. **Lipschitz 常数降低**：梯度变化更平缓
2. **局部极小值减少**：优化景观更平滑
3. **收敛速度提升**：更容易找到全局最优

**数学证明思路：**
设 $L(W)$ 为损失函数，$W$ 为网络参数。残差连接使得：
```math
||\nabla L(W_1) - \nabla L(W_2)|| \leq \beta ||W_1 - W_2||
```
其中 $\beta$ 相比无残差网络显著减小。

## 1.4 前馈网络的非线性映射

### 1.4.1 位置前馈网络的数学形式

```math
\text{FFN}(x) = \text{ReLU}(xW_1 + b_1)W_2 + b_2
```

**代码实现** (`src/model/transformer.py:119-121`)：

```python
def forward(self, x):
    # x: (batch_size, seq_len, d_model)
    return self.w_2(self.dropout(F.relu(self.w_1(x))))
```

**维度变换：**
- 输入：`d_model` → 隐藏：`d_ff` → 输出：`d_model`
- 典型比例：`d_ff = 4 * d_model`

### 1.4.2 激活函数的选择与影响

**ReLU 激活函数：**
```math
\text{ReLU}(x) = \max(0, x)
```

**优势：**
1. **计算高效**：简单的阈值操作
2. **梯度稳定**：正区间梯度为 1
3. **稀疏激活**：产生稀疏表示

**现代替代方案：**
- **GELU**：$\text{GELU}(x) = x \cdot \Phi(x)$，更平滑的激活
- **SwiGLU**：$\text{SwiGLU}(x) = \text{Swish}(xW_1) \odot (xW_2)$，门控机制

## 1.5 完整的 MiniGPT 架构

### 1.5.1 模型整体结构

**代码架构分析** (`src/model/transformer.py:194-285`)：

```python
class MiniGPT(nn.Module):
    def __init__(self, vocab_size, d_model=512, n_heads=8, 
                 n_layers=6, d_ff=2048, max_len=1024, dropout=0.1):
        # 词嵌入层
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        
        # 位置编码
        self.positional_encoding = PositionalEncoding(d_model, max_len)
        
        # Transformer层堆叠
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        # 输出层
        self.layer_norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)
```

### 1.5.2 前向传播流程

**数学表达：**
```math
\begin{align}
H_0 &= \text{Embedding}(x) + \text{PE}(x) \\
H_l &= \text{TransformerBlock}_l(H_{l-1}), \quad l = 1, ..., L \\
\text{logits} &= \text{LayerNorm}(H_L) W_{vocab}
\end{align}
```

**代码实现** (`src/model/transformer.py:251-285`)：

```python
def forward(self, input_ids, attention_mask=None):
    batch_size, seq_len = input_ids.size()
    
    # 1. 词嵌入
    x = self.token_embedding(input_ids)
    
    # 2. 位置编码
    x = self.positional_encoding(x)
    x = self.dropout(x)
    
    # 3. 创建因果掩码
    causal_mask = self.create_causal_mask(seq_len).to(input_ids.device)
    
    # 4. 通过Transformer块
    for transformer_block in self.transformer_blocks:
        x = transformer_block(x, causal_mask)
    
    # 5. 层归一化
    x = self.layer_norm(x)
    
    # 6. 输出投影
    logits = self.lm_head(x)
    
    return logits
```

### 1.5.3 因果掩码的实现机制

**数学定义：**
```math
\text{mask}_{ij} = \begin{cases}
1 & \text{if } i \geq j \\
0 & \text{if } i < j
\end{cases}
```

**代码实现** (`src/model/transformer.py:243-249`)：

```python
def create_causal_mask(self, seq_len):
    """创建因果掩码（下三角矩阵）
    
    防止模型在预测时看到未来的token
    """
    mask = torch.tril(torch.ones(seq_len, seq_len))
    return mask  # 1表示可见，0表示掩码
```

**作用机制：**
- 确保位置 $i$ 只能看到位置 $j \leq i$ 的信息
- 实现自回归语言建模的因果约束
- 训练时并行计算，推理时逐步生成

## 1.6 参数初始化与模型稳定性

### 1.6.1 权重初始化策略

**代码实现** (`src/model/transformer.py:230-241`)：

```python
def init_weights(self):
    """初始化模型参数"""
    for module in self.modules():
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
```

**初始化原理：**
1. **线性层**：小方差正态分布，避免梯度爆炸
2. **嵌入层**：正态分布初始化，保持词向量多样性
3. **LayerNorm**：权重初始化为 1，偏置为 0

### 1.6.2 模型规模与参数量分析

**参数量计算：**

对于配置 `(vocab_size=V, d_model=D, n_heads=H, n_layers=L, d_ff=F)`：

1. **嵌入层**：$V \times D$
2. **每个 Transformer 块**：
   - 注意力：$4 \times D^2$ (Q, K, V, O 投影)
   - 前馈：$2 \times D \times F$
   - LayerNorm：$4 \times D$ (两个 LayerNorm，每个有权重和偏置)
3. **输出层**：$D \times V$

**总参数量**：
```math
\text{Total} = 2VD + L(4D^2 + 2DF + 4D)
```

**代码验证** (`src/model/transformer.py:329-331`)：

```python
def get_num_params(self):
    """获取模型参数数量"""
    return sum(p.numel() for p in self.parameters())
```

## 小结

本章深入分析了 Transformer 架构的数学原理和实现细节：

1. **注意力机制**是信息选择和聚合的核心，通过 QKV 三元组实现高效的序列建模
2. **多头注意力**通过子空间分解增强表征能力，实现并行计算
3. **位置编码**基于三角函数的傅里叶分解，巧妙编码序列位置信息
4. **残差连接**和**层归一化**确保深层网络的训练稳定性
5. **因果掩码**实现自回归约束，支持并行训练和递归推理

这些组件的协同作用构成了现代大语言模型的基础架构，为后续的预训练、微调和强化学习奠定了坚实基础。