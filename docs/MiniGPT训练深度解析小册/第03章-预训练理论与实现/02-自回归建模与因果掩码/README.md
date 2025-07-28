# 02 自回归建模与因果掩码

> **时间的箭头：因果性约束在语言建模中的数学实现**

## 核心思想

自回归建模是现代语言模型的基础架构，它体现了**时间的因果性**：在预测第$t$个词时，模型只能使用前$t-1$个词的信息，不能"偷看"未来。这个看似简单的约束，实际上是语言生成任务的核心挑战。

**关键洞察**：
- **因果性原理**：信息流只能从过去向未来传播
- **掩码机制**：通过矩阵操作实现因果性约束
- **并行训练**：Teacher forcing使得可以并行计算所有位置的损失
- **串行生成**：推理时必须逐步生成，体现真实的序列建模

从数学角度，自回归建模将序列的联合概率分解为条件概率的连乘，每个条件概率都严格遵循因果性约束。

## 2.1 因果性的数学表达

### 信息流的有向无环图

**因果性定义**：在时刻$t$，模型状态只依赖于$t'<t$的历史状态：
$$P(x_t | x_1, x_2, ..., x_{t-1}, x_t, x_{t+1}, ..., x_n) = P(x_t | x_1, x_2, ..., x_{t-1})$$

这可以用**有向无环图(DAG)**表示：
```
x₁ → x₂ → x₃ → x₄ → ... → xₙ
```

每个节点只接收来自前驱节点的信息，不存在反向或跳跃连接。

**矩阵表示**：定义因果掩码矩阵$M \in \{0, 1\}^{n \times n}$：
$$M_{ij} = \begin{cases}
1 & \text{if } i \geq j \\
0 & \text{if } i < j
\end{cases}$$

这是一个**下三角矩阵**，确保位置$i$只能看到位置$j \leq i$的信息。

```python
# MiniGPT中的因果掩码实现 (src/model/transformer.py:243-249)
def create_causal_mask(self, seq_len: int) -> torch.Tensor:
    """创建因果掩码（下三角矩阵）
    
    防止模型在预测时看到未来的token
    """
    mask = torch.tril(torch.ones(seq_len, seq_len))
    return mask  # 1表示可见，0表示掩码

def analyze_causal_structure(seq_len=8):
    """分析因果结构的数学性质"""
    
    # 创建因果掩码
    mask = torch.tril(torch.ones(seq_len, seq_len))
    
    print("因果掩码矩阵:")
    print(mask.numpy().astype(int))
    
    # 分析连通性
    total_connections = seq_len * seq_len
    causal_connections = mask.sum().item()
    sparsity = 1 - (causal_connections / total_connections)
    
    print(f"\\n连通性分析:")
    print(f"  总可能连接数: {total_connections}")
    print(f"  因果连接数: {causal_connections}")
    print(f"  稀疏性: {sparsity:.2%}")
    
    # 每个位置的可见范围
    visible_counts = mask.sum(dim=-1)
    print(f"\\n每个位置的可见范围:")
    for i, count in enumerate(visible_counts):
        print(f"  位置 {i}: 可见 {count} 个位置 (包括自己)")
    
    # 信息传播路径分析
    print(f"\\n信息传播分析:")
    print(f"  第一个位置: 只能看到自己")
    print(f"  最后位置: 可以看到全部 {seq_len} 个位置")
    print(f"  平均可见范围: {visible_counts.float().mean():.1f} 个位置")
    
    return mask
```

### 因果性与马尔可夫性的关系

**马尔可夫链**：状态$x_t$只依赖于前一状态$x_{t-1}$
$$P(x_t | x_1, ..., x_{t-1}) = P(x_t | x_{t-1})$$

**高阶马尔可夫链**：状态$x_t$依赖于前$k$个状态
$$P(x_t | x_1, ..., x_{t-1}) = P(x_t | x_{t-k}, ..., x_{t-1})$$

**Transformer的优势**：理论上可以建模**无限阶马尔可夫链**（受序列长度限制）：
$$P(x_t | x_1, ..., x_{t-1}) = P(x_t | x_1, x_2, ..., x_{t-1})$$

```python
def compare_markov_orders(model, test_data, orders=[1, 2, 4, 8, 16]):
    """比较不同马尔可夫阶数对预测的影响"""
    
    model.eval()
    results = {}
    
    with torch.no_grad():
        for order in orders:
            total_loss = 0
            total_tokens = 0
            
            for batch in test_data:
                input_ids = batch['input_ids']
                seq_len = input_ids.size(1)
                
                # 创建限制历史窗口的掩码
                if order == float('inf'):
                    # 完整因果掩码
                    mask = torch.tril(torch.ones(seq_len, seq_len))
                else:
                    # 限制窗口大小的掩码
                    mask = torch.zeros(seq_len, seq_len)
                    for i in range(seq_len):
                        start = max(0, i - order + 1)
                        mask[i, start:i+1] = 1
                
                # 前向传播（使用限制掩码）
                outputs = model(input_ids, attention_mask=mask.unsqueeze(0).unsqueeze(0))
                
                # 计算损失
                targets = input_ids[:, 1:]
                logits = outputs[:, :-1]
                
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    targets.view(-1),
                    reduction='sum'
                )
                
                total_loss += loss.item()
                total_tokens += targets.numel()
            
            avg_loss = total_loss / total_tokens
            perplexity = math.exp(avg_loss)
            results[order] = {'loss': avg_loss, 'perplexity': perplexity}
            
            print(f"马尔可夫阶数 {order:2d}: 损失={avg_loss:.4f}, 困惑度={perplexity:.2f}")
    
    return results
```

## 2.2 Transformer中的因果掩码实现

### 注意力分数的掩码应用

在Transformer的self-attention中，因果掩码通过将未来位置的注意力分数设为$-\infty$来实现：

$$\text{scores}_{ij} = \begin{cases}
\frac{\mathbf{q}_i^T \mathbf{k}_j}{\sqrt{d_k}} & \text{if } i \geq j \\
-\infty & \text{if } i < j
\end{cases}$$

经过softmax后，$-\infty$位置的权重变为0：
$$\alpha_{ij} = \frac{\exp(\text{scores}_{ij})}{\sum_{k=1}^{i} \exp(\text{scores}_{ik})}$$

```python
# MiniGPT中注意力掩码的详细实现
class CausalSelfAttention:
    """因果自注意力的完整实现与分析"""
    
    def __init__(self, d_model, n_heads, dropout=0.1):
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.scale = math.sqrt(self.d_k)
        
        # 线性投影层
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
        # 注册因果掩码为buffer（不参与训练）
        self.register_buffer('causal_mask', torch.tril(torch.ones(1, 1, 2048, 2048)))
    
    def forward(self, x, return_attention=False):
        batch_size, seq_len, d_model = x.shape
        
        # 1. 计算Q, K, V投影
        Q = self.w_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # 2. 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        # 3. 应用因果掩码
        mask = self.causal_mask[:, :, :seq_len, :seq_len]
        scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # 4. 计算注意力权重
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # 5. 应用注意力权重
        output = torch.matmul(attention_weights, V)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        output = self.w_o(output)
        
        if return_attention:
            return output, attention_weights
        return output

def verify_causal_property(attention_weights):
    """验证注意力权重的因果性质"""
    
    batch_size, n_heads, seq_len, _ = attention_weights.shape
    
    print("=== 因果性验证 ===")
    
    # 检查上三角部分是否为0
    for head in range(min(n_heads, 3)):  # 检查前3个头
        attn_matrix = attention_weights[0, head]  # 取第一个样本
        
        # 提取上三角部分（不包括对角线）
        upper_triangular = torch.triu(attn_matrix, diagonal=1)
        max_future_attention = upper_triangular.max().item()
        nonzero_future = (upper_triangular > 1e-6).sum().item()
        
        print(f"Head {head + 1}:")
        print(f"  最大未来注意力: {max_future_attention:.8f}")
        print(f"  非零未来注意力数: {nonzero_future}")
        
        # 验证行和为1（概率性质）
        row_sums = attn_matrix.sum(dim=-1)
        print(f"  行和范围: [{row_sums.min():.6f}, {row_sums.max():.6f}]")
        
        # 分析对角线强度（自注意力）
        diagonal_strength = torch.diag(attn_matrix).mean().item()
        print(f"  平均对角线强度: {diagonal_strength:.4f}")
    
    # 整体统计  
    total_future_attention = torch.triu(attention_weights, diagonal=1).sum().item()
    print(f"\\n总体未来注意力: {total_future_attention:.8f} (应该接近0)")
    
    return total_future_attention < 1e-6  # 因果性验证通过
```

### 掩码的数值稳定性

使用$-\infty$可能导致数值问题。实践中使用大负数：

```python
def analyze_mask_numerical_stability():
    """分析掩码值对数值稳定性的影响"""
    
    seq_len = 10
    mask_values = [-1e9, -1e4, -100, -10, -1]
    
    # 创建测试注意力分数
    scores = torch.randn(1, 1, seq_len, seq_len) * 2  # 随机分数
    
    print("掩码值对注意力权重的影响:")
    print("位置 | " + " | ".join([f"mask={v}" for v in mask_values]))
    print("-" * 60)
    
    for pos in range(3):  # 检查前3个位置
        row_data = [f"{pos:2d}"]
        
        for mask_val in mask_values:
            # 应用掩码
            masked_scores = scores.clone()
            mask = torch.tril(torch.ones(seq_len, seq_len))
            masked_scores[0, 0] = masked_scores[0, 0].masked_fill(mask == 0, mask_val)
            
            # 计算softmax
            attention = F.softmax(masked_scores[0, 0], dim=-1)
            
            # 检查该位置对未来的注意力
            future_attention = attention[pos, pos+1:].sum().item()
            row_data.append(f"{future_attention:.2e}")
        
        print(" | ".join(row_data))
    
    print("\\n推荐使用 -1e9 作为掩码值，既保证因果性又维持数值稳定性")

def gradient_flow_analysis_with_masking():
    """分析掩码对梯度流的影响"""
    
    # 创建简单的自注意力层
    d_model = 64
    seq_len = 16
    attention = CausalSelfAttention(d_model, n_heads=4)
    
    # 随机输入
    x = torch.randn(2, seq_len, d_model, requires_grad=True)
    
    # 前向传播
    output, attn_weights = attention(x, return_attention=True)
    
    # 计算损失（简单示例）
    loss = output.mean()
    
    # 反向传播
    loss.backward()
    
    # 分析梯度
    input_grad_norm = x.grad.norm(dim=-1)  # (batch_size, seq_len)
    
    print("梯度流分析:")
    print(f"输入梯度范数分布:")
    for pos in range(min(seq_len, 8)):
        grad_norm = input_grad_norm[0, pos].item()
        print(f"  位置 {pos:2d}: 梯度范数 = {grad_norm:.6f}")
    
    # 分析梯度与位置的关系
    position_range = torch.arange(seq_len).float()
    grad_norms = input_grad_norm[0]
    
    # 计算相关性
    correlation = torch.corrcoef(torch.stack([position_range, grad_norms]))[0, 1]
    print(f"\\n位置与梯度范数的相关性: {correlation:.4f}")
    
    return input_grad_norm, attn_weights
```

## 2.3 训练中的Teacher Forcing

### 并行训练的数学原理

**Teacher Forcing**：训练时使用真实的历史token，而不是模型生成的token：

训练时：$P(x_t | x_1^{true}, x_2^{true}, ..., x_{t-1}^{true})$
推理时：$P(x_t | x_1^{pred}, x_2^{pred}, ..., x_{t-1}^{pred})$

这种**分布偏移**是自回归模型训练的核心挑战。

**并行计算优势**：Teacher Forcing使得可以并行计算所有位置的损失：

```python
def demonstrate_teacher_forcing():
    """演示Teacher Forcing的并行训练过程"""
    
    # 模拟训练数据
    batch_size, seq_len, vocab_size = 4, 8, 1000
    
    # 真实序列
    true_sequence = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    print("Teacher Forcing 并行训练演示:")
    print("真实序列 (前4个token):", true_sequence[0, :4].tolist())
    
    # 模拟模型输出 (logits)
    model_logits = torch.randn(batch_size, seq_len, vocab_size)
    
    # 计算所有位置的损失（并行）
    targets = true_sequence[:, 1:]  # 移除第一个token作为输入
    predictions = model_logits[:, :-1]  # 移除最后一个预测
    
    # 每个位置的损失
    position_losses = F.cross_entropy(
        predictions.view(-1, vocab_size),
        targets.view(-1),
        reduction='none'
    ).view(batch_size, seq_len-1)
    
    print("\\n各位置损失 (第一个样本):")
    for pos in range(seq_len-1):
        loss_val = position_losses[0, pos].item()
        true_token = true_sequence[0, pos+1].item()
        print(f"  位置 {pos+1}: 预测token {true_token}, 损失 = {loss_val:.4f}")
    
    # 总损失
    total_loss = position_losses.mean()
    print(f"\\n平均损失: {total_loss:.4f}")
    
    return position_losses

def analyze_exposure_bias():
    """分析暴露偏差问题"""
    
    print("=== 暴露偏差分析 ===")
    print("训练阶段:")
    print("  输入: [<BOS>, '今天', '天气', '很好']")
    print("  目标: ['今天', '天气', '很好', '<EOS>']")
    print("  特点: 每步都使用真实历史")
    
    print("\\n推理阶段:")
    print("  t=1: 输入<BOS> → 生成'今天'")
    print("  t=2: 输入<BOS>+'今天' → 生成'天气'")  
    print("  t=3: 输入<BOS>+'今天'+'天气' → 生成'很好'")
    print("  特点: 每步都使用生成历史")
    
    print("\\n问题分析:")
    print("1. 分布不匹配: 训练用真实分布，推理用模型分布")
    print("2. 错误累积: 早期错误影响后续生成")
    print("3. 长序列问题: 序列越长，累积误差越大")
    
    print("\\n缓解策略:")
    print("1. Scheduled Sampling: 训练时逐渐引入生成token")
    print("2. 对比学习: 学习好坏序列的差异")
    print("3. 强化学习: 直接优化生成质量指标")

def scheduled_sampling_demo(model, batch, schedule_ratio=0.5):
    """演示Scheduled Sampling训练策略"""
    
    input_ids = batch['input_ids']
    batch_size, seq_len = input_ids.shape
    
    # 创建混合输入序列
    mixed_input = input_ids.clone()
    
    for pos in range(1, seq_len):
        # 按比例决定使用真实token还是生成token
        use_generated = torch.rand(batch_size) < schedule_ratio
        
        if use_generated.any():
            # 获取前一位置的模型预测
            with torch.no_grad():
                logits = model(mixed_input[:, :pos])[:, -1]  # 最后一位的预测
                generated_tokens = logits.argmax(dim=-1)
            
            # 替换部分样本的当前token
            mixed_input[use_generated, pos] = generated_tokens[use_generated]
    
    print("Scheduled Sampling 效果:")
    print("原始序列:", input_ids[0, :8].tolist())
    print("混合序列:", mixed_input[0, :8].tolist())
    print(f"替换比例: {schedule_ratio:.1%}")
    
    return mixed_input
```

## 2.4 推理时的自回归生成

### 逐步生成的数学过程

推理时，模型必须**逐步生成**每个token：

```python
def autoregressive_generation(model, tokenizer, prompt, max_length=50, temperature=1.0):
    """完整的自回归生成过程"""
    
    model.eval()
    
    # 编码初始prompt
    input_ids = tokenizer.encode(prompt)
    input_tensor = torch.tensor([input_ids])
    
    print(f"自回归生成过程:")
    print(f"初始prompt: '{prompt}'")
    print(f"初始token ids: {input_ids}")
    print()
    
    generated_tokens = []
    
    with torch.no_grad():
        for step in range(max_length):
            # 前向传播
            outputs = model(input_tensor)
            
            # 获取最后一个位置的logits
            next_token_logits = outputs[0, -1, :] / temperature
            
            # 采样下一个token
            probs = F.softmax(next_token_logits, dim=-1)
            next_token_id = torch.multinomial(probs, num_samples=1).item()
            
            # 解码token
            next_token = tokenizer.decode([next_token_id])
            generated_tokens.append(next_token)
            
            print(f"步骤 {step+1:2d}: 生成 '{next_token}' (id={next_token_id}, p={probs[next_token_id]:.4f})")
            
            # 停止条件
            if next_token_id == tokenizer.eos_token_id:
                print("遇到结束符，停止生成")
                break
            
            # 更新输入序列
            input_tensor = torch.cat([input_tensor, torch.tensor([[next_token_id]])], dim=1)
            
            # 序列长度限制
            if input_tensor.size(1) > model.max_seq_len:
                print("达到最大序列长度，停止生成")
                break
    
    # 生成结果
    generated_text = ''.join(generated_tokens)
    full_text = prompt + generated_text
    
    print(f"\\n生成结果:")
    print(f"完整文本: '{full_text}'")
    print(f"生成长度: {len(generated_tokens)} tokens")
    
    return full_text, generated_tokens

def analyze_generation_statistics(model, tokenizer, prompts, num_samples=5):
    """分析生成统计特性"""
    
    model.eval()
    stats = {
        'lengths': [],
        'perplexities': [],
        'token_diversity': [],
        'repetition_rates': []
    }
    
    for prompt in prompts:
        for _ in range(num_samples):
            # 生成文本
            generated_text, tokens = autoregressive_generation(
                model, tokenizer, prompt, max_length=30, temperature=0.8
            )
            
            # 统计长度
            stats['lengths'].append(len(tokens))
            
            # 计算困惑度
            input_ids = tokenizer.encode(generated_text)
            if len(input_ids) > 1:
                with torch.no_grad():
                    outputs = model(torch.tensor([input_ids]))
                    loss = F.cross_entropy(
                        outputs[0, :-1].view(-1, outputs.size(-1)),
                        torch.tensor(input_ids[1:])
                    )
                    ppl = torch.exp(loss).item()
                    stats['perplexities'].append(ppl)
            
            # 计算词汇多样性
            unique_tokens = len(set(tokens))
            diversity = unique_tokens / len(tokens) if tokens else 0
            stats['token_diversity'].append(diversity)
            
            # 计算重复率
            if len(tokens) > 1:
                repetitions = sum(1 for i in range(1, len(tokens)) if tokens[i] == tokens[i-1])
                repetition_rate = repetitions / (len(tokens) - 1)
                stats['repetition_rates'].append(repetition_rate)
    
    # 输出统计结果
    print("=== 生成统计分析 ===")
    for metric, values in stats.items():
        if values:
            mean_val = np.mean(values)
            std_val = np.std(values)
            print(f"{metric:15s}: 均值={mean_val:.3f}, 标准差={std_val:.3f}")
    
    return stats
```

### 生成策略的数学分析

**贪心解码**：每步选择概率最大的token
$$x_t = \arg\max_{x} P(x | x_{<t})$$

**随机采样**：按概率分布采样
$$x_t \sim P(x | x_{<t})$$

**Top-k采样**：只从概率最高的k个token中采样
**Top-p采样**：从累积概率达到p的最小token集合中采样

```python
def compare_decoding_strategies(model, tokenizer, prompt, strategies):
    """比较不同解码策略"""
    
    model.eval()
    input_ids = tokenizer.encode(prompt)
    input_tensor = torch.tensor([input_ids])
    
    results = {}
    
    with torch.no_grad():
        # 获取下一个token的logits
        outputs = model(input_tensor)
        next_token_logits = outputs[0, -1, :]
        
        print(f"解码策略比较 (prompt: '{prompt}'):")
        print("-" * 50)
        
        for strategy_name, strategy_config in strategies.items():
            if strategy_name == 'greedy':
                # 贪心解码
                next_token_id = next_token_logits.argmax().item()
                prob = F.softmax(next_token_logits, dim=-1)[next_token_id].item()
                
            elif strategy_name == 'temperature':
                # 温度采样
                temp = strategy_config['temperature']
                scaled_logits = next_token_logits / temp
                probs = F.softmax(scaled_logits, dim=-1)
                next_token_id = torch.multinomial(probs, num_samples=1).item()
                prob = probs[next_token_id].item()
                
            elif strategy_name == 'top_k':
                # Top-k采样
                k = strategy_config['k']
                top_k_logits, top_k_indices = torch.topk(next_token_logits, k)
                top_k_probs = F.softmax(top_k_logits, dim=-1)
                selected_idx = torch.multinomial(top_k_probs, num_samples=1).item()
                next_token_id = top_k_indices[selected_idx].item()
                prob = top_k_probs[selected_idx].item()
                
            elif strategy_name == 'top_p':
                # Top-p (nucleus) 采样
                p = strategy_config['p']
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                # 找到累积概率超过p的位置
                sorted_indices_to_remove = cumulative_probs > p
                sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
                sorted_indices_to_remove[0] = 0
                
                # 构建nucleus
                nucleus_logits = sorted_logits.clone()
                nucleus_logits[sorted_indices_to_remove] = float('-inf')
                nucleus_probs = F.softmax(nucleus_logits, dim=-1)
                
                selected_idx = torch.multinomial(nucleus_probs, num_samples=1).item()
                next_token_id = sorted_indices[selected_idx].item()
                prob = nucleus_probs[selected_idx].item()
            
            # 解码token
            next_token = tokenizer.decode([next_token_id])
            results[strategy_name] = {
                'token': next_token,
                'token_id': next_token_id,
                'probability': prob
            }
            
            print(f"{strategy_name:12s}: '{next_token}' (id={next_token_id}, p={prob:.4f})")
    
    return results

def analyze_decoding_diversity(model, tokenizer, prompt, num_generations=10):
    """分析不同解码策略的多样性"""
    
    strategies = {
        'greedy': {},
        'temperature_0.8': {'temperature': 0.8},
        'temperature_1.2': {'temperature': 1.2},
        'top_k_10': {'k': 10},
        'top_p_0.9': {'p': 0.9}
    }
    
    diversity_stats = {}
    
    for strategy_name, config in strategies.items():
        generated_texts = []
        
        for _ in range(num_generations):
            if strategy_name == 'greedy':
                # 贪心解码总生成相同结果
                if not generated_texts:  # 只计算一次
                    text, _ = autoregressive_generation(
                        model, tokenizer, prompt, max_length=20, temperature=1.0
                    )
                    generated_texts = [text] * num_generations
                break
            else:
                # 随机策略
                temp = config.get('temperature', 1.0)
                text, _ = autoregressive_generation(
                    model, tokenizer, prompt, max_length=20, temperature=temp
                )
                generated_texts.append(text)
        
        # 计算多样性指标
        unique_texts = len(set(generated_texts))
        diversity_ratio = unique_texts / len(generated_texts)
        
        # 计算平均编辑距离
        edit_distances = []
        for i in range(len(generated_texts)):
            for j in range(i+1, len(generated_texts)):
                # 简化的编辑距离（字符级）
                dist = len(set(generated_texts[i]) ^ set(generated_texts[j]))
                edit_distances.append(dist)
        
        avg_edit_distance = np.mean(edit_distances) if edit_distances else 0
        
        diversity_stats[strategy_name] = {
            'unique_ratio': diversity_ratio,
            'avg_edit_distance': avg_edit_distance
        }
        
        print(f"{strategy_name:15s}: 唯一性={diversity_ratio:.2%}, 平均差异={avg_edit_distance:.1f}")
    
    return diversity_stats
```

## 2.5 实践：MiniGPT中的因果建模

### 完整的因果Transformer实现

```python
# MiniGPT中Transformer Block的因果建模实现
class TransformerBlock(nn.Module):
    """包含因果约束的Transformer块"""
    
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        
        # 因果自注意力
        self.self_attention = CausalSelfAttention(d_model, n_heads, dropout)
        
        # 前馈网络
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        
        # 层归一化
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # 自注意力子层（Pre-LN结构）
        attn_input = self.norm1(x)
        attn_output = self.self_attention(attn_input)
        x = x + self.dropout(attn_output)
        
        # 前馈网络子层
        ff_input = self.norm2(x)
        ff_output = self.feed_forward(ff_input)
        x = x + self.dropout(ff_output)
        
        return x

def test_causal_property_end_to_end():
    """端到端测试因果性质"""
    
    # 创建模型
    d_model, n_heads, n_layers = 128, 4, 6
    model = nn.ModuleList([
        TransformerBlock(d_model, n_heads, d_model*4) 
        for _ in range(n_layers)
    ])
    
    # 测试输入
    batch_size, seq_len = 2, 10
    x = torch.randn(batch_size, seq_len, d_model)
    
    print("=== 端到端因果性测试 ===")
    
    # 前向传播
    hidden_states = [x]
    for i, layer in enumerate(model):
        x = layer(x)
        hidden_states.append(x)
        
        print(f"Layer {i+1} 输出形状: {x.shape}")
    
    # 测试因果性：修改未来token，检查过去token的输出是否改变
    x_modified = x.clone()
    x_modified[:, -1, :] = torch.randn_like(x_modified[:, -1, :])  # 修改最后一个token
    
    # 重新前向传播
    x_test = hidden_states[0]  # 重新开始
    for layer in model:
        x_test = layer(x_test)
    
    # 比较前面位置的输出
    for pos in range(seq_len - 1):
        diff = torch.norm(x[0, pos] - x_test[0, pos]).item()
        if diff > 1e-6:
            print(f"警告: 位置 {pos} 受到未来信息影响，差异 = {diff}")
        else:
            print(f"位置 {pos}: 因果性保持 ✓")
    
    return hidden_states

def performance_comparison_causal_vs_bidirectional():
    """比较因果模型与双向模型的性能"""
    
    print("=== 因果 vs 双向模型比较 ===")
    
    # 模型配置
    config = {
        'd_model': 256,
        'n_heads': 8,
        'n_layers': 6,
        'vocab_size': 10000,
        'seq_len': 128
    }
    
    # 创建测试数据
    batch_size = 16
    test_input = torch.randint(0, config['vocab_size'], (batch_size, config['seq_len']))
    
    # 因果模型
    causal_model = GPTModel(**config)  # 假设的GPT模型
    
    # 双向模型（移除因果掩码）
    bidirectional_model = BERTModel(**config)  # 假设的BERT模型
    
    # 性能测试
    with torch.no_grad():
        # 因果模型推理
        start_time = time.time()
        causal_output = causal_model(test_input)
        causal_time = time.time() - start_time
        
        # 双向模型推理
        start_time = time.time()
        bidirectional_output = bidirectional_model(test_input)
        bidirectional_time = time.time() - start_time
    
    print(f"推理时间比较:")
    print(f"  因果模型:   {causal_time:.4f}s")
    print(f"  双向模型:   {bidirectional_time:.4f}s")
    print(f"  时间比值:   {causal_time/bidirectional_time:.2f}")
    
    print(f"\\n模型参数比较:")
    causal_params = sum(p.numel() for p in causal_model.parameters())
    bidirectional_params = sum(p.numel() for p in bidirectional_model.parameters()) 
    print(f"  因果模型:   {causal_params:,} 参数")
    print(f"  双向模型:   {bidirectional_params:,} 参数")
    
    print(f"\\n适用场景:")
    print(f"  因果模型:   文本生成、语言建模")
    print(f"  双向模型:   文本理解、分类任务")
```

## 小结与思考

本节深入探讨了自回归建模与因果掩码：

1. **因果性原理**：信息流的单向性确保了模型的时序建模能力
2. **掩码实现**：通过下三角矩阵实现注意力的因果约束
3. **Teacher Forcing**：训练与推理的分布差异带来的挑战
4. **生成策略**：不同解码方法对生成质量和多样性的影响

**关键洞察**：
- 因果性是语言生成任务的核心约束
- 掩码机制优雅地在并行架构中实现了序列依赖
- 训练推理不一致是自回归模型的固有问题
- 解码策略的选择需要平衡质量和多样性

**思考题**：
1. 为什么Transformer比RNN更适合实现因果建模？
2. Teacher Forcing的优缺点如何权衡？
3. 不同生成策略在什么场景下更适用？
4. 如何缓解长序列生成中的错误累积问题？

**下一节预告**：我们将学习分词策略与信息压缩，理解如何将原始文本转换为模型可处理的符号序列。

---

*因果性不仅是时间的约束，更是语言建模的智慧体现——在有限的历史中预测无限的未来。* ⏱️