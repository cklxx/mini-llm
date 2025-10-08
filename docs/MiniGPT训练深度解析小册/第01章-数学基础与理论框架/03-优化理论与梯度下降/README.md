# 03 优化理论与梯度下降

> **理解深度学习训练的数学本质**

## 核心思想

深度学习的训练本质上是一个优化问题：在参数空间中寻找使损失函数最小的点。理解优化理论，有助于我们设计更好的训练算法，诊断训练问题，并提高模型性能。

**关键洞察**：
- 梯度指向函数值增长最快的方向
- 学习率控制着参数更新的步长
- 自适应算法能够适应不同参数的更新频率
- 优化景观的几何性质决定了训练的难易程度

## 3.1 凸优化与非凸优化

### 凸函数的定义

函数 $f: \mathbb{R}^n \rightarrow \mathbb{R}$ 是凸函数，当且仅当对于任意 $\mathbf{x}, \mathbf{y} \in \mathbb{R}^n$ 和 $\lambda \in [0,1]$：

$$f(\lambda \mathbf{x} + (1-\lambda)\mathbf{y}) \leq \lambda f(\mathbf{x}) + (1-\lambda)f(\mathbf{y})$$

**几何直觉**：函数图像下方的区域是凸集。

**二阶条件**：如果 $f$ 二次可微，则 $f$ 为凸函数当且仅当其Hessian矩阵半正定：
$$\mathbf{H} = \nabla^2 f(\mathbf{x}) \succeq 0$$

### 深度学习中的非凸优化

**问题特征**：
- 损失函数是非凸的（存在多个局部最优）
- 参数空间维度极高（数百万到数千亿参数）
- 存在鞍点（Hessian矩阵既有正特征值又有负特征值）

**非凸优化的挑战**：
```python
def analyze_loss_landscape(model, dataloader, param_name='transformer_blocks.0.attention.w_q.weight'):
    """分析损失函数的景观特性"""
    
    # 获取目标参数
    target_param = None
    for name, param in model.named_parameters():
        if name == param_name:
            target_param = param
            break
    
    if target_param is None:
        return
    
    original_data = target_param.data.clone()
    
    # 在参数空间中采样
    perturbations = torch.randn(10, *target_param.shape) * 0.01
    losses = []
    
    model.eval()
    with torch.no_grad():
        for perturbation in perturbations:
            # 扰动参数
            target_param.data = original_data + perturbation
            
            # 计算损失
            total_loss = 0
            for batch in dataloader:
                if isinstance(batch, dict):
                    input_ids, labels = batch['input_ids'], batch['labels']
                else:
                    input_ids = batch
                    labels = torch.cat([input_ids[:, 1:], 
                                      torch.zeros(input_ids.size(0), 1, dtype=torch.long)], dim=1)
                
                logits = model(input_ids)
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), 
                                     labels.reshape(-1), ignore_index=0)
                total_loss += loss.item()
                break  # 只用一个batch近似
            
            losses.append(total_loss)
    
    # 恢复原参数
    target_param.data = original_data
    
    # 分析统计特性
    losses = torch.tensor(losses)
    print(f"损失函数统计 (参数: {param_name}):")
    print(f"  均值: {losses.mean():.4f}")
    print(f"  标准差: {losses.std():.4f}")
    print(f"  最小值: {losses.min():.4f}")
    print(f"  最大值: {losses.max():.4f}")
    
    return losses
```

## 3.2 梯度下降算法族

### 普通梯度下降(Gradient Descent)

**算法**：
$$\theta_{t+1} = \theta_t - \eta \nabla_\theta L(\theta_t)$$

其中 $\eta$ 是学习率，$\nabla_\theta L$ 是损失函数的梯度。

**收敛性分析**：
对于 $L$-光滑的凸函数，梯度下降的收敛率为 $O(1/T)$，其中 $T$ 是迭代次数。

**代码实现**：
```python
class SGD:
    def __init__(self, params, lr=0.01):
        self.params = list(params)
        self.lr = lr
    
    def step(self):
        """执行一步参数更新"""
        for param in self.params:
            if param.grad is not None:
                # θ = θ - η∇L
                param.data.add_(param.grad.data, alpha=-self.lr)
    
    def zero_grad(self):
        """清零梯度"""
        for param in self.params:
            if param.grad is not None:
                param.grad.zero_()
```

### 动量法(Momentum)

**动机**：在相关方向上加速，在震荡方向上减速。

**算法**：
$$\mathbf{v}_t = \beta \mathbf{v}_{t-1} + (1-\beta)\nabla_\theta L(\theta_t)$$
$$\theta_{t+1} = \theta_t - \eta \mathbf{v}_t$$

其中 $\beta$ 通常取值 0.9。

**物理类比**：想象一个球在损失函数表面滚动，动量帮助球穿越小的局部最优。

```python
class Momentum:
    def __init__(self, params, lr=0.01, momentum=0.9):
        self.params = list(params)
        self.lr = lr
        self.momentum = momentum
        
        # 初始化动量缓存
        self.velocity = {}
        for param in self.params:
            self.velocity[id(param)] = torch.zeros_like(param.data)
    
    def step(self):
        for param in self.params:
            if param.grad is not None:
                param_id = id(param)
                
                # v = βv + (1-β)∇L
                self.velocity[param_id] = (self.momentum * self.velocity[param_id] + 
                                         (1 - self.momentum) * param.grad.data)
                
                # θ = θ - ηv
                param.data.add_(self.velocity[param_id], alpha=-self.lr)
```

### AdaGrad算法

**动机**：为不同参数使用不同的学习率，频繁更新的参数学习率衰减更快。

**算法**：
$$G_t = G_{t-1} + \nabla_\theta L(\theta_t) \odot \nabla_\theta L(\theta_t)$$
$$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{G_t + \epsilon}} \odot \nabla_\theta L(\theta_t)$$

其中 $\odot$ 表示逐元素乘法，$\epsilon$ 是防止除零的小常数。

**问题**：学习率单调递减，可能过早停止学习。

```python
class AdaGrad:
    def __init__(self, params, lr=0.01, eps=1e-8):
        self.params = list(params)
        self.lr = lr
        self.eps = eps
        
        # 初始化梯度平方累积
        self.sum_squared_grads = {}
        for param in self.params:
            self.sum_squared_grads[id(param)] = torch.zeros_like(param.data)
    
    def step(self):
        for param in self.params:
            if param.grad is not None:
                param_id = id(param)
                
                # G = G + g²
                self.sum_squared_grads[param_id] += param.grad.data ** 2
                
                # θ = θ - η/(√G + ε) * g
                adaptive_lr = self.lr / (torch.sqrt(self.sum_squared_grads[param_id]) + self.eps)
                param.data.add_(param.grad.data * adaptive_lr, alpha=-1)
```

### Adam算法

**动机**：结合动量和自适应学习率的优点。

**算法**：
$$\mathbf{m}_t = \beta_1 \mathbf{m}_{t-1} + (1-\beta_1)\nabla_\theta L(\theta_t)$$
$$\mathbf{v}_t = \beta_2 \mathbf{v}_{t-1} + (1-\beta_2)[\nabla_\theta L(\theta_t)]^2$$
$$\hat{\mathbf{m}}_t = \frac{\mathbf{m}_t}{1-\beta_1^t}$$
$$\hat{\mathbf{v}}_t = \frac{\mathbf{v}_t}{1-\beta_2^t}$$
$$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{\hat{\mathbf{v}}_t} + \epsilon}\hat{\mathbf{m}}_t$$

**偏差修正**：$\hat{\mathbf{m}}_t$ 和 $\hat{\mathbf{v}}_t$ 修正了初始化偏差。

```python
class Adam:
    def __init__(self, params, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.params = list(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0  # 时间步
        
        # 初始化一阶和二阶动量
        self.m = {}
        self.v = {}
        for param in self.params:
            self.m[id(param)] = torch.zeros_like(param.data)
            self.v[id(param)] = torch.zeros_like(param.data)
    
    def step(self):
        self.t += 1
        
        for param in self.params:
            if param.grad is not None:
                param_id = id(param)
                grad = param.grad.data
                
                # 更新一阶动量（梯度的指数移动平均）
                self.m[param_id] = self.beta1 * self.m[param_id] + (1 - self.beta1) * grad
                
                # 更新二阶动量（梯度平方的指数移动平均）
                self.v[param_id] = self.beta2 * self.v[param_id] + (1 - self.beta2) * grad.pow(2)
                
                # 偏差修正
                m_hat = self.m[param_id] / (1 - self.beta1 ** self.t)
                v_hat = self.v[param_id] / (1 - self.beta2 ** self.t)
                
                # 参数更新
                param.data.add_(m_hat / (torch.sqrt(v_hat) + self.eps), alpha=-self.lr)
```

### AdamW：权重衰减的修正

**问题**：传统Adam中的L2正则化与自适应学习率相互作用，导致次优效果。

**解决方案**：将权重衰减与梯度更新解耦。

**算法**：
$$\theta_{t+1} = \theta_t - \eta \left(\frac{\hat{\mathbf{m}}_t}{\sqrt{\hat{\mathbf{v}}_t} + \epsilon} + \lambda \theta_t\right)$$

其中 $\lambda$ 是权重衰减系数。

```python
class AdamW:
    def __init__(self, params, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.01):
        self.params = list(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0
        
        self.m = {}
        self.v = {}
        for param in self.params:
            self.m[id(param)] = torch.zeros_like(param.data)
            self.v[id(param)] = torch.zeros_like(param.data)
    
    def step(self):
        self.t += 1
        
        for param in self.params:
            if param.grad is not None:
                param_id = id(param)
                grad = param.grad.data
                
                # Adam更新
                self.m[param_id] = self.beta1 * self.m[param_id] + (1 - self.beta1) * grad
                self.v[param_id] = self.beta2 * self.v[param_id] + (1 - self.beta2) * grad.pow(2)
                
                m_hat = self.m[param_id] / (1 - self.beta1 ** self.t)
                v_hat = self.v[param_id] / (1 - self.beta2 ** self.t)
                
                # 解耦的权重衰减
                param.data.mul_(1 - self.lr * self.weight_decay)
                
                # 梯度更新
                param.data.add_(m_hat / (torch.sqrt(v_hat) + self.eps), alpha=-self.lr)
```

## 3.3 学习率调度策略

### 固定学习率的问题

- **过大**：训练不稳定，可能发散
- **过小**：收敛过慢，陷入局部最优

### 学习率衰减策略

#### 1. 阶梯衰减(Step Decay)

$$\eta_t = \eta_0 \cdot \gamma^{\lfloor t/s \rfloor}$$

其中 $s$ 是衰减间隔，$\gamma$ 是衰减因子。

```python
class StepLR:
    def __init__(self, optimizer, step_size, gamma=0.1):
        self.optimizer = optimizer
        self.step_size = step_size
        self.gamma = gamma
        self.last_epoch = 0
    
    def step(self):
        if self.last_epoch % self.step_size == 0 and self.last_epoch > 0:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] *= self.gamma
        self.last_epoch += 1
```

#### 2. 指数衰减(Exponential Decay)

$$\eta_t = \eta_0 \cdot e^{-\lambda t}$$

```python
class ExponentialLR:
    def __init__(self, optimizer, gamma):
        self.optimizer = optimizer
        self.gamma = gamma
    
    def step(self):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] *= self.gamma
```

#### 3. 余弦退火(Cosine Annealing)

$$\eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})\left(1 + \cos\left(\frac{t}{T}\pi\right)\right)$$

**优势**：平滑衰减，有利于收敛到更好的局部最优。

```python
class CosineAnnealingLR:
    def __init__(self, optimizer, T_max, eta_min=0):
        self.optimizer = optimizer
        self.T_max = T_max
        self.eta_min = eta_min
        self.base_lr = optimizer.param_groups[0]['lr']
        self.last_epoch = 0
    
    def step(self):
        lr = self.eta_min + (self.base_lr - self.eta_min) * \
             (1 + math.cos(math.pi * self.last_epoch / self.T_max)) / 2
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        self.last_epoch += 1
```

#### 4. 暖启动(Warm-up)

在训练初期使用较小的学习率，然后逐渐增加到目标值。

**动机**：防止训练初期的梯度爆炸。

```python
class WarmupLR:
    def __init__(self, optimizer, warmup_steps, target_lr):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.target_lr = target_lr
        self.current_step = 0
    
    def step(self):
        if self.current_step < self.warmup_steps:
            lr = self.target_lr * (self.current_step + 1) / self.warmup_steps
        else:
            lr = self.target_lr
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        self.current_step += 1
```

## 3.4 梯度裁剪与数值稳定性

### 梯度爆炸问题

在深层网络中，梯度可能呈指数增长，导致参数更新过大。

**检测方法**：
```python
def check_gradient_explosion(model, threshold=10.0):
    """检测梯度爆炸"""
    total_norm = 0
    param_count = 0
    
    for param in model.parameters():
        if param.grad is not None:
            param_norm = param.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
            param_count += 1
    
    total_norm = total_norm ** (1. / 2)
    
    print(f"梯度范数: {total_norm:.4f}")
    if total_norm > threshold:
        print("⚠️  检测到梯度爆炸！")
    
    return total_norm
```

### 梯度裁剪策略

#### 1. 范数裁剪(Norm Clipping)

$$\mathbf{g} \leftarrow \begin{cases}
\mathbf{g} & \text{if } \|\mathbf{g}\| \leq \tau \\
\frac{\tau}{\|\mathbf{g}\|} \mathbf{g} & \text{if } \|\mathbf{g}\| > \tau
\end{cases}$$

```python
def clip_grad_norm(parameters, max_norm, norm_type=2):
    """梯度范数裁剪"""
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    
    if len(parameters) == 0:
        return 0.0
    
    device = parameters[0].grad.device
    
    # 计算总梯度范数
    if norm_type == float('inf'):
        total_norm = max(p.grad.data.abs().max() for p in parameters)
        total_norm = total_norm.to(device)
    else:
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.data, norm_type) 
                                           for p in parameters]), norm_type)
    
    # 裁剪
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        for p in parameters:
            p.grad.data.mul_(clip_coef)
    
    return total_norm.item()
```

#### 2. 值裁剪(Value Clipping)

$$g_i \leftarrow \text{clip}(g_i, -\tau, \tau)$$

```python
def clip_grad_value(parameters, clip_value):
    """梯度值裁剪"""
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    
    for p in filter(lambda p: p.grad is not None, parameters):
        p.grad.data.clamp_(-clip_value, clip_value)
```

## 3.5 实践：MiniGPT中的优化配置

### 优化器选择与配置

```python
# MiniGPT中的优化器配置 (src/training/trainer.py)
class PreTrainer:
    def __init__(self, model, tokenizer, device='cpu'):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        
        # AdamW优化器：权重衰减解耦
        self.optimizer = optim.AdamW(
            model.parameters(), 
            lr=1e-4,           # 基础学习率
            weight_decay=0.01, # 权重衰减
            betas=(0.9, 0.999), # 动量参数
            eps=1e-8           # 数值稳定性
        )
        
        # 余弦退火调度器
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, 
            T_max=1000  # 总训练步数
        )
```

### 训练循环中的优化实践

```python
def train_epoch(self, dataloader):
    """优化实践的训练循环"""
    self.model.train()
    total_loss = 0
    
    for batch_idx, batch in enumerate(dataloader):
        # 数据准备
        input_ids, labels = self.prepare_batch(batch)
        
        # 前向传播
        logits = self.model(input_ids)
        loss = self.compute_loss(logits, labels)
        
        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        
        # 梯度裁剪（防止梯度爆炸）
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), 
            max_norm=1.0
        )
        
        # 参数更新
        self.optimizer.step()
        self.scheduler.step()  # 学习率调度
        
        # 监控指标
        if batch_idx % 100 == 0:
            current_lr = self.scheduler.get_last_lr()[0]
            print(f"Batch {batch_idx}: Loss={loss:.4f}, "
                  f"LR={current_lr:.6f}, GradNorm={grad_norm:.4f}")
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)
```

### 优化器状态监控

```python
def monitor_optimizer_state(optimizer, step):
    """监控优化器内部状态"""
    if hasattr(optimizer, 'state') and len(optimizer.state) > 0:
        # 获取第一个参数的状态作为代表
        first_param = next(iter(optimizer.param_groups[0]['params']))
        state = optimizer.state[first_param]
        
        if 'exp_avg' in state:  # Adam类优化器
            # 一阶动量统计
            m_norm = state['exp_avg'].norm().item()
            # 二阶动量统计
            v_norm = state['exp_avg_sq'].norm().item()
            
            print(f"Step {step}: 一阶动量范数={m_norm:.6f}, "
                  f"二阶动量范数={v_norm:.6f}")
```

## 小结与思考

本节介绍了深度学习优化的核心理论和实践：

1. **梯度下降**是参数优化的基础算法
2. **自适应算法**（Adam/AdamW）适应不同参数的更新需求
3. **学习率调度**控制训练的节奏和收敛质量
4. **梯度裁剪**保证训练的数值稳定性
5. **优化配置**需要根据具体问题精心调整

**思考题**：
1. 为什么AdamW比Adam在大模型训练中表现更好？
2. 如何从损失函数的几何性质理解不同优化算法的行为？
3. 学习率调度策略如何影响模型的最终性能？

**下一节预告**：我们将学习统计学习理论，理解模型泛化能力的数学基础。

---

*优化理论为深度学习提供了寻找最优解的指南针，而实践经验让我们在复杂的参数空间中找到前进的方向。* 🎯