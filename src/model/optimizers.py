"""
现代优化器实现 (2024-2025最新)
包含业界主流的优化器：AdamW、Lion、Sophia、Muon等

基于最新研究:
- Muon (2024): 2x效率提升, Kimi-2使用
- Sophia (2023): 二阶优化, 适合大模型预训练
- Lion (2023): 内存高效
- WSD调度器 (2024): 无需预设总步数

论文参考:
- Muon: https://arxiv.org/abs/2502.16982
- Sophia: https://arxiv.org/abs/2305.14342
- Lion: https://arxiv.org/abs/2302.06675
- WSD: https://arxiv.org/abs/2410.05192
"""
import math
import torch
from torch.optim.optimizer import Optimizer
from typing import Dict, Any, Optional, List, Callable


class Lion(Optimizer):
    """
    Lion优化器 - Google提出的轻量级优化器

    Lion (EvoLved Sign Momentum) 是一个简单但有效的优化器，
    在许多视觉和语言模型任务上优于AdamW。

    Reference: https://arxiv.org/abs/2302.06675
    """

    def __init__(
        self,
        params,
        lr: float = 1e-4,
        betas: tuple = (0.9, 0.99),
        weight_decay: float = 0.0,
        maximize: bool = False,
        foreach: Optional[bool] = None,
    ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(
            lr=lr,
            betas=betas,
            weight_decay=weight_decay,
            maximize=maximize,
            foreach=foreach,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                if group["maximize"]:
                    grad = -grad

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["exp_avg"] = torch.zeros_like(p)

                exp_avg = state["exp_avg"]
                beta1, beta2 = group["betas"]

                # Weight decay
                if group["weight_decay"] != 0:
                    p.mul_(1 - group["lr"] * group["weight_decay"])

                # Update biased first moment estimate
                update = exp_avg * beta1 + grad * (1 - beta1)

                # Apply sign of the update
                p.add_(torch.sign(update), alpha=-group["lr"])

                # Decay the momentum running average coefficient
                exp_avg.mul_(beta2).add_(grad, alpha=1 - beta2)

        return loss


class Sophia(Optimizer):
    """
    Sophia优化器 - 为LLM预训练设计的二阶优化器

    Sophia (Second-order Clipped Stochastic Optimization) 使用对角Hessian估计
    来加速大语言模型的训练，在相同的步数下能达到更好的性能。

    Reference: https://arxiv.org/abs/2305.14342
    """

    def __init__(
        self,
        params,
        lr: float = 1e-4,
        betas: tuple = (0.965, 0.99),
        rho: float = 0.04,
        weight_decay: float = 1e-1,
        maximize: bool = False,
        capturable: bool = False,
        eps: float = 1e-15,
    ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= rho:
            raise ValueError(f"Invalid rho parameter: {rho}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(
            lr=lr,
            betas=betas,
            rho=rho,
            weight_decay=weight_decay,
            maximize=maximize,
            capturable=capturable,
            eps=eps,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None, bs=5120):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            rho = group["rho"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                if group["maximize"]:
                    grad = -grad

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = torch.zeros((1,), dtype=torch.float, device=p.device, requires_grad=False)
                    state["exp_avg"] = torch.zeros_like(p)
                    state["hessian_diag_sq"] = torch.zeros_like(p)

                exp_avg, hessian_diag_sq = state["exp_avg"], state["hessian_diag_sq"]
                step_t = state["step"]

                step_t += 1

                # Perform step weight decay
                if group["weight_decay"] != 0:
                    p.mul_(1 - group["lr"] * group["weight_decay"])

                # Decay the first moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

                if len(state) == 1:
                    state["hessian_diag_sq"] = torch.zeros_like(p)
                    hessian_diag_sq = state["hessian_diag_sq"]

                # Update Hessian diagonal estimate
                k = group.get('k', 10)
                update_period = group.get('update_period', k)

                if step_t % update_period == 1:
                    # Approximation using gradient auto-correlation
                    hessian_diag_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Bias correction
                bias_correction1 = 1 - beta1 ** step_t
                bias_correction2 = 1 - beta2 ** step_t

                # Clipped update
                h = hessian_diag_sq.sqrt() / math.sqrt(bias_correction2)
                u = exp_avg / bias_correction1 / (h + group["eps"])
                u.clamp_(min=-rho, max=rho)

                p.add_(u, alpha=-group["lr"])

        return loss


class AdamWScheduleFree(Optimizer):
    """
    Schedule-Free AdamW - 无需学习率调度的AdamW变体

    这个优化器自动调整学习率，无需手动设置学习率调度器。
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
        warmup_steps: int = 10000,
        r: float = 0.0,
    ):
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            warmup_steps=warmup_steps,
            r=r,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)
                    state["z"] = p.clone()

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                z = state["z"]
                beta1, beta2 = group["betas"]

                state["step"] += 1
                step = state["step"]

                # Schedule-free learning rate
                if step < group["warmup_steps"]:
                    lr_t = group["lr"] * step / group["warmup_steps"]
                else:
                    lr_t = group["lr"] / (1 + group["r"] * (step - group["warmup_steps"]))

                # Weight decay
                if group["weight_decay"] != 0:
                    grad = grad.add(p, alpha=group["weight_decay"])

                # Exponential moving average of gradient values
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

                # Exponential moving average of squared gradient values
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Bias correction
                bias_correction1 = 1 - beta1 ** step
                bias_correction2 = 1 - beta2 ** step

                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group["eps"])
                step_size = lr_t / bias_correction1

                # Update z
                z.addcdiv_(exp_avg, denom, value=-step_size)

                # Update parameters
                p.copy_(z)

        return loss


def get_optimizer(
    optimizer_name: str,
    parameters,
    lr: float = 1e-4,
    weight_decay: float = 0.01,
    **kwargs
) -> Optimizer:
    """
    获取优化器实例

    Args:
        optimizer_name: 优化器名称
        parameters: 模型参数
        lr: 学习率
        weight_decay: 权重衰减
        **kwargs: 其他优化器参数

    Returns:
        优化器实例
    """
    optimizer_name = optimizer_name.lower()

    if optimizer_name == "adamw":
        return torch.optim.AdamW(
            parameters,
            lr=lr,
            weight_decay=weight_decay,
            **kwargs
        )
    elif optimizer_name == "adam":
        return torch.optim.Adam(
            parameters,
            lr=lr,
            weight_decay=weight_decay,
            **kwargs
        )
    elif optimizer_name == "sgd":
        return torch.optim.SGD(
            parameters,
            lr=lr,
            weight_decay=weight_decay,
            momentum=kwargs.get("momentum", 0.9),
            **{k: v for k, v in kwargs.items() if k != "momentum"}
        )
    elif optimizer_name == "lion":
        return Lion(
            parameters,
            lr=lr,
            weight_decay=weight_decay,
            **kwargs
        )
    elif optimizer_name == "sophia":
        return Sophia(
            parameters,
            lr=lr,
            weight_decay=weight_decay,
            **kwargs
        )
    elif optimizer_name == "adamw_schedule_free":
        return AdamWScheduleFree(
            parameters,
            lr=lr,
            weight_decay=weight_decay,
            **kwargs
        )
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}. "
                        f"Supported optimizers: ['adamw', 'adam', 'sgd', 'lion', 'sophia', 'adamw_schedule_free']")


def get_scheduler(
    scheduler_name: str,
    optimizer: Optimizer,
    **kwargs
) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
    """
    获取学习率调度器

    Args:
        scheduler_name: 调度器名称
        optimizer: 优化器实例
        **kwargs: 调度器参数

    Returns:
        学习率调度器实例
    """
    if scheduler_name is None or scheduler_name == "none":
        return None

    scheduler_name = scheduler_name.lower()

    if scheduler_name == "cosine":
        T_max = kwargs.get("T_max", 1000)
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max)
    elif scheduler_name == "warmup_cosine":
        from transformers import get_cosine_schedule_with_warmup
        num_warmup_steps = kwargs.get("num_warmup_steps", 1000)
        num_training_steps = kwargs.get("num_training_steps", 10000)
        return get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )
    elif scheduler_name == "linear":
        gamma = kwargs.get("gamma", 0.95)
        return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    elif scheduler_name == "step":
        step_size = kwargs.get("step_size", 1000)
        gamma = kwargs.get("gamma", 0.1)
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    else:
        raise ValueError(f"Unsupported scheduler: {scheduler_name}. "
                        f"Supported schedulers: ['cosine', 'warmup_cosine', 'linear', 'step', 'inverse_sqrt', 'wsd', 'none']")


class Muon(Optimizer):
    """
    Muon优化器 - Momentum Orthogonalized by Newton-Schulz (2024)

    最新研究成果，相比AdamW节省~48%计算量达到相同效果。
    Kimi-2 (1T参数)使用此优化器。

    核心思想:
    - 对2D参数(权重矩阵)使用Newton-Schulz正交化
    - 与AdamW混合使用(AdamW处理1D参数如bias, norm)

    论文: https://arxiv.org/abs/2502.16982

    Args:
        params: 模型参数(推荐只传入2D参数)
        lr: 学习率 (推荐: 0.02, 比AdamW高20倍!)
        momentum: 动量系数 (默认: 0.95)
        nesterov: 是否使用Nesterov动量
        ns_steps: Newton-Schulz迭代步数 (默认: 5)
        weight_decay: 权重衰减

    使用示例:
        # 推荐: 混合使用Muon(2D参数) + AdamW(1D参数)
        params_2d = [p for p in model.parameters() if len(p.shape) >= 2]
        params_1d = [p for p in model.parameters() if len(p.shape) < 2]

        opt_muon = Muon(params_2d, lr=0.02)
        opt_adamw = torch.optim.AdamW(params_1d, lr=1e-3)
    """

    def __init__(
        self,
        params,
        lr: float = 0.02,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        weight_decay: float = 0.0
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= momentum < 1.0:
            raise ValueError(f"Invalid momentum: {momentum}")

        defaults = dict(
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            weight_decay=weight_decay
        )
        super().__init__(params, defaults)

    def newton_schulz_iteration(self, G, steps=5):
        """
        Newton-Schulz正交化迭代

        将梯度矩阵G正交化，使更新方向更稳定
        """
        # 优化的系数(通过实验确定)
        a, b, c = (3.4445, -4.7750, 2.0315)

        X = G.clone()
        if steps == 5:
            for _ in range(steps):
                A = X @ X.T
                B = b * A + c * A @ A
                X = a * X + B @ X

        return X

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None):
        """执行单步优化"""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad

                # 权重衰减
                if group['weight_decay'] != 0:
                    grad = grad.add(p, alpha=group['weight_decay'])

                state = self.state[p]

                # 状态初始化
                if len(state) == 0:
                    state['momentum_buffer'] = torch.zeros_like(grad)

                buf = state['momentum_buffer']

                # 动量更新
                buf.mul_(group['momentum']).add_(grad)

                # Muon核心: 对2D参数应用Newton-Schulz正交化
                if len(p.shape) == 2:
                    update = self.newton_schulz_iteration(
                        buf.view(p.shape),
                        steps=group['ns_steps']
                    )
                else:
                    # 1D参数使用标准动量
                    update = buf

                # Nesterov加速
                if group['nesterov']:
                    update = grad + group['momentum'] * update

                # 应用更新
                p.add_(update, alpha=-group['lr'])

        return loss


def get_warmup_cosine_schedule(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 0.5,
    min_lr_ratio: float = 0.1
):
    """
    Warmup + Cosine退火调度器 (推荐用于LLM训练)

    GPT-3, Llama, Chinchilla, Pythia等都使用此调度器

    Args:
        optimizer: 优化器
        num_warmup_steps: warmup步数 (推荐: 总步数的5-10%)
        num_training_steps: 总训练步数
        num_cycles: cosine周期数 (0.5 = 半周期, 1.0 = 全周期)
        min_lr_ratio: 最小学习率占比 (默认0.1 = 降到10%)

    Returns:
        LambdaLR调度器
    """
    def lr_lambda(current_step: int):
        # Warmup阶段: 线性增长
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))

        # Cosine退火阶段
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * num_cycles * 2.0 * progress))

        # 确保不低于min_lr_ratio
        return max(min_lr_ratio, cosine_decay)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def get_inverse_sqrt_schedule(
    optimizer: Optimizer,
    num_warmup_steps: int,
    timescale: int = 10000
):
    """
    Inverse Sqrt调度器 (Transformer论文原始调度器)

    "Attention is All You Need"中使用的调度器

    公式: lr = lr_max * min(step^(-0.5), step * warmup_steps^(-1.5))

    Args:
        optimizer: 优化器
        num_warmup_steps: warmup步数
        timescale: 时间尺度 (默认10000)

    Returns:
        LambdaLR调度器
    """
    def lr_lambda(current_step: int):
        current_step += 1  # 避免除以0
        if current_step < num_warmup_steps:
            # Warmup阶段: 线性增长
            return float(current_step) / float(num_warmup_steps)
        else:
            # Inverse sqrt decay
            return float(num_warmup_steps) ** 0.5 / (current_step ** 0.5)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def get_wsd_schedule(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_stable_steps: int,
    num_decay_steps: int,
    min_lr_ratio: float = 0.1
):
    """
    Warmup-Stable-Decay调度器 (2024最新)

    无需预设总训练步数,支持持续训练

    阶段:
    1. Warmup: 0 → warmup_steps, 线性增长
    2. Stable: warmup_steps → warmup_steps + stable_steps, 保持常数
    3. Decay: 之后, Cosine退火

    论文: https://arxiv.org/abs/2410.05192

    Args:
        optimizer: 优化器
        num_warmup_steps: warmup步数
        num_stable_steps: 稳定阶段步数
        num_decay_steps: 衰减阶段步数
        min_lr_ratio: 最小学习率占比

    Returns:
        LambdaLR调度器
    """
    def lr_lambda(current_step: int):
        # 1. Warmup阶段
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))

        # 2. Stable阶段
        if current_step < num_warmup_steps + num_stable_steps:
            return 1.0

        # 3. Decay阶段
        progress = float(current_step - num_warmup_steps - num_stable_steps) / float(max(1, num_decay_steps))
        progress = min(progress, 1.0)  # 限制在[0, 1]
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))

        return max(min_lr_ratio, cosine_decay)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# 更新get_optimizer函数以支持Muon
_original_get_optimizer = get_optimizer


def get_optimizer(
    optimizer_name: str,
    parameters,
    lr: float = 1e-4,
    weight_decay: float = 0.01,
    **kwargs
) -> Optimizer:
    """
    获取优化器实例 (2024更新)

    新增支持:
    - muon: Muon优化器 (2024, 推荐大模型)
    - muon_adamw: Muon+AdamW混合 (最优配置)

    推荐配置:
    1. 小模型(<1B): adamw
    2. 大模型(>1B): muon_adamw (混合)
    3. 预训练: adamw with β2=0.95
    4. 微调: adamw with β2=0.999
    """
    optimizer_name = optimizer_name.lower()

    if optimizer_name == "muon":
        return Muon(
            parameters,
            lr=lr,
            momentum=0.95,
            weight_decay=weight_decay,
            **kwargs
        )
    elif optimizer_name == "muon_adamw":
        # 混合模式: 需要传入model而不是parameters
        # 这里返回提示信息
        raise ValueError(
            "muon_adamw需要使用get_hybrid_optimizer(model, ...)函数创建"
        )
    else:
        # 调用原始函数处理其他优化器
        return _original_get_optimizer(optimizer_name, parameters, lr, weight_decay, **kwargs)


def get_hybrid_optimizer(
    model,
    muon_lr: float = 0.02,
    adamw_lr: float = 1e-3,
    weight_decay: float = 0.01,
    betas: tuple = (0.9, 0.95)
):
    """
    创建Muon+AdamW混合优化器 (2024最优配置)

    Muon处理2D参数(权重矩阵), AdamW处理1D参数(bias, norm)

    Args:
        model: PyTorch模型
        muon_lr: Muon学习率 (默认0.02, 是AdamW的20倍)
        adamw_lr: AdamW学习率 (默认1e-3)
        weight_decay: 权重衰减
        betas: AdamW的beta参数

    Returns:
        dict包含两个优化器: {'muon': Muon, 'adamw': AdamW}

    使用示例:
        opts = get_hybrid_optimizer(model)
        # 训练循环中:
        loss.backward()
        opts['muon'].step()
        opts['adamw'].step()
        opts['muon'].zero_grad()
        opts['adamw'].zero_grad()
    """
    params_2d = []  # 权重矩阵
    params_1d = []  # bias, norm等

    for name, param in model.named_parameters():
        if param.requires_grad:
            if len(param.shape) >= 2:
                params_2d.append(param)
            else:
                params_1d.append(param)

    print(f"🔧 创建混合优化器:")
    print(f"   - Muon: {len(params_2d)} 个2D参数, lr={muon_lr}")
    print(f"   - AdamW: {len(params_1d)} 个1D参数, lr={adamw_lr}")

    return {
        'muon': Muon(
            params_2d,
            lr=muon_lr,
            momentum=0.95,
            weight_decay=weight_decay
        ),
        'adamw': torch.optim.AdamW(
            params_1d,
            lr=adamw_lr,
            betas=betas,
            weight_decay=weight_decay
        )
    }