"""
现代优化器实现
包含业界主流的优化器：AdamW、Lion、Sophia等
"""
import math
import torch
from torch.optim.optimizer import Optimizer
from typing import Dict, Any, Optional, List


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
                        f"Supported schedulers: ['cosine', 'warmup_cosine', 'linear', 'step', 'none']")