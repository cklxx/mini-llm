"""
现代激活函数实现
支持业界主流的激活函数：SwiGLU、GELU、Mish、xIELU等
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class GELU(nn.Module):
    """
    GELU激活函数 - GPT、BERT等模型的标准选择

    GELU(x) = x * Φ(x) = x * 0.5 * (1 + erf(x / √2))
    """
    def __init__(self, approximate: str = "none"):
        super().__init__()
        self.approximate = approximate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(x, approximate=self.approximate)


class Mish(nn.Module):
    """
    Mish激活函数 - 在某些任务上优于ReLU和Swish

    Mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + e^x))
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.mish(x)


class xIELU(nn.Module):
    """
    xIELU激活函数 - 2024年新提出的激活函数
    在某些LLM任务上优于SwiGLU和ReLU2

    xIELU(x) = x * sigmoid(x) if x > 0 else alpha * (exp(x) - 1)
    """
    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        positive_mask = x > 0
        positive_part = x * torch.sigmoid(x)
        negative_part = self.alpha * (torch.exp(x) - 1)
        return torch.where(positive_mask, positive_part, negative_part)


class SwiGLU(nn.Module):
    """
    SwiGLU激活函数 - LLaMA、Mixtral等现代LLM的首选

    SwiGLU(x, gate) = Swish(gate) ⊙ x = (gate * sigmoid(gate)) ⊙ x
    """
    def forward(self, x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        return F.silu(gate) * x


class GLU(nn.Module):
    """
    标准GLU (Gated Linear Unit)

    GLU(x, gate) = x ⊙ σ(gate)
    """
    def forward(self, x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(gate)


class ReGLU(nn.Module):
    """
    ReGLU激活函数 - 使用ReLU而不是sigmoid

    ReGLU(x, gate) = x ⊙ ReLU(gate)
    """
    def forward(self, x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        return x * F.relu(gate)


class GEGLU(nn.Module):
    """
    GEGLU激活函数 - 使用GELU

    GEGLU(x, gate) = x ⊙ GELU(gate)
    """
    def forward(self, x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        return x * F.gelu(gate)


class SwiGLUFeedForward(nn.Module):
    """
    SwiGLU前馈网络 - 现代LLM的标准前馈层
    """
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.1, bias: bool = False):
        super().__init__()
        self.w_gate = nn.Linear(dim, hidden_dim, bias=bias)
        self.w_up = nn.Linear(dim, hidden_dim, bias=bias)
        self.w_down = nn.Linear(hidden_dim, dim, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.w_gate(x)
        up = self.w_up(x)
        gated_output = F.silu(gate) * up
        return self.w_down(self.dropout(gated_output))


class GEGLUFeedForward(nn.Module):
    """
    GEGLU前馈网络 - 在某些任务上表现良好
    """
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.1, bias: bool = False):
        super().__init__()
        self.w_gate = nn.Linear(dim, hidden_dim, bias=bias)
        self.w_up = nn.Linear(dim, hidden_dim, bias=bias)
        self.w_down = nn.Linear(hidden_dim, dim, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.w_gate(x)
        up = self.w_up(x)
        gated_output = F.gelu(gate) * up
        return self.w_down(self.dropout(gated_output))


class StandardFeedForward(nn.Module):
    """
    标准前馈网络 - 支持多种激活函数
    """
    def __init__(self, dim: int, hidden_dim: int, activation: str = "relu", dropout: float = 0.1, bias: bool = True):
        super().__init__()
        self.linear1 = nn.Linear(dim, hidden_dim, bias=bias)
        self.linear2 = nn.Linear(hidden_dim, dim, bias=bias)
        self.dropout = nn.Dropout(dropout)

        # 选择激活函数
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "gelu":
            self.activation = GELU()
        elif activation == "mish":
            self.activation = Mish()
        elif activation == "swish" or activation == "silu":
            self.activation = nn.SiLU()
        elif activation == "xielu":
            self.activation = xIELU()
        else:
            raise ValueError(f"Unsupported activation function: {activation}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        return self.linear2(x)


def get_activation_function(name: str) -> nn.Module:
    """
    获取激活函数实例

    Args:
        name: 激活函数名称

    Returns:
        激活函数实例
    """
    activation_map = {
        "relu": nn.ReLU(),
        "gelu": GELU(),
        "mish": Mish(),
        "swish": nn.SiLU(),
        "silu": nn.SiLU(),
        "xielu": xIELU(),
        "tanh": nn.Tanh(),
        "sigmoid": nn.Sigmoid(),
        "leaky_relu": nn.LeakyReLU(),
        "elu": nn.ELU(),
    }

    if name not in activation_map:
        raise ValueError(f"Unsupported activation function: {name}. "
                        f"Supported functions: {list(activation_map.keys())}")

    return activation_map[name]


def get_feedforward_layer(
    dim: int,
    hidden_dim: int,
    feedforward_type: str = "swiglu",
    activation: str = "relu",
    dropout: float = 0.1,
    bias: bool = False
) -> nn.Module:
    """
    获取前馈层实例

    Args:
        dim: 输入/输出维度
        hidden_dim: 隐藏层维度
        feedforward_type: 前馈层类型 ("swiglu", "geglu", "standard")
        activation: 激活函数名称（仅用于standard类型）
        dropout: dropout率
        bias: 是否使用bias

    Returns:
        前馈层实例
    """
    if feedforward_type == "swiglu":
        return SwiGLUFeedForward(dim, hidden_dim, dropout, bias)
    elif feedforward_type == "geglu":
        return GEGLUFeedForward(dim, hidden_dim, dropout, bias)
    elif feedforward_type == "standard":
        return StandardFeedForward(dim, hidden_dim, activation, dropout, bias)
    else:
        raise ValueError(f"Unsupported feedforward type: {feedforward_type}. "
                        f"Supported types: ['swiglu', 'geglu', 'standard']")
