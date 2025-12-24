from __future__ import annotations

import mlx.core as mx
import mlx.optimizers as optim


class AdamWFP32State(optim.AdamW):
    """
    AdamW that keeps optimizer state (m/v) in float32.

    This is more numerically stable but uses substantially more memory than the
    default MLX AdamW, which stores state in the parameter dtype.
    """

    def init_single(self, parameter: mx.array, state: dict):
        state["m"] = mx.zeros(parameter.shape, dtype=mx.float32)
        state["v"] = mx.zeros(parameter.shape, dtype=mx.float32)

    def apply_single(self, gradient: mx.array, parameter: mx.array, state: dict):
        # MLX's default AdamW keeps state in param dtype; force float32 state.
        if "m" in state and isinstance(state["m"], mx.array) and state["m"].dtype != mx.float32:
            state["m"] = state["m"].astype(mx.float32)
        if "v" in state and isinstance(state["v"], mx.array) and state["v"].dtype != mx.float32:
            state["v"] = state["v"].astype(mx.float32)
        return super().apply_single(gradient, parameter, state)


def make_optimizer(
    *,
    name: str,
    learning_rate: float,
    weight_decay: float,
    state_dtype: str = "float32",
) -> optim.Optimizer:
    """
    Create an MLX optimizer.

    Args:
        name: One of: adamw, adafactor, lion.
        state_dtype: For AdamW only: float32 (stable, memory-heavy) or param
          (stores state in parameter dtype; faster/cheaper).
    """
    name = str(name).lower().strip()
    state_dtype = str(state_dtype).lower().strip()

    if name == "adamw":
        if state_dtype == "float32":
            return AdamWFP32State(
                learning_rate=float(learning_rate),
                weight_decay=float(weight_decay),
            )
        if state_dtype in {"param", "params", "parameter"}:
            return optim.AdamW(
                learning_rate=float(learning_rate),
                weight_decay=float(weight_decay),
            )
        raise ValueError(f"Unknown AdamW state_dtype: {state_dtype} (use float32|param)")

    if name == "adafactor":
        # Make it behave more like a traditional optimizer with a fixed LR.
        return optim.Adafactor(
            learning_rate=float(learning_rate),
            weight_decay=float(weight_decay),
            relative_step=False,
            scale_parameter=False,
        )

    if name == "lion":
        return optim.Lion(
            learning_rate=float(learning_rate),
            weight_decay=float(weight_decay),
        )

    raise ValueError(f"Unknown optimizer: {name} (use adamw|adafactor|lion)")

