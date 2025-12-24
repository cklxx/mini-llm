from __future__ import annotations

from functools import lru_cache
from typing import Tuple

import mlx.core as mx
import mlx.core.fast as fast


_TG = 256

_ENABLED = True


def set_enabled(enabled: bool) -> None:
    global _ENABLED
    _ENABLED = bool(enabled)


def enabled() -> bool:
    return bool(_ENABLED)


@lru_cache(maxsize=128)
def _scalar_u32(value: int) -> mx.array:
    return mx.array([int(value)], dtype=mx.uint32)


@lru_cache(maxsize=128)
def _scalar_f32(value: float) -> mx.array:
    return mx.array([float(value)], dtype=mx.float32)


@lru_cache(maxsize=1)
def _rms_norm_kernel():
    source = r"""
        constexpr uint TG = 256;

        uint gid = thread_position_in_grid.x;
        uint row = gid / TG;
        uint lane = gid - row * TG;

        uint dimv = dim[0];
        float epsv = eps[0];

        uint base = row * dimv;

        threadgroup float partial[TG];
        float sum = 0.0f;

        for (uint j = lane; j < dimv; j += TG) {
            float v = float(inp[base + j]);
            sum += v * v;
        }

        partial[lane] = sum;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint stride = TG / 2; stride > 0; stride /= 2) {
            if (lane < stride) {
                partial[lane] += partial[lane + stride];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        float inv_rms = rsqrt(partial[0] / float(dimv) + epsv);

        for (uint j = lane; j < dimv; j += TG) {
            float v = float(inp[base + j]);
            float ww = float(w[j]);
            out[base + j] = T(v * inv_rms * ww);
        }
    """

    return fast.metal_kernel(
        name="mlx_train_rms_norm",
        input_names=["inp", "w", "dim", "eps"],
        output_names=["out"],
        source=source,
        ensure_row_contiguous=True,
    )


@lru_cache(maxsize=1)
def _silu_mul_kernel():
    source = r"""
        uint elem = thread_position_in_grid.x;
        float x = float(a[elem]);
        float y = float(b[elem]);
        float s = 1.0f / (1.0f + exp(-x));
        out[elem] = T((x * s) * y);
    """
    return fast.metal_kernel(
        name="mlx_train_silu_mul",
        input_names=["a", "b"],
        output_names=["out"],
        source=source,
        ensure_row_contiguous=True,
    )


@mx.custom_function
def _rms_norm(x: mx.array, w: mx.array, dim: mx.array, eps: mx.array) -> mx.array:
    if x.ndim < 1:
        raise ValueError("rms_norm expects x.ndim >= 1")
    dimv = int(x.shape[-1])
    if w.shape != (dimv,):
        raise ValueError(f"rms_norm weight shape mismatch: w={w.shape} vs dim={dimv}")

    x2 = x.reshape((-1, dimv))
    rows = int(x2.shape[0])

    out2 = _rms_norm_kernel()(
        inputs=[x2, w, dim, eps],
        template=[("T", x.dtype)],
        grid=(rows * _TG, 1, 1),
        threadgroup=(_TG, 1, 1),
        output_shapes=[x2.shape],
        output_dtypes=[x.dtype],
    )[0]
    return out2.reshape(x.shape)


@_rms_norm.vjp
def _rms_norm_vjp(
    primals: Tuple[mx.array, mx.array, mx.array, mx.array],
    cotangent: mx.array,
    output: mx.array,
):
    x, w, dim, eps = primals
    dy = cotangent

    x_f = x.astype(mx.float32)
    dy_f = dy.astype(mx.float32)
    w_f = w.astype(mx.float32)
    eps_f = eps.astype(mx.float32)

    d = int(x.shape[-1])
    r = mx.rsqrt(mx.mean(x_f * x_f, axis=-1, keepdims=True) + eps_f)
    gw = dy_f * w_f
    a = mx.sum(gw * x_f, axis=-1, keepdims=True)
    dx = gw * r - x_f * (r * r * r) * (a / float(d))

    reduce_axes = tuple(range(x.ndim - 1))
    dw = mx.sum(dy_f * x_f * r, axis=reduce_axes)

    return dx.astype(x.dtype), dw.astype(w.dtype), mx.zeros_like(dim), mx.zeros_like(eps)


@mx.custom_function
def _silu_mul(a: mx.array, b: mx.array) -> mx.array:
    if a.shape != b.shape:
        raise ValueError(f"silu_mul expects a.shape == b.shape, got a={a.shape} b={b.shape}")
    out = _silu_mul_kernel()(
        inputs=[a, b],
        template=[("T", a.dtype)],
        grid=(int(a.size), 1, 1),
        threadgroup=(_TG, 1, 1),
        output_shapes=[a.shape],
        output_dtypes=[a.dtype],
    )[0]
    return out


@_silu_mul.vjp
def _silu_mul_vjp(
    primals: Tuple[mx.array, mx.array],
    cotangent: mx.array,
    output: mx.array,
):
    x, y = primals
    dout = cotangent

    x_f = x.astype(mx.float32)
    y_f = y.astype(mx.float32)
    dout_f = dout.astype(mx.float32)

    sig = mx.sigmoid(x_f)
    silu = x_f * sig
    dsilu = sig * (1.0 + x_f * (1.0 - sig))

    dx = dout_f * y_f * dsilu
    dy = dout_f * silu
    return dx.astype(x.dtype), dy.astype(y.dtype)


def rms_norm(x: mx.array, w: mx.array, eps: float) -> mx.array:
    """
    RMSNorm via custom Metal kernel forward + custom VJP backward.
    """
    if w.dtype != x.dtype:
        w = w.astype(x.dtype)
    return _rms_norm(x, w, _scalar_u32(int(x.shape[-1])), _scalar_f32(float(eps)))


def silu_mul(a: mx.array, b: mx.array) -> mx.array:
    """
    SiLU(a) * b via custom Metal kernel forward + custom VJP backward.
    """
    if a.dtype != b.dtype:
        b = b.astype(a.dtype)
    return _silu_mul(a, b)
