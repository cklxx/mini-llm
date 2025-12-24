from __future__ import annotations

import argparse
import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import mlx.core as mx

from .config import MiniLLMConfig, minillm_200mb
from .model import MiniLLMForCausalLM, count_parameters


def _load_config(checkpoint_dir: Path) -> MiniLLMConfig:
    config_path = checkpoint_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config.json in checkpoint dir: {checkpoint_dir}")
    with open(config_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"config.json must be an object, got: {type(data).__name__}")
    return MiniLLMConfig.from_dict(data)


def _fmt_int(n: int) -> str:
    return f"{n:,}"


def _fmt_tflops(flops: float) -> str:
    return f"{flops / 1e12:.3f}"


def _fmt_gflops(flops: float) -> str:
    return f"{flops / 1e9:.3f}"


def _estimate_forward_flops(
    *,
    cfg: MiniLLMConfig,
    batch_size: int,
    seq_len: int,
    include_lm_head: bool,
) -> Tuple[float, float]:
    """
    Return (total_forward_flops, forward_flops_per_token) for a full-sequence forward.

    Notes:
    - Counts multiply-add as 2 FLOPs (GEMM convention).
    - Assumes full attention O(T^2) (training / prefill), not KV-cached decode.
    - Ignores small elementwise costs (norm/activation/rope/softmax).
    """

    cfg = cfg.finalize()
    h = int(cfg.hidden_size)
    l = int(cfg.num_hidden_layers)
    i = int(cfg.intermediate_size or 0)
    v = int(cfg.vocab_size)
    n_heads = int(cfg.num_attention_heads)
    n_kv = int(cfg.num_key_value_heads or n_heads)
    head_dim = h // n_heads
    h_kv = n_kv * head_dim

    # Per layer, per token:
    # - Q,O projections: 2*H*H each -> 4*H^2
    # - K,V projections: 2*H*H_kv each -> 4*H*H_kv
    # - MLP (gate/up/down): 2*H*I + 2*H*I + 2*I*H -> 6*H*I
    # - Attention matmuls (QK^T + AV): 4*T*H
    per_token_per_layer = (4 * h * h) + (4 * h * h_kv) + (6 * h * i) + (4 * seq_len * h)
    per_token = l * per_token_per_layer
    if include_lm_head:
        # logits = H @ W_vocab^T : 2*H*V per token
        per_token += 2 * h * v

    tokens = int(batch_size) * int(seq_len)
    total = float(tokens) * float(per_token)
    return total, float(per_token)


def _estimate_train_flops(
    *,
    forward_flops: float,
    accum_steps: int,
    train_mult: float,
) -> float:
    """
    Approx training FLOPs per optimizer step.

    `train_mult` is a heuristic multiplier converting forward FLOPs to (forward+backward).
    Typical rule-of-thumb for dense GEMMs is ~3x.
    """
    return float(forward_flops) * float(train_mult) * int(accum_steps)


@dataclass(frozen=True)
class DevicePeak:
    chip: str
    gpu_cores: Optional[int]
    fp32_tflops: Optional[float]
    fp16_tflops: Optional[float]


def _parse_gpu_cores_from_system_profiler(text: str) -> Optional[int]:
    # Matches: "Total Number of Cores: 16" under Graphics/Displays.
    # Use the last occurrence to prefer the GPU section over CPU section.
    cores: Optional[int] = None
    for m in re.finditer(r"Total Number of Cores:\s*(\d+)\b", text):
        try:
            cores = int(m.group(1))
        except ValueError:
            continue
    return cores


def _detect_device_peak() -> DevicePeak:
    info = mx.metal.device_info()
    chip = str(info.get("device_name") or "unknown")

    gpu_cores: Optional[int] = None
    fp32: Optional[float] = None
    fp16: Optional[float] = None

    if chip.startswith("Apple "):
        try:
            sp = subprocess.check_output(
                ["system_profiler", "SPHardwareDataType", "SPDisplaysDataType"],
                text=True,
                stderr=subprocess.DEVNULL,
            )
            gpu_cores = _parse_gpu_cores_from_system_profiler(sp)
        except Exception:
            gpu_cores = None

        # Community-reported peak FP32 per GPU core (TFLOPs/core) for Apple Silicon.
        # M1-family: 8-core ~= 2.6 TFLOPs -> 0.325 TFLOPs/core.
        per_core_fp32: Optional[float] = None
        if "M1" in chip:
            per_core_fp32 = 0.325
        elif "M2" in chip:
            per_core_fp32 = 0.360
        elif "M3" in chip:
            per_core_fp32 = 0.360

        if per_core_fp32 is not None and gpu_cores is not None:
            fp32 = per_core_fp32 * gpu_cores
            fp16 = 2.0 * fp32

    return DevicePeak(chip=chip, gpu_cores=gpu_cores, fp32_tflops=fp32, fp16_tflops=fp16)


def main() -> None:
    parser = argparse.ArgumentParser(description="MiniLLM (MLX) FLOPs estimator + device peak TFLOPs")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Checkpoint directory (or model.safetensors) to load config from. If omitted, use preset config.",
    )
    parser.add_argument(
        "--preset",
        type=str,
        choices=["200mb"],
        default="200mb",
        help="Preset to use when --checkpoint is not set.",
    )
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--seq_len", type=int, default=1024)
    parser.add_argument("--accum_steps", type=int, default=1, help="Only used for train-step FLOPs.")
    parser.add_argument(
        "--train_mult",
        type=float,
        default=3.0,
        help="Heuristic multiplier: forward FLOPs -> (forward+backward) FLOPs.",
    )
    parser.add_argument(
        "--include_lm_head",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include final logits matmul (H x vocab).",
    )
    parser.add_argument(
        "--tok_s",
        type=float,
        default=None,
        help="If set, also report estimated achieved TFLOPs/s at this token throughput.",
    )
    parser.add_argument(
        "--step_s",
        type=float,
        default=None,
        help="If set, also report achieved TFLOPs/s from optimizer-step wall time (seconds).",
    )
    parser.add_argument(
        "--step_ms",
        type=float,
        default=None,
        help="Same as --step_s but in milliseconds.",
    )

    args = parser.parse_args()
    if args.step_s is not None and args.step_ms is not None:
        raise ValueError("Pass only one of --step_s / --step_ms")

    if args.checkpoint:
        ckpt = Path(args.checkpoint)
        if ckpt.is_file() and ckpt.name.endswith(".safetensors"):
            ckpt = ckpt.parent
        cfg = _load_config(ckpt)
        ckpt_msg = str(ckpt)
    else:
        cfg = minillm_200mb()
        ckpt_msg = "(preset 200mb)"

    model = MiniLLMForCausalLM(cfg)
    n_params = count_parameters(model.parameters())

    fwd_total, fwd_per_tok = _estimate_forward_flops(
        cfg=cfg,
        batch_size=int(args.batch_size),
        seq_len=int(args.seq_len),
        include_lm_head=bool(args.include_lm_head),
    )
    train_step = _estimate_train_flops(
        forward_flops=fwd_total,
        accum_steps=int(args.accum_steps),
        train_mult=float(args.train_mult),
    )

    print(f"[model] cfg_from={ckpt_msg}")
    print(f"[model] params={_fmt_int(n_params)} ({n_params/1e9:.6f}B)")
    print(
        "[fwd] "
        f"batch={args.batch_size} seq_len={args.seq_len} tokens={args.batch_size*args.seq_len} "
        f"flops={_fmt_tflops(fwd_total)} TFLOPs ({_fmt_gflops(fwd_total)} GFLOPs) "
        f"per_token={_fmt_gflops(fwd_per_tok)} GFLOPs"
    )
    print(
        "[train_step] "
        f"accum_steps={args.accum_steps} train_mult={args.train_mult:g} "
        f"flops={_fmt_tflops(train_step)} TFLOPs ({_fmt_gflops(train_step)} GFLOPs) (per optimizer step)"
    )

    dev = _detect_device_peak()
    print(f"[device] {dev.chip}")
    if dev.gpu_cores is not None:
        print(f"[device] gpu_cores={dev.gpu_cores}")
    if dev.fp32_tflops is not None and dev.fp16_tflops is not None:
        print(f"[device] peak_fp32≈{dev.fp32_tflops:.2f} TFLOPs  peak_fp16≈{dev.fp16_tflops:.2f} TFLOPs")
    else:
        print("[device] peak TFLOPs: unknown (unsupported chip mapping)")

    achieved: Optional[float] = None
    if args.tok_s is not None and args.tok_s > 0:
        achieved = (fwd_per_tok * float(args.train_mult) * float(args.tok_s)) / 1e12
        print(f"[throughput] tok/s={args.tok_s:.0f} => est={achieved:.3f} TFLOPs/s (train)")
    elif args.step_s is not None or args.step_ms is not None:
        step_s = float(args.step_s) if args.step_s is not None else (float(args.step_ms) / 1000.0)
        if step_s <= 0:
            raise ValueError("--step_s/--step_ms must be > 0")
        achieved = (train_step / step_s) / 1e12
        print(f"[throughput] step_time={step_s:.3f}s => est={achieved:.3f} TFLOPs/s (train)")

    if achieved is not None and dev.fp16_tflops is not None and dev.fp16_tflops > 0:
        util_fp16 = 100.0 * achieved / float(dev.fp16_tflops)
        print(f"[util] vs peak_fp16 ≈ {util_fp16:.1f}%")
    if achieved is not None and dev.fp32_tflops is not None and dev.fp32_tflops > 0:
        util_fp32 = 100.0 * achieved / float(dev.fp32_tflops)
        print(f"[util] vs peak_fp32 ≈ {util_fp32:.1f}%")


if __name__ == "__main__":
    main()
