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


def _load_config(checkpoint_dir: Path) -> MiniLLMConfig:
    config_path = checkpoint_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config.json in checkpoint dir: {checkpoint_dir}")
    with open(config_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return MiniLLMConfig(**data).finalize()


def _is_valid_checkpoint_dir(path: Path) -> bool:
    try:
        if not path.is_dir():
            return False
        cfg = path / "config.json"
        weights = path / "model.safetensors"
        if not cfg.is_file() or cfg.stat().st_size <= 0:
            return False
        if not weights.is_file() or weights.stat().st_size <= 0:
            return False
        return True
    except OSError:
        return False


def _iter_step_dirs(checkpoints_dir: Path) -> list[tuple[int, Path]]:
    step_re = re.compile(r"^step_(\d+)$")
    if not checkpoints_dir.exists() or not checkpoints_dir.is_dir():
        return []
    out: list[tuple[int, Path]] = []
    for child in checkpoints_dir.iterdir():
        if not child.is_dir():
            continue
        m = step_re.match(child.name)
        if not m:
            continue
        try:
            step = int(m.group(1))
        except ValueError:
            continue
        out.append((step, child))
    out.sort(key=lambda x: x[0])
    return out


def _find_latest_checkpoint(out_dir: Path) -> Optional[Path]:
    candidates = [
        out_dir / "sft" / "checkpoints",
        out_dir / "pretrain" / "checkpoints",
        out_dir / "checkpoints",
    ]
    for ckpts in candidates:
        best: Optional[tuple[int, Path]] = None
        for step, path in _iter_step_dirs(ckpts):
            if not _is_valid_checkpoint_dir(path):
                continue
            if best is None or step > best[0]:
                best = (step, path)
        if best is not None:
            return best[1]
    return None


def _resolve_checkpoint_arg(value: str) -> Path:
    p = Path(value)
    if p.is_dir():
        return p
    if p.is_file() and p.name.endswith(".safetensors"):
        return p.parent
    raise FileNotFoundError(value)


def _fmt_int(n: int) -> str:
    return f"{n:,}"


def _fmt_b(n: float, *, digits: int = 6) -> str:
    return f"{n / 1e9:.{digits}f}"


def _fmt_params(n: int) -> str:
    return f"{_fmt_int(n)} ({_fmt_b(n)} B)"


def _fmt_flops(flops: float) -> str:
    if flops >= 1e12:
        return f"{flops/1e12:.3f} TFLOPs"
    if flops >= 1e9:
        return f"{flops/1e9:.3f} GFLOPs"
    if flops >= 1e6:
        return f"{flops/1e6:.3f} MFLOPs"
    return f"{flops:.0f} FLOPs"


def _fmt_tflops(flops: float) -> str:
    return f"{flops/1e12:.3f} TFLOPs"


def _fmt_gflops(flops: float) -> str:
    return f"{flops/1e9:.3f} GFLOPs"


def _fmt_gflops_per_token(flops_per_token: float) -> str:
    return f"{flops_per_token/1e9:.3f} GFLOPs/token"


def _fmt_flops_rate(flops_per_s: float) -> str:
    return f"{flops_per_s/1e12:.3f} TFLOPs/s"


@dataclass(frozen=True)
class ParamBreakdown:
    total: int
    embed: int
    final_norm: int
    layer_total: int
    layer_attn: int
    layer_mlp: int
    layer_norms: int
    n_layers: int
    q: int
    k: int
    v: int
    o: int
    gate: int
    up: int
    down: int


def _params_from_config(cfg: MiniLLMConfig) -> ParamBreakdown:
    cfg = cfg.finalize()
    h = int(cfg.hidden_size)
    v = int(cfg.vocab_size)
    l = int(cfg.num_hidden_layers)
    i = int(cfg.intermediate_size or 0)
    n_heads = int(cfg.num_attention_heads)
    n_kv = int(cfg.num_key_value_heads or n_heads)
    head_dim = h // n_heads
    h_kv = int(n_kv * head_dim)

    embed = v * h
    final_norm = h

    q = h * h
    k = h_kv * h
    vproj = h_kv * h
    o = h * h
    attn = q + k + vproj + o

    gate = i * h
    up = i * h
    down = h * i
    mlp = gate + up + down

    norms = 2 * h
    layer_total = attn + mlp + norms
    total = embed + (l * layer_total) + final_norm

    return ParamBreakdown(
        total=total,
        embed=embed,
        final_norm=final_norm,
        layer_total=layer_total,
        layer_attn=attn,
        layer_mlp=mlp,
        layer_norms=norms,
        n_layers=l,
        q=q,
        k=k,
        v=vproj,
        o=o,
        gate=gate,
        up=up,
        down=down,
    )


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
        per_token += 2 * h * v

    tokens = int(batch_size) * int(seq_len)
    total = float(tokens) * float(per_token)
    return total, float(per_token)


@dataclass(frozen=True)
class DevicePeak:
    chip: str
    gpu_cores: Optional[int]
    fp32_tflops: Optional[float]
    fp16_tflops: Optional[float]


def _parse_gpu_cores_from_system_profiler(text: str) -> Optional[int]:
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

        # Approx peak FP32 per GPU core (TFLOPs/core) for Apple Silicon.
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


def _line(char: str = "-", n: int = 60) -> str:
    return char * n


def _kv(label: str, value: str, *, w: int = 14) -> str:
    return f"{label:<{w}}: {value}"


def main() -> None:
    parser = argparse.ArgumentParser(description="MiniLLM (MLX) model stats (params + FLOPs + device)")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--out_dir", type=str, default="out/mlx")
    parser.add_argument("--preset", type=str, choices=["200mb"], default="200mb")

    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--seq_len", type=int, default=1024)
    parser.add_argument("--accum_steps", type=int, default=1)
    parser.add_argument("--train_mult", type=float, default=3.0)
    parser.add_argument(
        "--include_lm_head",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include final logits matmul (H x vocab) in FLOPs.",
    )
    parser.add_argument("--tok_s", type=float, default=None, help="Token throughput to estimate achieved TFLOPs/s.")
    parser.add_argument("--step_ms", type=float, default=None, help="Optimizer step time (ms) to estimate TFLOPs/s.")
    parser.add_argument("--no_tips", action="store_true", help="Hide simple performance tips.")

    args = parser.parse_args()

    ckpt: Optional[Path]
    if args.checkpoint:
        ckpt = _resolve_checkpoint_arg(args.checkpoint)
    else:
        ckpt = _find_latest_checkpoint(Path(args.out_dir))

    if ckpt is not None:
        cfg = _load_config(ckpt)
        ckpt_msg = str(ckpt)
    else:
        cfg = minillm_200mb() if args.preset == "200mb" else minillm_200mb()
        ckpt_msg = "(none)"

    cfg = cfg.finalize()
    p = _params_from_config(cfg)

    fwd_total, fwd_per_tok = _estimate_forward_flops(
        cfg=cfg,
        batch_size=int(args.batch_size),
        seq_len=int(args.seq_len),
        include_lm_head=bool(args.include_lm_head),
    )
    train_step_flops = fwd_total * float(args.train_mult) * int(args.accum_steps)

    dev = _detect_device_peak()

    print(_line("=", 60))
    print("MiniLLM (MLX) Stats")
    print(_line("=", 60))
    print(_kv("checkpoint", ckpt_msg))
    print(
        _kv(
            "config",
            f"vocab={cfg.vocab_size} hidden={cfg.hidden_size} inter={cfg.intermediate_size} "
            f"layers={cfg.num_hidden_layers} heads={cfg.num_attention_heads} kv_heads={cfg.num_key_value_heads}",
        )
    )
    print()

    print("Parameters")
    print(_line("-", 60))
    print(_kv("total", _fmt_params(p.total)))
    print(_kv("embedding", _fmt_params(p.embed)))
    print(_kv("final_norm", _fmt_params(p.final_norm)))
    print(
        _kv(
            "block",
            f"{_fmt_params(p.layer_total)}  x{p.n_layers} = {_fmt_params(p.layer_total * p.n_layers)}",
        )
    )
    print(_kv("  block.attn", _fmt_params(p.layer_attn)))
    print(_kv("  block.mlp", _fmt_params(p.layer_mlp)))
    print(_kv("  block.norms", _fmt_params(p.layer_norms)))
    print()

    print("Block (per-layer) detail")
    print(_line("-", 60))
    print(_kv("attn.q_proj", _fmt_params(p.q) + f"  x{p.n_layers} = {_fmt_params(p.q * p.n_layers)}"))
    print(_kv("attn.k_proj", _fmt_params(p.k) + f"  x{p.n_layers} = {_fmt_params(p.k * p.n_layers)}"))
    print(_kv("attn.v_proj", _fmt_params(p.v) + f"  x{p.n_layers} = {_fmt_params(p.v * p.n_layers)}"))
    print(_kv("attn.o_proj", _fmt_params(p.o) + f"  x{p.n_layers} = {_fmt_params(p.o * p.n_layers)}"))
    print(_kv("mlp.gate", _fmt_params(p.gate) + f"  x{p.n_layers} = {_fmt_params(p.gate * p.n_layers)}"))
    print(_kv("mlp.up", _fmt_params(p.up) + f"  x{p.n_layers} = {_fmt_params(p.up * p.n_layers)}"))
    print(_kv("mlp.down", _fmt_params(p.down) + f"  x{p.n_layers} = {_fmt_params(p.down * p.n_layers)}"))
    print(_kv("norms (2x)", _fmt_params(p.layer_norms) + f"  x{p.n_layers} = {_fmt_params(p.layer_norms * p.n_layers)}"))
    print()

    print("FLOPs (estimate; full attention / prefill)")
    print(_line("-", 60))
    print(_kv("batch/seq", f"batch={args.batch_size} seq_len={args.seq_len} tokens={args.batch_size*args.seq_len}"))
    print(_kv("accum", f"accum_steps={args.accum_steps} train_mult={args.train_mult:g}"))
    print(
        _kv(
            "forward",
            f"{_fmt_tflops(fwd_total)} ({_fmt_gflops(fwd_total)})  | {_fmt_gflops_per_token(fwd_per_tok)}",
        )
    )
    print(
        _kv(
            "train_step",
            f"{train_step_flops/1e12:.3f} TFLOPs/step ({train_step_flops/1e9:.3f} GFLOPs/step)",
        )
    )

    achieved: Optional[float] = None
    achieved_note: Optional[str] = None
    if args.tok_s is not None and args.tok_s > 0:
        achieved = (fwd_per_tok * float(args.train_mult) * float(args.tok_s))  # FLOPs/s
        achieved_note = f"tok/s={args.tok_s:.0f}"
    elif args.step_ms is not None and args.step_ms > 0:
        step_s = float(args.step_ms) / 1000.0
        achieved = train_step_flops / step_s
        tok_s = (int(args.batch_size) * int(args.seq_len) * int(args.accum_steps)) / step_s
        achieved_note = f"step_ms={args.step_ms:.1f} (tok/s≈{tok_s:.0f})"

    if achieved is not None:
        print(_kv("achieved", f"{_fmt_flops_rate(achieved)} ({achieved_note})"))
        if dev.fp16_tflops is not None and dev.fp16_tflops > 0:
            util = 100.0 * (achieved / 1e12) / float(dev.fp16_tflops)
            print(_kv("util@fp16_peak", f"{util:.1f}% (peak≈{dev.fp16_tflops:.2f} TFLOPs)"))
        if dev.fp32_tflops is not None and dev.fp32_tflops > 0:
            util = 100.0 * (achieved / 1e12) / float(dev.fp32_tflops)
            print(_kv("util@fp32_peak", f"{util:.1f}% (peak≈{dev.fp32_tflops:.2f} TFLOPs)"))
    else:
        print(_kv("achieved", "n/a (pass --tok_s from logs, or --step_ms)"))
    print()

    print("Device (Metal)")
    print(_line("-", 60))
    print(_kv("chip", dev.chip))
    if dev.gpu_cores is not None:
        print(_kv("gpu_cores", str(dev.gpu_cores)))
    if dev.fp32_tflops is not None and dev.fp16_tflops is not None:
        print(_kv("peak", f"fp32≈{dev.fp32_tflops:.2f} TFLOPs | fp16≈{dev.fp16_tflops:.2f} TFLOPs"))
    else:
        print(_kv("peak", "unknown"))

    info = mx.metal.device_info()
    mem = info.get("memory_size")
    if isinstance(mem, int):
        print(_kv("memory", f"{mem/1024/1024/1024:.1f} GiB"))
    arch = info.get("architecture")
    if isinstance(arch, str) and arch:
        print(_kv("arch", arch))
    print(_line("=", 60))

    if not args.no_tips:
        tips: list[str] = []
        if int(args.batch_size) <= 1:
            tips.append("提高利用率：尝试增大 micro-batch（--batch_size），同时等比例减小 --accum_steps。")
        tips.append("测真实速度别开 --profile_timing（它会强制同步，显著变慢）。")
        tips.append("MLX 训练已默认启用编译（--compile/--compile_optimizer）。")
        if tips:
            print("Tips")
            print(_line("-", 60))
            for t in tips:
                print(f"- {t}")


if __name__ == "__main__":
    main()
