from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import DefaultDict, Dict, Iterable, List, Optional, Tuple

import mlx.core as mx
import mlx.utils as mlx_utils

from .config import MiniLLMConfig, minillm_200mb
from .model import MiniLLMForCausalLM


def _fmt_int(n: int) -> str:
    return f"{n:,}"


def _fmt_params(n: int) -> str:
    return f"{_fmt_int(n)} ({n / 1e9:.6f}B)"


def _load_config(checkpoint_dir: Path) -> MiniLLMConfig:
    config_path = checkpoint_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config.json in checkpoint dir: {checkpoint_dir}")
    with open(config_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"config.json must be an object, got: {type(data).__name__}")
    return MiniLLMConfig.from_dict(data)


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


def _iter_step_dirs(checkpoints_dir: Path) -> Iterable[Tuple[int, Path]]:
    step_re = re.compile(r"^step_(\d+)$")
    if not checkpoints_dir.exists() or not checkpoints_dir.is_dir():
        return []
    out: List[Tuple[int, Path]] = []
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
        best: Optional[Tuple[int, Path]] = None
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


def _flatten_arrays(params: Dict[str, object]) -> Dict[str, mx.array]:
    flat: Dict[str, object] = {}
    mlx_utils.tree_flatten(params, destination=flat)
    out: Dict[str, mx.array] = {}
    for k, v in flat.items():
        if isinstance(v, mx.array):
            out[k] = v
    return out


def _group_key(name: str) -> str:
    return re.sub(r"model\.layers\.\d+\.", "model.layers.*.", name)


def _print_report(model: MiniLLMForCausalLM) -> None:
    arrays = _flatten_arrays(model.parameters())
    total_params = sum(int(v.size) for v in arrays.values())

    grouped_sizes: DefaultDict[str, List[int]] = defaultdict(list)
    grouped_shapes: Dict[str, Tuple[int, ...]] = {}
    for name, arr in arrays.items():
        g = _group_key(name)
        grouped_sizes[g].append(int(arr.size))
        if g not in grouped_shapes:
            grouped_shapes[g] = tuple(int(x) for x in arr.shape)

    # Per-transformer-layer totals (model.layers.<i>.*)
    layer_sizes: DefaultDict[int, int] = defaultdict(int)
    layer_re = re.compile(r"^model\.layers\.(\d+)\.")
    for name, arr in arrays.items():
        m = layer_re.match(name)
        if not m:
            continue
        layer_sizes[int(m.group(1))] += int(arr.size)

    cfg = model.config
    print(
        "[config] "
        f"vocab={cfg.vocab_size} hidden={cfg.hidden_size} intermediate={cfg.intermediate_size} "
        f"layers={cfg.num_hidden_layers} heads={cfg.num_attention_heads} kv_heads={cfg.num_key_value_heads}"
    )
    print(f"[total] params={_fmt_params(total_params)}")

    if layer_sizes:
        uniq = sorted(set(layer_sizes.values()))
        if len(uniq) == 1:
            per_layer = uniq[0]
            n = len(layer_sizes)
            print(f"[layers] transformer_block params={_fmt_params(per_layer)} x{n} = {_fmt_params(per_layer * n)}")
        else:
            print("[layers] transformer_block params per layer (uneven):")
            for i in range(min(len(layer_sizes), 128)):
                print(f"  - layer[{i}]: {_fmt_params(layer_sizes[i])}")

    # High-level buckets.
    embed = sum(
        sum(v) for k, v in grouped_sizes.items() if k == "model.embed_tokens.weight"
    )
    final_norm = sum(sum(v) for k, v in grouped_sizes.items() if k == "model.norm.weight")
    attn = sum(
        sum(v)
        for k, v in grouped_sizes.items()
        if k.startswith("model.layers.*.self_attn.")
    )
    mlp = sum(
        sum(v) for k, v in grouped_sizes.items() if k.startswith("model.layers.*.mlp.")
    )
    norms = sum(
        sum(v)
        for k, v in grouped_sizes.items()
        if k.startswith("model.layers.*.input_layernorm.")
        or k.startswith("model.layers.*.post_attention_layernorm.")
    )
    other = total_params - (embed + final_norm + attn + mlp + norms)
    print(
        "[summary] "
        f"embed={_fmt_params(embed)} attn={_fmt_params(attn)} mlp={_fmt_params(mlp)} "
        f"norms={_fmt_params(norms)} final_norm={_fmt_params(final_norm)} other={_fmt_params(other)}"
    )

    # Detailed repeated-weights breakdown.
    order: Dict[str, int] = {
        "model.embed_tokens.weight": 0,
        "model.layers.*.input_layernorm.weight": 10,
        "model.layers.*.self_attn.q_proj.weight": 20,
        "model.layers.*.self_attn.k_proj.weight": 21,
        "model.layers.*.self_attn.v_proj.weight": 22,
        "model.layers.*.self_attn.o_proj.weight": 23,
        "model.layers.*.post_attention_layernorm.weight": 30,
        "model.layers.*.mlp.gate_proj.weight": 40,
        "model.layers.*.mlp.up_proj.weight": 41,
        "model.layers.*.mlp.down_proj.weight": 42,
        "model.norm.weight": 90,
    }

    def sort_key(item: Tuple[str, List[int]]) -> Tuple[int, str]:
        k, _ = item
        return (order.get(k, 1000), k)

    print("[detail] (name shape params_each xN = params_total)")
    for name, sizes in sorted(grouped_sizes.items(), key=sort_key):
        n = len(sizes)
        total = sum(sizes)
        shape = grouped_shapes.get(name, ())
        if n == 1:
            print(f"  - {name} {shape} {_fmt_params(total)}")
            continue
        per = sizes[0]
        if any(s != per for s in sizes):
            print(f"  - {name} {shape} (uneven) total={_fmt_params(total)} n={n}")
            continue
        print(
            f"  - {name} {shape} {_fmt_int(per)} x{n} = {_fmt_params(total)}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="MiniLLM (MLX) parameter count report")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Checkpoint directory (or model.safetensors) to load config from; defaults to latest under out/mlx.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="out/mlx",
        help="Output root used to auto-find latest checkpoint when --checkpoint is not set.",
    )
    parser.add_argument(
        "--preset",
        type=str,
        choices=["200mb"],
        default="200mb",
        help="Fallback preset when no checkpoint is found.",
    )

    args = parser.parse_args()

    ckpt: Optional[Path] = None
    if args.checkpoint:
        ckpt = _resolve_checkpoint_arg(args.checkpoint)
    else:
        ckpt = _find_latest_checkpoint(Path(args.out_dir))

    if ckpt is not None:
        cfg = _load_config(ckpt)
        print(f"[ckpt] {ckpt}")
    else:
        cfg = minillm_200mb()
        print("[ckpt] (none) using preset config: 200mb")

    model = MiniLLMForCausalLM(cfg)
    _print_report(model)


if __name__ == "__main__":
    main()
