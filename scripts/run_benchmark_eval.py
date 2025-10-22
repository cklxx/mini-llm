#!/usr/bin/env python3
"""Convenience wrapper for launching the benchmark evaluator on a saved checkpoint."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch

# Ensure the repository root is on the module search path when executed directly.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.evaluation import BenchmarkEvaluator, BenchmarkSettings  # noqa: E402
from src.model.config import MiniGPTConfig  # noqa: E402
from src.model.transformer import MiniGPT  # noqa: E402
from src.tokenizer.bpe_tokenizer import BPETokenizer  # noqa: E402

DEFAULT_TASKS = [
    "wikitext2",
    "lambada_openai",
    "hellaswag",
    "arc_challenge",
    "winogrande",
    "piqa",
    "boolq",
]
DEFAULT_CACHE_DIR = REPO_ROOT / "data" / "eval" / "benchmarks"


class ConsoleBenchmarkLogger:
    """Minimal logger compatible with :class:`BenchmarkEvaluator`."""

    def log_benchmark(self, step: int, metrics: dict[str, float], *, task: str) -> None:
        header = f"📊 Benchmark[{task}] (step={step})"
        print(header)
        for key, value in metrics.items():
            print(f"  - {key}: {value}")


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():  # type: ignore[attr-defined]
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def _load_model(checkpoint_path: Path, *, device: torch.device, vocab_size: int) -> MiniGPT:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    config_dict: dict[str, Any] | None = None
    if isinstance(checkpoint.get("model_config"), dict):
        config_dict = checkpoint["model_config"]
    elif isinstance(checkpoint.get("config"), dict):
        config_dict = checkpoint["config"].get("model_config")  # type: ignore[union-attr]
    else:
        config_obj = checkpoint.get("config")
        model_config_obj = getattr(config_obj, "model_config", None)
        if model_config_obj is not None:
            if isinstance(model_config_obj, dict):
                config_dict = model_config_obj
            elif hasattr(model_config_obj, "to_dict"):
                config_dict = model_config_obj.to_dict()

    if config_dict is None:
        raise RuntimeError(
            "无法从 checkpoint 解析模型配置。请确认 checkpoint 由训练流水线保存，且包含 'model_config' 字段。"
        )

    model_config = MiniGPTConfig.from_dict(config_dict)
    model_config.vocab_size = vocab_size

    model = MiniGPT(model_config)
    state_dict = checkpoint.get("model_state_dict") or checkpoint
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def _load_tokenizer(tokenizer_path: Path) -> BPETokenizer:
    tokenizer = BPETokenizer()
    tokenizer.load(str(tokenizer_path))
    return tokenizer


def _resolve_tokenizer_path(checkpoint_path: Path, override: str | None) -> Path:
    candidates = []
    if override:
        candidates.append(Path(override).expanduser())
    checkpoint_dir = checkpoint_path.parent
    candidates.extend(
        [
            checkpoint_dir / "tokenizer",
            checkpoint_dir / "tokenizer.json",
        ]
    )

    for candidate in candidates:
        if candidate.is_dir():
            json_path = candidate / "tokenizer.json"
            if json_path.exists():
                return json_path
        elif candidate.exists():
            return candidate

    raise FileNotFoundError(
        "未找到分词器文件。请通过 --tokenizer 指定 tokenizer.json，"
        "或确保其与 checkpoint 位于同一目录。"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the built-in benchmark suite on a saved Mini-LLM checkpoint.",
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to the checkpoint file produced by scripts/train.py",
    )
    parser.add_argument(
        "--tokenizer",
        help=(
            "Path to tokenizer resources (tokenizer.json or directory). "
            "Defaults to the tokenizer next to the checkpoint."
        ),
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Computation device identifier. Defaults to auto-detection.",
    )
    parser.add_argument(
        "--tasks",
        nargs="*",
        default=DEFAULT_TASKS,
        help=(
            "Benchmark task names. If omitted, uses the default suite: "
            + ", ".join(DEFAULT_TASKS)
        ),
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=256,
        help="Maximum samples per task. Defaults to 256.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for perplexity-style tasks. Defaults to 4.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=None,
        help="Optional override for sequence length. Defaults to task-specific values.",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Where to cache downloaded datasets. Defaults to data/eval/benchmarks.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional path to write JSON results. Defaults to checkpoint directory.",
    )
    parser.add_argument(
        "--disable-auto-download",
        action="store_true",
        help="Disable automatic dataset downloads (reuse existing cache only).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"未找到 checkpoint 文件: {checkpoint_path}")

    tokenizer_path = _resolve_tokenizer_path(checkpoint_path, args.tokenizer)

    device = _resolve_device(args.device)
    print(f"🖥️  使用设备: {device}")

    tokenizer = _load_tokenizer(tokenizer_path)
    print(f"🔤 已加载分词器 (路径={tokenizer_path}, vocab_size={tokenizer.vocab_size})")

    model = _load_model(checkpoint_path, device=device, vocab_size=tokenizer.vocab_size)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"🧠 模型已加载，参数量: {total_params:,}")

    cache_dir = Path(args.cache_dir).expanduser().resolve() if args.cache_dir else DEFAULT_CACHE_DIR
    os.makedirs(cache_dir, exist_ok=True)

    tasks = args.tasks or DEFAULT_TASKS
    settings = BenchmarkSettings.from_task_names(
        tasks,
        frequency=1,
        max_samples=args.max_samples,
        batch_size=args.batch_size,
        max_length=args.max_length,
        overrides=None,
        cache_dir=str(cache_dir),
        auto_download=not args.disable_auto_download,
    )

    evaluator = BenchmarkEvaluator(device=device, tokenizer=tokenizer, settings=settings)
    monitor = ConsoleBenchmarkLogger()

    print("🚀 开始执行行业评测...")
    metrics = evaluator.maybe_run(model, step=1, monitor=monitor, force=True)

    if not metrics:
        print("⚠️  未得到任何评测结果，请检查数据集是否可用或任务配置是否正确。")
        return

    timestamp = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())
    output_path = (
        Path(args.output).expanduser().resolve()
        if args.output
        else checkpoint_path.parent / f"benchmark_results_{timestamp.replace(':', '')}.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        "timestamp": timestamp,
        "checkpoint": str(checkpoint_path),
        "tokenizer": str(tokenizer_path),
        "device": str(device),
        "tasks": list(metrics.keys()),
        "metrics": metrics,
        "settings": {
            "max_samples": args.max_samples,
            "batch_size": args.batch_size,
            "max_length": args.max_length,
            "cache_dir": str(cache_dir),
            "auto_download": not args.disable_auto_download,
        },
    }
    model_config = getattr(model, "config", None)
    if model_config is not None and hasattr(model_config, "to_dict"):
        payload["model_config"] = model_config.to_dict()

    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)

    print(f"✅ 评测完成，结果已保存到: {output_path}")


if __name__ == "__main__":
    main()
