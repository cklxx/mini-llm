#!/usr/bin/env python3
"""Run a minimal end-to-end training + inference smoke test.

The script prepares tiny synthetic datasets, runs the four training
modes (pretrain → sft → dpo → rlhf) with extremely small budgets, and
finally performs a single inference call on the RLHF checkpoint.  The
goal is to guarantee that every stage of the pipeline can execute
successfully on CPU-only environments within a few minutes.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"


DATASETS: dict[Path, list[dict[str, object]]] = {
    DATA_DIR / "pretrain_hq.jsonl": [
        {
            "text": "Mini-LLM focuses on compact transformer experiments for quick iteration.",
        },
        {
            "text": "Progressive training allows staged scaling without large compute.",
        },
        {
            "text": "Curriculum style pretraining can warm up tokenizers and pipelines.",
        },
    ],
    DATA_DIR / "sft_mini_512.jsonl": [
        {
            "conversations": [
                {"role": "user", "content": "请用一句话描述 Mini-LLM 框架。"},
                {
                    "role": "assistant",
                    "content": "Mini-LLM 是一个教学友好的小型语言模型训练与推理演示框架。",
                },
            ]
        },
        {
            "conversations": [
                {"role": "user", "content": "如何快速验证训练流程？"},
                {
                    "role": "assistant",
                    "content": "可以运行内置的烟雾测试脚本，依次执行预训练、SFT、DPO 和 RLHF。",
                },
            ]
        },
    ],
    DATA_DIR / "alex_identity.jsonl": [
        {
            "conversations": [
                {
                    "role": "system",
                    "content": "你是助教 Alex，负责解释 Mini-LLM 的设计。",
                },
                {"role": "user", "content": "请做个自我介绍。"},
                {
                    "role": "assistant",
                    "content": "大家好，我是助教 Alex，专注于讲解 Mini-LLM 的训练与推理流程。",
                },
            ]
        },
        {
            "conversations": [
                {
                    "role": "system",
                    "content": "你是助教 Alex，负责解释 Mini-LLM 的设计。",
                },
                {"role": "user", "content": "你能提供哪些帮助？"},
                {
                    "role": "assistant",
                    "content": "我可以带你运行完整的训练流水线，并在推理阶段演示文本生成。",
                },
            ]
        },
    ],
    DATA_DIR / "ultra_think.jsonl": [
        {
            "conversations": [
                {
                    "role": "user",
                    "content": "请深入分析为什么要进行渐进式训练。",
                },
                {
                    "role": "assistant",
                    "content": (
                        "<ultra_think>渐进式训练允许在保证稳定性的前提下逐步扩展模型："
                        "先用少量数据验证流程，再逐步放大规模，以便及时发现问题并控制资源。</ultra_think>"
                    ),
                },
            ]
        }
    ],
    DATA_DIR / "dpo.jsonl": [
        {
            "prompt": "用户: 帮我总结 Mini-LLM 的目标。",
            "chosen": "助手: Mini-LLM 旨在用最小的工程成本展示完整的训练与推理流程。",
            "rejected": "助手: 我不知道。",
        },
        {
            "prompt": "用户: 为什么要跑一遍全部流程？",
            "chosen": (
                "助手: 通过运行预训练、SFT、DPO 和 RLHF，可以确认数据管线、检查点与推理接口协同正常。"
            ),
            "rejected": "助手: 不需要。",
        },
    ],
}


@dataclass
class Stage:
    mode: str
    max_steps: int
    warmup_steps: int = 0
    extra_args: tuple[str, ...] = ()


STAGES: tuple[Stage, ...] = (
    Stage(mode="pretrain", max_steps=4),
    Stage(mode="sft", max_steps=3, extra_args=("--auto-resume",)),
    Stage(mode="dpo", max_steps=3, extra_args=("--auto-resume",)),
    Stage(mode="rlhf", max_steps=3, extra_args=("--auto-resume",)),
)


def ensure_jsonl(path: Path, records: Iterable[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def prepare_datasets() -> None:
    print("📦 Preparing synthetic datasets...")
    for path, records in DATASETS.items():
        ensure_jsonl(path, records)
        print(f"  • {path.relative_to(PROJECT_ROOT)} ({len(records)} samples)")


def clean_previous_runs() -> None:
    for stage in STAGES:
        ckpt_dir = CHECKPOINT_DIR / f"{stage.mode}_tiny"
        if ckpt_dir.exists():
            print(f"🧹 Removing stale checkpoints: {ckpt_dir.relative_to(PROJECT_ROOT)}")
            shutil.rmtree(ckpt_dir)


def run_stage(stage: Stage, env: dict[str, str]) -> None:
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "train.py"),
        "--mode",
        stage.mode,
        "--config",
        "tiny",
        "--max-steps",
        str(stage.max_steps),
        "--batch-size",
        "2",
        "--warmup-steps",
        str(stage.warmup_steps),
        *stage.extra_args,
    ]
    print(f"\n🚀 Running {stage.mode.upper()} stage (max_steps={stage.max_steps})")
    subprocess.run(cmd, check=True, cwd=PROJECT_ROOT, env=env)


def run_inference(model_path: Path, env: dict[str, str]) -> None:
    prompt = "请用两句话总结刚刚训练完成的模型流程。"
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "generate.py"),
        "--model-path",
        str(model_path),
        "--mode",
        "single",
        "--prompt",
        prompt,
        "--max-new-tokens",
        "32",
        "--device",
        "cpu",
    ]
    print("\n🧪 Running inference sanity check...")
    subprocess.run(cmd, check=True, cwd=PROJECT_ROOT, env=env)


def main() -> None:
    prepare_datasets()
    clean_previous_runs()

    env = os.environ.copy()
    env.update(
        {
            "MINIGPT_VAL_SPLIT": "0.0",
            "MINIGPT_PRETRAIN_HQ_RATIO": "1.0",
            "MINIGPT_PRETRAIN_WIKI_RATIO": "0.0",
            "MINIGPT_PRETRAIN_CHINA_RATIO": "0.0",
            "MINIGPT_PRETRAIN_PJ_RATIO": "0.0",
            "MINIGPT_SFT_MAIN_RATIO": "1.0",
            "MINIGPT_SFT_MAIN_MAX": "16",
            "MINIGPT_MEMORY_MONITOR": "0",
            "MINIGPT_DISABLE_MANIFEST": "1",
            "PYTHONUNBUFFERED": "1",
        }
    )

    for stage in STAGES:
        run_stage(stage, env)

    final_model = CHECKPOINT_DIR / f"{STAGES[-1].mode}_tiny" / "final_model.pt"
    if not final_model.exists():
        raise SystemExit(f"❌ Expected model checkpoint not found: {final_model}")

    run_inference(final_model, env)
    print("\n✅ Smoke test finished successfully!")


if __name__ == "__main__":
    main()
