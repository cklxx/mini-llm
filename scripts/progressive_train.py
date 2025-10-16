#!/usr/bin/env python3
"""Orchestrate progressive MiniGPT training runs aligned with the scaling plan."""

from __future__ import annotations

import argparse
import json
import sys
from collections import OrderedDict
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@contextmanager
def _project_import_context() -> Iterator[None]:
    """Temporarily prepend the repo root and src directory to ``sys.path``."""

    added: list[str] = []
    for candidate in (PROJECT_ROOT, PROJECT_ROOT / "src"):
        candidate_str = str(candidate)
        if candidate_str not in sys.path:
            sys.path.insert(0, candidate_str)
            added.append(candidate_str)
    try:
        yield
    finally:
        for candidate_str in added:
            try:
                sys.path.remove(candidate_str)
            except ValueError:  # pragma: no cover - defensive cleanup
                pass


@dataclass
class PhaseDefinition:
    """Describe a single training pass for a stage."""

    name: str
    mode: str
    description: str
    overrides: dict[str, int | float | bool] = field(default_factory=dict)
    resume_from_previous: bool = False
    auto_resume: bool = False
    retrain_tokenizer: bool = False
    config: str | None = None


@dataclass
class StageDefinition:
    """Describe a progressive stage composed of one or more phases."""

    id: str
    title: str
    target_params: str
    base_config: str
    focus: str
    dataset_scope: str
    checkpoints: str
    iteration_guidance: str
    phases: list[PhaseDefinition]


PROGRESSIVE_STAGES: OrderedDict[str, StageDefinition] = OrderedDict(
    (
        (
            "stage0",
            StageDefinition(
                id="stage0",
                title="Tokenizer 与数据管线冒烟验证",
                target_params="≤20M",
                base_config="small",
                focus="验证 Tokenizer/数据清洗/日志是否工作，快速观察损失曲线",
                dataset_scope="抽样 1–2B token，主语言 Python/TypeScript",
                checkpoints="首次全量训练，无需继承 checkpoint",
                iteration_guidance="在完成冒烟验证后再次迭代 1–2 次以确认修复后的数据/配置稳定",
                phases=[
                    PhaseDefinition(
                        name="阶段0 预训练",
                        mode="pretrain",
                        description="以小批量训练 2–3k 步，确认吞吐与指标采集",
                        overrides={
                            "max_steps": 3000,
                            "warmup_steps": 120,
                            "eval_steps": 200,
                            "save_steps": 600,
                        },
                    ),
                ],
            ),
        ),
        (
            "stage1",
            StageDefinition(
                id="stage1",
                title="轻量可复现基线",
                target_params="60–80M",
                base_config="medium",
                focus="建立稳定的多语言补全基线，并验证短 SFT 接口",
                dataset_scope="4–6B token 预训练 + 约 20K 条指令微调样本",
                checkpoints="从阶段 0 产出的权重开始可加速收敛",
                iteration_guidance="建议至少循环两轮：第一轮获得基线，第二轮针对评测反馈调整学习率/采样",
                phases=[
                    PhaseDefinition(
                        name="阶段1 预训练",
                        mode="pretrain",
                        description="跑满 40k 步左右，观察 HumanEval/MBPP 初始分数",
                        overrides={
                            "max_steps": 40000,
                            "warmup_steps": 800,
                            "eval_steps": 1000,
                            "save_steps": 4000,
                        },
                        resume_from_previous=True,
                        auto_resume=True,
                    ),
                    PhaseDefinition(
                        name="阶段1 指令微调",
                        mode="sft",
                        description="加载阶段1预训练权重，跑 2–3k 步短程指令微调",
                        overrides={
                            "max_steps": 3000,
                            "warmup_steps": 60,
                            "eval_steps": 200,
                            "save_steps": 600,
                        },
                        resume_from_previous=True,
                        auto_resume=True,
                    ),
                ],
            ),
        ),
        (
            "stage2",
            StageDefinition(
                id="stage2",
                title="200M 级性能冲刺",
                target_params="180–220M",
                base_config="foundation",
                focus="扩展上下文到 4K，覆盖 15–25B token 并扩大指令集",
                dataset_scope="主语言高质量仓库 + 80K–120K 指令/执行反馈",
                checkpoints="继承阶段1权重，保留评测挂钩与蒸馏接口",
                iteration_guidance="在不同数据混合或蒸馏策略之间做多次 A/B，对齐关键指标后再晋级下一阶段",
                phases=[
                    PhaseDefinition(
                        name="阶段2 预训练",
                        mode="pretrain",
                        description="1–1.5 天遍历数据，主要关注损失稳定性与吞吐",
                        overrides={
                            "max_steps": 120000,
                            "warmup_steps": 3600,
                            "eval_steps": 3000,
                            "save_steps": 6000,
                        },
                        resume_from_previous=True,
                        auto_resume=True,
                    ),
                    PhaseDefinition(
                        name="阶段2 指令微调",
                        mode="sft",
                        description="继续扩充高质量指令样本，确保 IDE 体验",
                        overrides={
                            "max_steps": 6000,
                            "warmup_steps": 120,
                            "eval_steps": 400,
                            "save_steps": 1200,
                        },
                        resume_from_previous=True,
                        auto_resume=True,
                    ),
                ],
            ),
        ),
        (
            "stage3",
            StageDefinition(
                id="stage3",
                title="向 350M+ 扩展试点",
                target_params=">=350M",
                base_config="large",
                focus="在确认 200M 满足应用需求后扩大深度与宽度",
                dataset_scope="沿用阶段2清洗成果，追加大模型特化任务",
                checkpoints="承接阶段2 指令微调后的最新 checkpoint",
                iteration_guidance="通过多轮缩短版训练 (如 20K 步) 快速验证显存/吞吐，再进入正式全量训练",
                phases=[
                    PhaseDefinition(
                        name="阶段3 预训练",
                        mode="pretrain",
                        description="以经过验证的超参启动，重点监控显存与吞吐",
                        overrides={
                            "max_steps": 160000,
                            "warmup_steps": 4800,
                            "eval_steps": 4000,
                            "save_steps": 8000,
                        },
                        resume_from_previous=True,
                        auto_resume=True,
                    ),
                ],
            ),
        ),
    )
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="根据渐进式配置执行 MiniGPT 训练阶段",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--stages",
        nargs="+",
        default=["stage0", "stage1", "stage2", "stage3"],
        choices=list(PROGRESSIVE_STAGES.keys()),
        help="选择需要运行的阶段，默认全量",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="执行实际训练。默认仅打印计划 (dry-run)",
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="首个阶段/阶段内首个子阶段起点的 checkpoint 路径",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="覆盖所有阶段的学习率",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="覆盖所有阶段的最大训练步数",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=None,
        help="覆盖所有阶段的 warmup 步数",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="覆盖所有阶段的 batch size",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=None,
        help="覆盖所有阶段的梯度累积步数",
    )
    parser.add_argument(
        "--auto-resume",
        action="store_true",
        help="强制所有子阶段启用 auto resume",
    )
    parser.add_argument(
        "--retrain-tokenizer",
        action="store_true",
        help="强制所有子阶段重新训练 tokenizer",
    )
    parser.add_argument(
        "--plan-output",
        type=Path,
        default=None,
        help="将训练阶段计划导出为 JSON 文件",
    )
    return parser.parse_args()


def apply_overrides(config, overrides: dict[str, int | float | bool]) -> None:
    for key, value in overrides.items():
        if value is None:
            continue
        if not hasattr(config, key):
            print(f"⚠️  配置 {config.model_size} 未包含属性 {key}，已跳过覆盖")
            continue
        setattr(config, key, value)


def plan_to_json(stage_ids: Iterable[str]) -> list[dict[str, Any]]:
    plan = []
    for stage_id in stage_ids:
        stage = PROGRESSIVE_STAGES[stage_id]
        plan.append(
            {
                "id": stage.id,
                "title": stage.title,
                "target_params": stage.target_params,
                "base_config": stage.base_config,
                "focus": stage.focus,
                "dataset_scope": stage.dataset_scope,
                "checkpoints": stage.checkpoints,
                "iteration_guidance": stage.iteration_guidance,
                "phases": [
                    {
                        "name": phase.name,
                        "mode": phase.mode,
                        "description": phase.description,
                        "overrides": phase.overrides,
                        "resume_from_previous": phase.resume_from_previous,
                        "auto_resume": phase.auto_resume,
                        "retrain_tokenizer": phase.retrain_tokenizer,
                        "config": phase.config or stage.base_config,
                    }
                    for phase in stage.phases
                ],
            }
        )
    return plan


def render_plan(stage_ids: Iterable[str]) -> None:
    plan = plan_to_json(stage_ids)
    print("\n=== 渐进式训练路线图 ===")
    for stage in plan:
        print(
            f"\n[{stage['id']}] {stage['title']}\n"
            f"  目标参数量: {stage['target_params']}\n"
            f"  训练配置: {stage['base_config']}\n"
            f"  核心目标: {stage['focus']}\n"
            f"  数据范围: {stage['dataset_scope']}\n"
            f"  checkpoint 策略: {stage['checkpoints']}\n"
            f"  迭代建议: {stage['iteration_guidance']}"
        )
        for phase in stage["phases"]:
            print(
                f"    - {phase['name']} ({phase['mode']})\n"
                f"      描述: {phase['description']}\n"
                f"      覆盖项: {phase['overrides'] or '无'}\n"
                f"      继承上一阶段: {'是' if phase['resume_from_previous'] else '否'}"
            )


def load_training_dependencies() -> tuple[object, type]:
    """Import heavy training dependencies lazily."""

    try:
        with _project_import_context():
            from config.training_config import get_config as get_training_config  # type: ignore
            from training.pipeline.app import MiniGPTTrainer  # type: ignore
    except ModuleNotFoundError as exc:  # pragma: no cover - import error messaging
        if exc.name == "torch":
            raise SystemExit(
                "未检测到 PyTorch (torch) 库。请先安装 GPU/CPU 版 PyTorch 后再使用 --execute 执行训练。"
            ) from exc
        raise

    return get_training_config, MiniGPTTrainer


def execute_plan(args: argparse.Namespace, stage_ids: Iterable[str]) -> None:
    get_training_config, MiniGPTTrainer = load_training_dependencies()
    last_checkpoint: str | None = args.resume_from

    for stage_id in stage_ids:
        stage = PROGRESSIVE_STAGES[stage_id]
        print(f"\n===== 执行 {stage.id.upper()} : {stage.title} =====")
        for phase in stage.phases:
            config_name = phase.config or stage.base_config
            config = get_training_config(config_name)

            combined_overrides: dict[str, int | float | bool] = {}
            combined_overrides.update(phase.overrides)

            # Apply CLI overrides last so they always win
            if args.max_steps is not None:
                combined_overrides["max_steps"] = args.max_steps
            if args.warmup_steps is not None:
                combined_overrides["warmup_steps"] = args.warmup_steps
            if args.batch_size is not None:
                combined_overrides["batch_size"] = args.batch_size
            if args.gradient_accumulation_steps is not None:
                combined_overrides["gradient_accumulation_steps"] = args.gradient_accumulation_steps
            if args.learning_rate is not None:
                combined_overrides["learning_rate"] = args.learning_rate

            print(
                f"\n→ {phase.name} [{phase.mode}] | 配置: {config_name}\n"
                f"   描述: {phase.description}"
            )
            apply_overrides(config, combined_overrides)

            for key in sorted(combined_overrides):
                value = getattr(config, key, None)
                print(f"   - {key}: {value}")

            resume_from: str | None = None
            if phase.resume_from_previous:
                resume_from = last_checkpoint
            if resume_from:
                print(f"   将从 {resume_from} 恢复")

            auto_resume = args.auto_resume or phase.auto_resume
            retrain_tokenizer = args.retrain_tokenizer or phase.retrain_tokenizer

            trainer = MiniGPTTrainer(config, mode=phase.mode)
            final_path = trainer.train(
                resume_from=resume_from,
                auto_resume=auto_resume,
                retrain_tokenizer=retrain_tokenizer,
            )
            last_checkpoint = final_path
            print(f"✅ {phase.name} 完成，模型已保存到 {final_path}")


def main() -> None:
    args = parse_args()
    stage_ids = args.stages
    plan_json = plan_to_json(stage_ids)

    if args.plan_output is not None:
        args.plan_output.parent.mkdir(parents=True, exist_ok=True)
        args.plan_output.write_text(json.dumps(plan_json, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"📄 渐进式计划已保存至 {args.plan_output}")

    if not args.execute:
        render_plan(stage_ids)
        print("\n💡 使用 --execute 以实际运行训练；支持 --resume-from 指定初始 checkpoint。")
        return

    execute_plan(args, stage_ids)


if __name__ == "__main__":
    main()
