"""Command line helper to run the Qwen identity fine-tuning workflow."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def _ensure_repo_import_path() -> None:
    """Add the repository root to ``sys.path`` when executed as a script."""

    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


_ensure_repo_import_path()


def _load_pipeline_modules():
    from examples.qwen_identity_finetune.pipeline import (  # noqa: E402
        IdentityFineTuneConfig,
        PersonaSpecification,
        run_identity_finetune_pipeline,
    )

    return IdentityFineTuneConfig, PersonaSpecification, run_identity_finetune_pipeline

_DEFAULT_PERSONA_STATEMENT = (
    "你是一位善于共情、用语温和的医学专家，擅长通过先思考再回答的方式"
    "为患者提供循序渐进的指导。请确保你的回复先展示思考过程，再给出最终答复。"
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Qwen persona identity fine-tuning workflow end-to-end.",
    )
    parser.add_argument(
        "--work-dir",
        default="./qwen_identity_workspace",
        help="Directory used to store datasets, checkpoints, and logs.",
    )
    parser.add_argument(
        "--cache-dir",
        default="./pretrained",
        help="Cache directory for downloaded base models from ModelScope.",
    )
    parser.add_argument(
        "--persona-name",
        default="Dr. Grace",
        help="Display name of the persona identity.",
    )
    parser.add_argument(
        "--persona-statement",
        default=_DEFAULT_PERSONA_STATEMENT,
        help="Natural language description that defines the persona identity.",
    )
    parser.add_argument(
        "--warmup-question",
        action="append",
        dest="warmup_questions",
        help="Optional extra question for generating additional persona-aligned samples."
        " Can be provided multiple times.",
    )
    parser.add_argument(
        "--identity-mix-ratio",
        type=float,
        default=0.25,
        help="Portion of persona-aligned samples to mix into the base training set.",
    )
    parser.add_argument(
        "--train-split-ratio",
        type=float,
        default=0.9,
        help="Proportion of the dataset used for training (remainder used for validation).",
    )
    parser.add_argument(
        "--max-identity-samples",
        type=int,
        default=None,
        help="Optional upper bound on the number of persona samples to generate.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for dataset splitting and sampling.",
    )
    parser.add_argument(
        "--base-model-repo",
        default="Qwen/Qwen3-1.7B",
        help="ModelScope repository name of the base Qwen model.",
    )
    parser.add_argument(
        "--base-model-revision",
        default="master",
        help="Revision identifier for the base model (branch, tag, or commit).",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="Learning rate used during fine-tuning.",
    )
    parser.add_argument(
        "--per-device-batch-size",
        type=int,
        default=1,
        help="Per-device batch size for both training and evaluation.",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=4,
        help="Number of gradient accumulation steps.",
    )
    parser.add_argument(
        "--train-epochs",
        type=int,
        default=1,
        help="Number of fine-tuning epochs to run.",
    )
    parser.add_argument(
        "--eval-steps",
        type=int,
        default=100,
        help="How often to run evaluation (in steps).",
    )
    parser.add_argument(
        "--logging-steps",
        type=int,
        default=10,
        help="Frequency of logging training metrics (in steps).",
    )
    parser.add_argument(
        "--save-steps",
        type=int,
        default=400,
        help="Checkpoint save interval (in steps).",
    )
    parser.add_argument(
        "--run-name",
        default="qwen3-identity",
        help="Run name used in SwanLab dashboards.",
    )
    parser.add_argument(
        "--predictions-to-log",
        type=int,
        default=3,
        help="Number of generated predictions to record in SwanLab at the end of training.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=2048,
        help="Maximum sequence length during tokenization.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    IdentityFineTuneConfig, PersonaSpecification, run_identity_finetune_pipeline = (
        _load_pipeline_modules()
    )

    warmup_questions = args.warmup_questions

    persona = PersonaSpecification(
        persona_name=args.persona_name,
        identity_statement=args.persona_statement,
        mix_ratio=args.identity_mix_ratio,
        max_identity_samples=args.max_identity_samples,
        warmup_questions=warmup_questions,
    )

    config = IdentityFineTuneConfig(
        work_dir=args.work_dir,
        persona=persona,
        cache_dir=args.cache_dir,
        seed=args.seed,
        base_model_repo=args.base_model_repo,
        base_model_revision=args.base_model_revision,
        learning_rate=args.learning_rate,
        per_device_batch_size=args.per_device_batch_size,
        per_device_eval_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        train_epochs=args.train_epochs,
        eval_steps=args.eval_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        run_name=args.run_name,
        predictions_to_log=args.predictions_to_log,
        max_length=args.max_length,
        train_split_ratio=args.train_split_ratio,
    )

    os.environ.setdefault("SWANLAB_PROJECT", args.run_name)

    run_identity_finetune_pipeline(config)


if __name__ == "__main__":
    main()
