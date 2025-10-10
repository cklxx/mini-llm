"""Command-line entry point for MiniGPT training."""
from __future__ import annotations

import argparse

from config.training_config import get_config

from .app import MiniGPTTrainer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MiniGPT训练脚本")
    parser.add_argument(
        "--mode",
        choices=["pretrain", "sft", "dpo", "rlhf"],
        default="sft",
        help="训练模式 (pretrain: 预训练, sft: 监督微调, dpo: 直接偏好优化, rlhf: 强化学习)",
    )
    parser.add_argument(
        "--config",
        choices=["tiny", "small", "small_30m", "medium", "large", "foundation", "moe"],
        default="medium",
        help="模型配置大小 (tiny: ~1M, small: ~25M, small_30m: ~30M, medium: ~80M, large: ~350M, foundation: ~200M, moe: MOE模型)",
    )
    parser.add_argument(
        "--retrain-tokenizer",
        action="store_true",
        help="重新训练分词器",
    )
    parser.add_argument(
        "--resume",
        "--resume-from-checkpoint",
        type=str,
        default=None,
        dest="resume_from_checkpoint",
        help="从指定checkpoint文件继续训练（例如: checkpoints/sft_medium/checkpoint_step_5000.pt）",
    )
    parser.add_argument(
        "--auto-resume",
        action="store_true",
        help="自动从最新的checkpoint恢复训练。注意：SFT/DPO/RLHF模式会自动加载pretrain权重作为初始化",
    )
    parser.add_argument("--learning-rate", type=float, default=None, help="学习率")
    parser.add_argument("--max-steps", type=int, default=None, help="最大训练步数")
    parser.add_argument("--batch-size", type=int, default=None, help="批次大小")
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=None,
        help="学习率warmup步数（如果不指定，将根据训练模式自动设置）",
    )
    return parser


def apply_mode_defaults(config, mode: str, overrides) -> None:
    if mode == "pretrain":
        config.max_steps = overrides.max_steps or config.max_steps or 50000
        if overrides.learning_rate is None:
            config.learning_rate = 1e-4
        config.warmup_steps = min(500, int(config.max_steps * 0.05))
        print("📚 预训练模式：建立基础语言理解能力")
        print(f"   学习率: {config.learning_rate:.2e}")
        print(f"   Warmup steps: {config.warmup_steps} (前{config.warmup_steps/config.max_steps*100:.1f}%)")
    elif mode == "sft":
        config.max_steps = overrides.max_steps or config.max_steps or 10000
        if overrides.learning_rate is None:
            config.learning_rate = 5e-5
        config.warmup_steps = min(200, int(config.max_steps * 0.02))
        print("🎯 监督微调模式：训练对话和特定任务能力")
        print(f"   学习率: {config.learning_rate:.2e} (比预训练低，保护已学知识)")
        print(f"   Warmup steps: {config.warmup_steps} (前{config.warmup_steps/config.max_steps*100:.1f}%)")
        print("   💡 模型已有预训练基础，使用短warmup快速进入衰减阶段")
    elif mode == "dpo":
        config.max_steps = overrides.max_steps or config.max_steps or 5000
        if overrides.learning_rate is None:
            config.learning_rate = 1e-5
        config.warmup_steps = min(100, int(config.max_steps * 0.02))
        print("⚖️  直接偏好优化模式：根据人类偏好调整响应")
        print(f"   学习率: {config.learning_rate:.2e}")
        print(f"   Warmup steps: {config.warmup_steps} (前{config.warmup_steps/config.max_steps*100:.1f}%)")
        print("   💡 在SFT基础上优化，使用极短warmup")
    elif mode == "rlhf":
        config.max_steps = overrides.max_steps or config.max_steps or 3000
        if overrides.learning_rate is None:
            config.learning_rate = 1e-5
        config.warmup_steps = min(100, int(config.max_steps * 0.02))
        print("🔄 强化学习微调模式：通过奖励模型优化")
        print(f"   学习率: {config.learning_rate:.2e}")
        print(f"   Warmup steps: {config.warmup_steps} (前{config.warmup_steps/config.max_steps*100:.1f}%)")
        print("   💡 在已训练模型上强化学习，使用极短warmup")
    else:
        raise ValueError(f"不支持的训练模式: {mode}")


def run_cli(argv: list[str] | None = None) -> str:
    parser = build_parser()
    args = parser.parse_args(argv)

    config = get_config(args.config)

    if args.learning_rate is not None:
        config.learning_rate = args.learning_rate
    if args.max_steps is not None:
        config.max_steps = args.max_steps
    if args.batch_size is not None:
        config.batch_size = args.batch_size

    apply_mode_defaults(config, args.mode, args)

    if args.warmup_steps is not None:
        config.warmup_steps = args.warmup_steps
        print(f"⚙️  使用自定义warmup步数: {config.warmup_steps} (前{config.warmup_steps/config.max_steps*100:.1f}%)")

    trainer = MiniGPTTrainer(config, mode=args.mode)

    if args.auto_resume:
        print("🔄 启用自动恢复模式")
    elif args.resume_from_checkpoint:
        print(f"🔄 将从checkpoint恢复: {args.resume_from_checkpoint}")

    final_model_path = trainer.train(
        resume_from=args.resume_from_checkpoint,
        auto_resume=args.auto_resume,
        retrain_tokenizer=args.retrain_tokenizer,
    )

    print(f"\n✅ 训练完成！模型保存在: {final_model_path}")

    if args.mode == "pretrain":
        print("\n💡 建议下一步运行SFT训练:")
        print(f"uv run python scripts/train.py --mode sft --config {args.config} --resume {final_model_path}")
    elif args.mode == "sft":
        print("\n💡 建议下一步运行DPO训练:")
        print(f"uv run python scripts/train.py --mode dpo --config {args.config} --resume {final_model_path}")

    return final_model_path
