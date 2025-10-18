"""High-level training orchestration for MiniGPT."""

from __future__ import annotations

import math
import os
import signal
import sys
import time
from datetime import datetime
from typing import Any

import torch

from model.config import MiniGPTConfig
from model.transformer import create_model
from training.training_monitor import TrainingMonitor

from .checkpointing import CheckpointManager
from .data_manager import DataResolver, DatasetPreparer
from .environment import TrainingEnvironment
from .memory_hooks import MemoryHooks
from .regression_suite import RegressionSuite
from .tokenizer_manager import TokenizerManager
from .training_loop import TrainingControl, TrainingLoopRunner


class MiniGPTTrainer:
    """MiniGPT训练器，封装数据、模型与训练流程。"""

    def __init__(self, config, mode: str = "pretrain"):
        self.config = config
        self.mode = mode
        self.environment = TrainingEnvironment(config, mode)
        self.device = self.environment.device
        self.output_dir = self.environment.output_dir
        self.control = TrainingControl()

        self.resolver = DataResolver(config, mode)
        self.tokenizer_manager = TokenizerManager(config, mode, self.output_dir, self.resolver)
        self.checkpoints = CheckpointManager(config, mode, self.output_dir, self.device)
        self.memory_hooks = MemoryHooks(config, self.device)
        self.regression_suite = RegressionSuite(config, self.output_dir, self.device)
        self.reference_model = None

        print(f"=== MiniGPT {mode.upper()} 训练 ===")
        print(f"模型配置: {config.model_size}")
        print(f"设备: {self.device}")
        print(f"输出目录: {self.output_dir}")

    # ------------------------------------------------------------------
    def setup_tokenizer(self, retrain: bool = False):
        return self.tokenizer_manager.setup(retrain=retrain)

    def setup_data_loader(self, tokenizer):
        preparer = DatasetPreparer(
            self.config,
            self.mode,
            tokenizer,
            self.resolver,
            seed=getattr(self.config, "random_seed", 42),
        )
        train_loader, val_loader, stats = preparer.build()
        self.environment.dataset_stats = stats
        self.environment.persist_dataset_stats()
        return train_loader, val_loader

    def setup_model(self, tokenizer):
        print("🧠 创建模型...")
        pretrain_path: str | None = None
        pretrain_metadata: dict[str, Any] = {}

        if self.mode in {"sft", "dpo", "rlhf"}:
            pretrain_path, pretrain_metadata = self.checkpoints.peek_pretrain_metadata()
            if pretrain_path:
                print(f"🔍 检测到 pretrain checkpoint: {pretrain_path}")
                stored_size = pretrain_metadata.get("model_size")
                if stored_size and stored_size != self.config.model_size:
                    print(
                        "⚠️  当前训练配置的 model_size="
                        f"{self.config.model_size} 与 pretrain checkpoint 的标记 {stored_size} 不一致"
                    )

        model_config: MiniGPTConfig | None = None
        stored_config = pretrain_metadata.get("model_config") if pretrain_metadata else None
        if isinstance(stored_config, dict):
            try:
                model_config = MiniGPTConfig.from_dict(stored_config)
                print("♻️  使用 pretrain checkpoint 中保存的模型配置。")
            except Exception as exc:
                print(f"⚠️  无法从 pretrain checkpoint 解析模型配置: {exc}")
                model_config = None

        model_label = (
            pretrain_metadata.get("model_size")
            if pretrain_metadata.get("model_size")
            else self.config.model_size
        )

        if model_config is not None:
            model_config.vocab_size = tokenizer.vocab_size
            model = create_model(
                vocab_size=tokenizer.vocab_size,
                model_size=model_label,
                config=model_config,
            )
        else:
            model = create_model(
                vocab_size=tokenizer.vocab_size,
                model_size=self.config.model_size,
            )

        model = model.to(self.device)

        self._validate_model_alignment(model, tokenizer, pretrain_metadata)

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"总参数量: {total_params:,}")
        print(f"可训练参数: {trainable_params:,}")

        if self.mode == "dpo":
            ref_config = None
            if hasattr(model, "config") and hasattr(model.config, "to_dict"):
                try:
                    ref_config = MiniGPTConfig.from_dict(model.config.to_dict())
                except Exception as exc:
                    print(f"⚠️  无法复制模型配置用于参考模型: {exc}")
                    ref_config = None
            self.reference_model = create_model(
                vocab_size=tokenizer.vocab_size,
                model_size=model_label,
                config=ref_config,
            ).to(self.device)
            for param in self.reference_model.parameters():
                param.requires_grad_(False)

        return model

    # ------------------------------------------------------------------
    def _build_scheduler(self, optimizer, start_step: int = 0):
        warmup_steps = self.config.warmup_steps
        max_steps = self.config.max_steps

        def lr_lambda(current_step: int):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            progress = float(current_step - warmup_steps) / float(max(1, max_steps - warmup_steps))
            return max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))

        return torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda,
            last_epoch=start_step - 1 if start_step > 0 else -1,
        )

    # ------------------------------------------------------------------
    def train(
        self,
        resume_from: str | None = None,
        auto_resume: bool = False,
        retrain_tokenizer: bool = False,
    ):
        print(f"🚀 开始{self.mode}训练...")
        signal.signal(signal.SIGINT, self._signal_handler)
        print("💡 按 Ctrl+C 可优雅地停止训练并保存模型")

        start_time = time.time()
        tokenizer = self.setup_tokenizer(retrain=retrain_tokenizer)
        train_loader, val_loader = self.setup_data_loader(tokenizer)
        model = self.setup_model(tokenizer)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            betas=(self.config.beta1, self.config.beta2),
            eps=self.config.eps,
        )

        criterion = torch.nn.CrossEntropyLoss(
            ignore_index=tokenizer.pad_id,
            label_smoothing=getattr(self.config, "label_smoothing", 0.0),
        )

        scaler = None
        if self.config.mixed_precision and self.device == "cuda":
            scaler = torch.amp.GradScaler("cuda")
            print("✅ 启用混合精度训练 (FP16)")

        if self.config.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
            print("✅ 启用梯度检查点")

        start_step, _ = self.checkpoints.resume(
            model,
            optimizer,
            resume_from=resume_from,
            auto_resume=auto_resume,
        )

        if self.mode == "dpo" and self.reference_model is not None:
            try:
                self.reference_model.load_state_dict(model.state_dict())
                self.reference_model.to(self.device)
                self.reference_model.eval()
            except Exception as exc:
                print(f"⚠️  无法将策略模型权重拷贝到参考模型: {exc}")

        # Set initial_lr for scheduler (required when resuming from checkpoint)
        for param_group in optimizer.param_groups:
            if "initial_lr" not in param_group:
                param_group["initial_lr"] = param_group["lr"]

        scheduler = self._build_scheduler(optimizer, start_step=start_step)
        self._log_scheduler_state(optimizer, start_step)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        tensorboard_dir = os.path.join(
            self.config.tensorboard_dir,
            f"{self.mode}_{self.config.model_size}_{timestamp}",
        )
        monitor = TrainingMonitor(
            model=model,
            log_dir=tensorboard_dir,
            enable_tensorboard=self.config.enable_tensorboard,
            enable_real_time_plots=False,
            lightweight_mode=True,
            log_interval=10,
        )

        if self.device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            print("🧹 GPU缓存已清理")

        runner = TrainingLoopRunner(
            self.config,
            self.device,
            self.checkpoints,
            self.mode,
            reference_model=self.reference_model,
            dpo_beta=getattr(self.config, "dpo_beta", 0.1),
        )
        final_path = runner.run(
            model,
            tokenizer,
            train_loader,
            val_loader,
            optimizer,
            scheduler,
            criterion,
            scaler,
            monitor,
            self.control,
            start_step,
            start_time,
            memory_hooks=self.memory_hooks,
            regression_suite=self.regression_suite,
        )

        monitor.close()
        print(f"📊 TensorBoard日志: {tensorboard_dir}")
        print(f"💡 查看训练过程: tensorboard --logdir={tensorboard_dir}")
        return final_path

    def _validate_model_alignment(
        self,
        model,
        tokenizer,
        pretrain_metadata: dict[str, Any] | None,
    ) -> None:
        config = getattr(model, "config", None)
        if config is None:
            return

        stored_config = (
            pretrain_metadata.get("model_config")
            if pretrain_metadata and "model_config" in pretrain_metadata
            else None
        )

        if stored_config:
            mismatches: list[tuple[str, Any, Any]] = []
            for key in (
                "hidden_size",
                "num_hidden_layers",
                "num_attention_heads",
                "num_key_value_heads",
                "intermediate_size",
                "max_position_embeddings",
            ):
                expected = stored_config.get(key)
                actual = getattr(config, key, None)
                if expected is None or actual is None:
                    continue
                if expected != actual:
                    mismatches.append((key, expected, actual))

            if mismatches:
                print("❌ 检测到 pretrain checkpoint 与当前模型配置不一致:")
                for key, expected, actual in mismatches:
                    print(f"   - {key}: pretrain={expected}, 当前={actual}")
                raise RuntimeError(
                    "当前模型架构与 pretrain checkpoint 不一致，请确保 SFT 与 pretrain 使用相同的模型配置。"
                )

            print("✅ 当前模型与 pretrain checkpoint 的核心架构参数一致。")
        elif self.mode in {"sft", "dpo", "rlhf"}:
            print("ℹ️  pretrain checkpoint 未提供模型配置，跳过自动架构对齐检查。")

        current_vocab = getattr(config, "vocab_size", None)
        if current_vocab is not None and current_vocab != tokenizer.vocab_size:
            raise RuntimeError(
                f"分词器词表大小 {tokenizer.vocab_size} 与模型配置 {current_vocab} 不一致，请重新对齐。"
            )

        max_positions = getattr(config, "max_position_embeddings", None)
        if max_positions is not None:
            if max_positions < self.config.max_seq_len:
                raise RuntimeError(
                    f"当前训练配置的 max_seq_len={self.config.max_seq_len} 超出 pretrain 模型支持的 {max_positions}。"
                )
            if max_positions != self.config.max_seq_len:
                print(
                    f"ℹ️  模型支持的最大序列长度为 {max_positions}，当前训练配置为 {self.config.max_seq_len}。"
                )

    # ------------------------------------------------------------------
    def _log_scheduler_state(self, optimizer, start_step: int) -> None:
        if start_step > 0:
            current_lr = optimizer.param_groups[0]["lr"]
            if start_step >= self.config.warmup_steps:
                phase = "Cosine Decay"
                progress = (
                    (start_step - self.config.warmup_steps)
                    / (self.config.max_steps - self.config.warmup_steps)
                    * 100
                )
            else:
                phase = "Warmup"
                progress = start_step / self.config.warmup_steps * 100
            print(f"📊 学习率调度器已恢复到第 {start_step} 步")
            print(f"   当前阶段: {phase} (已完成{progress:.1f}%)")
            print(f"   当前学习率: {current_lr:.2e}")
        else:
            warmup_ratio = self.config.warmup_steps / self.config.max_steps * 100
            print(
                f"✅ 学习率调度器: Warmup({self.config.warmup_steps}步, {warmup_ratio:.1f}%) + Cosine Decay"
            )
            print(
                f"   初始LR: 0 -> 峰值LR: {self.config.learning_rate:.2e} -> "
                f"最低LR: {self.config.learning_rate * 0.1:.2e}"
            )

    def _signal_handler(self, signum, frame):
        print("\n\n⚠️  收到中断信号 (Ctrl+C)")
        if not self.control.interrupted:
            self.control.interrupted = True
            print("🔄 正在优雅地停止训练...")
            print("💾 将保存当前模型状态...")
            print("   (再次按 Ctrl+C 可强制退出)")
        else:
            print("⚡ 强制退出！")
            sys.exit(1)
