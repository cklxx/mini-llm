"""Single training pipeline used across all MiniGPT stages."""

from __future__ import annotations

import json
import math
import os
import random
import signal
import sys
import time
from datetime import datetime
from typing import Any

import torch

from evaluation.benchmark_suite import BenchmarkEvaluator, BenchmarkSettings
from model.config import MiniGPTConfig
from model.transformer import create_model
from training.training_monitor import TrainingMonitor

from .checkpointing import CheckpointManager
from .data_manager import DataResolver, DatasetPreparer
from .tokenizer_manager import TokenizerManager
from .training_loop import TrainingControl, TrainingLoopRunner


class TrainingPipeline:
    """Coordinate data preparation, model setup and the training loop."""

    def __init__(self, config, mode: str = "pretrain") -> None:
        self.config = config
        self.mode = mode
        self.device = getattr(config, "device", self._detect_device())
        self.output_dir = os.path.join(
            self.config.checkpoint_dir, f"{mode}_{self.config.model_size}"
        )
        os.makedirs(self.output_dir, exist_ok=True)

        self.control = TrainingControl()
        self.resolver = DataResolver(config, mode)
        self.tokenizer_manager = TokenizerManager(
            config, mode, self.output_dir, self.resolver
        )
        self.checkpoints = CheckpointManager(config, mode, self.output_dir, self.device)
        self.reference_model = None
        self.latest_model = None
        self.latest_tokenizer = None

        self._set_random_seed(getattr(self.config, "random_seed", 42))
        self._save_config_snapshot()

        print(f"=== MiniGPT {mode.upper()} 训练 ===")
        print(f"模型配置: {config.model_size}")
        print(f"设备: {self.device}")
        print(f"输出目录: {self.output_dir}")

    # ------------------------------------------------------------------
    def _detect_device(self) -> str:
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _set_random_seed(self, seed: int) -> None:
        random.seed(seed)
        try:
            import numpy as np  # type: ignore

            np.random.seed(seed)
        except Exception:  # pragma: no cover - numpy optional
            pass

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _serialize_config_value(self, value: Any):
        if isinstance(value, (int, float, str, bool)) or value is None:
            return value
        if isinstance(value, (list, tuple)):
            return [self._serialize_config_value(v) for v in value]
        if isinstance(value, dict):
            return {str(k): self._serialize_config_value(v) for k, v in value.items()}
        return str(value)

    def _save_config_snapshot(self) -> None:
        snapshot_path = os.path.join(self.output_dir, "training_config_snapshot.json")
        try:
            config_dict = {
                key: self._serialize_config_value(val)
                for key, val in vars(self.config).items()
                if not key.startswith("_")
            }
            with open(snapshot_path, "w", encoding="utf-8") as handle:
                json.dump(config_dict, handle, indent=2, ensure_ascii=False)
            print(f"📝 配置快照已保存: {snapshot_path}")
        except Exception as exc:  # pragma: no cover - defensive logging
            print(f"⚠️  配置快照保存失败: {exc}")

    def _persist_dataset_stats(self, stats: list[dict[str, Any]]) -> None:
        if not stats:
            return
        stats_path = os.path.join(self.output_dir, "dataset_stats.json")
        try:
            with open(stats_path, "w", encoding="utf-8") as handle:
                json.dump(stats, handle, indent=2, ensure_ascii=False)
            print(f"📊 数据集统计已保存: {stats_path}")
        except Exception as exc:  # pragma: no cover - defensive logging
            print(f"⚠️  数据集统计保存失败: {exc}")

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
        self._persist_dataset_stats(stats)
        return train_loader, val_loader

    def _build_model(self, tokenizer):
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
        else:
            self.reference_model = None

        return model

    # ------------------------------------------------------------------
    def _create_optimizer(self, model):
        return torch.optim.AdamW(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            betas=(self.config.beta1, self.config.beta2),
            eps=self.config.eps,
        )

    def _create_criterion(self, tokenizer):
        return torch.nn.CrossEntropyLoss(
            ignore_index=tokenizer.pad_id,
            label_smoothing=getattr(self.config, "label_smoothing", 0.0),
        )

    def _create_scaler(self):
        if self.config.mixed_precision and self.device == "cuda":
            print("✅ 启用混合精度训练 (FP16)")
            return torch.amp.GradScaler("cuda")
        return None

    def _build_scheduler(self, optimizer, start_step: int = 0):
        warmup_steps = self.config.warmup_steps
        max_steps = self.config.max_steps

        def scheduler_lambda(current_step: int):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            progress = float(current_step - warmup_steps) / float(max(1, max_steps - warmup_steps))
            return max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))

        return torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            scheduler_lambda,
            last_epoch=start_step - 1 if start_step > 0 else -1,
        )

    # ------------------------------------------------------------------
    def train(
        self,
        *,
        resume_from: str | None = None,
        auto_resume: bool = False,
        retrain_tokenizer: bool = False,
        tokenizer_override=None,
        model_override=None,
        target_epochs: int | None = None,
    ):
        print(f"🚀 开始{self.mode}训练...")
        signal.signal(signal.SIGINT, self._signal_handler)
        print("💡 按 Ctrl+C 可优雅地停止训练并保存模型")

        start_time = time.time()
        tokenizer = tokenizer_override or self.setup_tokenizer(retrain=retrain_tokenizer)
        self.latest_tokenizer = tokenizer
        train_loader, val_loader = self.setup_data_loader(tokenizer)
        if target_epochs is not None:
            updates_per_epoch = max(
                1,
                math.ceil(
                    len(train_loader)
                    / max(1, getattr(self.config, "gradient_accumulation_steps", 1))
                ),
            )
            computed_steps = max(1, updates_per_epoch * max(1, int(target_epochs)))
            if computed_steps < self.config.max_steps:
                print(
                    "🎯 按照目标 epoch 调整训练步数: "
                    f"{self.config.max_steps} -> {computed_steps}"
                )
                self.config.max_steps = computed_steps

        if model_override is not None:
            model = model_override.to(self.device)
            self._validate_model_alignment(model, tokenizer, None)
            if self.mode == "dpo" and self.reference_model is None:
                ref_config = None
                base_config = getattr(model, "config", None)
                if base_config is not None and hasattr(base_config, "to_dict"):
                    try:
                        ref_config = MiniGPTConfig.from_dict(base_config.to_dict())
                    except Exception as exc:
                        print(f"⚠️  无法从提供的模型复制配置: {exc}")
                        ref_config = None
                self.reference_model = create_model(
                    vocab_size=tokenizer.vocab_size,
                    model_size=self.config.model_size,
                    config=ref_config,
                ).to(self.device)
                for param in self.reference_model.parameters():
                    param.requires_grad_(False)
                try:
                    self.reference_model.load_state_dict(model.state_dict())
                except Exception as exc:
                    print(f"⚠️  无法同步参考模型权重: {exc}")
        else:
            model = self._build_model(tokenizer)

        optimizer = self._create_optimizer(model)
        criterion = self._create_criterion(tokenizer)
        scaler = self._create_scaler()

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

        for param_group in optimizer.param_groups:
            param_group.setdefault("initial_lr", param_group["lr"])

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

        benchmark_evaluator = self._maybe_create_benchmark_evaluator(tokenizer)

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
            memory_hooks=None,
            regression_suite=None,
            benchmark_evaluator=benchmark_evaluator,
        )

        monitor.close()
        print(f"📊 TensorBoard日志: {tensorboard_dir}")
        print(f"💡 查看训练过程: tensorboard --logdir={tensorboard_dir}")
        self.latest_model = model
        return final_path

    # ------------------------------------------------------------------
    def _maybe_create_benchmark_evaluator(self, tokenizer):
        if not getattr(self.config, "benchmark_eval_enabled", False):
            return None

        max_length = self.config.benchmark_eval_max_length
        if max_length is None:
            max_length = getattr(self.config, "max_seq_len", None)

        try:
            settings = BenchmarkSettings.from_task_names(
                self.config.benchmark_eval_tasks,
                frequency=self.config.benchmark_eval_frequency,
                max_samples=self.config.benchmark_eval_max_samples,
                batch_size=self.config.benchmark_eval_batch_size,
                max_length=max_length,
                overrides=getattr(self.config, "benchmark_eval_overrides", None),
                cache_dir=getattr(self.config, "benchmark_eval_cache_dir", None),
                auto_download=getattr(self.config, "benchmark_eval_auto_download", True),
            )
        except ValueError as exc:
            print(f"⚠️  未能启用行业评测: {exc}")
            return None

        return BenchmarkEvaluator(
            device=self.device,
            tokenizer=tokenizer,
            settings=settings,
            autocast_dtype=getattr(self.config, "inference_autocast_dtype", None),
        )

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

    def _log_scheduler_state(self, optimizer, start_step: int) -> None:
        if start_step > 0:
            current_lr = optimizer.param_groups[0]["lr"]
            if start_step >= self.config.warmup_steps:
                phase = "Cosine Decay"
                progress = (
                    (start_step - self.config.warmup_steps)
                    / max(1, self.config.max_steps - self.config.warmup_steps)
                    * 100
                )
            else:
                phase = "Warmup"
                progress = start_step / max(1, self.config.warmup_steps) * 100
            print(f"📊 学习率调度器已恢复到第 {start_step} 步")
            print(f"   当前阶段: {phase} (已完成{progress:.1f}%)")
            print(f"   当前学习率: {current_lr:.2e}")
        else:
            warmup_ratio = (
                self.config.warmup_steps / max(1, self.config.max_steps) * 100
            )
            print(
                f"✅ 学习率调度器: Warmup({self.config.warmup_steps}步, {warmup_ratio:.1f}%) + Cosine Decay"
            )
            print(
                f"   初始LR: 0 -> 峰值LR: {self.config.learning_rate:.2e} -> "
                f"最低LR: {self.config.learning_rate * 0.1:.2e}"
            )

    def _signal_handler(self, signum, frame):  # pragma: no cover - signal handler
        print("\n\n⚠️  收到中断信号 (Ctrl+C)")
        if not self.control.interrupted:
            self.control.interrupted = True
            print("🔄 正在优雅地停止训练...")
            print("💾 将保存当前模型状态...")
            print("   (再次按 Ctrl+C 可强制退出)")
        else:
            print("⚡ 强制退出！")
            sys.exit(1)
