"""Checkpoint helpers for the training pipeline."""

from __future__ import annotations

import glob
import os
import re
import shutil
from typing import Any

import torch

from tokenizer.config_utils import canonicalize_tokenizer_config


class CheckpointManager:
    """Persist and restore checkpoints with cleanup helpers."""

    def __init__(self, config, mode: str, output_dir: str, device: str):
        self.config = config
        self.mode = mode
        self.output_dir = output_dir
        self.device = device
        self._pretrain_metadata_cache: tuple[str | None, dict[str, Any]] | None = None

    # ------------------------------------------------------------------
    def find_latest(self) -> str | None:
        pattern = os.path.join(self.output_dir, "checkpoint_step_*.pt")
        checkpoints = glob.glob(pattern)
        if not checkpoints:
            return None
        checkpoints.sort(key=os.path.getmtime, reverse=True)
        return checkpoints[0]

    def _locate_stage_checkpoint(self, stage: str) -> str | None:
        stage_dir = os.path.join(
            self.config.checkpoint_dir, f"{stage}_{self.config.model_size}"
        )
        search_patterns = [
            os.path.join(stage_dir, "final_model.pt"),
            os.path.join(stage_dir, "checkpoint_step_*.pt"),
        ]
        for pattern in search_patterns:
            matches = glob.glob(pattern)
            if not matches:
                continue
            if pattern.endswith("checkpoint_step_*.pt"):
                matches.sort(key=os.path.getmtime, reverse=True)
            return matches[0]
        return None

    def _locate_pretrain_checkpoint(self) -> str | None:
        return self._locate_stage_checkpoint("pretrain")

    def find_pretrain_source(self) -> str | None:
        path = self._locate_pretrain_checkpoint()
        if path:
            print(f"   ✅ 找到 pretrain checkpoint: {path}")
        return path

    def peek_pretrain_metadata(self) -> tuple[str | None, dict[str, Any]]:
        if self._pretrain_metadata_cache is not None:
            return self._pretrain_metadata_cache

        path = self._locate_pretrain_checkpoint()
        if not path:
            self._pretrain_metadata_cache = (None, {})
            return self._pretrain_metadata_cache

        metadata = self._extract_checkpoint_metadata(path)
        metadata["path"] = path
        self._pretrain_metadata_cache = (path, metadata)
        return self._pretrain_metadata_cache

    # ------------------------------------------------------------------
    def resume(
        self,
        model,
        optimizer,
        *,
        resume_from: str | None = None,
        auto_resume: bool = False,
    ) -> tuple[int, bool]:
        start_step = 0
        checkpoint_loaded = False

        if auto_resume:
            latest_checkpoint = self.find_latest()
            if latest_checkpoint:
                print(f"🔍 找到checkpoint: {latest_checkpoint}")
                start_step = self._load_checkpoint(latest_checkpoint, model, optimizer)
                checkpoint_loaded = True
            else:
                print("ℹ️  未找到当前模式的checkpoint")
        elif resume_from:
            if os.path.exists(resume_from):
                metadata = self._extract_checkpoint_metadata(resume_from)
                checkpoint_mode = metadata.get("mode")
                has_optimizer_state = metadata.get("has_optimizer_state", False)
                if checkpoint_mode and checkpoint_mode != self.mode:
                    print(
                        "ℹ️  检测到来自不同训练阶段的checkpoint，将其作为初始化权重使用。"
                    )
                    self._load_model_weights(resume_from, model)
                    print("✅ 已加载模型权重")
                    checkpoint_loaded = True
                    start_step = 0
                elif not has_optimizer_state:
                    print(
                        "ℹ️  指定的checkpoint不包含优化器状态，将仅加载模型权重并重新开始训练。"
                    )
                    self._load_model_weights(resume_from, model)
                    print("✅ 已加载模型权重")
                    checkpoint_loaded = True
                    start_step = 0
                else:
                    start_step = self._load_checkpoint(resume_from, model, optimizer)
                    checkpoint_loaded = True
            else:
                print(f"⚠️  Checkpoint文件不存在: {resume_from}")

        if not checkpoint_loaded and self.mode in {"sft", "dpo", "rlhf"}:
            for stage in self._initial_checkpoint_stages():
                stage_path = self._locate_stage_checkpoint(stage)
                if not stage_path:
                    continue
                print(
                    f"\n🎯 {self.mode.upper()} 模式：从 {stage.upper()} checkpoint 加载初始权重"
                )
                try:
                    self._load_model_weights(stage_path, model)
                    checkpoint_loaded = True
                    print("✅ 成功加载初始化模型权重")
                    break
                except Exception as exc:
                    print(f"⚠️  加载 {stage} 权重失败: {exc}")
                    print("   尝试其他可用的上游检查点...")
            if not checkpoint_loaded:
                print(
                    f"\n⚠️  未找到可用的上游 checkpoint 初始化 {self.mode} 模式"
                )
                print(
                    f"   建议先完成前置阶段训练，例如运行 pretrain 或 sft 流程。"
                )
                print(f"   现在将使用随机初始化的模型进行 {self.mode} 训练")
        elif not checkpoint_loaded and self.mode == "pretrain":
            print("\n📚 Pretrain 模式：从随机初始化开始训练")

        return start_step, checkpoint_loaded

    def _initial_checkpoint_stages(self) -> list[str]:
        if self.mode == "sft":
            return ["pretrain"]
        if self.mode == "dpo":
            return ["sft", "pretrain"]
        if self.mode == "rlhf":
            return ["dpo", "sft", "pretrain"]
        return []

    # ------------------------------------------------------------------
    def load(self, path: str, model, optimizer) -> int:
        return self._load_checkpoint(path, model, optimizer)

    def save(self, model, optimizer, step: int, loss: float, tokenizer=None) -> str:
        self._remove_old_checkpoints(current_step=step)
        checkpoint_path = os.path.join(self.output_dir, f"checkpoint_step_{step}.pt")
        payload = {
            "step": step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
            "config": self.config,
            "mode": self.mode,
            "model_size": getattr(self.config, "model_size", None),
        }

        model_config = self._serialize_model_config(model)
        if model_config is not None:
            payload["model_config"] = model_config
        if tokenizer is not None:
            try:
                payload.update(
                    {
                        "tokenizer_vocab_size": tokenizer.vocab_size,
                        "tokenizer_config": canonicalize_tokenizer_config(tokenizer.get_config()),
                        "tokenizer_checksum": tokenizer.checksum(),
                        "tokenizer_special_tokens": tokenizer.special_tokens_map(),
                    }
                )
            except Exception as exc:  # pragma: no cover - defensive
                print(f"⚠️  无法附加分词器信息到checkpoint: {exc}")
        torch.save(payload, checkpoint_path)
        print(f"💾 检查点已保存: {checkpoint_path}")
        return checkpoint_path

    def save_final(self, model, tokenizer, step: int) -> str:
        final_path = os.path.join(self.output_dir, "final_model.pt")
        payload = {
            "model_state_dict": model.state_dict(),
            "tokenizer_vocab_size": tokenizer.vocab_size,
            "tokenizer_config": canonicalize_tokenizer_config(tokenizer.get_config()),
            "tokenizer_checksum": tokenizer.checksum(),
            "tokenizer_special_tokens": tokenizer.special_tokens_map(),
            "config": self.config,
            "mode": self.mode,
            "step": step,
            "model_size": getattr(self.config, "model_size", None),
        }

        model_config = self._serialize_model_config(model)
        if model_config is not None:
            payload["model_config"] = model_config
        torch.save(payload, final_path)
        return final_path

    # ------------------------------------------------------------------
    def _remove_old_checkpoints(self, *, current_step: int) -> None:
        pattern = os.path.join(self.output_dir, "checkpoint_step_*.pt")
        archive_dir = os.path.join(self.output_dir, "archive")
        os.makedirs(archive_dir, exist_ok=True)

        step_pattern = re.compile(r"checkpoint_step_(\d+)\.pt$")

        for old_ckpt in glob.glob(pattern):
            match = step_pattern.search(os.path.basename(old_ckpt))
            step_value = int(match.group(1)) if match else None

            if step_value is not None and step_value >= current_step:
                target_path = os.path.join(archive_dir, os.path.basename(old_ckpt))
                if os.path.exists(target_path):
                    # Avoid overwriting an archived checkpoint with the same name.
                    continue
                try:
                    shutil.move(old_ckpt, target_path)
                    print(
                        "📦 保留较高步数checkpoint → "
                        f"{os.path.join('archive', os.path.basename(target_path))}"
                    )
                except Exception as exc:
                    print(f"⚠️  移动checkpoint到归档失败: {exc}")
                continue

            try:
                os.remove(old_ckpt)
                print(f"🗑️  删除旧checkpoint: {os.path.basename(old_ckpt)}")
            except Exception as exc:
                print(f"⚠️  删除旧checkpoint失败: {exc}")

    def _load_checkpoint(self, checkpoint_path, model, optimizer) -> int:
        print(f"🔄 正在加载checkpoint: {checkpoint_path}")
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            if "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
                print("✅ 模型权重已加载")
            else:
                model.load_state_dict(checkpoint)
                print("✅ 模型权重已加载（旧格式）")
            if "optimizer_state_dict" in checkpoint and optimizer is not None:
                try:
                    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                    # Set initial_lr for each param group (required for LR schedulers when resuming)
                    for group in optimizer.param_groups:
                        if "initial_lr" not in group:
                            group["initial_lr"] = group["lr"]
                    print("✅ 优化器状态已加载")
                except Exception as exc:
                    print(f"⚠️  优化器状态加载失败: {exc}")
                    print("   将使用新的优化器状态")
            start_step = checkpoint.get("step", checkpoint.get("global_step", 0))
            if start_step > 0:
                print(f"✅ 将从第 {start_step} 步继续训练")
            if "loss" in checkpoint:
                print(f"📊 Checkpoint损失: {checkpoint['loss']:.4f}")
            if "mode" in checkpoint:
                print(f"📝 训练模式: {checkpoint['mode']}")
            if "config" in checkpoint:
                print("ℹ️  Checkpoint包含训练配置，可用于恢复实验环境")
            else:
                print("ℹ️  Checkpoint不包含训练配置，请确保当前配置与原训练一致")
            return start_step
        except Exception as exc:
            print(f"❌ 加载checkpoint失败: {exc}")
            print("⚠️  将从头开始训练")
            return 0

    def _load_model_weights(self, checkpoint_path: str, model) -> None:
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)

    def _serialize_model_config(self, model) -> dict[str, Any] | None:
        config = getattr(model, "config", None)
        if config is None:
            return None
        if hasattr(config, "to_dict"):
            try:
                return config.to_dict()
            except Exception as exc:  # pragma: no cover - defensive logging
                print(f"⚠️  无法序列化模型配置: {exc}")
        return None

    def _extract_checkpoint_metadata(self, path: str) -> dict[str, Any]:
        metadata: dict[str, Any] = {}
        try:
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        except Exception as exc:  # pragma: no cover - defensive logging
            print(f"⚠️  读取checkpoint元数据失败: {exc}")
            return metadata

        try:
            model_config = checkpoint.get("model_config")
            if isinstance(model_config, dict):
                metadata["model_config"] = model_config
            model_size = checkpoint.get("model_size")
            if model_size:
                metadata["model_size"] = model_size
            config_obj = checkpoint.get("config")
            config_model_size = getattr(config_obj, "model_size", None)
            if config_model_size and "model_size" not in metadata:
                metadata["model_size"] = config_model_size
            metadata["mode"] = checkpoint.get("mode")
            metadata["step"] = checkpoint.get("step", checkpoint.get("global_step"))
            metadata["has_optimizer_state"] = "optimizer_state_dict" in checkpoint
        finally:
            del checkpoint

        return metadata
