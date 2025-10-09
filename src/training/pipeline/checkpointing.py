"""Checkpoint helpers for the training pipeline."""
from __future__ import annotations

import glob
import os
from typing import Optional, Tuple

import torch


class CheckpointManager:
    """Persist and restore checkpoints with cleanup helpers."""

    def __init__(self, config, mode: str, output_dir: str, device: str):
        self.config = config
        self.mode = mode
        self.output_dir = output_dir
        self.device = device

    # ------------------------------------------------------------------
    def find_latest(self) -> Optional[str]:
        pattern = os.path.join(self.output_dir, "checkpoint_step_*.pt")
        checkpoints = glob.glob(pattern)
        if not checkpoints:
            return None
        checkpoints.sort(key=os.path.getmtime, reverse=True)
        return checkpoints[0]

    def find_pretrain_source(self) -> Optional[str]:
        pretrain_dir = os.path.join(self.config.checkpoint_dir, f"pretrain_{self.config.model_size}")
        search_patterns = [
            os.path.join(pretrain_dir, "checkpoint_step_*.pt"),
            os.path.join(pretrain_dir, "final_model.pt"),
        ]
        for pattern in search_patterns:
            matches = glob.glob(pattern)
            if not matches:
                continue
            matches.sort(key=os.path.getmtime, reverse=True)
            print(f"   ✅ 找到 pretrain checkpoint: {matches[0]}")
            return matches[0]
        return None

    # ------------------------------------------------------------------
    def resume(
        self,
        model,
        optimizer,
        *,
        resume_from: Optional[str] = None,
        auto_resume: bool = False,
    ) -> Tuple[int, bool]:
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
                start_step = self._load_checkpoint(resume_from, model, optimizer)
                checkpoint_loaded = True
            else:
                print(f"⚠️  Checkpoint文件不存在: {resume_from}")

        if not checkpoint_loaded and self.mode in {"sft", "dpo", "rlhf"}:
            pretrain_model_path = self.find_pretrain_source()
            if pretrain_model_path:
                print(f"\n🎯 {self.mode.upper()} 模式：从 pretrain checkpoint 加载初始权重")
                try:
                    self._load_model_weights(pretrain_model_path, model)
                    checkpoint_loaded = True
                    print("✅ 成功加载 pretrain 模型权重")
                except Exception as exc:
                    print(f"⚠️  加载 pretrain 权重失败: {exc}")
                    print("   将使用随机初始化的模型")
            else:
                print(f"\n⚠️  未找到 pretrain 模型: {os.path.join(self.config.checkpoint_dir, f'pretrain_{self.config.model_size}')}")
                print(
                    f"   建议先运行 pretrain 模式训练基础模型：\n"
                    f"   uv run python scripts/train.py --mode pretrain --config {self.config.model_size}"
                )
                print(f"   现在将使用随机初始化的模型进行 {self.mode} 训练")
        elif not checkpoint_loaded and self.mode == "pretrain":
            print("\n📚 Pretrain 模式：从随机初始化开始训练")

        return start_step, checkpoint_loaded

    # ------------------------------------------------------------------
    def load(self, path: str, model, optimizer) -> int:
        return self._load_checkpoint(path, model, optimizer)

    def save(self, model, optimizer, step: int, loss: float) -> str:
        self._remove_old_checkpoints()
        checkpoint_path = os.path.join(self.output_dir, f"checkpoint_step_{step}.pt")
        torch.save(
            {
                "step": step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": loss,
                "config": self.config,
                "mode": self.mode,
            },
            checkpoint_path,
        )
        print(f"💾 检查点已保存: {checkpoint_path}")
        return checkpoint_path

    def save_final(self, model, tokenizer, step: int) -> str:
        final_path = os.path.join(self.output_dir, "final_model.pt")
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "tokenizer_vocab_size": tokenizer.vocab_size,
                "config": self.config,
                "mode": self.mode,
                "step": step,
            },
            final_path,
        )
        return final_path

    # ------------------------------------------------------------------
    def _remove_old_checkpoints(self) -> None:
        pattern = os.path.join(self.output_dir, "checkpoint_step_*.pt")
        for old_ckpt in glob.glob(pattern):
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
                        if 'initial_lr' not in group:
                            group['initial_lr'] = group['lr']
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
