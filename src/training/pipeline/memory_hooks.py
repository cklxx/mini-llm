"""Lightweight integration of the legacy memory optimizer helpers."""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch

from training.memory_optimizer import MemoryMonitor


@dataclass
class MemoryHookConfig:
    """Runtime knobs used by :class:`MemoryHooks`."""

    enabled: bool
    threshold: float
    cleanup_interval: int
    log_interval: int


class MemoryHooks:
    """Wrap :class:`MemoryMonitor` into the new training pipeline."""

    def __init__(self, config: any, device: str):
        enabled = bool(getattr(config, "memory_monitor_enabled", False))
        threshold = float(getattr(config, "memory_pressure_threshold", 0.92))
        cleanup_interval = int(getattr(config, "memory_cleanup_interval", 200))
        log_interval = int(getattr(config, "memory_log_interval", 100))
        self._hook_config = MemoryHookConfig(enabled, threshold, cleanup_interval, log_interval)
        self._device = torch.device(device)
        self._monitor: MemoryMonitor | None = None
        self._steps_since_log = 0
        self._steps_since_cleanup = 0

        if self._hook_config.enabled:
            self._monitor = MemoryMonitor(self._device)
            env_setting = os.environ.get("PYTORCH_CUDA_ALLOC_CONF")
            if env_setting:
                print(f"🧠 CUDA allocator config: {env_setting}")
            print(
                "🧮 Memory monitor enabled "
                f"(threshold={self._hook_config.threshold:.2f}, interval={self._hook_config.cleanup_interval})"
            )

    # ------------------------------------------------------------------
    def on_train_start(self) -> None:
        if not self._monitor:
            return
        summary = self._monitor.get_memory_summary()
        print(summary)

    def on_step_end(self, step: int) -> None:
        if not self._monitor:
            return

        self._steps_since_log += 1
        self._steps_since_cleanup += 1

        should_cleanup = False
        if (
            self._hook_config.cleanup_interval > 0
            and self._steps_since_cleanup >= self._hook_config.cleanup_interval
        ):
            should_cleanup = True
        elif self._monitor.check_memory_pressure(self._hook_config.threshold):
            should_cleanup = True

        if should_cleanup:
            self._monitor.force_cleanup()
            info = self._monitor.get_memory_info()
            print(
                "🧹 Memory cleanup triggered at step "
                f"{step}: GPU={info.get('gpu_allocated_gb', 0):.2f}GB, RAM={info['ram_used_gb']:.2f}GB"
            )
            self._steps_since_cleanup = 0

        if (
            self._hook_config.log_interval > 0
            and self._steps_since_log >= self._hook_config.log_interval
        ):
            summary = self._monitor.get_memory_summary()
            print(summary)
            self._steps_since_log = 0

    def on_oom(self) -> None:
        if not self._monitor:
            return
        print("🛟 Memory monitor captured an OOM event, forcing immediate cleanup")
        self._monitor.force_cleanup()
        summary = self._monitor.get_memory_summary()
        print(summary)
