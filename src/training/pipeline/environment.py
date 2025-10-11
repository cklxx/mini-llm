"""Runtime environment helpers used by the training pipeline."""

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class TrainingEnvironment:
    """Encapsulate device/seed setup and run bookkeeping."""

    config: Any
    mode: str
    output_dir: str = field(init=False)
    device: str = field(init=False)
    dataset_stats: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.output_dir = os.path.join(
            self.config.checkpoint_dir, f"{self.mode}_{self.config.model_size}"
        )
        os.makedirs(self.output_dir, exist_ok=True)
        self._set_random_seed(getattr(self.config, "random_seed", 42))
        self.device = self._setup_device()
        self._snapshot_config()

    # ------------------------------------------------------------------
    def _setup_device(self) -> str:
        if torch.backends.mps.is_available():
            device = "mps"
            print("🔧 使用Apple Silicon GPU (MPS)")
        elif torch.cuda.is_available():
            device = "cuda"
            print(f"🔧 使用CUDA GPU: {torch.cuda.get_device_name()}")
        else:
            device = "cpu"
            print("🔧 使用CPU")
        return device

    def _set_random_seed(self, seed: int) -> None:
        random.seed(seed)
        try:
            import numpy as np

            np.random.seed(seed)
        except Exception:
            pass

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    # ------------------------------------------------------------------
    def _serialize_config_value(self, value: Any):
        if isinstance(value, int | float | str | bool) or value is None:
            return value
        if isinstance(value, list | tuple):
            return [self._serialize_config_value(v) for v in value]
        if isinstance(value, dict):
            return {str(k): self._serialize_config_value(v) for k, v in value.items()}
        return str(value)

    def _snapshot_config(self) -> None:
        try:
            config_dict = {
                key: self._serialize_config_value(val)
                for key, val in vars(self.config).items()
                if not key.startswith("_")
            }
            snapshot_path = os.path.join(self.output_dir, "training_config_snapshot.json")
            with open(snapshot_path, "w", encoding="utf-8") as handle:
                json.dump(config_dict, handle, indent=2, ensure_ascii=False)
            print(f"📝 配置快照已保存: {snapshot_path}")
        except Exception as exc:
            print(f"⚠️  配置快照保存失败: {exc}")

    # ------------------------------------------------------------------
    def persist_dataset_stats(self) -> None:
        if not self.dataset_stats:
            return
        stats_path = os.path.join(self.output_dir, "dataset_stats.json")
        try:
            with open(stats_path, "w", encoding="utf-8") as handle:
                json.dump(self.dataset_stats, handle, indent=2, ensure_ascii=False)
            print(f"📊 数据集统计已保存: {stats_path}")
        except Exception as exc:
            print(f"⚠️  数据集统计保存失败: {exc}")
