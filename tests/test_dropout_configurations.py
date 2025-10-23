import inspect
import sys
from collections.abc import Callable
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.model import config as model_config

try:  # pragma: no cover - allow running tests without torch installed
    import torch  # noqa: F401
except ModuleNotFoundError:  # pragma: no cover - torch optional for GPU configs
    GPU_MODULES: list[object] = []
else:
    from config import rtx3090_config, rtx4090_config

    GPU_MODULES = [rtx3090_config, rtx4090_config]


@pytest.mark.parametrize("config_name", sorted(model_config.CONFIG_MAPPING.keys()))
def test_default_model_configs_define_dropout(config_name: str) -> None:
    cfg = model_config.get_config(config_name)
    assert hasattr(cfg, "dropout"), f"{config_name} 缺少 dropout 配置"
    assert cfg.dropout is not None and cfg.dropout >= 0.0, (
        f"{config_name} dropout 设置异常: {cfg.dropout}"
    )
    assert hasattr(cfg, "attention_dropout"), f"{config_name} 缺少 attention_dropout 配置"
    assert cfg.attention_dropout is not None and cfg.attention_dropout >= 0.0, (
        f"{config_name} attention_dropout 设置异常: {cfg.attention_dropout}"
    )


def _collect_gpu_specific_factories() -> list[tuple[str, Callable[[], object]]]:
    if not GPU_MODULES:
        return []

    factories: list[tuple[str, Callable[[], object]]] = []
    modules = GPU_MODULES
    for module in modules:
        for name, obj in inspect.getmembers(module, inspect.isfunction):
            if not name.startswith("get_"):
                continue
            signature = inspect.signature(obj)
            if signature.parameters:
                continue
            factories.append((f"{module.__name__}.{name}", obj))
    return factories


@pytest.mark.parametrize("factory_name,factory", _collect_gpu_specific_factories())
def test_gpu_specific_configs_define_dropout(factory_name: str, factory) -> None:
    cfg = factory()
    assert hasattr(cfg, "dropout"), f"{factory_name} 缺少 dropout 配置"
    assert cfg.dropout is not None and cfg.dropout >= 0.0, (
        f"{factory_name} dropout 设置异常: {cfg.dropout}"
    )
    assert hasattr(cfg, "attention_dropout"), f"{factory_name} 缺少 attention_dropout 配置"
    assert cfg.attention_dropout is not None and cfg.attention_dropout >= 0.0, (
        f"{factory_name} attention_dropout 设置异常: {cfg.attention_dropout}"
    )
