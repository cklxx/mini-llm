#!/usr/bin/env python3
"""Lightweight structural checks for the MiniGPT project."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))


def iter_required_modules() -> Iterable[str]:
    """Yield the core modules that should remain importable."""

    yield "src.model.config"
    yield "src.model.transformer"
    yield "src.model.rope"
    yield "src.model.gqa"


def check_python_requires() -> bool:
    """Ensure the project metadata pins a supported Python version."""

    pyproject = PROJECT_ROOT / "pyproject.toml"
    if not pyproject.exists():
        print(f"❌ pyproject.toml missing at {pyproject}")
        return False

    contents = pyproject.read_text(encoding="utf-8")
    if "requires-python" not in contents:
        print("❌ Missing requires-python declaration")
        return False

    return True


def check_core_imports() -> bool:
    """Verify critical modules can be imported without side effects."""

    all_good = True
    for module_name in iter_required_modules():
        try:
            importlib.import_module(module_name)
            print(f"✅ {module_name} import succeeded")
        except ModuleNotFoundError as exc:  # pragma: no cover - defensive logging
            if exc.name == "torch":
                print(f"⚠️ {module_name} import skipped (missing optional dependency: {exc.name})")
            else:
                all_good = False
                print(f"❌ {module_name} import failed: {exc}")
        except Exception as exc:  # pragma: no cover - defensive logging
            all_good = False
            print(f"❌ {module_name} import failed: {exc}")
    return all_good


def main() -> int:
    structure_ok = check_python_requires()
    imports_ok = check_core_imports()
    if structure_ok and imports_ok:
        print("🎉 Code structure looks good!")
        return 0
    print("⚠️ Issues detected in code structure checks")
    return 1


if __name__ == "__main__":
    sys.exit(main())

