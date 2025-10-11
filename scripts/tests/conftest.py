"""Test configuration helpers for MiniGPT."""

from __future__ import annotations

import warnings

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    """Provide no-op coverage options when pytest-cov is unavailable."""
    if parser.getgroup("cov") is not None:
        # pytest-cov already registered the coverage options.
        return

    group = parser.getgroup("cov")
    group.addoption(
        "--cov",
        action="store",
        default=None,
        help="Ignored because pytest-cov is not installed.",
    )
    group.addoption(
        "--cov-report",
        action="append",
        default=[],
        help="Ignored because pytest-cov is not installed.",
    )


def pytest_configure(config: pytest.Config) -> None:
    """Notify users when coverage options are ignored."""
    if config.pluginmanager.hasplugin("pytest_cov"):
        return

    if getattr(config.option, "cov", None) or getattr(config.option, "cov_report", None):
        warnings.warn(
            "pytest-cov is not installed; coverage options will be ignored.",
            RuntimeWarning,
            stacklevel=2,
        )
