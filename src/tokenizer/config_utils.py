"""Utilities for working with tokenizer configuration metadata."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

__all__ = ["canonicalize_tokenizer_config"]


def canonicalize_tokenizer_config(config: Mapping[str, Any] | None) -> dict[str, Any]:
    """Return a normalized copy of a tokenizer configuration mapping.

    Checkpoints may embed tokenizer configuration dictionaries that include
    environment-specific absolute paths (for example ``tokenizer_dir``).  These
    fields are informative but should not participate in strict equality checks,
    otherwise moving checkpoints between machines causes spurious mismatches.

    The canonical representation removes path-sensitive keys so downstream
    comparisons focus on semantic attributes such as vocabulary size and special
    token definitions.
    """

    if not config:
        return {}

    sanitized: dict[str, Any] = deepcopy(dict(config))
    sanitized.pop("tokenizer_dir", None)
    return sanitized

