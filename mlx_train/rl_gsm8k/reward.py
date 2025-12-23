from __future__ import annotations

import re
from typing import Optional, Tuple


_ANSWER_PATTERNS: Tuple[re.Pattern[str], ...] = (
    re.compile(r"####\s*([-+]?\d[\d,]*(?:\.\d+)?)"),
    re.compile(r"(?:最终答案|答案)\s*[:：]?\s*([-+]?\d[\d,]*(?:\.\d+)?)"),
)

_FALLBACK_NUMBER = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?")


def _normalize_number(num: str) -> Optional[str]:
    raw = (num or "").strip().replace(",", "")
    if not raw:
        return None

    # Prefer integer normalization (GSM8K is typically integer).
    if re.fullmatch(r"[-+]?\d+", raw):
        try:
            return str(int(raw))
        except Exception:
            return None

    if re.fullmatch(r"[-+]?\d+\.\d+", raw):
        try:
            val = float(raw)
        except Exception:
            return None
        if val.is_integer():
            return str(int(val))
        # Keep a compact decimal representation (no trailing zeros).
        s = f"{val:.10f}".rstrip("0").rstrip(".")
        return s if s else None

    return raw


def extract_final_answer(text: str) -> Optional[str]:
    s = (text or "").strip()
    if not s:
        return None

    for pat in _ANSWER_PATTERNS:
        matches = pat.findall(s)
        if matches:
            out = _normalize_number(matches[-1])
            if out is not None:
                return out

    # Fallback: last number in the output.
    matches = _FALLBACK_NUMBER.findall(s)
    if not matches:
        return None
    return _normalize_number(matches[-1])


def reward_gsm8k(pred_text: str, ref_answer_text: str) -> float:
    pred = extract_final_answer(pred_text)
    ref = extract_final_answer(ref_answer_text)
    if pred is None or ref is None:
        return 0.0
    return 1.0 if pred == ref else 0.0

