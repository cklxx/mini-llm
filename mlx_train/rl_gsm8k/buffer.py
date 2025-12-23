from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Union


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class RolloutRecord:
    example_id: str
    messages: list[dict[str, str]]
    response: str
    reward: float
    reference_answer: str
    pred_final: Optional[str]
    ref_final: Optional[str]
    meta: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "example_id": self.example_id,
            "messages": self.messages,
            "response": self.response,
            "reward": float(self.reward),
            "reference_answer": self.reference_answer,
            "pred_final": self.pred_final,
            "ref_final": self.ref_final,
            "meta": self.meta,
        }


class JsonlRolloutBuffer:
    def __init__(self, path: Union[str, os.PathLike[str]]):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, record: Union[RolloutRecord, Dict[str, Any]]) -> None:
        obj = record.to_dict() if isinstance(record, RolloutRecord) else record
        line = json.dumps(obj, ensure_ascii=False)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
            f.flush()

    def iter(self, *, follow: bool = False, poll_s: float = 0.25) -> Iterator[Dict[str, Any]]:
        """
        Iterate JSONL records. If follow=True, wait for new lines when reaching EOF
        (useful when rollouts are being appended by another process).
        """
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.touch(exist_ok=True)
        with open(self.path, "r", encoding="utf-8") as f:
            while True:
                pos = f.tell()
                line = f.readline()
                if not line:
                    if not follow:
                        break
                    time.sleep(poll_s)
                    f.seek(pos)
                    continue
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                if isinstance(obj, dict):
                    yield obj


def count_jsonl_lines(path: Union[str, os.PathLike[str]], *, max_lines: Optional[int] = None) -> int:
    p = Path(path)
    if not p.is_file():
        return 0
    n = 0
    with open(p, "r", encoding="utf-8") as f:
        for _ in f:
            n += 1
            if max_lines is not None and n >= int(max_lines):
                break
    return n

