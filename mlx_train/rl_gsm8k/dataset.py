from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence


@dataclass(frozen=True)
class GSM8KExample:
    example_id: str
    question: str
    answer: str
    meta: Dict[str, Any]


_QUESTION_KEYS = ("question", "query", "prompt", "problem", "input")
_ANSWER_KEYS = ("answer", "solution", "output", "completion", "target")


def _first_str(obj: Dict[str, Any], keys: Sequence[str]) -> Optional[str]:
    for k in keys:
        v = obj.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None


def _iter_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if not isinstance(obj, dict):
                continue
            yield obj


def _iter_json(path: Path) -> Iterator[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if isinstance(obj, list):
        for item in obj:
            if isinstance(item, dict):
                yield item
        return
    if isinstance(obj, dict):
        data = obj.get("data")
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    yield item
            return
        yield obj


def _iter_parquet(path: Path, *, batch_size: int = 1024) -> Iterator[Dict[str, Any]]:
    try:
        import pyarrow.parquet as pq
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "Reading .parquet requires `pyarrow`. Install: `pip install -U pyarrow`"
        ) from e

    pf = pq.ParquetFile(path)
    for batch in pf.iter_batches(batch_size=batch_size):
        for row in batch.to_pylist():
            if isinstance(row, dict):
                yield row


def find_split_files(dataset_dir: str, *, split: str) -> List[Path]:
    root = Path(dataset_dir)
    if not root.exists():
        raise FileNotFoundError(
            f"dataset_dir not found: {dataset_dir}. "
            "Download first (e.g. `python -m mlx_train.rl_gsm8k.download --local_dir <dir>`), "
            "or pass `--dataset_dir <dir>` / set `DATASET_DIR=<dir>` when using the bash script."
        )

    pats = [
        f"{split}.jsonl",
        f"{split}.json",
        f"{split}.parquet",
        f"{split}_*.jsonl",
        f"{split}_*.json",
        f"{split}_*.parquet",
        f"*{split}*.jsonl",
        f"*{split}*.json",
        f"*{split}*.parquet",
    ]
    found: List[Path] = []
    seen: set[Path] = set()
    for pat in pats:
        for p in root.rglob(pat):
            if p.is_file() and p.suffix in {".jsonl", ".json", ".parquet"} and p not in seen:
                found.append(p)
                seen.add(p)
    found.sort()
    return found


def iter_gsm8k(dataset_dir: str, *, split: str) -> Iterator[GSM8KExample]:
    files = find_split_files(dataset_dir, split=split)
    if not files:
        raise FileNotFoundError(
            f"No {split} data files found under {dataset_dir}. "
            "Expected *.jsonl / *.json / *.parquet containing GSM8K-style fields."
        )

    for path in files:
        if path.suffix == ".jsonl":
            it: Iterable[Dict[str, Any]] = _iter_jsonl(path)
        elif path.suffix == ".parquet":
            it = _iter_parquet(path)
        else:
            it = _iter_json(path)

        for idx, obj in enumerate(it):
            q = _first_str(obj, _QUESTION_KEYS)
            a = _first_str(obj, _ANSWER_KEYS)
            if not q or not a:
                continue
            ex_id = str(obj.get("id") or obj.get("_id") or f"{path.name}:{idx}")
            meta = {"source_file": os.fspath(path), "source_idx": idx}
            yield GSM8KExample(example_id=ex_id, question=q, answer=a, meta=meta)


def load_gsm8k_list(dataset_dir: str, *, split: str, limit: Optional[int] = None) -> List[GSM8KExample]:
    items: List[GSM8KExample] = []
    for ex in iter_gsm8k(dataset_dir, split=split):
        items.append(ex)
        if limit is not None and len(items) >= int(limit):
            break
    return items
