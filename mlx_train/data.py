from __future__ import annotations

import glob
import json
import os
import random
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple, cast


def resolve_jsonl_paths(data_path: str) -> List[str]:
    parts: List[str] = []
    for piece in (p.strip() for p in data_path.split(",")):
        if not piece:
            continue
        if os.path.isdir(piece):
            parts.extend(sorted(glob.glob(os.path.join(piece, "*.jsonl"))))
        else:
            expanded = sorted(glob.glob(piece))
            parts.extend(expanded if expanded else [piece])

    paths = [p for p in parts if os.path.isfile(p)]
    if not paths:
        raise FileNotFoundError(f"No JSONL files found from: {data_path}")
    return paths


def iter_jsonl(paths: Sequence[str]) -> Iterator[Dict[str, Any]]:
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if not isinstance(obj, dict):
                    raise ValueError(f"Expected a JSON object per line in {path}, got: {type(obj).__name__}")
                yield cast(Dict[str, Any], obj)


def shuffle_stream(
    items: Iterable[Dict[str, Any]], *, buffer_size: int, seed: int
) -> Iterator[Dict[str, Any]]:
    if buffer_size <= 0:
        yield from items
        return

    rng = random.Random(seed)
    buf: List[Dict[str, Any]] = []
    for item in items:
        if len(buf) < buffer_size:
            buf.append(item)
            continue
        idx = rng.randrange(len(buf))
        yield buf[idx]
        buf[idx] = item
    rng.shuffle(buf)
    yield from buf


def _pad_or_truncate(ids: List[int], *, length: int, pad_id: int) -> List[int]:
    if len(ids) >= length:
        return ids[:length]
    return ids + [pad_id] * (length - len(ids))


def _generate_sft_loss_mask(input_ids: Sequence[int], bos_id: Sequence[int], eos_id: Sequence[int], max_length: int) -> List[int]:
    loss_mask = [0] * len(input_ids)
    i = 0
    while i < len(input_ids):
        if list(input_ids[i : i + len(bos_id)]) == list(bos_id):
            start = i + len(bos_id)
            end = start
            while end < len(input_ids):
                if list(input_ids[end : end + len(eos_id)]) == list(eos_id):
                    break
                end += 1
            for j in range(start + 1, min(end + len(eos_id) + 1, max_length)):
                loss_mask[j] = 1
            i = end + len(eos_id) if end < len(input_ids) else len(input_ids)
        else:
            i += 1
    return loss_mask


@dataclass(frozen=True)
class TokenizedBatch:
    x: List[List[int]]
    y: List[List[int]]
    loss_mask: List[List[int]]


def tokenize_pretrain_sample(
    *,
    tokenizer,
    text: str,
    seq_len: int,
    pad_id: int,
    add_special_tokens: bool = False,
) -> Tuple[List[int], List[int], List[int]]:
    ids: List[int] = tokenizer.encode(text, add_special_tokens=add_special_tokens)
    ids = _pad_or_truncate(ids, length=seq_len + 1, pad_id=pad_id)
    x = ids[:-1]
    y = ids[1:]
    mask = [1 if t != pad_id else 0 for t in y]
    return x, y, mask


def tokenize_sft_sample(
    *,
    tokenizer,
    conversations: Sequence[Dict[str, Any]],
    seq_len: int,
    pad_id: int,
    bos_id: Sequence[int],
    eos_id: Sequence[int],
) -> Tuple[List[int], List[int], List[int]]:
    prompt = tokenizer.apply_chat_template(conversations, tokenize=False, add_generation_prompt=False)
    ids: List[int] = tokenizer.encode(prompt, add_special_tokens=False)
    ids = _pad_or_truncate(ids, length=seq_len + 1, pad_id=pad_id)

    loss_mask = _generate_sft_loss_mask(ids, bos_id, eos_id, seq_len + 1)
    x = ids[:-1]
    y = ids[1:]
    mask = loss_mask[1:]
    return x, y, mask


def make_batch_iterator(
    *,
    paths: Sequence[str],
    tokenizer,
    task: str,
    seq_len: int,
    batch_size: int,
    shuffle_buffer: int,
    seed: int,
) -> Iterator[TokenizedBatch]:
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        raise ValueError("Tokenizer must have pad_token_id set.")

    bos_id: Optional[List[int]] = None
    eos_id: Optional[List[int]] = None
    if task == "sft":
        bos_id = tokenizer.encode(f"{tokenizer.bos_token}assistant", add_special_tokens=False)
        eos_id = tokenizer.encode(f"{tokenizer.eos_token}", add_special_tokens=False)

    stream: Iterable[Dict[str, Any]] = iter_jsonl(paths)
    stream = shuffle_stream(stream, buffer_size=shuffle_buffer, seed=seed)

    cur_x: List[List[int]] = []
    cur_y: List[List[int]] = []
    cur_m: List[List[int]] = []

    for obj in stream:
        if task == "pretrain":
            if "text" in obj:
                text = str(obj["text"])
            elif "conversations" in obj:
                conversations = cast(Sequence[Dict[str, Any]], obj["conversations"])
                text = tokenizer.apply_chat_template(conversations, tokenize=False, add_generation_prompt=False)
            else:
                text = json.dumps(obj, ensure_ascii=False)
            x, y, m = tokenize_pretrain_sample(tokenizer=tokenizer, text=text, seq_len=seq_len, pad_id=pad_id)
        elif task == "sft":
            if "conversations" not in obj:
                raise ValueError("SFT task expects JSONL lines with a `conversations` field.")
            x, y, m = tokenize_sft_sample(
                tokenizer=tokenizer,
                conversations=cast(Sequence[Dict[str, Any]], obj["conversations"]),
                seq_len=seq_len,
                pad_id=pad_id,
                bos_id=bos_id or [],
                eos_id=eos_id or [],
            )
        else:
            raise ValueError(f"Unknown task: {task}")

        cur_x.append(x)
        cur_y.append(y)
        cur_m.append(m)

        if len(cur_x) >= batch_size:
            yield TokenizedBatch(x=cur_x, y=cur_y, loss_mask=cur_m)
            cur_x, cur_y, cur_m = [], [], []
