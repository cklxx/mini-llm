"""
Stream a large JSON array (e.g. Common Crawl or Wikipedia exports) and convert it
into JSONL so MiniGPT's data pipeline can process it. Handles memory-efficient
parsing and optional metadata retention. Also supports dropping short records.

Example (limit to the first few records during smoke tests):
    python scripts/convert_to_jsonl.py \
        --input data/external/wikipedia_zh/wiki_pretrain_part1.json \
        --output data/processed/wiki_zh_part1.jsonl \
        --text-field text --title-field title --join-with "\n\n" --max-records 1000
"""

import argparse
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any


def stream_json_array(path: Path) -> Iterator[dict[str, Any]]:
    decoder = json.JSONDecoder()
    with path.open("r", encoding="utf-8") as f:
        buffer = ""
        chunk = f.read(65536)
        while chunk:
            buffer += chunk
            while True:
                buffer = buffer.lstrip()
                if not buffer:
                    break
                if buffer[0] == "[":
                    buffer = buffer[1:]
                    continue
                if buffer[0] == ",":
                    buffer = buffer[1:]
                    continue
                if buffer[0] == "]":
                    return
                try:
                    obj, offset = decoder.raw_decode(buffer)
                except json.JSONDecodeError:
                    break  # need more data
                yield obj
                buffer = buffer[offset:]
            chunk = f.read(65536)
        buffer = buffer.strip()
        if buffer and buffer not in ("]", ""):
            if buffer[0] == ",":
                buffer = buffer[1:]
            buffer = buffer.strip()
            if buffer and buffer[0] != "]":
                obj, _ = decoder.raw_decode(buffer)
                yield obj


def build_text(
    obj: dict[str, Any],
    text_field: str,
    title_field: str | None,
    join_with: str,
) -> str:
    pieces = []
    if title_field:
        title = obj.get(title_field)
        if isinstance(title, str) and title:
            pieces.append(title.strip())
    text = obj.get(text_field)
    if isinstance(text, str) and text:
        pieces.append(text.strip())
    return join_with.join(pieces)


def main() -> None:
    parser = argparse.ArgumentParser(description="Stream JSON array to JSONL")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--title-field", default=None)
    parser.add_argument("--join-with", default="\n\n")
    parser.add_argument("--add-metadata", action="store_true", help="Keep non-text columns")
    parser.add_argument("--language", default=None)
    parser.add_argument(
        "--min-length",
        type=int,
        default=0,
        help="Drop records whose assembled text is shorter than this length.",
    )
    parser.add_argument(
        "--max-records",
        type=int,
        default=None,
        help="Write at most this many records (useful for smoke tests).",
    )
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.output.open("w", encoding="utf-8") as out:
        title_field = args.title_field
        if isinstance(title_field, str) and title_field.lower() in {"none", ""}:
            title_field = None

        written = 0
        for obj in stream_json_array(args.input):
            text = build_text(obj, args.text_field, title_field, args.join_with)
            if not text:
                continue
            if args.min_length and len(text.strip()) < args.min_length:
                continue
            record = {"text": text}
            if args.language:
                record["lang"] = args.language
            if args.add_metadata:
                for k, v in obj.items():
                    if k in {args.text_field, title_field}:
                        continue
                    record[k] = v
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1
            if args.max_records is not None and written >= args.max_records:
                break


if __name__ == "__main__":
    main()
