#!/usr/bin/env python3
"""Convert JSON array datasets to JSONL format.

Handles large files by streaming objects out of a JSON array (common in
Hugging Face exports). Allows concatenation of title/text fields.

Example:
    python scripts/convert_to_jsonl.py \
        --input data/external/wikipedia_zh/wiki_pretrain_part1.json \
        --output data/processed/wiki_zh_part1.jsonl \
        --text-field text --title-field title --join-with "\n\n"
"""

import argparse
import json
from collections.abc import Generator
from pathlib import Path


def stream_json_array(path: Path) -> Generator[dict, None, None]:
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
                obj, offset = decoder.raw_decode(buffer)
                yield obj


def build_text(obj: dict, text_field: str, title_field: str | None, join_with: str) -> str:
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
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.output.open("w", encoding="utf-8") as out:
        for obj in stream_json_array(args.input):
            text = build_text(obj, args.text_field, args.title_field, args.join_with)
            if not text:
                continue
            record = {"text": text}
            if args.language:
                record["lang"] = args.language
            if args.add_metadata:
                for k, v in obj.items():
                    if k in {args.text_field, args.title_field}:
                        continue
                    record[k] = v
            out.write(json.dumps(record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
