"""
Download code-focused datasets from Hugging Face and export them to disk for
MiniGPT experiments. Supports both streaming and in-memory modes, optional
language filters, and JSONL or Parquet export formats.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from ast import literal_eval
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path
from typing import Any

from datasets import Dataset, IterableDataset, load_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download code corpora from Hugging Face and export to disk.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "datasets",
        nargs="+",
        help="Dataset identifiers, e.g. bigcode/the-stack-smol codeparrot/codeparrot-clean",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Dataset configuration name (if the dataset exposes named subsets).",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train"],
        help="Dataset splits to download.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/raw/code_corpus"),
        help="Root directory where exported files are stored.",
    )
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="Use streaming mode to iterate over large datasets.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional cap on the number of samples exported per split.",
    )
    parser.add_argument(
        "--shuffle-seed",
        type=int,
        default=None,
        help="Shuffle the dataset with the provided seed before sampling (non-streaming only).",
    )
    parser.add_argument(
        "--languages",
        nargs="+",
        default=None,
        help="Limit to the listed programming languages (datasets that support the argument only).",
    )
    parser.add_argument(
        "--file-format",
        choices=("jsonl", "parquet"),
        default="jsonl",
        help="Output format for exported samples.",
    )
    parser.add_argument(
        "--compression",
        default=None,
        help="Compression codec for Parquet export (snappy, gzip, zstd, ...).",
    )
    parser.add_argument(
        "--extra-kwarg",
        action="append",
        default=None,
        help="Additional key=value arguments forwarded to datasets.load_dataset.",
    )
    parser.add_argument(
        "--save-cache",
        action="store_true",
        help="Keep the Hugging Face datasets cache on disk instead of removing files after export.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing export files instead of skipping them.",
    )
    parser.add_argument(
        "--stats-output",
        type=Path,
        default=None,
        help="Optional path to store a JSON summary of exported files.",
    )
    return parser.parse_args()


def parse_extra_kwargs(pairs: Sequence[str] | None) -> dict[str, Any]:
    if not pairs:
        return {}
    parsed: dict[str, Any] = {}
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"Invalid extra kwarg '{pair}', expected key=value format.")
        key, value = pair.split("=", 1)
        key = key.strip()
        value = value.strip()
        try:
            parsed[key] = literal_eval(value)
        except Exception:
            parsed[key] = value
    return parsed


def iter_samples(
    dataset: Dataset | IterableDataset,
    *,
    max_samples: int | None,
) -> Iterator[dict[str, Any]]:
    if isinstance(dataset, IterableDataset):
        iterable: Iterable[dict[str, Any]]
        if max_samples is not None:
            iterable = dataset.take(max_samples)
        else:
            iterable = dataset
        for example in iterable:
            yield example
        return

    materialized: Dataset = dataset
    if max_samples is not None:
        capped = min(max_samples, len(materialized))
        materialized = materialized.select(range(capped))
    for example in materialized:
        yield example


def export_jsonl(
    dataset: Dataset | IterableDataset,
    *,
    output_file: Path,
    max_samples: int | None,
) -> tuple[int, int]:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with output_file.open("w", encoding="utf-8") as fp:
        for example in iter_samples(dataset, max_samples=max_samples):
            fp.write(json.dumps(example, ensure_ascii=False) + "\n")
            written += 1
    size = output_file.stat().st_size if output_file.exists() else 0
    return written, size


def export_parquet(
    dataset: Dataset,
    *,
    output_file: Path,
    compression: str | None,
    max_samples: int | None,
) -> tuple[int, int]:
    subset = dataset
    if max_samples is not None:
        capped = min(max_samples, len(dataset))
        subset = dataset.select(range(capped))
    output_file.parent.mkdir(parents=True, exist_ok=True)
    subset.to_parquet(str(output_file), compression=compression)
    row_count = subset.num_rows
    size = output_file.stat().st_size if output_file.exists() else 0
    return row_count, size


def main() -> None:
    args = parse_args()
    extra_kwargs = parse_extra_kwargs(args.extra_kwarg)
    export_stats: list[dict[str, Any]] = []

    for dataset_name in args.datasets:
        print(f"==> Processing dataset: {dataset_name}")
        for split in args.splits:
            load_kwargs: dict[str, Any] = dict(extra_kwargs)
            if args.languages is not None:
                load_kwargs.setdefault("languages", args.languages)
            dataset = load_dataset(
                path=dataset_name,
                name=args.config,
                split=split,
                streaming=args.streaming,
                **load_kwargs,
            )

            if not isinstance(dataset, IterableDataset) and args.shuffle_seed is not None:
                dataset = dataset.shuffle(seed=args.shuffle_seed)

            split_dir = args.output_dir / dataset_name.replace("/", "_")
            split_dir.mkdir(parents=True, exist_ok=True)

            if args.file_format == "jsonl":
                output_file = split_dir / f"{split}.jsonl"
                if output_file.exists() and not args.force:
                    print(f"Skipping existing file {output_file} (use --force to overwrite)")
                    export_stats.append(
                        {
                            "dataset": dataset_name,
                            "split": split,
                            "path": str(output_file),
                            "format": "jsonl",
                            "skipped": True,
                        }
                    )
                else:
                    records, bytes_written = export_jsonl(
                        dataset, output_file=output_file, max_samples=args.max_samples
                    )
                    print(
                        f"Saved {dataset_name}:{split} to {output_file}"
                        f" ({records} records, {bytes_written} bytes)"
                    )
                    export_stats.append(
                        {
                            "dataset": dataset_name,
                            "split": split,
                            "path": str(output_file),
                            "format": "jsonl",
                            "records": records,
                            "bytes": bytes_written,
                            "max_samples": args.max_samples,
                            "streaming": args.streaming,
                        }
                    )
            else:
                if isinstance(dataset, IterableDataset):
                    raise ValueError("Parquet export is not supported in streaming mode.")
                output_file = split_dir / f"{split}.parquet"
                if output_file.exists() and not args.force:
                    print(f"Skipping existing file {output_file} (use --force to overwrite)")
                    export_stats.append(
                        {
                            "dataset": dataset_name,
                            "split": split,
                            "path": str(output_file),
                            "format": "parquet",
                            "skipped": True,
                        }
                    )
                else:
                    records, bytes_written = export_parquet(
                        dataset,
                        output_file=output_file,
                        compression=args.compression,
                        max_samples=args.max_samples,
                    )
                    print(
                        f"Saved {dataset_name}:{split} to {output_file}"
                        f" ({records} rows, {bytes_written} bytes)"
                    )
                    export_stats.append(
                        {
                            "dataset": dataset_name,
                            "split": split,
                            "path": str(output_file),
                            "format": "parquet",
                            "records": records,
                            "bytes": bytes_written,
                            "compression": args.compression,
                            "max_samples": args.max_samples,
                        }
                    )

    cache_dir = Path(os.getenv("HF_DATASETS_CACHE", Path.home() / ".cache/huggingface/datasets"))
    if not args.save_cache and cache_dir.exists():
        print(f"Removing Hugging Face datasets cache at {cache_dir}")
        shutil.rmtree(cache_dir, ignore_errors=True)
    elif cache_dir.exists():
        print(f"Cache retained at {cache_dir}")

    if export_stats:
        print("\nExport summary:")
        for item in export_stats:
            if item.get("skipped"):
                print(f" - {item['dataset']}:{item['split']} skipped (existing {item['path']})")
            else:
                print(
                    " - {dataset}:{split} -> {path} ({records} records, {bytes} bytes)".format(
                        dataset=item["dataset"],
                        split=item["split"],
                        path=item["path"],
                        records=item.get("records", "?"),
                        bytes=item.get("bytes", "?"),
                    )
                )
        if args.stats_output is not None:
            args.stats_output.parent.mkdir(parents=True, exist_ok=True)
            args.stats_output.write_text(
                json.dumps(export_stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            print(f"📄 导出统计已保存至 {args.stats_output}")


if __name__ == "__main__":
    main()
