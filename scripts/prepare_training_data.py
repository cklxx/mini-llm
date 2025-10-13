#!/usr/bin/env python3
"""One-click data preparation pipeline for MiniGPT training.

This script downloads raw corpora (Chinese Wiki, Chinese mixed-domain, SlimPajama
English subset), performs conversion to JSONL, runs SimHash-based near-duplicate
filtering, and produces cleaned datasets plus manifests ready for training.

Usage example:
    python scripts/prepare_training_data.py --workers 8

The pipeline is idempotent: existing files are reused unless --force is passed.
Downloads and heavy processing steps exploit multiple CPU cores using
ThreadPoolExecutor / ProcessPoolExecutor.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import multiprocessing
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import requests

# ---------------------------------------------------------------------------
# SimHash utilities (adapted from previous scripts)
# ---------------------------------------------------------------------------

def feature_hash(feature: str) -> int:
    digest = hashlib.md5(feature.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def tokenize_for_simhash(text: str, max_features: int = 32) -> List[str]:
    length = len(text)
    if not text or length < 2:
        return []
    stride = max(1, length // (max_features * 2))
    feats: List[str] = []
    for n in (2, 3):
        for i in range(0, length - n + 1, stride):
            feats.append(text[i : i + n])
            if len(feats) >= max_features:
                break
        if len(feats) >= max_features:
            break
    for token in text.split():
        if token:
            feats.append(token)
            if len(feats) >= max_features:
                break
    return feats


def simhash(feats: Iterable[str]) -> int:
    weights = [0] * 64
    for feat in feats:
        h = feature_hash(feat)
        for bit in range(64):
            weights[bit] += 1 if h & (1 << bit) else -1
    fp = 0
    for bit, weight in enumerate(weights):
        if weight >= 0:
            fp |= 1 << bit
    return fp


def hamming_distance(a: int, b: int) -> int:
    x = a ^ b
    count = 0
    while x:
        x &= x - 1
        count += 1
    return count


def ensure_config(bands: int, bits_per_band: int) -> None:
    if bands * bits_per_band != 64:
        raise ValueError("bands * bits_per_band must equal 64")


def band_keys(fp: int, bands: int, bits_per_band: int) -> Iterable[int]:
    mask = (1 << bits_per_band) - 1
    for i in range(bands):
        shift = i * bits_per_band
        yield (fp >> shift) & mask


def simhash_dedupe(
    input_paths: Sequence[Path],
    output_path: Path,
    casefold: bool = False,
    bands: int = 4,
    bits_per_band: int = 16,
    hamming_threshold: int = 3,
    max_bucket_size: int = 64,
    min_length: int = 10,
) -> Tuple[int, int]:
    """Stream multiple JSONL files and remove near duplicates."""
    ensure_config(bands, bits_per_band)
    band_maps: List[Dict[int, List[int]]] = [dict() for _ in range(bands)]
    kept = 0
    dropped = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as dst:
        for path in input_paths:
            with path.open("r", encoding="utf-8") as src:
                for raw in src:
                    raw_strip = raw.strip()
                    if not raw_strip:
                        continue
                    obj = json.loads(raw_strip)
                    text = obj.get("text", "")
                    if not isinstance(text, str):
                        continue
                    if len(text) < min_length:
                        dst.write(raw_strip + "\n")
                        kept += 1
                        continue
                    if casefold:
                        text = text.lower()
                    fp = simhash(tokenize_for_simhash(text))
                    is_dup = False
                    for idx, key in enumerate(band_keys(fp, bands, bits_per_band)):
                        bucket = band_maps[idx].setdefault(key, [])
                        for existing in bucket:
                            if hamming_distance(fp, existing) <= hamming_threshold:
                                is_dup = True
                                break
                        if is_dup:
                            break
                    if is_dup:
                        dropped += 1
                        continue
                    dst.write(raw_strip + "\n")
                    kept += 1
                    for idx, key in enumerate(band_keys(fp, bands, bits_per_band)):
                        bucket = band_maps[idx].setdefault(key, [])
                        if max_bucket_size and len(bucket) >= max_bucket_size:
                            continue
                        bucket.append(fp)
    return kept, dropped

# ---------------------------------------------------------------------------
# JSONL conversion utilities
# ---------------------------------------------------------------------------

def convert_json_array_to_jsonl(input_path: Path, output_path: Path, text_field: str = "text", title_field: Optional[str] = "title", join_with: str = "\n\n") -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    decoder = json.JSONDecoder()
    buffer = ""
    with input_path.open("r", encoding="utf-8") as src, output_path.open("w", encoding="utf-8") as dst:
        while True:
            chunk = src.read(65536)
            if not chunk:
                break
            buffer += chunk
            while buffer:
                buffer = buffer.lstrip()
                if not buffer:
                    break
                if buffer[0] in "[,":
                    buffer = buffer[1:]
                    continue
                if buffer[0] == ']':
                    buffer = buffer[1:]
                    break
                try:
                    obj, offset = decoder.raw_decode(buffer)
                except json.JSONDecodeError:
                    # need more data
                    break
                pieces = []
                if title_field:
                    title = obj.get(title_field)
                    if isinstance(title, str) and title:
                        pieces.append(title.strip())
                text = obj.get(text_field, "")
                if isinstance(text, str) and text:
                    pieces.append(text.strip())
                if pieces:
                    dst.write(json.dumps({"text": join_with.join(pieces)}, ensure_ascii=False) + "\n")
                buffer = buffer[offset:]
        buffer = buffer.strip()
        if buffer and buffer[0] not in "],":
            obj, _ = decoder.raw_decode(buffer)
            text = obj.get(text_field, "")
            if isinstance(text, str) and text:
                dst.write(json.dumps({"text": text.strip()}, ensure_ascii=False) + "\n")


def convert_chinacorpus_json(input_path: Path, output_path: Path) -> None:
    convert_json_array_to_jsonl(input_path, output_path, text_field="text", title_field=None, join_with="\n")

# ---------------------------------------------------------------------------
# Download utilities
# ---------------------------------------------------------------------------

CHUNK_SIZE = 1 << 20  # 1 MiB


def download_file(url: str, dest: Path, force: bool = False) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and not force:
        print(f"[download] skip existing {dest}")
        return
    print(f"[download] fetching {url} -> {dest}")
    with requests.get(url, stream=True, timeout=60) as resp:
        resp.raise_for_status()
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            for chunk in resp.iter_content(chunk_size=CHUNK_SIZE):
                if chunk:
                    tmp.write(chunk)
            tmp_path = Path(tmp.name)
    tmp_path.replace(dest)


def parallel_download(urls: Sequence[Tuple[str, Path]], workers: int, force: bool = False) -> None:
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(download_file, url, path, force) for url, path in urls]
        for fut in concurrent.futures.as_completed(futures):
            fut.result()

# ---------------------------------------------------------------------------
# Processing routines
# ---------------------------------------------------------------------------


def run_subprocess(cmd: Sequence[str]) -> None:
    print("[cmd]", " ".join(cmd))
    subprocess.run(cmd, check=True)


def decompress_zst(source: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    run_subprocess(["zstd", "-d", "--force", str(source), "-o", str(dest)])


def combine_files(inputs: Sequence[Path], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as dst:
        for path in inputs:
            with path.open("r", encoding="utf-8") as src:
                shutil.copyfileobj(src, dst)

# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------

WIKI_PARTS = [
    (f"wiki_pretrain_part{i}.json", f"https://huggingface.co/datasets/yuhuanstudio/wikipedia-pretrain-zh/resolve/main/wiki_pretrain_part{i}.json")
    for i in range(1, 7)
]

CHINACORPUS_PARTS = [
    (f"train_{i:04d}_of_0040.json", f"https://huggingface.co/datasets/ticoAg/ChineseCorpus-Kaggle-fanti/resolve/main/data/train_{i:04d}_of_0040.json")
    for i in range(1, 32)
]

SLIMPAJAMA_FILES = [
    (f"example_train_{i}.jsonl.zst", f"https://huggingface.co/datasets/cerebras/SlimPajama-627B/resolve/main/train/chunk1/example_train_{i}.jsonl.zst")
    for i in range(50)
]


def prepare_wiki(download_dir: Path, tmp_dir: Path, final_dir: Path, workers: int, download_workers: int, force: bool) -> Path:
    final_path = final_dir / "wiki_zh_full.simdedup.jsonl"
    if final_path.exists() and not force:
        print(f"[wiki] skip existing {final_path}")
        return final_path

    urls = [(url, download_dir / fname) for fname, url in WIKI_PARTS]
    parallel_download(urls, workers=download_workers, force=force)

    converted_paths: List[Path] = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        futures = []
        for fname, _ in WIKI_PARTS:
            src = download_dir / fname
            dst = tmp_dir / f"wiki_{fname.replace('.json', '.jsonl')}"
            if dst.exists() and not force:
                converted_paths.append(dst)
                continue
            futures.append(executor.submit(convert_json_array_to_jsonl, src, dst))
            converted_paths.append(dst)
        for fut in concurrent.futures.as_completed(futures):
            fut.result()

    combined = tmp_dir / "wiki_zh_full.jsonl"
    if not combined.exists() or force:
        combine_files(converted_paths, combined)

    kept, dropped = simhash_dedupe([combined], final_path)
    print(f"[wiki] kept {kept}, dropped {dropped}")
    return final_path


def prepare_chinacorpus(download_dir: Path, tmp_dir: Path, final_dir: Path, workers: int, download_workers: int, force: bool) -> Path:
    final_path = final_dir / "chinacorpus_full.simdedup.jsonl"
    if final_path.exists() and not force:
        print(f"[chinacorpus] skip existing {final_path}")
        return final_path

    urls = [(url, download_dir / fname) for fname, url in CHINACORPUS_PARTS]
    parallel_download(urls, workers=download_workers, force=force)

    converted: List[Path] = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        futures = []
        for fname, _ in CHINACORPUS_PARTS:
            src = download_dir / fname
            dst = tmp_dir / f"chinacorpus_{fname.replace('.json', '.jsonl')}"
            if dst.exists() and not force:
                converted.append(dst)
                continue
            converted.append(dst)
            futures.append(executor.submit(convert_chinacorpus_json, src, dst))
        for fut in concurrent.futures.as_completed(futures):
            fut.result()

    combined = tmp_dir / "chinacorpus_full.jsonl"
    if not combined.exists() or force:
        combine_files(converted, combined)

    kept, dropped = simhash_dedupe([combined], final_path)
    print(f"[chinacorpus] kept {kept}, dropped {dropped}")
    return final_path


def prepare_slimpajama(download_dir: Path, tmp_dir: Path, final_dir: Path, workers: int, download_workers: int, force: bool) -> Path:
    final_path = final_dir / "slimpajama_chunk1_part0_49.cleaned.jsonl"
    if final_path.exists() and not force:
        print(f"[slimpajama] skip existing {final_path}")
        return final_path

    urls = [(url, download_dir / fname) for fname, url in SLIMPAJAMA_FILES]
    parallel_download(urls, workers=download_workers, force=force)

    decompressed: List[Path] = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        futures = []
        for fname, _ in SLIMPAJAMA_FILES:
            src = download_dir / fname
            dst = tmp_dir / fname.replace(".zst", "")
            if dst.exists() and not force:
                decompressed.append(dst)
                continue
            decompressed.append(dst)
            futures.append(executor.submit(decompress_zst, src, dst))
        for fut in concurrent.futures.as_completed(futures):
            fut.result()

    combined = tmp_dir / "slimpajama_chunk1_part0_49.jsonl"
    if not combined.exists() or force:
        combine_files(decompressed, combined)

    kept, dropped = simhash_dedupe([combined], final_path, casefold=True)
    print(f"[slimpajama] kept {kept}, dropped {dropped}")
    return final_path


def clean_existing_jsonl(input_path: Path, output_path: Path, force: bool) -> Path:
    if output_path.exists() and not force:
        print(f"[clean] skip existing {output_path}")
        return output_path
    kept, dropped = simhash_dedupe([input_path], output_path)
    print(f"[clean] {input_path.name} kept {kept} dropped {dropped}")
    return output_path


def write_manifests(final_dir: Path, pretrain_paths: Dict[str, Path], token_targets: Dict[str, float], configs_dir: Path, sft_path: Path) -> None:
    total = sum(token_targets.values())
    datasets = []
    for name, path in pretrain_paths.items():
        datasets.append({
            "name": name,
            "path": str(path),
            "target_tokens": token_targets[name],
            "weight": round(token_targets[name] / total, 6)
        })
    pretrain_manifest = {
        "type": "pretrain",
        "total_target_tokens": total,
        "datasets": datasets
    }
    configs_dir.mkdir(parents=True, exist_ok=True)
    (configs_dir / "pretrain_manifest.json").write_text(json.dumps(pretrain_manifest, ensure_ascii=False, indent=2))

    sft_manifest = {
        "type": "sft",
        "datasets": [
            {
                "name": "sft_clean",
                "path": str(sft_path),
                "num_records": sum(1 for _ in sft_path.open("r", encoding="utf-8"))
            }
        ]
    }
    (configs_dir / "sft_manifest.json").write_text(json.dumps(sft_manifest, ensure_ascii=False, indent=2))

    datasets_manifest = {
        "generated_at": __import__('datetime').datetime.utcnow().isoformat() + 'Z',
        "datasets": [
            {
                "file": str(path),
                "source": name,
                "license": "TBD",
                "language": "zh" if "slimpajama" not in name else "en",
                "description": "auto-prepared corpus",
                "num_records": sum(1 for _ in path.open("r", encoding="utf-8")),
                "size_bytes": path.stat().st_size
            }
            for name, path in pretrain_paths.items()
        ] + [
            {
                "file": str(sft_path),
                "source": "sft_clean",
                "license": "unspecified",
                "language": "zh/en mix",
                "description": "cleaned SFT dataset",
                "num_records": sum(1 for _ in sft_path.open("r", encoding="utf-8")),
                "size_bytes": sft_path.stat().st_size
            }
        ]
    }
    (final_dir / "datasets_manifest.json").write_text(json.dumps(datasets_manifest, ensure_ascii=False, indent=2))

# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare MiniGPT training data")
    parser.add_argument("--workers", type=int, default=max(2, multiprocessing.cpu_count() // 2), help="Parallel workers for processing")
    parser.add_argument("--download-workers", type=int, default=8, help="Parallel downloads")
    parser.add_argument("--force-download", action="store_true", help="Re-download existing files")
    parser.add_argument("--output-dir", type=Path, default=Path("data/final"))
    parser.add_argument("--download-dir", type=Path, default=Path("data/raw_cache"))
    parser.add_argument("--tmp-dir", type=Path, default=Path("data/tmp"))
    parser.add_argument("--configs-dir", type=Path, default=Path("configs/data"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.tmp_dir.mkdir(parents=True, exist_ok=True)
    args.download_dir.mkdir(parents=True, exist_ok=True)

    print(f"[info] using {args.workers} processing workers")

    wiki_final = prepare_wiki(args.download_dir / "wiki", args.tmp_dir / "wiki", args.output_dir, args.workers, args.download_workers, args.force_download)
    china_final = prepare_chinacorpus(args.download_dir / "chinacorpus", args.tmp_dir / "chinacorpus", args.output_dir, args.workers, args.download_workers, args.force_download)
    slimpajama_final = prepare_slimpajama(args.download_dir / "slimpajama", args.tmp_dir / "slimpajama", args.output_dir, args.workers, args.download_workers, args.force_download)

    pretrain_hq_input = Path("data/pretrain_hq.jsonl")
    if not pretrain_hq_input.exists():
        raise FileNotFoundError("data/pretrain_hq.jsonl not found")
    pretrain_hq_final = clean_existing_jsonl(pretrain_hq_input, args.output_dir / "pretrain_hq.cleaned.jsonl", force=args.force_download)

    sft_input = Path("data/sft_mini_512.jsonl")
    if not sft_input.exists():
        raise FileNotFoundError("data/sft_mini_512.jsonl not found")
    sft_final = clean_existing_jsonl(sft_input, args.output_dir / "sft_mini_512.cleaned.jsonl", force=args.force_download)

    token_targets = {
        "wiki_zh": 0.9e9,
        "chinacorpus": 0.8e9,
        "pretrain_hq": 0.6e9,
        "slimpajama": 2.1e9,
    }
    pretrain_paths = {
        "wiki_zh": wiki_final,
        "chinacorpus": china_final,
        "pretrain_hq": pretrain_hq_final,
        "slimpajama": slimpajama_final,
    }
    write_manifests(args.output_dir, pretrain_paths, token_targets, args.configs_dir, sft_final)

    print("[done] data preparation complete")


if __name__ == "__main__":
    main()
