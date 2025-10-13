#!/usr/bin/env python3
"""Near-duplicate filtering for JSONL corpora via SimHash + LSH.

The tool streams an input JSONL file (expects a top-level "text" field by default)
, computes a 64-bit SimHash fingerprint for each entry, and removes samples whose
fingerprints fall within a Hamming distance threshold of previously written
examples.  Locality-Sensitive Hashing (LSH) on bands of the fingerprint reduces
pairwise comparisons so the script scales to millions of records.

Example:
    python scripts/dedupe_simhash.py \
        --input data/processed/wiki_zh_full.cleaned.jsonl \
        --output data/processed/wiki_zh_full.simdedup.jsonl \
        --text-field text --bands 4 --bits-per-band 16 --hamming-threshold 3

Parameters worth tuning:
- bands/bits-per-band: control the LSH granularity. `bands * bits_per_band` must
  equal 64. Fewer bands -> larger buckets (more comparisons); more bands -> fewer
  candidates but potentially higher miss-rate.
- hamming-threshold: maximum allowed distance (0 = exact match). Values between
  2 and 4 are typical for short textual duplicates.
- max-bucket-size: caps the number of fingerprints stored per bucket to bound
  memory usage on extremely large corpora. Older entries are kept while newer
  ones silently skip the overflowed bucket.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SimHash-based near-duplicate filter")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--bands", type=int, default=4, help="Number of LSH bands (must divide 64)")
    parser.add_argument("--bits-per-band", type=int, default=16, help="Bits per band; bands * bits-per-band = 64")
    parser.add_argument("--hamming-threshold", type=int, default=3)
    parser.add_argument("--max-bucket-size", type=int, default=64, help="Cap fingerprints stored per bucket (0 disables cap)")
    parser.add_argument("--min-length", type=int, default=10, help="Minimum text length to consider (shorter samples pass without dedupe)")
    parser.add_argument("--casefold", action="store_true", help="Lowercase text before fingerprinting")
    parser.add_argument("--state", type=Path, default=None, help="Optional pickle file to persist LSH buckets across shards")
    return parser.parse_args()


def tokenize(text: str) -> Iterable[str]:
    """Generate features for SimHash.

    Uses a mix of character bigrams/trigrams and whitespace tokens to better
    capture near duplicates in Chinese as well as spaced languages.
    """
    if not text:
        return []
    features: List[str] = []
    length = len(text)
    max_features = 32
    stride = max(1, length // max_features)
    for n in (2, 3):
        for i in range(0, length - n + 1, stride):
            features.append(text[i : i + n])
            if len(features) >= max_features:
                break
        if len(features) >= max_features:
            break
    for token in text.split():
        if token:
            features.append(token)
    return features


def feature_hash(feature: str) -> int:
    digest = hashlib.md5(feature.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def simhash(features: Iterable[str]) -> int:
    weights = [0] * 64
    for feat in features:
        h = feature_hash(feat)
        for bit in range(64):
            if h & (1 << bit):
                weights[bit] += 1
            else:
                weights[bit] -= 1
    fingerprint = 0
    for bit, weight in enumerate(weights):
        if weight >= 0:
            fingerprint |= 1 << bit
    return fingerprint


def hamming_distance(a: int, b: int) -> int:
    x = a ^ b
    count = 0
    while x:
        x &= x - 1
        count += 1
    return count


def fingerprint(text: str, casefold: bool) -> int:
    if casefold:
        text = text.lower()
    feats = tokenize(text)
    if not feats:
        return 0
    return simhash(feats)


def ensure_config(bands: int, bits_per_band: int) -> None:
    if bands * bits_per_band != 64:
        raise ValueError("bands * bits_per_band must equal 64")


def band_keys(fp: int, bands: int, bits_per_band: int) -> Iterable[int]:
    mask = (1 << bits_per_band) - 1
    for i in range(bands):
        shift = i * bits_per_band
        yield (fp >> shift) & mask


def dedupe(args: argparse.Namespace) -> Tuple[int, int, int]:
    ensure_config(args.bands, args.bits_per_band)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # bucket -> list of fingerprints
    band_maps: List[Dict[int, List[int]]] = [defaultdict(list) for _ in range(args.bands)]
    if args.state and args.state.exists():
        with args.state.open("rb") as state_file:
            saved = pickle.load(state_file)
            band_maps = saved.get("band_maps", band_maps)
            saved_kept = saved.get("kept", 0)
    else:
        saved_kept = 0

    kept = saved_kept
    dropped = 0
    total = 0

    with args.input.open("r", encoding="utf-8") as src, args.output.open(
        "w", encoding="utf-8"
    ) as dst:
        for line in src:
            raw = line.strip()
            if not raw:
                continue
            total += 1
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError:
                dropped += 1
                continue
            text = obj.get(args.text_field, "")
            if not isinstance(text, str):
                dropped += 1
                continue
            if len(text) < args.min_length:
                dst.write(raw + "\n")
                kept += 1
                continue
            fp = fingerprint(text, args.casefold)
            is_dup = False
            for idx, key in enumerate(band_keys(fp, args.bands, args.bits_per_band)):
                bucket = band_maps[idx][key]
                for existing in bucket:
                    if hamming_distance(fp, existing) <= args.hamming_threshold:
                        is_dup = True
                        break
                if is_dup:
                    break
            if is_dup:
                dropped += 1
                continue
            dst.write(raw + "\n")
            kept += 1
            for idx, key in enumerate(band_keys(fp, args.bands, args.bits_per_band)):
                bucket = band_maps[idx][key]
                if args.max_bucket_size and len(bucket) >= args.max_bucket_size:
                    continue
                bucket.append(fp)
    if args.state:
        with args.state.open("wb") as state_file:
            pickle.dump({"band_maps": band_maps, "kept": kept}, state_file)
    return total, kept, dropped


def main() -> None:
    args = parse_args()
    total, kept, dropped = dedupe(args)
    stats = {
        "total": total,
        "kept": kept,
        "dropped": dropped,
        "drop_rate": round(dropped / total, 6) if total else 0.0,
    }
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
