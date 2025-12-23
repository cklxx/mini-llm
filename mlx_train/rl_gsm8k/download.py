from __future__ import annotations

import argparse
import os
import shlex
import shutil
import subprocess
from dataclasses import dataclass
from typing import List, Optional


@dataclass(frozen=True)
class DownloadSpec:
    repo_id: str
    repo_type: str
    local_dir: str
    revision: Optional[str] = None


def build_huggingface_cli_command(spec: DownloadSpec) -> List[str]:
    cmd = [
        "huggingface-cli",
        "download",
        "--repo-type",
        spec.repo_type,
        spec.repo_id,
        "--local-dir",
        spec.local_dir,
    ]
    if spec.revision:
        cmd.extend(["--revision", spec.revision])
    return cmd


def ensure_huggingface_cli() -> str:
    exe = shutil.which("huggingface-cli")
    if not exe:
        raise FileNotFoundError(
            "Missing `huggingface-cli` in PATH. Install: `pip install -U huggingface_hub`"
        )
    return exe


def download_with_huggingface_cli(spec: DownloadSpec) -> None:
    ensure_huggingface_cli()
    os.makedirs(spec.local_dir, exist_ok=True)
    cmd = build_huggingface_cli_command(spec)
    print(f"[hf] {' '.join(shlex.quote(c) for c in cmd)}", flush=True)
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download GSM8K dataset repo via `huggingface-cli download`."
    )
    parser.add_argument(
        "--repo_id", type=str, default="zhuzilin/gsm8k", help="HF dataset repo id"
    )
    parser.add_argument("--repo_type", type=str, default="dataset")
    parser.add_argument(
        "--local_dir", type=str, default="/root/gsm8k", help="Local output directory"
    )
    parser.add_argument("--revision", type=str, default=None)
    args = parser.parse_args()

    download_with_huggingface_cli(
        DownloadSpec(
            repo_id=str(args.repo_id),
            repo_type=str(args.repo_type),
            local_dir=str(args.local_dir),
            revision=None if args.revision in (None, "") else str(args.revision),
        )
    )


if __name__ == "__main__":
    main()

