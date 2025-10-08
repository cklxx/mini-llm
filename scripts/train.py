#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""MiniGPT training entrypoint that delegates to the pipeline package."""

import os
import sys

project_root = os.path.dirname(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.append(project_root)
src_dir = os.path.join(project_root, "src")
if src_dir not in sys.path:
    sys.path.append(src_dir)

from training.pipeline.cli import run_cli


def main() -> None:
    run_cli()


if __name__ == "__main__":
    main()
