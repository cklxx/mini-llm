#!/usr/bin/env bash
set -euo pipefail

# Backward compatible wrapper; use `scripts/stats_mlx.sh` instead.
exec bash "$(dirname "${BASH_SOURCE[0]}")/stats_mlx.sh" "$@"
