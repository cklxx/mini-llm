#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
pushd "${ROOT_DIR}" >/dev/null
trap "popd >/dev/null" EXIT
export PYTHONPATH="${ROOT_DIR}/src:${PYTHONPATH:-}"

if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
fi

if [ ! -d "${ROOT_DIR}/.venv" ]; then
  uv venv "${ROOT_DIR}/.venv"
fi
uv sync
source "${ROOT_DIR}/.venv/bin/activate"

if ! python -c "import rustbpe" >/dev/null 2>&1; then
  uv run maturin develop --manifest-path "${ROOT_DIR}/rustbpe/Cargo.toml" --release
fi

ARGS=($@)
AUTO_FLAG=0
for arg in "${ARGS[@]}"; do
  if [ "$arg" = "--auto-resume" ]; then
    AUTO_FLAG=1
    break
  fi
done
if [ ${AUTO_FLAG} -eq 0 ]; then
  ARGS+=("--auto-resume")
fi

DELAY="${RESTART_DELAY:-30}"
ATTEMPT=1
while true; do
  echo "\n[resume] attempt ${ATTEMPT}: python scripts/train.py ${ARGS[*]}"
  set +e
  uv run python "${ROOT_DIR}/scripts/train.py" "${ARGS[@]}"
  STATUS=$?
  set -e
  if [ ${STATUS} -eq 0 ]; then
    echo "[resume] training finished successfully"
    break
  fi
  echo "[resume] training interrupted (exit code ${STATUS}). Restarting in ${DELAY}s..."
  sleep "${DELAY}"
  ATTEMPT=$((ATTEMPT + 1))
done
