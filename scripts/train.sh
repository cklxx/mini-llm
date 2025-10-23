#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
pushd "${ROOT_DIR}" >/dev/null
trap "popd >/dev/null" EXIT
export PYTHONPATH="${ROOT_DIR}/src:${PYTHONPATH:-}"

DATA_PATH="${DATA_PATH:-${ROOT_DIR}/data/pretrain_hq.jsonl}"
VOCAB_SIZE="${VOCAB_SIZE:-32000}"
TOKENIZER_OUTPUT="${TOKENIZER_OUTPUT:-${ROOT_DIR}/tokenizers/rust_bpe}"
CACHE_DIR="${TOKENIZER_CACHE_DIR:-${ROOT_DIR}/tokenizers}"
MODEL_SIZE="${MODEL_SIZE:-medium}"
TRAIN_MODE="${TRAIN_MODE:-pretrain}"
WORKERS="${WORKERS:-4}"
PREPARE_DATA="${PREPARE_DATA:-1}"
RETRAIN_TOKENIZER="${RETRAIN_TOKENIZER:-0}"
TRAIN_EXTRA_ARGS=(${TRAIN_EXTRA_ARGS:-})

log_step() {
  echo "\n[train] $1"
}

# -----------------------------------------------------------------------------
log_step "Ensuring uv virtual environment"
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
fi

if [ ! -d "${ROOT_DIR}/.venv" ]; then
  uv venv "${ROOT_DIR}/.venv"
fi
uv sync
source "${ROOT_DIR}/.venv/bin/activate"

# -----------------------------------------------------------------------------
log_step "Building rustbpe extension if necessary"
if ! python -c "import rustbpe" >/dev/null 2>&1; then
  uv run maturin develop --manifest-path "${ROOT_DIR}/rustbpe/Cargo.toml" --release
fi

# -----------------------------------------------------------------------------
if [ "${PREPARE_DATA}" -ne 0 ]; then
  log_step "Preparing datasets (workers=${WORKERS})"
  uv run python "${ROOT_DIR}/scripts/prepare_training_data.py" --workers "${WORKERS}"
fi

# -----------------------------------------------------------------------------
log_step "Training RustBPE tokenizer"
TOKENIZER_ARGS=("--data" "${DATA_PATH}" "--vocab-size" "${VOCAB_SIZE}" "--output" "${TOKENIZER_OUTPUT}" "--cache-dir" "${CACHE_DIR}")
if [ "${RETRAIN_TOKENIZER}" -ne 0 ]; then
  TOKENIZER_ARGS+=("--force")
fi
uv run python "${ROOT_DIR}/scripts/train_rust_tokenizer.py" "${TOKENIZER_ARGS[@]}"

# -----------------------------------------------------------------------------
log_step "Launching training pipeline (${TRAIN_MODE}, config=${MODEL_SIZE})"
TRAIN_CMD=(uv run python "${ROOT_DIR}/scripts/train.py" "--mode" "${TRAIN_MODE}" "--config" "${MODEL_SIZE}" "--auto-resume")
TRAIN_CMD+=("--retrain-tokenizer")
if [ ${#TRAIN_EXTRA_ARGS[@]} -gt 0 ]; then
  TRAIN_CMD+=("${TRAIN_EXTRA_ARGS[@]}")
fi
"${TRAIN_CMD[@]}"

log_step "Pipeline complete"
