#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT_DIR"

echo "[1/7] Ensuring uv toolchain..."
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.cargo/bin:$PATH"
fi

if [ ! -d ".venv" ]; then
  uv venv
fi

source .venv/bin/activate

echo "[2/7] Syncing Python dependencies"
uv pip sync requirements.txt

ENABLE_RUSTBPE=${ENABLE_RUSTBPE:-0}

if [[ "$ENABLE_RUSTBPE" == "1" ]]; then
  echo "[3/7] Building rustbpe extension"
  uv run maturin develop --release --manifest-path rustbpe/Cargo.toml
else
  echo "[3/7] Skipping RustBPE build (set ENABLE_RUSTBPE=1 to enable)"
fi

DATA_DIR=${ROOT_DIR}/data/processed
TF_DIR=${TF_DIR:-/openbayes/home/tf_dir}
TOKENIZER_DIR=${ROOT_DIR}/out/rustbpe_tokenizer
mkdir -p "$DATA_DIR"
mkdir -p "$TF_DIR"

DATA_ROOT=${DATA_ROOT:-/openbayes/input/input0}
PRETRAIN_JSON=${PRETRAIN_JSON:-"$DATA_ROOT/pretrain_hq.jsonl"}
SFT_JSON=${SFT_JSON:-"$DATA_ROOT/sft_mini_512.jsonl"}
DPO_JSON=${DPO_JSON:-"$DATA_ROOT/dpo_pairs.jsonl"}

if [ ! -s "$DPO_JSON" ]; then
  ALT_DPO="$DATA_ROOT/dpo.jsonl"
  if [ -s "$ALT_DPO" ]; then
    DPO_JSON="$ALT_DPO"
  fi
fi

NEED_LOCAL_DATA=0

if [ ! -s "$PRETRAIN_JSON" ]; then
  PRETRAIN_JSON="$DATA_DIR/pretrain_chinese.jsonl"
  NEED_LOCAL_DATA=1
fi

if [ ! -s "$SFT_JSON" ]; then
  SFT_JSON="$DATA_DIR/sft_chinese.jsonl"
  NEED_LOCAL_DATA=1
fi

if [ ! -s "$DPO_JSON" ]; then
  DPO_JSON="$DATA_DIR/dpo_chinese.jsonl"
  NEED_LOCAL_DATA=1
fi

if [ "$NEED_LOCAL_DATA" -eq 1 ]; then
  echo "[4/7] Preparing Chinese data mixtures"
  uv run python scripts/build_chinese_mix.py --output-dir "$DATA_DIR"
else
  echo "[4/7] Using datasets from $DATA_ROOT"
fi

TB_PRETRAIN_DIR="$TF_DIR/pretrain"
TB_SFT_DIR="$TF_DIR/sft"
TB_DPO_DIR="$TF_DIR/dpo"
mkdir -p "$TB_PRETRAIN_DIR" "$TB_SFT_DIR" "$TB_DPO_DIR"

if [ ! -s "$PRETRAIN_JSON" ]; then
  echo "Pretrain dataset not found at $PRETRAIN_JSON" >&2
  exit 1
fi

if [[ "$ENABLE_RUSTBPE" == "1" ]]; then
  mkdir -p "$TOKENIZER_DIR"
  echo "[5/7] Training RustBPE tokenizer"
  uv run python scripts/train_rustbpe_tokenizer.py --format pretrain --output "$TOKENIZER_DIR" "$PRETRAIN_JSON"

  EMB_DIR=${DATA_DIR}/embeddings
  mkdir -p "$EMB_DIR"

  echo "[6/7] Exporting token tensors"
  uv run python scripts/export_embeddings.py pretrain --input "$PRETRAIN_JSON" --output "$EMB_DIR/pretrain.pt" --tokenizer-dir "$TOKENIZER_DIR"
  if [ -s "$SFT_JSON" ]; then
    uv run python scripts/export_embeddings.py sft --input "$SFT_JSON" --output "$EMB_DIR/sft.pt" --tokenizer-dir "$TOKENIZER_DIR"
  fi
  if [ -s "$DPO_JSON" ]; then
    uv run python scripts/export_embeddings.py dpo --input "$DPO_JSON" --output "$EMB_DIR/dpo.pt" --tokenizer-dir "$TOKENIZER_DIR"
  fi
else
  echo "[5/7] Skipping tokenizer training (ENABLE_RUSTBPE=1 to enable)"
  echo "[6/7] Skipping tensor export (ENABLE_RUSTBPE=1 to enable)"
fi

read -r -a EXTRA_PRETRAIN <<< "${EXTRA_PRETRAIN_ARGS:-}"
read -r -a EXTRA_SFT <<< "${EXTRA_SFT_ARGS:-}"
read -r -a EXTRA_DPO <<< "${EXTRA_DPO_ARGS:-}"

echo "[7/7] Launching training stages"
uv run python trainer/train_pretrain.py --data_path "$PRETRAIN_JSON" --hidden_size 512 --num_hidden_layers 8 --tensorboard_dir "$TB_PRETRAIN_DIR" "${EXTRA_PRETRAIN[@]}"
if [ -s "$SFT_JSON" ]; then
  uv run python trainer/train_full_sft.py --data_path "$SFT_JSON" --hidden_size 512 --num_hidden_layers 8 --tensorboard_dir "$TB_SFT_DIR" "${EXTRA_SFT[@]}"
fi
if [ -s "$DPO_JSON" ]; then
  uv run python trainer/train_dpo.py --data_path "$DPO_JSON" --hidden_size 512 --num_hidden_layers 8 --tensorboard_dir "$TB_DPO_DIR" "${EXTRA_DPO[@]}"
fi

echo "Pipeline completed. Check the out/ directory for checkpoints."
