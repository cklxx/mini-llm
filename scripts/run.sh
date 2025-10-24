#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: scripts/run.sh [--smoke-test]

Options:
  --smoke-test    Run a fast CPU-only smoke test with tiny dataset slices and
                  a handful of optimizer steps per stage. Useful for local
                  verification that the end-to-end pipeline works without
                  requiring a GPU.
USAGE
}

SMOKE_TEST=0

while (($#)); do
  case "$1" in
    --smoke-test)
      SMOKE_TEST=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT_DIR"

export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"

VENV_DIR=${VENV_DIR:-.venv}

# Detect cloud environment and set defaults accordingly
# Check if running in OpenBayes environment
if [ -d "/openbayes/home" ] 2>/dev/null; then
  IS_CLOUD=1
  TF_DIR=${TF_DIR:-/openbayes/home/tf_dir}
  PRETRAIN_DEFAULT_ROOT=${DATA_ROOT:-/openbayes/input/input0}
else
  IS_CLOUD=0
  TF_DIR=${TF_DIR:-./tf_dir}
  PRETRAIN_DEFAULT_ROOT=${DATA_ROOT:-./data}
  echo "[env] Running in local environment (TF_DIR: $TF_DIR)"
fi

OUT_DIR=${OUT_DIR:-out}
DATA_DIR=${DATA_DIR:-data/processed}
RESULTS_FILE=${RESULTS_FILE:-"$TF_DIR/eval_results.jsonl"}

USE_UV=0

if [ -d "$VENV_DIR" ]; then
  echo "[env] Using existing virtual environment at $VENV_DIR"
else
  echo "[env] No virtual environment found at $VENV_DIR; bootstrapping with uv"
  USE_UV=1
  if ! command -v uv >/dev/null 2>&1; then
    echo "[env] Installing uv toolchain"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$PATH"
  fi
  uv venv "$VENV_DIR"
fi

if [ ! -x "$VENV_DIR/bin/python" ]; then
  echo "Python interpreter not found in $VENV_DIR" >&2
  exit 1
fi

# shellcheck disable=SC1090
source "$VENV_DIR/bin/activate"
trap 'deactivate >/dev/null 2>&1 || true' EXIT

if [ "$USE_UV" -eq 1 ]; then
  if [ ! -f "$VENV_DIR/.deps_installed" ]; then
    echo "[env] Syncing Python dependencies via uv"
    uv pip sync requirements.txt
    touch "$VENV_DIR/.deps_installed"
  else
    echo "[env] Dependencies already synced via uv, skipping"
  fi
else
  if [ ! -f "$VENV_DIR/.deps_installed" ]; then
    echo "[env] Installing Python dependencies via pip"
    python -m pip install --upgrade pip
    python -m pip install -r requirements.txt
    touch "$VENV_DIR/.deps_installed"
  else
    echo "[env] Dependencies already installed, skipping pip install"
  fi
fi

mkdir -p "$TF_DIR" || { echo "[warn] Could not create $TF_DIR directory"; }
mkdir -p "$OUT_DIR" || { echo "[error] Could not create $OUT_DIR directory" >&2; exit 1; }
mkdir -p "$DATA_DIR" || { echo "[error] Could not create $DATA_DIR directory" >&2; exit 1; }

# PRETRAIN_DEFAULT_ROOT was already set in cloud environment detection above
if [ -z "${PRETRAIN_DEFAULT_ROOT:-}" ]; then
  PRETRAIN_DEFAULT_ROOT=${DATA_ROOT:-./data}
fi
PRETRAIN_JSON=${PRETRAIN_JSON:-"$PRETRAIN_DEFAULT_ROOT/pretrain_hq.jsonl"}
SFT_JSON=${SFT_JSON:-"$PRETRAIN_DEFAULT_ROOT/sft_mini_512.jsonl"}
DPO_JSON=${DPO_JSON:-"$PRETRAIN_DEFAULT_ROOT/dpo_pairs.jsonl"}

if [ ! -s "$DPO_JSON" ]; then
  ALT_DPO="$PRETRAIN_DEFAULT_ROOT/dpo.jsonl"
  if [ -s "$ALT_DPO" ]; then
    DPO_JSON="$ALT_DPO"
  fi
fi

IDENTITY_DATA="data/chinese/identity_conversations.jsonl"
if [ -f "$IDENTITY_DATA" ]; then
  echo "[data] Using identity data at $IDENTITY_DATA"
else
  echo "[data] Identity data missing at $IDENTITY_DATA" >&2
  exit 1
fi

NEED_LOCAL_DATA=0

if [ ! -s "$PRETRAIN_JSON" ]; then
  echo "[data] Falling back to processed pretrain data"
  PRETRAIN_JSON="$DATA_DIR/pretrain_chinese.jsonl"
  NEED_LOCAL_DATA=1
fi

if [ ! -s "$SFT_JSON" ]; then
  echo "[data] Falling back to processed SFT data"
  SFT_JSON="$DATA_DIR/sft_chinese.jsonl"
  NEED_LOCAL_DATA=1
fi

if [ ! -s "$DPO_JSON" ]; then
  echo "[data] Falling back to processed DPO data"
  DPO_JSON="$DATA_DIR/dpo_chinese.jsonl"
  NEED_LOCAL_DATA=1
fi

if [ "$NEED_LOCAL_DATA" -eq 1 ]; then
  echo "[data] Building Chinese data mixtures"
  python scripts/build_chinese_mix.py --output-dir "$DATA_DIR"
fi

for path in "$PRETRAIN_JSON" "$SFT_JSON" "$DPO_JSON"; do
  if [ ! -s "$path" ]; then
    echo "[data] Required dataset not found: $path" >&2
    exit 1
  fi
done

if [ "$SMOKE_TEST" -eq 1 ]; then
  echo "[smoke] Enabling CPU smoke test mode"
  SMOKE_DIR="$OUT_DIR/smoke_data"
  mkdir -p "$SMOKE_DIR"

  SMOKE_PRETRAIN_LIMIT=${SMOKE_PRETRAIN_LIMIT:-64}
  SMOKE_SFT_LIMIT=${SMOKE_SFT_LIMIT:-16}
  SMOKE_DPO_LIMIT=${SMOKE_DPO_LIMIT:-8}

  python scripts/create_smoke_subset.py --input "$PRETRAIN_JSON" --output "$SMOKE_DIR/pretrain.jsonl" --limit "$SMOKE_PRETRAIN_LIMIT"
  PRETRAIN_JSON="$SMOKE_DIR/pretrain.jsonl"

  python scripts/create_smoke_subset.py --input "$SFT_JSON" --output "$SMOKE_DIR/sft.jsonl" --limit "$SMOKE_SFT_LIMIT"
  SFT_JSON="$SMOKE_DIR/sft.jsonl"

  python scripts/create_smoke_subset.py --input "$DPO_JSON" --output "$SMOKE_DIR/dpo.jsonl" --limit "$SMOKE_DPO_LIMIT"
  DPO_JSON="$SMOKE_DIR/dpo.jsonl"
fi

EXTRA_PRETRAIN_ARGS=()
EXTRA_SFT_ARGS=()
EXTRA_DPO_ARGS=()

if [ -n "${PRETRAIN_ARGS:-}" ]; then
  read -r -a EXTRA_PRETRAIN_ARGS <<<"${PRETRAIN_ARGS}"
fi
if [ -n "${SFT_ARGS:-}" ]; then
  read -r -a EXTRA_SFT_ARGS <<<"${SFT_ARGS}"
fi
if [ -n "${DPO_ARGS:-}" ]; then
  read -r -a EXTRA_DPO_ARGS <<<"${DPO_ARGS}"
fi

MODEL_HIDDEN_SIZE=${MODEL_HIDDEN_SIZE:-512}
MODEL_NUM_LAYERS=${MODEL_NUM_LAYERS:-8}
USE_MOE=${USE_MOE:-false}

CHECKPOINT_PRETRAIN="$OUT_DIR/pretrain_${MODEL_HIDDEN_SIZE}.pth"
CHECKPOINT_SFT="$OUT_DIR/full_sft_${MODEL_HIDDEN_SIZE}.pth"
CHECKPOINT_DPO="$OUT_DIR/rlhf_${MODEL_HIDDEN_SIZE}.pth"

# Auto-load pretrained checkpoints from /openbayes/home/out or environment
PRETRAINED_PATH=""
find_pretrained_checkpoint() {
  local stage=$1
  local model_size=$2
  local moe_suffix=""

  if [ "$USE_MOE" = "true" ]; then
    moe_suffix="_moe"
  fi

  # Check environment variable first
  if [ -n "${MINILLM_PRETRAINED_PATH:-}" ] && [ -f "$MINILLM_PRETRAINED_PATH" ]; then
    echo "$MINILLM_PRETRAINED_PATH"
    return 0
  fi

  # Check /openbayes/home/out (OpenBayes environment)
  local remote_path="/openbayes/home/out/${stage}_${model_size}${moe_suffix}.pth"
  if [ -f "$remote_path" ]; then
    echo "$remote_path"
    return 0
  fi

  # Check local out directory
  local local_path="$OUT_DIR/${stage}_${model_size}${moe_suffix}.pth"
  if [ -f "$local_path" ]; then
    echo "$local_path"
    return 0
  fi

  return 1
}

# Initialize PRETRAINED_PATH if available
PRETRAINED_PATH=$(find_pretrained_checkpoint "pretrain" "$MODEL_HIDDEN_SIZE" 2>/dev/null || true)
if [ -n "$PRETRAINED_PATH" ]; then
  echo "[checkpoint] Found pretrained model at: $PRETRAINED_PATH"
  EXTRA_PRETRAIN_ARGS+=(--pretrained_path "$PRETRAINED_PATH")
fi

TB_PRETRAIN_DIR="$TF_DIR/pretrain"
TB_SFT_DIR="$TF_DIR/sft"
TB_DPO_DIR="$TF_DIR/dpo"
TB_EVAL_DIR="$TF_DIR/eval"

mkdir -p "$TB_PRETRAIN_DIR" "$TB_SFT_DIR" "$TB_DPO_DIR" "$TB_EVAL_DIR"

EVAL_CMD_BASE=(python scripts/evaluate_stage.py --hidden-size "$MODEL_HIDDEN_SIZE" --num-hidden-layers "$MODEL_NUM_LAYERS" --results-file "$RESULTS_FILE")

PRETRAIN_EVAL_MAX_SAMPLES=128
PRETRAIN_EVAL_BATCH=8
SFT_EVAL_MAX_SAMPLES=128
SFT_EVAL_BATCH=4
DPO_EVAL_MAX_SAMPLES=64
DPO_EVAL_BATCH=2

if [ "$SMOKE_TEST" -eq 1 ]; then
  EXTRA_PRETRAIN_ARGS+=(--device cpu --dtype float32 --batch_size "${SMOKE_PRETRAIN_BATCH:-2}" --max_steps "${SMOKE_PRETRAIN_STEPS:-4}" --num_workers 0 --log_interval 1 --save_interval "${SMOKE_PRETRAIN_STEPS:-4}")
  EXTRA_SFT_ARGS+=(--device cpu --dtype float32 --batch_size "${SMOKE_SFT_BATCH:-2}" --max_steps "${SMOKE_SFT_STEPS:-4}" --num_workers 0 --log_interval 1 --save_interval "${SMOKE_SFT_STEPS:-4}")
  EXTRA_DPO_ARGS+=(--device cpu --dtype float32 --batch_size "${SMOKE_DPO_BATCH:-2}" --max_steps "${SMOKE_DPO_STEPS:-4}" --num_workers 0 --log_interval 1 --save_interval "${SMOKE_DPO_STEPS:-4}")

  PRETRAIN_EVAL_MAX_SAMPLES=${SMOKE_PRETRAIN_EVAL_SAMPLES:-8}
  PRETRAIN_EVAL_BATCH=${SMOKE_PRETRAIN_EVAL_BATCH:-2}
  SFT_EVAL_MAX_SAMPLES=${SMOKE_SFT_EVAL_SAMPLES:-8}
  SFT_EVAL_BATCH=${SMOKE_SFT_EVAL_BATCH:-2}
  DPO_EVAL_MAX_SAMPLES=${SMOKE_DPO_EVAL_SAMPLES:-4}
  DPO_EVAL_BATCH=${SMOKE_DPO_EVAL_BATCH:-1}

  EVAL_CMD_BASE+=(--device cpu)
fi

echo "[stage] Starting pretrain (2 epochs)"
python trainer/train_pretrain.py --data_path "$PRETRAIN_JSON" --hidden_size "$MODEL_HIDDEN_SIZE" --num_hidden_layers "$MODEL_NUM_LAYERS" --epochs 2 --out_dir "$OUT_DIR" --tensorboard_dir "$TB_PRETRAIN_DIR" "${EXTRA_PRETRAIN_ARGS[@]}"

if [ -f "$CHECKPOINT_PRETRAIN" ]; then
  echo "[eval] Pretrain evaluation"
  "${EVAL_CMD_BASE[@]}" --stage pretrain --checkpoint "$CHECKPOINT_PRETRAIN" --data-path "$PRETRAIN_JSON" --max-seq-len 512 --max-samples "$PRETRAIN_EVAL_MAX_SAMPLES" --batch-size "$PRETRAIN_EVAL_BATCH" --tensorboard-dir "$TB_EVAL_DIR/pretrain"
else
  echo "[warn] Pretrain checkpoint not found at $CHECKPOINT_PRETRAIN" >&2
fi

echo "[stage] Starting SFT"
# Auto-load SFT pretrained checkpoint
SFT_PRETRAINED_PATH=$(find_pretrained_checkpoint "full_sft" "$MODEL_HIDDEN_SIZE" 2>/dev/null || true)
SFT_ARGS_WITH_PRETRAIN=("${EXTRA_SFT_ARGS[@]}")
if [ -z "$SFT_PRETRAINED_PATH" ]; then
  # If no full_sft checkpoint, try pretrain checkpoint
  SFT_PRETRAINED_PATH=$(find_pretrained_checkpoint "pretrain" "$MODEL_HIDDEN_SIZE" 2>/dev/null || true)
fi
if [ -n "$SFT_PRETRAINED_PATH" ]; then
  echo "[checkpoint] Using pretrained model for SFT: $SFT_PRETRAINED_PATH"
  SFT_ARGS_WITH_PRETRAIN+=(--pretrained_path "$SFT_PRETRAINED_PATH")
fi
python trainer/train_full_sft.py --data_path "$SFT_JSON" --hidden_size "$MODEL_HIDDEN_SIZE" --num_hidden_layers "$MODEL_NUM_LAYERS" --out_dir "$OUT_DIR" --tensorboard_dir "$TB_SFT_DIR" "${SFT_ARGS_WITH_PRETRAIN[@]}"

if [ -f "$CHECKPOINT_SFT" ]; then
  echo "[eval] SFT evaluation"
  "${EVAL_CMD_BASE[@]}" --stage sft --checkpoint "$CHECKPOINT_SFT" --data-path "$SFT_JSON" --max-seq-len 512 --max-samples "$SFT_EVAL_MAX_SAMPLES" --batch-size "$SFT_EVAL_BATCH" --tensorboard-dir "$TB_EVAL_DIR/sft"
else
  echo "[warn] SFT checkpoint not found at $CHECKPOINT_SFT" >&2
fi

echo "[stage] Starting DPO"
# Auto-load DPO pretrained checkpoint
DPO_PRETRAINED_PATH=$(find_pretrained_checkpoint "full_sft" "$MODEL_HIDDEN_SIZE" 2>/dev/null || true)
DPO_ARGS_WITH_PRETRAIN=("${EXTRA_DPO_ARGS[@]}")
if [ -z "$DPO_PRETRAINED_PATH" ]; then
  # If no full_sft checkpoint, try rlhf checkpoint
  DPO_PRETRAINED_PATH=$(find_pretrained_checkpoint "rlhf" "$MODEL_HIDDEN_SIZE" 2>/dev/null || true)
fi
if [ -z "$DPO_PRETRAINED_PATH" ]; then
  # If no rlhf checkpoint, try pretrain checkpoint
  DPO_PRETRAINED_PATH=$(find_pretrained_checkpoint "pretrain" "$MODEL_HIDDEN_SIZE" 2>/dev/null || true)
fi
if [ -n "$DPO_PRETRAINED_PATH" ]; then
  echo "[checkpoint] Using pretrained model for DPO: $DPO_PRETRAINED_PATH"
  DPO_ARGS_WITH_PRETRAIN+=(--pretrained_path "$DPO_PRETRAINED_PATH")
fi
python trainer/train_dpo.py --data_path "$DPO_JSON" --hidden_size "$MODEL_HIDDEN_SIZE" --num_hidden_layers "$MODEL_NUM_LAYERS" --out_dir "$OUT_DIR" --tensorboard_dir "$TB_DPO_DIR" "${DPO_ARGS_WITH_PRETRAIN[@]}"

if [ -f "$CHECKPOINT_DPO" ]; then
  echo "[eval] DPO evaluation"
  "${EVAL_CMD_BASE[@]}" --stage dpo --checkpoint "$CHECKPOINT_DPO" --data-path "$DPO_JSON" --max-seq-len 1024 --max-samples "$DPO_EVAL_MAX_SAMPLES" --batch-size "$DPO_EVAL_BATCH" --tensorboard-dir "$TB_EVAL_DIR/dpo"
else
  echo "[warn] DPO checkpoint not found at $CHECKPOINT_DPO" >&2
fi

echo "[done] Training pipeline completed. Check $OUT_DIR for checkpoints and $RESULTS_FILE for evaluation logs."
