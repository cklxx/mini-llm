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

# Unset proxy variables that may interfere with package installation
unset http_proxy
unset https_proxy
unset HTTP_PROXY
unset HTTPS_PROXY
unset no_proxy
unset NO_PROXY

# Use Tsinghua mirror for faster package downloads in China
# Can be overridden by setting PIP_INDEX_URL environment variable
export PIP_INDEX_URL=${PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}

VENV_DIR=${VENV_DIR:-.venv}

# Check Python version compatibility (requires Python 3.9-3.12)
check_python_version() {
  local python_cmd=$1
  if ! command -v "$python_cmd" >/dev/null 2>&1; then
    return 1
  fi

  local py_version=$("$python_cmd" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
  local major=$(echo "$py_version" | cut -d. -f1)
  local minor=$(echo "$py_version" | cut -d. -f2)

  # Check if version is in supported range (3.9-3.12)
  if [ "$major" -eq 3 ] && [ "$minor" -ge 9 ] && [ "$minor" -le 12 ]; then
    echo "$python_cmd"
    return 0
  fi
  return 1
}

# Find compatible Python interpreter
PYTHON_CMD=""
for py_candidate in python3.12 python3.11 python3.10 python3.9 python3 python; do
  if PYTHON_CMD=$(check_python_version "$py_candidate" 2>/dev/null); then
    PY_VERSION=$("$PYTHON_CMD" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
    echo "[env] Using Python $PY_VERSION at $PYTHON_CMD"
    break
  fi
done

if [ -z "$PYTHON_CMD" ]; then
  echo "[error] No compatible Python version found (requires 3.9-3.12)" >&2
  echo "[error] Current Python versions detected:" >&2
  for py_test in python3 python; do
    if command -v "$py_test" >/dev/null 2>&1; then
      "$py_test" --version >&2 || true
    fi
  done
  exit 1
fi

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

# Validate virtual environment thoroughly
validate_venv() {
  local venv_path=$1

  # Check if venv directory exists
  if [ ! -d "$venv_path" ]; then
    return 1
  fi

  # Check if python interpreter exists and is executable
  if [ ! -x "$venv_path/bin/python" ]; then
    return 1
  fi

  # Check if pip is available (critical for dependency installation)
  if ! "$venv_path/bin/python" -m pip --version >/dev/null 2>&1; then
    echo "[env] Virtual environment at $venv_path is missing pip" >&2
    return 1
  fi

  # Check if venv Python version matches our requirements
  local venv_py_version=$("$venv_path/bin/python" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null)
  local major=$(echo "$venv_py_version" | cut -d. -f1)
  local minor=$(echo "$venv_py_version" | cut -d. -f2)

  if [ "$major" -eq 3 ] && [ "$minor" -ge 9 ] && [ "$minor" -le 12 ]; then
    return 0
  else
    echo "[env] Virtual environment Python version $venv_py_version is not compatible (requires 3.9-3.12)" >&2
    return 1
  fi
}

# Check if venv exists and is valid
if validate_venv "$VENV_DIR"; then
  VENV_PY_VERSION=$("$VENV_DIR/bin/python" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
  echo "[env] Using existing virtual environment at $VENV_DIR (Python $VENV_PY_VERSION)"
else
  # Venv doesn't exist or is broken, recreate it
  if [ -d "$VENV_DIR" ]; then
    echo "[env] Virtual environment at $VENV_DIR is broken or incompatible, removing it"
    rm -rf "$VENV_DIR"
  fi

  # Prefer uv for faster environment creation if available
  if command -v uv >/dev/null 2>&1; then
    echo "[env] Attempting to create virtual environment with uv using $PYTHON_CMD"
    # uv venv doesn't include pip by default, so we need to check if it works properly
    if uv venv "$VENV_DIR" --python "$PYTHON_CMD" --seed 2>/dev/null; then
      # Verify pip is available
      if "$VENV_DIR/bin/python" -m pip --version >/dev/null 2>&1; then
        USE_UV=1
        echo "[env] Virtual environment created successfully with uv"
      else
        echo "[env] uv venv missing pip, falling back to venv"
        rm -rf "$VENV_DIR" 2>/dev/null || true
        "$PYTHON_CMD" -m venv "$VENV_DIR"
      fi
    else
      echo "[env] uv failed, falling back to venv"
      rm -rf "$VENV_DIR" 2>/dev/null || true
      "$PYTHON_CMD" -m venv "$VENV_DIR"
    fi
  else
    echo "[env] Creating virtual environment with venv using $PYTHON_CMD"
    "$PYTHON_CMD" -m venv "$VENV_DIR"
  fi
fi

if [ ! -x "$VENV_DIR/bin/python" ]; then
  echo "[error] Python interpreter not found in $VENV_DIR after setup" >&2
  exit 1
fi

# shellcheck disable=SC1090
source "$VENV_DIR/bin/activate"
trap 'deactivate >/dev/null 2>&1 || true' EXIT

# Check if dependencies need to be installed or updated
REQUIREMENTS_HASH=$(shasum -a 256 requirements.txt 2>/dev/null | cut -d' ' -f1 || echo "unknown")
DEPS_MARKER="$VENV_DIR/.deps_installed"
DEPS_HASH_FILE="$VENV_DIR/.deps_hash"

NEED_INSTALL=0
if [ ! -f "$DEPS_MARKER" ]; then
  NEED_INSTALL=1
  echo "[env] No dependency marker found, will install"
elif [ ! -f "$DEPS_HASH_FILE" ]; then
  NEED_INSTALL=1
  echo "[env] No hash file found, will reinstall to ensure consistency"
elif [ "$REQUIREMENTS_HASH" != "$(cat "$DEPS_HASH_FILE" 2>/dev/null)" ]; then
  NEED_INSTALL=1
  echo "[env] requirements.txt has changed, will update dependencies"
else
  echo "[env] Dependencies are up to date, skipping installation"
fi

if [ "$NEED_INSTALL" -eq 1 ]; then
  INSTALL_SUCCESS=0

  if [ "$USE_UV" -eq 1 ] && command -v uv >/dev/null 2>&1; then
    echo "[env] Attempting to sync Python dependencies via uv"
    if uv pip sync requirements.txt 2>/dev/null; then
      INSTALL_SUCCESS=1
      echo "[env] Dependencies synced successfully via uv"
    else
      echo "[env] uv pip sync failed, falling back to pip"
    fi
  fi

  if [ "$INSTALL_SUCCESS" -eq 0 ]; then
    echo "[env] Installing Python dependencies via pip"
    python -m pip install --upgrade pip
    # Use --no-cache-dir to avoid potential corruption and ensure fresh install
    python -m pip install --no-cache-dir -r requirements.txt
    INSTALL_SUCCESS=1
  fi

  if [ "$INSTALL_SUCCESS" -eq 1 ]; then
    # Mark dependencies as installed and save hash
    touch "$DEPS_MARKER"
    echo "$REQUIREMENTS_HASH" > "$DEPS_HASH_FILE"
    echo "[env] Dependencies installed successfully"
  else
    echo "[error] Failed to install dependencies" >&2
    exit 1
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
