#!/bin/bash
#SBATCH --job-name=cap-eval
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --environment=vsch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --exclusive
#SBATCH --time=03:45:00
#SBATCH --signal=B:TERM@120
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#
# Held-out eval for all 78 capacity-sweep checkpoints (36 MLP + 42 XGB).
# Uses features.parquet with eval split; builds from ~/autotuner/ if missing.
# Node-local /dev/shm cache only — does NOT use $SCRATCH.
#
# Submit:
#   sbatch benchmarks/eval/run_capacity_eval.sh
#
# Resume (skip runs that already have eval_regret.csv + metrics.eval):
#   sbatch benchmarks/eval/run_capacity_eval.sh
#
# Force re-eval everything:
#   FORCE=1 sbatch benchmarks/eval/run_capacity_eval.sh
#
# Env overrides:
#   CAPACITY_DIR=artifacts/analysis/capacity
#   FEATURES=artifacts/analysis/paper/features.parquet
#   DB=~/autotuner/autotuner_bf16_eval.db
#   EVAL_TAG=bf16_eval
#   EVAL_MIN_CONFIGS=8000
#   NUM_GPUS=4
#   XGB_WORKERS=64

set -euo pipefail
_STUDY_JOB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=repo_paths.sh
source "${_STUDY_JOB_DIR}/../common/repo_paths.sh"
repo_paths_export "${CKS_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$REPO_ROOT"
export CKS_ROOT="$REPO_ROOT"
export PYTHONUNBUFFERED=1

VENV="${VENV:-${CKS_ROOT}/.venv}"
if [ ! -x "$VENV/bin/python" ]; then
    echo "Creating venv at $VENV ..."
    python3 -m venv "$VENV"
fi
PYTHON="$VENV/bin/python"
PIP="$VENV/bin/pip"

DEPS_MARKER="$VENV/.cap_eval_deps_ok"
if [ ! -f "$DEPS_MARKER" ]; then
    echo "Installing eval dependencies into $VENV ..."
    "$PIP" install -q --upgrade pip
    "$PIP" install -q \
        -r "${CKS_ROOT}/requirements.txt" \
        xgboost scikit-learn pyarrow pandas
    touch "$DEPS_MARKER"
fi

"$PYTHON" -c "import xgboost, torch, sklearn, pyarrow, pandas" \
    || { echo "ERROR: venv missing required packages" >&2; exit 1; }

FEATURES="${FEATURES:-${PAPER_DIR}/features.parquet}"
DB="${DB:-${HOME}/autotuner/autotuner_bf16_eval.db}"
EVAL_TAG="${EVAL_TAG:-bf16_eval}"
EVAL_MIN_CONFIGS="${EVAL_MIN_CONFIGS:-8000}"
NUM_GPUS="${NUM_GPUS:-4}"
XGB_WORKERS="${XGB_WORKERS:-64}"
SHM="/dev/shm/capacity_eval_${SLURM_JOB_ID:-local}"

mkdir -p "$SHM"

EXTRA_ARGS=()
if [ "${FORCE:-0}" = "1" ]; then
    EXTRA_ARGS+=(--force)
fi

echo "=== capacity eval ==="
echo "CKS_ROOT=$CKS_ROOT"
echo "CAPACITY_DIR=$CAPACITY_DIR"
echo "FEATURES=$FEATURES"
echo "DB=$DB"
echo "EVAL_TAG=$EVAL_TAG  EVAL_MIN_CONFIGS=$EVAL_MIN_CONFIGS"
echo "SHM=$SHM"
echo "NUM_GPUS=$NUM_GPUS  XGB_WORKERS=$XGB_WORKERS"
echo "PYTHON=$PYTHON"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-local}"
echo

"$PYTHON" "${BENCHMARKS}/eval/eval_capacity_checkpoints.py" \
    --capacity-dir "$CAPACITY_DIR" \
    --features "$FEATURES" \
    --db "$DB" \
    --eval-tag "$EVAL_TAG" \
    --eval-min-configs "$EVAL_MIN_CONFIGS" \
    --shm-dir "$SHM" \
    --num-gpus "$NUM_GPUS" \
    --xgb-workers "$XGB_WORKERS" \
    "${EXTRA_ARGS[@]}" \
    "$@"

echo
echo "Done. Per-run: eval_regret.csv + metrics.json[eval] under $CAPACITY_DIR"
echo "Summary: $CAPACITY_DIR/capacity_eval_summary.csv"
