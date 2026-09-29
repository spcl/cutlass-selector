#!/bin/bash
#SBATCH --job-name=deepbench-eval
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --environment=vsch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --time=02:00:00
#SBATCH --signal=B:TERM@300
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --cpus-per-task=288
#SBATCH --mem=450G

#
# DeepBench dense GEMM evaluation (151 problems in deepbench_compatibility.csv).
# Mirrors benchmarks/eval/slurm_eval.sh but uses explicit (M,N,K,layout) problems instead of
# shapes.json × 4 layouts.
#
#   sbatch benchmarks/deepbench/slurm_deepbench.sh
#
# Resume compile/bench only:
#   SKIP_PREP=1 EVAL_PREP_DIR=artifacts/eval/out/deepbench_<job_id> PHASE=compile,bench sbatch benchmarks/deepbench/slurm_deepbench.sh
#
# Methods (default: paper baselines via benchmarks/deepbench/propose_methods.py):
#   EVAL_METHODS=mlp_full,xgb_full sbatch benchmarks/deepbench/slurm_deepbench.sh
#
# Capacity sweep (additional MLP widths + XGB depths, seed 42):
#   sbatch benchmarks/deepbench/slurm_deepbench_capacity.sh

set -euo pipefail
: "${CKS_TAG:=deepbench}"
: "${PHASE:=plan,compile,bench}"
: "${SKIP_PREP:=0}"

if [ "${CKS_EVAL_DEBUG:-0}" = "1" ]; then
    : "${DEEPBENCH_LIMIT:=5}"
    : "${CKS_EVAL_FRESH_DB:=1}"
    : "${CKS_EVAL_FRESH_BUILD:=1}"
    : "${CKS_COMPILE_JOBS:=16}"
    : "${CKS_CKPT_INTERVAL:=0}"
fi

if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    source "${SLURM_SUBMIT_DIR}/benchmarks/common/slurm_common.sh"
else
    source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../common/slurm_common.sh"
fi

export CKS_TAG
export CKS_GPUS="${CKS_GPUS:-4}"
export CKS_COMPILE_JOBS="${CKS_COMPILE_JOBS:-64}"
export DB="${DB:-$HOME/autotuner/autotuner_deepbench.db}"
export NVMMH_GPU="${NVMMH_GPU:-H100_SXM}"

slurm_cd_repo
export MODELS_DIR="${MODELS_DIR:-$PAPER_DIR}"
export DEEPBENCH_INCLUDE_CAPACITY="${DEEPBENCH_INCLUDE_CAPACITY:-0}"
export DEEPBENCH_CAPACITY_ONLY="${DEEPBENCH_CAPACITY_ONLY:-0}"
export CAPACITY_SEED="${CAPACITY_SEED:-42}"

VENV="${VENV:-${CKS_ROOT}/.venv}"
if [ ! -x "$VENV/bin/python" ]; then
    echo "Creating venv at $VENV ..."
    python3 -m venv "$VENV"
fi
PIP="$VENV/bin/pip"
PYTHON="$VENV/bin/python"
export PYTHON

DEPS_MARKER="$VENV/.paper_eval_deps_ok"
if [ ! -f "$DEPS_MARKER" ]; then
    echo "Installing eval dependencies into $VENV ..."
    "$PIP" install -q --upgrade pip
    "$PIP" install -q \
        -r "${CKS_ROOT}/requirements.txt" \
        "nvidia-matmul-heuristics==0.1.0.27" \
        xgboost scikit-learn pandas
    touch "$DEPS_MARKER"
fi

"$PYTHON" -c "
import nvMatmulHeuristics
import jinja2, numpy, pandas, sklearn, torch, xgboost
print('venv ok:', '$VENV')
" || { echo "ERROR: venv missing required packages — rm $DEPS_MARKER and re-run" >&2; exit 1; }

slurm_require_cutlass
if ! command -v nvcc >/dev/null 2>&1; then
    echo "ERROR: nvcc not in PATH — submit with #SBATCH --environment=vsch" >&2
    exit 1
fi

EVAL_PREP_DIR="${EVAL_PREP_DIR:-$EVAL_OUT_DIR/deepbench_${SLURM_JOB_ID:-local}}"
export EVAL_PREP_DIR
mkdir -p "$EVAL_PREP_DIR" "$(dirname "$DB")"
export CKS_GPU_BUF_GB="${CKS_GPU_BUF_GB:-16}"

DEEPBENCH_CSV="${DEEPBENCH_CSV:-$DEEPBENCH_DIR/deepbench_compatibility.csv}"
PROBLEMS_JSON="${PROBLEMS_JSON:-$EVAL_PREP_DIR/problems.json}"
export DEEPBENCH_LIMIT="${DEEPBENCH_LIMIT:-}"

if [ "$SKIP_PREP" != "1" ]; then
    echo "=== export DeepBench problems ==="
    if [ ! -f "$DEEPBENCH_CSV" ]; then
        echo "ERROR: missing $DEEPBENCH_CSV" >&2
        exit 1
    fi
    EXPORT_ARGS=(--csv "$DEEPBENCH_CSV" --out "$PROBLEMS_JSON")
    if [ -n "${DEEPBENCH_SPLIT:-}" ]; then
        EXPORT_ARGS+=(--split "$DEEPBENCH_SPLIT")
    fi
    if [ -n "$DEEPBENCH_LIMIT" ]; then
        EXPORT_ARGS+=(--limit "$DEEPBENCH_LIMIT")
    fi
    "$PYTHON" "$BENCHMARKS/deepbench/export_problems.py" "${EXPORT_ARGS[@]}"
    "$PYTHON" -c "import json; p=json.load(open('$PROBLEMS_JSON')); assert p['problems']"

    PROPOSE_ARGS=(
        --prep-dir "$EVAL_PREP_DIR"
        --problems "$PROBLEMS_JSON"
        --paper-dir "$MODELS_DIR"
        --capacity-dir "$CAPACITY_DIR"
        --python "$PYTHON"
        --nvmmh-gpu "$NVMMH_GPU"
        --capacity-seed "$CAPACITY_SEED"
        --random-seed "${RANDOM_PICK_SEED:-42}"
        --methods "${EVAL_METHODS:-default}"
    )
    if [ "$DEEPBENCH_INCLUDE_CAPACITY" = "1" ]; then
        PROPOSE_ARGS+=(--include-capacity)
    fi
    if [ "$DEEPBENCH_CAPACITY_ONLY" = "1" ]; then
        PROPOSE_ARGS+=(--capacity-only)
    fi
    echo "=== proposals (include_capacity=$DEEPBENCH_INCLUDE_CAPACITY) ==="
    "$PYTHON" "$BENCHMARKS/deepbench/propose_methods.py" "${PROPOSE_ARGS[@]}"
fi

PROPOSALS=("$EVAL_PREP_DIR"/proposals_*.json)
if [ ! -e "${PROPOSALS[0]}" ]; then
    echo "ERROR: no proposals in $EVAL_PREP_DIR/proposals_*.json" >&2
    exit 1
fi

slurm_init_db_paths
if [ "${CKS_EVAL_DEBUG:-0}" != "1" ]; then
    export CKS_CKPT_INTERVAL="${CKS_CKPT_INTERVAL:-600}"
fi
if [ "${CKS_EVAL_FRESH_DB:-0}" = "1" ]; then
    echo "CKS_EVAL_FRESH_DB=1 — not restoring $CKS_HOME_DB"
    rm -f "$CKS_HOME_DB" "$CKS_LOCAL_DB"
else
    slurm_db_restore
fi
if [ "${CKS_CKPT_INTERVAL:-600}" -gt 0 ] 2>/dev/null; then
    slurm_db_sync_loop
else
    echo "Background DB checkpoint disabled (CKS_CKPT_INTERVAL=0)"
fi
slurm_print_paths
echo "ckpt loop : every ${CKS_CKPT_INTERVAL:-600}s"

BENCH_WORKERS="${CKS_GPUS}"
echo "prep dir  : $EVAL_PREP_DIR"
echo "problems  : $PROBLEMS_JSON"
echo "phases    : $PHASE"
echo "proposals : ${PROPOSALS[*]}"
echo "workers   : $BENCH_WORKERS bench / $CKS_COMPILE_JOBS compile"

if [ "${CKS_EVAL_FRESH_BUILD:-0}" = "1" ]; then
    echo "Clearing build dir $CKS_BUILD_DIR"
    rm -rf "${CKS_BUILD_DIR:?}"/*
fi

run_eval() {
    local pid
    ulimit -c 0
    "$PYTHON" -u "$SRC/eval/run.py" "$@" &
    pid=$!
    trap slurm_db_checkpoint_final EXIT
    trap '
        kill -TERM '"$pid"' 2>/dev/null || true
        sleep 5
        pkill -KILL -P '"$pid"' 2>/dev/null || true
        kill -KILL '"$pid"' 2>/dev/null || true
        wait '"$pid"' 2>/dev/null || true
        slurm_db_checkpoint_final
        trap - EXIT
        exit
    ' SIGTERM
    wait "$pid"
}

EVAL_PHASES="$(echo "$PHASE" | tr ',' '\n' | grep -E '^(plan|compile|bench)$' | paste -sd, -)"
if [ -n "$EVAL_PHASES" ]; then
    run_eval \
        --phase        "$EVAL_PHASES" \
        --db           "$CKS_LOCAL_DB" \
        --build-dir    "$CKS_BUILD_DIR" \
        --proposals    "${PROPOSALS[@]}" \
        --compile-jobs "$CKS_COMPILE_JOBS" \
        --bench-workers "$BENCH_WORKERS"
fi

slurm_db_checkpoint_final

if [[ "$PHASE" == *bench* ]]; then
    REPORT_CSV="$EVAL_PREP_DIR/report.csv"
    echo "=== report -> $REPORT_CSV ==="
    "$PYTHON" "$SRC/eval/report.py" --db "$CKS_HOME_DB" --out "$REPORT_CSV"
fi

echo "Done. DB: $CKS_HOME_DB"
