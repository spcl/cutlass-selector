#!/bin/bash
# Shared MLP inference GEMM case-study body. Sourced from slurm_case_study*.sh.
#
# Steps (CASE_STEPS):
#   prep      trace_gemms.py + select_kernels.py  (CPU)
#   bench     benchmark_gemms.py via src/eval/run.py  (SM90 GPU + nvcc)
#   summarize analyze.py                          (CPU)
#   all       prep → bench → summarize

: "${CASE_STEPS:=all}"
: "${NVMMH_GPU:=H100_SXM}"
: "${CKS_COMPILE_JOBS:=32}"
: "${CKS_GPUS:=1}"
: "${CASE_BENCH_WORKERS:=${CKS_GPUS:-1}}"
: "${CASE_PHASE:=plan,compile,bench}"
: "${CASE_FRESH_DB:=0}"

_STUDY_JOB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../repo_paths.sh
source "${_STUDY_JOB_DIR}/../common/repo_paths.sh"
repo_paths_export "${CKS_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$REPO_ROOT"
export CKS_ROOT="$REPO_ROOT"
export PYTHONUNBUFFERED=1
export PYTHONPATH="${SRC}/model:${SRC}/autotuner:${SRC}/baseline/nvmmh:${PYTHONPATH:-}"
export CUTLASS_DIR="${CUTLASS_DIR:-$SRC/extern/cutlass}"

if [ ! -d "$CUTLASS_DIR/include" ]; then
    echo "ERROR: cutlass missing at $CUTLASS_DIR. Run: git submodule update --init src/extern/cutlass" >&2
    exit 1
fi

# shellcheck source=/dev/null
source "${BENCHMARKS}/common/slurm_common.sh"

PYTHON="${PYTHON:-$REPO_ROOT/.venv/bin/python}"
if [ ! -x "$PYTHON" ]; then
    echo "ERROR: missing venv python at $PYTHON" >&2
    exit 1
fi
export PYTHON

SCRIPT_DIR="${BENCHMARKS}/mlp_case_study"
OUT_DIR="${PAPER_DIR}/mlp_case_study"
PROPOSALS="${OUT_DIR}/proposals_case_study.json"

case_db_home() {
    echo "${OUT_DIR}/case_study.db"
}

case_db_live() {
    echo "/dev/shm/mlp_case_study_${SLURM_JOB_ID:-local}.db"
}

case_build_dir() {
    local scratch="${SCRATCH:-/iopsstor/scratch/cscs/${USER:-unknown}}"
    echo "${CASE_BUILD_DIR:-$scratch/mlp_case_study_build_${SLURM_JOB_ID:-local}}"
}

case_db_restore() {
    local home live
    home="$(case_db_home)"
    live="$(case_db_live)"
    mkdir -p "$(dirname "$home")" "$(dirname "$live")"
    rm -f "$live" "$live-wal" "$live-shm"
    if [ -f "$home" ]; then
        echo "Restoring case-study DB: $home → $live"
        cp "$home" "$live"
    fi
    export CASE_DB_HOME="$home"
    export CASE_DB_LIVE="$live"
}

case_db_checkpoint() {
    local home="${CASE_DB_HOME:-}"
    local live="${CASE_DB_LIVE:-}"
    [ -n "$home" ] && [ -n "$live" ] && [ -f "$live" ] || return 0
    echo "Checkpoint case-study DB: $live → $home"
    local tmp="${home}.tmp.$$"
    slurm_db_checkpoint_copy "$live" "$tmp" && mv "$tmp" "$home"
    rm -f "$home-wal" "$home-shm"
}

run_prep() {
    echo "=== trace MLP internal GEMMs ==="
    "$PYTHON" "$SCRIPT_DIR/trace_gemms.py" --out-dir "$OUT_DIR"

    echo "=== select rank-1 + nvMMH 8-variant candidates ==="
    "$PYTHON" "$SCRIPT_DIR/select_kernels.py" \
        --out-dir "$OUT_DIR" \
        --gpu "$NVMMH_GPU"

    if [ ! -f "$PROPOSALS" ]; then
        echo "ERROR: missing $PROPOSALS after select_kernels.py" >&2
        exit 1
    fi
}

run_bench() {
    if ! command -v nvcc >/dev/null 2>&1; then
        echo "ERROR: nvcc not in PATH — submit bench with #SBATCH --environment=vsch" >&2
        exit 1
    fi

    if [ ! -f "$PROPOSALS" ]; then
        echo "ERROR: missing $PROPOSALS — run CASE_STEPS=prep first" >&2
        exit 1
    fi

    case_db_restore
    trap case_db_checkpoint EXIT

    local fresh_flag=()
    if [ "${CASE_FRESH_DB:-0}" = "1" ]; then
        fresh_flag=(--fresh-db)
    fi

    echo "=== benchmark our rank-1 + nvMMH variants ==="
    echo "build_dir=$(case_build_dir)"
    echo "live_db=${CASE_DB_LIVE}"

    "$PYTHON" "$SCRIPT_DIR/benchmark_gemms.py" \
        --proposals "$PROPOSALS" \
        --db "${CASE_DB_LIVE}" \
        --build-dir "$(case_build_dir)" \
        --out "${OUT_DIR}/per_gemm_results.csv" \
        --python "$PYTHON" \
        --compile-jobs "$CKS_COMPILE_JOBS" \
        --bench-workers "$CASE_BENCH_WORKERS" \
        --phase "$CASE_PHASE" \
        "${fresh_flag[@]}"
}

run_summarize() {
    if [ ! -f "${OUT_DIR}/per_gemm_results.csv" ]; then
        echo "ERROR: missing per_gemm_results.csv — run CASE_STEPS=bench first" >&2
        exit 1
    fi
    echo "=== analyze weighted exec + time-to-solution ==="
    "$PYTHON" "$SCRIPT_DIR/analyze.py" --out-dir "$OUT_DIR"
    "$PYTHON" "$SCRIPT_DIR/plot_case_study.py" --out-dir "$OUT_DIR" --also-quick
}

case "$CASE_STEPS" in
    prep) run_prep ;;
    bench) run_bench ;;
    summarize) run_summarize ;;
    all)
        run_prep
        run_bench
        run_summarize
        ;;
    *)
        echo "ERROR: CASE_STEPS=$CASE_STEPS (use prep|bench|summarize|all)" >&2
        exit 1
        ;;
esac

echo "MLP case study step(s) complete: CASE_STEPS=$CASE_STEPS"
