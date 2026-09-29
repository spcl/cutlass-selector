#!/bin/bash
# Shared fusion-transfer study eval body. Sourced from slurm_study_eval*.sh.
# Requires fusion eval scripts (run_eval_grid.py, etc.) — see STUDY.md.

: "${STUDY_DTYPE:=all}"
: "${STUDY_EVAL_SUITE:=eval}"    # eval | eval2
: "${STUDY_EVAL_STEPS:=all}"     # prep | eval | summarize | all
: "${STUDY_PHASE:=plan,compile,bench}"
: "${NVMMH_GPU:=H100_SXM}"
: "${CKS_COMPILE_JOBS:=64}"
: "${CKS_GPUS:=4}"
: "${EVAL_N_SHAPES:=1000}"
: "${SLURM_ARRAY_TASK_ID:=}"

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
if ! command -v nvcc >/dev/null 2>&1; then
    echo "ERROR: nvcc not in PATH — submit with #SBATCH --environment=vsch" >&2
    exit 1
fi

PYTHON="${PYTHON:-$REPO_ROOT/.venv/bin/python}"
if [ ! -x "$PYTHON" ]; then
    echo "ERROR: missing venv python at $PYTHON" >&2
    exit 1
fi
export PYTHON
export STUDY_EVAL_SUITE

SCRIPT_DIR="${BENCHMARKS}/transfer_fusion"
AUTOTUNER="${AUTOTUNER:-$HOME/autotuner}"
# shellcheck source=/dev/null
source "${BENCHMARKS}/common/slurm_common.sh"

if [ ! -f "$SCRIPT_DIR/run_eval_grid.py" ]; then
    echo "ERROR: fusion eval not wired yet — missing $SCRIPT_DIR/run_eval_grid.py" >&2
    echo "See benchmarks/transfer_fusion/STUDY.md." >&2
    exit 1
fi

study_eval_home_db() {
    local dtype="$1"
    local suite="${STUDY_EVAL_SUITE:-eval}"
    local suffix=""
    if [ "$suite" != "eval" ]; then
        suffix="_${suite}"
    fi
    echo "$PAPER_DIR/transfer_fusion_study/${suite}/autotuner_transfer_fusion${suffix}_${dtype}.db"
}

study_eval_local_db() {
    local dtype="$1"
    local suite="${STUDY_EVAL_SUITE:-eval}"
    echo "/dev/shm/transfer_fusion_${suite}_${dtype}_${SLURM_JOB_ID:-local}.db"
}

study_eval_db_restore() {
    local dtype="$1"
    export CKS_HOME_DB
    CKS_HOME_DB="$(study_eval_home_db "$dtype")"
    export CKS_LOCAL_DB
    CKS_LOCAL_DB="$(study_eval_local_db "$dtype")"
    mkdir -p "$(dirname "$CKS_HOME_DB")"
    rm -f "$CKS_LOCAL_DB" "$CKS_LOCAL_DB-wal" "$CKS_LOCAL_DB-shm" \
        "$CKS_HOME_DB-wal" "$CKS_HOME_DB-shm"
    if [ -f "$CKS_HOME_DB" ]; then
        echo "Restoring fusion transfer eval DB: $CKS_HOME_DB → $CKS_LOCAL_DB"
        cp "$CKS_HOME_DB" "$CKS_LOCAL_DB"
    fi
    export CKS_TRANSFER_EVAL_DB="$CKS_LOCAL_DB"
    echo "live DB: $CKS_TRANSFER_EVAL_DB"
}

study_eval_db_checkpoint() {
    local home="${CKS_HOME_DB:-}"
    local live="${CKS_TRANSFER_EVAL_DB:-}"
    [ -n "$home" ] && [ -n "$live" ] && [ -f "$live" ] || return 0
    echo "Checkpoint fusion transfer eval DB: $live → $home"
    local tmp="${home}.tmp.$$"
    slurm_db_checkpoint_copy "$live" "$tmp" && mv "$tmp" "$home"
    rm -f "$home-wal" "$home-shm"
}

run_prep() {
    study_eval_db_restore "$STUDY_DTYPE"
    trap study_eval_db_checkpoint EXIT

    if [ "${STUDY_EVAL_SUITE:-eval}" = "eval2" ]; then
        echo "=== eval2 zero-shot shapes (dtype=${STUDY_DTYPE}) ==="
        "$PYTHON" "$SCRIPT_DIR/gen_eval2_shapes.py" --autotuner "$AUTOTUNER" --dtype "$STUDY_DTYPE"
    else
        echo "=== eval shapes (TN fusion, n=${EVAL_N_SHAPES}, dtype=${STUDY_DTYPE}) ==="
        "$PYTHON" "$SCRIPT_DIR/gen_eval_shapes.py" --autotuner "$AUTOTUNER" --dtype "$STUDY_DTYPE" --n "$EVAL_N_SHAPES"
    fi

    echo "=== nvMMH proposals (dtype=${STUDY_DTYPE}) ==="
    "$PYTHON" "$SCRIPT_DIR/run_eval_grid.py" --dtype "$STUDY_DTYPE" --prep-only \
        --gpu "$NVMMH_GPU" --python "$PYTHON"

    echo "=== nvMMH bench (dtype=${STUDY_DTYPE}) ==="
    "$PYTHON" "$SCRIPT_DIR/run_eval_grid.py" --dtype "$STUDY_DTYPE" --nvmmh-only \
        --phase "$STUDY_PHASE" --gpu "$NVMMH_GPU" \
        --compile-jobs "$CKS_COMPILE_JOBS" --bench-workers "$CKS_GPUS" \
        --python "$PYTHON"
}

run_eval() {
    local array_flag=()
    if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
        array_flag=(--array-index "$SLURM_ARRAY_TASK_ID")
    fi
    "$PYTHON" "$SCRIPT_DIR/run_eval_grid.py" \
        --dtype "$STUDY_DTYPE" \
        --phase "$STUDY_PHASE" \
        --gpu "$NVMMH_GPU" \
        --compile-jobs "$CKS_COMPILE_JOBS" \
        --bench-workers "$CKS_GPUS" \
        --skip-missing \
        --skip-done \
        --python "$PYTHON" \
        "${array_flag[@]}"
}

run_summarize() {
    if [ "${STUDY_EVAL_SUITE:-eval}" = "eval2" ]; then
        "$PYTHON" "$SCRIPT_DIR/summarize_eval2.py" --dtype "$STUDY_DTYPE"
    else
        "$PYTHON" "$SCRIPT_DIR/summarize_study.py" --dtype "$STUDY_DTYPE"
        "$PYTHON" "$SCRIPT_DIR/plot_study.py" --also-quick
    fi
}

case "$STUDY_EVAL_STEPS" in
    prep) run_prep ;;
    eval) run_eval ;;
    summarize) run_summarize ;;
    all)
        run_prep
        run_eval
        run_summarize
        ;;
    *)
        echo "ERROR: STUDY_EVAL_STEPS=$STUDY_EVAL_STEPS (use prep|eval|summarize|all)" >&2
        exit 1
        ;;
esac

echo "Fusion study eval step(s) complete."
