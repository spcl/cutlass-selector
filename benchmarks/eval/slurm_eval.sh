#!/bin/bash
#SBATCH --job-name=paper-eval
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --environment=vsch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --time=06:00:00
#SBATCH --signal=B:TERM@300
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --cpus-per-task=288
#SBATCH --mem=450G

#
# End-to-end broad GEMM evaluation: held-out shapes, proposals for nvMMH + four
# trained selectors (MLP/XGB × full/structural), compile, benchmark, report.
#
# Copy model artifacts to the cluster first (see src/eval/README.md or job output).
#
#   sbatch benchmarks/eval/slurm_eval.sh
#
# Medium debug (200 problems = 50 shapes × 4 layouts):
#   N_SHAPES=50 sbatch benchmarks/eval/slurm_eval.sh
#
# Resume compile/bench only (proposals already generated):
#   SKIP_PREP=1 EVAL_PREP_DIR=artifacts/eval/out/run_<job_id> PHASE=compile,bench sbatch benchmarks/eval/slurm_eval.sh
#
# Re-bench models with nvMMH rank-1 scheduler (after graft + DB reset on login node):
#   python src/eval/graft_nvmmh_scheduler.py --prep-dir artifacts/eval/out/run_<job_id> --db ~/autotuner/autotuner_eval.db
#   SKIP_PREP=1 EVAL_PREP_DIR=artifacts/eval/out/run_<job_id> PHASE=bench sbatch benchmarks/eval/slurm_eval.sh

set -euo pipefail
: "${CKS_TAG:=eval}"
: "${PHASE:=plan,compile,bench,cublas}"
: "${SKIP_PREP:=0}"

# Small smoke run: CKS_EVAL_DEBUG=1 (5 shapes, fresh DB + build).
if [ "${CKS_EVAL_DEBUG:-0}" = "1" ]; then
    : "${N_SHAPES:=5}"
    : "${CKS_EVAL_FRESH_DB:=1}"
    : "${CKS_EVAL_FRESH_BUILD:=1}"
    : "${CKS_COMPILE_JOBS:=16}"
    : "${CKS_CKPT_INTERVAL:=0}"  # debug only — short smoke, no background writer
fi

if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    source "${SLURM_SUBMIT_DIR}/benchmarks/common/slurm_common.sh"
else
    source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../common/slurm_common.sh"
fi

export CKS_TAG
export CKS_GPUS="${CKS_GPUS:-4}"
export CKS_COMPILE_JOBS="${CKS_COMPILE_JOBS:-64}"
export DB="${DB:-$HOME/autotuner/autotuner_eval.db}"

# Hold out everything seen in the production BF16 DBs (no env vars needed on Daint).
AUTOTUNER="${AUTOTUNER:-$HOME/autotuner}"
if [ -z "${HOLDOUT_SPECS_CSV:-}" ]; then
    HOLDOUT_SPECS=(
        "$AUTOTUNER/autotuner_bf16_final.db:bf16_final"
        "$AUTOTUNER/autotuner_bf16_eval.db:bf16_eval"
    )
else
    IFS=',' read -r -a HOLDOUT_SPECS <<< "$HOLDOUT_SPECS_CSV"
fi
export N_SHAPES="${N_SHAPES:-2000}"
export NVMMH_GPU="${NVMMH_GPU:-H100_SXM}"
# "<method>:<dir under MODELS_DIR>"; backend is inferred from the method prefix.
: "${EVAL_MODELS:=mlp_full:mlp_mse mlp_structural:mlp_mse_structural xgb_full:xgb_mse xgb_structural:xgb_mse_structural}"

slurm_cd_repo
export MODELS_DIR="${MODELS_DIR:-$PAPER_DIR}"

# .venv — nvMMH, XGBoost, sklearn, etc. are not in the CSCS uenv.
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
    echo "Installing paper-eval dependencies into $VENV ..."
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

EVAL_PREP_DIR="${EVAL_PREP_DIR:-$EVAL_OUT_DIR/run_${SLURM_JOB_ID:-local}}"
export EVAL_PREP_DIR
SHAPES_JSON="$EVAL_PREP_DIR/shapes.json"
mkdir -p "$EVAL_PREP_DIR" "$(dirname "$DB")"
export CKS_GPU_BUF_GB="${CKS_GPU_BUF_GB:-16}"

if [ "$SKIP_PREP" != "1" ]; then
    echo "=== shapes (n=$N_SHAPES, holdout: ${HOLDOUT_SPECS[*]}) ==="
    for spec in "${HOLDOUT_SPECS[@]}"; do
        db_path="${spec%%:*}"
        if [ ! -f "$db_path" ]; then
            echo "ERROR: holdout DB missing: $db_path" >&2
            exit 1
        fi
    done
    rm -f "$SHAPES_JSON"
    HOLDOUT_ARGS=()
    for spec in "${HOLDOUT_SPECS[@]}"; do
        HOLDOUT_ARGS+=(--holdout "$spec")
    done
    "$PYTHON" "$SRC/eval/shapes.py" \
        --n "$N_SHAPES" \
        "${HOLDOUT_ARGS[@]}" \
        --out "$SHAPES_JSON"
    "$PYTHON" -c "import json; json.load(open('$SHAPES_JSON'))"

    echo "=== nvMMH proposals ==="
    "$PYTHON" "$SRC/eval/propose.py" --method nvmmh --gpu "$NVMMH_GPU" \
        --shapes "$SHAPES_JSON" --out "$EVAL_PREP_DIR/proposals_nvmmh.json"

    for spec in $EVAL_MODELS; do
        method="${spec%%:*}"
        model_dir="$MODELS_DIR/${spec#*:}"
        echo "=== $method proposals ($model_dir) ==="
        if [ ! -f "$model_dir/metrics.json" ]; then
            echo "ERROR: missing $model_dir/metrics.json" >&2
            echo "  MODELS_DIR=$MODELS_DIR  CKS_ROOT=$CKS_ROOT" >&2
            echo "  scp -r artifacts/analysis/paper/mlp_mse artifacts/analysis/paper/mlp_mse_structural \\" >&2
            echo "      artifacts/analysis/paper/xgb_mse artifacts/analysis/paper/xgb_mse_structural \\" >&2
            echo "      <cluster>:cutlass-selector/artifacts/analysis/paper/" >&2
            exit 1
        fi
        backend="${method%%_*}"
        "$PYTHON" "$SRC/eval/propose.py" --method "$method" --backend "$backend" \
            --model-dir "$model_dir" \
            --scheduler-from "$EVAL_PREP_DIR/proposals_nvmmh.json" \
            --shapes "$SHAPES_JSON" --out "$EVAL_PREP_DIR/proposals_${method}.json"
    done
fi

PROPOSALS=("$EVAL_PREP_DIR"/proposals_*.json)
if [ ! -e "${PROPOSALS[0]}" ]; then
    echo "ERROR: no proposals in $EVAL_PREP_DIR/proposals_*.json" >&2
    echo "  Run without SKIP_PREP=1, or set EVAL_PREP_DIR to a prior run_* directory." >&2
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

if [[ "$PHASE" == *cublas* ]]; then
    if [ ! -f "$SHAPES_JSON" ]; then
        echo "ERROR: $SHAPES_JSON missing (needed by the cuBLASLt baseline)" >&2
        exit 1
    fi
    echo "=== cuBLASLt baseline ==="
    make -C "$SRC/baseline/cublas_lt"
    "$PYTHON" -u "$SRC/eval/cublas.py" --db "$CKS_LOCAL_DB" --shapes "$SHAPES_JSON"
fi

slurm_db_checkpoint_final

if [[ "$PHASE" == *bench* ]] || [[ "$PHASE" == *cublas* ]]; then
    REPORT_CSV="$EVAL_PREP_DIR/report.csv"
    echo "=== report -> $REPORT_CSV ==="
    "$PYTHON" "$SRC/eval/report.py" --db "$CKS_HOME_DB" --out "$REPORT_CSV"
fi

echo "Done. DB: $CKS_HOME_DB"
