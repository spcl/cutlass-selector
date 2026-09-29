#!/bin/bash
# Shared fusion-transfer study training body. Sourced from slurm_study_train*.sh.
#
# Parallel dtype jobs:
#   sbatch benchmarks/transfer_fusion/slurm_study_train_fp16.sh
#   sbatch benchmarks/transfer_fusion/slurm_study_train_fp32.sh
#   sbatch benchmarks/transfer_fusion/slurm_study_train_fp8.sh

: "${STUDY_DTYPE:=all}"
: "${STUDY_STEPS:=all}"          # prep | train | all
: "${SLURM_ARRAY_TASK_ID:=}"

_STUDY_JOB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../repo_paths.sh
source "${_STUDY_JOB_DIR}/../common/repo_paths.sh"
repo_paths_export "${CKS_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$REPO_ROOT"
export CKS_ROOT="$REPO_ROOT"
export PYTHONUNBUFFERED=1
export PYTHONPATH="${SRC}/model:${CKS_ROOT}:${PYTHONPATH:-}"

PYTHON="${PYTHON:-$REPO_ROOT/.venv/bin/python}"
if [ ! -x "$PYTHON" ]; then
    echo "ERROR: missing venv python at $PYTHON" >&2
    exit 1
fi
export PYTHON

SCRIPT_DIR="${BENCHMARKS}/transfer_fusion"
AUTOTUNER="${AUTOTUNER:-$HOME/autotuner}"

dtype_flag() {
    if [ "$STUDY_DTYPE" != "all" ]; then
        echo --dtype "$STUDY_DTYPE"
    fi
}

run_prep() {
    echo "=== build fusion study features (dtype=${STUDY_DTYPE}) ==="
    "$PYTHON" "$SCRIPT_DIR/build_study_features.py" --autotuner "$AUTOTUNER" $(dtype_flag)

    echo "=== verify fusion features (dtype=${STUDY_DTYPE}) ==="
    "$PYTHON" "$SCRIPT_DIR/verify_fusion_features.py" --autotuner "$AUTOTUNER" $(dtype_flag)

    echo "=== nested shape subsets (dtype=${STUDY_DTYPE}) ==="
    "$PYTHON" "$SCRIPT_DIR/make_subsets.py" --autotuner "$AUTOTUNER" $(dtype_flag)

    if [ "$STUDY_DTYPE" = "fp16" ] || [ "$STUDY_DTYPE" = "all" ]; then
        echo "=== scaler policy comparison (FP16 25% MLP) ==="
        "$PYTHON" "$SCRIPT_DIR/scaler_compare.py"
    else
        echo "=== skip scaler compare (fp32/fp8 job; policy written by fp16 job) ==="
    fi

    echo "=== run manifest ==="
    "$PYTHON" -c "from scripts.transfer_fusion.study_common import write_run_manifest; print(write_run_manifest())"
}

run_train() {
    local dtype_flag_arr=()
    if [ "$STUDY_DTYPE" != "all" ]; then
        dtype_flag_arr=(--dtype "$STUDY_DTYPE")
    fi
    if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
        local run_id
        run_id="$("$PYTHON" "$SCRIPT_DIR/run_id_at.py" "$SLURM_ARRAY_TASK_ID" \
            --dtype "$STUDY_DTYPE")"
        echo "=== train array task ${SLURM_ARRAY_TASK_ID} -> ${run_id} ==="
        "$PYTHON" "$SCRIPT_DIR/train_one.py" --run-id "$run_id" --python "$PYTHON"
    else
        echo "=== train grid (dtype=${STUDY_DTYPE}) ==="
        "$PYTHON" "$SCRIPT_DIR/run_training.py" --python "$PYTHON" "${dtype_flag_arr[@]}"
    fi
}

case "$STUDY_STEPS" in
    prep) run_prep ;;
    train) run_train ;;
    all)
        run_prep
        run_train
        ;;
    *)
        echo "ERROR: STUDY_STEPS=$STUDY_STEPS (use prep|train|all)" >&2
        exit 1
        ;;
esac

echo "Fusion study training step(s) complete."
