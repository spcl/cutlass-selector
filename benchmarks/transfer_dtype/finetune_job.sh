#!/bin/bash
# Shared dtype-transfer finetune body (features + MLP + XGB). Sourced from slurm_finetune*.sh.

: "${TRANSFER_DTYPE:=all}"
: "${FINETUNE_EPOCHS:=48}"
: "${FINETUNE_LR:=4e-5}"
: "${FINETUNE_XGB_ESTIMATORS:=200}"
: "${SOURCE_CKPT:-}"

CKS_ROOT="${CKS_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
# shellcheck source=../common/repo_paths.sh
source "$CKS_ROOT/benchmarks/common/repo_paths.sh"
repo_paths_export "$CKS_ROOT"
cd "$REPO_ROOT" || exit 1
export CKS_ROOT="$REPO_ROOT"
export PYTHONUNBUFFERED=1
export PYTHONPATH="${SRC}/model:${PYTHONPATH:-}"

PYTHON="${PYTHON:-$REPO_ROOT/.venv/bin/python}"
if [ ! -x "$PYTHON" ]; then
    echo "ERROR: missing venv python at $PYTHON" >&2
    exit 1
fi
export PYTHON

SCRIPT_DIR="${BENCHMARKS}/transfer_dtype"
export FINETUNE_EPOCHS FINETUNE_LR FINETUNE_XGB_ESTIMATORS
if [ -n "$SOURCE_CKPT" ]; then
    export SOURCE_CKPT
fi

run_dtype() {
    local dtype="$1"
    echo ""
    echo "========== transfer finetune: $dtype =========="
    "$SCRIPT_DIR/build_features.sh" "$dtype"
    "$SCRIPT_DIR/finetune_mlp.sh" "$dtype"
    "$SCRIPT_DIR/finetune_xgb.sh" "$dtype"
}

case "$TRANSFER_DTYPE" in
    all)
        run_dtype fp32
        run_dtype fp8_e4m3
        ;;
    fp32|fp8_e4m3)
        run_dtype "$TRANSFER_DTYPE"
        ;;
    *)
        echo "ERROR: TRANSFER_DTYPE=$TRANSFER_DTYPE (use all|fp32|fp8_e4m3)" >&2
        exit 1
        ;;
esac

echo ""
echo "Finetune complete. Artifacts under artifacts/analysis/paper/transfer/dtype/*/mlp_mse/ and xgb_mse/"
