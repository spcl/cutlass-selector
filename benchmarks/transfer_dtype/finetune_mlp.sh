#!/bin/bash
# Fine-tune BF16 mlp_mse on a target dtype sweep (train-only features, no in-training eval).
#
# Usage:
#   benchmarks/transfer_dtype/finetune_mlp.sh fp32
#   benchmarks/transfer_dtype/finetune_mlp.sh fp8_e4m3
#   FINETUNE_EPOCHS=72 FINETUNE_LR=4e-5 benchmarks/transfer_dtype/finetune_mlp.sh fp32

# shellcheck source=../common/repo_paths.sh
source "$(cd "$(dirname "$0")" && pwd)/../common/repo_paths.sh"
repo_paths_export "$(dirname "$0")"
PYTHON="${PYTHON:-$REPO_ROOT/.venv/bin/python}"

DTYPE="${1:?usage: $0 fp32|fp8_e4m3}"

case "$DTYPE" in
    fp32)     ;;
    fp8_e4m3) ;;
    *)
        echo "ERROR: unsupported dtype $DTYPE (use fp32 or fp8_e4m3)" >&2
        exit 1
        ;;
esac

TRANSFER_DIR="${TRANSFER_DIR:-$PAPER_DIR/transfer/dtype/${DTYPE}}"
FEATURES="${FEATURES:-$TRANSFER_DIR/features.parquet}"
OUTDIR="${OUTDIR:-$TRANSFER_DIR/mlp_mse}"
SOURCE_CKPT="${SOURCE_CKPT:-$PAPER_DIR/mlp_mse/model_mlp.pt}"

# Pretrain reference: artifacts/analysis/paper/mlp_mse/metrics.json (epochs=144, lr≈1.19e-4)
FINETUNE_EPOCHS="${FINETUNE_EPOCHS:-48}"
FINETUNE_LR="${FINETUNE_LR:-4e-5}"
FINETUNE_BATCH="${FINETUNE_BATCH:-2048}"
HIDDEN=(1024 1024 512 256)
DROPOUT="0.043937900354189416"

if [ ! -f "$FEATURES" ]; then
    echo "ERROR: missing $FEATURES — run benchmarks/transfer_dtype/build_features.sh $DTYPE" >&2
    exit 1
fi
if [ ! -f "$SOURCE_CKPT" ]; then
    echo "ERROR: missing source checkpoint $SOURCE_CKPT" >&2
    exit 1
fi

mkdir -p "$OUTDIR"
LOG="$OUTDIR/mlp_mse.log"

echo "=== finetune MLP MSE ($DTYPE) ===" | tee "$LOG"
echo "  features : $FEATURES" | tee -a "$LOG"
echo "  init     : $SOURCE_CKPT" | tee -a "$LOG"
echo "  out      : $OUTDIR" | tee -a "$LOG"
echo "  epochs   : $FINETUNE_EPOCHS  lr=$FINETUNE_LR  batch=$FINETUNE_BATCH" | tee -a "$LOG"

"$PYTHON" "$SRC/model/train_mlp.py" \
    --features "$FEATURES" \
    --loss mse \
    --epochs "$FINETUNE_EPOCHS" \
    --lr "$FINETUNE_LR" \
    --batch-size "$FINETUNE_BATCH" \
    --dropout "$DROPOUT" \
    --hidden "${HIDDEN[@]}" \
    --init-checkpoint "$SOURCE_CKPT" \
    --scaler-policy fit \
    --skip-eval \
    --outdir "$OUTDIR" \
    2>&1 | tee -a "$LOG"

# Annotate metrics for transfer provenance.
METRICS="$OUTDIR/metrics.json"
if [ -f "$METRICS" ]; then
    "$PYTHON" - "$METRICS" "$DTYPE" "$SOURCE_CKPT" "$FINETUNE_EPOCHS" "$FINETUNE_LR" <<'PY'
import json, sys
path, dtype, ckpt, epochs, lr = sys.argv[1:6]
meta = json.loads(open(path).read_text())
meta["transfer"] = {
    "kind": "dtype",
    "target_dtype": dtype,
    "source_checkpoint": ckpt,
    "finetune_epochs": int(epochs),
    "finetune_lr": float(lr),
    "scaler_policy": "fit",
    "eval_protocol": "broad_gemm_vs_nvmmh",
}
open(path, "w").write(json.dumps(meta, indent=2) + "\n")
print(f"updated {path} with transfer metadata")
PY
fi

echo "Done. Deploy artifact: $OUTDIR/model_mlp.pt2" | tee -a "$LOG"
