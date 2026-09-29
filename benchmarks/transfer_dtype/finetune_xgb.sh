#!/bin/bash
# Fine-tune BF16 xgb_mse on a target dtype sweep (train-only features, no in-training eval).
#
# Usage:
#   benchmarks/transfer_dtype/finetune_xgb.sh fp32
#   FINETUNE_XGB_ESTIMATORS=200 benchmarks/transfer_dtype/finetune_xgb.sh fp8_e4m3

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
OUTDIR="${OUTDIR:-$TRANSFER_DIR/xgb_mse}"
SOURCE_MODEL="${SOURCE_XGB_MODEL:-$PAPER_DIR/xgb_mse/model_A.ubj}"

# Pretrain: 916 trees; finetune adds more rounds on top of the loaded booster.
FINETUNE_XGB_ESTIMATORS="${FINETUNE_XGB_ESTIMATORS:-200}"

if [ ! -f "$FEATURES" ]; then
    echo "ERROR: missing $FEATURES — run benchmarks/transfer_dtype/build_features.sh $DTYPE" >&2
    exit 1
fi
if [ ! -f "$SOURCE_MODEL" ]; then
    echo "ERROR: missing source model $SOURCE_MODEL" >&2
    exit 1
fi

mkdir -p "$OUTDIR"
LOG="$OUTDIR/xgb_mse.log"

echo "=== finetune XGB MSE ($DTYPE) ===" | tee "$LOG"
echo "  features : $FEATURES" | tee -a "$LOG"
echo "  init     : $SOURCE_MODEL" | tee -a "$LOG"
echo "  out      : $OUTDIR" | tee -a "$LOG"
echo "  +trees   : $FINETUNE_XGB_ESTIMATORS" | tee -a "$LOG"

"$PYTHON" "$SRC/model/train_xgb.py" \
    --features "$FEATURES" \
    --loss mse \
    --n-estimators "$FINETUNE_XGB_ESTIMATORS" \
    --init-model "$SOURCE_MODEL" \
    --skip-eval \
    --outdir "$OUTDIR" \
    2>&1 | tee -a "$LOG"

METRICS="$OUTDIR/metrics.json"
if [ -f "$METRICS" ]; then
    "$PYTHON" - "$METRICS" "$DTYPE" "$SOURCE_MODEL" "$FINETUNE_XGB_ESTIMATORS" <<'PY'
import json, sys
path, dtype, model, trees = sys.argv[1:5]
meta = json.loads(open(path).read_text())
meta["transfer"] = {
    "kind": "dtype",
    "target_dtype": dtype,
    "source_model": model,
    "finetune_extra_trees": int(trees),
    "eval_protocol": "broad_gemm_vs_nvmmh",
}
open(path, "w").write(json.dumps(meta, indent=2) + "\n")
print(f"updated {path} with transfer metadata")
PY
fi

echo "Done. Deploy artifact: $OUTDIR/model_A.ubj" | tee -a "$LOG"
