#!/bin/bash
# Build train-only features for dtype transfer finetuning.
#
# Usage:
#   benchmarks/transfer_dtype/build_features.sh fp32
#   benchmarks/transfer_dtype/build_features.sh fp8_e4m3
#   DB=~/autotuner/autotuner_sweep_fp32_tn.db benchmarks/transfer_dtype/build_features.sh fp32

# shellcheck source=../common/repo_paths.sh
source "$(cd "$(dirname "$0")" && pwd)/../common/repo_paths.sh"
repo_paths_export "$(dirname "$0")"
PYTHON="${PYTHON:-$REPO_ROOT/.venv/bin/python}"

DTYPE="${1:?usage: $0 fp32|fp8_e4m3}"
case "$DTYPE" in
    fp32)     TAG="${TRAIN_TAG:-sweep_fp32_tn}" ;;
    fp8_e4m3) TAG="${TRAIN_TAG:-sweep_fp8_tn}" ;;
    *)
        echo "ERROR: unsupported dtype $DTYPE (use fp32 or fp8_e4m3)" >&2
        exit 1
        ;;
esac

DB="${DB:-$HOME/autotuner/autotuner_${TAG}.db}"
OUT_DIR="${OUT_DIR:-$PAPER_DIR/transfer/dtype/${DTYPE}}"
OUT_PARQUET="$OUT_DIR/features.parquet"

if [ ! -f "$DB" ]; then
    echo "ERROR: missing DB $DB" >&2
    exit 1
fi

mkdir -p "$OUT_DIR"

echo "=== transfer features ==="
echo "  dtype : $DTYPE"
echo "  tag   : $TAG"
echo "  db    : $DB"
echo "  out   : $OUT_PARQUET"

"$PYTHON" "$SRC/model/features.py" \
    --db "$DB" \
    --train-tags "$TAG" \
    --dtype "$DTYPE" \
    --out "$OUT_PARQUET"

echo "Done."
