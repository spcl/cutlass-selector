#!/bin/bash
# Build features + finetune MLP for FP32 and FP8 dtype transfer.
#
# Usage:
#   benchmarks/transfer_dtype/run_finetune_dtype.sh
#   benchmarks/transfer_dtype/run_finetune_dtype.sh fp32    # one dtype only

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

run_one() {
    "$SCRIPT_DIR/build_features.sh" "$1"
    "$SCRIPT_DIR/finetune_mlp.sh" "$1"
}

case "${1:-all}" in
    all)
        run_one fp32
        run_one fp8_e4m3
        ;;
    fp32|fp8_e4m3)
        run_one "$1"
        ;;
    *)
        echo "usage: $0 [all|fp32|fp8_e4m3]" >&2
        exit 1
        ;;
esac

echo "All finetune runs complete."
