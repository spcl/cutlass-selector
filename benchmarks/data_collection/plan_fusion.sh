#!/bin/bash
# Plan fusion fine-tune sweeps (fp16 / fp32 / fp8_e4m3), TN + TMA, 5 ML fusions.
# Each of 593 shapes gets a random fusion (shared seed across dtypes).
# Shortlist k≈700 → ~1.25M pairs total across 3 dtypes (~FP32 walltime).
#
# Usage:
#   benchmarks/data_collection/plan_fusion.sh              # all three dtypes → separate DBs
#   benchmarks/data_collection/plan_fusion.sh fp16         # one dtype
#   FUSION_SEED=0xF0510A benchmarks/data_collection/plan_fusion.sh

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=../common/slurm_common.sh
source "$SCRIPT_DIR/../common/slurm_common.sh"
slurm_cd_repo
PYTHON="${PYTHON:-python3}"
SHAPES="${SHAPES:-$PLANS_DIR/shapes.json}"
FUSION_SEED="${FUSION_SEED:-0xF0510A}"
MODE="${1:-all}"

plan_one() {
    local dtype="$1"
    local tag="sweep_fusion_${dtype}"
    # fp8 tag uses shorter name
    if [ "$dtype" = "fp8_e4m3" ]; then
        tag="sweep_fusion_fp8"
    fi
    local db="${DB:-$HOME/autotuner/autotuner_${tag}.db}"
    mkdir -p "$(dirname "$db")"
    echo "=== fusion plan tag=$tag dtype=$dtype layouts=TN seed=$FUSION_SEED DB=$db ==="
    "$PYTHON" "$SRC/planning/plan/write_fusion.py" \
        --tag "$tag" \
        --dtype "$dtype" \
        --layouts TN \
        --shapes "$SHAPES" \
        --fusion-seed "$FUSION_SEED" \
        --wave-eff "$CKS_WAVE_EFF" \
        --db "$db"
}

case "$MODE" in
    fp16|fp32)
        plan_one "$MODE"
        ;;
    fp8|fp8_e4m3)
        plan_one fp8_e4m3
        ;;
    all)
        plan_one fp16
        plan_one fp32
        plan_one fp8_e4m3
        ;;
    *)
        echo "Usage: $0 [fp16|fp32|fp8|all]" >&2
        exit 2
        ;;
esac

echo "Done. Submit with benchmarks/data_collection/run_fusion.sh (set CKS_TAG + DB= to match)."
