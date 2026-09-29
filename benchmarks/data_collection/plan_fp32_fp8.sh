#!/bin/bash
# Write FP32 and FP8 TN-only sweep plans (593 shapes each).
# Each tag gets its own DB by default so sweeps can run in parallel.
#
# Usage:
#   benchmarks/data_collection/plan_fp32_fp8.sh
#   DB=$HOME/autotuner/autotuner_sweep_fp32_tn.db benchmarks/data_collection/plan_fp32_fp8.sh  # fp32 only via plan.py

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=../common/slurm_common.sh
source "$SCRIPT_DIR/../common/slurm_common.sh"
slurm_cd_repo
PYTHON="${PYTHON:-python3}"
SHAPES="${SHAPES:-$PLANS_DIR/shapes.json}"

plan() {
    local tag="$1" dtype="$2"
    local db="${DB:-$HOME/autotuner/autotuner_${tag}.db}"
    mkdir -p "$(dirname "$db")"
    echo "=== plan write tag=$tag dtype=$dtype layouts=TN wave_eff=$CKS_WAVE_EFF DB=$db ==="
    "$PYTHON" "$SRC/planning/plan.py" write \
        --tag "$tag" \
        --dtype "$dtype" \
        --layouts TN \
        --shapes "$SHAPES" \
        --wave-eff "$CKS_WAVE_EFF" \
        --db "$db"
}

plan sweep_fp32_tn fp32
plan sweep_fp8_tn fp8_e4m3

echo "Done. DBs: ~/autotuner/autotuner_sweep_fp32_tn.db, ~/autotuner/autotuner_sweep_fp8_tn.db"
