#!/bin/bash
# Write the BF16 primary sweep plan (593 shapes, 4 layouts).
#
# Usage:
#   benchmarks/data_collection/plan_sweep.sh
#   CKS_WAVE_EFF=old_we benchmarks/data_collection/plan_sweep.sh
#   DB=$HOME/autotuner/autotuner_sweep.db benchmarks/data_collection/plan_sweep.sh

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=../common/slurm_common.sh
source "$SCRIPT_DIR/../common/slurm_common.sh"

slurm_cd_repo
PYTHON="${PYTHON:-python3}"
TAG="${CKS_TAG:-sweep}"
DB="${DB:-$HOME/autotuner/autotuner_${TAG}.db}"
SHAPES="${SHAPES:-$PLANS_DIR/shapes.json}"

mkdir -p "$(dirname "$DB")"

echo "=== plan write tag=$TAG wave_eff=$CKS_WAVE_EFF DB=$DB ==="
"$PYTHON" "$SRC/planning/plan.py" write \
    --tag "$TAG" \
    --shapes "$SHAPES" \
    --wave-eff "$CKS_WAVE_EFF" \
    --db "$DB"

echo "Done. Tag: $TAG  wave_eff: $CKS_WAVE_EFF  DB: $DB"
