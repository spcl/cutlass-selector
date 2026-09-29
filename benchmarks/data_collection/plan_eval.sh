#!/bin/bash
# Write exhaustive BF16 eval plan (17 held-out shapes × 4 layouts × all valid configs).
#
# Shapes match src/planning/plan/shapes.py EVAL_SHAPES and src/autotuner/scheduler.py BENCHMARK_SHAPES.
#
# Usage:
#   benchmarks/data_collection/plan_eval.sh
#   DB=$HOME/autotuner/autotuner_bf16_eval.db benchmarks/data_collection/plan_eval.sh

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=../common/slurm_common.sh
source "$SCRIPT_DIR/../common/slurm_common.sh"

slurm_cd_repo
PYTHON="${PYTHON:-python3}"
TAG="${CKS_TAG:-bf16_eval}"
DB="${DB:-$HOME/autotuner/autotuner_${TAG}.db}"
SHAPES="${SHAPES:-$PLANS_DIR/eval_shapes.json}"

mkdir -p "$(dirname "$DB")"

echo "=== plan write (exhaustive) tag=$TAG DB=$DB ==="
"$PYTHON" "$SRC/planning/plan.py" write \
    --tag "$TAG" \
    --shapes "$SHAPES" \
    --exhaustive \
    --wave-eff "${CKS_WAVE_EFF:-new_we}" \
    --db "$DB"

echo "Done. Tag: $TAG  shapes: 17  DB: $DB"
