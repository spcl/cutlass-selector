#!/bin/bash
# Shared baseline-study body. Sourced from slurm_baselines*.sh.
#
# Steps (BASELINE_STEPS):
#   analysis    training-only feature correlations / redundancy / effect sizes (CPU)
#   train       ridge linear baselines: full + structural (CPU, RAM-heavy)
#   analytical  hardware analytical score: val pick → frozen test (CPU; needs analysis)
#   summarize   comparison table + figures + SUMMARY.md (CPU)
#   finish      analytical → summarize (needs analysis + train outputs)
#   all         analysis → train → analytical → summarize (monolithic; avoid under load)
#
# Env:
#   CKS_ROOT          repo root (default: the submit directory)
#   FEATURES             features.parquet (default: $PAPER_DIR/features.parquet)
#   BASELINE_OUT         output root (default: $PAPER_DIR/baselines)
#   VENV                 python venv (default: $CKS_ROOT/.venv)

: "${BASELINE_STEPS:=all}"

set -euo pipefail

_STUDY_JOB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../repo_paths.sh
source "${_STUDY_JOB_DIR}/../../common/repo_paths.sh"
repo_paths_export "${CKS_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$REPO_ROOT"
export CKS_ROOT="$REPO_ROOT"
export PYTHONUNBUFFERED=1
export PYTHONPATH="${SRC}/model:${BENCHMARKS}/transfer_dtype:${PYTHONPATH:-}"

SCRIPT_DIR="${BENCHMARKS}/baselines/learned_and_analytical"
BASELINE_OUT="${BASELINE_OUT:-${PAPER_DIR}/baselines}"
FEATURES_HINT="${FEATURES:-${PAPER_DIR}/features.parquet}"

VENV="${VENV:-${CKS_ROOT}/.venv}"
if [ ! -x "$VENV/bin/python" ]; then
    echo "Creating venv at $VENV ..."
    python3 -m venv "$VENV"
fi
PYTHON="$VENV/bin/python"
PIP="$VENV/bin/pip"

DEPS_MARKER="$VENV/.baseline_deps_ok"
if [ ! -f "$DEPS_MARKER" ]; then
    echo "Installing baseline dependencies into $VENV ..."
    "$PIP" install -q --upgrade pip
    "$PIP" install -q \
        -r "${CKS_ROOT}/requirements.txt" \
        scikit-learn pyarrow pandas matplotlib
    touch "$DEPS_MARKER"
fi

"$PYTHON" -c "import sklearn, pyarrow, pandas, matplotlib" \
    || { echo "ERROR: venv missing required packages (sklearn pyarrow pandas matplotlib)" >&2; exit 1; }
export PYTHON

ensure_features() {
    local resolved
    if resolved="$("$PYTHON" -c "
import sys
sys.path.insert(0, '${SCRIPT_DIR}')
from pathlib import Path
from common import resolve_features_path
print(resolve_features_path(Path('${FEATURES_HINT}')))
" 2>/dev/null)"; then
        FEATURES="$resolved"
        return 0
    fi

    local train_db="${TRAIN_DB:-$HOME/autotuner/autotuner_bf16_final.db}"
    local eval_db="${EVAL_DB:-$HOME/autotuner/autotuner_bf16_eval.db}"
    local out="${PAPER_DIR}/features.parquet"
    local lock="${PAPER_DIR}/.features_build.lock"
    if [ ! -f "$train_db" ]; then
        echo "ERROR: no usable features.parquet and missing train DB: $train_db" >&2
        return 1
    fi
    if [ ! -f "$eval_db" ]; then
        echo "ERROR: no usable features.parquet and missing eval DB: $eval_db" >&2
        return 1
    fi
    mkdir -p "$(dirname "$out")"
    exec 9>"$lock"
    if ! flock -w 7200 9; then
        echo "ERROR: timed out waiting for features.parquet build lock" >&2
        return 1
    fi
    if resolved="$("$PYTHON" -c "
import sys
sys.path.insert(0, '${SCRIPT_DIR}')
from pathlib import Path
from common import resolve_features_path
print(resolve_features_path(Path('${FEATURES_HINT}')))
" 2>/dev/null)"; then
        FEATURES="$resolved"
        return 0
    fi
    echo "Building $out from train=$train_db eval=$eval_db (bf16_final + bf16_eval) ..."
    "$PYTHON" "${SRC}/model/features.py" \
        --db "$train_db" \
        --eval-db "$eval_db" \
        --train-tags bf16_final \
        --eval-tag bf16_eval \
        --eval-min-configs 8000 \
        --dtype bf16 \
        --out "$out"
    FEATURES="$out"
}

ensure_features || exit 1

echo "=== baseline study ==="
echo "CKS_ROOT=$CKS_ROOT"
echo "FEATURES=$FEATURES"
echo "BASELINE_OUT=$BASELINE_OUT"
echo "BASELINE_STEPS=$BASELINE_STEPS"
echo "PYTHON=$PYTHON"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-local}"
echo

run_analysis() {
    echo "=== [1/4] training-only feature analysis ==="
    "$PYTHON" "$SCRIPT_DIR/feature_analysis.py" \
        --features "$FEATURES" \
        --outdir "$BASELINE_OUT"
}

run_train() {
    echo "=== [2/4] ridge linear baselines (full + structural) ==="
    "$PYTHON" "$SCRIPT_DIR/train_linear.py" \
        --features "$FEATURES" \
        --feature-set both \
        --outdir "$BASELINE_OUT"
}

run_analytical() {
    if [ ! -f "${BASELINE_OUT}/feature_correlations.csv" ]; then
        echo "ERROR: missing ${BASELINE_OUT}/feature_correlations.csv — run BASELINE_STEPS=analysis first" >&2
        exit 1
    fi
    echo "=== [3/4] hardware analytical score (agentic_analytical) ==="
    "$PYTHON" "$SCRIPT_DIR/agentic_analytical.py" \
        --features "$FEATURES" \
        --outdir "${BASELINE_OUT}/agentic_analytical" \
        --analysis-dir "$BASELINE_OUT"
}

run_summarize() {
    echo "=== [4/4] comparison table + figures + SUMMARY.md ==="
    for req in \
        "${BASELINE_OUT}/linear_full/metrics.json" \
        "${BASELINE_OUT}/linear_structural/metrics.json" \
        "${BASELINE_OUT}/agentic_analytical/metrics.json" \
        "${BASELINE_OUT}/analytical_additive/metrics.json" \
        "${BASELINE_OUT}/analytical_roofline/metrics.json"
    do
        if [ ! -f "$req" ]; then
            echo "ERROR: missing $req — run train + analytical first" >&2
            exit 1
        fi
    done
    "$PYTHON" "$SCRIPT_DIR/eval_random.py" --features "$FEATURES"
    "$PYTHON" "$SCRIPT_DIR/build_comparison_table.py"
    "$PYTHON" "$SCRIPT_DIR/plot_baselines.py"
}

case "$BASELINE_STEPS" in
    analysis) run_analysis ;;
    train) run_train ;;
    analytical) run_analytical ;;
    summarize) run_summarize ;;
    finish)
        run_analytical
        run_summarize
        ;;
    all)
        run_analysis
        run_train
        run_analytical
        run_summarize
        ;;
    *)
        echo "ERROR: BASELINE_STEPS=$BASELINE_STEPS (use analysis|train|analytical|summarize|finish|all)" >&2
        exit 1
        ;;
esac

echo "Baseline step(s) complete: BASELINE_STEPS=$BASELINE_STEPS"
