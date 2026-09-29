#!/bin/bash
# Canonical repository paths for the shell drivers (mirrors src/repo_paths.py).
# Source, then call repo_paths_export [start_dir].

repo_paths_find_root() {
    local start="${1:-$(pwd)}"
    local p
    p="$(cd "$start" 2>/dev/null && pwd)" || p="$(pwd)"
    while [ "$p" != "/" ]; do
        if [ -f "$p/src/repo_paths.py" ] && [ -d "$p/benchmarks" ]; then
            echo "$p"
            return 0
        fi
        p="$(dirname "$p")"
    done
    echo "ERROR: repo root not found (expected src/ and benchmarks/ siblings)" >&2
    return 1
}

repo_paths_export() {
    local start="${1:-${BASH_SOURCE[1]:-${BASH_SOURCE[0]}}}"
    if [ -f "$start" ]; then
        start="$(dirname "$start")"
    fi
    REPO_ROOT="$(repo_paths_find_root "$start")" || return 1
    export REPO_ROOT
    export CKS_ROOT="${CKS_ROOT:-$REPO_ROOT}"
    export SRC="${SRC:-$REPO_ROOT/src}"
    export BENCHMARKS="${BENCHMARKS:-$REPO_ROOT/benchmarks}"
    export ARTIFACTS="${ARTIFACTS:-$REPO_ROOT/artifacts}"
    export DATASETS="${DATASETS:-$REPO_ROOT/datasets}"
    export ANALYSIS_DIR="${ANALYSIS_DIR:-$ARTIFACTS/analysis}"
    export PAPER_DIR="${PAPER_DIR:-$ANALYSIS_DIR/paper}"
    export CAPACITY_DIR="${CAPACITY_DIR:-$ANALYSIS_DIR/capacity}"
    export SAMPLING_DIR="${SAMPLING_DIR:-$ANALYSIS_DIR/sampling}"
    export EVAL_ARTIFACTS_DIR="${EVAL_ARTIFACTS_DIR:-$ARTIFACTS/eval}"
    export EVAL_OUT_DIR="${EVAL_OUT_DIR:-$EVAL_ARTIFACTS_DIR/out}"
    export PLANS_DIR="${PLANS_DIR:-$DATASETS/plans}"
    export DEEPBENCH_DIR="${DEEPBENCH_DIR:-$DATASETS/deepbench}"
    export CUTLASS_DIR="${CUTLASS_DIR:-$SRC/extern/cutlass}"
    export PYTHONPATH="${SRC}/model:${SRC}:${BENCHMARKS}:${PYTHONPATH:-}"
}
