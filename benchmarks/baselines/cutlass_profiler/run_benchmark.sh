#!/bin/bash
# Run cutlass_profiler using heuristic-selected kernels from build/testlist.csv.
# Requires: build/tools/profiler/cutlass_profiler and build/testlist.csv (run build.sh first).
# Results written to results/cutlass_reference.gemm.csv.
# Run from the repository root:  bash benchmarks/baselines/cutlass_profiler/run_benchmark.sh

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
exec python3 "$SCRIPT_DIR/run_benchmark.py" "$@"
