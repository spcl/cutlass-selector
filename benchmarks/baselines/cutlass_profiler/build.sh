#!/bin/bash
# Build cutlass_profiler for SM90a using CUTLASS heuristics to select kernels.
# The heuristics analytically pick the best N configs per problem from problems.json,
# then only those kernels are compiled — much smaller build than a wildcard filter.
# Run once from baseline/cutlass/:  bash build.sh
# Requires: cmake, nvcc on PATH, nvidia-matmul-heuristics (ships with CUTLASS), extern/cutlass/ at project root.


SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CUTLASS_DIR="$(cd "$SCRIPT_DIR/../../../src/extern/cutlass" && pwd)"
BUILD_DIR="$SCRIPT_DIR/build"

mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# Use the active venv Python if available, otherwise fall back to system python3
PYTHON_BIN="$(command -v python3)"
if [[ -n "${VIRTUAL_ENV:-}" ]]; then
    PYTHON_BIN="$VIRTUAL_ENV/bin/python"
fi

cmake "$CUTLASS_DIR" \
    -DPython3_EXECUTABLE="$PYTHON_BIN" \
    -DCUTLASS_NVCC_ARCHS=90a \
    -DCUTLASS_UNITY_BUILD_ENABLED=OFF \
    -DCUTLASS_ENABLE_TESTS=OFF \
    -DCUTLASS_ENABLE_EXAMPLES=OFF \
    -DCUTLASS_LIBRARY_HEURISTICS_PROBLEMS_FILE="$SCRIPT_DIR/problems.json" \
    -DCUTLASS_LIBRARY_HEURISTICS_CONFIGS_PER_PROBLEM=5 \
    -DCUTLASS_LIBRARY_HEURISTICS_GPU=auto \
    -DCUTLASS_LIBRARY_HEURISTICS_RESTRICT_KERNELS=OFF \
    -DCUTLASS_LIBRARY_HEURISTICS_TESTLIST_FILE="$BUILD_DIR/testlist.csv" \
    -DCUTLASS_LIBRARY_IGNORE_KERNELS="*e4m3*ttn*"

make cutlass_profiler -j"$(( $(nproc) / 2 ))"

echo "Built: $BUILD_DIR/tools/profiler/cutlass_profiler"
echo "Testlist: $BUILD_DIR/testlist.csv"
