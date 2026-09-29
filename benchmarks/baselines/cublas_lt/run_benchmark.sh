#!/bin/bash
# cuBLASLt reference benchmark — BF16 and FP8 E4M3 configs across all 19 autotuner shapes.
# Expects cublas_profiler to be built (make -C src/baseline/cublas_lt).
# Run from the repository root:  bash benchmarks/baselines/cublas_lt/run_benchmark.sh

PROFILER="${PROFILER:-src/baseline/cublas_lt/cublas_profiler}"
OUT="${OUT:-artifacts/analysis/paper/baselines/cublas_reference.csv}"
mkdir -p "$(dirname "$OUT")"

SHAPES=(
    32    128   4096
    64    64    64
    64    12288 4096
    128   128   128
    256   256   256
    256   256   8192
    256   4096  4096
    256   12288 4096
    512   512   512
    512   3072  768
    2048  128   4096
    2048  2048  128
    2048  2048  2048
    4096  128   4096
    4096  4096  4096
    4096  11008 4096
    8192  8192  8192
    12288 128   4096
    12288 12288 12288
)

"$PROFILER" --header > "$OUT"

for config in bf16_f32_bf16 e4m3_e4m3_f32_e4m3; do
    echo "Config: $config"
    "$PROFILER" "$config" "${SHAPES[@]}" >> "$OUT"
done

echo "Done. Results in $OUT"
