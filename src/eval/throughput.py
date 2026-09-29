"""Throughput units for eval bench / report.

Eval databases store **GFLOP/s** in the legacy ``mean_tflops`` / ``std_tflops`` columns
(renaming the schema would break resume). Ratios and win rates are unchanged vs TFLOP/s.
"""

from __future__ import annotations

# GH200 BF16 tensor-core sustained peak (same 827 TFLOP/s as report/plot constants).
REPORTED_PEAK_GFLOPS = 827_000.0

FLOPS_UNIT_LABEL = "GFLOP/s"
FLOPS_UNIT_SHORT = "GF"


def calculate_gflops(M: int, N: int, K: int, avg_ms: float) -> float:
    """Sustained GEMM throughput in GFLOP/s."""
    return (2.0 * M * N * K) / (avg_ms / 1000.0) / 1e9


def tflops_to_gflops(tflops: float) -> float:
    """TFLOP/s to GFLOP/s."""
    return tflops * 1000.0
