#!/usr/bin/env python3
"""Verify dtype-dependent hardware-aware features for FP32 and FP8."""

from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

from repo_paths import BENCHMARKS, SRC

sys.path.insert(0, str(SRC / "model"))
from features import NUMERIC_FEATURES, featurize  # noqa: E402

sys.path.insert(0, str(SRC))
sys.path.insert(0, str(BENCHMARKS))
from transfer_dtype.study_common import STUDY_ROOT, features_path  # noqa: E402


def _row(dtype: str) -> dict:
    types = {
        "fp32": ("float", "float", "float"),
        "fp8_e4m3": ("cutlass::float_e4m3_t",) * 3,
    }
    ta, tb, tc = types[dtype]
    return {
        "M": 4096, "N": 4096, "K": 4096,
        "tile_m": 64, "tile_n": 128, "tile_k": 64,
        "stages": 4, "cluster_m": 1, "cluster_n": 1,
        "kernel_schedule": "cutlass::gemm::KernelTmaWarpSpecialized",
        "epilogue_schedule": "cutlass::epilogue::TmaWarpSpecialized",
        "scheduler": "cutlass::gemm::PersistentScheduler",
        "layout_a": "cutlass::layout::RowMajor",
        "layout_b": "cutlass::layout::ColumnMajor",
        "cutlass_type_a": ta, "cutlass_type_b": tb, "cutlass_type_c": tc,
    }


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--dtype", choices=["fp32", "fp8_e4m3", "all"], default="all")
    args = ap.parse_args()

    out = STUDY_ROOT / "verify_dtype_features.json"
    results: dict = {"checks": [], "ok": True, "dtype_filter": args.dtype}

    # Synthetic ratio checks (same as unit tests).
    bf16 = featurize(pd.DataFrame([{
        **_row("fp32"),
        "cutlass_type_a": "cutlass::bfloat16_t",
        "cutlass_type_b": "cutlass::bfloat16_t",
        "cutlass_type_c": "cutlass::bfloat16_t",
    }]))
    fp32 = featurize(pd.DataFrame([_row("fp32")]))
    fp8 = featurize(pd.DataFrame([_row("fp8_e4m3")]))

    def check(name: str, got: float, expect: float, tol: float = 1e-6) -> None:
        ok = abs(got - expect) <= tol * max(1.0, abs(expect))
        results["checks"].append({"name": name, "got": got, "expect": expect, "ok": ok})
        if not ok:
            results["ok"] = False

    b_smem = float(bf16["smem_total"].iloc[0])
    f_smem = float(fp32["smem_total"].iloc[0])
    e_smem = float(fp8["smem_total"].iloc[0])
    check("fp32_smem_vs_bf16", f_smem / b_smem, 2.0)
    check("fp8_smem_vs_bf16", e_smem / b_smem, 0.5)

    b_ai = float(bf16["problem_arith_intensity"].iloc[0])
    f_ai = float(fp32["problem_arith_intensity"].iloc[0])
    e_ai = float(fp8["problem_arith_intensity"].iloc[0])
    check("bf16_ai_vs_fp32", b_ai / f_ai, 2.0)
    check("fp8_ai_vs_fp32", e_ai / f_ai, 4.0)

    dtypes = ("fp32", "fp8_e4m3") if args.dtype == "all" else (args.dtype,)
    for dtype in dtypes:
        path = features_path(dtype)
        if not path.is_file():
            results["checks"].append({"name": f"missing_{dtype}", "ok": False})
            results["ok"] = False
            continue
        df = pd.read_parquet(path, columns=NUMERIC_FEATURES)
        bad = ~np.isfinite(df.to_numpy(dtype=float))
        ok = not bad.any()
        results["checks"].append({
            "name": f"parquet_finite_{dtype}",
            "rows": len(df),
            "ok": ok,
        })
        if not ok:
            results["ok"] = False

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))
    return 0 if results["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
