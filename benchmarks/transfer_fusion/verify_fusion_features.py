#!/usr/bin/env python3
"""Verify fusion featurizer vs config_space_fusion (especially shared memory)."""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from repo_paths import BENCHMARKS, SRC

sys.path.insert(0, str(SRC / "model"))
from features_fusion import NUMERIC_FEATURES, featurize  # noqa: E402

sys.path.insert(0, str(SRC / "autotuner"))
from config_space_fusion import (  # noqa: E402
    FUSION_KIND_NAMES,
    estimate_smem_total,
)

sys.path.insert(0, str(SRC))
sys.path.insert(0, str(BENCHMARKS))
from transfer_fusion.study_common import DTYPE_TAGS, STUDY_ROOT, features_path, sweep_db_path  # noqa: E402


def _parse_fusion(name: str) -> str:
    if "_fusion_" not in name:
        return "linear"
    return name.split("_fusion_", 1)[1]


def _row(dtype: str, fusion: str = "bias_relu") -> dict:
    types = {
        "fp16": ("cutlass::half_t",) * 3,
        "fp32": ("float",) * 3,
        "fp8_e4m3": ("cutlass::float_e4m3_t",) * 3,
    }
    ta, tb, tc = types[dtype]
    return {
        "name": f"cfg_fusion_{fusion}",
        "M": 4096,
        "N": 4096,
        "K": 4096,
        "tile_m": 64,
        "tile_n": 128,
        "tile_k": 64,
        "stages": 4,
        "cluster_m": 1,
        "cluster_n": 1,
        "kernel_schedule": "cutlass::gemm::KernelTmaWarpSpecialized",
        "epilogue_schedule": "cutlass::epilogue::TmaWarpSpecialized",
        "scheduler": "cutlass::gemm::PersistentScheduler",
        "layout_a": "cutlass::layout::RowMajor",
        "layout_b": "cutlass::layout::ColumnMajor",
        "cutlass_type_a": ta,
        "cutlass_type_b": tb,
        "cutlass_type_c": tc,
        "fusion": fusion,
    }


def _expected_smem(row: dict) -> float:
    from features_fusion import _dtype_bytes  # noqa: E402

    ba = float(_dtype_bytes(pd.Series([row["cutlass_type_a"]])).iloc[0])
    bb = float(_dtype_bytes(pd.Series([row["cutlass_type_b"]])).iloc[0])
    bc = float(_dtype_bytes(pd.Series([row["cutlass_type_c"]])).iloc[0])
    fusion = row.get("fusion") or _parse_fusion(row["name"])
    return estimate_smem_total(
        row["tile_m"],
        row["tile_n"],
        row["tile_k"],
        row["stages"],
        ba,
        bb,
        bc,
        row["kernel_schedule"],
        row["epilogue_schedule"],
        fusion,
    )


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--autotuner", type=Path, default=Path.home() / "autotuner")
    ap.add_argument("--dtype", choices=list(DTYPE_TAGS) + ["all"], default="all")
    ap.add_argument("--sample", type=int, default=500, help="rows from sweep DB to cross-check")
    args = ap.parse_args()

    out = STUDY_ROOT / "verify_fusion_features.json"
    results: dict = {"checks": [], "ok": True}

    def check(name: str, got: float, expect: float, tol: float = 1e-3) -> None:
        ok = bool(abs(got - expect) <= tol * max(1.0, abs(expect)))
        results["checks"].append({"name": name, "got": float(got), "expect": float(expect), "ok": ok})
        if not ok:
            results["ok"] = False

    # Synthetic dtype ratio checks (operand bytes / AI / smem scale).
    fp16 = featurize(pd.DataFrame([_row("fp16")]))
    fp32 = featurize(pd.DataFrame([_row("fp32")]))
    fp8 = featurize(pd.DataFrame([_row("fp8_e4m3")]))
    check("fp32_smem_vs_fp16", float(fp32["smem_total"].iloc[0]) / float(fp16["smem_total"].iloc[0]), 2.0)
    check("fp8_smem_vs_fp16", float(fp8["smem_total"].iloc[0]) / float(fp16["smem_total"].iloc[0]), 0.5)

    # Cooperative path sums mainloop + epilogue (no union max), so bias carveout is visible.
    coop_linear = _row("fp16", "linear")
    coop_linear["kernel_schedule"] = "cutlass::gemm::KernelTmaWarpSpecializedCooperative"
    coop_linear["epilogue_schedule"] = "cutlass::epilogue::TmaWarpSpecializedCooperative"
    coop_bias = dict(coop_linear, fusion="bias_relu", name="cfg_fusion_bias_relu")
    lin_smem = float(featurize(pd.DataFrame([coop_linear]))["smem_total"].iloc[0])
    bias_smem = float(featurize(pd.DataFrame([coop_bias]))["smem_total"].iloc[0])
    check("bias_fusion_adds_smem_coop", bias_smem - lin_smem, 256.0)  # ceil128(tile_n*bytes_c), tn=128 fp16

    # config_space_fusion vs featurize on a grid of fusion kinds.
    for fusion in FUSION_KIND_NAMES:
        row = _row("fp16", fusion)
        got = float(featurize(pd.DataFrame([row]))["smem_total"].iloc[0])
        expect = _expected_smem(row)
        check(f"smem_featurize_vs_config_space_{fusion}", got, expect)

    dtypes = list(DTYPE_TAGS) if args.dtype == "all" else [args.dtype]
    for dtype in dtypes:
        db = sweep_db_path(dtype, args.autotuner)
        if not db.is_file():
            results["checks"].append({"name": f"skip_{dtype}_db", "ok": True, "note": f"missing {db}"})
            continue

        conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        tag = DTYPE_TAGS[dtype][0]
        df = pd.read_sql(
            """
            SELECT r.name, r.M, r.N, r.K, c.tile_m, c.tile_n, c.tile_k, c.stages,
                   c.cluster_m, c.cluster_n, c.kernel_schedule, c.epilogue_schedule,
                   c.scheduler, c.cutlass_type_a, c.cutlass_type_b, c.cutlass_type_c,
                   c.layout_a, c.layout_b
            FROM runs r JOIN configs c ON r.name=c.name
            WHERE r.status='success' AND r.tag=? AND r.mean_tflops > 0
            ORDER BY RANDOM() LIMIT ?
        """,
            conn,
            params=(tag, args.sample),
        )
        conn.close()
        if df.empty:
            continue
        df["fusion"] = df["name"].map(_parse_fusion)
        feat = featurize(df)
        rel_err = []
        for i, row in df.iterrows():
            got = float(feat.loc[i, "smem_total"])
            expect = _expected_smem({**row.to_dict(), "fusion": row["fusion"]})
            if expect > 0:
                rel_err.append(abs(got - expect) / expect)
        if rel_err:
            med = float(np.median(rel_err))
            ok = med < 1e-6
            results["checks"].append({
                "name": f"smem_median_rel_err_{dtype}",
                "median_rel_err": med,
                "n": len(rel_err),
                "ok": bool(ok),
            })
            if not ok:
                results["ok"] = False

        path = features_path(dtype)
        if path.is_file():
            pq = pd.read_parquet(path, columns=["smem_total", "smem_frac"] + NUMERIC_FEATURES[:4])
            bad = pq["smem_total"].isna().sum()
            results["checks"].append({
                "name": f"parquet_{dtype}_smem_finite",
                "ok": bool(bad == 0),
                "n_bad": int(bad),
            })
            if bad:
                results["ok"] = False

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))
    print(f"wrote {out}")
    return 0 if results["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
