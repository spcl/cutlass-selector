#!/usr/bin/env python3
"""Benchmark our rank-1 kernels and all nvMMH variants for traced MLP GEMMs."""

from __future__ import annotations

import argparse
import json
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

from repo_paths import BENCHMARKS, SRC

sys.path.insert(0, str(SRC))
sys.path.insert(0, str(BENCHMARKS))

from mlp_case_study.common import BUILD_ROOT, DB_PATH, OUT_ROOT  # noqa: E402


def _run_eval(python: str, phase: str, proposals: Path, db: Path, build_dir: Path,
              compile_jobs: int, bench_workers: int) -> float:
    cmd = [
        python, "-u", str(SRC / "eval" / "run.py"),
        "--phase", phase,
        "--db", str(db),
        "--build-dir", str(build_dir),
        "--proposals", str(proposals),
        "--compile-jobs", str(compile_jobs),
        "--bench-workers", str(bench_workers),
    ]
    print(" ".join(cmd), flush=True)
    t0 = time.perf_counter()
    subprocess.check_call(cmd)
    return time.perf_counter() - t0


def _load_results(db: Path) -> pd.DataFrame:
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    df = pd.read_sql_query(
        """SELECT method, M, N, K, layout, rank, variant, config_name,
                  raster_order, swizzle_size, splits, score, status,
                  mean_ms, std_ms, mean_tflops, std_tflops, error_text, cutlass_reason,
                  ran_at
           FROM eval_runs""",
        conn,
    )
    conn.close()
    return df


def _compile_wall_s(db: Path) -> float | None:
    """Estimate compile wall time from configs.compiled_at deltas (paper protocol)."""
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        rows = conn.execute(
            "SELECT compiled_at FROM configs WHERE compiled_at IS NOT NULL ORDER BY compiled_at"
        ).fetchall()
    except sqlite3.OperationalError:
        conn.close()
        return None
    conn.close()
    if len(rows) < 2:
        return None
    from datetime import datetime

    ts = [datetime.fromisoformat(r[0]) for r in rows]
    return (ts[-1] - ts[0]).total_seconds()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--proposals", type=Path, default=OUT_ROOT / "proposals_case_study.json")
    ap.add_argument("--db", type=Path, default=DB_PATH)
    ap.add_argument("--build-dir", type=Path, default=BUILD_ROOT)
    ap.add_argument("--out", type=Path, default=OUT_ROOT / "per_gemm_results.csv")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--compile-jobs", type=int, default=32)
    ap.add_argument("--bench-workers", type=int, default=1)
    ap.add_argument("--phase", default="plan,compile,bench",
                    help="src/eval/run.py phases (comma-separated)")
    ap.add_argument("--fresh-db", action="store_true",
                    help="delete existing DB before plan (default: resume)")
    args = ap.parse_args()

    args.build_dir.mkdir(parents=True, exist_ok=True)
    args.db.parent.mkdir(parents=True, exist_ok=True)

    if args.fresh_db and args.db.is_file():
        args.db.unlink()

    timings: dict[str, float] = {}
    timings["plan_compile_bench_total_s"] = _run_eval(
        args.python, args.phase,
        args.proposals, args.db, args.build_dir,
        args.compile_jobs, args.bench_workers,
    )
    timings["compile_wall_est_s"] = _compile_wall_s(args.db)

    df = _load_results(args.db)
    shapes = pd.read_csv(OUT_ROOT / "mlp_gemm_shapes.csv")
    pd.read_csv(OUT_ROOT / "selection_timing.csv")

    # Attach gemm_id via M,N,K,model
    key_cols = ["model", "M", "N", "K", "layout"]
    shape_keys = shapes[key_cols + ["gemm_id", "occurrence_count"]].drop_duplicates()
    df["layout_upper"] = df["layout"].str.upper()
    df["model"] = df["method"].str.replace(r"_(ours|nvmmh)$", "", regex=True)
    merged = df.merge(
        shape_keys.rename(columns={"layout": "layout_upper"}),
        on=["model", "M", "N", "K", "layout_upper"],
        how="left",
    )
    merged["is_ours"] = merged["method"].str.endswith("_ours")
    merged["is_nvmmh"] = merged["method"].str.endswith("_nvmmh")
    merged.to_csv(args.out, index=False)

    # Per-shape nvMMH 8-variant benchmark wall time from ran_at span.
    from datetime import datetime

    def _span_s(group: pd.DataFrame) -> float:
        ts = [datetime.fromisoformat(t) for t in group["ran_at"].dropna()]
        return (max(ts) - min(ts)).total_seconds() if len(ts) >= 2 else 0.0

    nv = merged[merged["is_nvmmh"]]
    bench_by_gemm = (
        nv.groupby(["model", "gemm_id"])
        .apply(
            lambda g: pd.Series({
                "nvmmh_n_benchmarked": int((g["status"] == "success").sum()),
                "nvmmh_fastest_mean_ms": g.loc[g["status"] == "success", "mean_ms"].min(),
                "nvmmh_median_of_variants_ms": g.loc[g["status"] == "success", "mean_ms"].median(),
                "nvmmh_bench_wall_s": _span_s(g),
            }),
            include_groups=False,
        )
        .reset_index()
    )
    bench_by_gemm.to_csv(OUT_ROOT / "nvmmh_bench_summary.csv", index=False)

    meta = {
        "timings": timings,
        "n_rows": len(merged),
        "n_success": int((merged["status"] == "success").sum()),
    }
    (OUT_ROOT / "benchmark_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"wrote {args.out} ({len(merged)} rows, {meta['n_success']} success)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
