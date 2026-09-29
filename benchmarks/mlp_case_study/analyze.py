#!/usr/bin/env python3
"""Analyze MLP case study results: weighted exec time and time-to-solution."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from repo_paths import BENCHMARKS, SRC

sys.path.insert(0, str(SRC))
sys.path.insert(0, str(BENCHMARKS))

from mlp_case_study.common import OUT_ROOT  # noqa: E402

N_GRID = [1, 10, 100, 1000, 10000]


def _fmt_be(n: float | None) -> str:
    if n is None or not np.isfinite(n):
        return "—"
    if n == float("inf"):
        return "never"
    return f"{n:.1f}"


def _fmt_speedup(x: float) -> str:
    return f"{x:.3f}×" if np.isfinite(x) else "—"


def _break_even_n(t_ours_setup: float, t_exec_ours: float,
                  t_nvmmh_setup: float, t_exec_nvmmh: float) -> float | None:
    """Solve T_ours(N) < T_nvmmh(N) for integer N."""
    delta_setup = t_nvmmh_setup - t_ours_setup
    delta_exec = t_exec_ours - t_exec_nvmmh
    if delta_exec >= 0:
        return None if delta_setup <= 0 else float("inf")
    n = delta_setup / (-delta_exec)
    return max(1.0, n)


def analyze_model(
    model_id: str,
    shapes: pd.DataFrame,
    per_gemm: pd.DataFrame,
    selection: pd.DataFrame,
    trace_timing: pd.DataFrame,
    bench_meta: dict,
) -> dict:
    sub_shapes = shapes[shapes.model == model_id]
    sub_sel = selection[selection.model == model_id]
    sub_per = per_gemm[per_gemm["model"] == model_id]

    ours = sub_per[sub_per["is_ours"] & sub_per["status"].eq("success")].copy()
    nv = sub_per[sub_per["is_nvmmh"] & sub_per["status"].eq("success")].copy()

    # Per-GEMM latencies (ms)
    ours_lat = ours.set_index("gemm_id")["mean_ms"]
    nv_best = nv.loc[nv.groupby("gemm_id")["mean_ms"].idxmin()].set_index("gemm_id")["mean_ms"]

    rows = []
    t_exec_ours = 0.0
    t_exec_nvmmh = 0.0
    for _, s in sub_shapes.iterrows():
        gid = s.gemm_id
        occ = int(s.occurrence_count)
        o_ms = float(ours_lat.get(gid, np.nan))
        n_ms = float(nv_best.get(gid, np.nan))
        rows.append({
            "model": model_id,
            "gemm_id": gid,
            "layer": s.layer,
            "M": s.M, "N": s.N, "K": s.K,
            "occurrence_count": occ,
            "ours_mean_ms": o_ms,
            "nvmmh_best_mean_ms": n_ms,
            "speedup": n_ms / o_ms if o_ms > 0 and np.isfinite(o_ms) and np.isfinite(n_ms) else np.nan,
            "ours_contrib_ms": occ * o_ms if np.isfinite(o_ms) else np.nan,
            "nvmmh_contrib_ms": occ * n_ms if np.isfinite(n_ms) else np.nan,
        })
        if np.isfinite(o_ms):
            t_exec_ours += occ * o_ms
        if np.isfinite(n_ms):
            t_exec_nvmmh += occ * n_ms

    weighted = pd.DataFrame(rows)
    exec_speedup = t_exec_nvmmh / t_exec_ours if t_exec_ours > 0 else np.nan

    # Selection overhead (one-time per unique GEMM shape)
    catalogue_s = float(sub_sel["ours_catalogue_s"].iloc[0]) if len(sub_sel) else 0.0
    t_select_ours = catalogue_s + float(sub_sel["ours_select_s"].sum())
    t_select_nvmmh = float(sub_sel["nvmmh_recommend_s"].sum())

    # nvMMH 8-variant measurement: per-gemm wall from benchmark if available.
    nv_bench_path = OUT_ROOT / "nvmmh_bench_summary.csv"
    if nv_bench_path.is_file():
        nv_bench = pd.read_csv(nv_bench_path)
        sub_nv = nv_bench[nv_bench.model == model_id]
        t_measure_8 = float(sub_nv["nvmmh_bench_wall_s"].sum())
    else:
        bench_total = float(bench_meta.get("timings", {}).get("plan_compile_bench_total_s", 0))
        n_nv_variants = int((sub_per["is_nvmmh"]).sum())
        n_all_nv = int(per_gemm["is_nvmmh"].sum()) or 1
        t_measure_8 = bench_total * (n_nv_variants / n_all_nv)

    compile_wall = bench_meta.get("timings", {}).get("compile_wall_est_s")
    n_nv_variants = int((sub_per["is_nvmmh"]).sum())
    n_all_nv = int(per_gemm["is_nvmmh"].sum()) or 1
    t_compile_nvmmh = (compile_wall or 0) * (n_nv_variants / n_all_nv) if compile_wall else None

    t_setup_ours = t_select_ours
    t_setup_nvmmh_bench_only = t_select_nvmmh + t_measure_8
    t_setup_nvmmh_compile_bench = t_setup_nvmmh_bench_only + (t_compile_nvmmh or 0)

    tt_solutions = []
    for N in N_GRID:
        t_ours = t_setup_ours + t_exec_ours * N
        for label, t_setup_nv in [
            ("benchmark_only", t_setup_nvmmh_bench_only),
            ("compile_and_benchmark", t_setup_nvmmh_compile_bench),
        ]:
            t_nv = t_setup_nv + t_exec_nvmmh * N
            tt_solutions.append({
                "model": model_id,
                "N_inferences": N,
                "nvmmh_setup_mode": label,
                "T_ours_ms": t_ours,
                "T_nvmmh_ms": t_nv,
                "speedup_nv_over_ours": t_nv / t_ours if t_ours > 0 else np.nan,
            })

    be = _break_even_n(
        t_setup_ours, t_exec_ours,
        t_setup_nvmmh_bench_only, t_exec_nvmmh,
    )

    trace_row = trace_timing[trace_timing.model == model_id]
    return {
        "model": model_id,
        "n_unique_gemms": len(sub_shapes),
        "n_gemm_calls_per_inference": int(sub_shapes["occurrence_count"].sum()),
        "n_candidates": int(trace_row["n_candidates"].iloc[0]) if len(trace_row) else None,
        "t_exec_ours_ms": t_exec_ours,
        "t_exec_nvmmh_ms": t_exec_nvmmh,
        "steady_state_speedup": exec_speedup,
        "ours_selector_overhead_ms": t_select_ours * 1000,
        "nvmmh_recommend_overhead_ms": t_select_nvmmh * 1000,
        "nvmmh_benchmark_8_overhead_ms": t_measure_8 * 1000,
        "nvmmh_compile_overhead_ms": (t_compile_nvmmh or 0) * 1000,
        "break_even_n_benchmark_only": be,
        "weighted_df": weighted,
        "tt_solutions": pd.DataFrame(tt_solutions),
    }


def write_summary(results: list[dict], out_dir: Path) -> None:
    lines = [
        "# MLP inference GEMM case study",
        "",
        "Traces internal Linear GEMMs from real selector MLP forward passes, replays each "
        "unique shape through the paper CUTLASS eval harness (bf16 TN), and compares our "
        "learned rank-1 pick vs nvMMH best-of-8 measured variants.",
        "",
    ]
    for r in results:
        lines += [
            f"## {r['model']}",
            "",
            f"- Unique GEMM shapes: **{r['n_unique_gemms']}**",
            f"- GEMM calls per inference: **{r['n_gemm_calls_per_inference']}**",
            f"- Candidates per selector call: **{r['n_candidates']}**",
            f"- Occurrence-weighted exec (ours): **{r['t_exec_ours_ms']:.3f} ms**",
            f"- Occurrence-weighted exec (nvMMH best-of-8): **{r['t_exec_nvmmh_ms']:.3f} ms**",
            f"- Steady-state speedup (nvMMH/ours): **{_fmt_speedup(r['steady_state_speedup'])}**",
            f"- Our selector overhead: **{r['ours_selector_overhead_ms']:.1f} ms**",
            f"- nvMMH recommend overhead: **{r['nvmmh_recommend_overhead_ms']:.1f} ms**",
            f"- nvMMH 8-variant benchmark overhead (est.): **{r['nvmmh_benchmark_8_overhead_ms']:.1f} ms**",
            f"- nvMMH compile overhead (est.): **{r['nvmmh_compile_overhead_ms']:.1f} ms**",
            f"- Break-even N (benchmark-only setup): **{_fmt_be(r['break_even_n_benchmark_only'])}**",
            "",
            "### Time-to-solution speedup (nvMMH / ours)",
            "",
            "| N | benchmark-only setup | compile+benchmark setup |",
            "|---:|---:|---:|",
        ]
        tt = r["tt_solutions"]
        for N in N_GRID:
            b = tt[(tt.N_inferences == N) & (tt.nvmmh_setup_mode == "benchmark_only")]["speedup_nv_over_ours"].iloc[0]
            c = tt[(tt.N_inferences == N) & (tt.nvmmh_setup_mode == "compile_and_benchmark")]["speedup_nv_over_ours"].iloc[0]
            lines.append(f"| {N} | {b:.3f}× | {c:.3f}× |")
        lines.append("")

    (out_dir / "SUMMARY.md").write_text("\n".join(lines))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=OUT_ROOT)
    args = ap.parse_args()

    shapes = pd.read_csv(args.out_dir / "mlp_gemm_shapes.csv")
    per_gemm = pd.read_csv(args.out_dir / "per_gemm_results.csv")
    selection = pd.read_csv(args.out_dir / "selection_timing.csv")
    trace_timing = pd.read_csv(args.out_dir / "trace_timing.csv")
    bench_meta = json.loads((args.out_dir / "benchmark_meta.json").read_text())

    per_gemm["is_ours"] = per_gemm["method"].str.endswith("_ours")
    per_gemm["is_nvmmh"] = per_gemm["method"].str.endswith("_nvmmh")

    results: list[dict] = []
    summary_rows: list[dict] = []
    weighted_frames: list[pd.DataFrame] = []
    tt_frames: list[pd.DataFrame] = []

    for model_id in sorted(shapes.model.unique()):
        r = analyze_model(model_id, shapes, per_gemm, selection, trace_timing, bench_meta)
        results.append(r)
        summary_rows.append({
            k: v for k, v in r.items()
            if k not in ("weighted_df", "tt_solutions")
        })
        weighted_frames.append(r["weighted_df"])
        tt_frames.append(r["tt_solutions"])

    pd.concat(weighted_frames, ignore_index=True).to_csv(
        args.out_dir / "mlp_weighted_results.csv", index=False
    )
    pd.concat(tt_frames, ignore_index=True).to_csv(
        args.out_dir / "time_to_solution.csv", index=False
    )
    pd.DataFrame(summary_rows).to_csv(args.out_dir / "model_summary.csv", index=False)
    write_summary(results, args.out_dir)
    print(f"wrote {args.out_dir / 'SUMMARY.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
