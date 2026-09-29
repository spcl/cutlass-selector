#!/usr/bin/env python3
"""Paper-quality plots for the MLP inference GEMM case study."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from repo_paths import BENCHMARKS, SRC

sys.path.insert(0, str(SRC))
sys.path.insert(0, str(BENCHMARKS))

from mlp_case_study.common import OUT_ROOT  # noqa: E402
from paper_figure_style import (  # noqa: E402
    FIG_H,
    FIG_W,
    PAPER_FIGURES,
    save_figure,
    setup_rc,
    style_axes,
)

PAPER_MODEL = "paper_mlp_mse_full"
COL_OURS = "#1e3a8a"
COL_NVMMH = "#14532d"
COL_FAIL = "#b91c1c"


def _model_label(model_id: str) -> str:
    if model_id == PAPER_MODEL:
        return "Paper MLP (full)"
    if model_id == "paper_mlp_mse_structural":
        return "Paper MLP (structural)"
    if model_id.startswith("capacity_"):
        parts = model_id.replace("capacity_", "").rsplit("_s", 1)
        return parts[0].replace("_", " ")
    return model_id


def plot_time_to_solution(tt: pd.DataFrame, out_dir: Path, paper: bool = True) -> None:
    sub = tt[tt.model == PAPER_MODEL] if paper else tt
    if sub.empty:
        return

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    style_axes(ax)
    for mode, ls in [("benchmark_only", "-"), ("compile_and_benchmark", "--")]:
        msub = sub[sub.nvmmh_setup_mode == mode].sort_values("N_inferences")
        if msub.empty:
            continue
        label = "nvMMH (bench only)" if mode == "benchmark_only" else "nvMMH (compile+bench)"
        ax.plot(
            msub["N_inferences"],
            msub["T_nvmmh_ms"] / msub["T_ours_ms"],
            marker="o",
            lw=2.5,
            ls=ls,
            color=COL_NVMMH,
            label=label,
        )
    ax.axhline(1.0, color="#525252", ls=":", lw=1)
    ax.set_xscale("log")
    ax.set_xlabel("MLP inferences (N)")
    ax.set_ylabel("Time-to-solution ratio (nvMMH / ours)")
    ax.set_xticks([1, 10, 100, 1000, 10000])
    ax.legend(loc="best")
    tag = "paper_mlp" if paper else "all_models_mean"
    save_figure(fig, out_dir, f"mlp_case_study_time_to_solution_{tag}")


def plot_steady_state(summary: pd.DataFrame, out_dir: Path) -> None:
    df = summary.copy()
    df["label"] = df["model"].map(_model_label)
    df = df.sort_values("steady_state_speedup", na_position="last")

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    style_axes(ax)
    colors = [COL_OURS if m == PAPER_MODEL else "#7d94ad" for m in df["model"]]
    y = df["steady_state_speedup"].fillna(0)
    ax.barh(df["label"], y, color=colors)
    ax.axvline(1.0, color="#525252", ls="--", lw=1)
    ax.set_xlabel("Steady-state speedup (nvMMH / ours)")
    ax.set_ylabel("")
    save_figure(fig, out_dir, "mlp_case_study_steady_state_by_model")


def plot_per_gemm_paper(weighted: pd.DataFrame, out_dir: Path) -> None:
    sub = weighted[weighted.model == PAPER_MODEL].copy()
    if sub.empty:
        return
    sub["layer_short"] = sub["layer"].str.replace("net.", "L")
    sub = sub.sort_values("K")

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    style_axes(ax)
    x = np.arange(len(sub))
    w = 0.35
    ours = sub["ours_mean_ms"].fillna(0)
    nv = sub["nvmmh_best_mean_ms"].fillna(0)
    ax.bar(x - w / 2, ours, w, label="Ours (rank-1)", color=COL_OURS)
    ax.bar(x + w / 2, nv, w, label="nvMMH (best of 8)", color=COL_NVMMH)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{r.layer_short}\n{r.M}×{r.N}×{r.K}" for _, r in sub.iterrows()],
        fontsize=14,
    )
    ax.set_ylabel("Mean latency (ms)")
    ax.set_xlabel("MLP layer GEMM")
    ax.legend(loc="best")
    save_figure(fig, out_dir, "mlp_case_study_per_gemm_paper_mlp")


def plot_coverage(per_gemm: pd.DataFrame, out_dir: Path) -> None:
    rows = []
    for model, sub in per_gemm.groupby("model"):
        rows.append({
            "model": model,
            "label": _model_label(model),
            "ours_success_rate": (sub.is_ours & sub.status.eq("success")).sum() / max(sub.is_ours.sum(), 1),
            "nvmmh_success_rate": (sub.is_nvmmh & sub.status.eq("success")).sum() / max(sub.is_nvmmh.sum(), 1),
        })
    df = pd.DataFrame(rows).sort_values("label")

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    style_axes(ax)
    x = np.arange(len(df))
    w = 0.35
    ax.bar(x - w / 2, 100 * df["ours_success_rate"], w, label="Ours", color=COL_OURS)
    ax.bar(x + w / 2, 100 * df["nvmmh_success_rate"], w, label="nvMMH", color=COL_NVMMH)
    ax.set_ylim(0, 105)
    ax.set_ylabel("Benchmark success rate (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(df["label"], rotation=35, ha="right", fontsize=12)
    ax.legend(loc="lower right")
    save_figure(fig, out_dir, "mlp_case_study_bench_coverage")


def plot_overhead(summary: pd.DataFrame, out_dir: Path) -> None:
    row = summary[summary.model == PAPER_MODEL]
    if row.empty:
        row = summary.iloc[[0]]
    r = row.iloc[0]
    labels = ["Our selector", "nvMMH recommend", "nvMMH bench×8", "nvMMH compile"]
    vals = [
        r["ours_selector_overhead_ms"],
        r["nvmmh_recommend_overhead_ms"],
        r["nvmmh_benchmark_8_overhead_ms"],
        r["nvmmh_compile_overhead_ms"],
    ]
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    style_axes(ax)
    colors = [COL_OURS, COL_NVMMH, COL_NVMMH, "#6d9078"]
    x = np.arange(len(labels))
    ax.bar(x, vals, color=colors)
    ax.set_ylabel("One-time overhead (ms)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    save_figure(fig, out_dir, "mlp_case_study_setup_overhead_paper")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=OUT_ROOT)
    ap.add_argument("--fig-dir", type=Path, default=PAPER_FIGURES)
    ap.add_argument("--also-quick", action="store_true")
    args = ap.parse_args()

    setup_rc()
    tt = pd.read_csv(args.out_dir / "time_to_solution.csv")
    weighted = pd.read_csv(args.out_dir / "mlp_weighted_results.csv")
    per_gemm = pd.read_csv(args.out_dir / "per_gemm_results.csv")
    summary = pd.read_csv(args.out_dir / "model_summary.csv")

    plot_time_to_solution(tt, args.fig_dir)
    plot_steady_state(summary, args.fig_dir)
    plot_per_gemm_paper(weighted, args.fig_dir)
    plot_coverage(per_gemm, args.fig_dir)
    plot_overhead(summary, args.fig_dir)

    if args.also_quick:
        quick = args.out_dir / "plots"
        quick.mkdir(parents=True, exist_ok=True)
        plot_time_to_solution(tt, quick)
        plot_steady_state(summary, quick)
        plot_per_gemm_paper(weighted, quick)
        plot_coverage(per_gemm, quick)
        plot_overhead(summary, quick)

    print(f"wrote figures to {args.fig_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
