#!/usr/bin/env python3
"""Paper-quality plots for the dtype transfer study."""

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
from paper_figure_style import (  # noqa: E402
    FIG_H,
    FIG_TWIN,
    FIG_W,
    PAPER_FIGURES,
    finalize_axes,
    save_figure,
    setup_rc,
    style_axes,
)
from transfer_dtype.study_common import (  # noqa: E402
    CURVE_LABELS,
    CURVE_ORDER,
    PLOTS_ROOT,
    RESULTS_ROOT,
)

COL_NVMMH = "#525252"
COLORS = {
    "mlp_pretrain_full": "#1e3a8a",
    "mlp_pretrain_structural": "#7d94ad",
    "xgb_full": "#14532d",
    "xgb_structural": "#6d9078",
}
DTYPE_TITLES = {"fp32": "FP32 TN", "fp8_e4m3": "FP8 E4M3 TN"}
FRAC_PCT = [1, 5, 10, 25, 50, 100]

# Paper main figure: pretrain / warm-start only, full vs structural (no scratch).
PAPER_SPEEDUP_CURVES = [
    "mlp_pretrain_full",
    "mlp_pretrain_structural",
    "xgb_full",
    "xgb_structural",
]
PAPER_CURVE_LABELS = {
    "mlp_pretrain_full": "MLP (full)",
    "mlp_pretrain_structural": "MLP (structural)",
    "xgb_full": "XGBoost (full)",
    "xgb_structural": "XGBoost (structural)",
}


def _load_tables(curves: Path, runs: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not curves.is_file():
        raise SystemExit(f"missing {curves} — run summarize_study.py first")
    cdf = pd.read_csv(curves)
    rdf = pd.read_csv(runs) if runs.is_file() else pd.DataFrame()
    return cdf, rdf


def _filter_fraction(df: pd.DataFrame, min_fraction: float) -> pd.DataFrame:
    return df[df["fraction"] >= min_fraction].copy()


PROBLEM_KEYS = ["M", "N", "K", "layout"]


def _geomean(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x) & (x > 0)]
    return float(np.exp(np.mean(np.log(x)))) if len(x) else float("nan")


def _bootstrap_geomean_ci(
    ratios: np.ndarray,
    n_boot: int = 2000,
    ci: float = 0.95,
    seed: int = 0,
) -> tuple[float, float, float]:
    ratios = np.asarray(ratios, dtype=float)
    ratios = ratios[np.isfinite(ratios) & (ratios > 0)]
    if len(ratios) == 0:
        return float("nan"), float("nan"), float("nan")
    center = _geomean(ratios)
    if len(ratios) < 2:
        return center, center, center
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        sample = rng.choice(ratios, size=len(ratios), replace=True)
        boots[i] = np.exp(np.mean(np.log(sample)))
    alpha = (1.0 - ci) / 2.0
    lo, hi = np.percentile(boots, [100 * alpha, 100 * (1.0 - alpha)])
    return center, float(lo), float(hi)


def build_speedup_ci_table(
    per_problem: pd.DataFrame,
    runs: pd.DataFrame,
    curves: list[str] = PAPER_SPEEDUP_CURVES,
    n_boot: int = 2000,
    ci: float = 0.95,
) -> pd.DataFrame:
    """Bootstrap 95% CI on geomean speedup from per-eval-problem ratios."""
    meta = runs[["run_id", "dtype", "fraction", "curve_key"]].drop_duplicates()
    meta = meta[meta["curve_key"].isin(curves)]
    rows: list[dict] = []
    for dtype in per_problem["dtype"].dropna().unique():
        raw = per_problem[per_problem["dtype"] == dtype].copy()
        raw["mean_tflops"] = pd.to_numeric(raw["mean_tflops"], errors="coerce")
        nv = (
            raw[raw["method"] == "nvmmh"]
            .groupby(PROBLEM_KEYS, as_index=False)["mean_tflops"]
            .max()
            .rename(columns={"mean_tflops": "nv_tflops"})
        )
        for run_id in meta.loc[meta["dtype"] == dtype, "run_id"]:
            mp = raw[(raw["method"] == run_id) & (raw["rank"] == 1)].copy()
            if mp.empty:
                continue
            merged = mp.merge(nv, on=PROBLEM_KEYS, how="left")
            ok = merged["status"] == "success"
            both = ok & merged["nv_tflops"].notna() & (merged["nv_tflops"] > 0)
            both &= merged["mean_tflops"].notna() & (merged["mean_tflops"] > 0)
            ratios = (merged.loc[both, "mean_tflops"] / merged.loc[both, "nv_tflops"]).to_numpy()
            center, lo, hi = _bootstrap_geomean_ci(ratios, n_boot=n_boot, ci=ci, seed=hash(run_id) % 2**31)
            info = meta[(meta["run_id"] == run_id)].iloc[0]
            rows.append(
                {
                    "run_id": run_id,
                    "dtype": dtype,
                    "fraction": info["fraction"],
                    "curve_key": info["curve_key"],
                    "n_ratios": int(len(ratios)),
                    "geomean_speedup_vs_nvmmh": center,
                    "speedup_lo": lo,
                    "speedup_hi": hi,
                }
            )
    return pd.DataFrame(rows)


def _plot_curves_by_dtype(
    df: pd.DataFrame,
    metric: str,
    ylabel: str,
    curves: list[str],
    hline: float | None,
    title_suffix: str,
) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=FIG_TWIN, sharey=True)
    for ax, dtype in zip(axes, ("fp32", "fp8_e4m3")):
        style_axes(ax)
        sub = df[df["dtype"] == dtype]
        for curve in curves:
            csub = sub[sub["curve_key"] == curve].sort_values("fraction")
            if csub.empty:
                continue
            y = csub[metric]
            if y.isna().all():
                continue
            ax.plot(
                csub["fraction"] * 100,
                y,
                marker="o",
                ms=6,
                lw=2,
                label=CURVE_LABELS[curve],
                color=COLORS.get(curve),
            )
        if hline is not None:
            ax.axhline(hline, color=COL_NVMMH, ls="--", lw=1)
        ax.set_xlabel("Training data fraction (%)")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{DTYPE_TITLES[dtype]} — {title_suffix}")
        ax.set_xticks(FRAC_PCT)
        ax.grid(True, alpha=0.35)
    axes[1].legend(loc="best")
    finalize_axes(fig)
    return fig


def plot_speedup_single_dtype(
    df: pd.DataFrame,
    dtype: str,
    out_dir: Path,
    ci_df: pd.DataFrame | None = None,
) -> None:
    """One panel per dtype for the paper (4 curves, no grid, no title)."""
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    style_axes(ax)
    sub = df[df["dtype"] == dtype]
    ci_sub = ci_df[ci_df["dtype"] == dtype] if ci_df is not None and not ci_df.empty else None
    for curve in PAPER_SPEEDUP_CURVES:
        csub = sub[sub["curve_key"] == curve].sort_values("fraction")
        if csub.empty:
            continue
        y = csub["geomean_speedup_vs_nvmmh"]
        if y.isna().all():
            continue
        color = COLORS.get(curve)
        x = csub["fraction"] * 100
        if ci_sub is not None:
            band = ci_sub[ci_sub["curve_key"] == curve].sort_values("fraction")
            if not band.empty and band["speedup_lo"].notna().any():
                ax.fill_between(
                    band["fraction"] * 100,
                    band["speedup_lo"],
                    band["speedup_hi"],
                    color=color,
                    alpha=0.22,
                    linewidth=0,
                )
        ax.plot(
            x,
            y,
            marker="o",
            ms=8,
            lw=2.5,
            label=PAPER_CURVE_LABELS[curve],
            color=color,
            zorder=3,
        )
    ax.axhline(1.0, color=COL_NVMMH, ls="--", lw=1, zorder=1)
    ax.set_xlabel("Training data fraction (%)")
    ax.set_ylabel("Speedup vs nvMMH")
    ax.set_xticks(FRAC_PCT)
    ax.legend(loc="best")
    tag = "fp32" if dtype == "fp32" else "fp8"
    save_figure(fig, out_dir, f"transfer_speedup_{tag}_vs_training_fraction")


def plot_speedup(
    df: pd.DataFrame,
    out_dir: Path,
    ci_df: pd.DataFrame | None = None,
) -> None:
    plot_speedup_single_dtype(df, "fp32", out_dir, ci_df=ci_df)
    plot_speedup_single_dtype(df, "fp8_e4m3", out_dir, ci_df=ci_df)


def plot_win_rate(df: pd.DataFrame, out_dir: Path) -> None:
    fig = _plot_curves_by_dtype(
        df,
        "win_rate_vs_nvmmh",
        "Win rate vs. nvMMH",
        CURVE_ORDER,
        0.5,
        "win rate vs. nvMMH",
    )
    for ax in fig.axes:
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    save_figure(fig, out_dir, "transfer_win_rate_vs_training_fraction")


def plot_coverage(df: pd.DataFrame, out_dir: Path) -> None:
    fig = _plot_curves_by_dtype(
        df,
        "coverage",
        "Benchmark success rate",
        CURVE_ORDER,
        None,
        "eval coverage",
    )
    for ax in fig.axes:
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
        ax.set_ylim(0, 1.05)
    save_figure(fig, out_dir, "transfer_coverage_vs_training_fraction")


def plot_family_panels(df: pd.DataFrame, family: str, out_dir: Path) -> None:
    curves = [c for c in CURVE_ORDER if c.startswith(family)]
    label = "MLP" if family == "mlp" else "XGBoost"
    fig = _plot_curves_by_dtype(
        df,
        "geomean_speedup_vs_nvmmh",
        "Geometric-mean speedup vs. nvMMH",
        curves,
        1.0,
        f"{label} speedup",
    )
    save_figure(fig, out_dir, f"transfer_{family}_speedup_vs_training_fraction")


def plot_heatmap_100pct(runs: pd.DataFrame, out_dir: Path) -> None:
    sub = runs[runs["fraction"] == 1.0].copy()
    if sub.empty:
        return
    sub["label"] = sub["curve_key"].map(CURVE_LABELS)
    pivot = sub.pivot(index="label", columns="dtype", values="geomean_speedup_vs_nvmmh")
    pivot = pivot.reindex([CURVE_LABELS[c] for c in CURVE_ORDER if CURVE_LABELS[c] in pivot.index])
    pivot = pivot.rename(columns={"fp32": "FP32 TN", "fp8_e4m3": "FP8 E4M3 TN"})

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    style_axes(ax)
    data = pivot.to_numpy(dtype=float)
    im = ax.imshow(data, aspect="auto", cmap="YlGn", vmin=0.9, vmax=1.3)
    ax.set_xticks(range(pivot.shape[1]), pivot.columns)
    ax.set_yticks(range(pivot.shape[0]), pivot.index, fontsize=9)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i, j]
            if np.isfinite(val):
                ax.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=9)
    ax.set_title("Geometric-mean speedup vs. nvMMH at 100% training shapes")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Speedup")
    finalize_axes(fig)
    save_figure(fig, out_dir, "transfer_speedup_heatmap_100pct")


def plot_bar_key_fractions(runs: pd.DataFrame, out_dir: Path) -> None:
    key_curves = ["mlp_pretrain_full", "xgb_full", "mlp_pretrain_structural", "xgb_structural"]
    fracs = [0.05, 0.25, 1.0]
    sub = runs[(runs["curve_key"].isin(key_curves)) & (runs["fraction"].isin(fracs))].copy()
    if sub.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=FIG_TWIN, sharey=True)
    width = 0.18
    x = np.arange(len(fracs))
    for ax, dtype in zip(axes, ("fp32", "fp8_e4m3")):
        style_axes(ax)
        for i, curve in enumerate(key_curves):
            vals = []
            for frac in fracs:
                row = sub[(sub["dtype"] == dtype) & (sub["curve_key"] == curve) & (sub["fraction"] == frac)]
                vals.append(row["geomean_speedup_vs_nvmmh"].iloc[0] if len(row) else np.nan)
            ax.bar(
                x + (i - 1.5) * width,
                vals,
                width=width,
                label=CURVE_LABELS[curve],
                color=COLORS[curve],
            )
        ax.axhline(1.0, color=COL_NVMMH, ls="--", lw=1)
        ax.set_xticks(x, [f"{int(f * 100)}%" for f in fracs])
        ax.set_xlabel("Training data fraction")
        ax.set_ylabel("Geometric-mean speedup vs. nvMMH")
        ax.set_title(DTYPE_TITLES[dtype])
        ax.grid(True, axis="y", alpha=0.35)
    axes[1].legend(loc="best")
    finalize_axes(fig)
    save_figure(fig, out_dir, "transfer_speedup_bar_key_fractions")


def plot_all(
    curves_path: Path,
    runs_path: Path,
    out_dir: Path,
    min_fraction: float = 0.0,
    per_problem_path: Path | None = None,
    n_boot: int = 2000,
    ci_level: float = 0.95,
) -> None:
    setup_rc()
    curves, runs = _load_tables(curves_path, runs_path)
    curves = _filter_fraction(curves, min_fraction)
    if not runs.empty:
        runs = _filter_fraction(runs, min_fraction)

    ci_df = pd.DataFrame()
    pp_path = per_problem_path or (RESULTS_ROOT / "per_problem_raw.csv")
    if pp_path.is_file() and not runs.empty:
        per_problem = pd.read_csv(pp_path, low_memory=False)
        ci_df = build_speedup_ci_table(per_problem, runs, n_boot=n_boot, ci=ci_level)
        ci_df = _filter_fraction(ci_df, min_fraction)
        ci_out = RESULTS_ROOT / "speedup_bootstrap_ci.csv"
        ci_df.to_csv(ci_out, index=False)
        print(f"wrote {ci_out}")

    plot_speedup(curves, out_dir, ci_df=ci_df)
    plot_win_rate(curves, out_dir)
    plot_coverage(curves, out_dir)
    plot_family_panels(curves, "mlp", out_dir)
    plot_family_panels(curves, "xgb", out_dir)
    if not runs.empty:
        plot_heatmap_100pct(runs, out_dir)
        plot_bar_key_fractions(runs, out_dir)


def main() -> int:
    ap = argparse.ArgumentParser(description="Plot dtype transfer study results.")
    ap.add_argument("--curves", type=Path, default=RESULTS_ROOT / "curve_points.csv")
    ap.add_argument("--runs", type=Path, default=RESULTS_ROOT / "run_summary.csv")
    ap.add_argument("--out-dir", type=Path, default=PAPER_FIGURES)
    ap.add_argument("--also-quick", action="store_true", help="also write PNGs under transfer_study/plots/")
    ap.add_argument(
        "--min-fraction",
        type=float,
        default=0.0,
        help="drop curve points below this training fraction (e.g. 0.05 for paper)",
    )
    ap.add_argument("--per-problem", type=Path, default=RESULTS_ROOT / "per_problem_raw.csv")
    ap.add_argument("--n-boot", type=int, default=2000, help="bootstrap resamples for CI bands")
    ap.add_argument("--ci-level", type=float, default=0.95, help="confidence level for bands")
    args = ap.parse_args()

    plot_all(
        args.curves,
        args.runs,
        args.out_dir,
        min_fraction=args.min_fraction,
        per_problem_path=args.per_problem,
        n_boot=args.n_boot,
        ci_level=args.ci_level,
    )
    if args.also_quick:
        plot_all(
            args.curves,
            args.runs,
            PLOTS_ROOT,
            min_fraction=0.0,
            per_problem_path=args.per_problem,
            n_boot=args.n_boot,
            ci_level=args.ci_level,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
