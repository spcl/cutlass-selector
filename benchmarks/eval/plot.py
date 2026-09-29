#!/usr/bin/env python3
"""
plot.py — Figures for the evaluation, from report.csv only (never the database).

  budget_curve.pdf   what a measurement budget buys. x = kernels compiled and benchmarked
                     per problem, y = geometric mean of method@b / cuBLASLt best-of-8. The
                     reference line at 1.0 is cuBLASLt's own empirical best, so a point at
                     0.9 reads "90% of what an exhaustive cuBLASLt search achieves".
  roofline.pdf       % of peak vs cbrt(M·N·K). Series are labelled with
                     the budget each one spends, because they differ: ours and cuBLASLt are
                     single picks, nvMMH is the best of its 8 schedule variants.

Colour follows the entity in both figures: cuBLASLt blue, nvMMH
orange, ours aqua — categorical slots 1-3, assigned in fixed order.

    python benchmarks/eval/plot.py --csv artifacts/eval/out/report.csv --out-dir artifacts/eval/out
"""

import argparse
import sys
from pathlib import Path

import matplotlib

from repo_paths import EVAL_OUT, SRC

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(SRC / "eval"))

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, RULE, PAPER, GRID = "#0b0b0b", "#52514e", "#c3c2b7", "#fcfcfb", "#e5e4df"

from throughput import REPORTED_PEAK_GFLOPS  # noqa: E402

MAX_BUDGET = 8
N_BINS = 18
MIN_PER_BIN = 12

# (series prefix, label, colour) — fixed order, colour bound to the method.
METHODS = [
    ("cublas", "cuBLASLt", BLUE),
    ("nvmmh", "nvMMH", ORANGE),
    ("ours", "Ours", AQUA),
]


def _style(ax) -> None:
    ax.set_facecolor(PAPER)
    ax.tick_params(colors=MUTED)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(RULE)
    ax.grid(True, which="major", axis="y", color=GRID, linewidth=0.5, zorder=0)


def _geomean(x: np.ndarray) -> float:
    x = x[np.isfinite(x) & (x > 0)]
    return float(np.exp(np.mean(np.log(x)))) if len(x) else np.nan


def draw_budget_curve(ax, df: pd.DataFrame, reference: str) -> None:
    budgets = np.arange(1, MAX_BUDGET + 1)
    for prefix, label, colour in METHODS:
        cols = [f"{prefix}@{b}" for b in budgets]
        if not all(c in df for c in cols) or reference not in df:
            continue
        ys, ns = [], []
        for c in cols:
            sub = df[(df[c] > 0) & (df[reference] > 0)]
            ys.append(_geomean((sub[c] / sub[reference]).to_numpy()))
            ns.append(len(sub))
        ax.plot(budgets, ys, color=colour, linewidth=2.0, marker="o", markersize=5,
                markeredgecolor=PAPER, markeredgewidth=1.2, zorder=4,
                label=f"{label}  (n={max(ns):,})")
        ax.annotate(label, (budgets[-1], ys[-1]), textcoords="offset points",
                    xytext=(7, 0), va="center", color=colour, fontsize=8.5, zorder=5)

    ax.axhline(1.0, color=MUTED, linewidth=1.0, linestyle=(0, (4, 3)), zorder=2)
    _style(ax)
    ax.set_xlim(0.8, MAX_BUDGET + 2.0)
    ax.set_xticks(budgets)
    ax.set_xlabel("kernels compiled and benchmarked per problem", color=INK)
    ax.set_ylabel("geometric mean of achieved / cuBLASLt best-of-8", color=INK)
    ax.legend(frameon=False, loc="lower right", fontsize=8.5)


def _binned(x: pd.Series, y: pd.Series, edges: np.ndarray):
    idx = np.digitize(x, edges) - 1
    cx, lo, mid, hi = [], [], [], []
    for b in range(len(edges) - 1):
        sel = (idx == b) & y.notna().to_numpy() & (y > 0).to_numpy()
        if sel.sum() < MIN_PER_BIN:
            continue
        v = y[sel]
        cx.append(np.sqrt(edges[b] * edges[b + 1]))
        lo.append(v.quantile(0.25))
        mid.append(v.median())
        hi.append(v.quantile(0.75))
    return map(np.array, (cx, lo, mid, hi))


def draw_roofline(ax, df: pd.DataFrame, series, peak: float) -> None:
    edges = np.logspace(np.log10(df["cbrt_mnk"].min()), np.log10(df["cbrt_mnk"].max()), N_BINS + 1)
    for col, label, colour in series:
        if col not in df:
            continue
        pct = df[col] / peak * 100
        sub = df[(df[col] > 0)]
        ax.scatter(sub["cbrt_mnk"], sub[col] / peak * 100, s=2.5, alpha=0.12, color=colour,
                   edgecolors="none", zorder=2, rasterized=True)
        cx, lo, mid, hi = _binned(df["cbrt_mnk"], pct, edges)
        ax.fill_between(cx, lo, hi, color=colour, alpha=0.13, zorder=3, linewidth=0)
        ax.plot(cx, mid, color=colour, linewidth=2.0, zorder=4, label=f"{label}  (n={len(sub):,})")

    _style(ax)
    ax.set_xscale("log")
    ax.set_ylim(0, 100)
    ax.set_xlabel("$\\sqrt[3]{M \\cdot N \\cdot K}$  (log scale)", color=INK)
    ax.set_ylabel("% of peak BF16 tensor-core throughput", color=INK)
    ax.legend(frameon=False, loc="upper left", fontsize=8.5)


def save(fig, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, facecolor=fig.get_facecolor(), bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"wrote {path}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Evaluation figures")
    ap.add_argument("--csv", type=Path, default=EVAL_OUT / "report.csv")
    ap.add_argument("--out-dir", type=Path, default=EVAL_OUT)
    ap.add_argument("--peak", type=float, default=REPORTED_PEAK_GFLOPS,
                    help="BF16 tensor-core peak (GFLOP/s) for the %%-of-peak axis")
    ap.add_argument("--reference", default="cublas@8",
                    help="denominator series for the budget curve")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    fig.patch.set_facecolor(PAPER)
    draw_budget_curve(ax, df, args.reference)
    save(fig, args.out_dir / "budget_curve.pdf")

    # Each series labelled with the budget it spends: nvMMH ranks no schedules, so its
    # unit is the best of all 8 variants of its single recommendation.
    series = [
        ("cublas@1", "cuBLASLt (heuristic top-1)", BLUE),
        ("nvmmh@8", "nvMMH (best of 8 variants)", ORANGE),
        ("ours@1", "Ours (model top-1)", AQUA),
    ]
    fig, ax = plt.subplots(figsize=(7.4, 5.4))
    fig.patch.set_facecolor(PAPER)
    draw_roofline(ax, df, series, args.peak)
    save(fig, args.out_dir / "roofline.pdf")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
