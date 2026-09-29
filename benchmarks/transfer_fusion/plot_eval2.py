#!/usr/bin/env python3
"""Paper-quality plots for fusion transfer Eval2 (zero-shot fusion kinds)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from repo_paths import BENCHMARKS, SRC

sys.path.insert(0, str(SRC))
sys.path.insert(0, str(BENCHMARKS))
from paper_figure_style import (  # noqa: E402
    FIG_H,
    FIG_W,
    PAPER_FIGURES,
    save_figure,
    setup_rc,
    style_axes,
)
from transfer_fusion.plot_study import (  # noqa: E402
    COL_NVMMH,
    COLORS,
    PAPER_CURVE_LABELS,
    PAPER_SPEEDUP_CURVES,
)
from transfer_fusion.study_common import (  # noqa: E402
    PLOTS_ROOT,
    results_root,
)

SUITE = "eval2"
EVAL2_RESULTS = results_root(SUITE)
ZERO_SHOT_KINDS = ("silu", "bias_silu", "tanh", "bias_tanh")
KIND_LABELS = {
    "silu": "SiLU",
    "bias_silu": "Bias+SiLU",
    "tanh": "Tanh",
    "bias_tanh": "Bias+Tanh",
}


def plot_overall_speedup(curves: pd.DataFrame, out_dir: Path) -> None:
    overall = curves[curves["fusion_kind"] == "all"].copy()
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    style_axes(ax)
    from transfer_fusion.plot_study import FRAC_PCT as _FRAC_PCT  # noqa: PLC0415

    sub = overall[overall["dtype"] == "fp16"]
    for curve in PAPER_SPEEDUP_CURVES:
        csub = sub[sub["curve_key"] == curve].sort_values("fraction")
        if csub.empty:
            continue
        y = csub["geomean_speedup_vs_nvmmh"]
        if y.isna().all():
            continue
        ax.plot(
            csub["fraction"] * 100,
            y,
            marker="o",
            ms=8,
            lw=2.5,
            label=PAPER_CURVE_LABELS[curve],
            color=COLORS.get(curve),
            zorder=3,
        )
    ax.axhline(1.0, color=COL_NVMMH, ls="--", lw=1, zorder=1)
    ax.set_xlabel("Training data fraction (%)")
    ax.set_ylabel("Speedup vs nvMMH")
    ax.set_xticks(_FRAC_PCT)
    ax.legend(loc="best")
    save_figure(fig, out_dir, "transfer_fusion_eval2_speedup_fp16_vs_training_fraction")


def plot_kind_bars_at_fraction(
    curves: pd.DataFrame,
    out_dir: Path,
    fraction: float = 1.0,
    curve_key: str = "mlp_pretrain_full",
) -> None:
    sub = curves[
        (curves["dtype"] == "fp16")
        & (curves["fraction"] == fraction)
        & (curves["curve_key"] == curve_key)
        & curves["fusion_kind"].isin(ZERO_SHOT_KINDS)
    ].copy()
    if sub.empty:
        return

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    style_axes(ax)
    sub = sub.set_index("fusion_kind").reindex(ZERO_SHOT_KINDS)
    x = range(len(ZERO_SHOT_KINDS))
    y = sub["geomean_speedup_vs_nvmmh"].to_numpy()
    ax.bar(x, y, color=COLORS.get(curve_key, "#1e3a8a"), width=0.65)
    ax.axhline(1.0, color=COL_NVMMH, ls="--", lw=1)
    ax.set_xticks(list(x))
    ax.set_xticklabels([KIND_LABELS[k] for k in ZERO_SHOT_KINDS], rotation=15, ha="right")
    ax.set_ylabel("Speedup vs nvMMH")
    ax.set_xlabel("Zero-shot fusion kind")
    pct = int(round(fraction * 100))
    label = PAPER_CURVE_LABELS.get(curve_key, curve_key)
    ax.set_title(f"{label} at {pct}% fusion training coverage")
    tag = curve_key.replace("_pretrain", "").replace("_full", "")
    save_figure(fig, out_dir, f"transfer_fusion_eval2_speedup_by_kind_fp16_f{pct:03d}_{tag}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Plot fusion transfer Eval2 results.")
    ap.add_argument("--curves", type=Path, default=EVAL2_RESULTS / "curve_points.csv")
    ap.add_argument("--out-dir", type=Path, default=PAPER_FIGURES)
    ap.add_argument("--also-quick", action="store_true")
    args = ap.parse_args()

    if not args.curves.is_file():
        raise SystemExit(f"missing {args.curves} — run summarize_eval2.py first")

    setup_rc()
    curves = pd.read_csv(args.curves)
    plot_overall_speedup(curves, args.out_dir)
    plot_kind_bars_at_fraction(curves, args.out_dir, fraction=1.0, curve_key="mlp_pretrain_full")
    plot_kind_bars_at_fraction(curves, args.out_dir, fraction=1.0, curve_key="xgb_full")

    if args.also_quick:
        quick = PLOTS_ROOT / "eval2"
        quick.mkdir(parents=True, exist_ok=True)
        plot_overall_speedup(curves, quick)
        plot_kind_bars_at_fraction(curves, quick, fraction=1.0, curve_key="mlp_pretrain_full")
        plot_kind_bars_at_fraction(curves, quick, fraction=1.0, curve_key="xgb_full")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
