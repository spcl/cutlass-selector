#!/usr/bin/env python3
"""Figures for baseline comparison."""

from __future__ import annotations

import sys

import matplotlib.pyplot as plt

from repo_paths import BENCHMARKS

sys.path.insert(0, str(BENCHMARKS))
sys.path.insert(0, str(BENCHMARKS / "training"))
from plot_paper_results import (  # noqa: E402
    FIG_PANEL,
    _apply_panel_fonts,
    _save,
    _setup_rc,
    _style,
    plot_baseline_mean_regret,
)

COL_MLP = "#1e3a8a"
COL_LINEAR = "#4b6cb7"
COL_LINEAR_LIGHT = "#9aafd4"
COL_REF = "#737373"
COL_RANDOM = "#a8a8a8"


def plot_why_ml_ablation() -> None:
    """Controlled ablation chain + reference baselines (not part of the chain)."""
    entries = [
        ("MLP", 6.2, COL_MLP),
        ("Ridge", 23.7, COL_LINEAR),
        ("Ridge(s)", 60.8, COL_LINEAR_LIGHT),
        ("nvMMH", 17.3, COL_REF),
        ("Analytical", 63.7, COL_REF),
        ("random", 71.5, COL_RANDOM),
    ]
    labels = [e[0] for e in entries]
    vals = [e[1] for e in entries]
    colors = [e[2] for e in entries]
    _setup_rc()
    fig, ax = plt.subplots(figsize=FIG_PANEL)
    _style(ax)
    ax.barh(labels, vals, color=colors, edgecolor="white", height=0.65)
    ax.set_xlabel("Selection Regret (%)")
    ax.invert_yaxis()
    ax.axhline(2.5, color="#333333", lw=0.8, ls="--", alpha=0.45)
    _apply_panel_fonts(ax)
    _save(fig, "eval_why_ml_ablation")


def main() -> int:
    _setup_rc()
    plot_why_ml_ablation()
    plot_baseline_mean_regret()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
