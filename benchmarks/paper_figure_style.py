"""Shared dimensions and typography for paper figures (eval + transfer)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from repo_paths import FIGURES

PAPER_FIGURES = FIGURES

BG = "#ebebeb"
# All paper panels export at this exact size (inches).
FIG_W, FIG_H = 7.2, 5.6
FIG_TWIN = (FIG_W * 2, FIG_H)  # optional wide layouts only
PAPER_FONT = 20

# Identical on every export so LaTeX scaling is predictable (no bbox_inches='tight').
# Extra left margin for log-scale tick labels + 20pt y-axis titles.
FIG_MARGINS = {"left": 0.16, "right": 0.97, "bottom": 0.16, "top": 0.95}


def setup_rc(font_size: int = PAPER_FONT) -> None:
    plt.rcParams.update(
        {
            "font.size": font_size,
            "axes.labelsize": font_size,
            "legend.fontsize": font_size,
            "xtick.labelsize": font_size,
            "ytick.labelsize": font_size,
        }
    )


def style_axes(ax, grid: bool = False) -> None:
    ax.set_facecolor(BG)
    ax.figure.patch.set_facecolor("white")
    if grid:
        ax.grid(True, which="both", color="white", linewidth=0.6, alpha=0.4)
    else:
        ax.grid(False)


def finalize_axes(fig: plt.Figure) -> None:
    fig.subplots_adjust(**FIG_MARGINS)


def save_figure(fig: plt.Figure, out_dir: Path, name: str, dpi: int = 200) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    finalize_axes(fig)
    fig.savefig(out_dir / f"{name}.pdf", dpi=dpi)
    fig.savefig(out_dir / f"{name}.png", dpi=dpi)
    plt.close(fig)
    print(f"wrote {out_dir / name}.pdf")
