#!/usr/bin/env python3
"""Plot and summarize a DeepBench GPU eval run (eval/deepbench_<job_id>/)."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from repo_paths import BENCHMARKS, EVAL_ARTIFACTS, FIGURES, SRC

sys.path.insert(0, str(BENCHMARKS / "deepbench"))
sys.path.insert(0, str(BENCHMARKS))
from capacity_models import MLP_WIDTHS, PAPER_MLP_WIDTH, XGB_DEPTHS  # noqa: E402

sys.path.insert(0, str(BENCHMARKS / "training"))
from plot_paper_results import (  # noqa: E402
    _ERR_KW,
    COL_LINEAR,
    COL_LINEAR_LIGHT,
    COL_MLP,
    COL_MLP_LIGHT,
    COL_NVMMH,
    COL_RANDOM,
    COL_XGB,
    COL_XGB_LIGHT,
    FIG_PANEL,
    _apply_panel_fonts,
    _save,
    _setup_rc,
    _style,
)

PAPER_METHODS = [
    ("mlp_full", "MLP", COL_MLP),
    ("mlp_structural", "MLP(s)", COL_MLP_LIGHT),
    ("xgb_full", "XGB", COL_XGB),
    ("xgb_structural", "XGB(s)", COL_XGB_LIGHT),
    ("ridge_full", "Ridge", COL_LINEAR),
    ("ridge_structural", "Ridge(s)", COL_LINEAR_LIGHT),
    ("random_pick", "Random", COL_RANDOM),
]
REF_COL = "nvmmh@8"


def geomean(x: np.ndarray) -> float:
    x = np.asarray(x, float)
    x = x[x > 0]
    return float(np.exp(np.mean(np.log(x)))) if len(x) else float("nan")


def _problem_keys(problems_path: Path) -> set[tuple[int, int, int, str]]:
    payload = json.loads(problems_path.read_text())
    rows = payload["problems"] if isinstance(payload, dict) else payload
    return {(int(r["M"]), int(r["N"]), int(r["K"]), str(r["layout"]).lower()) for r in rows}


def _filter_problems(df: pd.DataFrame, problems_path: Path | None) -> pd.DataFrame:
    if problems_path is None:
        return df
    keys = _problem_keys(problems_path)
    out = df.copy()
    out["key"] = list(zip(out.M, out.N, out.K, out.layout.str.lower()))
    return out[out["key"].isin(keys)].drop(columns=["key"])


def _report_from_db(db_path: Path, out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [sys.executable, str(SRC / "eval" / "report.py"), "--db", str(db_path), "--out", str(out_csv)],
        check=True,
    )


def discover_method_ids(df: pd.DataFrame) -> list[str]:
    ids = []
    for col in df.columns:
        if not col.endswith("@1") or "pct_peak" in col or col.startswith("nvmmh"):
            continue
        mid = col[: -len("@1")]
        ok_col = f"{mid}_top1_ok"
        if ok_col in df.columns:
            ids.append(mid)
    return ids


def _cap_width(method_id: str) -> str | None:
    if not method_id.startswith("mlp_cap_"):
        return None
    body = method_id[len("mlp_cap_") :]
    if body.endswith("_full"):
        return body[: -len("_full")]
    if body.endswith("_structural"):
        return body[: -len("_structural")]
    return None


def _cap_depth(method_id: str) -> str | None:
    if not method_id.startswith("xgb_cap_"):
        return None
    body = method_id[len("xgb_cap_") :]
    if body.endswith("_full"):
        return body[: -len("_full")]
    if body.endswith("_structural"):
        return body[: -len("_structural")]
    return None


def _method_label_color(method_id: str) -> tuple[str, str]:
    paper = {mid: (label, color) for mid, label, color in PAPER_METHODS}
    if method_id in paper:
        return paper[method_id]

    if method_id.startswith("mlp_cap_"):
        width = _cap_width(method_id)
        structural = method_id.endswith("_structural")
        label = (width or method_id).replace("x", "×")
        if structural:
            label += " (s)"
        color = COL_MLP_LIGHT if structural else COL_MLP
        return label, color

    if method_id.startswith("xgb_cap_"):
        depth = _cap_depth(method_id)
        structural = method_id.endswith("_structural")
        label = depth or method_id
        if structural:
            label += " (s)"
        color = COL_XGB_LIGHT if structural else COL_XGB
        return label, color

    return method_id, COL_RANDOM


def _method_sort_key(method_id: str) -> tuple:
    if method_id.startswith("mlp_cap_"):
        width = _cap_width(method_id) or ""
        width_idx = MLP_WIDTHS.index(width) if width in MLP_WIDTHS else 999
        return (0, width_idx, 1 if method_id.endswith("_structural") else 0, method_id)
    if method_id == "mlp_full":
        return (0, 999, 0, method_id)
    if method_id == "mlp_structural":
        return (0, 999, 1, method_id)
    if method_id.startswith("xgb_cap_"):
        depth = _cap_depth(method_id) or ""
        depth_idx = XGB_DEPTHS.index(depth) if depth in XGB_DEPTHS else 999
        return (1, depth_idx, 1 if method_id.endswith("_structural") else 0, method_id)
    if method_id == "xgb_full":
        return (1, 999, 0, method_id)
    if method_id == "xgb_structural":
        return (1, 999, 1, method_id)
    if method_id.startswith("ridge"):
        return (2, 0 if method_id.endswith("_full") else 1, 0, method_id)
    if method_id == "random_pick":
        return (3, 0, 0, method_id)
    return (9, 0, 0, method_id)


def resolve_methods(df: pd.DataFrame) -> list[tuple[str, str, str]]:
    ids = sorted(discover_method_ids(df), key=_method_sort_key)
    return [(mid, *_method_label_color(mid)) for mid in ids]


def load_run(
    run_dir: Path,
    *,
    problems_path: Path | None = None,
) -> pd.DataFrame:
    df = pd.read_csv(run_dir / "report.csv")
    df = _filter_problems(df, problems_path)
    problems_path = problems_path or (run_dir / "problems.json")
    if problems_path.is_file():
        problems = pd.DataFrame(json.loads(problems_path.read_text())["problems"])
        problems["key"] = list(
            zip(problems.M, problems.N, problems.K, problems.layout.str.lower())
        )
        df = df.copy()
        df["key"] = list(zip(df.M, df.N, df.K, df.layout.str.lower()))
        df = df.merge(problems[["key", "split"]], on="key", how="left")
    return df


def _speedup_stats(ratio: np.ndarray) -> tuple[float, float, float]:
    """Geometric-mean speedup and ±1 SEM in log-space (asymmetric on ratio scale)."""
    ratio = np.asarray(ratio, float)
    ratio = ratio[ratio > 0]
    if len(ratio) == 0:
        return float("nan"), float("nan"), float("nan")
    log_r = np.log(ratio)
    mu = float(log_r.mean())
    sem = float(log_r.std(ddof=1) / np.sqrt(len(log_r))) if len(log_r) > 1 else 0.0
    g = float(np.exp(mu))
    lo = float(np.exp(mu - sem))
    hi = float(np.exp(mu + sem))
    return g, g - lo, hi - g


def summarize(df: pd.DataFrame, methods: list[tuple[str, str, str]]) -> pd.DataFrame:
    rows = []
    for mid, label, color in methods:
        col = f"{mid}@1"
        ok_col = f"{mid}_top1_ok"
        sub = df[(df[col] > 0) & (df[REF_COL] > 0)]
        ratio = sub[col] / sub[REF_COL]
        g, err_lo, err_hi = _speedup_stats(ratio.to_numpy())
        rows.append(
            {
                "method_id": mid,
                "label": label,
                "color": color,
                "n_comparable": int(len(sub)),
                "geomean_vs_nvmmh": g,
                "err_lo": err_lo,
                "err_hi": err_hi,
                "median_vs_nvmmh": float(ratio.median()) if len(ratio) else np.nan,
                "win_rate_pct": 100.0 * float((ratio >= 1.0).mean()) if len(ratio) else np.nan,
                "coverage_pct": 100.0 * float(df[ok_col].mean()) if ok_col in df.columns else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _summary_stats(summary: pd.DataFrame, method_id: str) -> tuple[float, float, float] | None:
    sub = summary.loc[summary.method_id == method_id]
    if sub.empty:
        return None
    row = sub.iloc[0]
    return float(row.geomean_vs_nvmmh), float(row.err_lo), float(row.err_hi)


def _has_capacity_methods(summary: pd.DataFrame) -> bool:
    return bool(summary.method_id.str.startswith(("mlp_cap_", "xgb_cap_")).any())


FAMILY_X = ("MLP", "MLP(s)", "XGB", "XGB(s)", "Ridge", "Ridge(s)", "Random")


def _capacity_method_ids(family: str, capacities: list[str], *, structural: bool) -> list[str]:
    suffix = "structural" if structural else "full"
    ids = [f"{family}_cap_{cap}_{suffix}" for cap in capacities]
    paper_id = f"{family}_{'structural' if structural else 'full'}"
    ids.append(paper_id)
    return ids


def _stagger_x(x: float, n: int, spread: float = 0.32) -> np.ndarray:
    if n <= 1:
        return np.array([x], dtype=float)
    return x + np.linspace(-spread / 2, spread / 2, n, dtype=float)


def _plot_capacity_column(
    ax: plt.Axes,
    x: float,
    method_ids: list[str],
    summary: pd.DataFrame,
    *,
    color: str,
) -> None:
    y_pts: list[float] = []
    err_lo: list[float] = []
    err_hi: list[float] = []
    for mid in method_ids:
        stats = _summary_stats(summary, mid)
        if stats is None:
            continue
        g, lo, hi = stats
        if not np.isfinite(g):
            continue
        y_pts.append(g)
        err_lo.append(lo)
        err_hi.append(hi)
    if not y_pts:
        return
    x_pts = _stagger_x(x, len(y_pts))
    yerr = np.array([err_lo, err_hi], dtype=float)
    ax.errorbar(
        x_pts,
        y_pts,
        yerr=yerr,
        fmt="o",
        color=color,
        linestyle="-",
        linewidth=1.8,
        markersize=4.5,
        capsize=2.5,
        elinewidth=_ERR_KW["elinewidth"],
        capthick=_ERR_KW["capthick"],
        ecolor=color,
        zorder=3,
    )


def _plot_baseline_point(
    ax: plt.Axes,
    x: float,
    method_id: str,
    summary: pd.DataFrame,
    *,
    color: str,
) -> None:
    stats = _summary_stats(summary, method_id)
    if stats is None:
        return
    g, lo, hi = stats
    if not np.isfinite(g):
        return
    ax.errorbar(
        [x],
        [g],
        yerr=np.array([[lo], [hi]]),
        fmt="o",
        color=color,
        linestyle="none",
        markersize=5,
        capsize=2.5,
        elinewidth=_ERR_KW["elinewidth"],
        capthick=_ERR_KW["capthick"],
        ecolor=color,
        zorder=3,
    )


def plot_speedup_capacity(summary: pd.DataFrame, figure_prefix: str) -> None:
    """One panel: capacity sweeps as vertical lines at MLP/XGB columns; baselines as points."""
    _setup_rc()
    fig, ax = plt.subplots(figsize=FIG_PANEL)
    _style(ax)

    xs = np.arange(len(FAMILY_X))
    mlp_widths = [w for w in MLP_WIDTHS if w != PAPER_MLP_WIDTH]
    xgb_depths = list(XGB_DEPTHS)

    _plot_capacity_column(
        ax,
        xs[0],
        _capacity_method_ids("mlp", mlp_widths, structural=False),
        summary,
        color=COL_MLP,
    )
    _plot_capacity_column(
        ax,
        xs[1],
        _capacity_method_ids("mlp", mlp_widths, structural=True),
        summary,
        color=COL_MLP_LIGHT,
    )
    _plot_capacity_column(
        ax,
        xs[2],
        _capacity_method_ids("xgb", xgb_depths, structural=False),
        summary,
        color=COL_XGB,
    )
    _plot_capacity_column(
        ax,
        xs[3],
        _capacity_method_ids("xgb", xgb_depths, structural=True),
        summary,
        color=COL_XGB_LIGHT,
    )
    _plot_baseline_point(ax, xs[4], "ridge_full", summary, color=COL_LINEAR)
    _plot_baseline_point(ax, xs[5], "ridge_structural", summary, color=COL_LINEAR_LIGHT)
    _plot_baseline_point(ax, xs[6], "random_pick", summary, color=COL_RANDOM)

    ax.axhline(1.0, color=COL_NVMMH, ls="--", lw=1.2, alpha=0.8, zorder=1)
    ax.set_xticks(xs, FAMILY_X, rotation=20, ha="right")
    ax.set_ylabel("Speedup vs nvMMH")
    vals = summary["geomean_vs_nvmmh"].to_numpy()
    errs = summary["err_hi"].to_numpy()
    ymax = float(np.nanmax(vals + errs)) if len(vals) else 1.2
    ax.set_ylim(0, max(1.2, ymax * 1.08))
    _apply_panel_fonts(ax)
    _save(fig, f"{figure_prefix}_geomean_speedup_vs_nvmmh", bottom=0.22)


def plot_speedup_bars(summary: pd.DataFrame, figure_prefix: str) -> None:
    if _has_capacity_methods(summary):
        plot_speedup_capacity(summary, figure_prefix)
        return

    _setup_rc()
    fig, ax = plt.subplots(figsize=FIG_PANEL)
    _style(ax)
    labels = summary["label"].tolist()
    vals = summary["geomean_vs_nvmmh"].to_numpy()
    err = np.array(
        [summary["err_lo"].to_numpy(), summary["err_hi"].to_numpy()],
        dtype=float,
    )
    colors = summary["color"].tolist()
    x = np.arange(len(labels))
    ax.bar(
        x,
        vals,
        width=0.65,
        color=colors,
        edgecolor="white",
        yerr=err,
        capsize=3,
        error_kw=_ERR_KW,
    )
    ax.axhline(1.0, color=COL_NVMMH, ls="--", lw=1.2, alpha=0.8)
    ax.set_xticks(x, labels, rotation=20, ha="right")
    ax.set_ylabel("Speedup vs nvMMH")
    ymax = float(np.nanmax(vals + err[1])) if len(vals) else 1.2
    ax.set_ylim(0, max(1.2, ymax * 1.08))
    _apply_panel_fonts(ax)
    _save(fig, f"{figure_prefix}_geomean_speedup_vs_nvmmh", bottom=0.22)


def plot_speedup_cdf(df: pd.DataFrame, figure_prefix: str) -> None:
    _setup_rc()
    fig, ax = plt.subplots(figsize=FIG_PANEL)
    _style(ax)
    for mid, label, color in PAPER_METHODS:
        if mid in ("ridge_structural", "random_pick"):
            continue
        col = f"{mid}@1"
        if col not in df.columns:
            continue
        sub = df[(df[col] > 0) & (df[REF_COL] > 0)]
        if sub.empty:
            continue
        ratio = np.sort(sub[col] / sub[REF_COL])
        y = np.arange(1, len(ratio) + 1) / len(ratio)
        ax.plot(ratio, y, label=label, color=color, lw=2.2)
    ax.axvline(1.0, color=COL_NVMMH, ls="--", lw=1.2, alpha=0.8)
    ax.set_xlabel("Speedup vs nvMMH")
    ax.set_ylabel("Fraction of problems")
    ax.set_xlim(left=0)
    ax.legend(frameon=False, loc="lower right")
    _apply_panel_fonts(ax)
    _save(fig, f"{figure_prefix}_speedup_cdf")


def plot_by_layout(df: pd.DataFrame, figure_prefix: str) -> None:
    layouts = sorted(df.layout.unique())
    focus = [m for m in PAPER_METHODS if m[0] in ("mlp_full", "mlp_structural", "xgb_full")]

    rows = []
    for lay in layouts:
        d = df[df.layout == lay]
        for mid, label, color in focus:
            col = f"{mid}@1"
            sub = d[(d[col] > 0) & (d[REF_COL] > 0)]
            if sub.empty:
                g = np.nan
            else:
                g = geomean((sub[col] / sub[REF_COL]).to_numpy())
            rows.append({"layout": lay.upper(), "label": label, "geomean": g, "color": color})

    plot_df = pd.DataFrame(rows)
    _setup_rc()
    fig, ax = plt.subplots(figsize=FIG_PANEL)
    _style(ax)
    labels = sorted(plot_df["label"].unique(), key=lambda x: [m[1] for m in focus].index(x))
    x = np.arange(len(layouts))
    width = 0.8 / max(len(labels), 1)
    for i, lab in enumerate(labels):
        sub = plot_df[plot_df.label == lab]
        vals = [
            sub[sub.layout == lay.upper()]["geomean"].iloc[0]
            if len(sub[sub.layout == lay.upper()])
            else np.nan
            for lay in layouts
        ]
        color = sub["color"].iloc[0]
        ax.bar(x + (i - (len(labels) - 1) / 2) * width, vals, width=width, label=lab, color=color)
    ax.axhline(1.0, color=COL_NVMMH, ls="--", lw=1.2, alpha=0.8)
    ax.set_xticks(x, [lay.upper() for lay in layouts])
    ax.set_ylabel("Speedup vs nvMMH")
    ax.legend(frameon=False, ncol=2, fontsize=10)
    _apply_panel_fonts(ax)
    _save(fig, f"{figure_prefix}_geomean_by_layout")


def write_summary_md(
    df: pd.DataFrame,
    summary: pd.DataFrame,
    run_dir: Path,
    out_path: Path,
) -> None:
    methods = [(r.method_id, r.label, r.color) for r in summary.itertuples()]
    lines = [
        "# DeepBench GPU eval summary",
        "",
        f"Run directory: `{run_dir.relative_to(SRC)}`",
        f"Problems benchmarked: **{len(df)}**",
        "",
        "Throughput is in **GFLOP/s** (stored in `mean_tflops` columns). "
        "Comparison baseline: **nvMMH** (best measured schedule variant of the rank-1 tile).",
        "",
        "## Headline (rank-1 vs nvMMH)",
        "",
        "| Method | Geo-mean | Median | Win % | Coverage % |",
        "|--------|----------|--------|-------|------------|",
    ]
    for _, r in summary.iterrows():
        lines.append(
            f"| {r['label']} | {r['geomean_vs_nvmmh']:.3f} | "
            f"{r['median_vs_nvmmh']:.3f} | {r['win_rate_pct']:.1f} | {r['coverage_pct']:.1f} |"
        )

    lines += ["", "## By DeepBench split", ""]
    for split in ["train", "inference_server", "inference_device"]:
        d = df[df.split == split]
        if d.empty:
            continue
        lines.append(f"### `{split}` (n={len(d)})")
        lines.append("")
        lines.append("| Method | Geo-mean | Win % |")
        lines.append("|--------|----------|-------|")
        for mid, label, _ in methods:
            col = f"{mid}@1"
            sub = d[(d[col] > 0) & (d[REF_COL] > 0)]
            if sub.empty:
                continue
            ratio = sub[col] / sub[REF_COL]
            lines.append(
                f"| {label} | {geomean(ratio.to_numpy()):.3f} | "
                f"{100*(ratio>=1).mean():.1f} |"
            )
        lines.append("")

    lines += [
        "## Notes",
        "",
        "- Results are **throughput generalization** only (no exhaustive oracle on DeepBench).",
        "- Comparison baseline: **nvMMH** (best measured schedule variant).",
        "",
    ]
    out_path.write_text("\n".join(lines))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--run-dir",
        type=Path,
        default=EVAL_ARTIFACTS / "deepbench_cap",
        help="directory with report.csv (and optional problems.json)",
    )
    ap.add_argument(
        "--db",
        type=Path,
        default=None,
        help="generate report.csv from eval SQLite DB before plotting",
    )
    ap.add_argument(
        "--problems",
        type=Path,
        default=None,
        help="restrict to the problems in this problems.json (default: all problems of the run)",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="figures + summary (default: <run-dir>)",
    )
    ap.add_argument(
        "--figure-prefix",
        default="deepbench",
        help="PDF/PNG basename prefix (default: deepbench)",
    )
    args = ap.parse_args()
    run_dir = args.run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    out_dir = (args.out_dir or run_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    figure_prefix = args.figure_prefix

    if args.db is not None:
        _report_from_db(args.db.resolve(), run_dir / "report.csv")

    problems_path = args.problems.resolve() if args.problems else None
    df = load_run(run_dir, problems_path=problems_path)
    methods = resolve_methods(df)
    summary = summarize(df, methods)
    summary.to_csv(out_dir / "summary.csv", index=False)

    plot_speedup_bars(summary, figure_prefix)
    plot_speedup_cdf(df, figure_prefix)
    plot_by_layout(df, figure_prefix)
    write_summary_md(df, summary, run_dir, out_dir / "SUMMARY.md")

    print(summary.to_string(index=False))
    print(f"\nWrote {out_dir / 'SUMMARY.md'}")
    fig_dir = FIGURES
    print(f"Figures -> {fig_dir}/{figure_prefix}_*.pdf")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
