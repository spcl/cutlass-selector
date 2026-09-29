#!/usr/bin/env python3
"""Summarize + plot capacity-sweep held-out eval from artifacts/analysis/capacity/."""

from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from repo_paths import CAPACITY, PAPER

OUT = CAPACITY / "figures"
PAPER_OUT = PAPER / "figures"

BG = "#ebebeb"
COL_MLP = "#1e3a8a"
COL_MLP_LIGHT = "#7d94ad"
COL_XGB = "#14532d"
COL_XGB_LIGHT = "#6d9078"

MLP_ORDER = ["16x8", "32x16", "64x32", "128x64", "256x128x64", "1024x1024x512x256"]
XGB_ORDER = ["d2", "d3", "d4", "d6", "d8", "d11", "d14"]

FIG_W, FIG_H = 7.2, 5.6
FONT = 20
MLP_SKIP_HIDDEN = {"128x64"}
MLP_PAPER_LARGEST = "1024x1024x512x256"
PAPER_MLP_RUNS = {
    "full": PAPER / "mlp_mse" / "metrics.json",
    "structural": PAPER / "mlp_mse_structural" / "metrics.json",
}


def _parse_run(run_dir: Path) -> dict | None:
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.exists():
        return None
    m = json.loads(metrics_path.read_text())
    ev = m.get("eval")
    if not ev or "regret_mean" not in ev:
        return None

    name = run_dir.name
    if name.startswith("mlp_"):
        family = "mlp"
        capacity = name.split("_")[2]
        params = m.get("n_params")
    elif name.startswith("xgb_"):
        family = "xgb"
        capacity = "d" + re.search(r"_d(\d+)_", name).group(1)
        params = m.get("n_leaves")
    else:
        return None

    feature_set = "structural" if "structural" in name else "full"
    seed = int(re.search(r"_s(\d+)$", name).group(1))
    return {
        "run_dir": name,
        "family": family,
        "capacity": capacity,
        "feature_set": feature_set,
        "seed": seed,
        "params": params,
        "eval_regret_mean": ev["regret_mean"],
        "eval_regret_p95": ev["regret_p95"],
        "eval_top1": ev["top1"],
        "eval_within5pct": ev["within5pct"],
        "eval_groups": ev["groups"],
    }


def load_runs(capacity_dir: Path = CAPACITY) -> pd.DataFrame:
    rows = []
    for d in sorted(capacity_dir.iterdir()):
        if d.is_dir():
            row = _parse_run(d)
            if row:
                rows.append(row)
    if not rows:
        raise FileNotFoundError(f"no eval metrics under {capacity_dir}")
    return pd.DataFrame(rows)


def merge_val(df: pd.DataFrame, capacity_dir: Path = CAPACITY) -> pd.DataFrame:
    val_path = capacity_dir / "capacity_runs.csv"
    if not val_path.exists():
        return df
    val = pd.read_csv(val_path)[["run_dir", "val_regret", "val_ndcg@10"]]
    return df.merge(val, on="run_dir", how="left")


def write_summary(df: pd.DataFrame, capacity_dir: Path = CAPACITY) -> Path:
    out = capacity_dir / "capacity_eval_summary.csv"
    summary = df.assign(
        status="ok",
        regret_mean=df["eval_regret_mean"],
        seconds=np.nan,
    )[["run_dir", "family", "status", "regret_mean", "seconds"]].sort_values(
        ["family", "run_dir"]
    )
    summary.to_csv(out, index=False)
    return out


def aggregate_scaling(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    agg = (
        df.groupby(["family", "capacity", "feature_set"], as_index=False)
        .agg(
            regret_mean=("eval_regret_mean", "mean"),
            regret_sd=("eval_regret_mean", "std"),
            params=("params", "mean"),
            n_seeds=("seed", "count"),
        )
    )

    def _pivot(family: str, order: list[str], label_col: str) -> pd.DataFrame:
        sub = agg[agg.family == family].copy()
        full = sub[sub.feature_set == "full"].set_index("capacity")
        struct = sub[sub.feature_set == "structural"].set_index("capacity")
        rows = []
        for cap in order:
            if cap not in full.index or cap not in struct.index:
                continue
            rows.append(
                {
                    label_col: cap,
                    "params": full.loc[cap, "params"],
                    "full": full.loc[cap, "regret_mean"],
                    "structural": struct.loc[cap, "regret_mean"],
                    "full_sd": full.loc[cap, "regret_sd"],
                    "structural_sd": struct.loc[cap, "regret_sd"],
                    "n_seeds": int(full.loc[cap, "n_seeds"]),
                }
            )
        return pd.DataFrame(rows)

    mlp = _pivot("mlp", MLP_ORDER, "hidden")
    xgb = _pivot("xgb", XGB_ORDER, "depth")
    return mlp, xgb


def write_eval_gap(df: pd.DataFrame, capacity_dir: Path = CAPACITY) -> Path:
    """Paired full vs structural gap on held-out eval (seed-matched)."""
    rows = []
    for family in ("mlp", "xgb"):
        order = MLP_ORDER if family == "mlp" else XGB_ORDER
        for cap in order:
            full = df[(df.family == family) & (df.capacity == cap) & (df.feature_set == "full")]
            struct = df[
                (df.family == family) & (df.capacity == cap) & (df.feature_set == "structural")
            ]
            if full.empty or struct.empty:
                continue
            full = full.set_index("seed").sort_index()
            struct = struct.set_index("seed").sort_index()
            seeds = sorted(set(full.index) & set(struct.index))
            if not seeds:
                continue
            f = full.loc[seeds, "eval_regret_mean"].to_numpy()
            s = struct.loc[seeds, "eval_regret_mean"].to_numpy()
            gaps = s - f
            if len(gaps) > 1:
                t_res = stats.ttest_rel(s, f)
                gap_p = float(t_res.pvalue)
                gap_sd = float(np.std(gaps, ddof=1))
                se = gap_sd / np.sqrt(len(gaps))
                gap_ci_lo = float(np.mean(gaps) - 1.96 * se)
                gap_ci_hi = float(np.mean(gaps) + 1.96 * se)
            else:
                gap_p = float("nan")
                gap_sd = 0.0
                gap_ci_lo = gap_ci_hi = float(gaps[0])

            rows.append(
                {
                    "family": family,
                    "capacity": cap,
                    "params_full": float(full["params"].mean()),
                    "params_structural": float(struct["params"].mean()),
                    "regret_full": float(np.mean(f)),
                    "regret_structural": float(np.mean(s)),
                    "gap": float(np.mean(gaps)),
                    "gap_ci_lo": gap_ci_lo,
                    "gap_ci_hi": gap_ci_hi,
                    "gap_p": gap_p,
                    "gap_seed_mean": float(np.mean(gaps)),
                    "gap_seed_sd": gap_sd,
                    "gap_seeds_positive": int(np.sum(gaps > 0)),
                    "n_seeds": len(seeds),
                }
            )

    out = capacity_dir / "capacity_eval_gap.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    return out


def _setup_rc() -> None:
    plt.rcParams.update(
        {
            "font.size": FONT,
            "axes.labelsize": FONT,
            "xtick.labelsize": FONT,
            "ytick.labelsize": FONT,
            "legend.fontsize": FONT,
        }
    )


def _style(ax) -> None:
    ax.set_facecolor(BG)
    ax.figure.patch.set_facecolor("white")


def _line_with_band(
    ax,
    x: np.ndarray,
    y: np.ndarray,
    y_sd: np.ndarray | None,
    n_seeds: int,
    *,
    color: str,
    label: str,
    marker: str = "o",
) -> None:
    y_pct = 100.0 * np.asarray(y, dtype=float)
    ax.plot(
        x,
        y_pct,
        color=color,
        marker=marker,
        label=label,
        linewidth=2.2,
        markersize=9,
        zorder=3,
    )
    if y_sd is None or n_seeds < 2:
        return
    sem = 100.0 * np.asarray(y_sd, dtype=float) / np.sqrt(n_seeds)
    lo = y_pct - sem
    hi = y_pct + sem
    mask = np.isfinite(lo) & np.isfinite(hi)
    if not mask.any():
        return
    ax.fill_between(x, lo, hi, color=color, alpha=0.22, linewidth=0, zorder=2)


def _save(fig: plt.Figure, stem: str, *, paper: bool = False) -> None:
    targets = [OUT]
    if paper:
        targets.append(PAPER_OUT)
    for base in targets:
        base.mkdir(parents=True, exist_ok=True)
        for ext in ("png", "pdf"):
            path = base / f"{stem}.{ext}"
            fig.savefig(path, dpi=160, bbox_inches="tight")
            print(f"wrote {path}")
    plt.close(fig)


def _apply_paper_mlp_largest(mlp: pd.DataFrame) -> pd.DataFrame:
    """Rightmost MLP point: use primary paper mlp_mse runs (same arch, held-out eval)."""
    df = mlp.copy()
    mask = df["hidden"] == MLP_PAPER_LARGEST
    if not mask.any():
        return df
    for feat, path in PAPER_MLP_RUNS.items():
        if not path.exists():
            print(f"warning: missing {path}, keeping capacity-sweep largest MLP")
            return df
        regret = json.loads(path.read_text())["eval"]["regret_mean"]
        df.loc[mask, feat] = regret
    full = float(df.loc[mask, "full"].iloc[0])
    struct = float(df.loc[mask, "structural"].iloc[0])
    print(
        f"largest MLP: paper mlp_mse values  full={full * 100:.2f}%  "
        f"structural={struct * 100:.2f}%"
    )
    return df


def mlp_scaling_for_seed(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    mlp = df[(df.family == "mlp") & (df.seed == seed)]
    rows = []
    for cap in MLP_ORDER:
        if cap in MLP_SKIP_HIDDEN:
            continue
        full = mlp[(mlp.capacity == cap) & (mlp.feature_set == "full")]
        struct = mlp[(mlp.capacity == cap) & (mlp.feature_set == "structural")]
        if full.empty or struct.empty:
            continue
        rows.append(
            {
                "hidden": cap,
                "params": float(full["params"].iloc[0]),
                "full": float(full["eval_regret_mean"].iloc[0]),
                "structural": float(struct["eval_regret_mean"].iloc[0]),
            }
        )
    return pd.DataFrame(rows).sort_values("params")


def _plot_mlp_lines(df: pd.DataFrame, stem: str, *, paper: bool = False) -> None:
    x = df["params"].to_numpy(dtype=float)
    n_seeds = int(df["n_seeds"].iloc[0]) if "n_seeds" in df.columns else 1
    full_sd = df["full_sd"].to_numpy() if "full_sd" in df.columns else None
    struct_sd = df["structural_sd"].to_numpy() if "structural_sd" in df.columns else None
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    _line_with_band(
        ax, x, df["full"].to_numpy(), full_sd, n_seeds, color=COL_MLP, label="Full",
    )
    _line_with_band(
        ax,
        x,
        df["structural"].to_numpy(),
        struct_sd,
        n_seeds,
        color=COL_MLP_LIGHT,
        label="Structural",
        marker="s",
    )
    ax.set_xscale("log")
    ax.set_xlabel("MLP Params")
    ax.set_ylabel("Regret (%)")
    ax.legend(framealpha=0.92, loc="upper right")
    _style(ax)
    fig.tight_layout()
    _save(fig, stem, paper=paper)


def plot_mlp_scaling(mlp: pd.DataFrame) -> None:
    df = _apply_paper_mlp_largest(mlp)
    df = df[~df["hidden"].isin(MLP_SKIP_HIDDEN)].sort_values("params")
    _plot_mlp_lines(df, "capacity_eval_mlp_scaling", paper=True)


def plot_mlp_scaling_per_seed(df: pd.DataFrame) -> None:
    seeds = sorted(df.loc[df.family == "mlp", "seed"].unique())
    for seed in seeds:
        scale = mlp_scaling_for_seed(df, seed)
        _plot_mlp_lines(scale, f"capacity_eval_mlp_scaling_s{seed}")
        print(f"MLP seed {seed}: {len(scale)} capacity points (capacity sweep only)")


def plot_xgb_scaling(xgb: pd.DataFrame) -> None:
    df = xgb.copy()
    df["depth_num"] = df["depth"].str[1:].astype(int)
    df = df.sort_values("depth_num")
    x = df["depth_num"].to_numpy()
    n_seeds = int(df["n_seeds"].iloc[0]) if "n_seeds" in df.columns else 1

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    _line_with_band(
        ax, x, df["full"].to_numpy(), df["full_sd"].to_numpy(), n_seeds,
        color=COL_XGB, label="Full",
    )
    _line_with_band(
        ax,
        x,
        df["structural"].to_numpy(),
        df["structural_sd"].to_numpy(),
        n_seeds,
        color=COL_XGB_LIGHT,
        label="Structural",
        marker="s",
    )
    ax.set_xticks(x)
    ax.set_xlabel("XGBoost Maximum Depth")
    ax.set_ylabel("Regret (%)")
    ax.legend(framealpha=0.92, loc="upper right")
    _style(ax)
    fig.tight_layout()
    _save(fig, "capacity_eval_xgb_scaling", paper=True)


def plot_val_vs_eval(df: pd.DataFrame) -> None:
    if "val_regret" not in df.columns:
        return
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    for family, color in [("mlp", COL_MLP), ("xgb", COL_XGB)]:
        sub = df[df.family == family]
        ax.scatter(
            100 * sub["val_regret"],
            100 * sub["eval_regret_mean"],
            c=color,
            alpha=0.75,
            label=family.upper(),
            edgecolors="white",
            linewidths=0.5,
        )
    lims = [
        0,
        max(100 * df["val_regret"].max(), 100 * df["eval_regret_mean"].max()) * 1.05,
    ]
    ax.plot(lims, lims, "--", color="#888888", linewidth=1)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("Validation regret (%)")
    ax.set_ylabel("Held-out eval regret (%)")
    ax.set_title("Capacity sweep: validation vs held-out eval")
    ax.legend()
    _style(ax)
    fig.tight_layout()
    path = OUT / "capacity_val_vs_eval.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def main() -> None:
    _setup_rc()
    OUT.mkdir(parents=True, exist_ok=True)

    df = merge_val(load_runs())
    print(f"loaded {len(df)} runs with eval  "
          f"({(df.family=='mlp').sum()} MLP, {(df.family=='xgb').sum()} XGB)")

    summary_path = write_summary(df)
    print(f"wrote {summary_path}")

    mlp_scale, xgb_scale = aggregate_scaling(df)
    mlp_scale.to_csv(CAPACITY / "mlp_capacity_scaling.csv", index=False)
    xgb_scale.to_csv(CAPACITY / "xgb_capacity_scaling.csv", index=False)
    gap_path = write_eval_gap(df)
    print(f"wrote {gap_path}")
    print(f"wrote {CAPACITY / 'mlp_capacity_scaling.csv'}")
    print(f"wrote {CAPACITY / 'xgb_capacity_scaling.csv'}")

    plot_mlp_scaling(mlp_scale)
    plot_mlp_scaling_per_seed(df)
    plot_xgb_scaling(xgb_scale)
    plot_val_vs_eval(df)
    print(f"figures in {OUT}")


if __name__ == "__main__":
    main()
