#!/usr/bin/env python3
"""Generate summary plots from artifacts/analysis/paper/ training sweep."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

from repo_paths import PAPER

OUT = PAPER / "figures"

RUNS = [
    "mlp_mse", "mlp_ranknet", "mlp_lambdarank",
    "mlp_mse_structural", "mlp_ranknet_structural", "mlp_lambdarank_structural",
    "xgb_mse", "xgb_ndcg", "xgb_pairwise",
    "xgb_mse_structural", "xgb_ndcg_structural", "xgb_pairwise_structural",
]

# Global palette
BG = "#ebebeb"
COL_MLP = "#1e3a8a"
COL_MLP_LIGHT = "#7d94ad"
COL_XGB = "#14532d"
COL_XGB_LIGHT = "#6d9078"
COL_NVMMH = "#525252"
COL_NVMMH_ALT = "#737373"
COL_RANDOM = "#a8a8a8"
COL_LINEAR = "#4b6cb7"
COL_LINEAR_LIGHT = "#9aafd4"

# Legacy figure names (pre-descriptive rename).
OLD_FIGURES = [
    "by_regime_best", "by_regime_nvmmh", "comparison_main", "full_vs_structural",
    "hardest_groups", "nvmmh_gpu_sweep", "regret_boxplot", "regret_by_run",
    "regret_cdf", "success_rates", "training_curves_mlp", "training_curves_xgboost",
    "val_ndcg_vs_eval_regret", "within5_comparison",
]

OBJ_LS_MLP = {"mse": "-", "ranknet": "--", "lambdarank": "-."}
OBJ_LS_XGB = {"mse": "-", "ndcg": "--", "pairwise": "-."}
OBJ_MARKERS = {"mse": "o", "ranknet": "s", "lambdarank": "D", "ndcg": "s", "pairwise": "D"}
OBJECTIVE_LABELS = {
    "mse": "MSE",
    "ranknet": "RankNet",
    "lambdarank": "LambdaRank",
    "ndcg": "NDCG",
    "pairwise": "Pairwise",
}

KEY_RUNS = {
    "mlp_full": "mlp_ranknet",
    "mlp_struct": "mlp_mse_structural",
    "xgb_full": "xgb_mse",
    "xgb_struct": "xgb_mse_structural",
}

# One panel size for all single-panel figures (symmetric in 2×2 LaTeX grids).
FIG_W, FIG_H = 7.2, 5.6
FIG_TWIN = (FIG_W * 2 + 0.9, FIG_H)
FIG_TALL = (FIG_W, FIG_H + 1.2)  # horizontal bar charts with many y labels
FIG_PANEL = (FIG_W, FIG_H)  # matched side-by-side main-text panels
PANEL_LABEL = 20
PANEL_TICK = 18
PANEL_LEGEND = 18


def _setup_rc() -> None:
    plt.rcParams.update(
        {
            "font.size": 13,
            "axes.labelsize": 14,
            "legend.fontsize": 11,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
        }
    )


def _style(ax) -> None:
    ax.set_facecolor(BG)
    ax.figure.patch.set_facecolor("white")


def _apply_panel_fonts(ax, *, legend=None) -> None:
    """Body-text scale for side-by-side main-text panels."""
    ax.xaxis.label.set_fontsize(PANEL_LABEL)
    ax.yaxis.label.set_fontsize(PANEL_LABEL)
    ax.tick_params(axis="both", which="major", labelsize=PANEL_TICK)
    if legend is not None:
        for text in legend.get_texts():
            text.set_fontsize(PANEL_LEGEND)


def _parse_run(run_dir: str) -> dict:
    parts = run_dir.split("_", 1)
    model = "MLP" if parts[0] == "mlp" else "XGBoost"
    rest = parts[1]
    structural = rest.endswith("_structural")
    objective = rest.replace("_structural", "") if structural else rest
    return {
        "model": model,
        "objective": objective,
        "feature_set": "structural" if structural else "full",
        "run_dir": run_dir,
    }


def _objective_label(objective: str) -> str:
    return OBJECTIVE_LABELS.get(objective, objective.replace("_", " ").title())


def _short_label(model: str, feature_set: str) -> str:
    if model == "MLP":
        return "MLP(s)" if feature_set == "structural" else "MLP"
    if model == "XGBoost":
        return "XGB(s)" if feature_set == "structural" else "XGB"
    if model == "nvMMH":
        return "nvMMH"
    return "random"


def _series_label(model: str, feature_set: str, *, show_objective: bool = False, objective: str = "") -> str:
    name = _short_label(model, feature_set)
    if show_objective and objective:
        return f"{name}·{objective}"
    return name


def _series_order(model: str, feature_set: str) -> tuple[int, int]:
    model_ord = {"MLP": 0, "XGBoost": 1, "nvMMH": 2, "random": 3}
    fs_ord = {"full": 0, "structural": 1, "n/a": 0}
    return model_ord.get(model, 9), fs_ord.get(feature_set, 0)


def _sort_runs(run_dirs: list[str]) -> list[str]:
    return sorted(run_dirs, key=lambda r: _series_order(_parse_run(r)["model"], _parse_run(r)["feature_set"]))


def _fill_color(model: str, feature_set: str) -> str:
    if model == "MLP":
        return COL_MLP_LIGHT if feature_set == "structural" else COL_MLP
    if model == "XGBoost":
        return COL_XGB_LIGHT if feature_set == "structural" else COL_XGB
    if model == "nvMMH":
        return COL_NVMMH
    return COL_RANDOM


def _bar_kwargs(model: str, feature_set: str) -> dict:
    c = _fill_color(model, feature_set)
    return {"color": c, "edgecolor": c, "linewidth": 0.6}


def _line_kwargs(model: str, feature_set: str, objective: str) -> dict:
    ls_map = OBJ_LS_MLP if model == "MLP" else OBJ_LS_XGB
    color = _fill_color(model, feature_set)
    return {
        "color": color,
        "linestyle": ls_map.get(objective, "-") if feature_set == "full" else "-",
        "marker": OBJ_MARKERS.get(objective, "o"),
        "markevery": 10,
        "markersize": 5,
        "linewidth": 2.0 if feature_set == "full" else 1.6,
    }


def _save(fig, name: str, *, bottom: float | None = None) -> None:
    fig.tight_layout(pad=1.0)
    if bottom is not None:
        fig.subplots_adjust(bottom=bottom)
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png", dpi=200)
    plt.close(fig)


def _legend_patches() -> list[Patch]:
    return [
        Patch(facecolor=COL_MLP, edgecolor=COL_MLP, label="MLP"),
        Patch(facecolor=COL_MLP_LIGHT, edgecolor=COL_MLP_LIGHT, label="MLP(s)"),
        Patch(facecolor=COL_XGB, edgecolor=COL_XGB, label="XGB"),
        Patch(facecolor=COL_XGB_LIGHT, edgecolor=COL_XGB_LIGHT, label="XGB(s)"),
        Patch(facecolor=COL_NVMMH, edgecolor=COL_NVMMH, label="nvMMH"),
        Patch(facecolor=COL_RANDOM, edgecolor=COL_RANDOM, label="random"),
    ]


def _cleanup_old_figures() -> None:
    for stem in OLD_FIGURES:
        for ext in (".pdf", ".png"):
            p = OUT / f"{stem}{ext}"
            if p.exists():
                p.unlink()


def load_summary() -> pd.DataFrame:
    df = pd.read_csv(PAPER / "summary_training.csv")
    for col in ("model", "objective", "feature_set", "run_dir"):
        if col not in df.columns:
            continue
    parsed = [_parse_run(r) for r in df["run_dir"]]
    df["model_family"] = [p["model"] for p in parsed]
    return df


def load_all_eval() -> pd.DataFrame:
    rows = []
    for run in RUNS:
        p = PAPER / run / "eval_regret.csv"
        if not p.exists():
            continue
        d = pd.read_csv(p)
        meta = _parse_run(run)
        d["run_dir"] = run
        for k, v in meta.items():
            d[k] = v
        rows.append(d)
    return pd.concat(rows, ignore_index=True)


def load_nvmmh_eval(mode: str = "top1") -> pd.DataFrame:
    p = PAPER / "nvmmh" / f"eval_regret_{mode}.csv"
    if not p.exists():
        return pd.DataFrame()
    d = pd.read_csv(p)
    d["run_dir"] = f"nvmmh_{mode}"
    d["model"] = "nvMMH"
    d["feature_set"] = "n/a"
    return d


def _hist_run_key(row: pd.Series) -> str:
    prefix = "mlp" if row["model"].lower() == "mlp" else "xgb"
    obj = row["objective"]
    if row["feature_set"] == "structural":
        return f"{prefix}_{obj}_structural"
    return f"{prefix}_{obj}"


def _eval_metric(run: str, key: str) -> float:
    return json.loads((PAPER / run / "metrics.json").read_text())["eval"][key]


def _column_mean_sem(path: Path, column: str) -> tuple[float, float]:
    values = pd.read_csv(path)[column].to_numpy(dtype=float)
    mean = float(np.mean(values))
    sem = float(np.std(values, ddof=1) / np.sqrt(len(values))) if len(values) > 1 else 0.0
    return mean, sem


def _regret_mean_sem(path: Path) -> tuple[float, float]:
    return _column_mean_sem(path, "regret")


def _eval_csv_for_key(key: str) -> Path:
    if key == "nvmmh":
        return PAPER / "nvmmh" / "eval_regret_top1.csv"
    return PAPER / key / "eval_regret.csv"


def _regime_column_mean_sem(run_dir: str, regime: str, column: str) -> tuple[float, float]:
    sub = pd.read_csv(PAPER / run_dir / "eval_regret.csv")
    sub = sub[sub["regime"] == regime][column].to_numpy(dtype=float)
    if len(sub) == 0:
        return float("nan"), 0.0
    mean = float(np.mean(sub))
    sem = float(np.std(sub, ddof=1) / np.sqrt(len(sub))) if len(sub) > 1 else 0.0
    return mean, sem


_ERR_KW = {
    "elinewidth": 1.2,
    "capthick": 1.2,
    "ecolor": "#333333",
    "ls": "-",
}


def plot_main_comparison() -> None:
    # Increasing regret top→bottom: full MLP, full XGB, structural MLP, structural XGB, nvMMH, random.
    entries = [
        ("mlp_ranknet", "MLP", "full"),
        ("xgb_mse", "XGBoost", "full"),
        ("mlp_mse_structural", "MLP", "structural"),
        ("xgb_mse_structural", "XGBoost", "structural"),
    ]
    labels, vals, errs, styles = [], [], [], []
    for run, model, fs in entries:
        labels.append(_series_label(model, fs))
        mean, sem = _regret_mean_sem(PAPER / run / "eval_regret.csv")
        vals.append(100 * mean)
        errs.append(100 * sem)
        styles.append(_bar_kwargs(model, fs))

    nvmmh_csv = PAPER / "nvmmh" / "eval_regret_top1.csv"
    if nvmmh_csv.exists():
        labels.append("nvMMH")
        mean, sem = _regret_mean_sem(nvmmh_csv)
        vals.append(100 * mean)
        errs.append(100 * sem)
        styles.append(_bar_kwargs("nvMMH", "full"))

    random_csv = PAPER / "random_pick" / "eval_regret.csv"
    if random_csv.exists():
        labels.append("random")
        mean, sem = _regret_mean_sem(random_csv)
        vals.append(100 * mean)
        errs.append(100 * sem)
        styles.append(_bar_kwargs("random", "full"))
    else:
        rnd = json.loads((PAPER / "mlp_mse" / "metrics.json").read_text())["baselines"]["random_pick"]["regret_mean"]
        labels.append("random")
        vals.append(100 * rnd)
        errs.append(0.0)
        styles.append(_bar_kwargs("random", "full"))

    fig, ax = plt.subplots(figsize=FIG_PANEL)
    _style(ax)
    for label, val, err, kw in zip(labels, vals, errs, styles):
        ax.barh(label, val, height=0.65, xerr=err, capsize=3, error_kw=_ERR_KW, **kw)
    ax.set_xlabel("Selection Regret (%)")
    ax.invert_yaxis()
    _apply_panel_fonts(ax)
    _save(fig, "eval_mean_regret_mlp_xgb_nvmmh_random")


def _ridge_bar_kwargs(structural: bool) -> dict:
    c = COL_LINEAR_LIGHT if structural else COL_LINEAR
    return {"color": c, "edgecolor": c, "linewidth": 0.6}


def plot_baseline_mean_regret() -> None:
    """Extended mean-regret panel (MLP/XGB/Ridge/nvMMH/random); matches FIG_PANEL styling."""
    baselines_dir = PAPER / "baselines"
    rows: list[tuple[str, Path, dict]] = [
        ("MLP", PAPER / "mlp_ranknet" / "eval_regret.csv", _bar_kwargs("MLP", "full")),
        ("XGB", PAPER / "xgb_mse" / "eval_regret.csv", _bar_kwargs("XGBoost", "full")),
        ("MLP(s)", PAPER / "mlp_mse_structural" / "eval_regret.csv", _bar_kwargs("MLP", "structural")),
        ("XGB(s)", PAPER / "xgb_mse_structural" / "eval_regret.csv", _bar_kwargs("XGBoost", "structural")),
        ("nvMMH", PAPER / "nvmmh" / "eval_regret_top1.csv", _bar_kwargs("nvMMH", "full")),
        ("Ridge", baselines_dir / "linear_full" / "eval_regret.csv", _ridge_bar_kwargs(False)),
        ("Ridge(s)", baselines_dir / "linear_structural" / "eval_regret.csv", _ridge_bar_kwargs(True)),
        ("random", PAPER / "random_pick" / "eval_regret.csv", _bar_kwargs("random", "full")),
    ]
    parsed: list[tuple[str, float, float, dict]] = []
    for label, path, style in rows:
        if not path.is_file():
            raise FileNotFoundError(f"missing {path}")
        mean, sem = _regret_mean_sem(path)
        parsed.append((label, mean, sem, style))
    parsed.sort(key=lambda r: r[1])

    fig, ax = plt.subplots(figsize=FIG_PANEL)
    _style(ax)
    for label, mean, sem, kw in parsed:
        ax.barh(label, 100 * mean, height=0.65, xerr=100 * sem, capsize=3, error_kw=_ERR_KW, **kw)
    ax.set_xlabel("Selection Regret (%)")
    ax.invert_yaxis()
    _apply_panel_fonts(ax)
    _save(fig, "baseline_mean_regret")


def plot_regret_summary(summary: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(FIG_W + 2.5, FIG_H + 1.5))
    _style(ax)
    d = summary.sort_values("eval_regret_mean")
    ylabels = []
    for _, r in d.iterrows():
        ylabels.append(_series_label(r.model, r.feature_set, show_objective=True, objective=r.objective))
    for i, (_, r) in enumerate(d.iterrows()):
        ax.barh(ylabels[i], 100 * r.eval_regret_mean, height=0.7, **_bar_kwargs(r.model, r.feature_set))
    ax.axvline(100 * d["eval_regret_mean"].min(), color="0.35", ls="--", lw=1.0)
    ax.axvline(71.5, color=COL_RANDOM, ls=":", lw=1.2)
    ax.set_xlabel("Mean Selection Regret (%)")
    ax.invert_yaxis()
    ax.legend(handles=_legend_patches(), loc="lower right")
    _save(fig, "eval_mean_regret_all_training_runs")


def plot_val_vs_eval(summary: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    _style(ax)
    for _, r in summary.iterrows():
        kw = _line_kwargs(r.model, r.feature_set, r.objective)
        ax.scatter(
            r["best_val_ndcg@10"], 100 * r["eval_regret_mean"],
            c=kw["color"], marker=kw["marker"], s=80, edgecolors="white", lw=1.0,
        )
    ax.set_xlabel("Best Validation NDCG@10")
    ax.set_ylabel("Held-Out Mean Regret (%)")
    ax.legend(handles=_legend_patches()[:4], loc="upper right")
    _save(fig, "val_ndcg_vs_eval_regret_scatter")


def plot_training_curves() -> None:
    for model, hist_name in [("MLP", "training_history_mlp.csv"), ("XGBoost", "training_history_xgb.csv")]:
        hist = pd.read_csv(PAPER / hist_name)
        hist["run_key"] = hist.apply(_hist_run_key, axis=1)
        fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
        _style(ax)
        step_col = "epoch" if model == "MLP" else "round"
        for run_dir, g in hist.groupby("run_key"):
            meta = _parse_run(run_dir)
            kw = _line_kwargs(meta["model"], meta["feature_set"], meta["objective"])
            ax.plot(
                g[step_col], g["val_ndcg@10"],
                label=_series_label(meta["model"], meta["feature_set"], show_objective=True, objective=meta["objective"]),
                **kw,
            )
        ax.set_xlabel("Epoch" if model == "MLP" else "Boosting Round")
        ax.set_ylabel("Validation NDCG@10")
        ax.legend(ncol=3, loc="lower right", framealpha=0.92, columnspacing=0.8, handletextpad=0.4)
        _save(fig, f"training_val_ndcg_{model.lower()}_by_objective_and_feature_set")


def plot_regret_boxplot(eval_df: pd.DataFrame, nvmmh_df: pd.DataFrame) -> None:
    key_runs = list(KEY_RUNS.values())
    combo = eval_df[eval_df.run_dir.isin(key_runs)].copy()
    if len(nvmmh_df):
        combo = pd.concat([combo, nvmmh_df], ignore_index=True)

    labels = []
    for run in key_runs:
        meta = _parse_run(run)
        labels.append(_series_label(meta["model"], meta["feature_set"]))
    if len(nvmmh_df):
        labels.append("nvMMH")

    order = []
    label_map = dict(zip(key_runs, labels[: len(key_runs)]))
    for run in key_runs:
        order.append(label_map[run])
    if len(nvmmh_df):
        order.append("nvMMH")

    data = []
    colors = []
    for run in key_runs:
        meta = _parse_run(run)
        data.append(100 * combo.loc[combo.run_dir == run, "regret"].values)
        colors.append(_bar_kwargs(meta["model"], meta["feature_set"])["color"])
    if len(nvmmh_df):
        data.append(100 * nvmmh_df.regret.values)
        colors.append(COL_NVMMH)

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    _style(ax)
    bp = ax.boxplot(data, tick_labels=order, vert=True, patch_artist=True, showfliers=False)
    for i, (box, run) in enumerate(zip(bp["boxes"], key_runs + (["nvmmh"] if len(nvmmh_df) else []))):
        if run == "nvmmh":
            box.set_facecolor(COL_NVMMH)
        else:
            meta = _parse_run(run)
            box.set_facecolor(_fill_color(meta["model"], meta["feature_set"]))
        box.set_alpha(0.85)
    ax.set_ylabel("Per-Group Regret (%)")
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=25, ha="right")
    _save(fig, "eval_regret_boxplot_mlp_xgb_nvmmh")


def plot_by_regime_nvmmh() -> None:
    runs = [
        ("mlp_ranknet", "MLP", "full"),
        ("mlp_mse_structural", "MLP", "structural"),
        ("xgb_mse", "XGBoost", "full"),
        ("xgb_mse_structural", "XGBoost", "structural"),
    ]
    metrics = {run: json.loads((PAPER / run / "metrics.json").read_text()) for run, _, _ in runs}
    nv = json.loads((PAPER / "nvmmh" / "metrics_top1.json").read_text())
    regimes = sorted(metrics["mlp_ranknet"]["by_regime"].keys())

    fig, ax = plt.subplots(figsize=(FIG_W + 1.5, FIG_H))
    _style(ax)
    x = np.arange(len(regimes))
    n = len(runs) + 1
    w = 0.8 / n
    offsets = np.linspace(-(n - 1) / 2, (n - 1) / 2, n) * w

    for i, (run, model, fs) in enumerate(runs):
        vals = [100 * metrics[run]["by_regime"][r]["regret_mean"] for r in regimes]
        ax.bar(x + offsets[i], vals, w, label=_series_label(model, fs), **_bar_kwargs(model, fs))

    nv_r = [100 * nv["by_regime"][r]["regret_mean"] for r in regimes]
    ax.bar(x + offsets[-1], nv_r, w, label="nvMMH", **_bar_kwargs("nvMMH", "full"))

    ax.set_xticks(x)
    ax.set_xticklabels(regimes)
    ax.set_ylabel("Mean Regret (%)")
    ax.set_xlabel("Shape Regime")
    ax.legend(ncol=2, loc="upper right", framealpha=0.92)
    _save(fig, "eval_regret_by_regime_mlp_xgb_nvmmh")


def plot_gpu_sweep() -> None:
    p = PAPER / "nvmmh" / "gpu_sweep.csv"
    if not p.exists():
        return
    df = pd.read_csv(p).sort_values("regret_mean")
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    _style(ax)
    colors = [COL_NVMMH if g == "H100_NVL" else COL_NVMMH_ALT for g in df.gpu]
    ax.barh(df.gpu, 100 * df.regret_mean, color=colors, edgecolor="white", height=0.65)
    ax.axvline(100 * df.regret_mean.min(), color="0.35", ls="--", lw=1.0)
    ax.set_xlabel("Mean Regret (%)")
    ax.invert_yaxis()
    _save(fig, "nvmmh_mean_regret_by_gpu_preset")


def plot_by_regime() -> None:
    rows = []
    for run in RUNS:
        mpath = PAPER / run / "metrics.json"
        if not mpath.exists():
            continue
        m = json.loads(mpath.read_text())
        meta = _parse_run(run)
        for regime, stats in m.get("by_regime", {}).items():
            rows.append({**meta, "regime": regime, "regret_mean": stats["regret_mean"],
                         "within5pct": stats.get("within5pct", np.nan)})
    df = pd.DataFrame(rows)
    if df.empty:
        return

    picks = []
    for model in ["MLP", "XGBoost"]:
        for fs in ["full", "structural"]:
            sub = df[(df.model == model) & (df.feature_set == fs)]
            picks.append(sub.groupby("run_dir")["regret_mean"].mean().idxmin())
    ordered_picks = _sort_runs(picks)
    sub = df[df.run_dir.isin(ordered_picks)]

    regime_col = {"regret_mean": "regret", "within5pct": "within5pct"}
    fig, axes = plt.subplots(1, 2, figsize=FIG_TWIN)
    for ax, metric, ylab in zip(axes, ["regret_mean", "within5pct"], ["Regret (%)", "Within 5% of Oracle (%)"]):
        _style(ax)
        pivot = sub.pivot(index="regime", columns="run_dir", values=metric)
        pivot = pivot[ordered_picks]
        regimes = list(pivot.index)
        x = np.arange(len(regimes))
        w = 0.8 / len(ordered_picks)
        for i, run_dir in enumerate(ordered_picks):
            meta = _parse_run(run_dir)
            vals, errs = [], []
            for regime in regimes:
                mean, sem = _regime_column_mean_sem(run_dir, regime, regime_col[metric])
                vals.append(100 * mean)
                errs.append(100 * sem)
            offset = (i - (len(ordered_picks) - 1) / 2) * w
            ax.bar(
                x + offset,
                np.array(vals),
                w,
                yerr=np.array(errs),
                capsize=3,
                error_kw=_ERR_KW,
                label=_series_label(meta["model"], meta["feature_set"]),
                **_bar_kwargs(meta["model"], meta["feature_set"]),
            )
        ax.set_xticks(x)
        ax.set_xticklabels(regimes)
        ax.set_xlabel("Shape Regime")
        ax.set_ylabel(ylab)
        ax.set_ylim(0, None)
        ax.legend(ncol=2, loc="upper right", framealpha=0.92)
    fig.subplots_adjust(wspace=0.32)
    _save(fig, "eval_by_regime_best_full_and_structural")


def plot_success_metrics(summary: pd.DataFrame) -> None:
    metrics = []
    for run in RUNS:
        m = json.loads((PAPER / run / "metrics.json").read_text())
        meta = _parse_run(run)
        e = m["eval"]
        metrics.append({**meta, "top1": 100 * e["top1"], "top5": 100 * e["top5"], "within5pct": 100 * e["within5pct"]})
    mdf = pd.DataFrame(metrics).sort_values("within5pct", ascending=False)

    fig, ax = plt.subplots(figsize=(FIG_W + 3.0, FIG_H + 1.5))
    _style(ax)
    x = np.arange(len(mdf))
    w = 0.25
    shades = {"top1": 1.0, "top5": 0.75, "within5pct": 0.5}
    for j, (metric, alpha_scale) in enumerate(shades.items()):
        for xi, (_, row) in enumerate(mdf.iterrows()):
            color = _fill_color(row.model, row.feature_set)
            ax.bar(xi + (j - 1) * w, row[metric], w, color=color, edgecolor=color, alpha=alpha_scale)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [_series_label(r.model, r.feature_set, show_objective=True, objective=r.objective) for _, r in mdf.iterrows()],
        rotation=45, ha="right",
    )
    ax.set_ylabel("Rate (%)")
    ax.legend(handles=_legend_patches()[:4], loc="upper right")
    _save(fig, "eval_success_rates_all_training_runs")


def plot_feature_set_comparison(summary: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=FIG_TWIN)
    for ax, model, col in zip(axes, ["MLP", "XGBoost"], ["eval_regret_mean", "best_val_ndcg@10"]):
        _style(ax)
        sub = summary[summary.model == model]
        objs = sorted(sub.objective.unique())
        x = np.arange(len(objs))
        w = 0.35
        scale = 100 if col == "eval_regret_mean" else 1
        for fs, offset in [("full", -w / 2), ("structural", w / 2)]:
            vals, errs = [], []
            for o in objs:
                row = sub[(sub.objective == o) & (sub.feature_set == fs)]
                if col == "eval_regret_mean":
                    run = row["run_dir"].iloc[0]
                    mean, sem = _regret_mean_sem(PAPER / run / "eval_regret.csv")
                    vals.append(100 * mean)
                    errs.append(100 * sem)
                else:
                    vals.append(float(row[col].iloc[0]) * scale)
                    errs.append(0.0)
            ax.bar(
                x + offset,
                np.array(vals),
                w,
                yerr=np.array(errs),
                capsize=3,
                error_kw=_ERR_KW,
                label=_series_label(model, fs),
                **_bar_kwargs(model, fs),
            )
        ax.set_xticks(x)
        ax.set_xticklabels([_objective_label(o) for o in objs])
        ax.set_ylabel("Regret (%)" if col == "eval_regret_mean" else "Val NDCG@10")
        ax.legend(loc="upper right", framealpha=0.92)
    fig.subplots_adjust(wspace=0.32)
    _save(fig, "eval_full_vs_structural_by_objective")


def plot_hard_groups(eval_df: pd.DataFrame) -> None:
    runs = ["mlp_ranknet", "mlp_mse_structural", "xgb_mse", "xgb_mse_structural"]
    pivot = eval_df[eval_df.run_dir.isin(runs)].pivot_table(
        index=["M", "N", "K", "regime"], columns="run_dir", values="regret"
    )
    pivot["best"] = pivot.min(axis=1)
    hard = pivot.nlargest(12, "best")

    fig, ax = plt.subplots(figsize=FIG_TALL)
    _style(ax)
    labels = [f"{r} ({m}×{n}×{k})" for (m, n, k, r) in hard.index]
    ax.barh(labels, 100 * hard["best"].values, color=COL_MLP, height=0.7)
    ax.set_xlabel("Best-Model Regret (%)")
    ax.invert_yaxis()
    _save(fig, "eval_hardest_groups_best_of_mlp_xgb_full_and_structural")


def plot_regret_cdf(eval_df: pd.DataFrame, nvmmh_df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=FIG_PANEL)
    _style(ax)
    picks = [
        ("mlp_ranknet", "MLP", "full", "ranknet"),
        ("mlp_mse_structural", "MLP", "structural", "mse"),
        ("xgb_mse", "XGBoost", "full", "mse"),
        ("xgb_mse_structural", "XGBoost", "structural", "mse"),
    ]
    for run, model, fs, obj in picks:
        r = np.sort(100 * eval_df.loc[eval_df.run_dir == run, "regret"].values)
        ax.plot(
            r, np.linspace(0, 1, len(r)),
            label=_series_label(model, fs),
            **_line_kwargs(model, fs, obj),
        )
    if len(nvmmh_df):
        r = np.sort(100 * nvmmh_df.regret.values)
        ax.plot(r, np.linspace(0, 1, len(r)), label="nvMMH", color=COL_NVMMH, lw=3.0, ls="-.")
    ridge_picks = [
        (PAPER / "baselines" / "linear_full" / "eval_regret.csv", "Ridge", COL_LINEAR, "-"),
        (PAPER / "baselines" / "linear_structural" / "eval_regret.csv", "Ridge(s)", COL_LINEAR_LIGHT, "--"),
    ]
    for path, label, color, ls in ridge_picks:
        if not path.exists():
            continue
        r = np.sort(100 * pd.read_csv(path)["regret"].values)
        ax.plot(r, np.linspace(0, 1, len(r)), label=label, color=color, lw=2.0, ls=ls)
    ax.set_xlim(0, 50)
    ax.set_xlabel("Per-Group Regret (%)")
    ax.set_ylabel("CDF")
    leg = ax.legend(loc="lower right", framealpha=0.92)
    _apply_panel_fonts(ax, legend=leg)
    _save(fig, "eval_regret_cdf_mlp_xgb_nvmmh")


def plot_mlp_width_scaling() -> None:
    """MLP capacity sweep: full vs structural regret vs parameter count."""
    for path in (PAPER / "mlp_width_scaling.csv", PAPER / "mlp_capacity_scaling.csv"):
        if not path.exists():
            continue
        df = pd.read_csv(path)
        req = {"hidden", "params", "full", "structural"}
        if not req.issubset(df.columns):
            print(f"skip {path.name}: need columns {sorted(req)}")
            return
        df = df.sort_values("params")
        fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
        _style(ax)
        x = np.arange(len(df))
        w = 0.35
        ax.bar(x - w / 2, 100 * df["full"], w, label="MLP", **_bar_kwargs("MLP", "full"))
        ax.bar(x + w / 2, 100 * df["structural"], w, label="MLP(s)", **_bar_kwargs("MLP", "structural"))
        ax.set_xticks(x)
        ax.set_xticklabels(df["hidden"], rotation=25, ha="right")
        ax.set_ylabel("Mean Regret (%)")
        ax.set_xlabel("Hidden layers")
        ax.legend()
        _save(fig, "eval_mlp_regret_vs_hidden_width_full_vs_structural")
        return
    print("no MLP width-scaling CSV in artifacts/analysis/paper/ (expected mlp_width_scaling.csv)")


def plot_within5_comparison() -> None:
    entries = [
        ("mlp_ranknet", "MLP", "full"),
        ("mlp_mse_structural", "MLP", "structural"),
        ("xgb_mse", "XGBoost", "full"),
        ("xgb_mse_structural", "XGBoost", "structural"),
        ("nvmmh", "nvMMH", "full"),
    ]
    labels, vals, errs, styles = [], [], [], []
    for key, model, fs in entries:
        label = "nvMMH" if key == "nvmmh" else _series_label(model, fs)
        mean, sem = _column_mean_sem(_eval_csv_for_key(key), "within5pct")
        labels.append(label)
        vals.append(100 * mean)
        errs.append(100 * sem)
        styles.append(_bar_kwargs(model, fs))

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    _style(ax)
    for label, val, err, kw in zip(labels, vals, errs, styles):
        ax.bar(label, val, width=0.55, yerr=err, capsize=3, error_kw=_ERR_KW, **kw)
    ax.set_ylabel("Groups Within 5% of Oracle (%)")
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=20, ha="right")
    _save(fig, "eval_within5pct_mlp_xgb_nvmmh")


def main() -> None:
    _setup_rc()
    OUT.mkdir(parents=True, exist_ok=True)
    _cleanup_old_figures()
    summary = load_summary()
    eval_df = load_all_eval()
    nvmmh_df = load_nvmmh_eval("top1")
    plot_main_comparison()
    plot_baseline_mean_regret()
    plot_regret_summary(summary)
    plot_val_vs_eval(summary)
    plot_training_curves()
    plot_regret_boxplot(eval_df, nvmmh_df)
    plot_by_regime()
    plot_by_regime_nvmmh()
    plot_gpu_sweep()
    plot_success_metrics(summary)
    plot_feature_set_comparison(summary)
    plot_within5_comparison()
    plot_hard_groups(eval_df)
    plot_regret_cdf(eval_df, nvmmh_df)
    plot_mlp_width_scaling()
    print(f"Wrote figures to {OUT}")


if __name__ == "__main__":
    main()
