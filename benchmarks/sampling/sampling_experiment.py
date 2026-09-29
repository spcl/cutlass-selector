#!/usr/bin/env python3
"""
Offline sampling-coverage experiment on exhaustive eval measurements.

Replays measurement-allocation policies against BF16 eval groups and writes
tables, plots and a JSON summary (tex_snippets.json).

Usage (local):
    python benchmarks/sampling/sampling_experiment.py --db ~/autotuner_bf16_eval.db

Usage (cluster, all cores):
    sbatch benchmarks/sampling/run_sampling_experiment.sh
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from repo_paths import SAMPLING, SRC

sys.path.insert(0, str(SRC / "planning" / "plan"))

from write import _build_arrays, _score_all  # noqa: E402

BUDGETS = [100, 250, 500, 1000, 2000, 3000, 5000, 10000]
PAPER_BUDGETS = [1000, 2000, 3000]
EPSILONS = [0.01, 0.05, 0.10, 0.20]
ALPHA_GRID = [0.0, 0.25, 0.5, 0.75, 1.0]
ALPHA_BUDGET = 2000
BIASED_TOP_FRAC = 0.75
LAYOUT_TAG = {
    ("cutlass::layout::RowMajor", "cutlass::layout::ColumnMajor"): "TN",
    ("cutlass::layout::RowMajor", "cutlass::layout::RowMajor"): "TT",
    ("cutlass::layout::ColumnMajor", "cutlass::layout::ColumnMajor"): "NN",
    ("cutlass::layout::ColumnMajor", "cutlass::layout::RowMajor"): "NT",
}


@dataclass(frozen=True)
class GroupData:
    key: tuple[int, int, int, str]
    tflops: np.ndarray
    sorted_idx: np.ndarray
    family_codes_sorted: np.ndarray
    bucket_splits: dict[int, list[np.ndarray]]  # B -> list of bucket index arrays (global idx)


def _log(msg: str) -> None:
    print(msg, flush=True)


def _layout_tag(la: str, lb: str) -> str:
    return LAYOUT_TAG[(la, lb)]


def _load_groups(db: Path, min_coverage: float) -> list[GroupData]:
    _log(f"Loading measurements from {db} ...")
    t0 = time.perf_counter()
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    df = pd.read_sql(
        """
        SELECT r.M, r.N, r.K, r.name, r.mean_tflops,
               c.layout_a, c.layout_b, c.tile_m, c.tile_n, c.tile_k,
               c.cluster_m, c.cluster_n, c.stages, c.kernel_schedule,
               c.epilogue_schedule, c.scheduler
        FROM runs r
        JOIN configs c ON r.name = c.name
        WHERE r.status = 'success' AND r.mean_tflops > 0
        """,
        conn,
    )
    plan = pd.read_sql(
        "SELECT M, N, K, layout, COUNT(*) AS planned FROM eval_plan GROUP BY M, N, K, layout",
        conn,
    )
    conn.close()
    _log(f"  read {len(df):,} successful runs in {time.perf_counter() - t0:.1f}s")

    df["layout"] = [_layout_tag(a, b) for a, b in zip(df.layout_a, df.layout_b)]
    plan_idx = plan.set_index(["M", "N", "K", "layout"])["planned"]

    groups: list[GroupData] = []
    for gi, (key, g) in enumerate(df.groupby(["M", "N", "K", "layout"], sort=False), 1):
        planned_n = int(plan_idx.get(key, 0))
        if planned_n and len(g) / planned_n < min_coverage:
            continue
        M, N, K, layout = key
        g = g.reset_index(drop=True)
        scores = _score_all(M, N, _build_arrays(g.to_dict("records")), "new_we")
        valid = np.flatnonzero(scores > -1)
        if len(valid) < 50:
            continue
        order = valid[np.argsort(scores[valid])[::-1]].astype(np.int64)
        families = (
            g["kernel_schedule"].astype(str) + "|" + g["scheduler"].astype(str)
        )
        codes, _ = pd.factorize(families, sort=False)
        groups.append(
            GroupData(
                key=(int(M), int(N), int(K), str(layout)),
                tflops=g["mean_tflops"].to_numpy(dtype=np.float64),
                sorted_idx=order,
                family_codes_sorted=codes[order].astype(np.int64),
                bucket_splits=_precompute_bucket_splits(order, BUDGETS, BIASED_TOP_FRAC),
            )
        )
        if gi % 10 == 0:
            _log(f"  prepared {len(groups)} groups so far ...")
    _log(f"Prepared {len(groups)} groups (min coverage {min_coverage:.0%})")
    return groups


def _precompute_bucket_splits(sorted_idx: np.ndarray, budgets: list[int], top_frac: float) -> dict[int, list[np.ndarray]]:
    out: dict[int, list[np.ndarray]] = {}
    n = len(sorted_idx)
    for B in budgets:
        if B > n:
            continue
        k_top = round(B * top_frac)
        remaining = sorted_idx[k_top:]
        if len(remaining) == 0:
            out[B] = []
            continue
        n_buckets = min(10, len(remaining))
        out[B] = list(np.array_split(remaining, n_buckets))
    return out


def _regret_vec(tflops: np.ndarray, picks: np.ndarray) -> np.ndarray:
    oracle = float(tflops.max())
    if oracle <= 0:
        return np.zeros(len(picks), dtype=np.float64)
    if picks.ndim == 1:
        return np.array([1.0 - float(tflops[picks].max()) / oracle])
    return 1.0 - tflops[picks].max(axis=1) / oracle


def _within_eps_vec(tflops: np.ndarray, picks: np.ndarray, eps: float) -> np.ndarray:
    oracle = float(tflops.max())
    if picks.ndim == 1:
        return np.array([float(tflops[picks].max() >= (1.0 - eps) * oracle)])
    return tflops[picks].max(axis=1) >= (1.0 - eps) * oracle


def _uniform_picks(sorted_idx: np.ndarray, B: int, n_resamples: int, seed: int) -> np.ndarray:
    n = len(sorted_idx)
    rng = np.random.default_rng(seed)
    keys = rng.random((n_resamples, n))
    order = np.argsort(keys, axis=1)
    return sorted_idx[order[:, :B]]


def _biased_picks(
    sorted_idx: np.ndarray,
    bucket_splits: list[np.ndarray],
    B: int,
    top_frac: float,
    n_resamples: int,
    seed: int,
) -> np.ndarray:
    n = len(sorted_idx)
    B = min(B, n)
    k_top = round(B * top_frac)
    top = sorted_idx[:k_top]
    k_bad = B - k_top
    out = np.empty((n_resamples, B), dtype=np.int64)
    if k_bad <= 0:
        out[:] = top[:B]
        return out
    rng = np.random.default_rng(seed)
    buckets = bucket_splits
    if not buckets:
        out[:] = top[:B]
        return out
    n_buckets = len(buckets)
    per_base = k_bad // n_buckets
    remainder = k_bad % n_buckets
    for r in range(n_resamples):
        chosen = list(top)
        sampled = 0
        for d, bucket in enumerate(buckets):
            if sampled >= k_bad:
                break
            n_samp = min(per_base + (1 if d < remainder else 0), len(bucket))
            if n_samp <= 0:
                continue
            chosen.extend(rng.choice(bucket, size=n_samp, replace=False).tolist())
            sampled += n_samp
        out[r, : len(chosen)] = chosen[:B]
    return out


def _stratified_picks(
    sorted_idx: np.ndarray,
    family_codes: np.ndarray,
    B: int,
    n_resamples: int,
    seed: int,
) -> np.ndarray:
    n = len(sorted_idx)
    B = min(B, n)
    rng = np.random.default_rng(seed)
    families = np.unique(family_codes)
    per = max(1, B // len(families))
    out = np.empty((n_resamples, B), dtype=np.int64)
    for r in range(n_resamples):
        chosen: list[int] = []
        for fam in families:
            fam_idx = sorted_idx[family_codes == fam]
            if len(fam_idx) == 0:
                continue
            n = min(per, len(fam_idx))
            chosen.extend(rng.choice(fam_idx, size=n, replace=False).tolist())
        if len(chosen) < B:
            rest = np.setdiff1d(sorted_idx, np.array(chosen, dtype=np.int64), assume_unique=False)
            need = min(B - len(chosen), len(rest))
            if need:
                chosen.extend(rng.choice(rest, size=need, replace=False).tolist())
        out[r, : len(chosen)] = chosen[:B]
    return out


def _process_group(group: GroupData, n_resamples: int) -> dict:
    M, N, K, layout = group.key
    tflops = group.tflops
    idx = group.sorted_idx
    n = len(idx)
    seed_base = hash(group.key) % (2**31 - 1)

    tail_row = {"M": M, "N": N, "K": K, "layout": layout, "n": n}
    oracle = float(tflops.max())
    for eps in EPSILONS:
        tail_row[f"frac_within_{int(eps * 100)}pct"] = float((tflops >= (1 - eps) * oracle).sum() / n)

    curve_rows = []
    table_rows = []
    alpha_rows = []

    for B in BUDGETS:
        if B > n:
            continue
        buckets = group.bucket_splits.get(B, [])
        top_pick = idx[:B]

        policy_picks = {
            "top_static": top_pick,
            "uniform": _uniform_picks(idx, B, n_resamples, seed_base + 17 * B),
            "biased_75_25": _biased_picks(idx, buckets, B, BIASED_TOP_FRAC, n_resamples, seed_base + 31 * B),
            "stratified_family": _stratified_picks(
                idx, group.family_codes_sorted, B, n_resamples, seed_base + 47 * B
            ),
        }

        for policy, picks in policy_picks.items():
            if picks.ndim == 1:
                regrets = _regret_vec(tflops, picks)
                within5 = _within_eps_vec(tflops, picks, 0.05)
            else:
                regrets = _regret_vec(tflops, picks)
                within5 = _within_eps_vec(tflops, picks, 0.05)
            row = {
                "M": M, "N": N, "K": K, "layout": layout,
                "budget": B, "policy": policy,
                "regret_mean": float(np.mean(regrets)),
                "regret_median": float(np.median(regrets)),
                "within5_pct": 100.0 * float(np.mean(within5)),
                "within1_pct": 100.0 * float(np.mean(_within_eps_vec(tflops, picks, 0.01))),
                "within10_pct": 100.0 * float(np.mean(_within_eps_vec(tflops, picks, 0.10))),
            }
            curve_rows.append(row)
            if B in PAPER_BUDGETS:
                table_rows.append(row)

    B = min(ALPHA_BUDGET, n)
    buckets = group.bucket_splits.get(B, [])
    for alpha in ALPHA_GRID:
        if alpha == 0.0:
            picks = _uniform_picks(idx, B, n_resamples, seed_base + 91)
        elif alpha == 1.0:
            picks = idx[:B]
        else:
            picks = _biased_picks(idx, buckets, B, alpha, n_resamples, seed_base + int(103 * alpha * 100))
        regrets = _regret_vec(tflops, picks)
        within5 = _within_eps_vec(tflops, picks, 0.05)
        alpha_rows.append({
            "M": M, "N": N, "K": K, "layout": layout,
            "alpha": alpha,
            "regret_mean": float(np.mean(regrets)),
            "within5_pct": 100.0 * float(np.mean(within5)),
        })

    return {"tail": tail_row, "curve": curve_rows, "table": table_rows, "alpha": alpha_rows}


def _run_parallel(groups: list[GroupData], n_resamples: int, jobs: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    tail_rows, curve_rows, table_rows, alpha_rows = [], [], [], []
    n = len(groups)
    t0 = time.perf_counter()

    if jobs <= 1:
        for i, g in enumerate(groups, 1):
            out = _process_group(g, n_resamples)
            tail_rows.append(out["tail"])
            curve_rows.extend(out["curve"])
            table_rows.extend(out["table"])
            alpha_rows.extend(out["alpha"])
            if i == 1 or i % 5 == 0 or i == n:
                _log(f"  [{i}/{n}] groups done  ({time.perf_counter() - t0:.1f}s elapsed)")
    else:
        _log(f"Running {n} groups on {jobs} workers ({n_resamples} resamples each) ...")
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            futures = {pool.submit(_process_group, g, n_resamples): g.key for g in groups}
            done = 0
            for fut in as_completed(futures):
                out = fut.result()
                tail_rows.append(out["tail"])
                curve_rows.extend(out["curve"])
                table_rows.extend(out["table"])
                alpha_rows.extend(out["alpha"])
                done += 1
                if done == 1 or done % 5 == 0 or done == n:
                    _log(f"  [{done}/{n}] groups done  ({time.perf_counter() - t0:.1f}s elapsed)")

    tail = pd.DataFrame(tail_rows)
    curve = pd.DataFrame(curve_rows)
    table = curve.groupby(["budget", "policy"], as_index=False).agg(
        regret_mean=("regret_mean", "mean"),
        regret_median=("regret_median", "mean"),
        within5_pct=("within5_pct", "mean"),
        within1_pct=("within1_pct", "mean"),
        within10_pct=("within10_pct", "mean"),
    )
    alpha = pd.DataFrame(alpha_rows).groupby("alpha", as_index=False).agg(
        regret_mean=("regret_mean", "mean"),
        within5_pct=("within5_pct", "mean"),
    )
    _log(f"Experiment finished in {time.perf_counter() - t0:.1f}s")
    return tail, curve, table, alpha


def _pct(x: float, digits: int = 1) -> str:
    return f"{100 * x:.{digits}f}"


def build_tex_snippets(tail: pd.DataFrame, table: pd.DataFrame, alpha: pd.DataFrame, n_resamples: int, n_groups: int) -> dict:
    def pol_at(budget: int, policy: str, col: str) -> float:
        row = table[(table.budget == budget) & (table.policy == policy)]
        return float(row[col].iloc[0]) if len(row) else float("nan")

    tail_stats = {}
    for eps in (1, 5, 10):
        col = f"frac_within_{eps}pct"
        tail_stats[f"tail_mean_{eps}"] = 100 * float(tail[col].mean())
        tail_stats[f"tail_median_{eps}"] = 100 * float(tail[col].median())
        tail_stats[f"tail_min_{eps}"] = 100 * float(tail[col].min())

    uni_2k = pol_at(2000, "uniform", "regret_mean")
    bias_2k = pol_at(2000, "biased_75_25", "regret_mean")
    top_2k = pol_at(2000, "top_static", "regret_mean")

    return {
        "n_groups": n_groups,
        "n_resamples": n_resamples,
        "biased_top_frac_pct": int(BIASED_TOP_FRAC * 100),
        "biased_explore_frac_pct": int((1 - BIASED_TOP_FRAC) * 100),
        **tail_stats,
        "uniform_regret_1k": 100 * pol_at(1000, "uniform", "regret_mean"),
        "uniform_regret_2k": 100 * uni_2k,
        "uniform_regret_3k": 100 * pol_at(3000, "uniform", "regret_mean"),
        "top_regret_2k": 100 * top_2k,
        "biased_regret_1k": 100 * pol_at(1000, "biased_75_25", "regret_mean"),
        "biased_regret_2k": 100 * bias_2k,
        "biased_regret_3k": 100 * pol_at(3000, "biased_75_25", "regret_mean"),
        "uniform_within5_2k": pol_at(2000, "uniform", "within5_pct"),
        "top_within5_2k": pol_at(2000, "top_static", "within5_pct"),
        "biased_within5_2k": pol_at(2000, "biased_75_25", "within5_pct"),
        "uniform_within5_3k": pol_at(3000, "uniform", "within5_pct"),
        "top_within5_3k": pol_at(3000, "top_static", "within5_pct"),
        "biased_within5_3k": pol_at(3000, "biased_75_25", "within5_pct"),
        "biased_vs_uniform_regret_reduction_2k_pct": 100 * (uni_2k - bias_2k) / uni_2k if uni_2k else float("nan"),
        "biased_vs_top_regret_reduction_2k_pct": 100 * (top_2k - bias_2k) / top_2k if top_2k else float("nan"),
        "alpha_rows": alpha.to_dict(orient="records"),
    }


def write_outputs(
    outdir: Path,
    tail: pd.DataFrame,
    curve: pd.DataFrame,
    table: pd.DataFrame,
    alpha: pd.DataFrame,
    snippets: dict,
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    tail.to_csv(outdir / "tail_sparsity.csv", index=False)
    curve.to_csv(outdir / "sampling_curve.csv", index=False)
    table.to_csv(outdir / "sampling_table.csv", index=False)
    alpha.to_csv(outdir / "alpha_ablation.csv", index=False)
    (outdir / "tex_snippets.json").write_text(json.dumps(snippets, indent=2))

    # Tail table (LaTeX)
    lines = [
        "% auto-generated by benchmarks/sampling/sampling_experiment.py",
        "\\begin{table}",
        "\\centering",
        "\\footnotesize",
        "\\caption{Fraction of valid kernels close to the exhaustive oracle.}",
        "\\label{tab:sampling-tail}",
        "\\begin{tabular}{@{}lccc@{}}",
        "\\toprule",
        "Threshold & Mean fraction & Median fraction & Minimum fraction \\\\",
        "\\midrule",
    ]
    for eps in (1, 5, 10):
        lines.append(
            f"Within {eps}\\%  & {snippets[f'tail_mean_{eps}']:.2f}\\% & "
            f"{snippets[f'tail_median_{eps}']:.2f}\\% & {snippets[f'tail_min_{eps}']:.2f}\\% \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    (outdir / "tail_table.tex").write_text("\n".join(lines) + "\n")

    try_plot(curve, tail, alpha, outdir)


def _plot_style():
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.labelsize": 13,
            "legend.fontsize": 11,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
        }
    )
    return plt


def _style_axes(ax) -> None:
    ax.set_facecolor("#ebebeb")
    ax.figure.patch.set_facecolor("white")


_POLICY_STYLE = {
    "uniform": ("Uniform", "o", "#1e3a8a"),
    "biased_75_25": ("75/25 Proxy", "s", "#14532d"),
    "top_static": ("Static Top-$B$", "^", "#525252"),
}


def _curve_agg_sem(curve: pd.DataFrame, metric: str) -> pd.DataFrame:
    rows = []
    for (budget, policy), grp in curve.groupby(["budget", "policy"]):
        vals = grp[metric].to_numpy(dtype=float)
        rows.append(
            {
                "budget": int(budget),
                "policy": policy,
                "mean": float(np.mean(vals)),
                "sem": float(np.std(vals, ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0,
            }
        )
    return pd.DataFrame(rows)


def _plot_curve_with_band(
    ax,
    agg: pd.DataFrame,
    policy: str,
    *,
    y_scale: float = 1.0,
    plot_kw: dict,
) -> None:
    label, marker, color = _POLICY_STYLE[policy]
    sub = agg[agg.policy == policy].sort_values("budget")
    if sub.empty:
        return
    x = sub["budget"].to_numpy(dtype=float)
    y = y_scale * sub["mean"].to_numpy(dtype=float)
    sem = y_scale * sub["sem"].to_numpy(dtype=float)
    ax.fill_between(x, y - sem, y + sem, color=color, alpha=0.22, linewidth=0, zorder=2)
    ax.plot(x, y, marker=marker, label=label, color=color, zorder=3, **plot_kw)


def try_plot(curve: pd.DataFrame, tail: pd.DataFrame, alpha: pd.DataFrame, outdir: Path) -> None:
    try:
        plt = _plot_style()
    except ImportError:
        _log("matplotlib not available — skipping plots")
        return

    regret_agg = _curve_agg_sem(curve, "regret_mean")
    within5_agg = _curve_agg_sem(curve, "within5_pct")
    plot_kw = {"markersize": 4, "linewidth": 1.6}

    fig, ax = plt.subplots(figsize=(6, 4))
    _style_axes(ax)
    for pol in _POLICY_STYLE:
        _plot_curve_with_band(ax, regret_agg, pol, y_scale=100.0, plot_kw=plot_kw)
    ax.set_xscale("log")
    ax.set_xlabel("Measurement Budget ($B$)")
    ax.set_ylabel("Sampling Regret (%)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "sampling_regret_vs_budget.pdf")
    fig.savefig(outdir / "sampling_regret_vs_budget.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    _style_axes(ax)
    for pol in _POLICY_STYLE:
        _plot_curve_with_band(ax, within5_agg, pol, y_scale=1.0, plot_kw=plot_kw)
    ax.set_xscale("log")
    ax.set_xlabel("Measurement Budget ($B$)")
    ax.set_ylabel("Groups With Kernel\nWithin 5% of Oracle (%)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "sampling_within5_vs_budget.pdf")
    fig.savefig(outdir / "sampling_within5_vs_budget.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 3.5))
    _style_axes(ax)
    cols = [f"frac_within_{p}pct" for p in (1, 5, 10, 20)]
    ax.boxplot([100 * tail[c].to_numpy() for c in cols], tick_labels=["1%", "5%", "10%", "20%"])
    ax.set_ylabel("Catalogue Fraction Within $\\epsilon$ of Oracle")
    ax.set_xlabel("Near-Optimal Band")
    fig.tight_layout()
    fig.savefig(outdir / "tail_sparsity_boxplot.pdf")
    fig.savefig(outdir / "tail_sparsity_boxplot.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    _style_axes(ax)
    ax.plot(
        100 * alpha.alpha.to_numpy(),
        100 * alpha.regret_mean.to_numpy(),
        marker="o",
        markersize=4,
        linewidth=1.6,
    )
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel("Fraction From Proxy Top Region (%)")
    ax.set_ylabel(f"Mean Sampling Regret @ $B$={ALPHA_BUDGET} (%)")
    fig.tight_layout()
    fig.savefig(outdir / "alpha_ablation.pdf")
    fig.savefig(outdir / "alpha_ablation.png", dpi=150)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", type=Path, default=Path.home() / "autotuner_bf16_eval.db")
    ap.add_argument("--outdir", type=Path, default=SAMPLING)
    ap.add_argument("--min-coverage", type=float, default=0.85)
    ap.add_argument("--resamples", type=int, default=100)
    ap.add_argument("--jobs", type=int, default=0, help="worker processes (0 = all CPUs)")
    ap.add_argument(
        "--replot",
        action="store_true",
        help="regenerate plots from existing CSVs in --outdir (skip DB/experiment)",
    )
    args = ap.parse_args()

    if args.replot:
        outdir = args.outdir
        curve = pd.read_csv(outdir / "sampling_curve.csv")
        tail = pd.read_csv(outdir / "tail_sparsity.csv")
        alpha = pd.read_csv(outdir / "alpha_ablation.csv")
        try_plot(curve, tail, alpha, outdir)
        _log(f"Replot complete: {outdir}")
        return 0

    jobs = args.jobs if args.jobs > 0 else max(1, (os.cpu_count() or 4) - 1)

    groups = _load_groups(args.db, args.min_coverage)
    tail, curve, table, alpha = _run_parallel(groups, args.resamples, jobs)
    snippets = build_tex_snippets(tail, table, alpha, args.resamples, len(groups))
    write_outputs(args.outdir, tail, curve, table, alpha, snippets)

    _log("\n=== Tail sparsity ===")
    for eps in (1, 5, 10):
        _log(f"  within {eps:>2}%: mean {snippets[f'tail_mean_{eps}']:.2f}%  "
             f"median {snippets[f'tail_median_{eps}']:.2f}%  min {snippets[f'tail_min_{eps}']:.2f}%")

    _log("\n=== Budget means (regret %) ===")
    _log(table.sort_values(["budget", "policy"]).to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    _log(f"\n=== Alpha ablation @ B={ALPHA_BUDGET} ===")
    _log(alpha.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    _log(f"\nWrote results to {args.outdir}")
    _log("  tex_snippets.json")
    _log("  sampling_regret_vs_budget.pdf / sampling_within5_vs_budget.pdf")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
