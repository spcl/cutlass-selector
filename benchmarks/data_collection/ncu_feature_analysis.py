#!/usr/bin/env python3
"""
Reproducible NCU profiling analysis behind the feature design.

Read-only against the autotuner SQLite DB. Joins successful ncu_runs with runs
and configs; throughput target is runs.mean_tflops (benchmark TFLOP/s).

Outputs (under --out-dir):
  profiling_summary.md
  counter_correlations.csv
  config_correlations.csv
  regime_correlations.csv
  candidate_effect_correlations.csv
  highlighted_counters.md
  plots/  (optional diagnostic scatter panels)

Usage:
  python benchmarks/data_collection/ncu_feature_analysis.py \\
    --db ~/autotuner.db.bak-pre-fp8 \\
    --out-dir analysis/ncu_profiling
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from repo_paths import ANALYSIS, SRC

# Repo imports
sys.path.insert(0, str(SRC / "autotuner"))
sys.path.insert(0, str(SRC / "model"))

from features import featurize  # noqa: E402
from ncu_worker import NCU_METRICS  # noqa: E402
from scheduler import BENCHMARK_SHAPES  # noqa: E402

MIN_GROUP_SAMPLES = 3
GROUP_COLS = ["M", "N", "K", "layout_a", "layout_b"]
TARGET = "mean_tflops"

# NCU columns stored in ncu_runs (exclude metadata)
NCU_SKIP = {"name", "M", "N", "K", "status", "error_text", "profiled_at"}

HIGHLIGHT_MAP = {
    "tensor-core / WGMMA activity": [
        "tensor_active_pct",
        "wgmma_inst_executed",
        "stall_gmma_pct",
    ],
    "DRAM throughput / bandwidth utilization": [
        "dram_throughput_pct",
        "dram_read_bytes",
        "dram_write_bytes",
    ],
    "achieved occupancy": [
        "achieved_occupancy_pct",
        "theoretical_occupancy_pct",
        "warps_active",
        "waves_per_sm",
    ],
    "barrier stalls": ["stall_barrier_pct"],
    "long-scoreboard / memory-dependency stalls": [
        "stall_long_scoreboard_pct",
        "stall_short_scoreboard_pct",
        "stall_mio_pct",
    ],
    "L2 hit rate / traffic": [
        "l2_hit_rate",
        "l2_read_hit_rate",
        "l2_write_hit_rate",
        "l2_read_sectors",
        "l2_write_sectors",
        "l2_throughput_pct",
    ],
}

CONFIG_NUMERIC = [
    "tile_m",
    "tile_n",
    "tile_k",
    "stages",
    "cluster_m",
    "cluster_n",
    "cluster_k",
    "alignment_a",
    "alignment_b",
    "alignment_c",
]

CONFIG_PROXY = [
    "problem_arith_intensity",
    "mainloop_compute_intensity",
    "restream_factor",
    "resident_panel_vs_L2",
    "working_set_vs_L2",
    "smem_total",
    "smem_frac",
    "blocks_per_sm",
    "true_blocks_per_sm",
    "reg_pressure_proxy",
    "bytes_per_stage",
    "pipeline_fill_frac",
    "stage_amortization",
    "cluster_size",
    "k_iters",
]

CANDIDATE_EFFECT_PAIRS = [
    ("tile_k", "achieved_occupancy_pct"),
    ("stages", "achieved_occupancy_pct"),
    ("smem_total", "achieved_occupancy_pct"),
    ("true_blocks_per_sm", "achieved_occupancy_pct"),
    ("tile_m", "tensor_active_pct"),
    ("tile_n", "tensor_active_pct"),
    ("tile_k", "tensor_active_pct"),
    ("stages", "tensor_active_pct"),
    ("problem_arith_intensity", "dram_throughput_pct"),
    ("mainloop_compute_intensity", "dram_throughput_pct"),
    ("stages", "stall_barrier_pct"),
    ("stages", "stall_long_scoreboard_pct"),
    ("bytes_per_stage", "stall_long_scoreboard_pct"),
    ("pipeline_fill_frac", "stall_barrier_pct"),
]

CATEGORICAL_CONFIG = ["kernel_schedule", "epilogue_schedule", "scheduler", "sched_class"]


def _spearman(x: pd.Series, y: pd.Series) -> float:
    mask = x.notna() & y.notna()
    xv, yv = x[mask], y[mask]
    if len(xv) < MIN_GROUP_SAMPLES:
        return np.nan
    if xv.nunique() < 2 or yv.nunique() < 2:
        return np.nan
    return float(xv.corr(yv, method="spearman"))


def load_profiles(db_path: Path, dtype: str | None) -> pd.DataFrame:
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    q = """
        SELECT
            n.*,
            r.mean_tflops, r.std_tflops, r.mean_ms, r.tag AS run_tag,
            c.cutlass_type_a, c.cutlass_type_b, c.cutlass_type_c,
            c.layout_a, c.layout_b,
            c.tile_m, c.tile_n, c.tile_k,
            c.cluster_m, c.cluster_n, c.cluster_k,
            c.stages, c.kernel_schedule, c.epilogue_schedule, c.scheduler,
            c.alignment_a, c.alignment_b, c.alignment_c
        FROM ncu_runs n
        JOIN runs r
          ON n.name = r.name AND n.M = r.M AND n.N = r.N AND n.K = r.K
        JOIN configs c ON n.name = c.name
        WHERE n.status = 'success' AND r.status = 'success'
    """
    df = pd.read_sql(q, conn)
    conn.close()
    if dtype:
        df = df[df["cutlass_type_a"] == dtype].copy()
    return df.reset_index(drop=True)


def ncu_numeric_columns(df: pd.DataFrame) -> list[str]:
    cols = []
    for c in df.columns:
        if c in NCU_SKIP or c in GROUP_COLS or c.startswith("cutlass"):
            continue
        if c in CONFIG_NUMERIC or c in {"mean_tflops", "std_tflops", "mean_ms"}:
            continue
        if c in {"layout_a", "layout_b", "kernel_schedule", "epilogue_schedule", "scheduler", "run_tag"}:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    return sorted(cols)


def correlation_table(
    df: pd.DataFrame,
    metrics: list[str],
    target: str = TARGET,
    group_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Global + within-group Spearman with target."""
    group_cols = group_cols or GROUP_COLS
    rows = []
    for metric in metrics:
        if metric not in df.columns:
            continue
        global_rho = _spearman(df[metric], df[target])
        within = []
        dropped = 0
        for _, g in df.groupby(group_cols, sort=False):
            rho = _spearman(g[metric], g[target])
            if np.isnan(rho):
                dropped += 1
            else:
                within.append(rho)
        n_groups = len(within)
        pos = sum(1 for r in within if r > 0)
        neg = sum(1 for r in within if r < 0)
        rows.append(
            {
                "metric": metric,
                "global_rho": global_rho,
                "mean_within_rho": float(np.mean(within)) if within else np.nan,
                "median_within_rho": float(np.median(within)) if within else np.nan,
                "positive_fraction": pos / n_groups if n_groups else np.nan,
                "negative_fraction": neg / n_groups if n_groups else np.nan,
                "n_groups": n_groups,
                "n_groups_dropped": dropped,
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        out["abs_global_minus_within"] = (out["global_rho"] - out["mean_within_rho"]).abs()
        out = out.sort_values("abs_global_minus_within", ascending=False)
    return out


def add_regime_labels(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    for col in ["M", "N", "K", "tile_m", "tile_n", "tile_k", "stages", "cluster_m", "cluster_n", "cluster_k"]:
        if col in d.columns:
            d[col] = d[col].astype("int64")
    f = featurize(d)
    d["problem_arith_intensity"] = f["problem_arith_intensity"]
    d["sched_class"] = f["sched_class"]

    flops = 2.0 * d["M"] * d["N"] * d["K"]
    d["size_metric"] = flops
    q33, q66 = d["size_metric"].quantile([0.33, 0.66])
    d["size_regime"] = np.where(
        d["size_metric"] <= q33,
        "small",
        np.where(d["size_metric"] <= q66, "medium", "large"),
    )

    m, n, k = d["M"].astype(float), d["N"].astype(float), d["K"].astype(float)
    aspect = np.maximum(m, n) / np.minimum(m, n)
    skinny_k = k < 0.25 * np.minimum(m, n)
    square = aspect <= 1.5
    tall = (m > 1.5 * n) & ~skinny_k
    wide = (n > 1.5 * m) & ~skinny_k
    shape = np.full(len(d), "other", dtype=object)
    shape[square] = "square"
    shape[tall] = "tall"
    shape[wide] = "wide"
    shape[skinny_k] = "skinny-K"
    d["shape_regime"] = shape

    ai_med = d["problem_arith_intensity"].median()
    d["intensity_regime"] = np.where(
        d["problem_arith_intensity"] >= ai_med, "compute_heavy", "memory_heavy"
    )
    return d


def selection_summary(df: pd.DataFrame) -> dict:
    """Recover how profiled configs were chosen (rank by benchmark TFLOPS within group)."""
    d = df.copy()
    d["tflops_rank_in_group"] = d.groupby(GROUP_COLS)["mean_tflops"].rank(
        ascending=False, method="first"
    )
    int(d["tflops_rank_in_group"].max())
    shapes = d[["M", "N", "K"]].drop_duplicates()
    benchmark_set = {tuple(s) for s in BENCHMARK_SHAPES}
    profiled_shapes = {tuple(x) for x in shapes.to_numpy()}
    return {
        "n_profiles": len(d),
        "n_unique_shapes": shapes.shape[0],
        "n_shape_layout_groups": d.groupby(GROUP_COLS).ngroups,
        "profiles_per_group_min": int(d.groupby(GROUP_COLS).size().min()),
        "profiles_per_group_median": float(d.groupby(GROUP_COLS).size().median()),
        "profiles_per_group_max": int(d.groupby(GROUP_COLS).size().max()),
        "max_tflops_rank_in_group": int(d["tflops_rank_in_group"].max()),
        "median_tflops_rank_in_group": float(d["tflops_rank_in_group"].median()),
        "all_shapes_in_BENCHMARK_SHAPES": profiled_shapes <= benchmark_set,
        "n_benchmark_shapes_profiled": len(profiled_shapes & benchmark_set),
        "selection_note": (
            "Configs appear to be the top performers by mean_tflops within each "
            f"(M,N,K,layout) group (max rank observed: {int(d['tflops_rank_in_group'].max())}). "
            "NCU scheduler uses BENCHMARK_SHAPES with --ncu-top-k (default 20)."
        ),
        "expected_408_note": (
            "408 = 17 BENCHMARK_SHAPES × 24 top configs is a plausible BF16-only budget; "
            "verify against actual row count rather than assuming."
        ),
    }


def categorical_throughput_table(df: pd.DataFrame, col: str, min_count: int = 5) -> pd.DataFrame:
    rows = []
    for cat, g in df.groupby(col, dropna=False):
        if len(g) < min_count:
            continue
        rows.append(
            {
                "category": str(cat),
                "n": len(g),
                "mean_tflops": g[TARGET].mean(),
                "median_tflops": g[TARGET].median(),
                "std_tflops": g[TARGET].std(),
            }
        )
    return pd.DataFrame(rows).sort_values("median_tflops", ascending=False)


def write_highlighted_counters(out_dir: Path, db_cols: set[str]) -> None:
    lines = ["# Highlighted NCU counter mapping\n"]
    lines.append("DB columns verified against `ncu_runs` schema and `NCU_METRICS` in `ncu_worker.py`.\n")
    for label, candidates in HIGHLIGHT_MAP.items():
        lines.append(f"## {label}\n")
        for col in candidates:
            if col not in db_cols:
                lines.append(f"- `{col}`: **missing from DB**\n")
                continue
            ncu_name = NCU_METRICS.get(col, "(no NCU_METRICS entry)")
            lines.append(f"- `{col}`: `{ncu_name}`\n")
        lines.append("")
    (out_dir / "highlighted_counters.md").write_text("".join(lines))


def maybe_plots(df: pd.DataFrame, counter_df: pd.DataFrame, out_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(exist_ok=True)
    picks = [
        ("tensor_active_pct", "Tensor-core activity"),
        ("achieved_occupancy_pct", "Achieved occupancy"),
        ("stall_barrier_pct", "Barrier stalls"),
        ("dram_throughput_pct", "DRAM throughput"),
    ]
    for col, title in picks:
        if col not in df.columns:
            continue
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(df[col], df[TARGET], s=8, alpha=0.35)
        ax.set_xlabel(col)
        ax.set_ylabel(TARGET)
        ax.set_title(f"Global: {title}")
        fig.tight_layout()
        fig.savefig(plot_dir / f"global_{col}.png", dpi=120)
        plt.close(fig)

    top_confound = counter_df.nlargest(3, "abs_global_minus_within")
    for _, row in top_confound.iterrows():
        col = row["metric"]
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].scatter(df[col], df[TARGET], s=8, alpha=0.35)
        axes[0].set_title(f"Global ρ={row['global_rho']:.3f}")
        axes[0].set_xlabel(col)
        axes[0].set_ylabel(TARGET)
        for _, g in df.groupby(GROUP_COLS):
            if len(g) >= MIN_GROUP_SAMPLES:
                axes[1].scatter(g[col], g[TARGET], s=12, alpha=0.6)
        axes[1].set_title(f"Within-group mean ρ={row['mean_within_rho']:.3f}")
        axes[1].set_xlabel(col)
        fig.suptitle(col)
        fig.tight_layout()
        fig.savefig(plot_dir / f"confound_{col}.png", dpi=120)
        plt.close(fig)


def plain_english_summary(
    counter_df: pd.DataFrame,
    config_df: pd.DataFrame,
    regime_tile_k: pd.DataFrame,
    sel: dict,
) -> str:
    lines = ["## Plain-English summary\n"]

    confound = counter_df.dropna(subset=["global_rho", "mean_within_rho"]).copy()
    confound["collapse"] = confound["global_rho"].abs() - confound["mean_within_rho"].abs()
    confound = confound.sort_values("collapse", ascending=False)

    lines.append("### Global confounds (strong globally, weak within-problem)\n")
    for _, r in confound.head(5).iterrows():
        lines.append(
            f"- **{r['metric']}**: global ρ={r['global_rho']:+.3f}, "
            f"within mean ρ={r['mean_within_rho']:+.3f}\n"
        )

    within_strong = counter_df.dropna(subset=["mean_within_rho"]).copy()
    within_strong["abs_within"] = within_strong["mean_within_rho"].abs()
    within_strong = within_strong.sort_values("abs_within", ascending=False)
    lines.append("\n### Strongest within-problem NCU signals\n")
    for _, r in within_strong.head(6).iterrows():
        lines.append(f"- **{r['metric']}**: mean within ρ={r['mean_within_rho']:+.3f} ({int(r['n_groups'])} groups)\n")

    stable = config_df.dropna(subset=["mean_within_rho"]).copy()
    stable["abs_within"] = stable["mean_within_rho"].abs()
    stable_cfg = stable[stable["metric"].isin(CONFIG_NUMERIC + ["tile_k"])].sort_values(
        "abs_within", ascending=False
    )
    lines.append("\n### Configuration knobs with stable within-group effects\n")
    for _, r in stable_cfg.head(6).iterrows():
        lines.append(f"- **{r['metric']}**: mean within ρ={r['mean_within_rho']:+.3f}\n")

    if not regime_tile_k.empty:
        lines.append("\n### tile_k regime dependence\n")
        for _, r in regime_tile_k.iterrows():
            lines.append(
                f"- **{r['regime']}**: mean within ρ(tile_k, TFLOPS)={r['mean_within_rho']:+.3f} "
                f"({int(r['n_groups'])} groups, {int(r['n_profiles'])} profiles)\n"
            )

    lines.append("\n### Implications for static proxies\n")
    lines.append(
        "- Tensor-core activity and DRAM throughput remain the main *measured* within-problem "
        "differentiators; the model needs analytical proxies for arithmetic intensity / bytes moved "
        "and pipeline supply rate (stages, SMEM, tile shape).\n"
    )
    lines.append(
        "- Occupancy and barrier stalls are poor direct ranking features within a fixed problem; "
        "they mostly track problem size globally.\n"
    )
    lines.append(f"\n_Actual profiled kernel count: {sel['n_profiles']} (expected 408 for BF16-only budget)._ \n")
    return "".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description="NCU profiling analysis for feature design")
    ap.add_argument("--db", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=ANALYSIS / "ncu_profiling")
    ap.add_argument(
        "--dtype",
        default="cutlass::bfloat16_t",
        help="Restrict to one cutlass_type_a (default: BF16). Use 'all' for every dtype.",
    )
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()

    dtype = None if args.dtype == "all" else args.dtype
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_profiles(args.db, dtype)
    if df.empty:
        print("No successful NCU profiles found.", file=sys.stderr)
        return 1

    df = add_regime_labels(df)
    featurized = featurize(df)
    for col in CONFIG_PROXY:
        if col in featurized.columns:
            df[col] = featurized[col].values

    sel = selection_summary(df)
    ncu_cols = ncu_numeric_columns(df)
    db_col_set = set(df.columns)

    counter_df = correlation_table(df, ncu_cols)
    counter_df.to_csv(out_dir / "counter_correlations.csv", index=False)

    config_metrics = [c for c in CONFIG_NUMERIC + CONFIG_PROXY if c in df.columns]
    config_df = correlation_table(df, config_metrics)
    config_df.to_csv(out_dir / "config_correlations.csv", index=False)

    regime_rows = []
    for regime_col in ["size_regime", "shape_regime", "intensity_regime"]:
        for regime, sub in df.groupby(regime_col):
            if len(sub) < 10:
                continue
            sub_corr = correlation_table(sub, ncu_cols + ["tile_k"])
            for metric in ["tensor_active_pct", "dram_throughput_pct", "achieved_occupancy_pct",
                             "stall_barrier_pct", "stall_long_scoreboard_pct", "tile_k"]:
                row = sub_corr[sub_corr["metric"] == metric]
                if row.empty:
                    continue
                r = row.iloc[0]
                regime_rows.append(
                    {
                        "regime_type": regime_col,
                        "regime": regime,
                        "metric": metric,
                        "n_profiles": len(sub),
                        "n_groups": sub.groupby(GROUP_COLS).ngroups,
                        "global_rho": r["global_rho"],
                        "mean_within_rho": r["mean_within_rho"],
                        "median_within_rho": r["median_within_rho"],
                        "n_groups_computed": r["n_groups"],
                    }
                )
    regime_df = pd.DataFrame(regime_rows)
    regime_df.to_csv(out_dir / "regime_correlations.csv", index=False)

    tile_k_regime = regime_df[
        (regime_df["metric"] == "tile_k") & (regime_df["regime_type"] == "intensity_regime")
    ]

    effect_rows = []
    for feat, effect in CANDIDATE_EFFECT_PAIRS:
        if feat not in df.columns or effect not in df.columns:
            continue
        row = correlation_table(df, [feat], target=effect).iloc[0]
        effect_rows.append(
            {
                "feature": feat,
                "effect": effect,
                "global_rho": row["global_rho"],
                "mean_within_rho": row["mean_within_rho"],
                "median_within_rho": row["median_within_rho"],
                "n_groups": row["n_groups"],
            }
        )
    effect_df = pd.DataFrame(effect_rows).sort_values("mean_within_rho", key=abs, ascending=False)
    effect_df.to_csv(out_dir / "candidate_effect_correlations.csv", index=False)

    cat_tables = {}
    for col in CATEGORICAL_CONFIG:
        if col in df.columns:
            cat_tables[col] = categorical_throughput_table(df, col)

    write_highlighted_counters(out_dir, db_col_set)

    md = ["# NCU profiling analysis summary\n"]
    md.append(f"- **Database**: `{args.db}` (read-only)\n")
    md.append(f"- **Dtype filter**: `{args.dtype}`\n")
    md.append(f"- **Profiled kernels**: {sel['n_profiles']}\n")
    md.append(f"- **Unique (M,N,K) shapes**: {sel['n_unique_shapes']}\n")
    md.append(f"- **Shape-layout groups**: {sel['n_shape_layout_groups']}\n")
    md.append(
        f"- **Profiles per group**: min={sel['profiles_per_group_min']}, "
        f"median={sel['profiles_per_group_median']:.1f}, max={sel['profiles_per_group_max']}\n"
    )
    md.append(f"- **BENCHMARK_SHAPES coverage**: {sel['n_benchmark_shapes_profiled']}/17\n")
    md.append(f"- **Selection**: {sel['selection_note']}\n")
    md.append(f"- **408 check**: actual={sel['n_profiles']} for this dtype filter; {sel['expected_408_note']}\n")
    md.append(f"- **Groups dropped (<{MIN_GROUP_SAMPLES} usable pairs)**: see `n_groups_dropped` in CSVs\n")
    md.append("\n## Key counters\n")
    for col in ["tensor_active_pct", "achieved_occupancy_pct", "stall_barrier_pct", "dram_throughput_pct"]:
        row = counter_df[counter_df["metric"] == col]
        if not row.empty:
            r = row.iloc[0]
            md.append(
                f"- `{col}`: global ρ={r['global_rho']:+.3f}, "
                f"within mean ρ={r['mean_within_rho']:+.3f}\n"
            )
    md.append("\n")
    md.append(plain_english_summary(counter_df, config_df, tile_k_regime, sel))
    md.append("\n## Categorical schedule throughput (median TFLOP/s, n≥5)\n")
    for col, tab in cat_tables.items():
        md.append(f"\n### {col}\n")
        if tab.empty:
            md.append("(insufficient support)\n")
        else:
            for _, r in tab.iterrows():
                md.append(
                    f"- {r['category']}: n={int(r['n'])}, median={r['median_tflops']:.1f}, "
                    f"mean={r['mean_tflops']:.1f}\n"
                )

    (out_dir / "profiling_summary.md").write_text("".join(md))

    if not args.no_plots:
        maybe_plots(df, counter_df, out_dir)

    print(f"Wrote analysis to {out_dir}")
    print(f"Profiles: {sel['n_profiles']} | groups: {sel['n_shape_layout_groups']}")
    for col in ["tensor_active_pct", "achieved_occupancy_pct", "stall_barrier_pct", "dram_throughput_pct"]:
        row = counter_df[counter_df["metric"] == col]
        if not row.empty:
            r = row.iloc[0]
            print(f"  {col}: global={r['global_rho']:+.3f} within_mean={r['mean_within_rho']:+.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
