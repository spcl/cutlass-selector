#!/usr/bin/env python3
"""Summarize dtype-transfer study eval results to CSV + markdown."""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from repo_paths import BENCHMARKS, SRC

sys.path.insert(0, str(SRC))
sys.path.insert(0, str(BENCHMARKS))
from transfer_dtype.study_common import (  # noqa: E402
    CURVE_LABELS,
    CURVE_ORDER,
    DTYPE_TAGS,
    EVAL_ROOT,
    RESULTS_ROOT,
    RunSpec,
    curve_key,
    eval_db_home_path,
    iter_run_specs,
    subset_path,
)

PROBLEM_KEYS = ["M", "N", "K", "layout"]


def geomean(x: np.ndarray) -> float:
    x = x[np.isfinite(x) & (x > 0)]
    return float(np.exp(np.mean(np.log(x)))) if len(x) else float("nan")


_EVAL_RUN_COLS = [
    "method", "M", "N", "K", "layout", "rank", "variant", "status", "config_name",
    "mean_tflops", "mean_ms", "error_text", "cutlass_reason",
]
_EVAL_RUN_KEYS = ["method", "M", "N", "K", "layout", "rank", "variant"]


def load_eval_db(db: Path) -> pd.DataFrame:
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        has_table = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='eval_runs'"
        ).fetchone()
        if not has_table:
            return pd.DataFrame(columns=_EVAL_RUN_COLS)
        df = pd.read_sql_query(
            f"""SELECT {", ".join(_EVAL_RUN_COLS)} FROM eval_runs""",
            conn,
        )
    except sqlite3.OperationalError:
        df = pd.DataFrame(columns=_EVAL_RUN_COLS)
    finally:
        conn.close()
    df["mean_tflops"] = pd.to_numeric(df["mean_tflops"], errors="coerce")
    df["tflops"] = np.where(df["status"] == "success", df["mean_tflops"], 0.0)
    return df


def load_dtype_eval(dtype: str) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    home = eval_db_home_path(dtype)
    if home.is_file():
        frames.append(load_eval_db(home))
    shard_dir = EVAL_ROOT / "shards" / dtype
    if shard_dir.is_dir():
        for shard in sorted(shard_dir.glob("*.db")):
            frames.append(load_eval_db(shard))
    if not frames:
        return pd.DataFrame(columns=_EVAL_RUN_COLS + ["tflops"])
    merged = pd.concat(frames, ignore_index=True)
    return merged.drop_duplicates(subset=_EVAL_RUN_KEYS, keep="last")


def nvmmh_per_problem(df: pd.DataFrame) -> pd.DataFrame:
    sub = df[df["method"] == "nvmmh"]
    return (
        sub.groupby(PROBLEM_KEYS, as_index=False)["tflops"]
        .max()
        .rename(columns={"tflops": "nvmmh_tflops"})
    )


def model_per_problem(df: pd.DataFrame, method: str) -> pd.DataFrame:
    sub = df[(df["method"] == method) & (df["rank"] == 1)].copy()
    return sub[PROBLEM_KEYS + ["tflops", "status", "config_name", "mean_tflops", "error_text"]]


def summarize_run(spec: RunSpec, raw: pd.DataFrame, nv: pd.DataFrame) -> dict:
    mp = model_per_problem(raw, spec.run_id)
    if mp.empty:
        return {"run_id": spec.run_id, "error": "no eval rows"}

    merged = mp.merge(nv, on=PROBLEM_KEYS, how="left")
    ok = merged["status"] == "success"
    both = ok & (merged["nvmmh_tflops"] > 0)
    ratio = merged.loc[both, "tflops"] / merged.loc[both, "nvmmh_tflops"]

    shapes_path = subset_path(spec.dtype, spec.seed, spec.fraction)
    n_shapes = len(json.loads(shapes_path.read_text())["shapes"]) if shapes_path.is_file() else None
    meta_path = spec.out_dir / "metrics.json"
    n_train_rows = None
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text())
        n_train_rows = meta.get("study", {}).get("n_rows") or meta.get("n_rows")

    return {
        "run_id": spec.run_id,
        "dtype": spec.dtype,
        "family": spec.family,
        "init": spec.init,
        "features": spec.features,
        "feature_set": "hardware-aware" if spec.features == "full" else "structural",
        "fraction": spec.fraction,
        "seed": spec.seed,
        "n_target_shapes": n_shapes,
        "n_train_kernels": n_train_rows,
        "n_eval_problems": len(mp),
        "coverage": float(ok.mean()),
        "geomean_tflops": geomean(merged.loc[ok, "tflops"].to_numpy()),
        "geomean_speedup_vs_nvmmh": geomean(ratio.to_numpy()) if both.any() else float("nan"),
        "win_rate_vs_nvmmh": float((merged["tflops"] > merged["nvmmh_tflops"]).mean()),
        "curve_key": curve_key(spec),
    }


def aggregate_curves(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dtype in DTYPE_TAGS:
        sub = summary[(summary["dtype"] == dtype) & summary["curve_key"].notna()].copy()
        if sub.empty:
            continue
        for curve in CURVE_ORDER:
            csub = sub[sub["curve_key"] == curve]
            if csub.empty:
                continue
            for fraction, grp in csub.groupby("fraction", sort=True):
                rows.append({
                    "dtype": dtype,
                    "curve_key": curve,
                    "curve_label": CURVE_LABELS[curve],
                    "fraction": fraction,
                    "n_target_shapes_mean": grp["n_target_shapes"].mean(),
                    "geomean_speedup_vs_nvmmh": geomean(grp["geomean_speedup_vs_nvmmh"].to_numpy()),
                    "win_rate_vs_nvmmh": grp["win_rate_vs_nvmmh"].mean(),
                    "coverage": grp["coverage"].mean(),
                    "n_seeds": len(grp),
                })
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dtype", choices=["fp32", "fp8_e4m3", "all"], default="all")
    args = ap.parse_args()

    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    dtypes = list(DTYPE_TAGS) if args.dtype == "all" else [args.dtype]

    all_raw: list[pd.DataFrame] = []
    run_rows: list[dict] = []

    for dtype in dtypes:
        raw = load_dtype_eval(dtype)
        if raw.empty:
            print(f"WARNING: no eval rows for {dtype}")
            continue
        raw["dtype"] = dtype
        all_raw.append(raw)
        nv = nvmmh_per_problem(raw)

        specs = [s for s in iter_run_specs() if s.dtype == dtype]
        for spec in specs:
            if spec.run_id not in set(raw["method"]):
                continue
            run_rows.append(summarize_run(spec, raw, nv))

    if all_raw:
        per_problem = pd.concat(all_raw, ignore_index=True)
        per_problem.to_csv(RESULTS_ROOT / "per_problem_raw.csv", index=False)

    summary = pd.DataFrame(run_rows)
    summary.to_csv(RESULTS_ROOT / "run_summary.csv", index=False)

    curves = aggregate_curves(summary)
    curves.to_csv(RESULTS_ROOT / "curve_points.csv", index=False)

    lines = [
        "# Dtype transfer study summary",
        "",
        f"Results directory: `{RESULTS_ROOT}`",
        "",
        "## Per-run (trained models with eval rows)",
        "",
    ]
    if summary.empty:
        lines.append("_No eval results yet._")
    else:
        cols = [
            "run_id", "dtype", "family", "init", "feature_set", "fraction", "seed",
            "n_target_shapes", "n_train_kernels", "coverage",
            "geomean_tflops", "geomean_speedup_vs_nvmmh", "win_rate_vs_nvmmh",
        ]
        lines.append("| " + " | ".join(cols) + " |")
        lines.append("|" + "|".join(["---"] * len(cols)) + "|")
        def _cell(col: str, val) -> str:
            if pd.isna(val):
                return "—"
            if col in ("coverage", "win_rate_vs_nvmmh"):
                return f"{100.0 * float(val):.1f}%"
            if col.startswith("geomean"):
                return f"{float(val):.3f}"
            return str(val)

        for _, row in summary.sort_values(["dtype", "fraction", "run_id"]).iterrows():
            lines.append("| " + " | ".join(_cell(c, row[c]) for c in cols) + " |")

    md_path = RESULTS_ROOT / "SUMMARY.md"
    md_path.write_text("\n".join(lines) + "\n")
    print(f"wrote {md_path}")
    print(f"wrote {RESULTS_ROOT / 'run_summary.csv'}")
    print(f"wrote {RESULTS_ROOT / 'curve_points.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
