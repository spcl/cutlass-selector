#!/usr/bin/env python3
"""Summarize Eval2 zero-shot fusion transfer results."""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from repo_paths import BENCHMARKS, SRC

sys.path.insert(0, str(SRC))
sys.path.insert(0, str(BENCHMARKS))
from transfer_fusion.study_common import (  # noqa: E402
    CURVE_LABELS,
    CURVE_ORDER,
    DTYPE_TAGS,
    RunSpec,
    curve_key,
    eval_db_home_path,
    eval_root,
    iter_run_specs,
    results_root,
    subset_path,
)

SUITE = "eval2"
EVAL2_ROOT = eval_root(SUITE)
EVAL2_RESULTS = results_root(SUITE)

PROBLEM_KEYS = ["M", "N", "K", "layout", "variant"]
_FUSION_RE = re.compile(r"_fusion_([a-z0-9_]+)$")


def geomean(x: np.ndarray) -> float:
    x = x[np.isfinite(x) & (x > 0)]
    return float(np.exp(np.mean(np.log(x)))) if len(x) else float("nan")


def fusion_from_config(name: str) -> str:
    m = _FUSION_RE.search(str(name))
    return m.group(1) if m else "unknown"


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
    df["fusion_kind"] = df["config_name"].map(fusion_from_config)
    return df


def load_dtype_eval(dtype: str) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    home = eval_db_home_path(dtype, SUITE)
    if home.is_file():
        frames.append(load_eval_db(home))
    shard_dir = EVAL2_ROOT / "shards" / dtype
    if shard_dir.is_dir():
        for shard in sorted(shard_dir.glob("*.db")):
            frames.append(load_eval_db(shard))
    if not frames:
        return pd.DataFrame(columns=_EVAL_RUN_COLS + ["tflops", "fusion_kind"])
    merged = pd.concat(frames, ignore_index=True)
    return merged.drop_duplicates(subset=_EVAL_RUN_KEYS, keep="last")


def nvmmh_per_problem(df: pd.DataFrame) -> pd.DataFrame:
    sub = df[df["method"] == "nvmmh"]
    keys = ["M", "N", "K", "layout"]
    return (
        sub.groupby(keys, as_index=False)["tflops"]
        .max()
        .rename(columns={"tflops": "nvmmh_tflops"})
    )


def model_per_problem(df: pd.DataFrame, method: str) -> pd.DataFrame:
    sub = df[(df["method"] == method) & (df["rank"] == 1)].copy()
    return sub[PROBLEM_KEYS + ["fusion_kind", "tflops", "status", "config_name", "mean_tflops", "error_text"]]


def summarize_run(spec: RunSpec, raw: pd.DataFrame, nv: pd.DataFrame) -> list[dict]:
    mp = model_per_problem(raw, spec.run_id)
    if mp.empty:
        return [{"run_id": spec.run_id, "error": "no eval rows"}]

    merged = mp.merge(nv, on=["M", "N", "K", "layout"], how="left")
    shapes_path = subset_path(spec.dtype, spec.seed, spec.fraction)
    n_shapes = len(json.loads(shapes_path.read_text())["shapes"]) if shapes_path.is_file() else None
    meta_path = spec.out_dir / "metrics.json"
    n_train_rows = None
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text())
        n_train_rows = meta.get("study", {}).get("n_rows") or meta.get("n_rows")

    rows: list[dict] = []
    groups = [("all", merged)] + list(merged.groupby("fusion_kind", sort=False))
    for label, grp in groups:
        ok = grp["status"] == "success"
        both = ok & (grp["nvmmh_tflops"] > 0)
        ratio = grp.loc[both, "tflops"] / grp.loc[both, "nvmmh_tflops"]
        rows.append({
            "run_id": spec.run_id,
            "dtype": spec.dtype,
            "family": spec.family,
            "init": spec.init,
            "features": spec.features,
            "feature_set": "hardware-aware" if spec.features == "full" else "structural",
            "fraction": spec.fraction,
            "seed": spec.seed,
            "fusion_kind": label,
            "n_target_shapes": n_shapes,
            "n_train_kernels": n_train_rows,
            "n_eval_problems": len(grp),
            "coverage": float(ok.mean()),
            "geomean_tflops": geomean(grp.loc[ok, "tflops"].to_numpy()),
            "geomean_speedup_vs_nvmmh": geomean(ratio.to_numpy()) if both.any() else float("nan"),
            "win_rate_vs_nvmmh": float((grp["tflops"] > grp["nvmmh_tflops"]).mean()),
            "curve_key": curve_key(spec),
        })
    return rows


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
            for (fraction, fusion_kind), grp in csub.groupby(["fraction", "fusion_kind"], sort=True):
                rows.append({
                    "dtype": dtype,
                    "curve_key": curve,
                    "curve_label": CURVE_LABELS[curve],
                    "fraction": fraction,
                    "fusion_kind": fusion_kind,
                    "n_target_shapes_mean": grp["n_target_shapes"].mean(),
                    "geomean_speedup_vs_nvmmh": geomean(grp["geomean_speedup_vs_nvmmh"].to_numpy()),
                    "win_rate_vs_nvmmh": grp["win_rate_vs_nvmmh"].mean(),
                    "coverage": grp["coverage"].mean(),
                    "n_seeds": len(grp),
                })
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dtype", choices=list(DTYPE_TAGS) + ["all"], default="all")
    args = ap.parse_args()

    EVAL2_RESULTS.mkdir(parents=True, exist_ok=True)
    dtypes = list(DTYPE_TAGS) if args.dtype == "all" else [args.dtype]

    all_raw: list[pd.DataFrame] = []
    run_rows: list[dict] = []

    for dtype in dtypes:
        raw = load_dtype_eval(dtype)
        if raw.empty:
            print(f"WARNING: no eval2 rows for {dtype}")
            continue
        raw["dtype"] = dtype
        all_raw.append(raw)
        nv = nvmmh_per_problem(raw)

        specs = [s for s in iter_run_specs() if s.dtype == dtype]
        for spec in specs:
            if spec.run_id not in set(raw["method"]):
                continue
            run_rows.extend(summarize_run(spec, raw, nv))

    if all_raw:
        per_problem = pd.concat(all_raw, ignore_index=True)
        per_problem.to_csv(EVAL2_RESULTS / "per_problem_raw.csv", index=False)

    summary = pd.DataFrame(run_rows)
    summary.to_csv(EVAL2_RESULTS / "run_summary.csv", index=False)

    curves = aggregate_curves(summary)
    curves.to_csv(EVAL2_RESULTS / "curve_points.csv", index=False)

    lines = [
        "# Fusion transfer Eval2 summary (zero-shot fusion kinds)",
        "",
        f"Results directory: `{EVAL2_RESULTS}`",
        "",
        "Zero-shot kinds: `silu`, `bias_silu`, `tanh`, `bias_tanh` (not in training sweep).",
        "",
        "## Per-run × fusion kind",
        "",
    ]
    if summary.empty:
        lines.append("_No eval2 results yet._")
    else:
        cols = [
            "run_id", "dtype", "family", "feature_set", "fraction",
            "fusion_kind", "n_eval_problems", "coverage",
            "geomean_speedup_vs_nvmmh", "win_rate_vs_nvmmh",
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

        for _, row in summary.sort_values(["dtype", "fraction", "fusion_kind", "run_id"]).iterrows():
            lines.append("| " + " | ".join(_cell(c, row[c]) for c in cols) + " |")

    md_path = EVAL2_RESULTS / "SUMMARY.md"
    md_path.write_text("\n".join(lines) + "\n")
    print(f"wrote {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
