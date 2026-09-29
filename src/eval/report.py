#!/usr/bin/env python3
"""
report.py — Reduce eval.db to one row per problem, plus the summary numbers for the paper.

Every series is "best measured GFLOP/s among the first b candidates the method proposed",
differing only in what orders the candidates:

    ours@b     the model's rank
    cublas@b   cuBLASLt's own heuristic rank (algo index; algo 0 = cublas_top1)
    nvmmh@b    see below
    nvmmh@8    best of all 8 schedule variants of nvMMH's rank-1 recommendation — the
               variants share its tile / cluster / raster / swizzle / splits and differ only
               in the schedules nvMMH does not emit. NOT a search over the config space.

A candidate that failed to compile or that CUTLASS rejected counts as 0 GFLOP/s: it consumed
budget and produced no usable kernel. So every series is a plain max, monotone in b, and the
fraction of problems whose top-1 was usable is reported separately as coverage.

nvmmh@b for b < 8 needs care. The model ranks its candidates and cuBLASLt ranks its algos, so
"the first b" is a property of those methods. nvMMH expresses no schedule preference, so the
variant order is ours, not its own — taking "the first b" would report an artefact of how
translate.py lists them. Instead we report the expectation over a uniformly random variant
order, computed exactly (no sampling): with values sorted ascending, the i-th is the maximum
of C(i-1, b-1) of the C(n, b) possible b-subsets.

    python src/eval/report.py --db artifacts/eval/out/eval.db --out artifacts/eval/out/report.csv
"""

import argparse
import sqlite3
import sys
from math import comb
from pathlib import Path

from repo_paths import EVAL_OUT

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from shapes import stratum  # noqa: E402
from throughput import REPORTED_PEAK_GFLOPS  # noqa: E402

MAX_BUDGET = 8
PROBLEM_KEYS = ["M", "N", "K", "layout"]


def expected_best_of_b(values: list[float], b: int) -> float:
    """E[max of a uniformly random b-subset] — exact, over all C(n, b) subsets."""
    n = len(values)
    if n == 0:
        return float("nan")
    if b >= n:
        return max(values)
    ordered = sorted(values)
    total = comb(n, b)
    return sum(v * comb(i, b - 1) for i, v in enumerate(ordered)) / total


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=? LIMIT 1", (name,)
    ).fetchone()
    return row is not None


def load_eval(conn) -> pd.DataFrame:
    """Load eval_runs; a failed candidate gets 0 GFLOP/s, since it spent budget without a usable kernel."""
    df = pd.read_sql_query(
        """SELECT method, M, N, K, layout, rank, variant, status, mean_tflops
           FROM eval_runs""", conn)
    # Failed candidate = budget spent, no usable kernel.
    df["gflops"] = np.where(df["status"] == "success", df["mean_tflops"], 0.0).astype(float)
    return df


def load_cublas(conn) -> pd.DataFrame:
    """Load cublas_runs (empty frame if the table does not exist), failed algorithms at 0 GFLOP/s."""
    cols = ["M", "N", "K", "layout", "algo", "status", "mean_tflops", "gflops"]
    if not _table_exists(conn, "cublas_runs"):
        return pd.DataFrame(columns=cols)
    df = pd.read_sql_query(
        """SELECT M, N, K, layout, algo, status, mean_tflops FROM cublas_runs""", conn)
    df["gflops"] = np.where(df["status"] == "success", df["mean_tflops"], 0.0).astype(float)
    return df


def ranked_series(df: pd.DataFrame, order_cols: list[str], prefix: str) -> pd.DataFrame:
    """best-of-first-b for a method whose candidate order is meaningful."""
    out = {}
    for keys, grp in df.groupby(PROBLEM_KEYS, sort=False):
        vals = grp.sort_values(order_cols)["gflops"].tolist()
        row = {}
        for b in range(1, MAX_BUDGET + 1):
            row[f"{prefix}@{b}"] = max(vals[:b]) if vals[:b] else np.nan
        row[f"{prefix}_n"] = len(vals)
        row[f"{prefix}_top1_ok"] = bool(vals and vals[0] > 0)
        out[keys] = row
    return pd.DataFrame.from_dict(out, orient="index").rename_axis(PROBLEM_KEYS).reset_index()


def unordered_series(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """best-of-b for a method that expresses no preference among its candidates."""
    out = {}
    for keys, grp in df.groupby(PROBLEM_KEYS, sort=False):
        vals = grp["gflops"].tolist()
        row = {f"{prefix}@{b}": expected_best_of_b(vals, b) for b in range(1, MAX_BUDGET + 1)}
        row[f"{prefix}_n"] = len(vals)
        row[f"{prefix}_any_ok"] = bool(vals and max(vals) > 0)
        out[keys] = row
    return pd.DataFrame.from_dict(out, orient="index").rename_axis(PROBLEM_KEYS).reset_index()


def geomean(x: np.ndarray) -> float:
    """Geometric mean; NaN for an empty array."""
    return float(np.exp(np.mean(np.log(x)))) if len(x) else float("nan")


def bootstrap_ci(ratios: np.ndarray, n_boot: int = 10000, seed: int = 0) -> tuple[float, float]:
    """Percentile CI on the geometric mean, resampling problems (paired: one ratio each)."""
    if len(ratios) < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    logs = np.log(ratios)
    draws = rng.choice(logs, size=(n_boot, len(logs)), replace=True).mean(axis=1)
    return tuple(float(np.exp(v)) for v in np.percentile(draws, [2.5, 97.5]))


def compare(df: pd.DataFrame, num: str, den: str, label: str) -> str:
    """One summary line for num/den: geometric-mean ratio, bootstrap 95% CI, win rate and n.

    Problems where either series is 0 are excluded and counted as unusable.
    """
    sub = df[(df[num] > 0) & (df[den] > 0)]
    if sub.empty:
        return f"  {label:34s} no comparable problems"
    ratios = (sub[num] / sub[den]).to_numpy()
    lo, hi = bootstrap_ci(ratios)
    ci = f"[{lo:.3f}, {hi:.3f}]" if np.isfinite(lo) else "[n too small]"
    win = 100.0 * float((ratios >= 1.0).mean())
    dropped = len(df) - len(sub)
    return (f"  {label:34s} {geomean(ratios):5.3f}x  {ci:>16s}   "
            f"win {win:5.1f}%   n={len(sub)}" + (f" (-{dropped} unusable)" if dropped else ""))


def main() -> int:
    ap = argparse.ArgumentParser(description="Per-problem evaluation report")
    ap.add_argument("--db", type=Path, default=EVAL_OUT / "eval.db")
    ap.add_argument("--out", type=Path, default=EVAL_OUT / "report.csv")
    ap.add_argument("--peak", type=float, default=REPORTED_PEAK_GFLOPS,
                    help="BF16 tensor-core peak (GFLOP/s) for %% of peak columns")
    args = ap.parse_args()

    conn = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    ev = load_eval(conn)
    cb = load_cublas(conn)
    conn.close()

    parts = []
    ranked_methods = sorted(m for m in ev["method"].unique() if m != "nvmmh")
    for method in ranked_methods:
        sub = ev[ev["method"] == method]
        if not sub.empty:
            parts.append(ranked_series(sub, ["rank"], method))
    nvmmh = ev[ev["method"] == "nvmmh"]
    if not nvmmh.empty:
        parts.append(unordered_series(nvmmh, "nvmmh"))
    if not cb.empty:
        parts.append(ranked_series(cb, ["algo"], "cublas"))

    if not parts:
        print("ERROR: no measured rows in the database", file=sys.stderr)
        return 1

    df = parts[0]
    for p in parts[1:]:
        df = df.merge(p, on=PROBLEM_KEYS, how="outer")

    df["cbrt_mnk"] = (df["M"] * df["N"] * df["K"]) ** (1 / 3)
    df["stratum"] = [stratum(m, n, k) for m, n, k in zip(df["M"], df["N"], df["K"])]
    for col in [c for c in df.columns if "@" in c]:
        df[col.replace("@", "_pct_peak@")] = df[col] / args.peak * 100
    df = df.sort_values(["cbrt_mnk", "layout"]).reset_index(drop=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"wrote {len(df)} problems -> {args.out}\n")

    print("coverage (top-1 produced a usable kernel):")
    for method in ranked_methods:
        col = f"{method}_top1_ok"
        if col in df:
            print(f"  {method:34s} {100.0 * df[col].mean():5.1f}%   n={df[col].notna().sum()}")
    if "nvmmh_any_ok" in df:
        print(f"  {'nvmmh (any variant)':34s} {100.0 * df['nvmmh_any_ok'].mean():5.1f}%   "
              f"n={df['nvmmh_any_ok'].notna().sum()}")
    if "cublas_top1_ok" in df:
        print(f"  {'cublas':34s} {100.0 * df['cublas_top1_ok'].mean():5.1f}%   "
              f"n={df['cublas_top1_ok'].notna().sum()}")

    print("\ngeometric mean of per-problem ratios vs nvMMH@8 (bootstrap 95% CI):")
    if "nvmmh@8" in df:
        for method in ranked_methods:
            num = f"{method}@1"
            if num in df:
                print(compare(df, num, "nvmmh@8", f"{method}@1 / nvMMH@8"))

    if "cublas@8" in df:
        print("\nvs cuBLAS (if measured):")
        for method in ranked_methods:
            num = f"{method}@1"
            if num in df:
                print(compare(df, num, "cublas@8", f"{method}@1 / cuBLAS@8"))

    print("\nby stratum (each method@1 / nvMMH@8):")
    if "nvmmh@8" in df:
        for name, grp in df.groupby("stratum"):
            for method in ranked_methods:
                num = f"{method}@1"
                if num in grp:
                    print(compare(grp, num, "nvmmh@8", f"{name}: {method}@1"))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
