#!/usr/bin/env python3
"""Evaluate nvMatmulHeuristics on held-out BF16 eval groups via DB lookup (no GPU runs).

For each (M,N,K,layout) group in the exhaustive eval plan:
  1. Query nvMMH for top-K tile recommendations (H100_PCIE, CUTLASS3, HSH/BF16).
  2. Match each recommendation to measured configs in the eval DB by
     (tile_m, tile_n, tile_k, cluster_m, cluster_n, stages, layout).
  3. Report regret vs the empirical oracle on the same catalogue.

Modes:
  top1     — rank-1 tile, best measured schedule variant in the catalogue
  bestofk  — best measured across all top-K tile matches (shortlist autotune)

Usage:
    python baseline/nvmmh/nvmmh_eval.py --db ~/autotuner_bf16_eval.db
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

from repo_paths import PAPER, SRC

sys.path.insert(0, str(SRC / "autotuner"))
sys.path.insert(0, str(SRC / "model"))

from config_space import LAYOUTS  # noqa: E402
from harness import regime, regime_breakdown, summarize  # noqa: E402

TAG = "bf16_eval"
DEFAULT_DB = Path.home() / "autotuner_bf16_eval.db"

# nvMMH MatmulLayout enum -> our layout tag
_NVMMH_LAYOUT = {
    "NN_ROW_MAJOR": "NN",
    "NT_ROW_MAJOR": "NT",
    "TN_ROW_MAJOR": "TN",
    "TT_ROW_MAJOR": "TT",
}


def _layout_pair(tag: str) -> tuple[str, str]:
    return LAYOUTS[tag][1], LAYOUTS[tag][2]


def _tile_key(
    tile_m: int,
    tile_n: int,
    tile_k: int,
    cluster_m: int,
    cluster_n: int,
    stages: int,
    layout_tag: str,
) -> tuple:
    la, lb = _layout_pair(layout_tag)
    return (tile_m, tile_n, tile_k, cluster_m, cluster_n, stages, la, lb)


def _load_eval_groups(conn: sqlite3.Connection) -> pd.DataFrame:
    df = pd.read_sql(
        """
        SELECT DISTINCT p.M, p.N, p.K, p.layout, p.regime
        FROM eval_plan p
        WHERE p.tag = ?
        ORDER BY p.M, p.N, p.K, p.layout
        """,
        conn,
        params=(TAG,),
    )
    return df


def _load_oracle_and_runs(conn: sqlite3.Connection) -> tuple[dict, dict, dict]:
    """Return oracle per group, runs (M,N,K,name)->tflops, config tile index."""
    runs_rows = pd.read_sql(
        """
        SELECT r.M, r.N, r.K, r.name, r.mean_tflops,
               c.layout_a, c.layout_b,
               c.tile_m, c.tile_n, c.tile_k,
               c.cluster_m, c.cluster_n, c.stages
        FROM runs r
        JOIN configs c ON r.name = c.name
        JOIN eval_plan p ON p.name = r.name AND p.M = r.M AND p.N = r.N AND p.K = r.K
        WHERE p.tag = ? AND r.status = 'success' AND r.mean_tflops > 0
        """,
        conn,
        params=(TAG,),
    )

    oracle: dict[tuple, float] = {}
    runs: dict[tuple, float] = {}
    tile_to_names: dict[tuple, set[str]] = defaultdict(set)

    for row in runs_rows.itertuples(index=False):
        la, lb = row.layout_a, row.layout_b
        # recover layout tag
        layout_tag = None
        for tag, (_, a, b) in LAYOUTS.items():
            if a == la and b == lb:
                layout_tag = tag
                break
        if layout_tag is None:
            continue
        gkey = (row.M, row.N, row.K, layout_tag)
        oracle[gkey] = max(oracle.get(gkey, 0.0), float(row.mean_tflops))
        runs[(row.M, row.N, row.K, row.name)] = float(row.mean_tflops)
        tkey = (row.tile_m, row.tile_n, row.tile_k, row.cluster_m, row.cluster_n, row.stages, la, lb)
        tile_to_names[tkey].add(row.name)

    return oracle, runs, tile_to_names


def _init_nvmmh(gpu: str):
    from nvMatmulHeuristics import (
        NvMatmulHeuristicsFlags,
        NvMatmulHeuristicsInterface,
        NvMatmulHeuristicsMatmulLayout,
        NvMatmulHeuristicsNvidiaGpu,
        NvMatmulHeuristicsTarget,
    )

    nvmmh = NvMatmulHeuristicsInterface(
        NvMatmulHeuristicsTarget.CUTLASS3,
        precision="HSH",
        flags=NvMatmulHeuristicsFlags.PERF_MODEL_BASED_AUTO_TUNING,
    )
    hw = nvmmh.createHardwareDescriptor()
    gpu_enum = getattr(NvMatmulHeuristicsNvidiaGpu, gpu)
    nvmmh.setHardwarePredefinedGpu(hw, gpu_enum)

    layout_iface: dict[str, object] = {}
    for enum_name, tag in _NVMMH_LAYOUT.items():
        layout_enum = getattr(NvMatmulHeuristicsMatmulLayout, enum_name)
        if not nvmmh.loadInternalDiscoverySet(layout_enum, hw):
            raise RuntimeError(f"loadInternalDiscoverySet failed for {enum_name}")
        layout_iface[tag] = layout_enum

    return nvmmh, hw, layout_iface


def _best_tflops_for_kernel(
    kernel,
    layout_tag: str,
    M: int,
    N: int,
    K: int,
    runs: dict,
    tile_to_names: dict,
) -> tuple[float | None, str | None, bool]:
    tkey = _tile_key(
        int(kernel.cta_tile_m),
        int(kernel.cta_tile_n),
        int(kernel.cta_tile_k),
        int(kernel.cluster_m),
        int(kernel.cluster_n),
        int(kernel.stages),
        layout_tag,
    )
    names = tile_to_names.get(tkey)
    if not names:
        return None, None, False
    best_name = None
    best_t = -1.0
    for name in names:
        t = runs.get((M, N, K, name))
        if t is not None and t > best_t:
            best_t = t
            best_name = name
    if best_name is None:
        return None, None, True  # tile matched configs but none measured at this shape
    return best_t, best_name, True


def eval_nvmmh(
    groups: pd.DataFrame,
    oracle: dict,
    runs: dict,
    tile_to_names: dict,
    nvmmh,
    hw,
    layout_iface: dict,
    top_k: int = 8,
    mode: str = "top1",
) -> pd.DataFrame:
    rows = []
    for group_id, g in groups.iterrows():
        M, N, K, layout_tag = int(g.M), int(g.N), int(g.K), g.layout
        gkey = (M, N, K, layout_tag)
        orc = oracle.get(gkey)
        if orc is None or orc <= 0:
            continue

        recs = nvmmh.get_with_mnk(M, N, K, layout_iface[layout_tag], top_k, hw)
        if not recs:
            rows.append(_missing_row(group_id, g, orc, "no_recommendations"))
            continue

        picks: list[tuple[float, str]] = []
        matched_any = False
        rec_slice = recs[:1] if mode == "top1" else recs
        for rec in rec_slice:
            tflops, name, matched = _best_tflops_for_kernel(
                rec["kernel"], layout_tag, M, N, K, runs, tile_to_names
            )
            matched_any = matched_any or matched
            if tflops is not None and name is not None:
                picks.append((tflops, name))

        if not picks:
            reason = "tile_not_in_catalogue" if not matched_any else "no_measured_match"
            rows.append(_missing_row(group_id, g, orc, reason))
            continue

        best_t, best_name = max(picks, key=lambda x: x[0])
        y_norm = best_t / orc
        regret = 1.0 - y_norm
        rows.append(
            {
                "group_id": group_id,
                "M": M,
                "N": N,
                "K": K,
                "layout": layout_tag,
                "regime": g.regime if pd.notna(g.regime) else regime(M, N, K),
                "selected_name": best_name,
                "selected_tflops": best_t,
                "oracle_tflops": orc,
                "regret": regret,
                "top1": int(regret <= 1e-9),
                "top5": 0,  # filled below if rank known
                "within1pct": int(regret <= 0.01),
                "within5pct": int(regret <= 0.05),
                "within10pct": int(regret <= 0.10),
                "nvmmh_mode": mode,
                "match_status": "ok",
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # rank_in_group for top5 (expensive but only 68 groups)
    return df


def _missing_row(group_id, g, orc: float, status: str) -> dict:
    M, N, K = int(g.M), int(g.N), int(g.K)
    return {
        "group_id": group_id,
        "M": M,
        "N": N,
        "K": K,
        "layout": g.layout,
        "regime": g.regime if pd.notna(g.regime) else regime(M, N, K),
        "selected_name": None,
        "selected_tflops": None,
        "oracle_tflops": orc,
        "regret": 1.0,
        "top1": 0,
        "top5": 0,
        "within1pct": 0,
        "within5pct": 0,
        "within10pct": 0,
        "nvmmh_mode": None,
        "match_status": status,
    }


def _add_ranks(df: pd.DataFrame, conn: sqlite3.Connection) -> pd.DataFrame:
    """Attach top5 flag from measured rank within group."""
    if df.empty:
        return df
    rank_rows = pd.read_sql(
        """
        SELECT r.M, r.N, r.K, c.layout_a, c.layout_b, r.name, r.mean_tflops
        FROM runs r
        JOIN configs c ON r.name = c.name
        JOIN eval_plan p ON p.name = r.name AND p.M = r.M AND p.N = r.N AND p.K = r.K
        WHERE p.tag = ? AND r.status = 'success' AND r.mean_tflops > 0
        """,
        conn,
        params=(TAG,),
    )
    la_to_tag = {LAYOUTS[t][1]: {} for t in LAYOUTS}
    for tag, (_, la, lb) in LAYOUTS.items():
        la_to_tag.setdefault(la, {})
    {(la, lb): tag for tag, (_, la, lb) in LAYOUTS.items()}

    def rank_in_group(row):
        if not row.selected_name:
            return 999
        la, lb = _layout_pair(row.layout)
        sub = rank_rows[
            (rank_rows.M == row.M)
            & (rank_rows.N == row.N)
            & (rank_rows.K == row.K)
            & (rank_rows.layout_a == la)
            & (rank_rows.layout_b == lb)
        ].sort_values("mean_tflops", ascending=False)
        names = sub.name.tolist()
        try:
            return names.index(row.selected_name) + 1
        except ValueError:
            return 999

    df = df.copy()
    df["rank_in_group"] = df.apply(rank_in_group, axis=1)
    df["top5"] = (df["rank_in_group"] <= 5).astype(int)
    df["top1"] = (df["rank_in_group"] == 1).astype(int)
    return df


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", type=Path, default=DEFAULT_DB)
    ap.add_argument("--outdir", type=Path, default=PAPER / "nvmmh")
    ap.add_argument("--gpu", default="H100_NVL", help="NvMatmulHeuristicsNvidiaGpu name (H100_NVL best on GH200 eval sweep)")
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--mode", choices=["top1", "bestofk", "both"], default="both")
    ap.add_argument("--tag", default=TAG)
    args = ap.parse_args()

    if not args.db.is_file():
        print(f"ERROR: DB not found: {args.db}", file=sys.stderr)
        return 1

    conn = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    groups = _load_eval_groups(conn)
    oracle, runs, tile_to_names = _load_oracle_and_runs(conn)
    print(f"Loaded {len(groups)} eval groups, {len(runs):,} measured runs, {len(tile_to_names):,} tile keys")

    nvmmh, hw, layout_iface = _init_nvmmh(args.gpu)
    print(f"nvMMH ready: gpu={args.gpu} top_k={args.top_k}")

    args.outdir.mkdir(parents=True, exist_ok=True)
    modes = ["top1", "bestofk"] if args.mode == "both" else [args.mode]
    summary_rows = []

    for mode in modes:
        print(f"\n== nvMMH eval ({mode}) ==")
        perg = eval_nvmmh(groups, oracle, runs, tile_to_names, nvmmh, hw, layout_iface, args.top_k, mode)
        perg = _add_ranks(perg, conn)
        label = f"nvMMH-{mode}"
        metrics = summarize(perg, label)
        metrics["gpu"] = args.gpu
        metrics["top_k"] = args.top_k
        metrics["mode"] = mode
        metrics["within1pct"] = float(perg["within1pct"].mean())
        metrics["within10pct"] = float(perg["within10pct"].mean())
        metrics["match_ok"] = int((perg.match_status == "ok").sum())
        metrics["match_fail"] = int((perg.match_status != "ok").sum())
        by_regime = regime_breakdown(perg)

        out_csv = args.outdir / f"eval_regret_{mode}.csv"
        perg.to_csv(out_csv, index=False)
        metrics_path = args.outdir / f"metrics_{mode}.json"
        payload = {"eval": metrics, "by_regime": by_regime, "gpu": args.gpu, "top_k": args.top_k}
        metrics_path.write_text(json.dumps(payload, indent=2))
        print(f"Wrote {out_csv} and {metrics_path}")

        summary_rows.append(
            {
                "method": label,
                "mode": mode,
                "regret_mean": metrics["regret_mean"],
                "regret_median": metrics["regret_median"],
                "within1pct": metrics.get("within1pct", float(perg.within1pct.mean())),
                "within5pct": metrics["within5pct"],
                "within10pct": metrics.get("within10pct", float(perg.within10pct.mean())),
                "top1": metrics["top1"],
                "top5": metrics["top5"],
                "groups": metrics["groups"],
                "match_ok": metrics["match_ok"],
            }
        )

        fail = perg[perg.match_status != "ok"]
        if len(fail):
            print(f"  WARNING: {len(fail)} groups without catalogue match:")
            for _, r in fail.iterrows():
                print(f"    {r.M}×{r.N}×{r.K} {r.layout}: {r.match_status}")

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(args.outdir / "summary.csv", index=False)

    # Append to paper summary if present
    paper_summary = PAPER / "summary_training.csv"
    if paper_summary.is_file() and len(summary_rows):
        # use top1 as primary nvMMH row for comparison table
        primary = summary_rows[0]
        print(f"\n=== Comparison anchor (nvMMH top1 regret {100*primary['regret_mean']:.1f}%) ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
