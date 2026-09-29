#!/usr/bin/env python3
"""Quick sweep: nvMMH regret vs GPU descriptor (top-1 tile, DB lookup only)."""

from __future__ import annotations

import argparse
import sqlite3
import sys
import time
from pathlib import Path

import pandas as pd

from repo_paths import PAPER, SRC

sys.path.insert(0, str(SRC / "baseline" / "nvmmh"))
from nvmmh_eval import (  # noqa: E402
    _add_ranks,
    _init_nvmmh,
    _load_eval_groups,
    _load_oracle_and_runs,
    eval_nvmmh,
)

# Hopper-era candidates relevant to GH200 / Alps
CANDIDATE_GPUS = [
    "H100_PCIE",
    "H100_SXM",
    "H100_NVL",
    "H200_SXM",
    "H20_SXM",
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", type=Path, default=Path.home() / "autotuner_bf16_eval.db")
    ap.add_argument("--gpus", nargs="*", default=CANDIDATE_GPUS)
    ap.add_argument("--out", type=Path, default=PAPER / "nvmmh" / "gpu_sweep.csv")
    args = ap.parse_args()

    t0 = time.perf_counter()
    conn = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    groups = _load_eval_groups(conn)
    oracle, runs, tile_to_names = _load_oracle_and_runs(conn)
    print(f"DB loaded in {time.perf_counter()-t0:.1f}s ({len(groups)} groups)")

    rows = []
    for gpu in args.gpus:
        t1 = time.perf_counter()
        try:
            nvmmh, hw, layout_iface = _init_nvmmh(gpu)
        except Exception as e:
            print(f"  {gpu}: SKIP ({e})")
            continue
        perg = eval_nvmmh(groups, oracle, runs, tile_to_names, nvmmh, hw, layout_iface, top_k=8, mode="top1")
        perg = _add_ranks(perg, conn)
        r = perg.regret
        rows.append(
            {
                "gpu": gpu,
                "regret_mean": float(r.mean()),
                "regret_median": float(r.median()),
                "regret_p95": float(r.quantile(0.95)),
                "within5pct": float(perg.within5pct.mean()),
                "within10pct": float(perg.within10pct.mean()),
                "top1": float(perg.top1.mean()),
                "top5": float(perg.top5.mean()),
                "match_ok": int((perg.match_status == "ok").sum()),
                "seconds": round(time.perf_counter() - t1, 1),
            }
        )
        print(
            f"  {gpu:12s}  regret={100*r.mean():5.1f}%  med={100*r.median():5.1f}%  "
            f"<5%={100*perg.within5pct.mean():4.1f}%  ({time.perf_counter()-t1:.1f}s)"
        )

    df = pd.DataFrame(rows).sort_values("regret_mean")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"\nBest GPU for nvMMH on this eval set: {df.iloc[0]['gpu']} ({100*df.iloc[0]['regret_mean']:.1f}% mean regret)")
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
