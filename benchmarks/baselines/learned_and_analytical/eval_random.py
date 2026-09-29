#!/usr/bin/env python3
"""Evaluate uniform random kernel selection on the 68-group eval split."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from repo_paths import BENCHMARKS, SRC

sys.path.insert(0, str(SRC / "model"))
sys.path.insert(0, str(BENCHMARKS / "baselines" / "learned_and_analytical"))

from common import (  # noqa: E402
    PAPER,
    eval_metrics_bundle,
    load_splits,
    summarize_extended,
    write_run_outputs,
)
from harness import random_pick_expected_per_group, summarize  # noqa: E402

DEFAULT_OUT = PAPER / "random_pick"
DEFAULT_SEED = 42


def _extend_perg(perg):
    perg = perg.copy()
    perg["within1pct"] = (perg["regret"] <= 0.01).astype(int)
    perg["within10pct"] = (perg["regret"] <= 0.10).astype(int)
    perg["ndcg@1"] = 0.0
    perg["ndcg@5"] = 0.0
    perg["ndcg@10"] = 0.0
    return perg


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--features", type=Path, default=None)
    ap.add_argument("--outdir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument(
        "--mode",
        choices=("sample", "expected"),
        default="expected",
        help="expected: 1-mean(y_norm) per group (paper default); sample: one random draw per group",
    )
    args = ap.parse_args()

    _, ev, _, features_path = load_splits(args.features)
    expected = random_pick_expected_per_group(ev)
    expected_metrics = summarize(expected, "random-expected")

    if args.mode == "expected":
        perg = _extend_perg(expected)
        metrics = summarize_extended(perg, "random-expected")
    else:
        rng = np.random.default_rng(args.seed)
        scores = rng.random(len(ev))
        perg, metrics = eval_metrics_bundle(ev, scores, f"random-sample-s{args.seed}")

    extra = {
        "model": "random_pick",
        "mode": args.mode,
        "seed": args.seed if args.mode == "sample" else None,
        "features": str(features_path),
        "expected_regret_mean": expected_metrics["regret_mean"],
        "expected_regret_median": expected_metrics["regret_median"],
    }
    if args.mode == "sample":
        seed_means = []
        for s in range(args.seed, args.seed + 20):
            rng = np.random.default_rng(s)
            sm, _ = eval_metrics_bundle(ev, rng.random(len(ev)), f"random-s{s}")
            seed_means.append(sm["regret"].mean())
        extra["sample_mean_over_20_seeds"] = {
            "seed_start": args.seed,
            "n_seeds": 20,
            "mean_of_means": float(np.mean(seed_means)),
            "std_of_means": float(np.std(seed_means, ddof=1)),
        }

    write_run_outputs(args.outdir, perg, metrics, ev, extra)
    print(f"wrote {args.outdir}  mode={args.mode}  mean={metrics['regret_mean']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
