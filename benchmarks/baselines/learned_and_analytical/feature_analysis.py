#!/usr/bin/env python3
"""Training-only feature correlation / redundancy / effect-size analysis."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from repo_paths import SRC

sys.path.insert(0, str(SRC / "model"))
from common import (  # noqa: E402
    BASELINES,
    DEFAULT_FEATURES,
    MIN_GROUP_SAMPLES,
    load_splits,
    roofline_bound_regime,
    spearman,
)
from features import NUMERIC_FEATURES  # noqa: E402
from harness import regime  # noqa: E402

NUMERIC_FOR_ANALYSIS = [c for c in NUMERIC_FEATURES if c not in {"log2_M", "log2_N", "log2_K"}]


def _within_group_correlations(train: pd.DataFrame, features: list[str], group_label: str) -> pd.DataFrame:
    rows = []
    for feat in features:
        rhos = []
        for _, g in train.groupby("group_id", sort=False):
            if len(g) < MIN_GROUP_SAMPLES:
                continue
            rho = spearman(g[feat], g["mean_tflops"])
            if np.isfinite(rho):
                rhos.append(rho)
        if not rhos:
            continue
        arr = np.asarray(rhos, dtype=np.float64)
        rows.append(
            {
                "feature": feat,
                "group_label": group_label,
                "n_groups": int(len(arr)),
                "mean_rho": float(arr.mean()),
                "median_rho": float(np.median(arr)),
                "std_rho": float(arr.std(ddof=0)),
                "iqr_rho": float(np.quantile(arr, 0.75) - np.quantile(arr, 0.25)),
                "frac_positive": float((arr > 0).mean()),
                "frac_abs_gt_0.1": float((np.abs(arr) > 0.1).mean()),
                "frac_abs_gt_0.2": float((np.abs(arr) > 0.2).mean()),
                "frac_abs_gt_0.3": float((np.abs(arr) > 0.3).mean()),
                "global_spearman": spearman(train[feat], train["mean_tflops"]),
            }
        )
    return pd.DataFrame(rows)


def _add_roofline_regime(train: pd.DataFrame) -> pd.DataFrame:
    d = train.copy()
    if "problem_arith_intensity" in d.columns:
        d["roofline_regime"] = d["problem_arith_intensity"].map(roofline_bound_regime)
    else:
        d["roofline_regime"] = "unknown"
    if "regime" not in d.columns:
        d["regime"] = [regime(m, n, k) for m, n, k in d[["M", "N", "K"]].to_numpy()]
    return d


def redundancy_analysis(train: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    corr = train[features].corr(method="spearman")
    rows = []
    for i, a in enumerate(features):
        for b in features[i + 1 :]:
            rho = corr.loc[a, b]
            if np.isfinite(rho) and abs(rho) >= 0.9:
                rows.append({"feature_a": a, "feature_b": b, "spearman_rho": float(rho)})
    return pd.DataFrame(rows).sort_values("spearman_rho", key=np.abs, ascending=False)


def top5_effect_sizes(train: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    rows = []
    for feat in features:
        effects = []
        for _, g in train.groupby("group_id", sort=False):
            if len(g) < 20:
                continue
            thresh = g["mean_tflops"].quantile(0.95)
            top = g[g["mean_tflops"] >= thresh][feat]
            rest = g[g["mean_tflops"] < thresh][feat]
            if len(top) < 2 or len(rest) < 2:
                continue
            pooled = float(np.sqrt((top.var(ddof=1) + rest.var(ddof=1)) / 2.0))
            if pooled <= 1e-12:
                continue
            effects.append((float(top.mean()) - float(rest.mean())) / pooled)
        if effects:
            arr = np.asarray(effects)
            rows.append(
                {
                    "feature": feat,
                    "n_groups": int(len(arr)),
                    "mean_effect": float(arr.mean()),
                    "median_effect": float(np.median(arr)),
                    "std_effect": float(arr.std(ddof=0)),
                }
            )
    return pd.DataFrame(rows).sort_values("mean_effect", key=np.abs, ascending=False)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    ap.add_argument("--outdir", type=Path, default=BASELINES)
    args = ap.parse_args()

    train, _, _, features_path = load_splits(args.features)
    train = _add_roofline_regime(train)
    args.outdir.mkdir(parents=True, exist_ok=True)

    print(
        f"feature analysis: {features_path}  "
        f"train rows={len(train):,} groups={train['group_id'].nunique():,}"
    )

    overall = _within_group_correlations(train, NUMERIC_FOR_ANALYSIS, "all")
    if overall.empty:
        raise RuntimeError("no within-group correlations computed — check train data and feature columns")
    overall.to_csv(args.outdir / "feature_correlations.csv", index=False)

    regime_rows: list[pd.DataFrame] = []
    for label, sub in train.groupby("regime", sort=False):
        if sub.empty:
            continue
        part = _within_group_correlations(sub, NUMERIC_FOR_ANALYSIS, str(label))
        if not part.empty:
            regime_rows.append(part)
    for label, sub in train.groupby("roofline_regime", sort=False):
        if label == "unknown" or sub.empty:
            continue
        part = _within_group_correlations(sub, NUMERIC_FOR_ANALYSIS, f"roofline:{label}")
        if not part.empty:
            regime_rows.append(part)
    if regime_rows:
        by_regime = pd.concat(regime_rows, ignore_index=True)
    else:
        by_regime = overall.assign(group_label="all").copy()
    by_regime.to_csv(args.outdir / "feature_correlations_by_regime.csv", index=False)

    redundancy_analysis(train, NUMERIC_FOR_ANALYSIS).to_csv(args.outdir / "feature_redundancy.csv", index=False)
    top5_effect_sizes(train, NUMERIC_FOR_ANALYSIS).to_csv(args.outdir / "top5_effect_sizes.csv", index=False)

    print(f"wrote analysis CSVs under {args.outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
