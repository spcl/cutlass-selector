#!/usr/bin/env python3
"""
Analysis for the feature-vs-capacity study (benchmarks/training/capacity_sweep.sh).

Re-scores every saved checkpoint on the shared validation split and writes:

    capacity_runs.csv  one row per run: size, validation regret / NDCG, convergence
    capacity_gap.csv   one row per (family, capacity): full-vs-structural gap with a
                       paired test over validation groups and the spread over seeds

Regret, not NDCG, is the headline: a selector compiles one kernel, and the two
metrics can disagree (see the paper sweep). The paired test first averages each
group's regret over seeds, then compares full vs structural over the groups, so it
uses the seeds to reduce noise without treating them as independent samples.

Convergence: relative drop of the training loss over the last quarter of training.
A run still falling fast is optimization-limited, not capacity-limited, and would
confound the curve.

Usage:
    python benchmarks/training/analyze_capacity.py --artifacts artifacts/analysis/capacity
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import stats
from sklearn.preprocessing import StandardScaler

from repo_paths import PAPER, SRC

sys.path.insert(0, str(SRC / "model"))
from features import CATEGORY_LEVELS, feature_columns
from harness import grouped_train_val_split, select_and_regret
from train_mlp import MLP, build_X

RUN_RE = re.compile(r"^(mlp|xgb)_mse_(.+)_(full|structural)_s(\d+)$")


def load_validation(features: Path, manifest: dict, val_frac: float, val_seed: int) -> pd.DataFrame:
    cols = list(dict.fromkeys(
        manifest["numeric_features"] + manifest["categorical_features"]
        + ["split", "group_id", "M", "N", "K", "mean_tflops", "group_best_tflops",
           "y_norm", "rank_in_group"]
    ))
    df = pd.read_parquet(features, columns=cols, filters=[("split", "==", "train")])
    for c in manifest["categorical_features"]:
        df[c] = pd.Categorical(df[c].astype(str), categories=CATEGORY_LEVELS[c])
    _, val = grouped_train_val_split(df, val_frac, val_seed)
    return val


def score(run_dir: Path, family: str, meta: dict, val: pd.DataFrame, manifest: dict) -> np.ndarray:
    num, cat = feature_columns(manifest, meta["feature_set"])
    if family == "xgb":
        import xgboost as xgb
        model = xgb.XGBRegressor()
        model.load_model(str(run_dir / "model_A.ubj"))
        return model.predict(val[num + cat])
    ck = torch.load(run_dir / "model_mlp.pt", weights_only=False)
    sc = StandardScaler()
    sc.mean_, sc.scale_ = ck["scaler_mean"], ck["scaler_scale"]
    sc.var_, sc.n_features_in_ = sc.scale_ ** 2, len(num)
    X, _ = build_X(val, num, cat, scaler=sc)
    net = MLP(ck["n_in"], tuple(ck["hidden"]), ck["dropout"])
    net.load_state_dict(ck["model_state"])
    net.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), 65536):
            out.append(net(torch.from_numpy(X[i:i + 65536])).numpy())
    return np.concatenate(out)


def convergence(run_dir: Path, family: str) -> dict:
    h = pd.read_csv(run_dir / ("training_history_xgb.csv" if family == "xgb"
                               else "training_history_mlp.csv"))
    step = "round" if family == "xgb" else "epoch"
    h = h.sort_values(step)
    tr = h["train_metric"].to_numpy()
    q = int(0.75 * len(tr))
    tail_drop = (tr[q] - tr[-1]) / max(abs(tr[-1]), 1e-12)
    vn = h["val_ndcg@10"].dropna()
    return {
        "final_train_loss": float(tr[-1]),
        "train_loss_tail_drop": float(tail_drop),
        "best_val_ndcg@10_step": int(h.loc[vn.idxmax(), step]),
        "steps": int(h[step].max()),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", type=Path, required=True)
    ap.add_argument("--features", type=Path, default=PAPER / "features.parquet")
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--val-seed", type=int, default=42)
    ap.add_argument("--n-boot", type=int, default=4000)
    args = ap.parse_args()

    manifest = json.loads(Path(str(args.features).rsplit(".", 1)[0] + ".manifest.json").read_text())
    val = load_validation(args.features, manifest, args.val_frac, args.val_seed)
    print(f"validation: {len(val):,} rows / {val['group_id'].nunique()} groups")

    runs, pergroup = [], {}
    for mpath in sorted(args.artifacts.glob("*/metrics.json")):
        run_dir = mpath.parent
        m = RUN_RE.match(run_dir.name)
        if not m:
            continue
        family, cap, fs, seed = m.group(1), m.group(2), m.group(3), int(m.group(4))
        meta = json.loads(mpath.read_text())
        if meta.get("val_seed", args.val_seed) != args.val_seed:
            raise SystemExit(f"{run_dir.name} was validated with val_seed={meta['val_seed']}")
        perg = select_and_regret(val, score(run_dir, family, meta, val, manifest), ndcg_ks=(10,))
        pergroup[(family, cap, fs, seed)] = perg.set_index("group_id")["regret"].sort_index()
        runs.append({
            "family": family, "capacity": cap, "feature_set": fs, "seed": seed,
            "params": meta.get("n_params") if family == "mlp" else meta.get("n_leaves"),
            "val_regret": float(perg["regret"].mean()),
            "val_regret_p90": float(perg["regret"].quantile(0.9)),
            "val_within5pct": float(perg["within5pct"].mean()),
            "val_ndcg@10": float(perg["ndcg@10"].mean()),
            "train_seconds": meta.get("train_seconds"),
            **convergence(run_dir, family),
            "run_dir": run_dir.name,
        })
        print(f"  {run_dir.name:44} regret={runs[-1]['val_regret']:.4f}")

    runs_df = pd.DataFrame(runs)
    order = runs_df.groupby(["family", "capacity"])["params"].mean()
    runs_df["_ord"] = [order[(f, c)] for f, c in zip(runs_df.family, runs_df.capacity)]
    runs_df = runs_df.sort_values(["family", "_ord", "feature_set", "seed"]).drop(columns="_ord")
    runs_df.to_csv(args.artifacts / "capacity_runs.csv", index=False)

    rows = []
    rng = np.random.default_rng(0)
    for (family, cap), _ in order.sort_values().groupby(level=[0, 1], sort=False):
        seeds = sorted({k[3] for k in pergroup if k[0] == family and k[1] == cap})
        full = [pergroup[(family, cap, "full", s)] for s in seeds if (family, cap, "full", s) in pergroup]
        stru = [pergroup[(family, cap, "structural", s)] for s in seeds if (family, cap, "structural", s) in pergroup]
        if not full or not stru:
            continue
        fm = pd.concat(full, axis=1).mean(axis=1)
        sm = pd.concat(stru, axis=1).mean(axis=1)
        d = (sm - fm).to_numpy()
        boot = rng.choice(d, (args.n_boot, len(d))).mean(axis=1)
        per_seed = [float(pergroup[(family, cap, "structural", s)].mean()
                          - pergroup[(family, cap, "full", s)].mean())
                    for s in seeds
                    if (family, cap, "full", s) in pergroup and (family, cap, "structural", s) in pergroup]
        sub = runs_df[(runs_df.family == family) & (runs_df.capacity == cap)]
        rows.append({
            "family": family, "capacity": cap,
            "params_full": float(sub[sub.feature_set == "full"]["params"].mean()),
            "params_structural": float(sub[sub.feature_set == "structural"]["params"].mean()),
            "regret_full": float(fm.mean()), "regret_structural": float(sm.mean()),
            "gap": float(d.mean()),
            "gap_ci_lo": float(np.quantile(boot, 0.025)),
            "gap_ci_hi": float(np.quantile(boot, 0.975)),
            "gap_p": float(stats.wilcoxon(sm, fm).pvalue) if np.any(d != 0) else 1.0,
            "gap_seed_mean": float(np.mean(per_seed)),
            "gap_seed_sd": float(np.std(per_seed, ddof=1)) if len(per_seed) > 1 else float("nan"),
            "gap_seeds_positive": int(sum(g > 0 for g in per_seed)),
            "n_seeds": len(per_seed),
            "groups_full_better": float((d > 1e-12).mean()),
            "groups_tied": float((np.abs(d) <= 1e-12).mean()),
        })
    gap_df = pd.DataFrame(rows)
    gap_df.to_csv(args.artifacts / "capacity_gap.csv", index=False)

    print("\nFull-vs-structural validation REGRET gap (structural - full; >0 = hardware-aware helps)")
    print(f"{'family':6} {'capacity':18} {'params(full)':>12} {'full':>7} {'struct':>7} "
          f"{'gap':>8} {'95% CI':>18} {'p':>9} {'seed sd':>8} {'+seeds':>7}")
    for _, r in gap_df.iterrows():
        print(f"{r.family:6} {r.capacity:18} {int(r.params_full):12,} {r.regret_full:7.4f} "
              f"{r.regret_structural:7.4f} {r.gap:+8.4f} [{r.gap_ci_lo:+.4f},{r.gap_ci_hi:+.4f}] "
              f"{r.gap_p:9.1e} {r.gap_seed_sd:8.4f} {r.gap_seeds_positive:>3}/{r.n_seeds}")
    print(f"\nwrote {args.artifacts / 'capacity_runs.csv'} and {args.artifacts / 'capacity_gap.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
