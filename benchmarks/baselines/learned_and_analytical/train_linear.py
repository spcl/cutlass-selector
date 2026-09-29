#!/usr/bin/env python3
"""Ridge linear baseline over the same feature matrix as MLP/XGBoost."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from repo_paths import SRC

sys.path.insert(0, str(SRC / "model"))

from common import (
    BASELINES,
    DEFAULT_FEATURES,
    VAL_FRAC,
    VAL_SEED,
    eval_metrics_bundle,
    feature_column_names,
    feature_sets,
    load_splits,
    mean_ndcg,
    mean_selection_regret,
    train_val_split,
    write_run_outputs,
)
from feature_manifest import category_levels_for  # noqa: E402
from train_mlp import build_X  # noqa: E402

ALPHAS = [0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]


class RidgeLinear:
    """Closed-form ridge: (X'X + αI)^{-1} X'y with intercept via column centering."""

    def __init__(self, alpha: float = 1.0):
        self.alpha = float(alpha)
        self.coef_: np.ndarray | None = None
        self.intercept_: float = 0.0
        self._x_mean: np.ndarray | None = None
        self._y_mean: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> RidgeLinear:
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        self._x_mean = X.mean(axis=0)
        self._y_mean = float(y.mean())
        Xc = X - self._x_mean
        yc = y - self._y_mean
        n_feat = Xc.shape[1]
        aI = self.alpha * np.eye(n_feat, dtype=np.float64)
        self.coef_ = np.linalg.solve(Xc.T @ Xc + aI, Xc.T @ yc)
        self.intercept_ = self._y_mean - float(self._x_mean @ self.coef_)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        return (X @ self.coef_ + self.intercept_).astype(np.float32)


def ridge_coefficients(
    model: RidgeLinear,
    scaler,
    feature_names: list[str],
    out_path: Path,
) -> pd.DataFrame:
    coef_std = model.coef_
    intercept_std = float(model.intercept_)
    n_num = len(scaler.mean_)
    scale = scaler.scale_
    mean = scaler.mean_
    coef_orig = np.empty_like(coef_std, dtype=np.float64)
    coef_orig[:n_num] = coef_std[:n_num] / scale
    coef_orig[n_num:] = coef_std[n_num:]  # one-hot block is unscaled
    intercept_orig = intercept_std - float(np.sum(coef_orig[:n_num] * mean))

    rows = []
    for name, c_std, c_orig in zip(feature_names, coef_std, coef_orig):
        rows.append(
            {
                "feature": name,
                "coefficient": float(c_orig),
                "coefficient_standardized": float(c_std),
                "abs_coefficient": float(abs(c_orig)),
                "abs_coefficient_standardized": float(abs(c_std)),
                "sign": int(np.sign(c_std)) if c_std != 0 else 0,
            }
        )
    df = pd.DataFrame(rows).sort_values("abs_coefficient_standardized", ascending=False)
    meta = pd.DataFrame(
        [
            {
                "feature": "__intercept__",
                "coefficient": intercept_orig,
                "coefficient_standardized": intercept_std,
                "abs_coefficient": abs(intercept_orig),
                "abs_coefficient_standardized": abs(intercept_std),
                "sign": int(np.sign(intercept_std)) if intercept_std != 0 else 0,
            }
        ]
    )
    out = pd.concat([df, meta], ignore_index=True)
    out.to_csv(out_path, index=False)
    return out


def tune_ridge_alpha(
    X_fit: np.ndarray,
    y_fit: np.ndarray,
    val_df: pd.DataFrame,
    X_val: np.ndarray,
) -> tuple[float, list[dict]]:
    records = []
    best_alpha, best_ndcg = ALPHAS[0], -1.0
    for alpha in ALPHAS:
        model = RidgeLinear(alpha=alpha).fit(X_fit, y_fit)
        val_scores = model.predict(X_val)
        val_ndcg = mean_ndcg(val_df, val_scores, k=10)
        val_regret = mean_selection_regret(val_df, val_scores)
        records.append(
            {
                "alpha": alpha,
                "val_ndcg@10": val_ndcg,
                "val_regret_mean": val_regret,
            }
        )
        if val_ndcg > best_ndcg:
            best_ndcg = val_ndcg
            best_alpha = alpha
    return best_alpha, records


def train_linear_baseline(
    features_path: Path,
    feature_set: str,
    outdir: Path,
) -> dict:
    train, ev, manifest, features_path = load_splits(features_path)
    print(f"using features: {features_path}")
    category_levels = category_levels_for(manifest)
    num_cols, cat_cols = feature_sets(manifest)[feature_set]
    feat_names = feature_column_names(num_cols, cat_cols, category_levels)

    fit, val = train_val_split(train, val_frac=VAL_FRAC, seed=VAL_SEED)
    print(
        f"linear ({feature_set}): fit={len(fit):,}/{fit['group_id'].nunique()} groups  "
        f"val={len(val):,}/{val['group_id'].nunique()} groups  eval={ev['group_id'].nunique()} groups"
    )

    X_fit, scaler = build_X(fit, num_cols, cat_cols, category_levels)
    X_val, _ = build_X(val, num_cols, cat_cols, category_levels, scaler)
    X_train, _ = build_X(train, num_cols, cat_cols, category_levels, scaler)
    X_eval, _ = build_X(ev, num_cols, cat_cols, category_levels, scaler)
    y_fit = fit["y_norm"].to_numpy(dtype=np.float64)

    t0 = time.perf_counter()
    best_alpha, alpha_sweep = tune_ridge_alpha(X_fit, y_fit, val, X_val)
    chosen_val = next(r for r in alpha_sweep if r["alpha"] == best_alpha)
    print(
        f"  selected alpha={best_alpha} (val NDCG@10={chosen_val['val_ndcg@10']:.4f}, "
        f"val regret={chosen_val['val_regret_mean']:.4f})"
    )

    model = RidgeLinear(alpha=best_alpha).fit(X_train, train["y_norm"].to_numpy(dtype=np.float64))
    train_seconds = time.perf_counter() - t0

    scores = model.predict(X_eval)
    label = f"Linear-ridge-{feature_set}"
    perg, metrics = eval_metrics_bundle(ev, scores, label)

    outdir.mkdir(parents=True, exist_ok=True)
    ridge_coefficients(model, scaler, feat_names, outdir / "coefficients.csv")
    assert len(feat_names) == len(model.coef_)
    pd.DataFrame(alpha_sweep).to_csv(outdir / "alpha_sweep.csv", index=False)

    extra = {
        "model": "ridge_linear",
        "feature_set": feature_set,
        "alpha": best_alpha,
        "validation_ndcg@10": chosen_val["val_ndcg@10"],
        "validation_regret_mean": chosen_val["val_regret_mean"],
        "alpha_sweep": alpha_sweep,
        "val_frac": VAL_FRAC,
        "val_seed": VAL_SEED,
        "numeric_features": num_cols,
        "categorical_features": cat_cols,
        "input_dim": int(X_train.shape[1]),
        "train_seconds": train_seconds,
        "data_split_protocol": {
            "alpha_tuning": "train fit split (85% base shapes)",
            "alpha_selection_metric": "validation NDCG@10",
            "final_fit": "full training split (excludes 68 eval groups)",
            "exhaustive_eval": "68 held-out groups (split==eval)",
        },
    }
    write_run_outputs(outdir, perg, metrics, ev, extra)
    (outdir / "model_ridge.json").write_text(
        json.dumps(
            {
                "alpha": best_alpha,
                "coef": model.coef_.tolist(),
                "intercept": float(model.intercept_),
                "scaler_mean": scaler.mean_.tolist(),
                "scaler_scale": scaler.scale_.tolist(),
                "feature_names": feat_names,
            },
            indent=2,
        )
    )
    print(f"wrote {outdir}")
    return metrics


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    ap.add_argument("--feature-set", choices=["full", "structural", "both"], default="both")
    ap.add_argument("--outdir", type=Path, default=BASELINES)
    args = ap.parse_args()

    sets = ["full", "structural"] if args.feature_set == "both" else [args.feature_set]
    for fs in sets:
        sub = args.outdir / f"linear_{fs}"
        train_linear_baseline(args.features, fs, sub)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
