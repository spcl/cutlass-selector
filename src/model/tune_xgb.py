#!/usr/bin/env python3
"""
Hyperparameter tuner for the XGBoost ranker.

Uses Optuna TPE to minimise shape-grouped CV regret_mean over the training split.
The eval split is never touched during tuning.

Single-node multi-GPU usage via shared SQLite study (workers collaborate):
    STORAGE="sqlite:////dev/shm/tune_$SLURM_JOB_ID/optuna.db"
    for GPU in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=$GPU python src/model/tune_xgb.py --features ... \\
            --device cuda --trials 75 --seed $((42+GPU)) \\
            --storage "$STORAGE" --study-name tune_A --suffix gpu$GPU &
    done
    wait

Local CPU usage:
    python src/model/tune_xgb.py --features artifacts/analysis/paper/features.parquet --trials 60 --cv 3

By default --loss is unset: each trial also picks its training objective
(ndcg / pairwise / mse) as part of the search, so TPE allocates more trials
to whichever objective performs best. Pass --loss to pin a single objective.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import train_xgb as train_module  # noqa: E402

# ── CV objective (no global state; safe for parallel workers) ─────────────

def _cv_regret(
    train_df: pd.DataFrame,
    feat_cols,
    grade_max: int,
    params: dict,
    n_estimators: int,
    rel_top_bands: int,
    n_splits: int,
    seed: int,
    device: str,
    loss: str,
) -> float:
    from sklearn.model_selection import GroupKFold

    groups = train_module.base_shape_id(train_df)
    gkf = GroupKFold(n_splits=n_splits)
    fold_regrets = []
    for i, (tr, va) in enumerate(gkf.split(train_df, groups=groups), 1):
        print(f"    fold {i}/{n_splits} ...", flush=True)
        a = train_module._sorted_by_group(train_df.iloc[tr])
        b = train_df.iloc[va]
        if loss == "mse":
            m = train_module.fit_regressor(a[feat_cols], a["y_norm"].to_numpy(), params, n_estimators, seed, device=device)
            pred = m.predict(b[feat_cols])
        else:
            m = train_module.fit_ranker(
                a[feat_cols],
                train_module.make_relevance(a["relevance_grade"].to_numpy(), grade_max, rel_top_bands),
                a["group_id"].to_numpy(),
                params,
                n_estimators,
                seed,
                device=device,
                objective=train_module.LOSS_OBJECTIVES[loss],
            )
            import xgboost as xgb
            dval = xgb.DMatrix(b[feat_cols], enable_categorical=True)
            pred = m.get_booster().predict(dval)
        perg = train_module.select_and_regret(b, pred)
        fold_regrets.append(float(perg["regret"].mean()))
        print(f"    fold {i}/{n_splits}  regret={fold_regrets[-1]:.4f}", flush=True)
    return float(np.mean(fold_regrets))


# ── Optuna objective ────────────────────────────────────────────────────────

def make_objective(train_df, feat_cols, grade_max, n_splits, seed, device, loss):
    """loss=None: the training objective is itself part of the search (per-trial
    categorical choice). loss='ndcg'/'pairwise'/'mse': pinned for every trial."""
    def objective(trial):
        params = dict(
            eta=trial.suggest_float("eta", 0.005, 0.3, log=True),
            max_depth=trial.suggest_int("max_depth", 4, 12),
            min_child_weight=trial.suggest_float("min_child_weight", 1.0, 30.0, log=True),
            subsample=trial.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
            reg_lambda=trial.suggest_float("reg_lambda", 0.01, 20.0, log=True),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            gamma=trial.suggest_float("gamma", 0.0, 5.0),
        )
        n_estimators = trial.suggest_int("n_estimators", 300, 1200)
        trial_loss = loss if loss is not None else trial.suggest_categorical("loss", ["ndcg", "pairwise", "mse"])
        # rel_top_bands is a ranking-relevance-band knob; meaningless for pointwise mse.
        rel_top_bands = trial.suggest_int("rel_top_bands", 2, 5) if trial_loss != "mse" else None

        print(
            f"\ntrial {trial.number}  loss={trial_loss}  n_est={n_estimators} depth={params['max_depth']} "
            f"eta={params['eta']:.4f} rtb={rel_top_bands}",
            flush=True,
        )
        return _cv_regret(train_df, feat_cols, grade_max, params, n_estimators,
                          rel_top_bands, n_splits, seed, device, trial_loss)

    return objective


# ── main ────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--trials", type=int, default=100)
    ap.add_argument("--cv", type=int, default=5)
    ap.add_argument("--loss", choices=["ndcg", "pairwise", "mse"], default=None,
                    help="pin a single training objective; default searches all three")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cpu", help="'cpu' or 'cuda'")
    ap.add_argument("--storage", default=None, help="Optuna storage URL, e.g. sqlite:////dev/shm/.../optuna.db")
    ap.add_argument("--study-name", default="tune_A", help="shared study name when using --storage")
    ap.add_argument("--suffix", default="", help="appended to output filename, e.g. 'gpu0'")
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()

    # XGBoost version check — device='cuda' requires >= 2.0.
    if args.device == "cuda":
        import xgboost as xgb
        major = int(xgb.__version__.split(".")[0])
        if major < 2:
            print(f"ERROR: device='cuda' requires XGBoost >= 2.0, found {xgb.__version__}", flush=True)
            return 1

    fpath = Path(args.features)
    outdir = Path(args.outdir) if args.outdir else fpath.parent
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(fpath)
    manifest = json.loads(Path(str(fpath).rsplit(".", 1)[0] + ".manifest.json").read_text())
    grade_max = int(manifest["hw_constants"]["GRADE_MAX"])
    feat_cols = manifest["numeric_features"] + manifest["categorical_features"]

    for c in manifest["categorical_features"]:
        df[c] = df[c].astype("category")

    train_df = df[df["split"] == "train"].copy()
    print(f"train={len(train_df):,} rows / {train_df['group_id'].nunique():,} groups")
    print(f"device={args.device}  trials={args.trials}  cv={args.cv}-fold  "
          f"loss={args.loss or 'searched (ndcg/pairwise/mse)'}")
    print("CV oracle caveat: fold-val oracle is sample-best (proxy), not true-oracle; "
          "CV regret valid for RELATIVE comparison only.")
    if args.storage:
        print(f"storage={args.storage}  study={args.study_name}")
    print()

    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    sampler = optuna.samplers.TPESampler(seed=args.seed)
    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        direction="minimize",
        sampler=sampler,
        load_if_exists=True,
    )

    best_so_far = [float("inf")]

    def callback(study, trial):
        if trial.value is not None and trial.value < best_so_far[0]:
            best_so_far[0] = trial.value
            print(f"  --> new best: trial {trial.number}  regret={trial.value:.4f}", flush=True)

    study.optimize(
        make_objective(train_df, feat_cols, grade_max, args.cv, args.seed, args.device, args.loss),
        n_trials=args.trials,
        n_jobs=1,
        show_progress_bar=False,
        callbacks=[callback],
    )

    best = study.best_trial
    print(f"\n== Best trial #{best.number}  CV regret_mean={best.value:.4f} ==")
    best_params = dict(best.params)
    best_loss = best_params.pop("loss", args.loss)
    rel_top_bands = best_params.pop("rel_top_bands", None)
    n_estimators = best_params.pop("n_estimators")
    print(f"  loss         : {best_loss}")
    print(f"  n_estimators : {n_estimators}")
    print(f"  rel_top_bands: {rel_top_bands}")
    for k, v in best_params.items():
        print(f"  {k:20}: {v:.5g}")

    print("\n== Top-10 trials ==")
    trials_df = study.trials_dataframe(attrs=("number", "value", "params"))
    print(trials_df.sort_values("value").head(10).to_string(index=False))

    suffix = f"_{args.suffix}" if args.suffix else ""
    out_path = outdir / f"tune_results{suffix}.json"
    result = {
        "best_cv_regret_mean": best.value,
        "best_trial": best.number,
        "loss": best_loss,
        "xgb_params": best_params,
        "n_estimators": n_estimators,
        "rel_top_bands": rel_top_bands,
        "all_trials": [
            {"number": t.number, "value": t.value, "params": t.params}
            for t in study.trials
            if t.value is not None
        ],
    }
    out_path.write_text(json.dumps(result, indent=2))
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
