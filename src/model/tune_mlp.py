#!/usr/bin/env python3
"""
Hyperparameter tuner for the LambdaRank MLP.

Uses Optuna TPE to minimise shape-grouped CV regret_mean over the training split.
The eval split is never touched during tuning.

Single-node multi-GPU usage via shared SQLite study (workers collaborate):
    STORAGE="sqlite:////dev/shm/tune_mlp_$SLURM_JOB_ID/optuna.db"
    for GPU in 0 1 2 3; do
        CUDA_VISIBLE_DEVICES=$GPU python src/model/tune_mlp.py --features ... \\
            --device cuda --trials 100 --sampler-seed $((42+GPU)) \\
            --train-seeds 0 1 --cv 3 \\
            --storage "$STORAGE" --study-name tune_mlp --suffix gpu$GPU &
    done
    wait

Search uses 3-fold/2-seed (6 fits/trial) for relative ranking.
The winner is re-validated at 5-fold/3-seed in the separate lock step.

Local smoke test:
    python src/model/tune_mlp.py --features artifacts/analysis/paper/features.parquet \\
        --trials 12 --cv 3 --train-seeds 0 1 --device mps
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# model/ on path for harness, features, train_mlp.
sys.path.insert(0, str(Path(__file__).parent))

import harness
from features import CATEGORY_LEVELS
from train_mlp import MLP, GroupDataset, build_X, predict, train_lambdarank, train_mlp

_HIDDEN_CHOICES = [
    (128, 64),
    (256, 128, 64),
    (256, 256, 256),
    (512, 256, 128),
    (512, 512, 256),
    (1024, 512, 256),
    (256, 128, 64, 32),
    (512, 256, 128, 64),
    (1024, 512, 256, 128),
    (1024, 1024, 512, 256),
]

_HAND_SET = dict(
    lr=5e-4,
    hidden=(256, 128, 64),
    dropout=0.1,
    weight_decay=0.0,
    groups_per_batch=64,
    epochs=30,
    grad_clip=0.0,
    max_pairs_per_group=2000,
)


# ── CV objective (no global state; safe for parallel workers) ─────────────

def _cv_regret_mlp(
    train_df: pd.DataFrame,
    num_cols: list[str],
    cat_cols: list[str],
    trial_params: dict,
    n_splits: int,
    train_seeds: list[int],
    device: str,
    trial=None,   # optuna.Trial; pass for pruning, None outside Optuna (e.g. final revalidation)
) -> float:
    import optuna
    from sklearn.model_selection import GroupKFold

    loss = trial_params["loss"]
    hidden = trial_params["hidden"]
    lr = trial_params["lr"]
    dropout = trial_params["dropout"]
    weight_decay = trial_params["weight_decay"]
    epochs = trial_params["epochs"]

    groups = harness.base_shape_id(train_df)
    gkf = GroupKFold(n_splits=n_splits)
    # Pre-materialise folds: deterministic (fold-major, seed-minor) order is
    # required so every trial reports intermediate values at identical step indices.
    fold_splits = list(gkf.split(train_df, groups=groups))

    all_regrets: list[float] = []
    unit = 0  # step index reported to pruner; incremented after each (fold, seed) unit

    for fold_i, (tr, va) in enumerate(fold_splits, 1):
        fold_tr = train_df.iloc[tr]
        fold_va = train_df.iloc[va]

        X_tr, scaler = build_X(fold_tr, num_cols, cat_cols)
        X_va, _ = build_X(fold_va, num_cols, cat_cols, scaler=scaler)
        y_tr = fold_tr["y_norm"].to_numpy(dtype="float32")
        group_ids = fold_tr["group_id"].to_numpy()

        for seed in train_seeds:
            if trial is not None and trial.number == 0 and unit == 0:
                print(
                    "  [GPU check] pair construction is CPU-side — "
                    "run `nvidia-smi` to verify GPU utilisation",
                    flush=True,
                )
            print(f"    fold {fold_i}/{n_splits}  seed={seed}  loss={loss} ...", flush=True)
            torch.manual_seed(seed)
            if loss == "mse":
                model = train_mlp(
                    X_tr, y_tr,
                    epochs=epochs,
                    batch_size=trial_params["batch_size"],
                    lr=lr,
                    hidden=hidden,
                    dropout=dropout,
                    seed=seed,
                    device=device,
                    weight_decay=weight_decay,
                    verbose=False,
                )
            else:
                dataset = GroupDataset(X_tr, y_tr, group_ids, max_rows=0, seed=seed)
                model = MLP(X_tr.shape[1], hidden, dropout).to(device)
                model = train_lambdarank(
                    dataset, model,
                    epochs=epochs,
                    groups_per_batch=trial_params["groups_per_batch"],
                    lr=lr,
                    max_pairs=trial_params["max_pairs_per_group"],
                    top_k=8,
                    device=device,
                    seed=seed,
                    grad_clip=trial_params["grad_clip"],
                    gain_mode=trial_params["gain"],
                    weighted=(loss == "lambdarank"),
                    weight_decay=weight_decay,
                    verbose=False,
                )
            scores = predict(model, X_va, device)
            perg = harness.select_and_regret(fold_va, scores)
            regret = float(perg["regret"].mean())
            all_regrets.append(regret)
            print(f"    fold {fold_i}/{n_splits}  seed={seed}  regret={regret:.4f}", flush=True)

            if trial is not None:
                running_mean = float(np.mean(all_regrets))
                trial.report(running_mean, step=unit)
                if trial.should_prune():
                    raise optuna.TrialPruned()
            unit += 1

    return float(np.mean(all_regrets))


# ── Optuna objective ────────────────────────────────────────────────────────

def make_objective(train_df, num_cols, cat_cols, n_splits, train_seeds, device, loss):
    """loss=None: the training objective is itself part of the search (per-trial
    categorical choice). loss='lambdarank'/'ranknet'/'mse': pinned for every trial."""
    def objective(trial):
        trial_loss = loss if loss is not None else trial.suggest_categorical(
            "loss", ["lambdarank", "ranknet", "mse"]
        )
        lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
        hidden_idx = trial.suggest_categorical("hidden_idx", list(range(len(_HIDDEN_CHOICES))))
        hidden = _HIDDEN_CHOICES[hidden_idx]
        dropout = trial.suggest_float("dropout", 0.0, 0.4)
        weight_decay = trial.suggest_categorical(
            "weight_decay", [0.0, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3]
        )
        epochs = trial.suggest_int("epochs", 20, 150)

        trial_params = dict(
            loss=trial_loss,
            lr=lr,
            hidden=hidden,
            dropout=dropout,
            weight_decay=weight_decay,
            epochs=epochs,
        )
        if trial_loss == "mse":
            trial_params["batch_size"] = trial.suggest_categorical(
                "batch_size", [1024, 2048, 4096, 8192]
            )
            print(
                f"\ntrial {trial.number}  loss={trial_loss}  lr={lr:.2e}  hidden={hidden}  "
                f"dropout={dropout:.2f}  wd={weight_decay}  epochs={epochs}  "
                f"batch_size={trial_params['batch_size']}",
                flush=True,
            )
        else:
            trial_params["groups_per_batch"] = trial.suggest_categorical(
                "groups_per_batch", [32, 64, 128, 256]
            )
            trial_params["grad_clip"] = trial.suggest_categorical(
                "grad_clip", [0.0, 0.5, 1.0, 5.0, 10.0]
            )
            trial_params["max_pairs_per_group"] = trial.suggest_categorical(
                "max_pairs_per_group", [2000, 4000, 8000, 16000, 32000, 64000]
            )
            # gain is meaningless for ranknet (no NDCG delta weight); keep it fixed there.
            trial_params["gain"] = (
                trial.suggest_categorical("gain", ["linear", "exp"])
                if trial_loss == "lambdarank" else "linear"
            )
            print(
                f"\ntrial {trial.number}  loss={trial_loss}  lr={lr:.2e}  hidden={hidden}  "
                f"dropout={dropout:.2f}  wd={weight_decay}  gpb={trial_params['groups_per_batch']}  "
                f"epochs={epochs}  clip={trial_params['grad_clip']}  "
                f"max_pairs={trial_params['max_pairs_per_group']}  gain={trial_params['gain']}",
                flush=True,
            )
        return _cv_regret_mlp(
            train_df, num_cols, cat_cols, trial_params, n_splits, train_seeds, device,
            trial=trial,
        )

    return objective


# ── main ────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--trials", type=int, default=100)
    ap.add_argument("--loss", choices=["lambdarank", "ranknet", "mse"], default=None,
                    help="pin a single training objective; default searches all three")
    ap.add_argument("--cv", type=int, default=3,
                    help="folds during search (3); re-validate winner at 5-fold in lock step")
    ap.add_argument("--train-seeds", type=int, nargs="+", default=[0, 1],
                    help="seeds averaged per trial (2 for search, 3 for final lock step)")
    ap.add_argument("--sampler-seed", type=int, default=42,
                    help="TPE exploration seed; use different values per worker")
    ap.add_argument("--device", default="cpu", help="'cpu', 'cuda', or 'mps'")
    ap.add_argument("--storage", default=None,
                    help="Optuna storage URL, e.g. sqlite:////dev/shm/.../optuna.db")
    ap.add_argument("--study-name", default="tune_mlp",
                    help="shared study name when using --storage")
    ap.add_argument("--suffix", default="",
                    help="appended to output filename, e.g. 'gpu0'")
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()

    fpath = Path(args.features)
    outdir = Path(args.outdir) if args.outdir else fpath.parent
    outdir.mkdir(parents=True, exist_ok=True)

    manifest = json.loads(Path(str(fpath).rsplit(".", 1)[0] + ".manifest.json").read_text())
    num_cols: list[str] = manifest["numeric_features"]
    cat_cols: list[str] = manifest["categorical_features"]

    df = pd.read_parquet(fpath)
    for c in cat_cols:
        df[c] = pd.Categorical(df[c].astype(str), categories=CATEGORY_LEVELS[c])

    train_df = df[df["split"] == "train"].copy()
    print(f"train={len(train_df):,} rows / {train_df['group_id'].nunique():,} groups")
    print(f"device={args.device}  trials={args.trials}  cv={args.cv}-fold  "
          f"train_seeds={args.train_seeds}  loss={args.loss or 'searched (lambdarank/ranknet/mse)'}")
    print("CV oracle caveat: fold-val oracle is sample-best (proxy); "
          "regret valid for RELATIVE comparison only.")
    if args.storage:
        print(f"storage={args.storage}  study={args.study_name}")
    print()

    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    sampler = optuna.samplers.TPESampler(seed=args.sampler_seed)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=2)
    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True,
    )

    best_so_far = [float("inf")]

    def callback(study, trial):
        if trial.value is not None and trial.value < best_so_far[0]:
            best_so_far[0] = trial.value
            print(
                f"  --> new best: trial {trial.number}  regret={trial.value:.4f}",
                flush=True,
            )

    study.optimize(
        make_objective(train_df, num_cols, cat_cols, args.cv, args.train_seeds, args.device, args.loss),
        n_trials=args.trials,
        n_jobs=1,
        show_progress_bar=False,
        callbacks=[callback],
    )

    best = study.best_trial
    print(f"\n== Best trial #{best.number}  CV regret_mean={best.value:.4f} ==")

    best_params_raw = dict(best.params)
    best_loss = best_params_raw.pop("loss", args.loss)
    hidden = list(_HIDDEN_CHOICES[best_params_raw.pop("hidden_idx")])
    best_mlp = {
        "loss":         best_loss,
        "lr":           best_params_raw.pop("lr"),
        "hidden":       hidden,
        "dropout":      best_params_raw.pop("dropout"),
        "weight_decay": best_params_raw.pop("weight_decay"),
        "epochs":       best_params_raw.pop("epochs"),
    }
    if best_loss == "mse":
        best_mlp["batch_size"] = best_params_raw.pop("batch_size", None)
    else:
        best_mlp["groups_per_batch"]    = best_params_raw.pop("groups_per_batch", None)
        best_mlp["grad_clip"]           = best_params_raw.pop("grad_clip", None)
        best_mlp["max_pairs_per_group"] = best_params_raw.pop("max_pairs_per_group", None)
        best_mlp["gain"]                = best_params_raw.pop("gain", "linear")
    for k, v in best_mlp.items():
        print(f"  {k:22}: {v}")

    print("\n== Top-10 trials (by CV regret_mean) ==")
    trials_df = study.trials_dataframe(attrs=("number", "value", "params"))
    print(trials_df.sort_values("value").head(10).to_string(index=False))

    if best_loss == "lambdarank":
        print("\n== Winner vs. hand-set config ==")
        print(f"  {'param':22}  {'hand-set':>22}  {'winner':>22}")
        print("  " + "-" * 70)
        for k in ("lr", "hidden", "dropout", "weight_decay", "groups_per_batch",
                  "epochs", "grad_clip", "max_pairs_per_group"):
            hs_val = _HAND_SET[k]
            w_val  = best_mlp[k]
            moved  = " *" if str(hs_val) != str(w_val) else "  "
            print(f"  {k:22}  {str(hs_val):>22}  {str(w_val):>22}{moved}")
        print(f"  {'ablation arm-A CV':22}  {'0.0370 (baseline)':>22}  "
              f"{'(winner measured above)':>22}")
        print("  (* = tuning moved this parameter)")

    suffix = f"_{args.suffix}" if args.suffix else ""
    out_path = outdir / f"tune_results_mlp{suffix}.json"
    result = {
        "best_cv_regret_mean": best.value,
        "best_trial": best.number,
        "mlp_params": best_mlp,
        "train_seeds": args.train_seeds,
        "n_folds": args.cv,
        "cv_oracle_caveat": (
            "fold-val oracle is sample-best (proxy), not true-oracle; "
            "CV regret valid for RELATIVE comparison only"
        ),
        "all_trials": [
            {"number": t.number, "value": t.value, "params": dict(t.params)}
            for t in study.trials
            if t.state == optuna.trial.TrialState.COMPLETE
        ],
    }
    out_path.write_text(json.dumps(result, indent=2))
    print(f"\nwrote {out_path}")
    print("\nNext step: restart-locking + single eval pass (deliberate, separate step).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
