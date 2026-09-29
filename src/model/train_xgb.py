#!/usr/bin/env python3
"""
XGBoost ranker for CUTLASS SM90 GEMM kernel selection.

The model ranks the configurations of one (M, N, K, layout) problem; it is judged by
selection quality, not prediction error (harness.py):

  - regret = 1 - chosen.mean_tflops / best, per (shape, layout) evaluation group
  - top-1 / top-5 recall, and "within 5% of the best"
  - p95 / max tail of the regret distribution
  - a regret-by-regime breakdown (square / tall / wide / skinny-K / large)

Evaluation uses the groups with split == 'eval', which were measured exhaustively, so
their group_best_tflops is the true optimum. --val-frac holds out a shape-grouped
validation split from training for the learning curve; --cv reports shape-grouped
cross-validation. Neither touches an evaluation shape.

Inputs : features.parquet (+ .manifest.json) from features.py
Outputs: model_A.ubj, metrics.json, eval_regret.csv, training_history_xgb.csv (with --val-frac)

Usage:
    python src/model/train_xgb.py --features artifacts/analysis/paper/features.parquet

--loss: ndcg (default, rank:ndcg) | pairwise (rank:pairwise) | mse (reg:squarederror, pointwise on y_norm)
--feature-set: full (hardware-aware, default) | structural (config/problem only)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from feature_manifest import category_levels_for
from features import FEATURE_SETS, feature_columns
from harness import (
    NDCG_KS,
    base_shape_id,
    baselines,
    grouped_train_val_split,
    make_relevance,
    ndcg_per_group,
    regime_breakdown,
    select_and_regret,
    summarize,
)

REL_TOP_BANDS = 2  # tuned: narrower top-band focus (Optuna trial #202, CV regret=0.0219)

DEFAULT_PARAMS = dict(
    eta=0.14994338035621416,
    max_depth=7,
    min_child_weight=2.129510939594175,
    subsample=0.8006253365599527,
    colsample_bytree=0.905985078442277,
    reg_lambda=0.0329689343120563,
    reg_alpha=0.08879582214664276,
    gamma=0.019394121795509252,
)
DEFAULT_N_ESTIMATORS = 916


# ----------------------------------------------------------------------
# Model (xgboost imported lazily)
# ----------------------------------------------------------------------
LOSS_OBJECTIVES = {"ndcg": "rank:ndcg", "pairwise": "rank:pairwise"}


def fit_ranker(X, y_rel, qid, params, n_estimators, seed, device="cpu", objective="rank:ndcg",
               eval_set=None, eval_qid=None, callbacks=None, eval_metric="ndcg@10",
               init_model: Path | None = None):
    """Fit an XGBRanker; X must be sorted by qid. init_model continues boosting from a saved model."""
    import xgboost as xgb

    model = xgb.XGBRanker(
        objective=objective,
        tree_method="hist",
        device=device,
        enable_categorical=True,
        n_estimators=n_estimators,
        random_state=seed,
        eval_metric=eval_metric,
        callbacks=callbacks,
        **params,
    )
    # X must be sorted by qid (ascending, contiguous); so must every eval_set entry.
    fit_kwargs = {"xgb_model": str(init_model)} if init_model is not None else {}
    model.fit(X, y_rel, qid=qid, eval_set=eval_set, eval_qid=eval_qid, verbose=False, **fit_kwargs)
    return model


def fit_regressor(X, y, params, n_estimators, seed, device="cpu",
                  eval_set=None, callbacks=None, eval_metric="rmse",
                  init_model: Path | None = None):
    """mse loss arm: pointwise regression on y_norm (not a ranker)."""
    import xgboost as xgb

    model = xgb.XGBRegressor(
        objective="reg:squarederror",
        tree_method="hist",
        device=device,
        enable_categorical=True,
        n_estimators=n_estimators,
        random_state=seed,
        eval_metric=eval_metric,
        callbacks=callbacks,
        **params,
    )
    fit_kwargs = {"xgb_model": str(init_model)} if init_model is not None else {}
    model.fit(X, y, eval_set=eval_set, verbose=False, **fit_kwargs)
    return model


def _sorted_by_group(df: pd.DataFrame):
    return df.sort_values("group_id", kind="stable")


# ----------------------------------------------------------------------
# Per-round training history
# ----------------------------------------------------------------------
def _make_history_callback(val_df: pd.DataFrame, val_X, ndcg_ks, every: int):
    """Records harness NDCG on the validation split every `every` boosting rounds.

    XGBoost's own ndcg@k uses exponential gain over the integer relevance labels, so it
    is not comparable with the MLP's. Recomputing it here through harness.ndcg_per_group
    keeps one NDCG definition across both model families and both feature sets, which is
    what makes the §2 objective figure meaningful.
    """
    import xgboost as xgb

    dval = xgb.DMatrix(val_X, enable_categorical=True)

    class _History(xgb.callback.TrainingCallback):
        def __init__(self):
            self.rows: dict[int, dict] = {}
            self.t0 = time.perf_counter()

        def after_iteration(self, model, epoch: int, evals_log) -> bool:
            if every <= 0:
                return False
            if (epoch + 1) % every and (epoch + 1) != 1:
                return False
            pred = model.predict(dval, iteration_range=(0, epoch + 1))
            nd = ndcg_per_group(val_df, pred, ndcg_ks).mean()
            row = {f"val_{k}": float(v) for k, v in nd.items()}
            row["elapsed_s"] = time.perf_counter() - self.t0
            self.rows[epoch] = row
            return False

    return _History()


def _history_frame(evals_result: dict, cb_rows: dict, meta: dict) -> pd.DataFrame:
    """evals_result (XGBoost's own train/val metric per round) + callback NDCG -> tidy rows."""
    train_log = evals_result.get("validation_0", {})
    val_log = evals_result.get("validation_1", {})
    metric = next(iter(train_log), None)
    if metric is None:
        return pd.DataFrame()
    rows = []
    for i, tr_v in enumerate(train_log[metric]):
        row = dict(meta)
        row["round"] = i + 1
        # Name the metric in a column rather than in the column name: the objectives
        # report different builtin metrics (rmse vs ndcg@10), and a single tidy
        # training_history_xgb.csv has to stack all of them.
        row["metric"] = metric
        row["train_metric"] = float(tr_v)
        row["val_metric"] = float(val_log[metric][i]) if metric in val_log else None
        row.update(cb_rows.get(i, {}))
        rows.append(row)
    return pd.DataFrame(rows)


def cv_report(train: pd.DataFrame, feat_cols, grade_max, params, n_estimators, n_splits, seed, rel_top_bands):
    """Shape-grouped k-fold cross-validation on the training split; returns per-fold and mean selection metrics."""
    from sklearn.model_selection import GroupKFold

    groups = base_shape_id(train)
    gkf = GroupKFold(n_splits=n_splits)
    print(f"\n== Shape-grouped {n_splits}-fold CV (validation = held-out training shapes) ==")
    fold_metrics = []
    for i, (tr, va) in enumerate(gkf.split(train, groups=groups), 1):
        a = _sorted_by_group(train.iloc[tr])
        b = train.iloc[va]
        m = fit_ranker(
            a[feat_cols],
            make_relevance(a["relevance_grade"].to_numpy(), grade_max, rel_top_bands),
            a["group_id"].to_numpy(),
            params,
            n_estimators,
            seed,
        )
        perg = select_and_regret(b, m.predict(b[feat_cols]))
        s = summarize(perg, f"fold {i}")
        fold_metrics.append(s)
    mean = {
        k: float(np.mean([f[k] for f in fold_metrics]))
        for k in ("regret_mean", "regret_p95", "top1", "top5", "within5pct")
    }
    print(
        f"  {'CV mean':18}  regret={mean['regret_mean']:.3f} p95={mean['regret_p95']:.3f} "
        f"top1={mean['top1']:.2f} top5={mean['top5']:.2f} <5%={mean['within5pct']:.2f}"
    )
    return {"folds": fold_metrics, "mean": mean}


# ----------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True, help="features.parquet from features.py")
    ap.add_argument("--cv", type=int, default=0, help="shape-grouped CV folds (0 = skip)")
    ap.add_argument("--no-bad", action="store_true", help="ablate the stratified_bad training slice")
    ap.add_argument("--loss", choices=["ndcg", "pairwise", "mse"], default="ndcg",
                    help="training objective ablation")
    ap.add_argument("--feature-set", choices=list(FEATURE_SETS), default="full",
                    help="full = hardware-aware; structural = config/problem only")
    ap.add_argument("--val-frac", type=float, default=0.0,
                    help="shape-grouped validation fraction for the training history "
                         "(0 = train on all shapes, no history)")
    ap.add_argument("--val-seed", type=int, default=42, help="seed for the validation split")
    ap.add_argument("--history-every", type=int, default=1,
                    help="record validation NDCG every N boosting rounds")
    ap.add_argument("--no-eval", "--skip-eval", dest="no_eval", action="store_true",
                    help="skip scoring the held-out eval groups. Their rows are millions "
                         "of the parquet and dominate peak memory; validation is unaffected.")
    ap.add_argument("--train-metric-rows", type=int, default=0,
                    help="subsample the training rows used for the per-round TRAIN metric "
                         "(0 = all). The extra eval_set entry is a second full copy of the "
                         "design matrix; the curve does not need every row.")
    ap.add_argument("--n-estimators", type=int, default=DEFAULT_N_ESTIMATORS)
    ap.add_argument("--max-depth", type=int, default=None,
                    help="override DEFAULT_PARAMS['max_depth']; the capacity knob for "
                         "the feature-vs-capacity study")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--outdir", default=None, help="defaults to <features file's directory>/xgb")
    ap.add_argument("--init-model", type=Path, default=None,
                    help="fine-tune: continue boosting from an existing model_A.ubj "
                         "(--n-estimators = extra rounds)")
    args = ap.parse_args()

    fpath = Path(args.features)
    suffix = "" if args.feature_set == "full" else f"_{args.feature_set}"
    outdir = Path(args.outdir) if args.outdir else fpath.parent / f"xgb_{args.loss}{suffix}"
    outdir.mkdir(parents=True, exist_ok=True)

    params = dict(DEFAULT_PARAMS)
    if args.max_depth is not None:
        params["max_depth"] = args.max_depth
    n_estimators = args.n_estimators

    manifest = json.loads(Path(str(fpath).rsplit(".", 1)[0] + ".manifest.json").read_text())
    grade_max = int(manifest["hw_constants"]["GRADE_MAX"])
    num_cols, cat_cols = feature_columns(manifest, args.feature_set)
    feat_cols = num_cols + cat_cols
    if args.no_eval:
        # Same reasoning as train_mlp.py: push the split filter and column projection
        # into the read. The eval rows and the provenance columns (notably the long
        # `name` strings) are dead weight and this is the run's peak memory.
        needed = list(dict.fromkeys(
            feat_cols + ["split", "group_id", "M", "N", "K", "mean_tflops",
                         "group_best_tflops", "y_norm", "rank_in_group", "relevance_grade",
                         "sampling_method"]
        ))
        df = pd.read_parquet(fpath, columns=needed, filters=[("split", "==", "train")])
    else:
        df = pd.read_parquet(fpath)
    print(f"feature set: {args.feature_set} "
          f"({len(num_cols)} numeric + {len(cat_cols)} categorical)")

    # Pinned category levels so train/eval/inference codes align under
    # enable_categorical regardless of which values are present.
    category_levels = category_levels_for(manifest)
    for c in cat_cols:
        df[c] = pd.Categorical(df[c].astype(str), categories=category_levels[c])

    train = df[df["split"] == "train"].copy()
    ev = df[df["split"] == "eval"].copy() if not args.no_eval else df.iloc[0:0].copy()
    del df
    if args.no_bad and "sampling_method" in train:
        before = len(train)
        train = train[train["sampling_method"] != "stratified_bad"].copy()
        print(f"ablation: dropped stratified_bad ({before - len(train):,} rows)")
    print(
        f"train={len(train):,} rows / {train['group_id'].nunique():,} groups   "
        f"eval={len(ev):,} rows / {ev['group_id'].nunique()} groups"
    )

    init_model = args.init_model
    if init_model is not None and not init_model.is_file():
        raise FileNotFoundError(init_model)
    if init_model is not None:
        print(f"fine-tune init: {init_model}  (+{n_estimators} trees)")

    metrics = {
        "n_estimators": n_estimators,
        "params": params,
        "no_bad": bool(args.no_bad),
        "loss": args.loss,
        "rel_top_bands": REL_TOP_BANDS,
        "feature_set": args.feature_set,
        "numeric_features": num_cols,
        "categorical_features": cat_cols,
        "seed": args.seed,
        "init_model": str(init_model) if init_model is not None else None,
    }

    if args.cv:
        metrics["cv"] = cv_report(train, feat_cols, grade_max, params, n_estimators, args.cv, args.seed, REL_TOP_BANDS)

    # Final fit. With --val-frac > 0 a shape-grouped slice is held out so the run has a
    # validation curve; the same model is then scored on the untouched eval groups, so
    # the reported model is exactly the one whose training history we plot.
    if args.val_frac > 0:
        fit_rows, val_rows = grouped_train_val_split(train, args.val_frac, args.val_seed)
        val_rows = _sorted_by_group(val_rows)
        print(
            f"validation split: fit={len(fit_rows):,} rows / "
            f"{fit_rows['group_id'].nunique():,} groups / "
            f"{pd.unique(base_shape_id(fit_rows)).size} shapes   "
            f"val={len(val_rows):,} rows / {val_rows['group_id'].nunique():,} groups / "
            f"{pd.unique(base_shape_id(val_rows)).size} shapes"
        )
    else:
        fit_rows, val_rows = train, None

    a = _sorted_by_group(fit_rows)
    hist_cb = (
        _make_history_callback(val_rows, val_rows[feat_cols], NDCG_KS, args.history_every)
        if val_rows is not None
        else None
    )
    callbacks = [hist_cb] if hist_cb is not None else None
    t0 = time.perf_counter()
    # The train-metric eval_set entry is a second full copy of the design matrix.
    # The curve does not need every row, and on a memory-bound box this copy is the
    # difference between running and being OOM-killed.
    a_tm = a
    if args.train_metric_rows and len(a) > args.train_metric_rows:
        idx = np.sort(np.random.default_rng(args.seed).choice(
            len(a), args.train_metric_rows, replace=False))
        a_tm = a.iloc[idx]
        print(f"train-metric eval_set subsampled to {len(a_tm):,} of {len(a):,} rows")

    if args.loss == "mse":
        eval_set = [(a_tm[feat_cols], a_tm["y_norm"].to_numpy())]
        if val_rows is not None:
            eval_set.append((val_rows[feat_cols], val_rows["y_norm"].to_numpy()))
        model = fit_regressor(
            a[feat_cols], a["y_norm"].to_numpy(), params, n_estimators, args.seed,
            eval_set=eval_set, callbacks=callbacks, init_model=init_model,
        )
    else:
        y_fit = make_relevance(a["relevance_grade"].to_numpy(), grade_max, REL_TOP_BANDS)
        eval_set = [(a_tm[feat_cols],
                     make_relevance(a_tm["relevance_grade"].to_numpy(), grade_max, REL_TOP_BANDS))]
        eval_qid = [a_tm["group_id"].to_numpy()]
        if val_rows is not None:
            eval_set.append((
                val_rows[feat_cols],
                make_relevance(val_rows["relevance_grade"].to_numpy(), grade_max, REL_TOP_BANDS),
            ))
            eval_qid.append(val_rows["group_id"].to_numpy())
        model = fit_ranker(
            a[feat_cols], y_fit, a["group_id"].to_numpy(),
            params, n_estimators, args.seed,
            objective=LOSS_OBJECTIVES[args.loss],
            eval_set=eval_set, eval_qid=eval_qid, callbacks=callbacks, init_model=init_model,
        )
    train_seconds = time.perf_counter() - t0
    metrics["train_seconds"] = train_seconds
    metrics["val_frac"] = args.val_frac
    metrics["val_seed"] = args.val_seed
    print(f"trained {n_estimators} rounds in {train_seconds:.1f}s")

    history = pd.DataFrame()
    if hist_cb is not None:
        meta = {
            "model": "xgboost",
            "objective": args.loss,
            "feature_set": args.feature_set,
            "seed": args.seed,
        }
        history = _history_frame(getattr(model, "evals_result_", {}), hist_cb.rows, meta)
        if not history.empty:
            history.to_csv(outdir / "training_history_xgb.csv", index=False)
            best_col = "val_ndcg@10"
            if best_col in history:
                best = history.loc[history[best_col].idxmax()]
                metrics["best_val_ndcg@10"] = float(best[best_col])
                metrics["best_round"] = int(best["round"])
                print(f"best {best_col}={best[best_col]:.4f} at round {int(best['round'])}")

    perg = None
    if not ev.empty:
        print(f"\n== Eval on {ev['group_id'].nunique()} true-oracle groups ==")
        perg = select_and_regret(ev, model.predict(ev[feat_cols]))
        metrics["eval"] = summarize(perg, "XGBoost")
        metrics["baselines"] = baselines(ev)
        metrics["by_regime"] = regime_breakdown(perg)
    else:
        metrics["eval"] = {"label": "XGBoost", "groups": 0, "skipped": True}
        print("\n== No true-oracle eval groups scored"
              + (" (--no-eval)" if args.no_eval else "") + " ==")

    # Total leaf count = the booster's free real-valued parameters. Gives the
    # feature-vs-capacity study an x-axis comparable with the MLP's weight count.
    metrics["n_leaves"] = sum(d.count("leaf=") for d in model.get_booster().get_dump())
    metrics["max_depth"] = params["max_depth"]
    print(f"model size: {metrics['n_leaves']:,} leaves over {n_estimators} trees "
          f"(max_depth={params['max_depth']})")

    model.save_model(str(outdir / "model_A.ubj"))
    if not history.empty:
        print(f"wrote {outdir / 'training_history_xgb.csv'} ({len(history)} rounds)")
    if perg is not None:
        perg.to_csv(outdir / "eval_regret.csv", index=False)
    (outdir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    written = "model_A.ubj" + (", eval_regret.csv" if perg is not None else "") + ", metrics.json"
    print(f"\nwrote {written} in {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
