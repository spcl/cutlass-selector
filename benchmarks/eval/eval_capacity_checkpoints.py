#!/usr/bin/env python3
"""Held-out eval for all capacity-sweep checkpoints (MLP + XGBoost).

Loads eval rows from features.parquet, or builds them from
~/autotuner/autotuner_bf16_eval.db via eval_plan tag bf16_eval,
scores each checkpoint under artifacts/analysis/capacity/, and writes eval_regret.csv +
updates metrics.json in place.

Designed for a single Slurm node with multiple GPUs: XGB runs on CPU workers,
MLP runs round-robin across GPUs.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sqlite3
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from repo_paths import CAPACITY, PAPER, SRC

DEFAULT_DB = Path.home() / "autotuner" / "autotuner_bf16_eval.db"
DEFAULT_EVAL_TAG = "bf16_eval"
DEFAULT_EVAL_MIN_CONFIGS = 8000
sys.path.insert(0, str(SRC / "model"))

from features import (  # noqa: E402
    CATEGORY_LEVELS,
    _layout_label,
    add_labels,
    featurize,
    load_eval,
)
from harness import (
    baselines,
    regime_breakdown,
    select_and_regret,
    summarize,
)  # noqa: E402


def _discover_runs(capacity_dir: Path) -> list[Path]:
    runs: list[Path] = []
    for d in sorted(capacity_dir.iterdir()):
        if not d.is_dir():
            continue
        if (d / "model_mlp.pt").exists() or (d / "model_A.ubj").exists():
            runs.append(d)
    return runs


def _needs_eval(run_dir: Path, force: bool) -> bool:
    if force:
        return True
    regret = run_dir / "eval_regret.csv"
    metrics = run_dir / "metrics.json"
    if not regret.exists() or not metrics.exists():
        return True
    try:
        meta = json.loads(metrics.read_text())
    except json.JSONDecodeError:
        return True
    ev = meta.get("eval")
    return not (isinstance(ev, dict) and "regret_mean" in ev)


def _resolve_db(db: Path | None) -> Path:
    path = DEFAULT_DB if db is None else db
    if not path.exists():
        raise FileNotFoundError(f"DB not found: {path}")
    return path


def _parquet_has_eval(path: Path) -> bool:
    if not path.is_file():
        return False
    df = pd.read_parquet(path, columns=["split"])
    return int((df["split"] == "eval").sum()) > 0


def _build_eval_features(
    db: Path, eval_tag: str, out: Path, min_configs: int
) -> Path:
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        ev = load_eval(conn, min_configs, eval_tag=eval_tag)
    finally:
        conn.close()
    if ev.empty:
        raise RuntimeError(
            f"no eval rows for tag={eval_tag!r} in {db} "
            f"(min_configs={min_configs}). "
            "bf16_eval runs are tagged in eval_plan, not tag IS NULL."
        )
    ev["layout"] = np.where(ev["layout"].notna(), ev["layout"], _layout_label(ev))
    ev = featurize(ev)
    ev = add_labels(ev)
    ev["split"] = "eval"
    out.parent.mkdir(parents=True, exist_ok=True)
    ev.to_parquet(out, index=False)
    print(
        f"built eval features: {out}  rows={len(ev):,}  "
        f"groups={ev['group_id'].nunique()}"
    )
    return out


def _ensure_eval_features(
    features_hint: Path,
    db: Path,
    eval_tag: str,
    cache_dir: Path | None,
    min_configs: int,
) -> Path:
    candidates = [
        features_hint,
        features_hint.parent / "eval_features.parquet",
    ]
    if cache_dir is not None:
        candidates.insert(0, cache_dir / "eval_features.parquet")

    for path in candidates:
        if _parquet_has_eval(path):
            df = pd.read_parquet(path, columns=["split"])
            n_eval = int((df["split"] == "eval").sum())
            print(f"using eval features: {path}  eval_rows={n_eval:,}")
            return path

    if features_hint.is_file():
        print(f"{features_hint} is train-only; building eval split separately")

    out = (cache_dir or features_hint.parent) / "eval_features.parquet"
    return _build_eval_features(db, eval_tag, out, min_configs)


def _prepare_eval_parquet(features: Path, shm_dir: Path | None) -> Path:
    """Write eval-only parquet (smaller) for worker processes."""
    df = pd.read_parquet(features)
    ev = df[df["split"] == "eval"].copy()
    if ev.empty:
        raise RuntimeError(f"no eval split in {features}")
    print(
        f"eval split: {len(ev):,} rows / {ev['group_id'].nunique()} groups",
        flush=True,
    )

    if shm_dir is not None:
        shm_dir.mkdir(parents=True, exist_ok=True)
        out = shm_dir / "eval_features.parquet"
        ev.to_parquet(out, index=False)
        print(f"cached eval rows → {out}", flush=True)
        return out
    return features


def _load_eval(ev_path: Path) -> pd.DataFrame:
    df = pd.read_parquet(ev_path)
    if "split" in df.columns:
        df = df[df["split"] == "eval"].copy()
    return df


def _apply_categoricals(df: pd.DataFrame, cat_cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cat_cols:
        out[c] = pd.Categorical(out[c].astype(str), categories=CATEGORY_LEVELS[c])
    return out


def _update_metrics(
    run_dir: Path, eval_metrics: dict, perg: pd.DataFrame, ev: pd.DataFrame
) -> None:
    metrics_path = run_dir / "metrics.json"
    meta = json.loads(metrics_path.read_text())
    meta["eval"] = eval_metrics
    meta["baselines"] = baselines(ev)
    meta["by_regime"] = regime_breakdown(perg)
    metrics_path.write_text(json.dumps(meta, indent=2))
    perg.to_csv(run_dir / "eval_regret.csv", index=False)


def eval_xgb_run(run_dir: str, ev_path: str, force: bool) -> dict:
    run = Path(run_dir)
    t0 = time.perf_counter()
    if not _needs_eval(run, force):
        meta = json.loads((run / "metrics.json").read_text())
        return {
            "run_dir": run.name,
            "family": "xgb",
            "status": "skipped",
            "regret_mean": meta.get("eval", {}).get("regret_mean"),
            "seconds": 0.0,
        }

    import xgboost as xgb

    meta = json.loads((run / "metrics.json").read_text())
    feat_cols = meta["numeric_features"] + meta["categorical_features"]
    ev = _load_eval(Path(ev_path))
    ev = _apply_categoricals(ev, meta["categorical_features"])

    model = xgb.XGBRegressor()
    model.load_model(str(run / "model_A.ubj"))
    scores = model.predict(ev[feat_cols])
    perg = select_and_regret(ev, scores)
    label = f"XGB-{meta.get('loss', 'mse')}"
    eval_metrics = summarize(perg, label)
    _update_metrics(run, eval_metrics, perg, ev)

    dt = time.perf_counter() - t0
    print(
        f"[xgb] {run.name}  regret={eval_metrics['regret_mean']:.4f}  ({dt:.1f}s)",
        flush=True,
    )
    return {
        "run_dir": run.name,
        "family": "xgb",
        "status": "ok",
        "regret_mean": eval_metrics["regret_mean"],
        "seconds": dt,
    }


def eval_mlp_gpu_worker(
    gpu_id: int, run_dirs: list[str], ev_path: str, force: bool
) -> list[dict]:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    import torch
    from sklearn.preprocessing import StandardScaler
    from train_mlp import MLP, build_X, predict

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    ev = _load_eval(Path(ev_path))
    results: list[dict] = []

    for run_dir in run_dirs:
        run = Path(run_dir)
        t0 = time.perf_counter()
        if not _needs_eval(run, force):
            meta = json.loads((run / "metrics.json").read_text())
            results.append(
                {
                    "run_dir": run.name,
                    "family": "mlp",
                    "status": "skipped",
                    "regret_mean": meta.get("eval", {}).get("regret_mean"),
                    "seconds": 0.0,
                }
            )
            continue

        meta = json.loads((run / "metrics.json").read_text())
        num_cols = meta["numeric_features"]
        cat_cols = meta["categorical_features"]
        ckpt = torch.load(run / "model_mlp.pt", map_location="cpu", weights_only=False)

        scaler = StandardScaler()
        scaler.mean_ = np.asarray(ckpt["scaler_mean"], dtype=np.float64)
        scaler.scale_ = np.asarray(ckpt["scaler_scale"], dtype=np.float64)
        n_in = int(ckpt["n_in"])
        hidden = tuple(ckpt["hidden"])
        dropout = float(ckpt["dropout"])

        X_eval, _ = build_X(ev, num_cols, cat_cols, scaler=scaler)
        model = MLP(n_in, hidden, dropout)
        model.load_state_dict(ckpt["model_state"])
        model.to(device)
        scores = predict(model, X_eval, device)
        perg = select_and_regret(ev, scores)
        label = f"MLP-{meta.get('loss', 'mse')}"
        eval_metrics = summarize(perg, label)
        _update_metrics(run, eval_metrics, perg, ev)

        dt = time.perf_counter() - t0
        print(
            f"[mlp gpu={gpu_id}] {run.name}  regret={eval_metrics['regret_mean']:.4f}  ({dt:.1f}s)",
            flush=True,
        )
        results.append(
            {
                "run_dir": run.name,
                "family": "mlp",
                "status": "ok",
                "regret_mean": eval_metrics["regret_mean"],
                "seconds": dt,
            }
        )
    return results


def _chunk_round_robin(items: list[Path], n: int) -> list[list[Path]]:
    buckets: list[list[Path]] = [[] for _ in range(n)]
    for i, item in enumerate(items):
        buckets[i % n].append(item)
    return buckets


def _write_summary(capacity_dir: Path, rows: list[dict]) -> Path:
    """Write summary for all runs with eval metrics (not just this job's workers)."""
    out = capacity_dir / "capacity_eval_summary.csv"
    by_run = {r["run_dir"]: r for r in rows if r.get("run_dir")}
    for run_dir in _discover_runs(capacity_dir):
        if run_dir.name in by_run:
            continue
        if not _needs_eval(run_dir, force=False):
            meta = json.loads((run_dir / "metrics.json").read_text())
            by_run[run_dir.name] = {
                "run_dir": run_dir.name,
                "family": "mlp" if (run_dir / "model_mlp.pt").exists() else "xgb",
                "status": "skipped",
                "regret_mean": meta.get("eval", {}).get("regret_mean"),
                "seconds": 0.0,
            }
    summary = pd.DataFrame(list(by_run.values())).sort_values(["family", "run_dir"])
    summary.to_csv(out, index=False)
    print(f"wrote {out}  ({len(summary)} rows)")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Eval all capacity checkpoints on held-out groups."
    )
    ap.add_argument(
        "--capacity-dir", type=Path, default=CAPACITY
    )
    ap.add_argument(
        "--features",
        type=Path,
        default=PAPER / "features.parquet",
    )
    ap.add_argument(
        "--db",
        type=Path,
        default=None,
        help="autotuner sqlite (default: ~/autotuner/autotuner_bf16_eval.db)",
    )
    ap.add_argument(
        "--eval-tag",
        default=DEFAULT_EVAL_TAG,
        help="eval_plan tag in the bf16 eval DB (default: bf16_eval)",
    )
    ap.add_argument(
        "--eval-min-configs",
        type=int,
        default=DEFAULT_EVAL_MIN_CONFIGS,
        help="min measured configs per (shape,layout) group",
    )
    ap.add_argument(
        "--shm-dir",
        type=Path,
        default=None,
        help="node-local cache (e.g. /dev/shm/…); not $SCRATCH",
    )
    ap.add_argument("--num-gpus", type=int, default=4)
    ap.add_argument("--xgb-workers", type=int, default=64)
    ap.add_argument(
        "--force", action="store_true", help="re-eval even if eval_regret.csv exists"
    )
    ap.add_argument("--only", choices=["mlp", "xgb", "all"], default="all")
    args = ap.parse_args()

    capacity_dir = args.capacity_dir.resolve()
    runs = _discover_runs(capacity_dir)
    if not runs:
        print(f"no checkpoints under {capacity_dir}", file=sys.stderr)
        return 1

    mlp_runs = [r for r in runs if (r / "model_mlp.pt").exists()]
    xgb_runs = [r for r in runs if (r / "model_A.ubj").exists()]
    print(f"discovered {len(runs)} runs: {len(mlp_runs)} MLP, {len(xgb_runs)} XGB")

    db = _resolve_db(args.db)
    features = _ensure_eval_features(
        args.features.resolve(),
        db,
        args.eval_tag,
        args.shm_dir,
        args.eval_min_configs,
    )
    ev_path = _prepare_eval_parquet(features, args.shm_dir)

    pending_mlp = [r for r in mlp_runs if _needs_eval(r, args.force)]
    pending_xgb = [r for r in xgb_runs if _needs_eval(r, args.force)]
    print(
        f"pending: {len(pending_mlp)} MLP, {len(pending_xgb)} XGB  (force={args.force})"
    )

    all_results: list[dict] = []
    t_all = time.perf_counter()

    if args.only in ("xgb", "all") and xgb_runs:
        workers = max(1, min(args.xgb_workers, len(pending_xgb) or 1))
        print(
            f"\n== XGB eval ({len(pending_xgb)} pending, {workers} workers) ==",
            flush=True,
        )
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = {
                ex.submit(eval_xgb_run, str(r), str(ev_path), args.force): r.name
                for r in xgb_runs
            }
            for fut in as_completed(futs):
                all_results.append(fut.result())

    if args.only in ("mlp", "all") and mlp_runs:
        ngpu = max(1, args.num_gpus)
        buckets = _chunk_round_robin(mlp_runs, ngpu)
        print(f"\n== MLP eval ({len(pending_mlp)} pending, {ngpu} GPUs) ==", flush=True)
        mlp_ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=ngpu, mp_context=mlp_ctx) as ex:
            futs = [
                ex.submit(
                    eval_mlp_gpu_worker,
                    gpu_id,
                    [str(r) for r in bucket],
                    str(ev_path),
                    args.force,
                )
                for gpu_id, bucket in enumerate(buckets)
                if bucket
            ]
            for fut in as_completed(futs):
                all_results.extend(fut.result())

    _write_summary(capacity_dir, all_results)
    elapsed = time.perf_counter() - t_all
    ok = sum(1 for r in all_results if r["status"] == "ok")
    skipped = sum(1 for r in all_results if r["status"] == "skipped")
    print(
        f"\nDone: {ok} evaluated, {skipped} skipped, {len(all_results)} total in {elapsed:.1f}s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

