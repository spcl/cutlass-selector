#!/usr/bin/env python3
"""Train a single epilogue-fusion transfer study run."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

from repo_paths import BENCHMARKS, SRC

sys.path.insert(0, str(SRC / "model"))
from features_fusion import CATEGORICAL_FEATURES, CATEGORY_LEVELS, NUMERIC_FEATURES  # noqa: E402

sys.path.insert(0, str(SRC))
sys.path.insert(0, str(BENCHMARKS))
from transfer_fusion.study_common import (  # noqa: E402
    MLP_EPOCHS_PRETRAIN,
    MLP_LR_PRETRAIN,
    PRETRAIN_MLP,
    PRETRAIN_XGB,
    STRUCTURAL_CATEGORICAL,
    STRUCTURAL_NUMERIC,
    XGB_EXTRA_TREES,
    RunSpec,
    features_path,
    load_scaler_policy,
    load_shapes,
    subset_path,
)


def load_feature_manifest(path: Path) -> dict:
    """Load a feature manifest; tolerate a duplicated trailing JSON object."""
    text = path.read_text()
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        if "Extra data" not in str(exc):
            raise
        obj, end = json.JSONDecoder().raw_decode(text)
        print(f"warning: {path} contains extra JSON after offset {end}; using first object")
        return obj


def feature_columns(feature_set: str) -> tuple[list[str], list[str]]:
    if feature_set == "full":
        return list(NUMERIC_FEATURES), list(CATEGORICAL_FEATURES)
    if feature_set == "structural":
        return list(STRUCTURAL_NUMERIC), list(STRUCTURAL_CATEGORICAL)
    raise ValueError(feature_set)


def prepare_run_parquet(
    full_parquet: Path,
    shapes: list[tuple[int, int, int]],
    feature_set: str,
    out_parquet: Path,
) -> Path:
    shape_set = set(shapes)
    df = pd.read_parquet(full_parquet)
    mask = [tuple(x) in shape_set for x in df[["M", "N", "K"]].to_numpy()]
    sub = df[mask].copy()
    if sub.empty:
        raise SystemExit(f"no rows after shape filter for {out_parquet}")

    num_cols, cat_cols = feature_columns(feature_set)
    meta_cols = [
        "M", "N", "K", "layout_a", "layout_b", "layout", "group_id",
        "mean_tflops", "std_tflops", "group_best_tflops", "y_norm",
        "relevance_grade", "rank_in_group", "split",
    ]
    keep = list(dict.fromkeys(num_cols + cat_cols + meta_cols))
    keep = [c for c in keep if c in sub.columns]
    sub = sub[keep]
    for c in cat_cols:
        vals = sub[c].astype(str)
        levels = list(CATEGORY_LEVELS[c])
        extra = sorted(set(vals.unique()) - set(levels))
        if extra:
            print(f"warning: extending {c} categories with {extra}")
            levels = levels + extra
        sub[c] = pd.Categorical(vals, categories=levels)

    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    sub.to_parquet(out_parquet, index=False)
    manifest_path = full_parquet.with_suffix(".manifest.json")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing feature manifest: {manifest_path}")
    base_manifest = load_feature_manifest(manifest_path)
    manifest = {
        **base_manifest,
        "numeric_features": num_cols,
        "categorical_features": cat_cols,
        "feature_set": feature_set,
        "n_rows": len(sub),
        "n_shapes": len(shapes),
        "n_groups": int(sub["group_id"].nunique()) if "group_id" in sub.columns else None,
    }
    manifest_path = Path(str(out_parquet).rsplit(".", 1)[0] + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return out_parquet


def train_run(spec: RunSpec, python: str = sys.executable, force: bool = False) -> None:
    out = spec.out_dir
    done = out / ("model_mlp.pt2" if spec.family == "mlp" else "model_A.ubj")
    if done.is_file() and not force:
        print(f"skip existing {spec.run_id}")
        return

    shapes = load_shapes(subset_path(spec.dtype, spec.seed, spec.fraction))
    pq = prepare_run_parquet(
        features_path(spec.dtype),
        shapes,
        spec.features,
        out / "features.parquet",
    )

    if spec.family == "mlp":
        train_py = SRC / "model" / "train_mlp.py"
        cmd = [
            python, str(train_py),
            "--features", str(pq),
            "--loss", "mse",
            "--skip-eval",
            "--outdir", str(out),
            "--init-checkpoint", str(PRETRAIN_MLP[spec.features]),
            "--scaler-policy", load_scaler_policy(),
            "--epochs", str(MLP_EPOCHS_PRETRAIN),
            "--lr", str(MLP_LR_PRETRAIN),
        ]
        subprocess.check_call(cmd)
        artifact = out / "model_mlp.pt2"
    else:
        train_py = SRC / "model" / "train_xgb.py"
        cmd = [
            python, str(train_py),
            "--features", str(pq),
            "--loss", "mse",
            "--skip-eval",
            "--outdir", str(out),
            "--init-model", str(PRETRAIN_XGB[spec.features]),
            "--n-estimators", str(XGB_EXTRA_TREES),
        ]
        subprocess.check_call(cmd)
        artifact = out / "model_A.ubj"

    meta_path = out / "metrics.json"
    meta = json.loads(meta_path.read_text()) if meta_path.is_file() else {}
    meta["study"] = {
        "run_id": spec.run_id,
        "dtype": spec.dtype,
        "fraction": spec.fraction,
        "seed": spec.seed,
        "family": spec.family,
        "features": spec.features,
        "init": spec.init,
        "n_shapes": len(shapes),
        "n_rows": int(pd.read_parquet(pq).shape[0]),
        "artifact": str(artifact),
    }
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    print(f"wrote {artifact}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--python", default=sys.executable)
    args = ap.parse_args()

    from transfer_fusion.study_common import iter_run_specs

    for spec in iter_run_specs():
        if spec.run_id == args.run_id:
            train_run(spec, python=args.python, force=args.force)
            return 0
    raise SystemExit(f"unknown run_id {args.run_id}")


if __name__ == "__main__":
    raise SystemExit(main())
