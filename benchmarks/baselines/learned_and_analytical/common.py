"""Shared utilities for linear and analytical baseline experiments."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from repo_paths import BENCHMARKS, PAPER, SRC

BASELINES = PAPER / "baselines"
DEFAULT_FEATURES = PAPER / "features.parquet"

sys.path.insert(0, str(SRC / "model"))
sys.path.insert(0, str(BENCHMARKS / "transfer_dtype"))

from feature_manifest import category_levels_for  # noqa: E402
from features import DRAM_BW, PEAK_TC_BF16  # noqa: E402
from harness import base_shape_id, baselines, regime_breakdown, select_and_regret  # noqa: E402
from study_common import STRUCTURAL_CATEGORICAL, STRUCTURAL_NUMERIC  # noqa: E402

VAL_FRAC = 0.15
VAL_SEED = 42
MIN_GROUP_SAMPLES = 3
ROOFLINE_RIDGE_AI = PEAK_TC_BF16 / DRAM_BW


def load_manifest(features_path: Path) -> dict:
    manifest_path = Path(str(features_path).rsplit(".", 1)[0] + ".manifest.json")
    return json.loads(manifest_path.read_text())


def feature_sets(manifest: dict) -> dict[str, tuple[list[str], list[str]]]:
    full_num = manifest["numeric_features"]
    full_cat = manifest["categorical_features"]
    return {
        "full": (full_num, full_cat),
        "structural": (list(STRUCTURAL_NUMERIC), list(STRUCTURAL_CATEGORICAL)),
    }


def feature_column_names(num_cols: list[str], cat_cols: list[str], category_levels: dict) -> list[str]:
    names = list(num_cols)
    for col in cat_cols:
        for lvl in category_levels[col]:
            names.append(f"{col}={lvl}")
    return names


def split_counts(features_path: Path) -> dict[str, int]:
    df = pd.read_parquet(features_path, columns=["split"])
    return {str(k): int(v) for k, v in df["split"].value_counts().items()}


def resolve_features_path(hint: Path | None = None) -> Path:
    """Pick a parquet that has both train and eval rows (paper training set + 68 groups)."""
    candidates: list[Path] = []
    if hint is not None:
        candidates.append(Path(hint))
    for p in (
        SRC / "model" / "artifacts" / "paper" / "features.parquet",
        PAPER / "features.parquet",
        DEFAULT_FEATURES,
    ):
        if p not in candidates:
            candidates.append(p)

    found: list[tuple[Path, dict[str, int]]] = []
    for p in candidates:
        if not p.is_file():
            continue
        counts = split_counts(p)
        found.append((p, counts))
        n_train = counts.get("train", 0)
        n_eval = counts.get("eval", 0)
        if n_train > 0 and n_eval > 0:
            print(f"features: {p}  splits={counts}")
            return p

    if found:
        lines = [f"  {p}: {counts}" for p, counts in found]
        raise FileNotFoundError(
            "No features.parquet with both train and eval splits.\n"
            + "\n".join(lines)
            + "\nBuild with src/model/features.py (see artifacts/analysis/paper/features.manifest.json) "
            "or set FEATURES=artifacts/analysis/paper/features.parquet"
        )
    raise FileNotFoundError(
        f"No features.parquet found (hint={hint}). "
        "Expected artifacts/analysis/paper/features.parquet on the cluster."
    )


def load_splits(features_path: Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame, dict, Path]:
    path = resolve_features_path(features_path)
    manifest = load_manifest(path)
    category_levels = category_levels_for(manifest)
    df = pd.read_parquet(path)
    for col in manifest["categorical_features"]:
        df[col] = pd.Categorical(df[col].astype(str), categories=category_levels[col])
    train = df[df["split"] == "train"].copy()
    ev = df[df["split"] == "eval"].copy()
    if train.empty:
        counts = split_counts(path)
        raise ValueError(
            f"{path} has no train rows (splits={counts}). "
            "Baselines require the full paper features matrix, not an eval-only cache."
        )
    if ev.empty:
        raise ValueError(f"{path} has no eval rows — need the 68 held-out groups.")
    return train, ev, manifest, path


def train_val_split(train: pd.DataFrame, val_frac: float = VAL_FRAC, seed: int = VAL_SEED) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hold out entire base (M,N,K) shapes — all four layouts stay together."""
    if train.empty:
        raise ValueError("train_val_split: train dataframe is empty")
    shape_ids = base_shape_id(train)
    unique = np.unique(shape_ids)
    if len(unique) < 2:
        raise ValueError(f"train_val_split: need >=2 base shapes, got {len(unique)}")
    rng = np.random.default_rng(seed)
    n_val = max(1, int(round(len(unique) * val_frac)))
    n_val = min(n_val, len(unique) - 1)
    val_shapes = set(rng.choice(unique, size=n_val, replace=False))
    mask = np.isin(shape_ids, list(val_shapes))
    fit = train.loc[~mask].copy()
    val = train.loc[mask].copy()
    return fit, val


def roofline_bound_regime(ai: float) -> str:
    if not np.isfinite(ai) or ai <= 0:
        return "unknown"
    if ai < ROOFLINE_RIDGE_AI / 3.0:
        return "memory-bound"
    if ai > ROOFLINE_RIDGE_AI * 3.0:
        return "compute-bound"
    return "transition"


def ndcg_at_k(y_true: np.ndarray, scores: np.ndarray, k: int = 10) -> float:
    n = len(y_true)
    if n == 0:
        return 0.0
    k = min(k, n)
    order = np.argsort(-scores)
    gains = y_true[order[:k]]
    discounts = 1.0 / np.log2(1.0 + np.arange(1, k + 1, dtype=np.float64))
    dcg = float(np.sum(gains * discounts))
    ideal = np.sort(y_true)[::-1][:k]
    idcg = float(np.sum(ideal * discounts))
    return dcg / idcg if idcg > 0 else 0.0


def mean_ndcg(df_grp: pd.DataFrame, scores: np.ndarray, k: int = 10) -> float:
    d = df_grp.copy()
    d["score"] = scores
    vals = []
    for _, g in d.groupby("group_id", sort=False):
        if len(g) < 2:
            continue
        vals.append(ndcg_at_k(g["y_norm"].to_numpy(dtype=np.float64), g["score"].to_numpy(dtype=np.float64), k=k))
    return float(np.mean(vals)) if vals else 0.0


def mean_selection_regret(df_grp: pd.DataFrame, scores: np.ndarray) -> float:
    """Mean top-1 selection regret over groups (same deployed metric as exhaustive eval)."""
    perg = select_and_regret(df_grp, scores)
    return float(perg["regret"].mean()) if len(perg) else float("nan")


def select_and_regret_extended(df_grp: pd.DataFrame, scores: np.ndarray) -> pd.DataFrame:
    d = df_grp.copy()
    d["score"] = scores
    perg = select_and_regret(df_grp, scores)
    perg["within1pct"] = (perg["regret"] <= 0.01).astype(int)
    perg["within10pct"] = (perg["regret"] <= 0.10).astype(int)

    ndcg_rows = []
    for gid, g in d.groupby("group_id", sort=False):
        y = g["y_norm"].to_numpy(dtype=np.float64)
        s = g["score"].to_numpy(dtype=np.float64)
        ndcg_rows.append(
            {
                "group_id": gid,
                "ndcg@1": ndcg_at_k(y, s, k=1),
                "ndcg@5": ndcg_at_k(y, s, k=5),
                "ndcg@10": ndcg_at_k(y, s, k=10),
            }
        )
    ndcg_df = pd.DataFrame(ndcg_rows)
    return perg.merge(ndcg_df, on="group_id", how="left")


def summarize_extended(perg: pd.DataFrame, label: str) -> dict:
    from harness import summarize

    s = summarize(perg, label)
    s["within1pct"] = float(perg["within1pct"].mean())
    s["within10pct"] = float(perg["within10pct"].mean())
    s["ndcg@1"] = float(perg["ndcg@1"].mean())
    s["ndcg@5"] = float(perg["ndcg@5"].mean())
    s["ndcg@10"] = float(perg["ndcg@10"].mean())
    return s


def write_regret_cdf(perg: pd.DataFrame, out_path: Path) -> None:
    r = np.sort(perg["regret"].to_numpy())
    cdf = pd.DataFrame({"regret": r, "cdf": (np.arange(1, len(r) + 1) / len(r))})
    cdf.to_csv(out_path, index=False)


def eval_metrics_bundle(
    ev: pd.DataFrame,
    scores: np.ndarray,
    label: str,
) -> tuple[pd.DataFrame, dict]:
    perg = select_and_regret_extended(ev, scores)
    metrics = summarize_extended(perg, label)
    return perg, metrics


def write_run_outputs(
    outdir: Path,
    perg: pd.DataFrame,
    metrics: dict,
    ev: pd.DataFrame,
    extra: dict,
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    perg.to_csv(outdir / "eval_regret.csv", index=False)
    write_regret_cdf(perg, outdir / "regret_cdf.csv")
    payload = {
        **extra,
        "eval": metrics,
        "baselines": baselines(ev),
        "by_regime": regime_breakdown(perg),
    }
    (outdir / "metrics.json").write_text(json.dumps(payload, indent=2))


def spearman(x: pd.Series, y: pd.Series) -> float:
    mask = x.notna() & y.notna()
    xv, yv = x[mask], y[mask]
    if len(xv) < MIN_GROUP_SAMPLES or xv.nunique() < 2 or yv.nunique() < 2:
        return np.nan
    return float(xv.corr(yv, method="spearman"))
