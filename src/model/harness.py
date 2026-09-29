"""Shared evaluation harness for GEMM kernel-selection models (XGBoost and MLP alike).

Judges a model by *selection quality* on a group of candidate configs, not
prediction error:

  - regret = 1 - chosen.mean_tflops / oracle, per (shape,layout) eval group
  - top-1 / top-5 recall, and "within 1% / 5% / 10% of oracle"
  - NDCG@k over the candidate ranking
  - p90 / p95 / max tail of the regret distribution
  - a regret-by-regime breakdown (square / tall / wide / skinny-K / large)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

NDCG_KS = (1, 5, 10)


def regime(M: int, N: int, K: int) -> str:
    """Replicates planning/plan/shapes._regime exactly."""
    if M >= 6144 and N >= 6144 and K >= 6144:
        return "large"
    if M / N >= 4:
        return "tall"
    if N / M >= 4:
        return "wide"
    if K <= min(M, N) / 4:
        return "skinny-K"
    return "square"


def make_relevance(grade: np.ndarray, grade_max: int, rel_top_bands: int = 2) -> np.ndarray:
    """Tie-collapsed grade (top band = grade_max) -> 0..rel_top_bands, top-focused."""
    return np.clip(grade - (grade_max - rel_top_bands), 0, rel_top_bands).astype("int64")


def base_shape_id(df: pd.DataFrame) -> np.ndarray:
    """One id per base (M,N,K) so all layouts of a shape share a CV fold."""
    return df.groupby(["M", "N", "K"], sort=False).ngroup().to_numpy()


def grouped_train_val_split(
    df: pd.DataFrame,
    val_frac: float = 0.15,
    seed: int = 42,
    stratify: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split training rows into (train, val) by *base shape*, never by row.

    All four layouts of an (M,N,K) share a side, so a layout of a shape can never be
    used to predict its sibling. Stratified by regime by default: the 593 training
    shapes run from 182 'tall' to 46 'large', and an unstratified draw makes the
    validation regime mix -- and so validation NDCG -- vary between runs that are
    supposed to differ only in training objective.

    Pass training rows only. Eval rows are the held-out test set and must not be
    reachable from here, so a mixed-split frame is rejected.
    """
    if not 0.0 < val_frac < 1.0:
        raise ValueError(f"val_frac must be in (0,1), got {val_frac}")
    if "split" in df.columns:
        present = set(pd.unique(df["split"].astype(str)))
        if present - {"train"}:
            raise ValueError(f"expected training rows only, found splits {sorted(present)}")

    ids = base_shape_id(df)
    shapes = df[["M", "N", "K"]].to_numpy()
    uniq, first_pos = np.unique(ids, return_index=True)
    rng = np.random.default_rng(seed)

    def _draw(pool: np.ndarray) -> np.ndarray:
        n = int(round(val_frac * len(pool)))
        n = min(max(n, 1), len(pool) - 1) if len(pool) > 1 else 0
        return rng.choice(pool, n, replace=False) if n else np.empty(0, dtype=pool.dtype)

    if stratify:
        regs = np.array([regime(*shapes[pos]) for pos in first_pos])
        picks = [_draw(uniq[regs == r]) for r in np.unique(regs)]
        val_ids = np.concatenate(picks) if picks else np.empty(0, dtype=uniq.dtype)
    else:
        val_ids = _draw(uniq)

    if len(val_ids) == 0:
        raise ValueError(
            f"val_frac={val_frac} yielded an empty validation set from "
            f"{len(uniq)} base shapes; every stratum would have been emptied. "
            "Raise val_frac or pass stratify=False."
        )
    mask = np.isin(ids, val_ids)
    return df[~mask].copy(), df[mask].copy()


def ndcg_at_k(gains: np.ndarray, scores: np.ndarray, k: int) -> float:
    """NDCG@k for one group, with *linear* gain = y_norm (throughput / group best).

    y_norm is already a normalized performance measure in [0,1], so linear gain keeps
    NDCG on the same scale as regret: NDCG@1 is exactly 1 - regret, because the ideal
    top-1 gain is 1.0 by construction. An exponential 2^g-1 transform would add nothing
    here (the gains are not graded relevance labels) while breaking that identity.
    """
    n = len(gains)
    kk = min(k, n)
    if kk == 0:
        return 0.0
    top = np.argpartition(-scores, kk - 1)[:kk]
    top = top[np.argsort(-scores[top], kind="stable")]
    disc = 1.0 / np.log2(np.arange(2, kk + 2))
    dcg = float((gains[top] * disc).sum())
    ideal = np.sort(gains)[::-1][:kk]
    idcg = float((ideal * disc).sum())
    return dcg / idcg if idcg > 0 else 0.0


def ndcg_per_group(df_grp: pd.DataFrame, scores: np.ndarray, ks=NDCG_KS) -> pd.DataFrame:
    """Per-group NDCG@k for each k, indexed by group_id."""
    gains = df_grp["y_norm"].to_numpy()
    rows = {}
    for gid, pos in df_grp.groupby("group_id").indices.items():
        g, sc = gains[pos], scores[pos]
        rows[gid] = {f"ndcg@{k}": ndcg_at_k(g, sc, k) for k in ks}
    return pd.DataFrame.from_dict(rows, orient="index").rename_axis("group_id")


def select_and_regret(df_grp: pd.DataFrame, scores: np.ndarray, ndcg_ks=NDCG_KS) -> pd.DataFrame:
    """Argmax the model score within each group; return per-group selection metrics.
    Regret uses group_best_tflops (true oracle on eval groups)."""
    d = df_grp[["group_id", "M", "N", "K", "mean_tflops", "group_best_tflops", "y_norm", "rank_in_group"]].copy()
    d["score"] = scores
    chosen = d.loc[d.groupby("group_id")["score"].idxmax()].reset_index(drop=True)
    out = chosen[["group_id", "M", "N", "K"]].copy()
    out["regret"] = 1.0 - chosen["y_norm"].to_numpy()
    out["top1"] = (chosen["rank_in_group"].to_numpy() == 1).astype(int)
    out["top5"] = (chosen["rank_in_group"].to_numpy() <= 5).astype(int)
    out["within1pct"] = (out["regret"] <= 0.01).astype(int)
    out["within5pct"] = (out["regret"] <= 0.05).astype(int)
    out["within10pct"] = (out["regret"] <= 0.10).astype(int)
    out["regime"] = [regime(m, n, k) for m, n, k in out[["M", "N", "K"]].to_numpy()]
    if ndcg_ks:
        nd = ndcg_per_group(df_grp, scores, ndcg_ks)
        for col in nd.columns:
            out[col] = out["group_id"].map(nd[col]).to_numpy()
    return out


def summarize(perg: pd.DataFrame, label: str) -> dict:
    """Selection-quality summary over per-group results: regret mean/median/p90/p95/max, top-1,
    top-5 and share within 5%.
    """
    r = perg["regret"].to_numpy()
    s = {
        "label": label,
        "groups": int(len(perg)),
        "regret_mean": float(np.mean(r)),
        "regret_median": float(np.median(r)),
        "regret_p90": float(np.quantile(r, 0.90)),
        "regret_p95": float(np.quantile(r, 0.95)),
        "regret_max": float(np.max(r)),
        "top1": float(perg["top1"].mean()),
        "top5": float(perg["top5"].mean()),
        "within1pct": float(perg["within1pct"].mean()),
        "within5pct": float(perg["within5pct"].mean()),
        "within10pct": float(perg["within10pct"].mean()),
    }
    for col in [c for c in perg.columns if c.startswith("ndcg@")]:
        s[col] = float(perg[col].mean())
    ndcg_str = "  ".join(f"{c}={s[c]:.4f}" for c in s if c.startswith("ndcg@"))
    print(
        f"  {label:18}  regret mean={s['regret_mean']:.3f} med={s['regret_median']:.3f} "
        f"p90={s['regret_p90']:.3f} p95={s['regret_p95']:.3f} max={s['regret_max']:.3f} | "
        f"top1={s['top1']:.2f} top5={s['top5']:.2f} "
        f"<1%={s['within1pct']:.2f} <5%={s['within5pct']:.2f} <10%={s['within10pct']:.2f}"
        f"  (n={s['groups']})"
    )
    if ndcg_str:
        print(f"  {'':18}  {ndcg_str}")
    return s


def random_pick_expected_per_group(df_eval: pd.DataFrame) -> pd.DataFrame:
    """Per-group expected regret under uniform random selection: 1 - mean(y_norm)."""
    rows = []
    for gid, g in df_eval.groupby("group_id", sort=False):
        row = g.iloc[0]
        regret = float(1.0 - g["y_norm"].mean())
        rows.append(
            {
                "group_id": int(gid),
                "M": int(row["M"]),
                "N": int(row["N"]),
                "K": int(row["K"]),
                "regret": regret,
                "top1": 0,
                "top5": 0,
                "within5pct": int(regret <= 0.05),
                "regime": regime(int(row["M"]), int(row["N"]), int(row["K"])),
            }
        )
    return pd.DataFrame(rows)


def random_pick_sampled_per_group(df_eval: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """One uniformly random valid config per group (reproducible given seed)."""
    rng = np.random.default_rng(seed)
    scores = rng.random(len(df_eval))
    return select_and_regret(df_eval, scores)


def baselines(df_eval: pd.DataFrame) -> dict:
    """Cheap reference points (no nvMMH yet): random pick and best single fixed config."""
    out = {}
    # Random pick: expected regret per group = 1 - mean(y_norm).
    expected = random_pick_expected_per_group(df_eval)
    out["random_pick"] = {
        "regret_mean": float(expected["regret"].mean()),
        "regret_median": float(expected["regret"].median()),
    }
    print(f"  {'random_pick':18}  regret mean={out['random_pick']['regret_mean']:.3f}")
    # Best single fixed config: the config name with best mean y_norm over EVAL groups
    # it appears in (a 'one config to rule them all' reference; needs a `name` column).
    if "name" in df_eval.columns:
        per_name = df_eval.groupby("name")["y_norm"]
        # require the config to appear in most eval groups to be a fair fixed choice
        cov = per_name.size()
        eligible = cov[cov >= 0.8 * df_eval["group_id"].nunique()].index
        if len(eligible):
            best = df_eval[df_eval["name"].isin(eligible)].groupby("name")["y_norm"].mean().idxmax()
            sub = df_eval[df_eval["name"] == best]
            regret = 1 - sub.groupby("group_id")["y_norm"].max()
            out["static_best_config"] = {
                "name": str(best),
                "coverage": int(sub["group_id"].nunique()),
                "regret_mean": float(regret.mean()),
            }
            print(
                f"  {'static_best_config':18}  regret mean={out['static_best_config']['regret_mean']:.3f} "
                f"(config covers {out['static_best_config']['coverage']}/{df_eval['group_id'].nunique()} groups)"
            )
    return out


def regime_breakdown(perg: pd.DataFrame) -> dict:
    """Mean regret, top-5 and within-5% per shape regime; also printed."""
    print("  regret by regime:")
    out = {}
    for reg, g in perg.groupby("regime"):
        out[reg] = {
            "groups": int(len(g)),
            "regret_mean": float(g["regret"].mean()),
            "top5": float(g["top5"].mean()),
            "within5pct": float(g["within5pct"].mean()),
        }
        print(
            f"      {reg:9} n={len(g):2d}  regret={out[reg]['regret_mean']:.3f} "
            f"top5={out[reg]['top5']:.2f} <5%={out[reg]['within5pct']:.2f}"
        )
    return out
