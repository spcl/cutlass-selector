#!/usr/bin/env python3
"""Hardware-aware analytical score: training analysis -> candidate formulas -> val pick -> frozen test."""

from __future__ import annotations

import argparse
import json
import sys
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
    load_splits,
    mean_ndcg,
    mean_selection_regret,
    train_val_split,
    write_run_outputs,
)

# Always report exhaustive eval for these two structures (independent val weight pick each).
TABLE_VARIANTS: dict[str, tuple[str, str]] = {
    "additive_balanced": (
        "analytical_additive",
        "Hardware Analytical (additive)",
    ),
    "multiplicative_roofline": (
        "analytical_roofline",
        "Hardware Analytical (roofline)",
    ),
}
from features import DRAM_BW, PEAK_TC_BF16  # noqa: E402

WEIGHT_GRID = (0.5, 1.0, 2.0)
ROOFLINE_RIDGE = PEAK_TC_BF16 / DRAM_BW


class NormBounds:
    def __init__(self) -> None:
        self._bounds: dict[str, tuple[float, float]] = {}

    def fit(self, train: pd.DataFrame, cols: set[str], q_lo: float = 0.05, q_hi: float = 0.95) -> None:
        for col in cols:
            s = train[col].astype(float)
            lo, hi = float(s.quantile(q_lo)), float(s.quantile(q_hi))
            self._bounds[col] = (lo, hi if hi > lo else lo + 1.0)

    def transform(self, s: pd.Series, col: str) -> pd.Series:
        lo, hi = self._bounds[col]
        if hi <= lo:
            return pd.Series(0.5, index=s.index)
        return ((s.astype(float) - lo) / (hi - lo)).clip(0.0, 1.0)


def _term(df: pd.DataFrame, col: str, bounds: NormBounds, invert: bool = False) -> pd.Series:
    x = bounds.transform(df[col], col)
    return 1.0 - x if invert else x


def _build_additive(df: pd.DataFrame, weights: dict[str, float], term_map: dict[str, pd.Series]) -> pd.Series:
    score = pd.Series(0.0, index=df.index)
    for key, series in term_map.items():
        score = score + weights.get(key, 1.0) * series
    return score


def _build_multiplicative(df: pd.DataFrame, weights: dict[str, float], term_map: dict[str, pd.Series]) -> pd.Series:
    # weights act as exponents in {0.5, 1, 2} grid — interpret as emphasis, not learned coeffs
    score = pd.Series(1.0, index=df.index)
    for key, series in term_map.items():
        w = weights.get(key, 1.0)
        score = score * np.power(series.clip(1e-6, 1.0), w)
    return score


def _balanced_terms(df: pd.DataFrame, bounds: NormBounds) -> dict[str, pd.Series]:
    work = 1.0 - 0.5 * (_term(df, "m_quant_waste", bounds, invert=True) + _term(df, "n_quant_waste", bounds, invert=True))
    return {
        "work": work.clip(0.0, 1.0),
        "wave": _term(df, "last_wave_eff", bounds),
        "mem": _term(df, "mainloop_compute_intensity", bounds),
        "occ": _term(df, "true_blocks_per_sm", bounds),
        "pipe": 0.5 * (_term(df, "stage_amortization", bounds) + _term(df, "pipeline_fill_frac", bounds)),
    }


def _roofline_terms(df: pd.DataFrame, bounds: NormBounds) -> dict[str, pd.Series]:
    ai = df["mainloop_compute_intensity"].astype(float).clip(lower=1e-6)
    roof = pd.Series(np.minimum(1.0, ai / ROOFLINE_RIDGE), index=df.index)
    return {
        "roof": bounds.transform(roof, "mainloop_compute_intensity"),
        "work": 1.0 - _term(df, "m_quant_waste", bounds, invert=True),
        "wave": _term(df, "last_wave_eff", bounds),
        "occ": _term(df, "true_blocks_per_sm", bounds),
        "pipe": _term(df, "pipeline_fill_frac", bounds),
    }


def _memory_terms(df: pd.DataFrame, bounds: NormBounds) -> dict[str, pd.Series]:
    return {
        "mem_traffic": 1.0 - _term(df, "restream_factor", bounds, invert=True),
        "l2": 1.0 - _term(df, "working_set_vs_L2", bounds, invert=True),
        "wave": _term(df, "last_wave_eff", bounds),
        "occ": _term(df, "true_blocks_per_sm", bounds),
        "pipe": _term(df, "stage_amortization", bounds),
    }


def _occupancy_terms(df: pd.DataFrame, bounds: NormBounds) -> dict[str, pd.Series]:
    return {
        "occ": _term(df, "true_blocks_per_sm", bounds),
        "wave": _term(df, "last_wave_eff", bounds),
        "reg": 1.0 - _term(df, "reg_pressure_proxy", bounds, invert=True),
        "pipe": _term(df, "pipeline_fill_frac", bounds),
        "work": 1.0 - _term(df, "m_quant_waste", bounds, invert=True),
    }


def candidate_formulas() -> list[tuple[str, str, str, tuple[str, ...]]]:
    return [
        (
            "additive_balanced",
            "Hardware Analytical Score (additive)",
            "Sum of normalized work, wave, memory-intensity, occupancy, and pipeline terms.",
            ("work", "wave", "mem", "occ", "pipe"),
        ),
        (
            "multiplicative_roofline",
            "Hardware Analytical Score (roofline product)",
            "Roofline efficiency × work × wave × occupancy × pipeline fill.",
            ("roof", "work", "wave", "occ", "pipe"),
        ),
        (
            "memory_traffic_product",
            "Hardware Analytical Score (memory-aware)",
            "Penalizes restreaming and L2 pressure; multiplies wave, occupancy, pipeline.",
            ("mem_traffic", "l2", "wave", "occ", "pipe"),
        ),
        (
            "occupancy_first",
            "Hardware Analytical Score (occupancy-first)",
            "Occupancy and register headroom with wave/pipeline modulation.",
            ("occ", "wave", "reg", "pipe", "work"),
        ),
    ]


def _all_raw_cols() -> set[str]:
    cols = {
        "m_quant_waste", "n_quant_waste", "last_wave_eff", "mainloop_compute_intensity",
        "true_blocks_per_sm", "stage_amortization", "pipeline_fill_frac", "restream_factor",
        "working_set_vs_L2", "reg_pressure_proxy",
    }
    return cols


def score_formula(
    name: str,
    df: pd.DataFrame,
    weights: dict[str, float],
    bounds: NormBounds,
) -> pd.Series:
    if name == "additive_balanced":
        return _build_additive(df, weights, _balanced_terms(df, bounds))
    if name == "multiplicative_roofline":
        return _build_multiplicative(df, weights, _roofline_terms(df, bounds))
    if name == "memory_traffic_product":
        return _build_multiplicative(df, weights, _memory_terms(df, bounds))
    if name == "occupancy_first":
        return _build_additive(df, weights, _occupancy_terms(df, bounds))
    raise KeyError(name)


def _weight_combos(keys: tuple[str, ...]) -> list[dict[str, float]]:
    """Coarse grid: unit weights plus single-term emphasis/de-emphasis."""
    base = {k: 1.0 for k in keys}
    combos = [dict(base)]
    for key in keys:
        for scale in (0.5, 2.0):
            w = dict(base)
            w[key] = scale
            combos.append(w)
    seen = set()
    uniq = []
    for c in combos:
        sig = tuple(sorted(c.items()))
        if sig not in seen:
            seen.add(sig)
            uniq.append(c)
    return uniq


def _validation_metrics(val: pd.DataFrame, scores: np.ndarray) -> tuple[float, float]:
    return mean_ndcg(val, scores, k=10), mean_selection_regret(val, scores)


def select_on_validation(
    train: pd.DataFrame,
    bounds: NormBounds,
) -> tuple[str, dict[str, float], float, float, pd.DataFrame]:
    _, val = train_val_split(train)
    records = []
    best_name, best_weights = "", {k: 1.0 for k in ("work",)}
    best_ndcg, best_regret = -1.0, float("nan")
    for name, _paper, _desc, keys in candidate_formulas():
        for weights in _weight_combos(keys):
            scores = score_formula(name, val, weights, bounds).to_numpy(dtype=np.float64)
            ndcg, regret = _validation_metrics(val, scores)
            records.append(
                {
                    "formula": name,
                    "weights": json.dumps(weights),
                    "val_ndcg@10": ndcg,
                    "val_regret_mean": regret,
                }
            )
            if ndcg > best_ndcg:
                best_name, best_weights, best_ndcg, best_regret = name, weights, ndcg, regret
    return best_name, best_weights, best_ndcg, best_regret, pd.DataFrame(records)


def select_best_weights_for_formula(
    formula_name: str,
    train: pd.DataFrame,
    bounds: NormBounds,
) -> tuple[dict[str, float], float, float, pd.DataFrame]:
    """Pick weights for one formula on validation (NDCG@10); return sweep records."""
    _, val = train_val_split(train)
    keys = next(t for n, _p, _d, t in candidate_formulas() if n == formula_name)
    records = []
    best_weights = {k: 1.0 for k in keys}
    best_ndcg, best_regret = -1.0, float("nan")
    for weights in _weight_combos(keys):
        scores = score_formula(formula_name, val, weights, bounds).to_numpy(dtype=np.float64)
        ndcg, regret = _validation_metrics(val, scores)
        records.append(
            {
                "formula": formula_name,
                "weights": json.dumps(weights),
                "val_ndcg@10": ndcg,
                "val_regret_mean": regret,
            }
        )
        if ndcg > best_ndcg:
            best_weights, best_ndcg, best_regret = weights, ndcg, regret
    return best_weights, best_ndcg, best_regret, pd.DataFrame(records)


def write_readme(
    path: Path,
    corr_path: Path,
    effect_path: Path,
    chosen_name: str,
    weights: dict[str, float],
    val_ndcg: float,
    val_regret: float,
) -> None:
    top_corr = pd.read_csv(corr_path).sort_values("mean_rho", key=lambda s: s.abs(), ascending=False).head(8)
    top_eff = pd.read_csv(effect_path).head(8)
    lines = [
        "# Hardware Analytical Score — design trace",
        "",
        "## Data splits (leakage protocol)",
        "",
        "- **Training analysis** (`feature_correlations*.csv`, redundancy, effect sizes): `split==train` only.",
        f"- **Formula selection**: validation holdout ({int(100*VAL_FRAC)}% of base shapes, seed={VAL_SEED}); metric = mean NDCG@10.",
        "- **Exhaustive evaluation**: 68 held-out groups (`split==eval`); run once after freezing.",
        "",
        "## Training statistics motivating terms",
        "",
        "Top features by |mean within-group Spearman| with throughput (training only):",
        "",
        "| Feature | Mean ρ | Frac positive |",
        "|---|---:|---:|",
    ]
    for _, r in top_corr.iterrows():
        lines.append(f"| {r.feature} | {r.mean_rho:.3f} | {r.frac_positive:.2f} |")
    lines += [
        "",
        "Top-5% vs rest effect sizes (training):",
        "",
        "| Feature | Mean effect |",
        "|---|---:|",
    ]
    for _, r in top_eff.iterrows():
        lines.append(f"| {r.feature} | {r.mean_effect:.3f} |")
    lines += [
        "",
        "## Candidate formulas",
        "",
    ]
    for name, _paper, desc, terms in candidate_formulas():
        lines.append(f"### `{name}`")
        lines.append(desc)
        lines.append(f"- Terms: {', '.join(terms)}")
        lines.append("")
    lines += [
        "## Validation selection",
        "",
        f"- Chosen formula: **{chosen_name}**",
        f"- Validation NDCG@10: **{val_ndcg:.4f}**",
        f"- Validation mean regret: **{val_regret:.4f}**",
        f"- Weights: `{json.dumps(weights)}`",
        "",
        "## Final frozen score",
        "",
        f"score(c) = {chosen_name} with weights {json.dumps(weights)}",
        "",
        "Normalization: per-term robust min–max to [0,1] using training quantiles (5th–95th).",
        "",
        "See `candidate_formulas.csv` for the full validation grid and `final_formula.json` for the frozen artifact.",
    ]
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    ap.add_argument("--outdir", type=Path, default=BASELINES / "agentic_analytical")
    ap.add_argument("--analysis-dir", type=Path, default=BASELINES)
    args = ap.parse_args()

    train, ev, _, features_path = load_splits(args.features)
    print(f"using features: {features_path}")
    bounds = NormBounds()
    bounds.fit(train, _all_raw_cols())

    chosen_name, weights, val_ndcg, val_regret, grid = select_on_validation(train, bounds)
    grid.to_csv(args.analysis_dir / "candidate_formulas.csv", index=False)

    paper_name = "Hardware Analytical Score"
    desc = next(d for n, _p, d, _t in candidate_formulas() if n == chosen_name)
    terms = next(t for n, _p, _d, t in candidate_formulas() if n == chosen_name)

    frozen = {
        "internal_name": "agentic_analytical",
        "paper_name": paper_name,
        "formula": chosen_name,
        "description": desc,
        "terms": list(terms),
        "weights": weights,
        "weight_grid": list(WEIGHT_GRID),
        "validation_ndcg@10": val_ndcg,
        "validation_regret_mean": val_regret,
        "normalization": "robust_minmax_q05_q95_fit_on_train",
        "selection_protocol": "validation NDCG@10 over coarse weight grid {0.5,1,2}",
    }
    args.outdir.mkdir(parents=True, exist_ok=True)
    (args.outdir / "final_formula.json").write_text(json.dumps(frozen, indent=2))

    test_scores = score_formula(chosen_name, ev, weights, bounds).to_numpy(dtype=np.float64)
    perg, metrics = eval_metrics_bundle(ev, test_scores, paper_name)
    extra = {
        "model": "agentic_analytical",
        "internal_name": "agentic_analytical",
        "paper_name": paper_name,
        "frozen_formula": frozen,
    }
    write_run_outputs(args.outdir, perg, metrics, ev, extra)

    write_readme(
        args.outdir / "README.md",
        args.analysis_dir / "feature_correlations.csv",
        args.analysis_dir / "top5_effect_sizes.csv",
        chosen_name,
        weights,
        val_ndcg,
        val_regret,
    )
    print(
        f"selected {chosen_name}  val_ndcg@10={val_ndcg:.4f}  val_regret={val_regret:.4f}  "
        f"test regret_mean={metrics['regret_mean']:.4f}"
    )
    print(f"wrote {args.outdir}")

    for formula_name, (subdir, label) in TABLE_VARIANTS.items():
        v_weights, v_ndcg, v_regret, v_grid = select_best_weights_for_formula(
            formula_name, train, bounds
        )
        v_out = args.analysis_dir / subdir
        v_out.mkdir(parents=True, exist_ok=True)
        v_grid.to_csv(v_out / "weight_sweep.csv", index=False)
        v_frozen = {
            "internal_name": subdir,
            "paper_name": label,
            "formula": formula_name,
            "weights": v_weights,
            "validation_ndcg@10": v_ndcg,
            "validation_regret_mean": v_regret,
            "selection_protocol": "per-formula validation NDCG@10 (weights only)",
        }
        (v_out / "final_formula.json").write_text(json.dumps(v_frozen, indent=2))
        v_scores = score_formula(formula_name, ev, v_weights, bounds).to_numpy(dtype=np.float64)
        v_perg, v_metrics = eval_metrics_bundle(ev, v_scores, label)
        v_extra = {
            "model": "agentic_analytical_variant",
            "internal_name": subdir,
            "paper_name": label,
            "frozen_formula": v_frozen,
        }
        write_run_outputs(v_out, v_perg, v_metrics, ev, v_extra)
        print(
            f"variant {formula_name} → {subdir}  val_ndcg@10={v_ndcg:.4f}  "
            f"val_regret={v_regret:.4f}  test regret_mean={v_metrics['regret_mean']:.4f}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
