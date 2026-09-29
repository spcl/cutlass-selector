#!/usr/bin/env python3
"""Combined exhaustive-eval table for baselines + existing paper methods."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

from repo_paths import PAPER

BASELINES = PAPER / "baselines"
OUT_CSV = BASELINES / "exhaustive_baseline_results.csv"


def _row_from_eval(m: dict, label: str) -> dict:
    return {
        "method": label,
        "regret_mean": m["regret_mean"],
        "regret_median": m["regret_median"],
        "within1pct": m.get("within1pct", float("nan")),
        "within5pct": m["within5pct"],
        "top1": m["top1"],
    }


def _load_eval(run_dir: Path, label: str) -> dict:
    m = json.loads((run_dir / "metrics.json").read_text())["eval"]
    return _row_from_eval(m, label)


def _try_load_eval(run_dir: Path, label: str) -> dict | None:
    path = run_dir / "metrics.json"
    if not path.is_file():
        print(f"warning: skipping {label} — missing {path}", file=sys.stderr)
        return None
    return _load_eval(run_dir, label)


def _static_best(metrics_path: Path) -> dict | None:
    if not metrics_path.is_file():
        return None
    m = json.loads(metrics_path.read_text())
    sb = m.get("baselines", {}).get("static_best_config")
    if not sb:
        return None
    return {
        "method": "CUTLASS static best",
        "regret_mean": sb["regret_mean"],
        "regret_median": float("nan"),
        "within1pct": float("nan"),
        "within5pct": float("nan"),
        "top1": float("nan"),
    }


def _random_regret() -> float:
    for path in (
        PAPER / "mlp_mse" / "metrics.json",
        BASELINES / "linear_full" / "metrics.json",
        BASELINES / "agentic_analytical" / "metrics.json",
    ):
        if not path.is_file():
            continue
        rnd = json.loads(path.read_text()).get("baselines", {}).get("random_pick", {}).get("regret_mean")
        if rnd is not None:
            return float(rnd)
    raise FileNotFoundError("random_pick baseline not found in mlp_mse or baseline metrics.json")


def main() -> int:
    rows: list[dict] = []

    rows.append(
        {
            "method": "Random",
            "regret_mean": _random_regret(),
            "regret_median": float("nan"),
            "within1pct": float("nan"),
            "within5pct": float("nan"),
            "top1": float("nan"),
        }
    )

    static = _static_best(PAPER / "mlp_mse" / "metrics.json")
    if static is None:
        static = _static_best(BASELINES / "linear_full" / "metrics.json")
    if static:
        rows.append(static)

    nv_path = PAPER / "nvmmh" / "metrics_top1.json"
    if nv_path.is_file():
        nv = json.loads(nv_path.read_text())["eval"]
        rows.append(_row_from_eval(nv, "nvMMH"))
    else:
        print(f"warning: skipping nvMMH — missing {nv_path}", file=sys.stderr)

    for run_dir, label in [
        (BASELINES / "agentic_analytical", "Hardware Analytical Score"),
        (BASELINES / "analytical_additive", "Hardware Analytical (additive)"),
        (BASELINES / "analytical_roofline", "Hardware Analytical (roofline)"),
        (BASELINES / "linear_structural", "Linear (structural)"),
        (BASELINES / "linear_full", "Linear (hardware-aware)"),
        (PAPER / "mlp_mse_structural", "MLP (structural)"),
        (PAPER / "mlp_ranknet", "MLP (hardware-aware)"),
        (PAPER / "xgb_mse_structural", "XGBoost (structural)"),
        (PAPER / "xgb_mse", "XGBoost (hardware-aware)"),
    ]:
        row = _try_load_eval(run_dir, label)
        if row is not None:
            rows.append(row)

    df = pd.DataFrame(rows)
    BASELINES.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}" if pd.notna(x) else "—"))
    print(f"\nWrote {OUT_CSV}")

    _write_summary(df)
    return 0


def _row(df: pd.DataFrame, method: str) -> pd.Series | None:
    sub = df.loc[df.method == method]
    return sub.iloc[0] if len(sub) else None


def _write_summary(df: pd.DataFrame) -> None:
    lin_full = _row(df, "Linear (hardware-aware)")
    lin_struct = _row(df, "Linear (structural)")
    anal = _row(df, "Hardware Analytical Score")
    nv = _row(df, "nvMMH")
    mlp = _row(df, "MLP (hardware-aware)")

    frozen = json.loads((BASELINES / "agentic_analytical" / "final_formula.json").read_text())
    ridge_meta = json.loads((BASELINES / "linear_full" / "metrics.json").read_text())

    h1 = (
        anal is not None
        and nv is not None
        and anal.regret_mean < nv.regret_mean - 0.02
    )
    h2 = (
        lin_full is not None
        and anal is not None
        and lin_full.regret_mean < anal.regret_mean - 0.01
    )
    h3 = (
        mlp is not None
        and lin_full is not None
        and mlp.regret_mean < lin_full.regret_mean - 0.01
    )

    lines = [
        "# Baseline study summary",
        "",
        "## Protocol",
        "",
        "- Features: `artifacts/analysis/paper/features.parquet` (train + 68-group eval).",
        "- Linear ridge: α tuned on 15% shape holdout (seed 42) via validation NDCG@10; refit on full train.",
        "- Analytical score: training-only correlation analysis; formula chosen on validation NDCG@10; frozen before test.",
        "- Exhaustive eval: 68 held-out `(M,N,K,layout)` groups — never used for formula design.",
        "",
        "## Selected linear regularization (full features)",
        "",
        f"- Ridge α = **{ridge_meta.get('alpha', 'n/a')}**",
        f"- Validation NDCG@10: {ridge_meta.get('validation_ndcg@10', float('nan')):.4f}",
        f"- Validation mean regret: {ridge_meta.get('validation_regret_mean', float('nan')):.4f}",
        "",
        "## Final analytical formula",
        "",
        f"- Name: **{frozen['paper_name']}** (`{frozen['formula']}`)",
        f"- Weights: `{json.dumps(frozen['weights'])}`",
        f"- Validation NDCG@10: {frozen['validation_ndcg@10']:.4f}",
        f"- Validation mean regret: {frozen.get('validation_regret_mean', float('nan')):.4f}",
        "",
        "## Headline exhaustive-eval numbers",
        "",
        "| Method | Mean regret | Within 5% | Top-1 |",
        "|---|---:|---:|---:|",
    ]
    for _, r in df.iterrows():
        lines.append(
            f"| {r.method} | {100*r.regret_mean:.1f}% | "
            f"{100*r.within5pct:.1f}% | {100*r.top1:.1f}% |"
            if pd.notna(r.within5pct) and pd.notna(r.top1)
            else f"| {r.method} | {100*r.regret_mean:.1f}% | — | — |"
        )
    lines += ["", "## Hypothesis check", ""]
    if anal is not None and nv is not None:
        lines.append(
            f"1. **Hardware-aware features encode architectural knowledge** (analytical ≫ random/nvMMH): "
            f"{'supported' if h1 else 'not clearly supported'} "
            f"(analytical {100*anal.regret_mean:.1f}% vs nvMMH {100*nv.regret_mean:.1f}%)."
        )
    else:
        lines.append("1. **Hardware-aware features** — nvMMH row missing; compare analytical vs random in table.")
    if lin_full is not None and anal is not None:
        lines.append(
            f"2. **Learning feature weights helps** (linear full > analytical): "
            f"{'supported' if h2 else 'not clearly supported'} "
            f"(linear {100*lin_full.regret_mean:.1f}% vs analytical {100*anal.regret_mean:.1f}%)."
        )
    if mlp is not None and lin_full is not None:
        lines.append(
            f"3. **Nonlinear interactions matter** (MLP > linear full): "
            f"{'supported' if h3 else 'not clearly supported'} "
            f"(MLP {100*mlp.regret_mean:.1f}% vs linear {100*lin_full.regret_mean:.1f}%)."
        )
    if lin_struct is not None:
        lines.append("")
        lines.append(f"Structural linear reference: {100*lin_struct.regret_mean:.1f}% mean regret.")
    (BASELINES / "SUMMARY.md").write_text("\n".join(lines) + "\n")
    print(f"Wrote {BASELINES / 'SUMMARY.md'}")


if __name__ == "__main__":
    raise SystemExit(main())
