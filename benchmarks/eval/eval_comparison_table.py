#!/usr/bin/env python3
"""Build main eval comparison table (models + nvMMH) from artifacts/analysis/paper/."""

from __future__ import annotations

import json

import pandas as pd

from repo_paths import PAPER

PRIMARY_RUNS = [
    ("MLP (full)", "mlp_ranknet"),
    ("MLP (structural)", "mlp_mse_structural"),
    ("XGBoost (full)", "xgb_mse"),
    ("XGBoost (structural)", "xgb_mse_structural"),
]


def _load_metrics(run_dir: str) -> dict:
    m = json.loads((PAPER / run_dir / "metrics.json").read_text())
    e = m["eval"]
    return {
        "method": run_dir,
        "regret_mean": e["regret_mean"],
        "regret_median": e["regret_median"],
        "within1pct": e.get("within1pct", float("nan")),
        "within5pct": e["within5pct"],
        "within10pct": e.get("within10pct", float("nan")),
        "top1": e["top1"],
        "ndcg@10": e.get("ndcg@10", float("nan")),
    }


def main() -> None:
    rows = []
    for label, run in PRIMARY_RUNS:
        r = _load_metrics(run)
        r["method"] = label
        rows.append(r)

    for mode, label in [("top1", "nvMMH (top-1 tile)"), ("bestofk", "nvMMH (best-of-K tiles)")]:
        m = json.loads((PAPER / "nvmmh" / f"metrics_{mode}.json").read_text())["eval"]
        rows.append(
            {
                "method": label,
                "regret_mean": m["regret_mean"],
                "regret_median": m["regret_median"],
                "within1pct": m["within1pct"],
                "within5pct": m["within5pct"],
                "within10pct": m["within10pct"],
                "top1": m["top1"],
                "ndcg@10": float("nan"),
            }
        )

    # random baseline from any model metrics
    rnd = json.loads((PAPER / "mlp_mse" / "metrics.json").read_text())["baselines"]["random_pick"]["regret_mean"]
    rows.append(
        {
            "method": "Random pick",
            "regret_mean": rnd,
            "regret_median": float("nan"),
            "within1pct": float("nan"),
            "within5pct": float("nan"),
            "within10pct": float("nan"),
            "top1": float("nan"),
            "ndcg@10": float("nan"),
        }
    )

    df = pd.DataFrame(rows)
    out = PAPER / "comparison_eval.csv"
    df.to_csv(out, index=False)

    md = PAPER / "comparison_eval.md"
    lines = [
        "# Held-out eval comparison (68 groups)",
        "",
        "nvMMH: H100_NVL analytical picks (best Hopper target in GPU sweep), matched to measured CUTLASS catalogue on GH200 eval DB.",
        "",
        "| Method | Mean regret | Median regret | Within 1% | Within 5% | Within 10% | Top-1 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, r in df.iterrows():
        def pct(x):
            return f"{100*x:.1f}%" if pd.notna(x) else "—"

        lines.append(
            f"| {r.method} | {pct(r.regret_mean)} | {pct(r.regret_median)} | "
            f"{pct(r.within1pct)} | {pct(r.within5pct)} | {pct(r.within10pct)} | {pct(r.top1)} |"
        )
    md.write_text("\n".join(lines) + "\n")
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nWrote {out} and {md}")


if __name__ == "__main__":
    main()
