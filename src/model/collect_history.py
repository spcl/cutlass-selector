#!/usr/bin/env python3
"""
Reduce a training sweep to stacked training curves and a per-run summary.

Reads every per-run directory written by train_xgb.py / train_mlp.py under an
artifacts root and emits:

    training_history_xgb.csv   all XGBoost runs stacked (one row per boosting round)
    training_history_mlp.csv   all MLP runs stacked (one row per epoch)
    summary_training.md        best validation NDCG + training time per run

Per-run directories are named <family>_<objective>[_<feature_set>], which is what
the trainers' default --outdir produces, but the family/objective/feature-set
labels are read from each run's metrics.json rather than parsed out of the path.

Usage:
    python src/model/collect_history.py --artifacts artifacts/analysis/paper
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

HISTORY_FILES = {"xgboost": "training_history_xgb.csv", "mlp": "training_history_mlp.csv"}


def _capacity(metrics: dict, family: str) -> str:
    """Human-readable capacity label. Distinguishes runs that differ only in model size."""
    if family == "mlp":
        return "x".join(str(h) for h in metrics.get("hidden", []))
    depth = metrics.get("max_depth")
    return f"depth{depth}" if depth is not None else ""


def _params(metrics: dict, family: str):
    """Free real-valued parameters: MLP weights, or total booster leaves.

    Not an exact equivalence -- a leaf value and a weight are both one number but are
    not equally expressive -- so plot it as 'model size', not as interchangeable counts.
    """
    return metrics.get("n_params") if family == "mlp" else metrics.get("n_leaves")


def _family(metrics: dict, run_dir: Path) -> str:
    if "n_estimators" in metrics:
        return "xgboost"
    if "epochs" in metrics:
        return "mlp"
    raise ValueError(f"cannot tell model family from {run_dir / 'metrics.json'}")


def load_runs(artifacts: Path) -> list[dict]:
    """One record per run directory holding a metrics.json."""
    runs = []
    for mpath in sorted(artifacts.glob("*/metrics.json")):
        metrics = json.loads(mpath.read_text())
        run_dir = mpath.parent
        family = _family(metrics, run_dir)
        hist_path = run_dir / HISTORY_FILES[family]
        runs.append(
            {
                "dir": run_dir,
                "family": family,
                "metrics": metrics,
                "history": pd.read_csv(hist_path) if hist_path.exists() else None,
            }
        )
    return runs


def combined_history(runs: list[dict], family: str) -> pd.DataFrame:
    """Stack every run's history, tagging each row with the run's capacity and size.

    Without these columns a capacity sweep is ambiguous once stacked: rows from a
    16x8 net and a 1024x1024x512x256 net differ only in their numbers, and would
    silently average together in any downstream group-by.
    """
    frames = []
    for r in runs:
        if r["family"] != family or r["history"] is None:
            continue
        h = r["history"].copy()
        h["capacity"] = _capacity(r["metrics"], "mlp" if family == "mlp" else "xgb")
        h["params"] = _params(r["metrics"], "mlp" if family == "mlp" else "xgb")
        h["run_dir"] = r["dir"].name
        frames.append(h)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _fmt_time(seconds) -> str:
    if seconds is None:
        return "-"
    seconds = float(seconds)
    if seconds < 90:
        return f"{seconds:.0f} s"
    if seconds < 5400:
        return f"{seconds / 60:.1f} min"
    return f"{seconds / 3600:.2f} h"


def summary_table(runs: list[dict]) -> pd.DataFrame:
    """One row per training run: model, objective, feature set, capacity, best validation NDCG,
    training time and evaluation regret.
    """
    rows = []
    for r in runs:
        m = r["metrics"]
        ev = m.get("eval") or {}
        rows.append(
            {
                "model": "XGBoost" if r["family"] == "xgboost" else "MLP",
                "objective": m.get("loss"),
                "feature_set": m.get("feature_set", "full"),
                "capacity": _capacity(m, "mlp" if r["family"] == "mlp" else "xgb"),
                "params": _params(m, "mlp" if r["family"] == "mlp" else "xgb"),
                "seed": m.get("seed"),
                "best_val_ndcg@10": m.get("best_val_ndcg@10"),
                "best_step": m.get("best_round", m.get("best_epoch")),
                "steps": m.get("n_estimators", m.get("epochs")),
                "train_seconds": m.get("train_seconds"),
                "eval_regret_mean": ev.get("regret_mean"),
                "eval_ndcg@10": ev.get("ndcg@10"),
                "eval_groups": ev.get("groups"),
                "run_dir": r["dir"].name,
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values(
        ["model", "objective", "feature_set", "params", "seed"], na_position="first"
    ).reset_index(drop=True)


def _capacity_section(summary: pd.DataFrame) -> list[str]:
    """Capacity sweep view: one row per capacity, aggregated over seeds.

    Keyed on `capacity` alone -- the knob we set. `params` is a *consequence* and
    legitimately differs between the feature sets (structural has fewer inputs, so
    fewer first-layer weights) and between seeds for XGBoost, so keying on it would
    split full from structural and the gap could never be computed.
    """
    out = ["## Capacity sweep", "",
           "Aggregated over seeds; `gap` is full minus structural validation NDCG@10,",
           "so a positive gap means the hardware-aware features are still helping.", ""]
    for model, block in summary.groupby("model"):
        block = block[block["capacity"].astype(str) != ""]
        if block["capacity"].nunique() < 2:
            continue  # not a capacity sweep for this family
        g = block.groupby(["capacity", "feature_set"])
        agg = g["best_val_ndcg@10"].agg(["mean", "std", "count"])
        par = g["params"].mean()
        caps = (block.groupby("capacity")["params"].mean().sort_values().index)
        out += [f"### {model}", "",
                "| Capacity | full params | full NDCG@10 | structural params | structural NDCG@10 | gap | seeds |",
                "|---|---:|---:|---:|---:|---:|---:|"]
        for cap in caps:
            cells, seeds = {}, []
            for fs in ("full", "structural"):
                if (cap, fs) in agg.index:
                    row = agg.loc[(cap, fs)]
                    sd = f" ±{row['std']:.4f}" if pd.notna(row["std"]) else ""
                    pv = par.loc[(cap, fs)] if (cap, fs) in par.index else float("nan")
                    cells[fs] = (f"{row['mean']:.4f}{sd}", row["mean"],
                                 "-" if pd.isna(pv) else f"{int(pv):,}")
                    seeds.append(int(row["count"]))
                else:
                    cells[fs] = ("-", float("nan"), "-")
            gap = cells["full"][1] - cells["structural"][1]
            gap_s = "-" if pd.isna(gap) else f"{gap:+.4f}"
            out.append(f"| {cap} | {cells['full'][2]} | {cells['full'][0]} | "
                       f"{cells['structural'][2]} | {cells['structural'][0]} | {gap_s} | "
                       f"{max(seeds) if seeds else 0} |")
        out.append("")
    return out


def markdown_summary(summary: pd.DataFrame) -> str:
    """Render the summary table as markdown (grouped by capacity when the runs form a capacity sweep)."""
    out = ["# Training sweep summary", ""]
    if summary.empty:
        out.append("No runs found.")
        return "\n".join(out) + "\n"

    # A capacity sweep has many runs per (objective, feature_set); the flat table
    # below would repeat identical-looking rows, so switch views. Judged per model
    # family and ignoring blank labels, so a mixed or older artifacts directory
    # (where one family records no capacity at all) is not mistaken for a sweep.
    labelled = summary[summary["capacity"].astype(str) != ""]
    multi_capacity = (
        bool(labelled.groupby("model")["capacity"].nunique().max() > 1)
        if not labelled.empty else False
    )
    if multi_capacity:
        out += _capacity_section(summary)

    show_seed = summary["seed"].nunique() > 1
    for feature_set, block in summary.groupby("feature_set"):
        out += [f"## Feature set: {feature_set}", ""]
        head = "| Model | Objective |"
        sep = "|---|---|"
        if multi_capacity:
            head += " Capacity | Params |"
            sep += "---|---:|"
        if show_seed:
            head += " Seed |"
            sep += "---:|"
        out.append(head + " Best validation NDCG@10 | At step | Training time |")
        out.append(sep + "---:|---:|---:|")
        for _, r in block.iterrows():
            ndcg = "-" if pd.isna(r["best_val_ndcg@10"]) else f"{r['best_val_ndcg@10']:.4f}"
            step = "-" if pd.isna(r["best_step"]) else f"{int(r['best_step'])}/{int(r['steps'])}"
            row = f"| {r['model']} | {r['objective']} |"
            if multi_capacity:
                par = "-" if pd.isna(r["params"]) else f"{int(r['params']):,}"
                row += f" {r['capacity']} | {par} |"
            if show_seed:
                row += f" {r['seed']} |"
            out.append(row + f" {ndcg} | {step} | {_fmt_time(r['train_seconds'])} |")
        out.append("")

    scored = summary[summary["eval_regret_mean"].notna()]
    if not scored.empty:
        out += [
            "## Held-out eval groups",
            "",
            "Reported for reference only -- the objective for section 3 is chosen on",
            "validation NDCG, never on these numbers.",
            "",
            "| Model | Objective | Feature set | Mean regret | NDCG@10 | Groups |",
            "|---|---|---|---:|---:|---:|",
        ]
        for _, r in scored.iterrows():
            out.append(
                f"| {r['model']} | {r['objective']} | {r['feature_set']} | "
                f"{r['eval_regret_mean']:.4f} | {r['eval_ndcg@10']:.4f} | {int(r['eval_groups'])} |"
            )
        out.append("")

    for feature_set, block in summary.groupby("feature_set"):
        picks = block.dropna(subset=["best_val_ndcg@10"])
        # Picking a "best objective" across capacities would just report the biggest
        # model, which is not what the objective comparison means.
        if picks.empty or multi_capacity:
            continue
        out.append(f"**Best objective per family ({feature_set} features, by validation NDCG@10):**")
        out.append("")
        for model, g in picks.groupby("model"):
            best = g.loc[g["best_val_ndcg@10"].idxmax()]
            out.append(f"- {model}: `{best['objective']}` (NDCG@10 {best['best_val_ndcg@10']:.4f})")
        out.append("")
    return "\n".join(out) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", type=Path, required=True,
                    help="directory holding the per-run output dirs")
    ap.add_argument("--out", type=Path, default=None,
                    help="where to write the combined files (default: --artifacts)")
    args = ap.parse_args()
    out = args.out or args.artifacts
    out.mkdir(parents=True, exist_ok=True)

    runs = load_runs(args.artifacts)
    if not runs:
        print(f"no runs found under {args.artifacts}")
        return 1
    print(f"found {len(runs)} run(s) under {args.artifacts}")

    for family, fname in HISTORY_FILES.items():
        hist = combined_history(runs, family)
        if hist.empty:
            print(f"  {family:8} no training history")
            continue
        hist.to_csv(out / fname, index=False)
        n_runs = hist["run_dir"].nunique()
        print(f"  {family:8} {len(hist):,} rows / {n_runs} run(s) -> {out / fname}")

    summary = summary_table(runs)
    (out / "summary_training.md").write_text(markdown_summary(summary))
    summary.to_csv(out / "summary_training.csv", index=False)
    print(f"  summary  -> {out / 'summary_training.md'}")
    print()
    cols = ["model", "objective", "feature_set", "capacity", "params", "seed",
            "best_val_ndcg@10", "train_seconds"]
    cols = [c for c in cols if summary[c].notna().any()]
    print(summary[cols].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
