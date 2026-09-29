#!/usr/bin/env python3
"""Compare MLP scaler policies on FP16 25% hardware-aware (pretrain proxy)."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys

from repo_paths import BENCHMARKS, SRC

sys.path.insert(0, str(SRC))
sys.path.insert(0, str(BENCHMARKS))
from transfer_fusion.study_common import (  # noqa: E402
    MLP_EPOCHS_PRETRAIN,
    MLP_LR_PRETRAIN,
    PRETRAIN_MLP,
    STUDY_ROOT,
    features_path,
    scaler_policy_path,
    subset_path,
)
from transfer_fusion.train_one import prepare_run_parquet  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--choose", choices=["fit", "source", "auto"], default="auto")
    args = ap.parse_args()

    dtype = "fp16"
    fraction = 0.25
    seed = 0
    shapes = json.loads(subset_path(dtype, seed, fraction).read_text())["shapes"]
    shapes = [tuple(s) for s in shapes]

    base = STUDY_ROOT / "scaler_compare"
    base.mkdir(parents=True, exist_ok=True)
    pq = prepare_run_parquet(
        features_path(dtype),
        shapes,
        "full",
        base / "features_frac025_seed0_full.parquet",
    )

    results = []
    train_py = SRC / "model" / "train_mlp.py"
    for policy in ("fit", "source"):
        out = base / f"mlp_full_pretrain_{policy}"
        cmd = [
            args.python, str(train_py),
            "--features", str(pq),
            "--loss", "mse",
            "--epochs", str(MLP_EPOCHS_PRETRAIN),
            "--lr", str(MLP_LR_PRETRAIN),
            "--init-checkpoint", str(PRETRAIN_MLP["full"]),
            "--scaler-policy", policy,
            "--skip-eval",
            "--outdir", str(out),
        ]
        print(" ".join(cmd))
        subprocess.check_call(cmd)
        meta = json.loads((out / "metrics.json").read_text())
        results.append({"policy": policy, "train_seconds": meta.get("train_seconds")})

    chosen = "fit" if args.choose == "auto" else args.choose
    payload = {
        "comparison": results,
        "policy": chosen,
        "note": "fit refits StandardScaler on target fusion features; source reuses BF16 scaler.",
    }
    scaler_policy_path().parent.mkdir(parents=True, exist_ok=True)
    scaler_policy_path().write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
