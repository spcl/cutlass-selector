#!/usr/bin/env python3
"""Build full feature parquets for fusion FP16 / FP32 / FP8 sweeps."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from repo_paths import BENCHMARKS, SRC

sys.path.insert(0, str(SRC))
sys.path.insert(0, str(BENCHMARKS))
from transfer_fusion.study_common import DTYPE_TAGS, features_path, sweep_db_path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--autotuner", type=Path, default=Path.home() / "autotuner")
    ap.add_argument("--dtype", choices=list(DTYPE_TAGS) + ["all"], default="all")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument(
        "--holdout-frac",
        type=float,
        default=0.0,
        help="hold out train groups for eval when DB has no oracle (typical for fusion sweeps)",
    )
    args = ap.parse_args()

    features_py = SRC / "model" / "features_fusion.py"

    dtypes = list(DTYPE_TAGS) if args.dtype == "all" else [args.dtype]
    for dtype in dtypes:
        tag, feat_dtype = DTYPE_TAGS[dtype]
        db = sweep_db_path(dtype, args.autotuner)
        if not db.is_file():
            raise SystemExit(f"missing {db}")
        out = features_path(dtype)
        out.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            args.python,
            str(features_py),
            "--db",
            str(db),
            "--train-tags",
            tag,
            "--dtype",
            feat_dtype,
            "--out",
            str(out),
        ]
        if args.holdout_frac > 0:
            cmd.extend(["--holdout-frac", str(args.holdout_frac)])
        print(" ".join(cmd))
        subprocess.check_call(cmd)
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
