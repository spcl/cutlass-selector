#!/usr/bin/env python3
"""Local dev driver only. On the cluster see artifacts/analysis/paper/baselines/README.md."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from repo_paths import PAPER

SCRIPTS = Path(__file__).resolve().parent


def _run(cmd: list[str]) -> None:
    print(f"\n>> {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", type=Path, default=PAPER / "features.parquet")
    ap.add_argument("--skip-linear", action="store_true")
    ap.add_argument("--skip-analytical", action="store_true")
    ap.add_argument("--skip-plots", action="store_true")
    args = ap.parse_args()

    py = sys.executable
    _run([py, str(SCRIPTS / "feature_analysis.py"), "--features", str(args.features)])

    if not args.skip_linear:
        _run([py, str(SCRIPTS / "train_linear.py"), "--features", str(args.features)])

    if not args.skip_analytical:
        _run([py, str(SCRIPTS / "agentic_analytical.py"), "--features", str(args.features)])

    _run([py, str(SCRIPTS / "build_comparison_table.py")])

    if not args.skip_plots:
        _run([py, str(SCRIPTS / "plot_baselines.py")])

    print("\nBaseline pipeline complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
