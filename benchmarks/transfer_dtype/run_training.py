#!/usr/bin/env python3
"""Run the full dtype-transfer training grid."""

from __future__ import annotations

import argparse
import sys

from repo_paths import BENCHMARKS, SRC

sys.path.insert(0, str(SRC))
sys.path.insert(0, str(BENCHMARKS))
from transfer_dtype.study_common import iter_run_specs, write_run_manifest
from transfer_dtype.train_one import train_run


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dtype", choices=["fp32", "fp8_e4m3", "all"], default="all")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--python", default=sys.executable)
    args = ap.parse_args()

    specs = iter_run_specs()
    if args.dtype != "all":
        specs = [s for s in specs if s.dtype == args.dtype]

    manifest = write_run_manifest()
    print(f"manifest: {manifest}")
    print(f"training {len(specs)} runs")
    for i, spec in enumerate(specs, 1):
        print(f"[{i}/{len(specs)}] {spec.run_id}")
        train_run(spec, python=args.python, force=args.force)
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
