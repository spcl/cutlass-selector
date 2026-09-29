#!/usr/bin/env python3
"""Print run_id for SLURM array task index."""

from __future__ import annotations

import argparse
import sys

from repo_paths import BENCHMARKS, SRC

sys.path.insert(0, str(SRC))
sys.path.insert(0, str(BENCHMARKS))
from transfer_fusion.study_common import DTYPE_TAGS, iter_run_specs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("index", type=int)
    ap.add_argument("--dtype", choices=list(DTYPE_TAGS) + ["all"], default="all")
    ap.add_argument("--count-only", action="store_true")
    args = ap.parse_args()

    specs = iter_run_specs()
    if args.dtype != "all":
        specs = [s for s in specs if s.dtype == args.dtype]
    if args.count_only:
        print(len(specs))
        return 0
    if args.index < 0 or args.index >= len(specs):
        raise SystemExit(f"index {args.index} out of range [0, {len(specs)})")
    print(specs[args.index].run_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
