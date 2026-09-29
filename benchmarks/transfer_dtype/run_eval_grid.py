#!/usr/bin/env python3
"""Evaluate all trained dtype-transfer study runs (or a filtered subset)."""

from __future__ import annotations

import argparse
import sys

from repo_paths import BENCHMARKS, SRC

sys.path.insert(0, str(SRC))
sys.path.insert(0, str(BENCHMARKS))
from transfer_dtype.eval_one import (  # noqa: E402
    artifact_ok,
    bench_nvmmh,
    ensure_nvmmh,
    eval_run,
    run_rows_present,
)
from transfer_dtype.study_common import (  # noqa: E402
    eval_shapes_path,
    iter_run_specs,
    write_run_manifest,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dtype", choices=["fp32", "fp8_e4m3", "all"], default="all")
    ap.add_argument("--run-id", action="append", default=[], help="repeatable run_id filter")
    ap.add_argument("--array-index", type=int, default=None, help="SLURM array task id")
    ap.add_argument("--phase", default="plan,compile,bench")
    ap.add_argument("--gpu", default="H100_SXM")
    ap.add_argument("--compile-jobs", type=int, default=64)
    ap.add_argument("--bench-workers", type=int, default=4)
    ap.add_argument("--force-propose", action="store_true")
    ap.add_argument("--skip-missing", action="store_true")
    ap.add_argument("--prep-only", action="store_true", help="shapes + nvmmh proposals only")
    ap.add_argument("--nvmmh-only", action="store_true", help="benchmark nvmmh baseline only")
    ap.add_argument("--skip-done", action="store_true", help="skip runs already present in eval DB")
    ap.add_argument("--force-nvmmh", action="store_true", help="re-benchmark nvmmh even if DB rows exist")
    ap.add_argument("--python", default=sys.executable)
    args = ap.parse_args()

    manifest = write_run_manifest()
    specs = iter_run_specs()
    if args.dtype != "all":
        specs = [s for s in specs if s.dtype == args.dtype]
    if args.run_id:
        wanted = set(args.run_id)
        specs = [s for s in specs if s.run_id in wanted]
    if args.array_index is not None:
        specs = [specs[args.array_index]]

    dtypes = sorted({s.dtype for s in specs})
    for dtype in dtypes:
        if not eval_shapes_path(dtype).is_file():
            raise FileNotFoundError(f"missing {eval_shapes_path(dtype)} — run gen_eval_shapes.py")
        ensure_nvmmh(dtype, args.python, args.gpu, force=args.force_propose)

    if args.prep_only:
        print(f"prep complete ({manifest})")
        return 0

    for dtype in dtypes:
        bench_nvmmh(
            dtype,
            python=args.python,
            gpu=args.gpu,
            phase=args.phase,
            compile_jobs=args.compile_jobs,
            bench_workers=args.bench_workers,
            force_propose=args.force_propose,
            force_bench=args.force_nvmmh,
        )

    if args.nvmmh_only:
        print("nvmmh bench complete")
        return 0

    print(f"evaluating {len(specs)} runs")
    for i, spec in enumerate(specs, 1):
        if args.skip_done and run_rows_present(spec):
            print(f"[{i}/{len(specs)}] skip {spec.run_id}: already in eval DB")
            continue
        try:
            artifact_ok(spec)
        except FileNotFoundError as exc:
            if args.skip_missing:
                print(f"[{i}/{len(specs)}] skip {spec.run_id}: {exc}")
                continue
            raise
        print(f"[{i}/{len(specs)}] {spec.run_id}")
        eval_run(
            spec,
            python=args.python,
            gpu=args.gpu,
            phase=args.phase,
            compile_jobs=args.compile_jobs,
            bench_workers=args.bench_workers,
            force_propose=args.force_propose,
        )
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
