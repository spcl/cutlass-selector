#!/usr/bin/env python3
"""Orchestrate the MLP inference GEMM case study."""

from __future__ import annotations

import argparse
import subprocess
import sys

from repo_paths import BENCHMARKS


def _run(script: str, extra: list[str] | None = None) -> None:
    cmd = [sys.executable, str(BENCHMARKS / "mlp_case_study" / script)] + (extra or [])
    print(" ".join(cmd), flush=True)
    subprocess.check_call(cmd)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="trace,select,bench,analyze",
                    help="comma-separated: trace,select,bench,analyze")
    ap.add_argument("--model", default=None, help="single model_id for trace/select")
    ap.add_argument("--bench-workers", type=int, default=1)
    ap.add_argument("--compile-jobs", type=int, default=32)
    ap.add_argument("--bench-phase", default="plan,compile,bench",
                    help="phases passed to benchmark_gemms.py")
    ap.add_argument("--fresh-db", action="store_true")
    ap.add_argument("--skip-bench", action="store_true")
    args = ap.parse_args()

    phases = set(args.phase.split(","))
    extra = ["--model", args.model] if args.model else []

    if "trace" in phases:
        _run("trace_gemms.py", extra)
    if "select" in phases:
        _run("select_kernels.py")
    if "bench" in phases and not args.skip_bench:
        bench_args = [
            "--bench-workers", str(args.bench_workers),
            "--compile-jobs", str(args.compile_jobs),
            "--phase", args.bench_phase,
        ]
        if args.fresh_db:
            bench_args.append("--fresh-db")
        _run("benchmark_gemms.py", bench_args)
    if "analyze" in phases:
        if args.skip_bench and "bench" not in phases:
            print("WARNING: analyze without bench requires existing per_gemm_results.csv")
        _run("analyze.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
