#!/usr/bin/env python3
"""Propose and benchmark one fusion-transfer study run."""

from __future__ import annotations

import argparse
import sqlite3
import subprocess
import sys
from pathlib import Path

from repo_paths import BENCHMARKS, EVAL_ARTIFACTS, SRC

UNFUSED_TEMPLATE = "hopper_template.cu.j2"
FUSION_TEMPLATE = "hopper_template_fusion.cu.j2"

sys.path.insert(0, str(SRC))
sys.path.insert(0, str(BENCHMARKS))
from transfer_fusion.study_common import (  # noqa: E402
    RunSpec,
    eval_db_home_path,
    eval_db_live_path,
    eval_db_shard_path,
    eval_nvmmh_shapes_path,
    eval_shapes_path,
    iter_run_specs,
    proposals_model_path,
    proposals_nvmmh_path,
)


def artifact_ok(spec: RunSpec) -> Path:
    if spec.family == "mlp":
        for name in ("model_mlp.pt2", "model_mlp.pt"):
            p = spec.out_dir / name
            if p.is_file():
                return p
        raise FileNotFoundError(f"no MLP artifact in {spec.out_dir}")
    p = spec.out_dir / "model_A.ubj"
    if not p.is_file():
        raise FileNotFoundError(p)
    return p


def _eval_run_count(db: Path, where: str, params: tuple = ()) -> int:
    if not db.is_file():
        return 0
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        has_table = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='eval_runs'"
        ).fetchone()
        if not has_table:
            return 0
        return conn.execute(f"SELECT COUNT(*) FROM eval_runs WHERE {where}", params).fetchone()[0]
    except sqlite3.OperationalError:
        return 0
    finally:
        conn.close()


def nvmmh_rows_present(dtype: str) -> bool:
    return _eval_run_count(eval_db_home_path(dtype), "method='nvmmh'") > 0


def run_rows_present(spec: RunSpec) -> bool:
    shard = eval_db_shard_path(spec)
    if _eval_run_count(shard, "method=?", (spec.run_id,)) > 0:
        return True
    return _eval_run_count(eval_db_home_path(spec.dtype), "method=?", (spec.run_id,)) > 0


def ensure_nvmmh(dtype: str, python: str, gpu: str, force: bool) -> Path:
    out = proposals_nvmmh_path(dtype)
    if out.is_file() and not force:
        return out
    shapes = eval_nvmmh_shapes_path(dtype)
    cmd = [
        python, str(SRC / "eval" / "propose_fusion.py"),
        "--method", "nvmmh",
        "--dtype", dtype,
        "--layouts", "TN",
        "--gpu", gpu,
        "--shapes", str(shapes),
        "--out", str(out),
    ]
    print(" ".join(cmd))
    subprocess.check_call(cmd)
    return out


def propose_model(spec: RunSpec, python: str, gpu: str, force: bool) -> Path:
    out = proposals_model_path(spec)
    if out.is_file() and not force:
        return out
    artifact_ok(spec)
    nvmmh = ensure_nvmmh(spec.dtype, python, gpu, force=False)
    cmd = [
        python, str(SRC / "eval" / "propose_fusion.py"),
        "--method", spec.run_id,
        "--backend", spec.family,
        "--model-dir", str(spec.out_dir),
        "--dtype", spec.dtype,
        "--layouts", "TN",
        "--scheduler-from", str(nvmmh),
        "--shapes", str(eval_shapes_path(spec.dtype)),
        "--out", str(out),
    ]
    print(" ".join(cmd))
    subprocess.check_call(cmd)
    return out


def _run_proposals(
    dtype: str,
    method: str,
    proposals: Path,
    python: str,
    phase: str,
    compile_jobs: int,
    bench_workers: int,
    db: Path,
    *,
    template: str | None = None,
) -> None:
    db.parent.mkdir(parents=True, exist_ok=True)
    build_dir = EVAL_ARTIFACTS / "build" / f"transfer_fusion_{dtype}" / method
    cmd = [
        python, "-u", str(SRC / "eval" / "run.py"),
        "--phase", phase,
        "--db", str(db),
        "--build-dir", str(build_dir),
        "--proposals", str(proposals),
        "--compile-jobs", str(compile_jobs),
        "--bench-workers", str(bench_workers),
    ]
    if template is not None:
        cmd.extend(["--template", template])
    print(" ".join(cmd))
    subprocess.check_call(cmd)


def bench_nvmmh(
    dtype: str,
    python: str,
    gpu: str,
    phase: str = "plan,compile,bench",
    compile_jobs: int = 64,
    bench_workers: int = 4,
    force_propose: bool = False,
    force_bench: bool = False,
) -> Path:
    prop = ensure_nvmmh(dtype, python, gpu, force=force_propose)
    if not force_bench and nvmmh_rows_present(dtype):
        print(f"nvmmh already benchmarked in {eval_db_home_path(dtype)}")
        return prop
    # nvMMH is unfused — fusion template + NoSmem epilogues do not compile.
    _run_proposals(
        dtype, "nvmmh", prop, python, phase, compile_jobs, bench_workers,
        eval_db_live_path(dtype), template=UNFUSED_TEMPLATE,
    )
    return prop


def run_eval(spec: RunSpec, python: str, phase: str, compile_jobs: int, bench_workers: int) -> None:
    _run_proposals(
        spec.dtype,
        spec.run_id,
        proposals_model_path(spec),
        python,
        phase,
        compile_jobs,
        bench_workers,
        eval_db_shard_path(spec),
        template=FUSION_TEMPLATE,
    )


def eval_run(
    spec: RunSpec,
    python: str = sys.executable,
    gpu: str = "H100_SXM",
    phase: str = "plan,compile,bench",
    compile_jobs: int = 64,
    bench_workers: int = 4,
    force_propose: bool = False,
    skip_bench: bool = False,
) -> None:
    if not eval_shapes_path(spec.dtype).is_file():
        raise FileNotFoundError(f"missing eval shapes for {spec.dtype}")
    propose_model(spec, python, gpu, force=force_propose)
    if skip_bench:
        run_eval(spec, python, "plan", compile_jobs, bench_workers)
        return
    run_eval(spec, python, phase, compile_jobs, bench_workers)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--phase", default="plan,compile,bench")
    ap.add_argument("--gpu", default="H100_SXM")
    ap.add_argument("--compile-jobs", type=int, default=64)
    ap.add_argument("--bench-workers", type=int, default=4)
    ap.add_argument("--force-propose", action="store_true")
    ap.add_argument("--skip-bench", action="store_true")
    ap.add_argument("--python", default=sys.executable)
    args = ap.parse_args()

    for spec in iter_run_specs():
        if spec.run_id == args.run_id:
            eval_run(
                spec,
                python=args.python,
                gpu=args.gpu,
                phase=args.phase,
                compile_jobs=args.compile_jobs,
                bench_workers=args.bench_workers,
                force_propose=args.force_propose,
                skip_bench=args.skip_bench,
            )
            return 0
    raise SystemExit(f"unknown run_id {args.run_id}")


if __name__ == "__main__":
    raise SystemExit(main())
