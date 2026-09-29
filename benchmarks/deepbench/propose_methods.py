#!/usr/bin/env python3
"""Generate nvMMH + learned-method proposal JSONs for a DeepBench run."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from repo_paths import BENCHMARKS, CAPACITY, PAPER, SRC

sys.path.insert(0, str(BENCHMARKS / "deepbench"))

from capacity_models import discover_capacity_models, write_manifest  # noqa: E402
from methods import resolve_methods  # noqa: E402


def _run_propose(
    python: str,
    *,
    method: str,
    backend: str,
    model_dir: Path | None,
    problems: Path,
    out: Path,
    scheduler_from: Path,
    nvmmh_gpu: str,
    random_seed: int | None,
) -> None:
    cmd = [
        python,
        str(SRC / "eval" / "propose.py"),
        "--method",
        method,
        "--backend",
        backend,
        "--problems",
        str(problems),
        "--out",
        str(out),
    ]
    if model_dir is not None:
        cmd += ["--model-dir", str(model_dir)]
    if backend == "nvmmh":
        cmd += ["--gpu", nvmmh_gpu]
    elif scheduler_from.is_file():
        cmd += ["--scheduler-from", str(scheduler_from)]
    if backend == "random" and random_seed is not None:
        cmd += ["--random-seed", str(random_seed)]
    print(f"=== {method} ({backend}) ===", flush=True)
    subprocess.run(cmd, check=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--prep-dir", type=Path, required=True)
    ap.add_argument("--problems", type=Path, required=True)
    ap.add_argument("--paper-dir", type=Path, default=PAPER)
    ap.add_argument("--capacity-dir", type=Path, default=CAPACITY)
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--nvmmh-gpu", default="H100_SXM")
    ap.add_argument("--capacity-seed", type=int, default=42)
    ap.add_argument("--random-seed", type=int, default=42)
    ap.add_argument(
        "--methods",
        default="default",
        help="comma-separated method ids, or 'default' for paper baselines",
    )
    ap.add_argument(
        "--include-capacity",
        action="store_true",
        help="also propose for capacity-sweep MLP/XGB checkpoints",
    )
    ap.add_argument(
        "--capacity-only",
        action="store_true",
        help="with --include-capacity, skip paper baselines (capacity checkpoints only)",
    )
    ap.add_argument("--skip-nvmmh", action="store_true")
    args = ap.parse_args()

    prep_dir = args.prep_dir.resolve()
    prep_dir.mkdir(parents=True, exist_ok=True)
    problems = args.problems.resolve()
    scheduler_from = prep_dir / "proposals_nvmmh.json"

    if not args.skip_nvmmh:
        _run_propose(
            args.python,
            method="nvmmh",
            backend="nvmmh",
            model_dir=None,
            problems=problems,
            out=scheduler_from,
            scheduler_from=scheduler_from,
            nvmmh_gpu=args.nvmmh_gpu,
            random_seed=None,
        )
    elif not scheduler_from.is_file():
        print(f"ERROR: --skip-nvmmh but missing {scheduler_from}", file=sys.stderr)
        return 1

    if args.capacity_only:
        method_ids = [
            m.method_id
            for m in discover_capacity_models(
                args.capacity_dir, seed=args.capacity_seed, skip_paper_mlp_width=True
            )
        ]
        include_capacity = False
    else:
        raw = args.methods.strip()
        method_ids = None if raw in ("", "default") else [x.strip() for x in raw.split(",")]
        include_capacity = args.include_capacity

    methods = resolve_methods(
        method_ids,
        include_capacity=include_capacity,
        capacity_dir=args.capacity_dir,
        paper_dir=args.paper_dir,
        capacity_seed=args.capacity_seed,
        random_seed=args.random_seed,
    )

    if args.include_capacity or args.capacity_only:
        cap_models = discover_capacity_models(
            args.capacity_dir, seed=args.capacity_seed, skip_paper_mlp_width=True
        )
        write_manifest(cap_models, prep_dir / "capacity_models.json")

    for em in methods:
        if em.model_dir is not None and not em.model_dir.is_dir():
            print(f"ERROR: missing model dir for {em.method_id}: {em.model_dir}", file=sys.stderr)
            return 1
        if em.backend == "ridge" and not (em.model_dir / "model_ridge.json").is_file():
            print(
                f"ERROR: missing ridge weights for {em.method_id}: "
                f"{em.model_dir / 'model_ridge.json'}",
                file=sys.stderr,
            )
            return 1
        _run_propose(
            args.python,
            method=em.method_id,
            backend=em.backend,
            model_dir=em.model_dir,
            problems=problems,
            out=prep_dir / f"proposals_{em.method_id}.json",
            scheduler_from=scheduler_from,
            nvmmh_gpu=args.nvmmh_gpu,
            random_seed=em.random_seed,
        )

    manifest = {
        "methods": [
            {
                "method_id": em.method_id,
                "backend": em.backend,
                "model_dir": str(em.model_dir) if em.model_dir else None,
            }
            for em in methods
        ]
    }
    (prep_dir / "methods.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {len(methods)} method proposal files -> {prep_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
