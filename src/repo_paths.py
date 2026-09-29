"""Canonical repository paths.

Layout:
  <repo>/
    src/         — library code (this file lives here)
    benchmarks/  — cluster drivers and the experiments behind the paper
    artifacts/   — generated outputs (models, figures, eval runs, …); local only
    datasets/    — inputs (shape lists, DeepBench catalogue, …); local only

Import with ``src/`` on ``PYTHONPATH``::

    from repo_paths import SRC, ARTIFACTS, PAPER
"""

from __future__ import annotations

from pathlib import Path

SRC = Path(__file__).resolve().parent
REPO_ROOT = SRC.parent
BENCHMARKS = REPO_ROOT / "benchmarks"
ARTIFACTS = REPO_ROOT / "artifacts"
DATASETS = REPO_ROOT / "datasets"

ANALYSIS = ARTIFACTS / "analysis"
PAPER = ANALYSIS / "paper"
CAPACITY = ANALYSIS / "capacity"
SAMPLING = ANALYSIS / "sampling"
FIGURES = PAPER / "figures"
BASELINES = PAPER / "baselines"

EVAL_ARTIFACTS = ARTIFACTS / "eval"
EVAL_OUT = EVAL_ARTIFACTS / "out"

PLANS = DATASETS / "plans"
DEEPBENCH_DATA = DATASETS / "deepbench"
