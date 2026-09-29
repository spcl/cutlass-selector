#!/usr/bin/env python3
"""Basic transfer-study curves (legacy PNG entry point)."""

from __future__ import annotations

import sys

from repo_paths import BENCHMARKS, SRC

sys.path.insert(0, str(SRC))
sys.path.insert(0, str(BENCHMARKS))
from transfer_dtype.plot_study import plot_all  # noqa: E402
from transfer_dtype.study_common import PLOTS_ROOT, RESULTS_ROOT  # noqa: E402


def main() -> int:
    plot_all(
        RESULTS_ROOT / "curve_points.csv",
        RESULTS_ROOT / "run_summary.csv",
        PLOTS_ROOT,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
