#!/usr/bin/env python3
"""Build nested shape subsets from fusion sweep DBs (593 TN base shapes)."""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

from repo_paths import BENCHMARKS, SRC

sys.path.insert(0, str(SRC))
sys.path.insert(0, str(BENCHMARKS))
from transfer_fusion.study_common import (
    DTYPE_TAGS,
    FRACTIONS,
    STUDY_ROOT,
    nested_subset,
    save_shapes,
    seeds_for_fraction,
    sweep_db_path,
)


def load_training_shapes(db: Path, tag: str) -> list[tuple[int, int, int]]:
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    rows = conn.execute(
        """SELECT DISTINCT M, N, K FROM runs
           WHERE status='success' AND tag=? AND mean_tflops > 0
           ORDER BY M, N, K""",
        (tag,),
    ).fetchall()
    conn.close()
    if not rows:
        raise SystemExit(f"no shapes in {db} for tag {tag}")
    return [tuple(r) for r in rows]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--autotuner", type=Path, default=Path.home() / "autotuner")
    ap.add_argument("--dtype", choices=list(DTYPE_TAGS) + ["all"], default="all")
    args = ap.parse_args()

    dtypes = list(DTYPE_TAGS) if args.dtype == "all" else [args.dtype]
    for dtype in dtypes:
        tag, _ = DTYPE_TAGS[dtype]
        db = sweep_db_path(dtype, args.autotuner)
        if not db.is_file():
            print(f"WARNING: skip {dtype}, missing {db}")
            continue
        shapes = load_training_shapes(db, tag)
        master = STUDY_ROOT / "dtype" / dtype / "training_shapes.json"
        save_shapes(master, shapes, {"dtype": dtype, "tag": tag, "n": len(shapes)})
        print(f"{dtype}: {len(shapes)} training base shapes -> {master}")

        for fraction in FRACTIONS:
            for seed in seeds_for_fraction(fraction):
                sub = nested_subset(shapes, fraction, seed)
                out = STUDY_ROOT / "subsets" / dtype / f"seed{seed}" / f"f{int(round(fraction * 100)):03d}.json"
                save_shapes(
                    out,
                    sub,
                    {
                        "dtype": dtype,
                        "fraction": fraction,
                        "seed": seed,
                        "n_shapes": len(sub),
                        "n_base": len(shapes),
                    },
                )
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
