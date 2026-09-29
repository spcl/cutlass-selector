#!/usr/bin/env python3
"""Generate broad held-out TN evaluation shapes per target dtype."""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

import numpy as np

from repo_paths import BENCHMARKS, SRC

sys.path.insert(0, str(SRC / "eval"))
from shapes import GRID, _draw, stratum, write_json_atomic  # noqa: E402

sys.path.insert(0, str(SRC))
sys.path.insert(0, str(BENCHMARKS))
from transfer_dtype.study_common import (  # noqa: E402
    DTYPE_TAGS,
    EVAL_N_SHAPES,
    EVAL_SHAPE_SEED,
    STUDY_ROOT,
    eval_shapes_path,
    load_shapes,
    sweep_db_path,
)


def align_for_dtype(dtype: str) -> int:
    if dtype == "fp32":
        return 4
    if dtype == "fp8_e4m3":
        return 16
    raise ValueError(dtype)


def sweep_benched_shapes(db: Path) -> set[tuple[int, int, int]]:
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    rows = conn.execute(
        "SELECT DISTINCT M, N, K FROM runs WHERE status='success' AND mean_tflops > 0"
    ).fetchall()
    conn.close()
    return {tuple(r) for r in rows}


def gen_shapes_tn(n: int, seed: int, exclude: set[tuple[int, int, int]], align: int) -> list[tuple[int, int, int]]:
    rng = np.random.default_rng(seed)
    taken = set(exclude)
    n_aligned = n // 2
    shapes = _draw(rng, n_aligned, GRID, False, taken)
    shapes += _draw(rng, n - n_aligned, align, True, taken)
    return sorted(shapes)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--autotuner", type=Path, default=Path.home() / "autotuner")
    ap.add_argument("--n", type=int, default=EVAL_N_SHAPES)
    ap.add_argument("--seed", type=int, default=EVAL_SHAPE_SEED)
    ap.add_argument("--dtype", choices=["fp32", "fp8_e4m3", "all"], default="all")
    args = ap.parse_args()

    dtypes = list(DTYPE_TAGS) if args.dtype == "all" else [args.dtype]
    for dtype in dtypes:
        master = STUDY_ROOT / "dtype" / dtype / "training_shapes.json"
        if not master.is_file():
            raise SystemExit(f"missing {master} — run make_subsets.py first")
        train_shapes = set(load_shapes(master))
        db = sweep_db_path(dtype, args.autotuner)
        if not db.is_file():
            raise SystemExit(f"missing {db}")
        exclude = train_shapes | sweep_benched_shapes(db)
        align = align_for_dtype(dtype)
        shapes = gen_shapes_tn(args.n, args.seed, exclude, align)

        bad = [s for s in shapes if s[0] % align or s[2] % align]
        if bad:
            raise SystemExit(f"{dtype}: {len(bad)} TN shapes violate align={align}, e.g. {bad[:3]}")
        leaked = [s for s in shapes if s in exclude]
        if leaked:
            raise SystemExit(f"{dtype}: {len(leaked)} shapes overlap training/holdout, e.g. {leaked[:3]}")

        out = eval_shapes_path(dtype)
        meta = {
            "dtype": dtype,
            "layout": "TN",
            "n": len(shapes),
            "seed": args.seed,
            "align": align,
            "n_exclude": len(exclude),
            "n_train_shapes": len(train_shapes),
        }
        write_json_atomic(out, [list(s) for s in shapes])
        meta_path = out.with_suffix(".meta.json")
        meta_path.write_text(json.dumps(meta, indent=2) + "\n")

        n_aligned = sum(1 for s in shapes if stratum(*s) == "aligned32")
        print(f"{dtype}: wrote {len(shapes)} TN shapes -> {out}")
        print(f"  held out {len(exclude)} shapes ({len(train_shapes)} finetune base shapes)")
        print(f"  aligned32: {n_aligned}  ragged: {len(shapes) - n_aligned}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
