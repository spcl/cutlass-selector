#!/usr/bin/env python3
"""
shapes.py — Generate the held-out (M, N, K) problem set for the evaluation.

ALIGNMENT. CUTLASS SM90 TMA requires the *contiguous* dimension of each tensor to be a
multiple of 128 bits — 8 elements for bf16. The check is in sm90_mma_*.hpp::can_implement()
and ignores the config's declared AlignmentA/B. Our D operand is always ColumnMajor, so M is
always constrained; each layout adds one more:

    tn:  A RowMajor(K)  B ColMajor(K)  ->  K%8, M%8
    tt:  A RowMajor(K)  B RowMajor(N)  ->  K%8, N%8, M%8
    nn:  A ColMajor(M)  B ColMajor(K)  ->  M%8, K%8
    nt:  A ColMajor(M)  B RowMajor(N)  ->  M%8, N%8

Every shape is evaluated under all 4 layouts, so the union applies: M, N and K must each be
a multiple of 8. Violating it makes can_implement() reject every kernel in the space.

TWO STRATA. The training sweep snaps essentially every dimension to a multiple of 32 (of its
593 shapes, 564 are fully 32-aligned and K is 32-aligned in all of them; the exceptions are
M=48 / N=48). So half the draw sits on the 32-grid (in distribution) and half on the 8-grid
excluding the 32-grid (off distribution), letting the headline comparison and the
generalization question be answered separately with equal power.

The stratum is a pure function of the shape and is not stored — recover it with stratum(),
so shapes.json stays a plain [[M, N, K], ...] list for every consumer.

HELD OUT. --exclude-db and --train-tags are required, with no defaults: pointing this at the
wrong database or tag silently produces an evaluation set overlapping the training data,
which is the one mistake that invalidates every number downstream.

    python src/eval/shapes.py --holdout build_cache/autotuner.db:sweep
"""

import argparse
import json
import sqlite3
import sys
from pathlib import Path

from repo_paths import EVAL_OUT


def write_json_atomic(path: Path, payload) -> None:
    """Write JSON through a temporary file and rename, so readers never see a partial file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload))
    tmp.replace(path)

import numpy as np

MIN_DIM, MAX_DIM = 32, 16384
ALIGN = 8       # bf16 TMA: 128 bits / 16 bits per element
GRID = 32       # the training sweep's dimension grid

DEFAULT_OUT = EVAL_OUT / "shapes.json"


def stratum(m: int, n: int, k: int) -> str:
    """aligned32 if every dimension is a multiple of 32 (the training grid), else ragged8."""
    return "aligned32" if (m % GRID == 0 and n % GRID == 0 and k % GRID == 0) else "ragged8"


def _snap(v: float, base: int) -> int:
    return int(max(MIN_DIM, min(MAX_DIM, round(v / base) * base)))


def training_shapes(db_path: Path, tags: list[str]) -> tuple[set, set]:
    """(planned under `tags`, benchmarked at all) — both are held out of the draw.

    Returned separately so the caller can tell an unmatched tag (empty `planned`) from a
    genuinely empty database; unioning them here would let a typo'd tag pass unnoticed.
    """
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        marks = ",".join("?" * len(tags))
        planned = {
            (m, n, k)
            for m, n, k in conn.execute(
                f"SELECT DISTINCT M, N, K FROM eval_plan WHERE tag IN ({marks})", tags
            )
        }
        benched = {(m, n, k) for m, n, k in conn.execute("SELECT DISTINCT M, N, K FROM runs")}
    finally:
        conn.close()
    return planned, benched


def _draw(rng, n: int, base: int, exclude_grid_aligned: bool,
          taken: set) -> list[tuple[int, int, int]]:
    out: list[tuple[int, int, int]] = []
    while len(out) < n:
        raw = np.exp(rng.uniform(np.log(MIN_DIM), np.log(MAX_DIM),
                                 size=(max(n - len(out), 64) * 4, 3)))
        for row in raw:
            shape = tuple(_snap(v, base) for v in row)
            if exclude_grid_aligned and stratum(*shape) == "aligned32":
                continue
            if shape in taken:
                continue
            taken.add(shape)
            out.append(shape)
            if len(out) == n:
                break
    return out


def gen_shapes(n: int, seed: int, exclude: set[tuple[int, int, int]]) -> list[tuple[int, int, int]]:
    """Draw n held-out shapes, half aligned32 and half ragged8, none of them in exclude."""
    rng = np.random.default_rng(seed)
    taken = set(exclude)
    n_aligned = n // 2
    shapes = _draw(rng, n_aligned, GRID, False, taken)
    shapes += _draw(rng, n - n_aligned, ALIGN, True, taken)
    return sorted(shapes)


def _parse_holdout(spec: str) -> tuple[Path, list[str]]:
    if ":" not in spec:
        raise ValueError(f"holdout spec must be DB:TAG[,TAG...], got {spec!r}")
    db_s, tags_s = spec.split(":", 1)
    tags = [t.strip() for t in tags_s.split(",") if t.strip()]
    if not tags:
        raise ValueError(f"holdout spec needs at least one tag: {spec!r}")
    return Path(db_s), tags


def collect_holdout(holdouts: list[tuple[Path, list[str]]]) -> set[tuple[int, int, int]]:
    """Union of every shape planned under the tags or benchmarked in each holdout DB.

    Exits if a DB is missing or has no plan rows for its tags, since that would silently
    weaken the holdout.
    """
    exclude: set[tuple[int, int, int]] = set()
    for db_path, tags in holdouts:
        if not db_path.exists():
            print(f"ERROR: {db_path} not found — cannot verify the shapes are held out "
                  f"of training.", file=sys.stderr)
            raise SystemExit(1)
        planned, benched = training_shapes(db_path, tags)
        if not planned:
            print(f"ERROR: no eval_plan rows in {db_path} for tags {tags} — "
                  f"wrong database or tag?", file=sys.stderr)
            raise SystemExit(1)
        print(f"  holdout {db_path.name} tag={tags}: "
              f"{len(planned)} planned + {len(benched)} benched -> "
              f"{len(planned | benched)} unique shapes")
        exclude |= planned | benched
    return exclude


def main() -> int:
    ap = argparse.ArgumentParser(description="Held-out evaluation shapes")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--holdout", action="append", metavar="DB:TAG",
                    help="repeatable holdout source (eval_plan under TAG + all benched shapes)")
    ap.add_argument("--exclude-db", type=Path,
                    help="deprecated alias for a single --holdout (use with --train-tags)")
    ap.add_argument("--train-tags", nargs="+", metavar="TAG",
                    help="eval_plan tag(s) for --exclude-db")
    args = ap.parse_args()

    holdouts: list[tuple[Path, list[str]]] = []
    if args.holdout:
        for spec in args.holdout:
            holdouts.append(_parse_holdout(spec))
    if args.exclude_db is not None:
        if not args.train_tags:
            print("ERROR: --exclude-db requires --train-tags", file=sys.stderr)
            return 1
        holdouts.append((args.exclude_db, args.train_tags))
    if not holdouts:
        print("ERROR: pass at least one --holdout DB:TAG (or --exclude-db with --train-tags)",
              file=sys.stderr)
        return 1

    exclude = collect_holdout(holdouts)

    shapes = gen_shapes(args.n, args.seed, exclude)

    bad = [s for s in shapes if any(d % ALIGN for d in s)]
    assert not bad, f"{len(bad)} shapes violate the %{ALIGN} TMA requirement, e.g. {bad[:3]}"
    leaked = [s for s in shapes if s in exclude]
    assert not leaked, f"{len(leaked)} shapes appear in training, e.g. {leaked[:3]}"

    write_json_atomic(args.out, [list(s) for s in shapes])

    n_aligned = sum(1 for s in shapes if stratum(*s) == "aligned32")
    print(f"wrote {len(shapes)} shapes -> {args.out}")
    print(f"  held out from training ({len(exclude)} seen shapes): 0 overlap")
    print(f"  aligned32 (M,N,K all %{GRID}) : {n_aligned}")
    print(f"  ragged8   (all %{ALIGN}, not %{GRID}): {len(shapes) - n_aligned}")
    for axis, i in (("M", 0), ("N", 1), ("K", 2)):
        print(f"  {axis} range [{min(s[i] for s in shapes)}, {max(s[i] for s in shapes)}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
