#!/usr/bin/env python3
"""
Copy nvMMH rank-1 tile-scheduler args onto model proposals and (optionally) reset
benchmark rows so a bench-only rerun picks them up.

Scheduler fields are shared across nvMMH schedule variants of the same recommendation,
so any rank-1 nvMMH proposal per (M, N, K, layout) suffices.
"""

import argparse
import json
import sqlite3
import sys
from pathlib import Path

MODEL_METHODS = ("mlp_full", "mlp_structural", "xgb_full", "xgb_structural")


def sched_map_from_nvmmh(path: Path) -> dict[tuple[int, int, int, str], tuple[int, int, int]]:
    """Map (M, N, K, layout) to nvMMH rank-1 (raster_order, swizzle_size, splits) from a proposals file."""
    payload = json.loads(path.read_text())
    sched: dict[tuple[int, int, int, str], tuple[int, int, int]] = {}
    for p in payload["proposals"]:
        if p.get("rank") != 1:
            continue
        key = (p["M"], p["N"], p["K"], p["layout"])
        sched.setdefault(
            key,
            (p["raster_order"], p["swizzle_size"], p["splits"]),
        )
    if not sched:
        raise SystemExit(f"ERROR: no rank-1 proposals in {path}")
    return sched


def graft_proposal_file(path: Path, sched: dict[tuple[int, int, int, str], tuple[int, int, int]]) -> int:
    """Overwrite the scheduler arguments of the rank-1 model proposals in a file; return how many changed."""
    payload = json.loads(path.read_text())
    n = 0
    for p in payload["proposals"]:
        if p.get("rank") != 1:
            continue
        key = (p["M"], p["N"], p["K"], p["layout"])
        if key not in sched:
            raise SystemExit(f"ERROR: no nvMMH scheduler for {key} when grafting {path}")
        raster, swizzle, splits = sched[key]
        if (p["raster_order"], p["swizzle_size"], p["splits"]) != (raster, swizzle, splits):
            p["raster_order"] = raster
            p["swizzle_size"] = swizzle
            p["splits"] = splits
            n += 1
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n")
    tmp.replace(path)
    return n


def reset_eval_runs(db_path: Path, sched: dict[tuple[int, int, int, str], tuple[int, int, int]]) -> int:
    """Apply the grafted scheduler arguments to the model rows in eval_runs and reset them to pending.

    Returns the number of rows reset, so a bench-only rerun measures them again.
    """
    conn = sqlite3.connect(str(db_path), timeout=120)
    conn.execute("PRAGMA busy_timeout=120000;")
    placeholders = ",".join("?" * len(MODEL_METHODS))
    rows = conn.execute(
        f"""SELECT method, M, N, K, layout, rank, variant
            FROM eval_runs WHERE method IN ({placeholders}) AND rank = 1""",
        MODEL_METHODS,
    ).fetchall()
    updates = []
    for method, M, N, K, layout, rank, variant in rows:
        key = (M, N, K, layout)
        if key not in sched:
            raise SystemExit(f"ERROR: no nvMMH scheduler for {key} in DB reset")
        raster, swizzle, splits = sched[key]
        updates.append((raster, swizzle, splits, method, M, N, K, layout, rank, variant))
    conn.executemany(
        """UPDATE eval_runs
           SET raster_order = ?, swizzle_size = ?, splits = ?,
               status = 'pending',
               mean_ms = NULL, std_ms = NULL,
               mean_tflops = NULL, std_tflops = NULL,
               cutlass_status_code = NULL, cutlass_reason = NULL,
               error_text = NULL, ran_at = NULL
           WHERE method = ? AND M = ? AND N = ? AND K = ? AND layout = ?
             AND rank = ? AND variant = ?""",
        updates,
    )
    conn.commit()
    conn.close()
    return len(updates)


def main() -> int:
    ap = argparse.ArgumentParser(description="Graft nvMMH rank-1 scheduler onto model eval rows")
    ap.add_argument("--nvmmh", type=Path, help="proposals_nvmmh.json (default: <prep-dir>/proposals_nvmmh.json)")
    ap.add_argument("--prep-dir", type=Path, help="patch proposals_{mlp,xgb}_*.json in this directory")
    ap.add_argument("--db", type=Path, help="reset model eval_runs rows to pending with new scheduler args")
    args = ap.parse_args()

    if not args.prep_dir and not args.db:
        print("ERROR: pass --prep-dir and/or --db", file=sys.stderr)
        return 1

    nvmmh = args.nvmmh
    if nvmmh is None:
        if args.prep_dir is None:
            print("ERROR: --nvmmh required when --prep-dir is omitted", file=sys.stderr)
            return 1
        nvmmh = args.prep_dir / "proposals_nvmmh.json"
    if not nvmmh.is_file():
        print(f"ERROR: missing {nvmmh}", file=sys.stderr)
        return 1

    sched = sched_map_from_nvmmh(nvmmh)
    print(f"nvMMH sched map: {len(sched)} problems from {nvmmh.name}")

    if args.prep_dir:
        if not args.prep_dir.is_dir():
            print(f"ERROR: not a directory: {args.prep_dir}", file=sys.stderr)
            return 1
        for method in MODEL_METHODS:
            path = args.prep_dir / f"proposals_{method}.json"
            if not path.is_file():
                print(f"WARNING: skip missing {path.name}")
                continue
            n = graft_proposal_file(path, sched)
            print(f"  {path.name}: updated {n} proposals")

    if args.db:
        if not args.db.is_file():
            print(f"ERROR: missing DB {args.db}", file=sys.stderr)
            return 1
        n = reset_eval_runs(args.db, sched)
        print(f"DB {args.db}: reset {n} model rows to pending")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
