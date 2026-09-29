#!/usr/bin/env python3
"""Merge per-node shard SQLite DBs into one durable registry."""

from __future__ import annotations

import argparse
import shutil
import sqlite3
import sys
import tempfile
from pathlib import Path


def _merge_one_shard(conn: sqlite3.Connection, shard_path: Path, alias: str) -> None:
    uri = f"file:{shard_path.resolve()}?mode=ro"
    conn.execute(f"ATTACH DATABASE ? AS {alias}", (uri,))
    try:
        conn.execute(
            f"""
            INSERT OR REPLACE INTO main.configs
            SELECT * FROM {alias}.configs
            WHERE compile_status != 'pending' OR so_file IS NOT NULL
            """
        )
        conn.execute(f"INSERT OR REPLACE INTO main.runs SELECT * FROM {alias}.runs")
        conn.execute(f"INSERT OR REPLACE INTO main.ncu_runs SELECT * FROM {alias}.ncu_runs")
        conn.commit()
    finally:
        conn.execute(f"DETACH DATABASE {alias}")


def merge_shards(dest: Path, shard_paths: list[Path], work_dir: Path | None = None) -> int:
    """Merge per-node shard registries into dest and return the number of shards merged.

    Work happens on a local copy in work_dir (a temporary directory by default), and dest is
    replaced atomically at the end, so a slow or shared filesystem never sees a half-written DB.
    If dest does not exist yet, the first shard seeds it.
    """
    shard_paths = sorted(p for p in shard_paths if p.is_file())
    if not shard_paths:
        print("No shard DBs found — nothing to merge.", file=sys.stderr)
        return 0

    tmp_ctx = None
    if work_dir is None:
        tmp_ctx = tempfile.TemporaryDirectory(prefix="merge_shards_")
        work_dir = Path(tmp_ctx.name)
    else:
        work_dir.mkdir(parents=True, exist_ok=True)

    work_dest = work_dir / "merged.db"
    if dest.is_file():
        print(f"Copying {dest} → {work_dest}")
        shutil.copy2(dest, work_dest)
    else:
        print(f"Bootstrapping {work_dest} from {shard_paths[0]}")
        shutil.copy2(shard_paths[0], work_dest)
        shard_paths = shard_paths[1:]
        if not shard_paths:
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(work_dest, dest)
            return 1

    conn = sqlite3.connect(work_dest, timeout=120)
    conn.execute("PRAGMA busy_timeout=120000")
    conn.execute("PRAGMA journal_mode=DELETE")
    conn.execute("PRAGMA synchronous=NORMAL")

    merged = 0
    for i, shard_path in enumerate(shard_paths):
        local_shard = work_dir / f"shard_{i}.db"
        print(f"Merging {shard_path} → {work_dest}")
        shutil.copy2(shard_path, local_shard)
        _merge_one_shard(conn, local_shard, f"s{i}")
        merged += 1

    conn.close()
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp_dest = dest.with_suffix(dest.suffix + ".tmp")
    shutil.copy2(work_dest, tmp_dest)
    tmp_dest.replace(dest)
    print(f"Merged {merged} shard(s) into {dest}")

    if tmp_ctx is not None:
        tmp_ctx.cleanup()
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dest", type=Path, required=True, help="Merged output DB path")
    parser.add_argument(
        "--shards",
        type=Path,
        nargs="+",
        required=True,
        help="Shard DB paths (globs expanded by shell)",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=None,
        help="Local scratch for merge I/O (default: temp dir, use /dev/shm on cluster)",
    )
    args = parser.parse_args()
    n = merge_shards(args.dest, args.shards, args.work_dir)
    if n == 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
