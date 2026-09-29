#!/usr/bin/env python3
"""
cublas.py — Run the cuBLASLt baseline over the evaluation shapes and store the results
alongside the CUTLASS measurements in eval.db.

The profiler binary reports every cuBLASLt heuristic candidate (algo 0 = its own top pick),
each measured with the same protocol as the CUTLASS side. That makes the two comparable and
leaves the reduction to report.py: algo 0 is the pure-heuristic number, and the minimum over
the first b candidates is cuBLASLt under a measurement budget of b.

The timing protocol is not duplicated here — warmup / rounds / iters-per-round are read from
src/autotuner/profile_worker.py and passed to the binary, so the two harnesses cannot drift apart.

Resumable: (M, N, K, layout) pairs already stored are skipped.

    python src/eval/cublas.py --db artifacts/eval/out/eval.db
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from repo_paths import EVAL_OUT, SRC

sys.path.insert(0, str(SRC / "autotuner"))

import profile_worker as pw  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run import _connect  # noqa: E402
from throughput import tflops_to_gflops  # noqa: E402

DEFAULT_PROFILER = SRC / "baseline" / "cublas_lt" / "cublas_profiler"
DEFAULT_SHAPES = EVAL_OUT / "shapes.json"
DEFAULT_DB = EVAL_OUT / "eval.db"
LAYOUTS = ("tn", "tt", "nn", "nt")
CONFIG_KEY = "bf16_f32_bf16"

_CUBLAS_RUNS_DDL = """
CREATE TABLE IF NOT EXISTS cublas_runs (
    M            INTEGER NOT NULL,
    N            INTEGER NOT NULL,
    K            INTEGER NOT NULL,
    layout       TEXT    NOT NULL,
    config       TEXT    NOT NULL,
    algo         INTEGER NOT NULL,
    status       TEXT    NOT NULL,
    mean_ms      REAL,
    std_ms       REAL,
    mean_tflops  REAL,
    std_tflops   REAL,
    ran_at       TEXT,
    PRIMARY KEY (M, N, K, layout, algo)
);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_rows(stdout: str, config: str) -> list[tuple]:
    """CSV -> cublas_runs tuples. Columns after M,N,K are algo,mean_ms,std_ms,mean_tf,std_tf."""
    rows = []
    for line in stdout.splitlines():
        parts = line.strip().split(",")
        if len(parts) != 14 or parts[0] != config:
            continue
        layout = parts[1]
        M, N, K, algo = (int(x) for x in parts[6:10])
        if parts[10] == "UNSUPPORTED":
            rows.append((M, N, K, layout, config, algo, "unsupported", None, None, None, None, _now()))
        else:
            mean_ms, std_ms, mean_tf, std_tf = (float(x) for x in parts[10:14])
            rows.append((M, N, K, layout, config, algo, "success",
                         mean_ms, std_ms,
                         round(tflops_to_gflops(mean_tf), 3),
                         round(tflops_to_gflops(std_tf), 3),
                         _now()))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description="cuBLASLt baseline over the evaluation shapes")
    ap.add_argument("--shapes", type=Path, default=DEFAULT_SHAPES)
    ap.add_argument("--db", type=Path, default=DEFAULT_DB)
    ap.add_argument("--profiler", type=Path, default=DEFAULT_PROFILER)
    ap.add_argument("--config", default=CONFIG_KEY)
    ap.add_argument("--layouts", nargs="+", default=list(LAYOUTS))
    ap.add_argument("--batch", type=int, default=25, help="shapes per profiler invocation")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    if not args.profiler.exists():
        print(f"ERROR: {args.profiler} not built — run 'make' in baseline/cublas_lt/",
              file=sys.stderr)
        return 1

    shapes = [tuple(s) for s in json.loads(args.shapes.read_text())]
    if args.limit:
        shapes = shapes[: args.limit]

    args.db.parent.mkdir(parents=True, exist_ok=True)
    conn = _connect(args.db)
    conn.executescript(_CUBLAS_RUNS_DDL)
    conn.commit()

    protocol = ["--warmup", str(pw.WARMUP_ITERS),
                "--rounds", str(pw.NUM_ROUNDS),
                "--iters-per-round", str(pw.ITERS_PER_ROUND)]
    print(f"protocol from profile_worker: warmup={pw.WARMUP_ITERS} rounds={pw.NUM_ROUNDS} "
          f"iters_per_round={pw.ITERS_PER_ROUND}")

    for layout in args.layouts:
        done = {(m, n, k) for m, n, k in conn.execute(
            "SELECT DISTINCT M, N, K FROM cublas_runs WHERE layout = ?", (layout,))}
        todo = [s for s in shapes if s not in done]
        print(f"layout={layout}: {len(todo)} shapes to run ({len(done)} already stored)",
              flush=True)

        for i in range(0, len(todo), args.batch):
            batch = todo[i: i + args.batch]
            cmd = [str(args.profiler), args.config, "--layout", layout, *protocol]
            for (M, N, K) in batch:
                cmd += [str(M), str(N), str(K)]
            proc = subprocess.run(cmd, capture_output=True, text=True)
            if proc.returncode != 0:
                print(f"  profiler failed on batch {i // args.batch}: "
                      f"{proc.stderr.strip()[-300:]}", file=sys.stderr)
                continue
            rows = parse_rows(proc.stdout, args.config)
            conn.executemany(
                """INSERT OR REPLACE INTO cublas_runs
                   (M, N, K, layout, config, algo, status, mean_ms, std_ms,
                    mean_tflops, std_tflops, ran_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""", rows)
            conn.commit()
            print(f"  {min(i + args.batch, len(todo))}/{len(todo)} shapes, "
                  f"{len(rows)} rows", flush=True)

    counts = dict(conn.execute(
        "SELECT status, COUNT(*) FROM cublas_runs GROUP BY status").fetchall())
    n_problems = conn.execute(
        "SELECT COUNT(*) FROM (SELECT DISTINCT M,N,K,layout FROM cublas_runs)").fetchone()[0]
    print(f"cublas_runs: {n_problems} problems, rows by status: {counts}")
    conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
