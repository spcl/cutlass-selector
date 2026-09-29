#!/usr/bin/env python3
"""
run.py — Plan, compile and benchmark the proposed kernels.

Three phases, independently runnable so GPU time is only spent on the last one:

  plan     read src/eval/propose.py's output, register every proposed kernel in `configs`,
           and insert one pending `eval_runs` row per proposal. CPU only.
  compile  nvcc every registered kernel; failures are bisected down to the individual
           kernel so a bad config is attributed precisely instead of poisoning a batch.
           Cross-compiles for sm_90a on any host.
  bench    measure each (kernel, shape, scheduler-args) job once and write the result to
           every proposal row that asked for it. REQUIRES SM90.

Everything below the phase driver is the autotuner's: SQLiteRegistry for the config
registry and compile bookkeeping, try_compile for nvcc, and profile_worker's benchmark /
_BufferPool / status decoding for the measurement, so the eval measures kernels exactly the
way the training sweep did.

`eval_runs` is keyed by (method, M, N, K, layout, rank, variant) — the provenance — while
benchmark *jobs* are keyed by (kernel, shape, raster, swizzle, splits). Those differ: two
methods proposing the same kernel for the same shape with the same scheduler args share one
measurement, but the same kernel with nvMMH's rasterization and with CUTLASS defaults is two
distinct measurements, which is why the scheduler args are part of the job key.

    python src/eval/run.py --phase plan,compile --proposals artifacts/eval/out/proposals_*.json
    python src/eval/run.py --phase bench
"""

import argparse
import json
import logging
import multiprocessing
import sqlite3
import statistics
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from repo_paths import EVAL_ARTIFACTS, EVAL_OUT, SRC

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SRC / "autotuner"))

import profile_worker as pw  # noqa: E402
from jinja2 import Environment, FileSystemLoader  # noqa: E402
from registry import SQLiteRegistry  # noqa: E402
from scheduler import _family_key, try_compile  # noqa: E402
from throughput import FLOPS_UNIT_SHORT, calculate_gflops  # noqa: E402

logger = logging.getLogger("eval")

DEFAULT_DB = EVAL_OUT / "eval.db"
DEFAULT_BUILD_DIR = EVAL_ARTIFACTS / "build"

# Generous next to the autotuner's 10s: our shapes reach 16384 per dimension, where CUTLASS's
# own workspace allocation can legitimately outlast a quick sweep step.
BENCH_WORKER_TIMEOUT_S = 30

_EVAL_RUNS_DDL = """
CREATE TABLE IF NOT EXISTS eval_runs (
    method              TEXT    NOT NULL,
    M                   INTEGER NOT NULL,
    N                   INTEGER NOT NULL,
    K                   INTEGER NOT NULL,
    layout              TEXT    NOT NULL,
    rank                INTEGER NOT NULL,
    variant             INTEGER NOT NULL,
    config_name         TEXT    NOT NULL,
    raster_order        INTEGER NOT NULL,
    swizzle_size        INTEGER NOT NULL,
    splits              INTEGER NOT NULL,
    score               REAL,
    status              TEXT    NOT NULL DEFAULT 'pending',
    cutlass_status_code INTEGER,
    cutlass_reason      TEXT,
    error_text          TEXT,
    mean_ms             REAL,
    std_ms              REAL,
    mean_tflops         REAL,  -- GFLOP/s (legacy column name)
    std_tflops          REAL,  -- GFLOP/s
    ran_at              TEXT,
    PRIMARY KEY (method, M, N, K, layout, rank, variant)
);
CREATE INDEX IF NOT EXISTS idx_eval_runs_status ON eval_runs(status);
CREATE INDEX IF NOT EXISTS idx_eval_runs_config ON eval_runs(config_name);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _connect(db_path: Path) -> sqlite3.Connection:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), timeout=120)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA journal_mode=WAL;")
    except sqlite3.OperationalError:
        # NFS/Lustre home dirs often reject WAL ("locking protocol").
        conn.execute("PRAGMA journal_mode=DELETE;")
    conn.execute("PRAGMA busy_timeout=120000;")
    conn.executescript(_EVAL_RUNS_DDL)
    conn.commit()
    return conn


def _chunks(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


def _format_compile_error(err: str | None, max_chars: int = 2400) -> str:
    """nvcc puts the diagnosis first; don't log only the template-instantiation tail."""
    if not err:
        return "compile failed (no stderr)"
    text = err.strip()
    if not text:
        return "compile failed (empty stderr)"
    lines = text.splitlines()
    key = [
        ln for ln in lines
        if "error:" in ln.lower()
        or "static assertion" in ln.lower()
        or ln.strip().startswith("note:")
    ]
    if key:
        summary = "\n".join(key[:12])
        if len(summary) <= max_chars:
            return summary
        return summary[:max_chars] + "\n... [truncated]"
    if len(text) <= max_chars:
        return text
    half = max_chars // 2
    return text[:half] + "\n... [truncated] ...\n" + text[-half:]


def _write_rows(conn, pks, *, status, mean_ms=None, std_ms=None, mean_tflops=None,
                std_tflops=None, cutlass_status_code=None, cutlass_reason=None, error_text=None):
    conn.executemany(
        """UPDATE eval_runs SET status=?, mean_ms=?, std_ms=?, mean_tflops=?, std_tflops=?,
               cutlass_status_code=?, cutlass_reason=?, error_text=?, ran_at=?
           WHERE method=? AND M=? AND N=? AND K=? AND layout=? AND rank=? AND variant=?""",
        [(status, mean_ms, std_ms, mean_tflops, std_tflops, cutlass_status_code,
          cutlass_reason, error_text, _now(), *pk) for pk in pks],
    )
    conn.commit()


def _sync_compile_failures(conn) -> int:
    cur = conn.execute(
        """UPDATE eval_runs
           SET status='compile_failed',
               error_text=(SELECT compile_error FROM configs WHERE configs.name = eval_runs.config_name),
               ran_at=?
           WHERE status='pending'
             AND config_name IN (SELECT name FROM configs WHERE compile_status='failed')""",
        (_now(),),
    )
    conn.commit()
    return cur.rowcount


# ── plan ──────────────────────────────────────────────────────────────────────

def phase_plan(conn, registry, args) -> None:
    """Register every proposed config and write one pending eval_runs row per proposal (idempotent)."""
    configs: dict[str, dict] = {}
    rows: list[tuple] = []

    for path in args.proposals:
        payload = json.loads(Path(path).read_text())
        configs.update(payload["configs"])
        for p in payload["proposals"]:
            rows.append((
                p["method"], p["M"], p["N"], p["K"], p["layout"], p["rank"], p["variant"],
                p["config_name"], p["raster_order"], p["swizzle_size"], p["splits"], p["score"],
            ))
        logger.info("plan: %s -> %d proposals", Path(path).name, len(payload["proposals"]))

    registry.register(list(configs.values()))
    conn.executemany(
        """INSERT OR IGNORE INTO eval_runs
           (method, M, N, K, layout, rank, variant, config_name,
            raster_order, swizzle_size, splits, score)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
        rows,
    )
    conn.commit()

    n_rows = conn.execute("SELECT COUNT(*) FROM eval_runs").fetchone()[0]
    logger.info("plan: %d proposals over %d unique kernels; eval_runs holds %d rows",
                len(rows), len(configs), n_rows)


# ── compile ───────────────────────────────────────────────────────────────────

def _compile_isolating(registry, configs, args, template) -> tuple[int, int]:
    """Compile a batch; on failure bisect so the error lands on the exact kernel."""
    if not configs:
        return 0, 0
    ok, err, so = try_compile(configs, args.build_dir, args.cutlass_dir, template)
    if ok:
        for kernel_id, cfg in enumerate(configs):
            registry.record_compile_success(cfg["name"], so.name, kernel_id)
        return len(configs), 0
    if len(configs) == 1:
        registry.record_compile_failure(configs[0]["name"], err or "compile failed")
        logger.warning("compile FAILED: %s\n%s", configs[0]["name"], _format_compile_error(err))
        return 0, 1
    mid = len(configs) // 2
    a = _compile_isolating(registry, configs[:mid], args, template)
    b = _compile_isolating(registry, configs[mid:], args, template)
    return a[0] + b[0], a[1] + b[1]


def _enrich_for_template(configs: list[dict], template_name: str) -> list[dict]:
    if "fusion" not in template_name:
        return configs
    from config_space_fusion import enrich_fusion_from_name

    return [enrich_fusion_from_name(dict(c)) for c in configs]


def phase_compile(conn, registry, args) -> None:
    """Compile all registered configs that are not built yet (re-queues configs whose .so is missing)."""
    template_name = getattr(args, "template", "hopper_template.cu.j2")
    template = Environment(
        loader=FileSystemLoader(str(args.template_dir))
    ).get_template(template_name)

    n_requeued = registry.requeue_missing_so_files(args.build_dir)
    if n_requeued:
        logger.info("compile: requeued %d configs with missing .so in %s",
                    n_requeued, args.build_dir)

    pending = _enrich_for_template(registry.pending_compile(), template_name)
    if not pending:
        logger.info("compile: nothing pending")
        _sync_compile_failures(conn)
        return

    groups: dict[tuple, list] = defaultdict(list)
    for cfg in pending:
        groups[_family_key(cfg)].append(cfg)
    batches = [b for cfgs in groups.values() for b in _chunks(cfgs, args.max_batch_size)]
    logger.info("compile: %d kernels in %d batches, %d nvcc jobs",
                len(pending), len(batches), args.compile_jobs)

    counts = {"ok": 0, "fail": 0}
    lock = threading.Lock()

    def _run(i: int) -> None:
        ok, fail = _compile_isolating(registry, batches[i], args, template)
        with lock:
            counts["ok"] += ok
            counts["fail"] += fail
            logger.info("compile: [%d/%d] %d ok, %d failed",
                        i + 1, len(batches), counts["ok"], counts["fail"])

    with ThreadPoolExecutor(max_workers=args.compile_jobs) as pool:
        list(pool.map(_run, range(len(batches))))

    n_marked = _sync_compile_failures(conn)
    logger.info("compile: done — %d ok, %d failed (%d eval rows marked compile_failed)",
                counts["ok"], counts["fail"], n_marked)


# ── bench ─────────────────────────────────────────────────────────────────────

def _bench_worker(gpu_id: int, job_q, build_dir: Path, db_path: Path) -> None:
    sys.stdout.reconfigure(line_buffering=True)
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(message)s",
                        datefmt="%H:%M:%S", stream=sys.stdout)
    import torch

    torch.cuda.set_device(gpu_id)
    conn = _connect(db_path)
    registry = SQLiteRegistry(db_path)
    buf_pool = pw._BufferPool(gpu_id)
    lib_cache: dict[str, object] = {}

    while True:
        item = job_q.get()
        if item is None:
            job_q.task_done()
            break

        key, pks, dtypes, label = item
        M, N, K, so_file, kernel_id, raster, swizzle, splits = key

        registry.set_worker_state(gpu_id, json.dumps({"key": list(key), "pks": pks}))

        so_path = build_dir / so_file
        if not so_path.is_file():
            _write_rows(conn, pks, status="compile_failed",
                        error_text=f"missing {so_path}")
            logger.warning("[GPU %d] missing .so %s — skipped %d row(s)",
                           gpu_id, so_path, len(pks))
            registry.clear_worker_state(gpu_id)
            job_q.task_done()
            continue

        if so_file not in lib_cache:
            lib_cache[so_file] = pw.load_lib(so_path)
        lib = lib_cache[so_file]

        dtype_a, dtype_b, dtype_c = (pw.CUTLASS_TO_TORCH[t] for t in dtypes)
        As, Bs, D = buf_pool.get(M, N, K, dtype_a, dtype_b, dtype_c)

        _write_rows(conn, pks, status="running")
        status, latencies = pw.benchmark(lib, kernel_id, As, Bs, D, M, N, K,
                                         raster_order=raster, swizzle_size=swizzle, splits=splits)

        if latencies is not None:
            gflops = [calculate_gflops(M, N, K, ms) for ms in latencies]
            mean_gf = statistics.mean(gflops)
            _write_rows(conn, pks, status="success",
                        mean_ms=round(statistics.mean(latencies), 4),
                        std_ms=round(statistics.stdev(latencies), 4),
                        mean_tflops=round(mean_gf, 3),
                        std_tflops=round(statistics.stdev(gflops), 3))
            outcome = f"{mean_gf:.1f} {FLOPS_UNIT_SHORT}"
        else:
            reason = pw.CUTLASS_STATUS.get(status, f"Unknown Status Code {status}")
            _write_rows(conn, pks, status="rejected",
                        cutlass_status_code=status, cutlass_reason=reason)
            outcome = f"REJECTED ({status})"

        logger.info("[GPU %d] [%dx%dx%d %s] r=%d sw=%d sp=%d -> %s",
                    gpu_id, M, N, K, label, raster, swizzle, splits, outcome)
        registry.clear_worker_state(gpu_id)
        job_q.task_done()

    conn.close()


def _bench_monitor(worker_procs, job_q, registry, build_dir, db_path, stop_evt) -> None:
    """Restart crashed or hung bench workers and account for the job they died on.

    Each queue item is one atomic job, so a dead worker's rows are simply marked and the
    job dropped — no re-queueing of a partially finished config, unlike the autotuner's
    monitor, which resumes a config's remaining shapes.
    """
    conn = _connect(db_path)
    while not stop_evt.is_set():
        now = time.time()
        for i, proc in enumerate(worker_procs):
            try:
                state, updated_at = registry.get_worker_state_and_time(i)
                elapsed = (now - datetime.fromisoformat(updated_at).timestamp()) if updated_at else 0.0
                timed_out = proc.is_alive() and bool(state) and elapsed > BENCH_WORKER_TIMEOUT_S

                if not timed_out and (proc.is_alive() or proc.exitcode == 0):
                    continue

                if timed_out:
                    logger.warning("[Monitor] worker %d timed out after %.0fs (pid=%s) — killing",
                                   i, elapsed, proc.pid)
                    proc.kill()
                    proc.join(timeout=5)

                if state:
                    info = json.loads(state)
                    pks = [tuple(pk) for pk in info["pks"]]
                    if timed_out:
                        _write_rows(conn, pks, status="hung", error_text="worker timed out")
                    else:
                        _write_rows(conn, pks, status="crashed",
                                    error_text=f"worker exited {proc.exitcode}")
                    logger.warning("[Monitor] worker %d %s on %s — marked %d row(s)",
                                   i, "hung" if timed_out else "crashed", info["key"], len(pks))
                    try:
                        job_q.task_done()
                    except ValueError:
                        logger.warning("[Monitor] worker %d: task_done already balanced", i)
                    registry.clear_worker_state(i)
                else:
                    logger.warning("[Monitor] worker %d died while idle (exit %s)", i, proc.exitcode)

                new_proc = multiprocessing.Process(
                    target=_bench_worker, args=(i, job_q, build_dir, db_path),
                    name=f"bench-worker-{i}")
                new_proc.start()
                worker_procs[i] = new_proc
                logger.info("[Monitor] restarted worker %d (pid=%d)", i, new_proc.pid)
            except Exception:
                logger.exception("[Monitor] worker %d handler failed — continuing", i)
        stop_evt.wait(timeout=1.0)
    conn.close()


def phase_bench(conn, registry, args) -> None:
    """Benchmark every pending eval_runs row on --bench-workers GPU worker processes.

    Rows left running by a dead process are marked crashed first, and missing libraries are
    recompiled before benchmarking.
    """
    conn.execute("UPDATE eval_runs SET status='crashed', error_text='process died during benchmark', "
                 "ran_at=? WHERE status='running'", (_now(),))
    conn.commit()

    n_requeued = registry.requeue_missing_so_files(args.build_dir)
    if n_requeued:
        logger.warning("bench: %d configs marked compile-success but .so missing in %s — "
                       "running compile phase", n_requeued, args.build_dir)
        phase_compile(conn, registry, args)

    _sync_compile_failures(conn)

    rows = conn.execute(
        """SELECT e.*, c.so_file, c.kernel_id, c.cutlass_type_a, c.cutlass_type_b, c.cutlass_type_c
           FROM eval_runs e JOIN configs c ON c.name = e.config_name
           WHERE e.status = 'pending' AND c.compile_status = 'success'"""
    ).fetchall()

    jobs: dict[tuple, list] = defaultdict(list)
    meta: dict[tuple, tuple] = {}
    for r in rows:
        key = (r["M"], r["N"], r["K"], r["so_file"], r["kernel_id"],
               r["raster_order"], r["swizzle_size"], r["splits"])
        jobs[key].append((r["method"], r["M"], r["N"], r["K"], r["layout"], r["rank"], r["variant"]))
        meta[key] = ((r["cutlass_type_a"], r["cutlass_type_b"], r["cutlass_type_c"]),
                     f"{r['layout']} {r['method']}")

    logger.info("bench: %d pending rows -> %d unique jobs, %d worker(s)",
                len(rows), len(jobs), args.bench_workers)
    if not jobs:
        return

    job_q = multiprocessing.JoinableQueue()
    for key, pks in jobs.items():
        dtypes, label = meta[key]
        job_q.put((key, pks, dtypes, label))

    stop_evt = threading.Event()
    worker_procs = []
    for i in range(args.bench_workers):
        p = multiprocessing.Process(target=_bench_worker,
                                    args=(i, job_q, args.build_dir, args.db),
                                    name=f"bench-worker-{i}")
        p.start()
        worker_procs.append(p)

    monitor = threading.Thread(target=_bench_monitor, daemon=True, name="bench-monitor",
                               args=(worker_procs, job_q, registry, args.build_dir,
                                     args.db, stop_evt))
    monitor.start()

    job_q.join()
    stop_evt.set()
    monitor.join(timeout=5)
    for _ in worker_procs:
        job_q.put(None)
    for p in worker_procs:
        p.join(timeout=30)
        if p.is_alive():
            p.terminate()

    counts = dict(conn.execute("SELECT status, COUNT(*) FROM eval_runs GROUP BY status").fetchall())
    logger.info("bench: done — %s", counts)


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(description="Plan, compile and benchmark proposed kernels")
    ap.add_argument("--phase", default="plan,compile,bench")
    ap.add_argument("--db", type=Path, default=DEFAULT_DB)
    ap.add_argument("--build-dir", type=Path, default=DEFAULT_BUILD_DIR)
    ap.add_argument("--template-dir", type=Path, default=SRC / "autotuner" / "templates")
    ap.add_argument(
        "--template",
        default="hopper_template.cu.j2",
        help="Jinja template under --template-dir (use hopper_template_fusion.cu.j2 for fusion)",
    )
    ap.add_argument("--cutlass-dir", type=Path, default=SRC / "extern" / "cutlass")
    ap.add_argument("--proposals", nargs="+", type=Path, default=[],
                    help="proposal JSON files from src/eval/propose.py (plan phase)")
    ap.add_argument("--compile-jobs", type=int, default=6)
    ap.add_argument("--max-batch-size", type=int, default=50)
    ap.add_argument("--bench-workers", type=int, default=None,
                    help="one per GPU, device ids 0..N-1 (default: number of visible GPUs)")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(message)s",
                        datefmt="%H:%M:%S", stream=sys.stdout)
    multiprocessing.set_start_method("spawn", force=True)
    if args.bench_workers is None:
        import torch
        args.bench_workers = max(1, torch.cuda.device_count())

    phases = set(args.phase.split(","))
    if "plan" in phases and not args.proposals:
        print("ERROR: --phase plan requires --proposals", file=sys.stderr)
        return 1

    args.db.parent.mkdir(parents=True, exist_ok=True)
    args.build_dir.mkdir(parents=True, exist_ok=True)
    conn = _connect(args.db)
    registry = SQLiteRegistry(args.db)

    if "plan" in phases:
        phase_plan(conn, registry, args)
    if "compile" in phases:
        phase_compile(conn, registry, args)
    if "bench" in phases:
        phase_bench(conn, registry, args)

    conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
