"""SQLite registry: the single store shared by planning, compilation, benchmarking and profiling.

Tables:
  configs       one row per kernel configuration, with its compile state, ``.so`` file and
                kernel id inside that library
  eval_plan     the planned (config, M, N, K) pairs per tag, written by planning/plan.py
  runs          one benchmark result per (config, M, N, K): status (success, rejected,
                crashed, hung), mean/std latency and throughput
  ncu_runs      Nsight Compute counters per profiled (config, M, N, K), one column per metric
  worker_state  the kernel each GPU worker is currently running, used to attribute crashes

Every phase reads what the previous one wrote, so each can be interrupted and resumed.
``SQLiteRegistry`` is the implementation; ``Registry`` is its interface.
"""

import logging
import sqlite3
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol, runtime_checkable

logger = logging.getLogger(__name__)

_DB_BUSY_RETRIES = 8
_DB_BUSY_BACKOFF_S = 0.05


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


_SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS eval_plan (
    name            TEXT    NOT NULL,
    M               INTEGER NOT NULL,
    N               INTEGER NOT NULL,
    K               INTEGER NOT NULL,
    tag             TEXT    NOT NULL,
    proxy_rank      INTEGER,
    k_used          INTEGER,
    regime          TEXT,
    layer           TEXT,
    sampling_method TEXT,
    layout          TEXT,
    PRIMARY KEY (name, M, N, K, tag)
);

CREATE INDEX IF NOT EXISTS idx_eval_plan_tag  ON eval_plan(tag);
CREATE INDEX IF NOT EXISTS idx_eval_plan_name ON eval_plan(name, tag);

CREATE TABLE IF NOT EXISTS configs (
    name              TEXT PRIMARY KEY,
    cutlass_type_a    TEXT NOT NULL,
    cutlass_type_b    TEXT NOT NULL,
    cutlass_type_c    TEXT NOT NULL,
    cutlass_type_acc  TEXT NOT NULL,
    layout_a          TEXT NOT NULL,
    layout_b          TEXT NOT NULL,
    alignment_a       INTEGER NOT NULL,
    alignment_b       INTEGER NOT NULL,
    alignment_c       INTEGER NOT NULL,
    op_class          TEXT NOT NULL,
    tile_m            INTEGER NOT NULL,
    tile_n            INTEGER NOT NULL,
    tile_k            INTEGER NOT NULL,
    cluster_m         INTEGER NOT NULL,
    cluster_n         INTEGER NOT NULL,
    cluster_k         INTEGER NOT NULL,
    kernel_schedule   TEXT NOT NULL,
    stages            INTEGER NOT NULL,
    epilogue_schedule TEXT NOT NULL,
    scheduler         TEXT NOT NULL,
    compile_status    TEXT NOT NULL DEFAULT 'pending',
    compile_error     TEXT,
    so_file           TEXT,
    kernel_id         INTEGER,
    compiled_at       TEXT
);

CREATE TABLE IF NOT EXISTS runs (
    name                TEXT NOT NULL REFERENCES configs(name),
    M                   INTEGER NOT NULL,
    N                   INTEGER NOT NULL,
    K                   INTEGER NOT NULL,
    status              TEXT NOT NULL,
    cutlass_status_code INTEGER,
    cutlass_reason      TEXT,
    error_text          TEXT,
    mean_ms             REAL,
    std_ms              REAL,
    mean_tflops         REAL,
    std_tflops          REAL,
    ran_at              TEXT,
    tag                 TEXT,
    PRIMARY KEY (name, M, N, K)
);

CREATE TABLE IF NOT EXISTS ncu_runs (
    name                       TEXT NOT NULL REFERENCES configs(name),
    M                          INTEGER NOT NULL,
    N                          INTEGER NOT NULL,
    K                          INTEGER NOT NULL,
    status                     TEXT NOT NULL DEFAULT 'pending',
    error_text                 TEXT,

    -- Throughput / roofline
    compute_throughput_pct     REAL,
    dram_throughput_pct        REAL,
    sm_active_pct              REAL,
    tensor_active_pct          REAL,
    tma_active_pct             REAL,
    fma_active_pct             REAL,
    smem_pipe_active_pct       REAL,

    -- Absolute timing (enables arithmetic intensity and TFLOPS verification)
    duration_ns                REAL,

    -- Memory hierarchy
    l2_throughput_pct          REAL,
    l2_hit_rate                REAL,
    l2_read_hit_rate           REAL,
    l2_write_hit_rate          REAL,
    l2_read_sectors            REAL,
    l2_write_sectors           REAL,
    dram_read_bytes            REAL,
    dram_write_bytes           REAL,
    l1_hit_rate                REAL,
    l1_read_sectors            REAL,

    -- Shared memory
    smem_bank_conflicts_ld     REAL,
    smem_bank_conflicts_st     REAL,
    shmem_ld_wavefronts        REAL,
    shmem_st_wavefronts        REAL,

    -- Occupancy / launch config
    achieved_occupancy_pct     REAL,
    theoretical_occupancy_pct  REAL,
    warps_active               REAL,
    eligible_warps_per_cycle   REAL,
    registers_per_thread       REAL,
    smem_static_bytes          REAL,
    smem_dynamic_bytes         REAL,
    grid_size                  REAL,
    block_size                 REAL,
    waves_per_sm               REAL,

    -- Warp stalls
    stall_mio_pct              REAL,
    stall_long_scoreboard_pct  REAL,
    stall_short_scoreboard_pct REAL,
    stall_barrier_pct          REAL,
    stall_gmma_pct             REAL,
    stall_drain_pct            REAL,
    stall_membar_pct           REAL,
    stall_wait_pct             REAL,
    stall_not_selected_pct     REAL,
    stall_no_instructions_pct  REAL,
    stall_tex_throttle_pct     REAL,

    -- Instruction mix
    issued_ipc                 REAL,
    executed_ipc               REAL,
    wgmma_inst_executed        REAL,
    lsu_inst_executed          REAL,

    profiled_at                TEXT,
    PRIMARY KEY (name, M, N, K)
);

CREATE TABLE IF NOT EXISTS worker_state (
    gpu_id      INTEGER PRIMARY KEY,
    config_name TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_configs_compile_status ON configs(compile_status);
CREATE INDEX IF NOT EXISTS idx_configs_so_file        ON configs(so_file);
CREATE INDEX IF NOT EXISTS idx_runs_status            ON runs(status);
CREATE INDEX IF NOT EXISTS idx_ncu_status             ON ncu_runs(status);
"""

_CONFIG_COLS = (
    "name",
    "cutlass_type_a",
    "cutlass_type_b",
    "cutlass_type_c",
    "cutlass_type_acc",
    "layout_a",
    "layout_b",
    "alignment_a",
    "alignment_b",
    "alignment_c",
    "op_class",
    "tile_m",
    "tile_n",
    "tile_k",
    "cluster_m",
    "cluster_n",
    "cluster_k",
    "kernel_schedule",
    "stages",
    "epilogue_schedule",
    "scheduler",
)


@runtime_checkable
class Registry(Protocol):
    """State tracker for autotuner configs, benchmark runs, and NCU profiles.

    Single source of truth across all phases: compile → benchmark → NCU.
    Implementations must be safe for concurrent use (multiple threads or processes
    writing to the same backing store).

    Run statuses:
        success  — benchmarked successfully, timing recorded
        rejected — CUTLASS returned a non-zero status code (soft runtime rejection)
        crashed  — worker subprocess died while running this (config, shape)
        hung     — worker timed out; kernel suspected to deadlock on this config
        NULL     — not yet attempted (no row in the runs table)
    """

    def register(self, configs: list[dict]) -> None:
        """Insert new configs, skipping any already known (idempotent)."""
        ...

    def pending_compile(self) -> list[dict]:
        """Return configs with compile_status = 'pending'."""
        ...

    def record_compile_success(self, name: str, so_file: str, kernel_id: int) -> None:
        """Mark a config compiled: its library path and kernel id inside that library."""
        ...

    def record_compile_failure(self, name: str, error: str) -> None:
        """Mark a config as failed to compile, storing the tail of the nvcc error."""
        ...

    def pending_runs(self, shapes: list[tuple]) -> list[tuple[dict, tuple]]:
        """Return (config, shape) pairs with no run row yet (never attempted)."""
        ...

    def pending_eval_names(self, shapes: list[tuple]) -> list[str]:
        """Distinct names of compiled configs with at least one unstarted shape."""
        ...

    def record_success(self, name: str, M: int, N: int, K: int, metrics: dict) -> None:
        """Record a successful benchmark: metrics holds mean_ms, std_ms, mean_tflops, std_tflops."""
        ...

    def record_rejected(self, name: str, M: int, N: int, K: int, code: int, reason: str) -> None:
        """Record a CUTLASS runtime rejection (non-zero status code, kernel ran but declined)."""
        ...

    def record_crashed(self, name: str, M: int, N: int, K: int, error: str) -> None:
        """Record a crash for this (config, shape). Safe to call when no row exists yet.
        Never overwrites an existing success or rejected result."""
        ...

    def record_hung(self, name: str, M: int, N: int, K: int) -> None:
        """Record that this (config, shape) caused a worker hang (timeout).
        Safe to call when no row exists yet. Never overwrites success or rejected."""
        ...

    def pending_ncu(self, shapes: list[tuple]) -> list[tuple[dict, tuple]]:
        """Return (config, shape) pairs with a successful run but no NCU data yet."""
        ...

    def record_ncu(self, name: str, M: int, N: int, K: int, metrics: dict | None, error: str | None) -> None:
        """Record NCU counters for a (config, shape), or the error if profiling failed."""
        ...

    def configs_in_so(self, so_file: str) -> list[dict]:
        """Return all successfully compiled configs in a given .so, ordered by kernel_id."""
        ...

    def run_status(self, name: str, M: int, N: int, K: int) -> str | None:
        """Return current run status, or None if no row exists."""
        ...

    def lookup(self, name: str) -> dict | None:
        """Return the full config dict (including so_file and kernel_id) for a compiled config."""
        ...

    def set_worker_state(self, gpu_id: int, config_name: str) -> None:
        """Record that this GPU worker is currently processing config_name."""
        ...

    def get_worker_state(self, gpu_id: int) -> str | None:
        """Return the config_name this GPU worker was processing, or None if idle."""
        ...

    def get_worker_state_and_time(self, gpu_id: int) -> tuple[str | None, str | None]:
        """Return (config_name, updated_at ISO string), or (None, None) if idle."""
        ...

    def clear_worker_state(self, gpu_id: int) -> None:
        """Clear the worker state for this GPU (worker finished cleanly)."""
        ...

    def summary(self) -> dict:
        """Return counts per compile_status and per run status for progress logging."""
        ...


class SQLiteRegistry:
    """SQLite-backed registry for single-node autotuner runs.

    Thread-safe: a single connection is shared across compiler threads via a lock.
    Process-safe: WAL mode allows concurrent writes from worker subprocesses, each
    of which opens its own connection to the same .db file.
    """

    def __init__(self, db_path: Path) -> None:
        self._path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False, timeout=120)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        with self._lock:
            self._conn.execute("PRAGMA busy_timeout=120000")
            self._conn.executescript(_SCHEMA)
            self._conn.commit()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _exec(self, sql: str, params: tuple = ()) -> sqlite3.Cursor:
        last_err: sqlite3.OperationalError | None = None
        for attempt in range(_DB_BUSY_RETRIES):
            try:
                with self._lock:
                    cur = self._conn.execute(sql, params)
                    self._conn.commit()
                    return cur
            except sqlite3.OperationalError as exc:
                if "locked" not in str(exc).lower():
                    raise
                last_err = exc
                time.sleep(_DB_BUSY_BACKOFF_S * (2**attempt))
        raise last_err  # type: ignore[misc]

    def _execmany(self, sql: str, rows: list) -> None:
        last_err: sqlite3.OperationalError | None = None
        for attempt in range(_DB_BUSY_RETRIES):
            try:
                with self._lock:
                    self._conn.executemany(sql, rows)
                    self._conn.commit()
                    return
            except sqlite3.OperationalError as exc:
                if "locked" not in str(exc).lower():
                    raise
                last_err = exc
                time.sleep(_DB_BUSY_BACKOFF_S * (2**attempt))
        raise last_err  # type: ignore[misc]

    def _fetch(self, sql: str, params: tuple = ()) -> list[dict]:
        with self._lock:
            return [dict(r) for r in self._conn.execute(sql, params).fetchall()]

    # ------------------------------------------------------------------
    # Config registration
    # ------------------------------------------------------------------

    def register(self, configs: list[dict]) -> None:
        """Insert new configs, skipping any already known (idempotent)."""
        cols = ", ".join(_CONFIG_COLS)
        placeholders = ", ".join(["?"] * len(_CONFIG_COLS))
        rows = [[c[col] for col in _CONFIG_COLS] for c in configs]
        self._execmany(f"INSERT OR IGNORE INTO configs ({cols}) VALUES ({placeholders})", rows)

    def pending_compile(self) -> list[dict]:
        """Return configs with compile_status = 'pending'."""
        return self._fetch("SELECT * FROM configs WHERE compile_status = 'pending'")

    def record_compile_success(self, name: str, so_file: str, kernel_id: int) -> None:
        """Mark a config compiled: its library path and kernel id inside that library."""
        self._exec(
            "UPDATE configs SET compile_status='success', so_file=?, kernel_id=?, compiled_at=? WHERE name=?",
            (so_file, kernel_id, _now(), name),
        )

    def record_compile_failure(self, name: str, error: str) -> None:
        """Mark a config as failed to compile, storing the last 2000 characters of the error."""
        self._exec(
            "UPDATE configs SET compile_status='failed', compile_error=? WHERE name=?",
            (error[-2000:], name),
        )

    # ------------------------------------------------------------------
    # Benchmark runs
    # ------------------------------------------------------------------

    def pending_runs(self, shapes: list[tuple]) -> list[tuple[dict, tuple]]:
        """Return (config, shape) pairs of compiled configs with no run row for that shape yet."""
        result = []
        for shape in shapes:
            M, N, K = shape
            rows = self._fetch(
                """
                SELECT c.* FROM configs c
                LEFT JOIN runs r ON c.name = r.name AND r.M = ? AND r.N = ? AND r.K = ?
                WHERE c.compile_status = 'success'
                  AND r.status IS NULL
                """,
                (M, N, K),
            )
            for row in rows:
                result.append((row, shape))
        return result


    def pending_eval_names(self, shapes: list[tuple]) -> list[str]:
        """Distinct names of compiled configs with at least one unstarted shape."""
        return list({config["name"] for config, _ in self.pending_runs(shapes)})

    def record_success(self, name: str, M: int, N: int, K: int, metrics: dict, tag: str | None = None) -> None:
        """Record a successful benchmark (overwrites any earlier result for this pair).

        metrics holds mean_ms, std_ms, mean_tflops, std_tflops; tag is the plan tag the run belongs to.
        """
        self._exec(
            """
            INSERT INTO runs (name, M, N, K, status, mean_ms, std_ms, mean_tflops, std_tflops, ran_at, tag)
            VALUES (?, ?, ?, ?, 'success', ?, ?, ?, ?, ?, ?)
            ON CONFLICT(name, M, N, K) DO UPDATE SET
                status='success', mean_ms=excluded.mean_ms, std_ms=excluded.std_ms,
                mean_tflops=excluded.mean_tflops, std_tflops=excluded.std_tflops,
                ran_at=excluded.ran_at
            """,
            (
                name,
                M,
                N,
                K,
                metrics["mean_ms"],
                metrics["std_ms"],
                metrics["mean_tflops"],
                metrics["std_tflops"],
                _now(),
                tag,
            ),
        )

    def record_rejected(self, name: str, M: int, N: int, K: int, code: int, reason: str, tag: str | None = None) -> None:
        """Record a CUTLASS runtime rejection: the kernel returned a non-zero status code."""
        self._exec(
            """
            INSERT OR REPLACE INTO runs (name, M, N, K, status, cutlass_status_code, cutlass_reason, ran_at, tag)
            VALUES (?, ?, ?, ?, 'rejected', ?, ?, ?, ?)
            """,
            (name, M, N, K, code, reason, _now(), tag),
        )

    def record_crashed(self, name: str, M: int, N: int, K: int, error: str, tag: str | None = None) -> None:
        # Upsert: insert if no row yet, update only if not already success/rejected.
        # The first shape with no run row is the one that caused the crash — subsequent
        # shapes stay NULL so they will be retried on the next run.
        """Record that this (config, shape) crashed its worker. Never overwrites a success or rejection."""
        self._exec(
            """
            INSERT INTO runs (name, M, N, K, status, error_text, ran_at, tag)
            VALUES (?, ?, ?, ?, 'crashed', ?, ?, ?)
            ON CONFLICT(name, M, N, K) DO UPDATE SET
                status='crashed', error_text=excluded.error_text, ran_at=excluded.ran_at
            WHERE runs.status NOT IN ('success', 'rejected')
            """,
            (name, M, N, K, error[-2000:], _now(), tag),
        )

    def record_hung(self, name: str, M: int, N: int, K: int, tag: str | None = None) -> None:
        """Record that this (config, shape) made its worker time out. Never overwrites a success or rejection."""
        self._exec(
            """
            INSERT INTO runs (name, M, N, K, status, error_text, ran_at, tag)
            VALUES (?, ?, ?, ?, 'hung', 'worker timed out', ?, ?)
            ON CONFLICT(name, M, N, K) DO UPDATE SET
                status='hung', error_text='worker timed out', ran_at=excluded.ran_at
            WHERE runs.status NOT IN ('success', 'rejected')
            """,
            (name, M, N, K, _now(), tag),
        )

    # ------------------------------------------------------------------
    # NCU runs
    # ------------------------------------------------------------------

    def pending_ncu(self, shapes: list[tuple], top_k: int | None = None) -> list[tuple[dict, tuple]]:
        """Return (config, shape) pairs with a successful run but no NCU data yet.

        If top_k is given, the top-k configs per shape per dtype (cutlass_type_a /
        cutlass_type_b / cutlass_type_c) by mean_tflops are returned, so each
        precision config contributes its own top-k independently.
        """
        result = []
        for shape in shapes:
            M, N, K = shape
            if top_k:
                rows = self._fetch(
                    """
                    SELECT * FROM (
                        SELECT c.*, ROW_NUMBER() OVER (
                            PARTITION BY c.cutlass_type_a, c.cutlass_type_b, c.cutlass_type_c
                            ORDER BY r.mean_tflops DESC
                        ) AS rn
                        FROM configs c
                        JOIN runs r ON c.name = r.name AND r.M = ? AND r.N = ? AND r.K = ?
                        LEFT JOIN ncu_runs n ON c.name = n.name AND n.M = ? AND n.N = ? AND n.K = ?
                        WHERE r.status = 'success'
                          AND n.name IS NULL
                    )
                    WHERE rn <= ?
                    """,
                    (M, N, K, M, N, K, top_k),
                )
            else:
                rows = self._fetch(
                    """
                    SELECT c.* FROM configs c
                    JOIN runs r ON c.name = r.name AND r.M = ? AND r.N = ? AND r.K = ?
                    LEFT JOIN ncu_runs n ON c.name = n.name AND n.M = ? AND n.N = ? AND n.K = ?
                    WHERE r.status = 'success'
                      AND n.name IS NULL
                    ORDER BY r.mean_tflops DESC
                    """,
                    (M, N, K, M, N, K),
                )
            for row in rows:
                result.append((row, shape))
        return result

    def record_ncu(self, name: str, M: int, N: int, K: int, metrics: dict | None, error: str | None) -> None:
        """Store one NCU profile: one column per metric on success, the error text otherwise."""
        if metrics:
            cols = ", ".join(metrics.keys())
            placeholders = ", ".join(["?"] * len(metrics))
            self._exec(
                f"""
                INSERT OR REPLACE INTO ncu_runs (name, M, N, K, status, {cols}, profiled_at)
                VALUES (?, ?, ?, ?, 'success', {placeholders}, ?)
                """,
                (name, M, N, K, *metrics.values(), _now()),
            )
        else:
            self._exec(
                """
                INSERT OR REPLACE INTO ncu_runs (name, M, N, K, status, error_text, profiled_at)
                VALUES (?, ?, ?, ?, 'failed', ?, ?)
                """,
                (name, M, N, K, error, _now()),
            )

    # ------------------------------------------------------------------
    # Lookups and utilities
    # ------------------------------------------------------------------

    def configs_in_so(self, so_file: str) -> list[dict]:
        """Return all successfully compiled configs in a given .so, ordered by kernel_id."""
        return self._fetch(
            """
            SELECT * FROM configs
            WHERE so_file = ? AND compile_status = 'success'
            ORDER BY kernel_id
            """,
            (so_file,),
        )

    def set_worker_state(self, gpu_id: int, config_name: str) -> None:
        """Record that this GPU worker is currently processing config_name."""
        self._exec(
            "INSERT OR REPLACE INTO worker_state (gpu_id, config_name, updated_at) VALUES (?, ?, ?)",
            (gpu_id, config_name, _now()),
        )

    def get_worker_state(self, gpu_id: int) -> str | None:
        """Return the config_name this GPU worker was processing, or None if idle."""
        rows = self._fetch("SELECT config_name FROM worker_state WHERE gpu_id = ?", (gpu_id,))
        return rows[0]["config_name"] if rows else None

    def get_worker_state_and_time(self, gpu_id: int) -> tuple[str | None, str | None]:
        """Return (config_name, updated_at ISO string), or (None, None) if idle."""
        rows = self._fetch("SELECT config_name, updated_at FROM worker_state WHERE gpu_id = ?", (gpu_id,))
        if not rows:
            return None, None
        return rows[0]["config_name"], rows[0]["updated_at"]

    def clear_worker_state(self, gpu_id: int) -> None:
        """Clear the worker state for this GPU (worker finished cleanly)."""
        self._exec("DELETE FROM worker_state WHERE gpu_id = ?", (gpu_id,))

    def run_status(self, name: str, M: int, N: int, K: int) -> str | None:
        """Return the run status of this (config, shape), or None if it was never attempted."""
        rows = self._fetch(
            "SELECT status FROM runs WHERE name=? AND M=? AND N=? AND K=?",
            (name, M, N, K),
        )
        return rows[0]["status"] if rows else None

    def lookup(self, name: str) -> dict | None:
        """Return the full row (including so_file and kernel_id) of a compiled config, or None."""
        rows = self._fetch("SELECT * FROM configs WHERE name=? AND compile_status='success'", (name,))
        return rows[0] if rows else None

    # ------------------------------------------------------------------
    # Eval plan (sweep planner)
    # ------------------------------------------------------------------

    def plan_delete_tag(self, tag: str) -> int:
        """Remove all eval_plan rows for a tag (used before replanning a sweep)."""
        with self._lock:
            cur = self._conn.execute("DELETE FROM eval_plan WHERE tag = ?", (tag,))
            self._conn.commit()
            return cur.rowcount

    def plan_put(self, entries: list[dict], batch_size: int = 50_000) -> None:
        """Bulk-insert (name, M, N, K, tag, …) into eval_plan. Idempotent."""
        sql = """INSERT OR IGNORE INTO eval_plan
               (name, M, N, K, tag, proxy_rank, k_used, regime, layer,
                sampling_method, layout)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"""
        total = len(entries)
        with self._lock:
            # Bulk load: home/scratch DBs on shared FS can take 30+ min without this.
            self._conn.execute("PRAGMA synchronous=OFF")
            self._conn.execute("PRAGMA temp_store=MEMORY")
            try:
                for start in range(0, total, batch_size):
                    batch = entries[start : start + batch_size]
                    rows = [
                        (
                            e["name"], e["M"], e["N"], e["K"], e["tag"],
                            e.get("proxy_rank"), e.get("k_used"),
                            e.get("regime"), e.get("layer"),
                            e.get("sampling_method"), e.get("layout"),
                        )
                        for e in batch
                    ]
                    self._conn.executemany(sql, rows)
                    self._conn.commit()
                    done = min(start + batch_size, total)
                    print(f"  {done:,}/{total:,}", flush=True)
            finally:
                self._conn.execute("PRAGMA synchronous=NORMAL")
                self._conn.commit()

    def plan_pending_compile(self, tag: str) -> list[dict]:
        """Configs in the plan for this tag that still need compilation."""
        return self._fetch(
            """
            SELECT DISTINCT c.* FROM configs c
            INNER JOIN eval_plan p ON c.name = p.name
            WHERE p.tag = ? AND c.compile_status = 'pending'
            """,
            (tag,),
        )

    def plan_pending_compile_for_shard(
        self,
        tag: str,
        shard_index: int,
        num_shards: int,
    ) -> list[dict]:
        """Pending compile configs that have at least one eval pair on this shard."""
        from shard import pair_on_shard

        if num_shards <= 1:
            return self.plan_pending_compile(tag)

        names: set[str] = set()
        rows = self._fetch(
            """
            SELECT DISTINCT p.name, p.M, p.N, p.K
            FROM eval_plan p
            JOIN configs c ON c.name = p.name
            WHERE p.tag = ? AND c.compile_status = 'pending'
            """,
            (tag,),
        )
        for r in rows:
            if pair_on_shard(r["name"], r["M"], r["N"], r["K"], shard_index, num_shards):
                names.add(r["name"])
        if not names:
            return []
        placeholders = ",".join("?" * len(names))
        return self._fetch(
            f"""
            SELECT * FROM configs
            WHERE name IN ({placeholders}) AND compile_status = 'pending'
            """,
            tuple(sorted(names)),
        )

    def requeue_compile_failures_for_tag(self, tag: str) -> int:
        """Reset failed → pending for kernels in this tag's plan (e.g. after hipcc fix)."""
        cur = self._exec(
            """
            UPDATE configs
            SET compile_status='pending', compile_error=NULL
            WHERE compile_status='failed'
              AND name IN (SELECT DISTINCT name FROM eval_plan WHERE tag=?)
            """,
            (tag,),
        )
        return cur.rowcount

    def requeue_config_missing_so(self, name: str) -> None:
        """Mark one compiled config pending when its .so is absent from the build dir."""
        self._exec(
            """
            UPDATE configs
            SET compile_status='pending', so_file=NULL, kernel_id=NULL,
                compiled_at=NULL, compile_error=NULL
            WHERE name=? AND compile_status='success'
            """,
            (name,),
        )

    def requeue_missing_so_files(self, build_dir: Path, tag: str | None = None) -> int:
        """Reset compile_status for configs whose .so is missing from build_dir.

        Needed on resume: the merged HOME DB records global compile success, but each
        node's scratch build dir only contains kernels compiled locally (and old job-
        scoped paths are empty after a new Slurm job id).
        """
        build_dir = Path(build_dir)
        if tag:
            rows = self._fetch(
                """
                SELECT DISTINCT c.name, c.so_file FROM configs c
                INNER JOIN eval_plan p ON c.name = p.name
                WHERE p.tag = ? AND c.compile_status = 'success' AND c.so_file IS NOT NULL
                """,
                (tag,),
            )
        else:
            rows = self._fetch(
                """
                SELECT name, so_file FROM configs
                WHERE compile_status = 'success' AND so_file IS NOT NULL
                """
            )
        missing = [r["name"] for r in rows if not (build_dir / r["so_file"]).is_file()]
        if not missing:
            return 0
        placeholders = ",".join("?" * len(missing))
        self._exec(
            f"""
            UPDATE configs
            SET compile_status='pending', so_file=NULL, kernel_id=NULL,
                compiled_at=NULL, compile_error=NULL
            WHERE name IN ({placeholders}) AND compile_status='success'
            """,
            tuple(missing),
        )
        return len(missing)

    def clear_retryable_runs_for_tag(self, tag: str) -> int:
        """Delete crashed/hung run rows for a tag so eval can retry those shapes."""
        cur = self._exec(
            """
            DELETE FROM runs
            WHERE status IN ('crashed', 'hung')
              AND name IN (SELECT DISTINCT name FROM eval_plan WHERE tag=?)
            """,
            (tag,),
        )
        return cur.rowcount

    def plan_pending_eval_names(
        self,
        tag: str,
        shard_index: int | None = None,
        num_shards: int | None = None,
    ) -> list[str]:
        """Compiled config names in the plan that have at least one unstarted shape."""
        from shard import pair_on_shard

        rows = self._fetch(
            """
            SELECT DISTINCT p.name, p.M, p.N, p.K
            FROM eval_plan p
            JOIN configs c ON c.name = p.name
            LEFT JOIN runs r ON r.name = p.name AND r.M = p.M AND r.N = p.N AND r.K = p.K
            WHERE p.tag = ? AND c.compile_status = 'success' AND r.status IS NULL
            """,
            (tag,),
        )
        if num_shards is not None and num_shards > 1:
            if shard_index is None:
                raise ValueError("shard_index required when num_shards > 1")
            rows = [
                r
                for r in rows
                if pair_on_shard(r["name"], r["M"], r["N"], r["K"], shard_index, num_shards)
            ]
        return sorted({r["name"] for r in rows})

    def plan_shapes_for_config(
        self,
        name: str,
        tag: str,
        shard_index: int | None = None,
        num_shards: int | None = None,
    ) -> list[tuple[int, int, int]]:
        """Shapes planned for (name, tag) that have not yet been benchmarked."""
        from shard import pair_on_shard

        rows = self._fetch(
            """
            SELECT p.M, p.N, p.K FROM eval_plan p
            LEFT JOIN runs r ON r.name = p.name AND r.M = p.M AND r.N = p.N AND r.K = p.K
            WHERE p.name = ? AND p.tag = ? AND r.status IS NULL
            ORDER BY p.M, p.N, p.K
            """,
            (name, tag),
        )
        shapes = [(r["M"], r["N"], r["K"]) for r in rows]
        if num_shards is not None and num_shards > 1:
            if shard_index is None:
                raise ValueError("shard_index required when num_shards > 1")
            shapes = [
                (M, N, K)
                for M, N, K in shapes
                if pair_on_shard(name, M, N, K, shard_index, num_shards)
            ]
        return shapes

    def plan_distinct_shapes(self, tag: str) -> list[tuple[int, int, int]]:
        """Distinct (M, N, K) shapes appearing in the plan for this tag."""
        rows = self._fetch(
            "SELECT DISTINCT M, N, K FROM eval_plan WHERE tag = ?",
            (tag,),
        )
        return [(r["M"], r["N"], r["K"]) for r in rows]

    def plan_distinct_config_count(self, tag: str) -> int:
        """Number of distinct config names in eval_plan for this tag."""
        rows = self._fetch(
            "SELECT COUNT(DISTINCT name) AS cnt FROM eval_plan WHERE tag = ?",
            (tag,),
        )
        return int(rows[0]["cnt"]) if rows else 0

    def summary(self) -> dict:
        """Return {'compile': ..., 'runs': ..., 'ncu': ...}, each a count per status."""
        compile_counts = {
            r["compile_status"]: r["cnt"]
            for r in self._fetch("SELECT compile_status, COUNT(*) AS cnt FROM configs GROUP BY compile_status")
        }
        run_counts = {
            r["status"]: r["cnt"] for r in self._fetch("SELECT status, COUNT(*) AS cnt FROM runs GROUP BY status")
        }
        ncu_counts = {
            r["status"]: r["cnt"] for r in self._fetch("SELECT status, COUNT(*) AS cnt FROM ncu_runs GROUP BY status")
        }
        return {"compile": compile_counts, "runs": run_counts, "ncu": ncu_counts}
