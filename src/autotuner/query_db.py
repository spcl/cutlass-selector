"""
query_db.py — Query the autotuner registry (autotuner.db) for debugging and analysis.

Usage:
    python src/autotuner/query_db.py summary
    python src/autotuner/query_db.py summary --tag sweep
    python src/autotuner/query_db.py shapes
    python src/autotuner/query_db.py shapes --tag sweep
    python src/autotuner/query_db.py best 4096 4096 4096
    python src/autotuner/query_db.py best 4096 4096 4096 --tag sweep
    python src/autotuner/query_db.py ncu 4096 4096 4096
    python src/autotuner/query_db.py failures
    python src/autotuner/query_db.py failures --tag sweep
    python src/autotuner/query_db.py export --out results.parquet
    python src/autotuner/query_db.py export --out results.parquet --tag sweep
    python src/autotuner/query_db.py estimate
    python src/autotuner/query_db.py estimate --tag sweep

All commands accept --db to point at a non-default registry:
    python src/autotuner/query_db.py --db /path/to/autotuner.db summary

Default DB search order:
    1. <repo>/build_cache/autotuner.db  (scheduler/plan default)
    2. <repo>/autotuner.db              (manually copied)
"""

from __future__ import annotations

import argparse
import math
import sqlite3
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from repo_paths import REPO_ROOT, SRC

sys.path.insert(0, str(SRC / "autotuner"))
from scheduler import BENCHMARK_SHAPES

_DB_CANDIDATES = [
    REPO_ROOT / "build_cache" / "autotuner.db",
    REPO_ROOT / "autotuner.db",
]


def _find_db() -> Path:
    for p in _DB_CANDIDATES:
        if p.exists():
            return p
    return _DB_CANDIDATES[0]


# ── Formatting helpers ────────────────────────────────────────────────────────

def _table(rows: list[dict], cols: list[str] | None = None) -> str:
    if not rows:
        return "  (no rows)"
    cols = cols or list(rows[0].keys())
    widths = [max(len(str(c)), max(len(str(r.get(c, ""))) for r in rows)) for c in cols]
    sep  = "  ".join("-" * w for w in widths)
    head = "  ".join(str(c).ljust(w) for c, w in zip(cols, widths))
    body = "\n".join(
        "  ".join(str(r.get(c, "")).ljust(w) for c, w in zip(cols, widths))
        for r in rows
    )
    return f"{head}\n{sep}\n{body}"


def _bar(value: int, total: int, width: int = 24) -> str:
    filled = round(width * value / total) if total else 0
    return f"[{'█' * filled}{'░' * (width - filled)}] {value:>6,}/{total:,}"


def _bar_pct(value: int, total: int, width: int = 24) -> str:
    filled = round(width * value / total) if total else 0
    pct = value / total * 100 if total else 0
    return f"[{'█' * filled}{'░' * (width - filled)}] {pct:.1f}%"


def _pct(value: int, total: int) -> str:
    return f"{value / total * 100:.1f}%" if total else "—"


def _age(path: Path) -> str:
    try:
        sec = time.time() - path.stat().st_mtime
        if sec < 60:
            return f"{int(sec)}s ago"
        if sec < 3600:
            return f"{int(sec / 60)}m ago"
        if sec < 86400:
            return f"{int(sec / 3600)}h {int(sec % 3600 / 60)}m ago"
        return f"{int(sec / 86400)}d {int(sec % 86400 / 3600)}h ago"
    except OSError:
        return "unknown"


def _connect(db: Path) -> sqlite3.Connection:
    if not db.exists():
        candidates = "\n  ".join(str(p) for p in _DB_CANDIDATES)
        sys.exit(
            f"Registry not found: {db}\n"
            f"(file does not exist — check path and that the Slurm job copied DB to $HOME)\n"
            f"Default search paths when --db is omitted:\n  {candidates}\n"
            f"Run the scheduler first, or pass --db <path>."
        )
    conn = sqlite3.connect(str(db))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA temp_store=MEMORY")
    conn.execute("PRAGMA cache_size=-1048576")
    return conn


# ── Subcommands ───────────────────────────────────────────────────────────────

def cmd_summary(conn: sqlite3.Connection, args) -> None:
    """Overall pipeline progress at a glance."""
    tag = args.tag

    cc_all = {r["compile_status"]: r["n"] for r in conn.execute(
        "SELECT compile_status, COUNT(*) AS n FROM configs GROUP BY compile_status"
    ).fetchall()}
    nc = {r["status"]: r["n"] for r in conn.execute(
        "SELECT status, COUNT(*) AS n FROM ncu_runs GROUP BY status"
    ).fetchall()}

    print(f"\n  DB            : {args.db}  (last modified: {_age(args.db)})")
    if tag:
        print(f"  Tag           : {tag}")

    # ── Compile ──────────────────────────────────────────────────────────────
    print("\n── Compile ─────────────────────────────────────────────")
    if tag:
        plan_configs = conn.execute(
            "SELECT COUNT(DISTINCT name) FROM eval_plan WHERE tag=?", (tag,)
        ).fetchone()[0]
        cc_tag = {r["compile_status"]: r["n"] for r in conn.execute("""
            SELECT c.compile_status, COUNT(DISTINCT c.name) AS n
            FROM configs c JOIN eval_plan p ON c.name = p.name
            WHERE p.tag = ?
            GROUP BY c.compile_status
        """, (tag,)).fetchall()}
        print(f"  Plan configs   : {plan_configs:,}")
        for status in ("success", "failed", "pending"):
            n = cc_tag.get(status, 0)
            bar = _bar(n, plan_configs) if status == "success" else ""
            print(f"  {status:<14} : {n:>7,}  {bar}")
    else:
        total_configs = sum(cc_all.values())
        print(f"  Total configs  : {total_configs:,}")
        for status in ("success", "failed", "pending"):
            n = cc_all.get(status, 0)
            bar = _bar(n, total_configs) if status == "success" else ""
            print(f"  {status:<14} : {n:>7,}  {bar}")

    # ── Benchmark runs ────────────────────────────────────────────────────────
    print("\n── Benchmark runs ──────────────────────────────────────")
    if tag:
        total_plan = conn.execute(
            "SELECT COUNT(*) FROM eval_plan WHERE tag=?", (tag,)
        ).fetchone()[0]
        n_shapes = conn.execute(
            "SELECT COUNT(DISTINCT M||','||N||','||K) FROM eval_plan WHERE tag=?", (tag,)
        ).fetchone()[0]
        rc = {r["status"]: r["n"] for r in conn.execute(
            "SELECT status, COUNT(*) AS n FROM runs WHERE tag=? GROUP BY status", (tag,)
        ).fetchall()}
        rc["pending"] = max(total_plan - sum(rc.values()), 0)
        print(f"  {'Total pairs':<14} : {total_plan:>7,}  ({n_shapes} shapes)")
        for status in ("success", "rejected", "crashed", "hung", "pending"):
            n = rc.get(status, 0)
            if not n:
                continue
            bar = _bar_pct(n, total_plan) if status in ("success", "rejected") else ""
            print(f"  {status:<14} : {n:>7,}  {bar}")
    else:
        compiled = cc_all.get("success", 0)
        n_shapes = len(BENCHMARK_SHAPES)
        total_expected = compiled * n_shapes
        shape_filter = " OR ".join(f"(M={M} AND N={N} AND K={K})" for M, N, K in BENCHMARK_SHAPES)
        rc = {r["status"]: r["n"] for r in conn.execute(
            f"SELECT status, COUNT(*) AS n FROM runs WHERE {shape_filter} GROUP BY status"
        ).fetchall()}
        total_runs = sum(rc.values())
        if total_runs:
            print(f"  {'Total pairs':<14} : {total_expected:>7,}  ({compiled:,} configs × {n_shapes} shapes)")
            for status in ("success", "rejected", "crashed", "hung"):
                n = rc.get(status, 0)
                if not n:
                    continue
                bar = _bar_pct(n, total_expected) if status in ("success", "rejected") else ""
                print(f"  {status:<14} : {n:>7,}  {bar}")
            pending = total_expected - total_runs
            if pending > 0:
                print(f"  {'pending':<14} : {pending:>7,}")
        else:
            print("  (no runs recorded yet)")

    # ── NCU profiles ──────────────────────────────────────────────────────────
    print("\n── NCU profiles ────────────────────────────────────────")
    shape_filter_r = " OR ".join(f"(r.M={M} AND r.N={N} AND r.K={K})" for M, N, K in BENCHMARK_SHAPES)
    n_ncu_done = nc.get("success", 0) + nc.get("failed", 0)
    n_ncu_pending = conn.execute(f"""
        SELECT COUNT(*) FROM configs c
        JOIN runs r ON c.name = r.name AND r.status = 'success' AND ({shape_filter_r})
        LEFT JOIN ncu_runs n ON c.name = n.name AND n.M = r.M AND n.N = r.N AND n.K = r.K
        WHERE n.name IS NULL
    """).fetchone()[0]
    if n_ncu_done > 0:
        total_ncu = n_ncu_done + n_ncu_pending
        for status in ("success", "failed"):
            n = nc.get(status, 0)
            if n:
                bar = _bar(n, total_ncu) if status == "success" else ""
                print(f"  {status:<14} : {n:>7,}  {bar}")
        print(f"  {'pending':<14} : {n_ncu_pending:>7,}")
    else:
        print("  (not started — run scheduler with --phase ncu)")

    print()


def cmd_shapes(conn: sqlite3.Connection, args) -> None:
    """Per-shape benchmark progress."""
    tag = args.tag

    if tag:
        rows = conn.execute("""
            SELECT
                p.M, p.N, p.K,
                COUNT(*)                                                   AS planned,
                SUM(CASE WHEN r.status = 'success'  THEN 1 ELSE 0 END)    AS success,
                SUM(CASE WHEN r.status = 'rejected' THEN 1 ELSE 0 END)    AS rejected,
                SUM(CASE WHEN r.status = 'crashed'  THEN 1 ELSE 0 END)    AS crashed,
                SUM(CASE WHEN r.status = 'hung'     THEN 1 ELSE 0 END)    AS hung,
                SUM(CASE WHEN r.status IS NULL      THEN 1 ELSE 0 END)    AS pending
            FROM eval_plan p
            LEFT JOIN runs r ON r.name=p.name AND r.M=p.M AND r.N=p.N AND r.K=p.K
            WHERE p.tag = ?
            GROUP BY p.M, p.N, p.K
            ORDER BY p.M, p.N, p.K
        """, (tag,)).fetchall()

        if not rows:
            print(f"\n  No shapes found for tag '{tag}'.\n")
            return

        data = [
            {
                "shape":    f"{r['M']}×{r['N']}×{r['K']}",
                "planned":  r["planned"],
                "success":  r["success"],
                "rejected": r["rejected"],
                "crashed":  r["crashed"],
                "hung":     r["hung"],
                "pending":  r["pending"],
                "done %":   _pct(r["success"] + r["rejected"], r["planned"]),
            }
            for r in rows
        ]
    else:
        rows = conn.execute("""
            SELECT
                r.M, r.N, r.K,
                SUM(r.status = 'success')   AS success,
                SUM(r.status = 'rejected')  AS rejected,
                SUM(r.status = 'crashed')   AS crashed,
                SUM(r.status = 'hung')      AS hung,
                COUNT(*)                    AS total
            FROM runs r
            GROUP BY r.M, r.N, r.K
            ORDER BY r.M, r.N, r.K
        """).fetchall()

        if not rows:
            compiled = conn.execute(
                "SELECT COUNT(*) FROM configs WHERE compile_status='success'"
            ).fetchone()[0]
            print(f"\n  No runs recorded yet. ({compiled:,} configs compiled.)\n")
            return

        data = [
            {
                "shape":    f"{r['M']}×{r['N']}×{r['K']}",
                "success":  r["success"],
                "rejected": r["rejected"],
                "crashed":  r["crashed"],
                "hung":     r["hung"],
                "total":    r["total"],
                "done %":   _pct(r["success"] + r["rejected"], r["total"]),
            }
            for r in rows
        ]

    print()
    print(_table(data))
    print()


def cmd_best(conn: sqlite3.Connection, args) -> None:
    """Top-N configs for a given shape, sorted by TFLOPS."""
    M, N, K = args.M, args.N, args.K
    tag = args.tag

    if tag:
        rows = conn.execute("""
            SELECT
                c.name, c.cutlass_type_a, c.tile_m, c.tile_n, c.tile_k,
                c.stages, c.cluster_m, c.cluster_n,
                c.kernel_schedule, c.scheduler,
                r.mean_tflops, r.std_tflops, r.mean_ms,
                p.proxy_rank, p.sampling_method, p.layout
            FROM runs r
            JOIN configs c USING(name)
            JOIN eval_plan p ON r.name=p.name AND r.M=p.M AND r.N=p.N AND r.K=p.K AND p.tag=?
            WHERE r.status = 'success' AND r.M = ? AND r.N = ? AND r.K = ?
            ORDER BY r.mean_tflops DESC
            LIMIT ?
        """, (tag, M, N, K, args.n)).fetchall()
    else:
        rows = conn.execute("""
            SELECT
                c.name, c.cutlass_type_a, c.tile_m, c.tile_n, c.tile_k,
                c.stages, c.cluster_m, c.cluster_n,
                c.kernel_schedule, c.scheduler,
                r.mean_tflops, r.std_tflops, r.mean_ms,
                NULL AS proxy_rank, NULL AS sampling_method, NULL AS layout
            FROM runs r
            JOIN configs c USING(name)
            WHERE r.status = 'success' AND r.M = ? AND r.N = ? AND r.K = ?
            ORDER BY r.mean_tflops DESC
            LIMIT ?
        """, (M, N, K, args.n)).fetchall()

    if not rows:
        print(f"\n  No successful runs found for shape {M}×{N}×{K}"
              + (f" (tag={tag})" if tag else "") + ".\n")
        return

    peak = {"cutlass::bfloat16_t": 827.2, "cutlass::float_e4m3_t": 1654.4,
            "cutlass::half_t": 827.2, "float": 67.0}

    data = []
    for r in rows:
        p = peak.get(r["cutlass_type_a"])
        sol = f"{r['mean_tflops'] / p * 100:.1f}%" if p else "—"
        sched = r["kernel_schedule"].split("::")[-1].replace("KernelTma", "")
        tile  = f"{r['tile_m']}×{r['tile_n']}×{r['tile_k']}"
        clust = f"{r['cluster_m']}×{r['cluster_n']}"
        sk    = "StreamK" if "StreamK" in (r["scheduler"] or "") else ""
        row = {
            "TFLOPS":   f"{r['mean_tflops']:.2f}",
            "±":        f"{r['std_tflops']:.2f}",
            "SOL":      sol,
            "ms":       f"{r['mean_ms']:.3f}",
            "tile":     tile,
            "stages":   r["stages"],
            "cluster":  clust,
            "schedule": sched,
            "sk":       sk,
        }
        if tag:
            row["p_rank"] = r["proxy_rank"] if r["proxy_rank"] is not None else "—"
            row["method"] = r["sampling_method"] or "—"
            row["layout"] = r["layout"] or "—"
        data.append(row)

    label = f"Top {args.n} for {M}×{N}×{K}" + (f"  [tag={tag}]" if tag else "")
    print(f"\n  {label}\n")
    print(_table(data))
    print()


def cmd_ncu(conn: sqlite3.Connection, args) -> None:
    """Top-N profiled configs for a shape, with key NCU metrics."""
    M, N, K = args.M, args.N, args.K
    rows = conn.execute("""
        SELECT
            c.name, c.tile_m, c.tile_n, c.tile_k, c.stages,
            c.cluster_m, c.cluster_n, c.kernel_schedule,
            r.mean_tflops,
            n.compute_throughput_pct,
            n.dram_throughput_pct,
            n.sm_active_pct,
            n.tensor_active_pct,
            n.achieved_occupancy_pct,
            n.stall_gmma_pct,
            n.stall_barrier_pct,
            n.stall_long_scoreboard_pct,
            n.stall_mio_pct
        FROM ncu_runs n
        JOIN configs c ON n.name = c.name
        JOIN runs r ON n.name = r.name AND r.M = n.M AND r.N = n.N AND r.K = n.K
        WHERE n.status = 'success' AND n.M = ? AND n.N = ? AND n.K = ?
          AND r.status = 'success'
        ORDER BY r.mean_tflops DESC
        LIMIT ?
    """, (M, N, K, args.n)).fetchall()

    if not rows:
        print(f"\n  No NCU profiles found for shape {M}×{N}×{K}.\n")
        return

    def _f(v, fmt=".1f"):
        return f"{v:{fmt}}" if v is not None else "—"

    data = []
    for r in rows:
        sched = r["kernel_schedule"].split("::")[-1].replace("KernelTma", "")
        tile  = f"{r['tile_m']}×{r['tile_n']}×{r['tile_k']}"
        clust = f"{r['cluster_m']}×{r['cluster_n']}"
        data.append({
            "TFLOPS":    _f(r["mean_tflops"]),
            "tile":      tile,
            "stg":       r["stages"],
            "clu":       clust,
            "sched":     sched,
            "comp%":     _f(r["compute_throughput_pct"]),
            "dram%":     _f(r["dram_throughput_pct"]),
            "sm%":       _f(r["sm_active_pct"]),
            "tensor%":   _f(r["tensor_active_pct"]),
            "occ%":      _f(r["achieved_occupancy_pct"]),
            "gmma_st%":  _f(r["stall_gmma_pct"]),
            "bar_st%":   _f(r["stall_barrier_pct"]),
            "lsb_st%":   _f(r["stall_long_scoreboard_pct"]),
            "mio_st%":   _f(r["stall_mio_pct"]),
        })

    print(f"\n  Top {args.n} NCU profiles for {M}×{N}×{K}\n")
    print(_table(data))
    print()


def cmd_failures(conn: sqlite3.Connection, args) -> None:
    """Show compile failures and runtime crashes/hangs."""
    tag = args.tag

    cf = conn.execute("""
        SELECT name, compile_error FROM configs
        WHERE compile_status = 'failed'
        LIMIT 50
    """).fetchall()

    print(f"\n── Compile failures ({len(cf)}) ─────────────────────────────")
    if cf:
        for r in cf:
            err = r["compile_error"] or ""
            import re
            m = re.search(r'static assertion failed with "(.*?)"', err)
            reason = m.group(1) if m else err[:120].replace("\n", " ")
            print(f"  {r['name'][:70]}")
            print(f"    → {reason}")
    else:
        print("  None.")

    if tag:
        crash_q = """
            SELECT c.name, r.M, r.N, r.K, r.error_text
            FROM runs r JOIN configs c USING(name)
            WHERE r.status = 'crashed' AND r.tag = ?
            ORDER BY r.ran_at DESC
            LIMIT 30
        """
        crash_params = (tag,)
        hung_q = """
            SELECT c.name, r.M, r.N, r.K
            FROM runs r JOIN configs c USING(name)
            WHERE r.status = 'hung' AND r.tag = ?
            ORDER BY r.ran_at DESC
            LIMIT 30
        """
        hung_params = (tag,)
    else:
        crash_q = """
            SELECT c.name, r.M, r.N, r.K, r.error_text
            FROM runs r JOIN configs c USING(name)
            WHERE r.status = 'crashed'
            ORDER BY r.ran_at DESC
            LIMIT 30
        """
        crash_params = ()
        hung_q = """
            SELECT c.name, r.M, r.N, r.K
            FROM runs r JOIN configs c USING(name)
            WHERE r.status = 'hung'
            ORDER BY r.ran_at DESC
            LIMIT 30
        """
        hung_params = ()

    rc = conn.execute(crash_q, crash_params).fetchall()
    tag_label = f" [tag={tag}]" if tag else ""
    print(f"\n── Runtime crashes ({len(rc)}){tag_label} ──────────────────────────────")
    if rc:
        for r in rc:
            err = (r["error_text"] or "")[:120].replace("\n", " ")
            print(f"  {r['M']}×{r['N']}×{r['K']}  {r['name'][:60]}")
            print(f"    → {err}")
    else:
        print("  None.")

    rh = conn.execute(hung_q, hung_params).fetchall()
    print(f"\n── Runtime hangs ({len(rh)}){tag_label} ────────────────────────────────")
    if rh:
        for r in rh:
            print(f"  {r['M']}×{r['N']}×{r['K']}  {r['name'][:60]}")
    else:
        print("  None.")

    rj_q = "SELECT cutlass_reason, COUNT(*) AS n FROM runs WHERE status='rejected'"
    rj_params: tuple = ()
    if tag:
        rj_q += " AND tag=?"
        rj_params = (tag,)
    rj_q += " GROUP BY cutlass_reason ORDER BY n DESC"
    rj = conn.execute(rj_q, rj_params).fetchall()
    print(f"\n── Runtime rejections by reason{tag_label} ─────────────────────────")
    if rj:
        for r in rj:
            print(f"  {r['n']:>7,}  {r['cutlass_reason']}")
    else:
        print("  None.")

    print()


def cmd_export(conn: sqlite3.Connection, args) -> None:
    """Export successful runs joined with config params and NCU metrics to Parquet or CSV."""
    try:
        import pandas as pd
    except ImportError:
        sys.exit("pandas is required for export. Install it with: pip install pandas pyarrow")

    tag = args.tag
    tag_filter = "AND r.tag = :tag" if tag else ""

    df = pd.read_sql(f"""
        SELECT
            r.M, r.N, r.K,
            r.mean_tflops, r.std_tflops, r.mean_ms, r.std_ms,
            r.tag,
            c.name, c.cutlass_type_a, c.cutlass_type_b, c.cutlass_type_c,
            c.tile_m, c.tile_n, c.tile_k,
            c.cluster_m, c.cluster_n, c.cluster_k,
            c.stages, c.kernel_schedule, c.epilogue_schedule,
            c.scheduler, c.layout_a, c.layout_b,
            c.alignment_a, c.alignment_b,
            n.compute_throughput_pct,
            n.dram_throughput_pct,
            n.sm_active_pct,
            n.tensor_active_pct,
            n.tma_active_pct,
            n.l2_throughput_pct,
            n.l2_hit_rate,
            n.l2_read_sectors,
            n.l2_write_sectors,
            n.dram_read_bytes,
            n.dram_write_bytes,
            n.smem_bank_conflicts_ld,
            n.smem_bank_conflicts_st,
            n.shmem_ld_wavefronts,
            n.shmem_st_wavefronts,
            n.achieved_occupancy_pct,
            n.warps_active,
            n.registers_per_thread,
            n.smem_static_bytes,
            n.smem_dynamic_bytes,
            n.grid_size,
            n.block_size,
            n.stall_mio_pct,
            n.stall_long_scoreboard_pct,
            n.stall_short_scoreboard_pct,
            n.stall_barrier_pct,
            n.stall_gmma_pct,
            n.issued_ipc,
            n.executed_ipc
        FROM runs r
        JOIN configs c USING(name)
        LEFT JOIN ncu_runs n ON r.name = n.name AND r.M = n.M AND r.N = n.N AND r.K = n.K
            AND n.status = 'success'
        WHERE r.status = 'success'
        {tag_filter}
        ORDER BY r.M, r.N, r.K, r.mean_tflops DESC
    """, conn, params={"tag": tag} if tag else {})

    out = Path(args.out)
    fmt = args.format or ("csv" if out.suffix == ".csv" else "parquet")
    out.parent.mkdir(parents=True, exist_ok=True)

    if fmt == "parquet":
        df.to_parquet(out, index=False)
    else:
        df.to_csv(out, index=False)

    ncu_rows = df["compute_throughput_pct"].notna().sum()
    tag_note = f"  (tag={tag})" if tag else ""
    print(f"\n  Exported {len(df):,} rows → {out}  ({out.stat().st_size / 1024:.0f} KB){tag_note}")
    print(f"  NCU metrics present in {ncu_rows:,} / {len(df):,} rows\n")


def cmd_estimate(conn: sqlite3.Connection, args) -> None:
    """Estimate remaining wall-clock time from live registry state."""
    MIN_SAMPLES = 30
    CI_BUCKETS  = 5
    GAP_SEC     = 15 * 60
    tag = args.tag

    def _pts(s: str) -> float:
        return datetime.fromisoformat(s).timestamp()

    def _fmt_dur(sec) -> str:
        if sec is None:
            return "—"
        sec = max(0, int(sec))
        d, r = divmod(sec, 86400)
        h, r = divmod(r, 3600)
        m    = r // 60
        if d:
            return f"{d}d {h}h {m}m"
        if h:
            return f"{h}h {m}m"
        return f"{m}m"

    def _fmt_ci(lo, hi) -> str:
        return f"[{_fmt_dur(lo)} – {_fmt_dur(hi)}]"

    def _detect_session(ts_iso: list[str]):
        if not ts_iso:
            return None, True, False
        ts  = sorted(_pts(t) for t in ts_iso)
        now = datetime.now(timezone.utc).timestamp()
        is_paused = (now - ts[-1]) > GAP_SEC
        if len(ts) < 2:
            return ts[0], is_paused, False
        gaps    = [ts[i + 1] - ts[i] for i in range(len(ts) - 1)]
        max_gap = max(gaps)
        had     = max_gap > GAP_SEC
        start   = ts[gaps.index(max_gap) + 1] if had else ts[0]
        return start, is_paused, had

    def _compute_rate(ts_sorted: list[float]):
        n = len(ts_sorted)
        if n < MIN_SAMPLES:
            return None, n
        t0, t1 = ts_sorted[0], ts_sorted[-1]
        dur = t1 - t0
        if dur < 1.0:
            return None, n
        rate = (n - 1) / dur
        bw   = dur / CI_BUCKETS
        bucket_rates = [
            sum(1 for t in ts_sorted if t0 + i * bw <= t < t0 + (i + 1) * bw) / bw
            for i in range(CI_BUCKETS)
        ]
        margin = 1.96 * statistics.stdev(bucket_rates) / math.sqrt(CI_BUCKETS)
        ci_lo = rate - margin
        ci_hi = rate + margin
        if ci_lo < 0.1 * rate:
            return (rate, None, None), n
        return (rate, max(1e-9, ci_lo), ci_hi), n

    compile_ts_iso = [r[0] for r in conn.execute(
        "SELECT compiled_at FROM configs WHERE compiled_at IS NOT NULL ORDER BY compiled_at"
    ).fetchall()]

    if tag:
        eval_ts_iso = [r[0] for r in conn.execute("""
            SELECT r.ran_at FROM runs r
            JOIN eval_plan p ON r.name=p.name AND r.M=p.M AND r.N=p.N AND r.K=p.K
            WHERE p.tag=? AND r.ran_at IS NOT NULL
            ORDER BY r.ran_at
        """, (tag,)).fetchall()]
    else:
        eval_ts_iso = [r[0] for r in conn.execute(
            "SELECT ran_at FROM runs WHERE ran_at IS NOT NULL ORDER BY ran_at"
        ).fetchall()]

    ncu_ts_iso = [r[0] for r in conn.execute(
        "SELECT profiled_at FROM ncu_runs WHERE profiled_at IS NOT NULL ORDER BY profiled_at"
    ).fetchall()]

    all_ts_iso = sorted(set(compile_ts_iso + eval_ts_iso + ncu_ts_iso))
    session_start, is_paused, had_restart = _detect_session(all_ts_iso)

    def _in_session(iso_list):
        ts = sorted(_pts(t) for t in iso_list)
        return [t for t in ts if session_start is None or t >= session_start]

    c_ts = _in_session(compile_ts_iso)
    e_ts = _in_session(eval_ts_iso)
    n_ts = _in_session(ncu_ts_iso)

    c_rate_tup, c_n = _compute_rate(c_ts)
    e_rate_tup, e_n = _compute_rate(e_ts)
    n_rate_tup, n_n = _compute_rate(n_ts)

    if tag:
        n_pending_compile = conn.execute("""
            SELECT COUNT(DISTINCT c.name) FROM configs c
            JOIN eval_plan p ON c.name=p.name
            WHERE p.tag=? AND c.compile_status='pending'
        """, (tag,)).fetchone()[0]
        cc = dict(conn.execute("""
            SELECT c.compile_status, COUNT(DISTINCT c.name) FROM configs c
            JOIN eval_plan p ON c.name=p.name
            WHERE p.tag=? AND c.compile_status IN ('success','failed')
            GROUP BY c.compile_status
        """, (tag,)).fetchall())
    else:
        n_pending_compile = conn.execute(
            "SELECT COUNT(*) FROM configs WHERE compile_status='pending'"
        ).fetchone()[0]
        cc = dict(conn.execute(
            "SELECT compile_status, COUNT(*) FROM configs "
            "WHERE compile_status IN ('success','failed') GROUP BY compile_status"
        ).fetchall())

    n_ok  = cc.get("success", 0)
    n_bad = cc.get("failed", 0)
    p_suc = n_ok / (n_ok + n_bad) if (n_ok + n_bad) else None

    if tag:
        n_pending_eval = conn.execute("""
            SELECT COUNT(*) FROM eval_plan p
            LEFT JOIN runs r ON r.name=p.name AND r.M=p.M AND r.N=p.N AND r.K=p.K
            WHERE p.tag=? AND r.status IS NULL
        """, (tag,)).fetchone()[0]
        n_shapes = conn.execute(
            "SELECT COUNT(DISTINCT M||','||N||','||K) FROM eval_plan WHERE tag=?", (tag,)
        ).fetchone()[0]
        n_eval_future = None   # plan is fixed; no future pairs from pending compiles
        n_eval_total  = n_pending_eval
    else:
        shapes   = BENCHMARK_SHAPES
        n_shapes = len(shapes)
        n_pending_eval = sum(
            conn.execute("""
                SELECT COUNT(*) FROM configs c
                LEFT JOIN runs r ON c.name = r.name AND r.M = ? AND r.N = ? AND r.K = ?
                WHERE c.compile_status = 'success' AND r.status IS NULL
            """, (M, N, K)).fetchone()[0]
            for M, N, K in shapes
        )
        n_eval_future = int(n_pending_compile * p_suc * n_shapes) if p_suc else None
        n_eval_total  = n_pending_eval + (n_eval_future or 0)

    run_counts = dict(conn.execute("SELECT status, COUNT(*) FROM runs GROUP BY status").fetchall())
    n_runs_total   = sum(run_counts.values())
    n_runs_success = run_counts.get("success", 0)

    ncu_counts = dict(conn.execute("SELECT status, COUNT(*) FROM ncu_runs GROUP BY status").fetchall())
    n_done_ncu    = ncu_counts.get("success", 0) + ncu_counts.get("failed", 0)
    n_pending_ncu = conn.execute("""
        SELECT COUNT(*) FROM configs c
        JOIN runs r ON c.name = r.name AND r.status = 'success'
        LEFT JOIN ncu_runs n ON c.name = n.name AND n.M = r.M AND n.N = r.N AND n.K = r.K
        WHERE n.name IS NULL OR n.status = 'pending'
    """).fetchone()[0]

    now = datetime.now(timezone.utc).timestamp()

    print(f"\n  DB            : {args.db}  (last modified: {_age(args.db)})")
    if tag:
        print(f"  Tag           : {tag}")
    print("\n── Estimate ──────────────────────────────────────────────────────────────")
    if session_start is not None:
        last_ts   = _pts(all_ts_iso[-1])
        restart_s = "  (restart detected — using post-restart data only)" if had_restart else ""
        print(f"  Session   :  {_fmt_dur(last_ts - session_start)} active"
              f" · last event {_fmt_dur(now - last_ts)} ago"
              f"  [{'PAUSED' if is_paused else 'LIVE'}]{restart_s}")
    else:
        print("  Session   :  no activity recorded yet")

    print("\n── Compile ───────────────────────────────────────────────────────────────")
    print(f"  Pending   :  {n_pending_compile:,} configs")
    if p_suc is not None:
        print(f"  p(success):  {p_suc * 100:.0f}%  ({n_ok:,} success · {n_bad:,} failed)")

    t_compile = t_c_lo = t_c_hi = None

    if c_rate_tup is not None:
        rate, lo, hi = c_rate_tup
        ci_str = (f"  [95% CI: {lo * 60:.1f} – {hi * 60:.1f}]"
                  if lo is not None else "  [CI unreliable — events clustered]")
        print(f"  Rate      :  {rate * 60:.1f} / min{ci_str}  (n={c_n})")
        if n_pending_compile > 0:
            t_compile = n_pending_compile / rate
            t_c_lo    = n_pending_compile / hi if hi else None
            t_c_hi    = n_pending_compile / lo if lo else None
            eta_ci    = f"  {_fmt_ci(t_c_lo, t_c_hi)}" if t_c_lo is not None else ""
            print(f"  ETA       :  ~{_fmt_dur(t_compile)}{eta_ci}")
        else:
            t_compile = t_c_lo = t_c_hi = 0.0
            print("  ETA       :  done")
    else:
        print(f"  Rate      :  not enough data  ({c_n} / {MIN_SAMPLES} samples needed)")

    print("\n── Eval ──────────────────────────────────────────────────────────────────")
    if n_shapes == 0:
        n_compiled_waiting = conn.execute(
            "SELECT COUNT(*) FROM configs WHERE compile_status='success'"
        ).fetchone()[0]
        print(f"  Status             :  not started  ({n_compiled_waiting:,} compiled configs waiting)")
    else:
        print(f"  Pending pairs      :  {n_pending_eval:,}  ({n_shapes} shapes)")
        if n_eval_future is not None:
            print(f"  Future (estimated) :  ~{n_eval_future:,} pairs"
                  f"  ({n_pending_compile:,} pending × {p_suc * 100:.0f}% × {n_shapes} shapes)")
        print(f"  Total to evaluate  :  ~{n_eval_total:,} pairs")

    t_eval = t_e_lo = t_e_hi = t_ef = None

    if e_rate_tup is not None:
        rate, lo, hi = e_rate_tup
        ci_str = (f"  [95% CI: {lo * 60:.1f} – {hi * 60:.1f}]"
                  if lo is not None else "  [CI unreliable — events clustered]")
        print(f"  Rate               :  {rate * 60:.1f} pairs/min{ci_str}  (n={e_n})")
        if n_eval_total > 0:
            t_eval = n_eval_total / rate
            t_e_lo = n_eval_total / hi if hi else None
            t_e_hi = n_eval_total / lo if lo else None
            t_ef   = (n_eval_future / rate) if n_eval_future else 0.0
            eta_ci = f"  {_fmt_ci(t_e_lo, t_e_hi)}" if t_e_lo is not None else ""
            print(f"  Time for total     :  ~{_fmt_dur(t_eval)}{eta_ci}"
                  f"  (if all pairs available now)")
        else:
            t_eval = t_e_lo = t_e_hi = t_ef = 0.0
            print("  Time for total     :  done")
    else:
        print(f"  Rate               :  not enough data  ({e_n} / {MIN_SAMPLES} samples needed)")

    print("\n── NCU ───────────────────────────────────────────────────────────────────")
    if n_done_ncu == 0 and n_pending_ncu == 0:
        print("  Status    :  not started")
    else:
        print(f"  Done      :  {n_done_ncu:,} pairs")
        print(f"  Pending   :  {n_pending_ncu:,} pairs")
        if n_rate_tup is not None:
            rate, lo, hi = n_rate_tup
            ci_str = (f"  [95% CI: {lo * 60:.1f} – {hi * 60:.1f}]"
                      if lo is not None else "  [CI unreliable — events clustered]")
            print(f"  Rate      :  {rate * 60:.1f} pairs/min{ci_str}  (n={n_n})")
            if n_pending_ncu > 0:
                t_ncu  = n_pending_ncu / rate
                t_n_lo = n_pending_ncu / hi if hi else None
                t_n_hi = n_pending_ncu / lo if lo else None
                eta_ci = f"  {_fmt_ci(t_n_lo, t_n_hi)}" if t_n_lo is not None else ""
                print(f"  ETA       :  ~{_fmt_dur(t_ncu)}{eta_ci}")
            else:
                t_ncu = t_n_lo = t_n_hi = 0.0
                print("  ETA       :  done")
        else:
            print(f"  Rate      :  not enough data  ({n_n} / {MIN_SAMPLES} samples needed)")

    if t_compile is not None and t_eval is not None:
        ef     = t_ef or 0.0
        ef_lo  = (n_eval_future / e_rate_tup[2]) if n_eval_future and e_rate_tup[2] else 0.0
        ef_hi  = (n_eval_future / e_rate_tup[1]) if n_eval_future and e_rate_tup[1] else 0.0

        compile_path = t_compile + ef
        eval_path    = t_eval
        t_wall       = max(compile_path, eval_path)
        bottleneck   = "eval-bound" if eval_path >= compile_path else "compile-bound"

        if bottleneck == "eval-bound":
            t_w_lo, t_w_hi = t_e_lo, t_e_hi
        else:
            t_w_lo = (t_c_lo + ef_lo) if t_c_lo is not None else None
            t_w_hi = (t_c_hi + ef_hi) if t_c_hi is not None else None

        wall_ci = f"  {_fmt_ci(t_w_lo, t_w_hi)}" if t_w_lo is not None else ""
        print("\n── Critical path ─────────────────────────────────────────────────────────")
        print(f"  Compile finishes  :  +{_fmt_dur(t_compile)}")
        print(f"  All eval done     :  +{_fmt_dur(t_wall)}  ← {bottleneck}")
        t_ncu_val = locals().get("t_ncu")
        if t_ncu_val is not None and t_ncu_val > 0:
            t_n_lo_val = locals().get("t_n_lo")
            t_n_hi_val = locals().get("t_n_hi")
            t_total    = t_wall + t_ncu_val
            t_total_lo = (t_w_lo + t_n_lo_val) if (t_w_lo and t_n_lo_val) else None
            t_total_hi = (t_w_hi + t_n_hi_val) if (t_w_hi and t_n_hi_val) else None
            total_ci   = f"  {_fmt_ci(t_total_lo, t_total_hi)}" if t_total_lo else ""
            print(f"  NCU done          :  +{_fmt_dur(t_total)}{total_ci}")
            print(f"  Wall-clock ETA    :  ~{_fmt_dur(t_total)}{total_ci}")
        else:
            print(f"  Wall-clock ETA    :  ~{_fmt_dur(t_wall)}{wall_ci}")

    warns = []
    if p_suc is not None and p_suc < 0.5:
        warns.append(f"compile success rate is only {p_suc * 100:.0f}% — check config generation")
    if n_runs_total > 50 and n_runs_success / n_runs_total < 0.3:
        warns.append(
            f"only {n_runs_success * 100 // n_runs_total}% of benchmark runs succeeded"
            f" — many kernels rejected or crashed"
        )
    if is_paused and (t_compile or t_eval):
        warns.append("run appears paused — verify it will restart before committing node hours")
    if had_restart and (c_n < 50 or e_n < 50):
        warns.append("few post-restart samples — estimate will improve as session matures")

    if warns:
        print()
        for w in warns:
            print(f"  ⚠  {w}")

    print()


def cmd_repair(conn: sqlite3.Connection, args) -> None:
    """Clear retryable crashed/hung runs for a tag (safe before resume)."""
    tag = args.tag
    if not tag:
        print("repair requires --tag", file=sys.stderr)
        sys.exit(1)
    n = conn.execute(
        """
        DELETE FROM runs
        WHERE status IN ('crashed', 'hung')
          AND name IN (SELECT DISTINCT name FROM eval_plan WHERE tag=?)
        """,
        (tag,),
    ).rowcount
    conn.commit()
    print(f"Cleared {n:,} crashed/hung run rows for tag={tag}")
    print("Missing .so requeue happens automatically at scheduler startup per node.")


# ── Entry point ───────────────────────────────────────────────────────────────

def _pop_global_args(argv: list[str]) -> tuple[list[str], Path | None, str | None]:
    """Pull --db / --tag out of argv (any position) so argparse subcommands stay simple."""
    rest: list[str] = []
    db_path: Path | None = None
    tag: str | None = None
    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg == "--db":
            if i + 1 >= len(argv):
                sys.exit("query_db.py: --db requires a path")
            db_path = Path(argv[i + 1])
            i += 2
        elif arg.startswith("--db="):
            db_path = Path(arg.split("=", 1)[1])
            i += 1
        elif arg == "--tag":
            if i + 1 >= len(argv):
                sys.exit("query_db.py: --tag requires a value")
            tag = argv[i + 1]
            i += 2
        elif arg.startswith("--tag="):
            tag = arg.split("=", 1)[1]
            i += 1
        else:
            rest.append(arg)
            i += 1
    return rest, db_path, tag


def main():
    argv, db_override, tag_override = _pop_global_args(sys.argv[1:])

    parser = argparse.ArgumentParser(
        description="Inspect the autotuner registry (autotuner.db)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("summary",  help="Overall pipeline progress")
    sub.add_parser("shapes",   help="Per-shape benchmark progress")
    sub.add_parser("failures", help="Compile failures and runtime crashes/hangs")
    sub.add_parser("estimate", help="Estimate remaining wall-clock time")
    sub.add_parser("repair", help="Clear crashed/hung runs before resume (requires --tag)")

    p_best = sub.add_parser("best", help="Top configs for a shape by TFLOPS")
    p_best.add_argument("M", type=int)
    p_best.add_argument("N", type=int)
    p_best.add_argument("K", type=int)
    p_best.add_argument("--n", type=int, default=15, help="Number of results")

    p_ncu = sub.add_parser("ncu", help="Top profiled configs for a shape with NCU metrics")
    p_ncu.add_argument("M", type=int)
    p_ncu.add_argument("N", type=int)
    p_ncu.add_argument("K", type=int)
    p_ncu.add_argument("--n", type=int, default=15, help="Number of results")

    p_exp = sub.add_parser("export", help="Export runs + NCU metrics to Parquet or CSV")
    p_exp.add_argument("--out",    default="results.parquet", help="Output file path")
    p_exp.add_argument("--format", choices=["parquet", "csv"],
                       help="Output format (inferred from --out extension if omitted)")

    args = parser.parse_args(argv)
    args.db = db_override or _find_db()
    args.tag = tag_override
    conn = _connect(args.db)

    dispatch = {
        "summary":  cmd_summary,
        "shapes":   cmd_shapes,
        "best":     cmd_best,
        "ncu":      cmd_ncu,
        "failures": cmd_failures,
        "export":   cmd_export,
        "estimate": cmd_estimate,
        "repair":   cmd_repair,
    }
    dispatch[args.cmd](conn, args)
    conn.close()



if __name__ == "__main__":
    main()
