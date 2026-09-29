"""Autotuner for fused-epilogue GEMMs (bias, ReLU, GELU, ...).

Same compile / eval / ncu phases as scheduler.py, over the fusion config space
(config_space_fusion.py) and templates/hopper_template_fusion.cu.j2. Kernels are
batched per fusion kind. Single node only.

Usage:
    python src/autotuner/scheduler_fusion.py --tag sweep_fusion_fp16 --phase compile,eval --num-gpus 4
"""

import argparse
import hashlib
import json
import logging
import multiprocessing
import os
import queue
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

from jinja2 import Environment, FileSystemLoader
from ncu_worker import NCU_TIMEOUT_S, ncu_worker_main
from profile_worker import profile_worker_main
from registry import SQLiteRegistry

from repo_paths import REPO_ROOT, SRC

logger = logging.getLogger(__name__)

_shutdown = threading.Event()


def _handle_signal(sig, frame):
    _shutdown.set()


SCRIPT_DIR = Path(__file__).resolve().parent

from config_space_fusion import (
    enrich_fusion_from_name,
    generate_search_space,
)

WORKER_TIMEOUT_S = 10

# Benchmark shapes (M, N, K)
#
# Compute-bound squares: roofline peak, tests tile efficiency at scale
# Memory-bound squares: small footprint, tests occupancy and latency hiding
# Tall-skinny (large M, small N): token-level projections in LLM decode/prefill
# Wide-short (small M, large N): opposite projection direction, tests transposed path
# Weird/interesting: real workload shapes that stress specific hardware behaviors
BENCHMARK_SHAPES = [
    # Compute-bound squares
    (2048, 2048, 2048),
    (4096, 4096, 4096),
    # Memory-bound squares
    (64, 64, 64),
    (128, 128, 128),
    (256, 256, 256),
    (512, 512, 512),
    # Tall-skinny: large M, small N (e.g. LLM decode down-projections)
    # K=4096 assumed
    (32, 128, 4096),
    (2048, 128, 4096),
    (4096, 128, 4096),
    (12288, 128, 4096),
    # Wide-short: small M, large N (opposite projection direction)
    # K=4096 assumed
    (256, 4096, 4096),
    (256, 12288, 4096),
    (64, 12288, 4096),
    # Weird/interesting cases from real workloads
    # Attention QK^T: large square M×N but tiny K (head dimension) — unusual roofline point where cuBLAS heuristics often pick poorly
    (2048, 2048, 128),
    # LLaMA 2 FFN intermediate dim (11008 is not a power of 2) — stresses tile alignment
    (4096, 11008, 4096),
    # Wide-K reduction: K >> M, N — dominates reduction path, rare but real in some attention variants
    (256, 256, 8192),
    # BERT base FFN: different scale and aspect ratio than LLaMA-scale shapes
    (512, 3072, 768),
]


def _batch_hash(configs: list) -> str:
    """Content hash used only for naming .so files — not a cache key."""
    return hashlib.sha256(json.dumps(configs, sort_keys=True).encode()).hexdigest()[:16]


def _family_key(config: dict) -> tuple:
    """Stable group key for batch compilation (includes fusion kind)."""
    fusion = config.get("fusion")
    if fusion is None and "_fusion_" in config["name"]:
        fusion = config["name"].rsplit("_fusion_", 1)[1]
    return (
        config["cutlass_type_a"],
        config["kernel_schedule"],
        config["epilogue_schedule"],
        config["scheduler"],
        config["layout_a"],
        config["layout_b"],
        fusion or "linear",
    )




def try_compile(
    configs: list[dict],
    build_dir: Path,
    cutlass_dir: Path,
    template,
) -> tuple[bool, str, Path]:
    """Compile a batch of configs into a shared library.

    Returns (success, error_text, so_file). The .so filename is derived from
    a content hash of the configs — same search space always produces the same name.
    If the .so already exists on disk it is reused without recompiling.
    """
    hash_name = _batch_hash(configs)
    cu_file = build_dir / f"batch_{hash_name}.cu"
    so_file = build_dir / f"batch_{hash_name}.so"

    if so_file.exists():
        return True, "", so_file

    cu_file.write_text(template.render(kernels=configs))
    cmd = [
        "nvcc",
        "-shared",
        "-Xcompiler=-fPIC",
        "-O3",
        "-arch=sm_90a",
        "--std=c++17",
        "--expt-relaxed-constexpr",
        "-Xptxas=-w",
        f"-I{cutlass_dir / 'include'}",
        f"-I{cutlass_dir / 'tools' / 'util' / 'include'}",
        str(cu_file),
        "-o",
        str(so_file),
    ]

    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True)
        return True, "", so_file
    except subprocess.CalledProcessError as e:
        so_file.unlink(missing_ok=True)
        if _shutdown.is_set():
            return None, "", so_file
        return False, e.stdout + e.stderr, so_file


def compiler_worker(compile_q, eval_q, registry, args, template, run_eval):
    """Compile thread: build batches from compile_q and hand compiled configs to eval_q.

    A failing batch is split into single-config compilations so one bad config does not
    cost the whole batch. Stops on a None sentinel.
    """
    while True:
        item = compile_q.get()
        if item is None:
            break

        cancelled = False
        try:
            batch_idx, total_batches, batch = item
            prefix = f"[{batch_idx + 1:0{len(str(total_batches))}d}/{total_batches}]"
            success, _, so_file = try_compile(batch, args.build_dir, args.cutlass_dir, template)

            if success:
                logger.info("[Compiler] %s Built %d kernels → %s", prefix, len(batch), so_file.name)
                for kid, config in enumerate(batch):
                    registry.record_compile_success(config["name"], so_file.name, kid)
                    if run_eval:
                        eval_q.put(config["name"])
            elif success is None:
                cancelled = True  # job cancelled, configs stay pending
            else:
                logger.warning(
                    "[Compiler] %s Batch failed — shattering into %d individual compilations",
                    prefix,
                    len(batch),
                )
                ok_count = 0
                for config in batch:
                    ok, err, single_so = try_compile([config], args.build_dir, args.cutlass_dir, template)
                    if ok:
                        ok_count += 1
                        registry.record_compile_success(config["name"], single_so.name, 0)
                        if run_eval:
                            eval_q.put(config["name"])
                    elif ok is None:
                        cancelled = True  # job cancelled mid-shatter, configs stay pending
                        break
                    else:
                        registry.record_compile_failure(config["name"], err)
                        logger.warning("  [Compiler] Failed: %s", config["name"])
                if not cancelled:
                    logger.info("[Compiler] %s Shatter done: %d/%d succeeded", prefix, ok_count, len(batch))
        finally:
            compile_q.task_done()

        if cancelled:
            break


def monitor_workers(worker_procs, eval_q, registry, shapes, build_dir, db_path, stop_evt, tag=None):
    """Watch worker processes and restart any that crash or hang.

    On crash: attributes the failure to the kernel being benchmarked (read from
    worker_state), marks its first unrecorded shape as crashed, calls task_done()
    on behalf of the dead worker, re-queues the kernel so remaining shapes are
    still attempted, then starts a fresh replacement process.

    On hang (timeout): marks the first pending shape as 'hung', re-queues the
    config so remaining shapes are still attempted on a fresh worker. Kills the
    stuck process with SIGKILL.
    """
    while not stop_evt.is_set():
        now = time.time()
        for i, proc in enumerate(worker_procs):
            # The monitor is the only thing that restarts crashed workers and balances
            # task_done() for them; if it ever dies, eval_q.join() deadlocks for the rest
            # of the job. Nothing inside this body is allowed to kill the thread — any
            # unexpected error is logged and the next worker is checked on the next tick.
            try:
                # Timeout is measured from when the worker last called set_worker_state()
                config_name, updated_at_iso = registry.get_worker_state_and_time(i)
                elapsed = (now - datetime.fromisoformat(updated_at_iso).timestamp()) if updated_at_iso else 0.0
                timed_out = proc.is_alive() and bool(config_name) and elapsed > WORKER_TIMEOUT_S

                if not timed_out and (proc.is_alive() or proc.exitcode == 0):
                    continue  # running normally, or exited cleanly via None sentinel

                if timed_out:
                    logger.warning(
                        "[Monitor] GPU %d worker timed out after %.0fs on config (pid=%d) — killing",
                        i, elapsed, proc.pid,
                    )
                    proc.kill()
                    proc.join(timeout=5)

                gpu_id = i
                if config_name:
                    if tag:
                        pending_shapes = registry.plan_shapes_for_config(config_name, tag)
                    else:
                        pending_shapes = [
                            (M, N, K) for M, N, K in shapes
                            if registry.run_status(config_name, M, N, K) is None
                        ]

                    if timed_out:
                        logger.warning("[Monitor] GPU %d hung on %s — marking current shape as hung, retrying remainder", gpu_id, config_name)
                        if pending_shapes:
                            M, N, K = pending_shapes[0]
                            registry.record_hung(config_name, M, N, K, tag=tag)
                    else:
                        logger.warning("[Monitor] GPU %d crashed on %s (exit %d)", gpu_id, config_name, proc.exitcode)
                        if pending_shapes:
                            M, N, K = pending_shapes[0]
                            registry.record_crashed(config_name, M, N, K, f"worker exited {proc.exitcode}", tag=tag)
                    # Re-queue only if shapes remain after this one
                    if len(pending_shapes) > 1:
                        eval_q.put(config_name)
                    try:
                        eval_q.task_done()  # the dead/hung worker never called this
                    except ValueError:
                        # Racy read: get_worker_state_and_time() is sampled at the top of the
                        # loop, but proc liveness is checked a moment later. In that window the
                        # worker can finish this config (calling its own task_done) and die on
                        # the *next* item before writing it to worker_state — so we see a stale
                        # name whose task_done is already balanced. task_done() raises *before*
                        # decrementing, so swallowing this leaves the queue count correct.
                        logger.warning("[Monitor] GPU %d: task_done already balanced for %s — ignoring", gpu_id, config_name)
                    registry.clear_worker_state(gpu_id)
                else:
                    # Worker died while idle (between clear_worker_state and task_done,
                    # or at startup before first get). task_done() was already called.
                    logger.warning("[Monitor] GPU %d died while idle (exit %s)", gpu_id, "timeout" if timed_out else proc.exitcode)

                new_proc = multiprocessing.Process(
                    target=profile_worker_main,
                    args=(gpu_id, eval_q, build_dir, shapes, db_path, tag),
                    name=f"worker-gpu{gpu_id}",
                )
                new_proc.start()
                worker_procs[i] = new_proc
                logger.info("[Monitor] Restarted GPU %d worker (pid=%d)", gpu_id, new_proc.pid)
            except Exception:
                logger.exception("[Monitor] GPU %d handler hit an unexpected error — continuing", i)

        stop_evt.wait(timeout=1.0)


def _run_compile_eval_phase(registry, args, db_path, run_compile, run_eval, batches, num_compilers, pending_eval, shapes=None, tag=None):
    if shapes is None:
        shapes = BENCHMARK_SHAPES

    eval_q = multiprocessing.JoinableQueue()
    stop_evt = threading.Event()

    if run_eval:
        worker_procs: list[multiprocessing.Process] = []
        for gpu_id in range(args.num_gpus):
            proc = multiprocessing.Process(
                target=profile_worker_main,
                args=(gpu_id, eval_q, args.build_dir, shapes, db_path, tag),
                name=f"worker-gpu{gpu_id}",
            )
            proc.start()
            worker_procs.append(proc)

        monitor = threading.Thread(
            target=monitor_workers,
            args=(worker_procs, eval_q, registry, shapes, args.build_dir, db_path, stop_evt, tag),
            daemon=True,
            name="monitor",
        )
        monitor.start()

        for name in pending_eval:
            eval_q.put(name)

    if run_compile:
        template = Environment(loader=FileSystemLoader(str(args.template_dir))).get_template(args.template)
        compile_q = queue.Queue()
        compilers = [
            threading.Thread(target=compiler_worker, args=(compile_q, eval_q, registry, args, template, run_eval))
            for _ in range(num_compilers)
        ]
        for t in compilers:
            t.start()
        for i, batch in enumerate(batches):
            compile_q.put((i, len(batches), batch))
        for _ in range(num_compilers):
            compile_q.put(None)
        for t in compilers:
            t.join()

    if run_eval:
        eval_q.join()
        stop_evt.set()
        monitor.join()
        for _ in worker_procs:
            eval_q.put(None)
        for proc in worker_procs:
            proc.join(timeout=30)
            if proc.is_alive():
                proc.terminate()


def _run_ncu_phase(registry, args, db_path):
    pending_ncu = registry.pending_ncu(BENCHMARK_SHAPES, top_k=args.ncu_top_k)
    filter_note = f"  (top-{args.ncu_top_k} per shape per dtype)" if args.ncu_top_k else ""
    print("=" * 50)
    print(f"NCU phase       : {len(pending_ncu):,} pairs{filter_note}")
    print(f"NCU binary      : {args.ncu_bin}")
    print(f"GPUs            : {args.num_gpus}")
    print("=" * 50)

    ncu_q = multiprocessing.JoinableQueue()
    for config, (M, N, K) in pending_ncu:
        ncu_q.put((config["name"], M, N, K))

    ncu_procs: list[multiprocessing.Process] = []
    for gpu_id in range(args.num_gpus):
        proc = multiprocessing.Process(
            target=ncu_worker_main,
            args=(gpu_id, ncu_q, args.build_dir, db_path, args.ncu_bin),
            name=f"ncu-gpu{gpu_id}",
        )
        proc.start()
        ncu_procs.append(proc)

    ncu_q.join()

    for _ in ncu_procs:
        ncu_q.put(None)
    for proc in ncu_procs:
        proc.join(timeout=NCU_TIMEOUT_S + 10)
        if proc.is_alive():
            proc.terminate()


def main():
    # spawn is required for CUDA: forking after CUDA is initialised causes
    # undefined behaviour. spawn starts each worker with a fresh interpreter.
    multiprocessing.set_start_method("spawn")
    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    # fmt: off
    parser = argparse.ArgumentParser(description="CUTLASS GEMM Auto-Tuner (fusion fine-tune)")
    parser.add_argument("--build-dir",      type=Path, default=REPO_ROOT / "build_cache")
    parser.add_argument("--db-path",        type=Path, default=None,    help="SQLite registry path (default: build_dir/autotuner.db)")
    parser.add_argument("--template-dir",   type=Path, default=SCRIPT_DIR / "templates")
    parser.add_argument("--template",       default="hopper_template_fusion.cu.j2",
                        help="Jinja template filename under --template-dir")
    parser.add_argument("--cutlass-dir",    type=Path, default=Path(os.environ.get("CUTLASS_DIR", SRC / "extern" / "cutlass")))
    parser.add_argument("--num-gpus",       type=int,  default=0,       help="Number of GPUs available on this node")
    parser.add_argument("--compile-jobs",   type=int,  default=6,       help="Concurrent nvcc threads")
    parser.add_argument("--max-batch-size", type=int,  default=50,      help="Max configs per .cu file — limits nvcc memory usage")
    parser.add_argument("--phase",                     default="compile,eval", help="Comma-separated phases to run: compile, eval, ncu. Or 'all'.")
    parser.add_argument("--ncu-top-k",      type=int,  default=20,      help="NCU phase: profile only the top-k kernels per shape by TFLOPS (default: 20)")
    parser.add_argument("--ncu-bin",                   default="ncu",   help="Path to the ncu executable (default: ncu, assumed on PATH)")
    parser.add_argument("--tag",                       default=None,    help="Eval-plan tag (e.g. 'sweep'). When set, only compile/eval configs in eval_plan for this tag.")
    # fmt: on
    args = parser.parse_args()

    phases = set(args.phase.split(",")) if args.phase != "all" else {"compile", "eval", "ncu"}
    run_compile = "compile" in phases
    run_eval = "eval" in phases and args.num_gpus > 0
    run_ncu = "ncu" in phases and args.num_gpus > 0

    logging.basicConfig(
        level=logging.INFO, format="[%(asctime)s] %(levelname)s %(message)s", datefmt="%H:%M:%S", stream=sys.stdout
    )

    args.build_dir.mkdir(parents=True, exist_ok=True)
    db_path = args.db_path or args.build_dir / "autotuner.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)

    registry = SQLiteRegistry(db_path)

    for gpu_id in range(args.num_gpus):
        registry.clear_worker_state(gpu_id)

    # Tag-scoped runs compile/eval only configs in eval_plan (written by src/planning/plan.py write).
    # Untagged runs register and compile the full BF16 search space.
    if args.tag:
        pending = registry.plan_pending_compile(args.tag) if run_compile else []
        plan_config_count = registry.plan_distinct_config_count(args.tag)
        shapes = registry.plan_distinct_shapes(args.tag)
        pending_eval = registry.plan_pending_eval_names(args.tag) if run_eval else []
    else:
        space = generate_search_space()
        registry.register(space)
        pending = registry.pending_compile() if run_compile else []
        plan_config_count = len(space)
        shapes = BENCHMARK_SHAPES
        pending_eval = registry.pending_eval_names(BENCHMARK_SHAPES) if run_eval else []

    pending = [enrich_fusion_from_name(dict(c)) for c in pending]
    families: dict[tuple, list[dict]] = {}
    for config in pending:
        families.setdefault(_family_key(config), []).append(config)
    batches = [
        family[i : i + args.max_batch_size]
        for family in families.values()
        for i in range(0, len(family), args.max_batch_size)
    ]
    num_compilers = min(args.compile_jobs, len(batches)) if batches else 0

    # fmt: off
    print("=" * 50)
    print("CUTLASS 3.x Auto-Tuner")
    print("=" * 50)
    if args.tag:
        print(f"Plan tag        : {args.tag}")
        print(f"Plan configs    : {plan_config_count:,}")
        print(f"Planned shapes  : {len(shapes):,}")
    else:
        print(f"Search space    : {plan_config_count:,} configs")
    print(f"Phases          : {', '.join(sorted(phases))}")
    print(f"Pending compile : {len(pending):,} configs in {len(batches)} batches")
    print(f"Pending eval    : {len(pending_eval):,} configs")
    print(f"Registry        : {db_path}")
    print("=" * 50)
    print(f"Compilers       : {num_compilers}")
    print(f"GPUs            : {args.num_gpus}")
    print("=" * 50)
    # fmt: on

    if args.tag and plan_config_count == 0 and (run_compile or run_eval):
        print(
            f"ERROR: eval_plan has no entries for tag {args.tag!r}. "
            f"Run src/planning/plan.py write --tag {args.tag} first.",
            file=sys.stderr,
        )
        sys.exit(1)

    if run_compile or run_eval:
        _run_compile_eval_phase(registry, args, db_path, run_compile, run_eval, batches, num_compilers, pending_eval, shapes=shapes, tag=args.tag)

    if run_ncu and not _shutdown.is_set():
        _run_ncu_phase(registry, args, db_path)

    summary = registry.summary()
    c, r, n = summary.get("compile", {}), summary.get("runs", {}), summary.get("ncu", {})
    # fmt: off
    print("=" * 50)
    print("Run complete.")
    if run_compile or run_eval:
        print(f"  Compile : {c.get('success', 0):,} success  {c.get('failed', 0):,} failed  {c.get('pending', 0):,} pending")
        print(f"  Runs    : {r.get('success', 0):,} success  {r.get('rejected', 0):,} rejected  {r.get('crashed', 0):,} crashed  {r.get('hung', 0):,} hung")
    if run_ncu:
        print(f"  NCU     : {n.get('success', 0):,} success  {n.get('failed', 0):,} failed")
    print(f"  DB      : {db_path}")
    print("=" * 50)
    # fmt: on


if __name__ == "__main__":
    main()
