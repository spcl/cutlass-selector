"""
ncu_worker.py — NCU profiling worker for the CUTLASS GEMM autotuner.

Two modes:

  Worker mode (imported by scheduler.py):
    Persistent per-GPU process. Pulls (name, M, N, K) items from a queue,
    invokes ncu as a subprocess, parses the CSV output, and records metrics
    to the registry.

  NCU target mode (run as __main__):
    Minimal CUDA launcher invoked by ncu itself. Loads a compiled .so,
    allocates buffers, calls run_kernel_once exactly once, then exits.
    ncu intercepts that single kernel launch and replays it internally for
    every hardware counter section — we never need to loop.

    Preferred: ncu_launcher binary (compiled from ncu_launcher.cu).
    Eliminates ~2 s of Python + torch import overhead per pair.
    Falls back to this script (__main__) if the binary is not available.

    Invoked as (binary path):
      ncu --csv --metrics <...> --kernel-name device_kernel --launch-count 1 \\
          --clock-control none --device <id> \\
          <build_dir>/ncu_launcher --build-dir <dir> --so <file> --kernel-id <id> \\
          --dtype-a bf16 --dtype-b bf16 --dtype-c bf16 --device-id <id> \\
          --M <M> --N <N> --K <K>
"""

import argparse
import csv
import ctypes
import io
import logging
import os
import signal
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from registry import SQLiteRegistry

logger = logging.getLogger(__name__)

SCRIPT_PATH = Path(__file__).resolve()
LAUNCHER_SRC = Path(__file__).resolve().parent / "ncu_launcher.cu"
NCU_TIMEOUT_S = 120  # ~2 s × replay passes per metric section; 120 s is safe for all shapes

# ── NCU metric definitions ─────────────────────────────────────────────────────
# Maps DB column name → NCU metric ID (SM90 / Hopper).
# Verify names on target node with: ncu --query-metrics | grep <fragment>

NCU_METRICS = {
    # ── Throughput / roofline ─────────────────────────────────────────────────
    "compute_throughput_pct":    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "dram_throughput_pct":       "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed",
    "sm_active_pct":             "sm__cycles_active.avg.pct_of_peak_sustained_elapsed",
    "tensor_active_pct":         "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed",
    "tma_active_pct":            "sm__pipe_tma_cycles_active.avg.pct_of_peak_sustained_elapsed",
    "fma_active_pct":            "sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_elapsed",
    "smem_pipe_active_pct":      "sm__pipe_shared_cycles_active.avg.pct_of_peak_sustained_elapsed",
    # ── Absolute timing ───────────────────────────────────────────────────────
    # Enables arithmetic intensity = 2MNK / (dram_read_bytes + dram_write_bytes)
    # and achieved TFLOPS = 2MNK / duration_ns independently of the runs table.
    "duration_ns":               "gpu__time_duration.sum",
    # ── Memory hierarchy ─────────────────────────────────────────────────────
    "l2_throughput_pct":         "lts__throughput.avg.pct_of_peak_sustained_elapsed",
    "l2_hit_rate":               "lts__t_sector_hit_rate.pct",
    "l2_read_hit_rate":          "lts__t_sector_op_read_hit_rate.pct",
    "l2_write_hit_rate":         "lts__t_sector_op_write_hit_rate.pct",
    "l2_read_sectors":           "lts__t_sectors_op_read.sum",
    "l2_write_sectors":          "lts__t_sectors_op_write.sum",
    "dram_read_bytes":           "dram__bytes_read.sum",
    "dram_write_bytes":          "dram__bytes_write.sum",
    "l1_hit_rate":               "l1tex__t_sector_hit_rate.pct",
    "l1_read_sectors":           "l1tex__t_sectors_op_read.sum",
    # ── Shared memory ─────────────────────────────────────────────────────────
    "smem_bank_conflicts_ld":    "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum",
    "smem_bank_conflicts_st":    "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum",
    "shmem_ld_wavefronts":       "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum",
    "shmem_st_wavefronts":       "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum",
    # ── Occupancy / launch config ─────────────────────────────────────────────
    "achieved_occupancy_pct":    "sm__warps_active.avg.pct_of_peak_sustained_active",
    "warps_active":              "sm__warps_active.avg.per_cycle_active",
    "eligible_warps_per_cycle":  "smsp__warps_eligible.avg.per_cycle_active",
    "registers_per_thread":      "launch__registers_per_thread",
    "smem_static_bytes":         "launch__shared_mem_per_block_static",
    "smem_dynamic_bytes":        "launch__shared_mem_per_block_dynamic",
    "grid_size":                 "launch__grid_size",
    "block_size":                "launch__block_size",
    "waves_per_sm":              "launch__waves_per_multiprocessor",
    # ── Warp stalls ───────────────────────────────────────────────────────────
    "stall_mio_pct":             "smsp__warp_issue_stalled_mio_throttle_per_warp_active.pct",
    "stall_long_scoreboard_pct": "smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct",
    "stall_short_scoreboard_pct":"smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct",
    "stall_barrier_pct":         "smsp__warp_issue_stalled_barrier_per_warp_active.pct",
    "stall_gmma_pct":            "smsp__warp_issue_stalled_gmma_per_warp_active.pct",
    "stall_drain_pct":           "smsp__warp_issue_stalled_drain_per_warp_active.pct",
    "stall_membar_pct":          "smsp__warp_issue_stalled_membar_per_warp_active.pct",
    "stall_wait_pct":            "smsp__warp_issue_stalled_wait_per_warp_active.pct",
    "stall_not_selected_pct":    "smsp__warp_issue_stalled_not_selected_per_warp_active.pct",
    "stall_no_instructions_pct": "smsp__warp_issue_stalled_no_instructions_per_warp_active.pct",
    "stall_tex_throttle_pct":    "smsp__warp_issue_stalled_tex_throttle_per_warp_active.pct",
    # ── Instruction mix ───────────────────────────────────────────────────────
    "issued_ipc":                "smsp__inst_issued.avg.per_cycle_active",
    "executed_ipc":              "smsp__inst_executed.avg.per_cycle_active",
    "wgmma_inst_executed":       "smsp__inst_executed_pipe_tensor_op_hmma.sum",
    "lsu_inst_executed":         "smsp__inst_executed_pipe_lsu.sum",
}

_METRICS_ARG = ",".join(NCU_METRICS.values())

_SHORT_TO_TORCH = {
    "f16": torch.float16,
    "bf16": torch.bfloat16,
    "f32": torch.float32,
    "tf32": torch.float32,
    "e4m3": getattr(torch, "float8_e4m3fn", None),
    "e5m2": getattr(torch, "float8_e5m2", None),
    "s8": torch.int8,
    "s32": torch.int32,
}

_CUTLASS_TO_SHORT = {
    "cutlass::half_t": "f16",
    "cutlass::bfloat16_t": "bf16",
    "float": "f32",
    "cutlass::tfloat32_t": "tf32",
    "cutlass::float_e4m3_t": "e4m3",
    "cutlass::float_e5m2_t": "e5m2",
    "int8_t": "s8",
    "int32_t": "s32",
}

# ── NCU target mode ────────────────────────────────────────────────────────────


def _run_once_mode(args) -> None:
    """Load .so, run kernel once (ncu intercepts this launch), exit."""
    torch.cuda.set_device(args.device_id)

    lib = ctypes.CDLL(os.path.abspath(str(Path(args.build_dir) / args.so)))
    lib.run_kernel_once.restype = ctypes.c_int
    lib.run_kernel_once.argtypes = [
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
    ]

    dtype_a = _SHORT_TO_TORCH[args.dtype_a]
    dtype_b = _SHORT_TO_TORCH[args.dtype_b]
    dtype_c = _SHORT_TO_TORCH[args.dtype_c]
    M, N, K = args.M, args.N, args.K

    # Single copy — ncu handles replays internally, not us
    A = torch.zeros(M * K, dtype=dtype_a, device="cuda")
    B = torch.zeros(K * N, dtype=dtype_b, device="cuda")
    D = torch.zeros(M * N, dtype=dtype_c, device="cuda")

    ptrs_A = (ctypes.c_void_p * 1)(A.data_ptr())
    ptrs_B = (ctypes.c_void_p * 1)(B.data_ptr())

    status = lib.run_kernel_once(args.kernel_id, ptrs_A, ptrs_B, D.data_ptr(), 1, M, N, K)
    sys.exit(status)


# ── NCU subprocess helpers ─────────────────────────────────────────────────────


def _ensure_launcher(build_dir: Path) -> Path | None:
    """Build ncu_launcher binary in build_dir if not already present.

    Returns the binary path on success, None if compilation fails (caller
    falls back to the Python target).
    """
    binary = build_dir / "ncu_launcher"
    if binary.exists():
        return binary

    logger.info("[NCU] Compiling ncu_launcher binary (one-time)...")
    cmd = [
        "nvcc", "-O2",
        "-o", str(binary),
        str(LAUNCHER_SRC),
        "-ldl", "-lcuda",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if result.returncode == 0:
            logger.info("[NCU] ncu_launcher compiled → %s", binary)
            return binary
        logger.warning("[NCU] ncu_launcher compilation failed — falling back to Python target:\n%s",
                       result.stderr[-1000:])
    except Exception as e:
        logger.warning("[NCU] ncu_launcher compilation error — falling back to Python target: %s", e)
    return None


def _build_ncu_cmd(config: dict, M: int, N: int, K: int, gpu_id: int, build_dir: Path, ncu_bin: str,
                   launcher: Path | None) -> list[str]:
    # fmt: off
    common_flags = [
        ncu_bin,
        "--csv",
        "--metrics",       _METRICS_ARG,
        "--kernel-name",   "device_kernel",
        "--launch-count",  "1",
        "--clock-control", "none",
        "--device",        str(gpu_id),
    ]
    target_args = [
        "--build-dir",   str(build_dir),
        "--so",          config["so_file"],
        "--kernel-id",   str(config["kernel_id"]),
        "--dtype-a",     _CUTLASS_TO_SHORT[config["cutlass_type_a"]],
        "--dtype-b",     _CUTLASS_TO_SHORT[config["cutlass_type_b"]],
        "--dtype-c",     _CUTLASS_TO_SHORT[config["cutlass_type_c"]],
        "--device-id",   str(gpu_id),
        "--M",           str(M),
        "--N",           str(N),
        "--K",           str(K),
    ]
    if launcher:
        target = [str(launcher)]
    else:
        target = [sys.executable, str(SCRIPT_PATH)]
    return common_flags + target + target_args
    # fmt: on


def _parse_ncu_csv(output: str) -> dict[str, float]:
    """Extract metric values from ncu --csv stdout. Returns {} on parse failure."""
    lines = [line for line in output.splitlines() if not line.startswith("==")]
    if not lines:
        return {}

    reverse = {v: k for k, v in NCU_METRICS.items()}
    metrics: dict[str, float] = {}
    try:
        for row in csv.DictReader(io.StringIO("\n".join(lines))):
            ncu_id = row.get("Metric Name", "").strip('"').strip()
            val = row.get("Metric Value", "").strip('"').strip().replace(",", "")
            if ncu_id in reverse and val:
                try:
                    metrics[reverse[ncu_id]] = float(val)
                except ValueError:
                    pass
    except Exception:
        pass

    return metrics


# ── Worker mode ────────────────────────────────────────────────────────────────


def ncu_worker_main(gpu_id: int, ncu_q, build_dir: Path, db_path: Path, ncu_bin: str) -> None:
    """Persistent NCU worker — one per GPU. Pulls items from ncu_q, profiles, records."""
    sys.stdout.reconfigure(line_buffering=True)
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )

    shutdown = threading.Event()
    signal.signal(signal.SIGTERM, lambda s, f: shutdown.set())
    signal.signal(signal.SIGINT, lambda s, f: shutdown.set())

    # Build the C launcher once per worker (only GPU 0 actually compiles;
    # the others find the binary already on disk).
    launcher = _ensure_launcher(build_dir)

    registry = SQLiteRegistry(db_path)

    while True:
        item = ncu_q.get()
        if item is None:
            ncu_q.task_done()
            break

        name, M, N, K = item
        config = registry.lookup(name)
        cmd = _build_ncu_cmd(config, M, N, K, gpu_id, build_dir, ncu_bin, launcher)

        cancelled = False
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=NCU_TIMEOUT_S, cwd=tmpdir)
            if result.returncode == 0:
                metrics = _parse_ncu_csv(result.stdout)
                if metrics:
                    registry.record_ncu(name, M, N, K, metrics, None)
                    logger.info("[NCU GPU %d] [%dx%dx%d] %s  (%d metrics)", gpu_id, M, N, K, name, len(metrics))
                else:
                    err = "ncu produced no parseable metrics"
                    registry.record_ncu(name, M, N, K, None, err)
                    logger.warning("[NCU GPU %d] [%dx%dx%d] %s  → no metrics", gpu_id, M, N, K, name)
                if shutdown.is_set():
                    cancelled = True
            else:
                if shutdown.is_set():
                    cancelled = True
                else:
                    err = (result.stderr or result.stdout)[-2000:]
                    registry.record_ncu(name, M, N, K, None, err)
                    logger.warning(
                        "[NCU GPU %d] [%dx%dx%d] %s  → ncu failed (rc=%d)", gpu_id, M, N, K, name, result.returncode
                    )
        except subprocess.TimeoutExpired:
            registry.record_ncu(name, M, N, K, None, f"timeout after {NCU_TIMEOUT_S}s")
            logger.warning("[NCU GPU %d] [%dx%dx%d] %s  → timeout", gpu_id, M, N, K, name)
        except Exception as e:
            if shutdown.is_set():
                cancelled = True
            else:
                registry.record_ncu(name, M, N, K, None, str(e))
                logger.error("[NCU GPU %d] [%dx%dx%d] %s  → %s", gpu_id, M, N, K, name, e)
        finally:
            ncu_q.task_done()

        if cancelled:
            break


# ── Entry point (NCU target mode only) ────────────────────────────────────────

if __name__ == "__main__":
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--so",        type=str,  required=True)
    parser.add_argument("--kernel-id", type=int,  required=True)
    parser.add_argument("--dtype-a",   type=str,  required=True)
    parser.add_argument("--dtype-b",   type=str,  required=True)
    parser.add_argument("--dtype-c",   type=str,  required=True)
    parser.add_argument("--device-id", type=int,  default=0)
    parser.add_argument("--M",         type=int,  required=True)
    parser.add_argument("--N",         type=int,  required=True)
    parser.add_argument("--K",         type=int,  required=True)
    # fmt: on
    _run_once_mode(parser.parse_args())
