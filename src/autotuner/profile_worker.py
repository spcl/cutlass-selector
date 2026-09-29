"""Benchmark worker: times compiled CUTLASS kernels on one GPU.

Each worker process loads the ``.so`` libraries built by the scheduler and calls their
``benchmark_kernel`` entry point for every planned shape. Operands are rotated through
enough copies to exceed L2 (``ceil(3 * L2 / (A + B))``, at most MAX_NUM_COPIES) so timed
iterations read cold memory. A measurement is WARMUP_ITERS warm-up launches followed by
NUM_ROUNDS rounds of ITERS_PER_ROUND launches; mean and standard deviation over rounds
are stored in the registry. The tile scheduler runs with raster=Heuristic and
max_swizzle_size=8, the settings all training data was collected under.
"""

import ctypes
import logging
import math
import os
import statistics
import sys
import time
from pathlib import Path

import torch

# Allow this module to be imported in a spawned child process where autotuner/
# may not yet be on sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from registry import SQLiteRegistry

logger = logging.getLogger(__name__)

# Map CUTLASS C++ types to PyTorch dtypes
CUTLASS_TO_TORCH = {
    "cutlass::half_t": torch.float16,
    "cutlass::bfloat16_t": torch.bfloat16,
    "float": torch.float32,
    "cutlass::tfloat32_t": torch.float32,
    "cutlass::float_e4m3_t": getattr(torch, "float8_e4m3fn", None),
    "cutlass::float_e5m2_t": getattr(torch, "float8_e5m2", None),
    "int8_t": torch.int8,
    "int32_t": torch.int32,
    "double": torch.float64,
}


# CUTLASS non-zero status codes returned by benchmark_kernel
CUTLASS_STATUS = {
    1: "Misaligned Operand (pointers don't meet 128-byte TMA alignment)",
    2: "Invalid Data Type",
    3: "Invalid Layout (stride doesn't match requirement)",
    4: "Invalid Problem Size",
    5: "Not Supported (tile size or stages invalid for this type)",
    6: "Workspace Null",
    7: "Error Internal (usually SMEM limit exceeded on Hopper)",
    8: "Arch Mismatch (needs SM90 but running on older GPU)",
    9: "Insufficient Driver",
    10: "Memory Allocation Failed",
    11: "Invalid / Unspecified (problem likely too small for tile dimensions)",
}

WARMUP_ITERS = 5
NUM_ROUNDS = 5
ITERS_PER_ROUND = 3

MAX_NUM_COPIES = 64  # cap L2-rotation copies when problem is tiny
TMA_ALIGN_BYTES = 128
DEFAULT_GPU_BUF_GB = float(os.environ.get("CKS_GPU_BUF_GB", "16"))


def calculate_tflops(M: int, N: int, K: int, avg_ms: float) -> float:
    """Throughput of an M x N x K GEMM (2*M*N*K flops) that took avg_ms milliseconds, in TFLOP/s."""
    return (2.0 * M * N * K) / (avg_ms / 1000.0) / 1e12


def load_lib(path: str | Path) -> ctypes.CDLL:
    """Load a compiled kernel .so and wire up the benchmark_kernel C signature."""
    lib = ctypes.CDLL(os.path.abspath(path))
    # int benchmark_kernel(int kernel_id,
    #     void** ptrs_A, void** ptrs_B, void* ptr_D,
    #     int num_copies, int M, int N, int K,
    #     int warmup, int num_rounds, int iters_per_round,
    #     int raster_order, int swizzle_size, int splits, float* out_ms)
    lib.benchmark_kernel.restype = ctypes.c_int
    lib.benchmark_kernel.argtypes = [
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_float),
    ]
    return lib


def _l2_bytes(device_id: int) -> int:
    props = torch.cuda.get_device_properties(device_id)
    l2 = getattr(props, "l2_cache_size", None) or getattr(props, "L2_cache_size", None)
    return int(l2) if l2 else 50 * 1024 * 1024


def _num_copies(M, N, K, dtype_a, dtype_b, device_id: int) -> int:
    buf = M * K * dtype_a.itemsize + K * N * dtype_b.itemsize
    if buf <= 0:
        return 1
    return max(1, min(math.ceil(3 * _l2_bytes(device_id) / buf), MAX_NUM_COPIES))


def _align128(offset: int) -> int:
    return ((offset + TMA_ALIGN_BYTES - 1) // TMA_ALIGN_BYTES) * TMA_ALIGN_BYTES


class _BufferPool:
    """One pre-allocated GPU byte pool; benchmark views are slices (no per-shape alloc)."""

    def __init__(self, gpu_id: int, size_gb: float = DEFAULT_GPU_BUF_GB) -> None:
        self._gpu_id = gpu_id
        self._size_bytes = int(size_gb * 1024**3)
        dev = f"cuda:{gpu_id}"
        self._pool = torch.zeros(self._size_bytes, dtype=torch.uint8, device=dev)
        logger.info("[GPU %d] Buffer pool: %.1f GiB", gpu_id, size_gb)

    def get(
        self, M: int, N: int, K: int, dtype_a, dtype_b, dtype_c
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], torch.Tensor]:
        """Carve A, B and D views for an M x N x K problem out of the preallocated pool.

        Returns num_copies A and B tensors (enough to rotate past L2) and one D tensor, each
        starting on a 128-byte boundary as TMA requires. Raises if the pool is too small.
        """
        num_copies = _num_copies(M, N, K, dtype_a, dtype_b, self._gpu_id)
        a_bytes = M * K * dtype_a.itemsize
        b_bytes = K * N * dtype_b.itemsize
        d_bytes = M * N * dtype_c.itemsize

        offset = 0
        As: list[torch.Tensor] = []
        for _ in range(num_copies):
            offset = _align128(offset)
            end = offset + a_bytes
            if end > self._size_bytes:
                raise RuntimeError(
                    f"GPU {self._gpu_id} buffer pool exhausted at A slice "
                    f"({end} > {self._size_bytes}); increase CKS_GPU_BUF_GB"
                )
            As.append(self._pool[offset:end].view(dtype_a).reshape(M * K))
            offset = end

        Bs: list[torch.Tensor] = []
        for _ in range(num_copies):
            offset = _align128(offset)
            end = offset + b_bytes
            if end > self._size_bytes:
                raise RuntimeError(
                    f"GPU {self._gpu_id} buffer pool exhausted at B slice; increase CKS_GPU_BUF_GB"
                )
            Bs.append(self._pool[offset:end].view(dtype_b).reshape(K * N))
            offset = end

        offset = _align128(offset)
        d_end = offset + d_bytes
        if d_end > self._size_bytes:
            raise RuntimeError(
                f"GPU {self._gpu_id} buffer pool exhausted at D slice; increase CKS_GPU_BUF_GB"
            )
        D = self._pool[offset:d_end].view(dtype_c).reshape(M * N)
        return As, Bs, D


def benchmark(lib, kernel_id, As, Bs, D, M, N, K,
              raster_order=-1, swizzle_size=1, splits=1) -> tuple[int, list[float] | None]:
    """Run a single kernel and return (status_code, round_latencies).

    On success (status == 0), round_latencies is a list of NUM_ROUNDS per-iteration
    average latencies (total_round_ms / ITERS_PER_ROUND). Otherwise None.

    Tile-scheduler runtime args. The defaults (-1, 1, 1) are resolved by the kernel
    template to raster=Heuristic, max_swizzle_size=8, the settings every sweep
    measurement was taken under; changing that makes new data incomparable with
    the existing DBs.
        raster_order — 0 = AlongM, 1 = AlongN, otherwise Heuristic (-1).
        swizzle_size — threadblock swizzle (1, 2, 4, 8).
        splits       — Stream-K split count (ignored by non-Stream-K schedulers).
    """
    ptrs_A = (ctypes.c_void_p * len(As))(*[a.data_ptr() for a in As])
    ptrs_B = (ctypes.c_void_p * len(Bs))(*[b.data_ptr() for b in Bs])
    out_ms = (ctypes.c_float * NUM_ROUNDS)()

    status = lib.benchmark_kernel(
        kernel_id,
        ptrs_A,
        ptrs_B,
        D.data_ptr(),
        len(As),
        M,
        N,
        K,
        WARMUP_ITERS,
        NUM_ROUNDS,
        ITERS_PER_ROUND,
        raster_order,
        swizzle_size,
        splits,
        out_ms,
    )

    latencies = [out_ms[r] for r in range(NUM_ROUNDS)] if status == 0 else None
    return status, latencies


def profile_worker_main(
    gpu_id: int,
    eval_q,
    build_dir: Path,
    shapes: list[tuple],
    db_path: Path,
    tag: str | None = None,
    shard_index: int | None = None,
    num_shards: int | None = None,
) -> None:
    """Persistent benchmark worker — lives for the entire autotuner run.

    Pulls config names from eval_q, benchmarks all shapes, and writes results to
    the registry. CUDA context, loaded .so libraries, and allocated buffers are
    all reused across kernels.
    """
    sys.stdout.reconfigure(line_buffering=True)
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )

    torch.cuda.set_device(gpu_id)
    registry = SQLiteRegistry(db_path)
    lib_cache: dict[str, ctypes.CDLL] = {}  # so_filename → loaded library
    buf_pool = _BufferPool(gpu_id)

    while True:
        config_name = eval_q.get()
        if config_name is None:
            eval_q.task_done()
            break

        # Write to DB before any benchmark work. The monitor reads this on crash
        # to identify which kernel died and which shape to mark as crashed.
        registry.set_worker_state(gpu_id, config_name)

        config = registry.lookup(config_name)
        if config["compile_status"] != "success" or not config.get("so_file"):
            registry.clear_worker_state(gpu_id)
            eval_q.put(config_name)
            eval_q.task_done()
            time.sleep(1.0)
            continue

        so_file = config["so_file"]
        so_path = build_dir / so_file
        if not so_path.is_file():
            logger.error(
                "[GPU %d] Missing .so %s for %s — requeueing for compile",
                gpu_id,
                so_path,
                config_name,
            )
            registry.requeue_config_missing_so(config_name)
            registry.clear_worker_state(gpu_id)
            eval_q.task_done()
            continue

        if so_file not in lib_cache:
            lib_cache[so_file] = load_lib(so_path)
        lib = lib_cache[so_file]
        dtype_a = CUTLASS_TO_TORCH[config["cutlass_type_a"]]
        dtype_b = CUTLASS_TO_TORCH[config["cutlass_type_b"]]
        dtype_c = CUTLASS_TO_TORCH[config["cutlass_type_c"]]
        name = config["name"]
        kernel_id = config["kernel_id"]

        shapes_to_run = (
            registry.plan_shapes_for_config(name, tag, shard_index, num_shards)
            if tag
            else shapes
        )
        for M, N, K in shapes_to_run:
            if registry.run_status(name, M, N, K) in ("success", "rejected", "crashed", "hung"):
                continue

            # Heartbeat per shape — monitor timeout is per worker_state update;
            # without this, multi-shape configs trip WORKER_TIMEOUT_S mid-config.
            registry.set_worker_state(gpu_id, config_name)

            As, Bs, D = buf_pool.get(M, N, K, dtype_a, dtype_b, dtype_c)

            status, latencies = benchmark(lib, kernel_id, As, Bs, D, M, N, K)

            if latencies is not None:
                mean_ms = statistics.mean(latencies)
                std_ms = statistics.stdev(latencies)
                tflops = [calculate_tflops(M, N, K, ms) for ms in latencies]
                mean_tf = statistics.mean(tflops)
                std_tf = statistics.stdev(tflops)
                logger.info("[GPU %d] [%dx%dx%d] %s -> %.2f TFLOPS", gpu_id, M, N, K, name, mean_tf)
                registry.record_success(
                    name,
                    M,
                    N,
                    K,
                    {
                        "mean_ms": round(mean_ms, 4),
                        "std_ms": round(std_ms, 4),
                        "mean_tflops": round(mean_tf, 2),
                        "std_tflops": round(std_tf, 2),
                    },
                    tag=tag,
                )
            else:
                reason = CUTLASS_STATUS.get(status, f"Unknown Status Code {status}")
                logger.warning("[GPU %d] [%dx%dx%d] %s -> REJECTED (%s)", gpu_id, M, N, K, name, reason)
                registry.record_rejected(name, M, N, K, status, reason, tag=tag)

        registry.clear_worker_state(gpu_id)
        eval_q.task_done()
