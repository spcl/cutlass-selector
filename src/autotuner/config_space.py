"""CUTLASS SM90 config enumeration + validity rules (single source of truth)."""

from __future__ import annotations

import itertools
import math

# ── Type / layout / suffix tables (verbatim from src/autotuner/scheduler.py) ──────

TYPE_BYTES = {
    "cutlass::half_t": 2,
    "cutlass::bfloat16_t": 2,
    "cutlass::tfloat32_t": 4,
    "cutlass::float_e4m3_t": 1,
    "cutlass::float_e5m2_t": 1,
    "float": 4,
    "int8_t": 1,
    "int32_t": 4,
    "double": 8,
}

# Follows CUTLASS profiler DataTypeNames
TYPE_SHORT_NAME = {
    "cutlass::half_t": "f16",
    "cutlass::bfloat16_t": "bf16",
    "cutlass::tfloat32_t": "tf32",
    "cutlass::float_e4m3_t": "e4m3",
    "cutlass::float_e5m2_t": "e5m2",
    "float": "f32",
    "int8_t": "s8",
    "int32_t": "s32",
}

# Follows CUTLASS profiler ShortLayoutTypeNames
LAYOUT_SHORT = {
    "cutlass::layout::RowMajor": "t",
    "cutlass::layout::ColumnMajor": "n",
}

# Follows CUTLASS profiler KernelScheduleSuffixes
KERNEL_SCHEDULE_SUFFIX = {
    "cutlass::gemm::KernelTmaWarpSpecialized": "_warpspecialized",
    "cutlass::gemm::KernelTmaWarpSpecializedPingpong": "_warpspecialized_pingpong",
    "cutlass::gemm::KernelTmaWarpSpecializedCooperative": "_warpspecialized_cooperative",
}

# Follows CUTLASS profiler TileSchedulerSuffixes
TILE_SCHEDULER_SUFFIX = {
    "cutlass::gemm::PersistentScheduler": "",
    "cutlass::gemm::StreamKScheduler": "_stream_k",
}

# Follows CUTLASS profiler EpilogueScheduleSuffixes (python/cutlass_library/library.py)
EPILOGUE_SCHEDULE_SUFFIX = {
    "cutlass::epilogue::TmaWarpSpecialized": "_epi_tma",
    "cutlass::epilogue::TmaWarpSpecializedCooperative": "_epi_tma",
    "cutlass::epilogue::NoSmemWarpSpecialized": "_epi_nosmem",
}

HOPPER_SMEM_LIMIT_BYTES = 232448  # 227 KiB per-CTA cap on GH200 / SM90 Hopper
SMEM_BARRIER_RESERVE_BYTES = 2048  # add to tile estimate (barriers, static TMA smem)
MINIMUM_TMA_OVERHEAD = 1024
OP_CLASS = "cutlass::arch::OpClassTensorOp"

# ── Precision tuples, keyed by dtype name ─────────────────────────────────────
# (name_prefix, type_a, type_b, type_c_and_d, type_acc).
PRECISIONS = {
    "bf16": (
        "BF16_F32_BF16",
        "cutlass::bfloat16_t",
        "cutlass::bfloat16_t",
        "cutlass::bfloat16_t",
        "float",
    ),
    "fp32": (
        "F32_F32_F32",
        "float",
        "float",
        "float",
        "float",
    ),
    "fp8_e4m3": (
        "E4M3_E4M3_E4M3",
        "cutlass::float_e4m3_t",
        "cutlass::float_e4m3_t",
        "cutlass::float_e4m3_t",
        "float",
    ),
}

# Memory layouts for A and B (name, layout_a, layout_b). C/D is always ColMajor —
# row-major output is the transpose+swap trick (C^T = B^T * A^T).
LAYOUTS = {
    "TN": ("TN", "cutlass::layout::RowMajor", "cutlass::layout::ColumnMajor"),
    "TT": ("TT", "cutlass::layout::RowMajor", "cutlass::layout::RowMajor"),
    "NN": ("NN", "cutlass::layout::ColumnMajor", "cutlass::layout::ColumnMajor"),
    "NT": ("NT", "cutlass::layout::ColumnMajor", "cutlass::layout::RowMajor"),
}

# Tile / cluster / schedule / stage axes
TILE_SHAPES_MNK = [
    (m, n, k)
    for m in [64, 128, 192, 256]
    for n in [16, 32, 48, 64, 80, 96, 128, 192, 256]
    for k in [32, 64, 128, 256, 512]
]
CLUSTER_SHAPES_MNK = [(m, n, 1) for m in [1, 2, 4, 8, 16] for n in [1, 2, 4, 8, 16] if m * n <= 16]
KERNEL_SCHEDULES = [
    "cutlass::gemm::KernelTmaWarpSpecialized",
    "cutlass::gemm::KernelTmaWarpSpecializedPingpong",
    "cutlass::gemm::KernelTmaWarpSpecializedCooperative",
]
STAGE_COUNTS = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
EPILOGUE_SCHEDULES = [
    "cutlass::epilogue::NoSmemWarpSpecialized",
    "cutlass::epilogue::TmaWarpSpecialized",
    "cutlass::epilogue::TmaWarpSpecializedCooperative",
]
TILE_SCHEDULERS = [
    "cutlass::gemm::PersistentScheduler",
    "cutlass::gemm::StreamKScheduler",
]


# ── SMEM helpers (shared with src/model/features.py) ──────────────────────────────

def _epilogue_tile_mn(
    tile_m: float,
    tile_n: float,
    bytes_c: float,
    epilogue_schedule: str,
) -> tuple[int, int]:
    """EpilogueTileAuto path from sm90_compute_tile_shape_or_override."""
    tm, tn = int(tile_m), int(tile_n)
    if "Cooperative" in epilogue_schedule:
        epi_m = min(128, tm)
        epi_n = math.gcd(min(32, tn), tn)
    else:
        n_perf = 64 if bytes_c == 1 and tn % 64 == 0 else 32
        epi_m = min(64, tm)
        epi_n = math.gcd(min(n_perf, tn), tn)
    return epi_m, epi_n


def _epi_tiles(tile_m: float, tile_n: float, epi_m: int, epi_n: int) -> int:
    return math.ceil(tile_m / epi_m) * math.ceil(tile_n / epi_n)


def _subtile_smem_bytes(epi_m: int, epi_n: int, bytes_c: float, stages: int) -> float:
    """Swizzled epilogue buffer; 128 B alignment per stage (conservative vs ss_smem_selector)."""
    per_stage = epi_m * epi_n * bytes_c
    per_stage_aligned = ((int(per_stage) + 127) // 128) * 128
    return float(per_stage_aligned * stages)


def estimate_epilogue_smem_bytes(
    tile_m: float,
    tile_n: float,
    bytes_c: float,
    epilogue_schedule: str,
    *,
    has_source_c: bool = True,
) -> float:
    """TMA epilogue smem from EpilogueTile + StagesC/D + ReuseSmem (sm90_builder.inl)."""
    if "NoSmem" in epilogue_schedule:
        return 0.0
    epi_m, epi_n = _epilogue_tile_mn(tile_m, tile_n, bytes_c, epilogue_schedule)
    epi_tiles = _epi_tiles(tile_m, tile_n, epi_m, epi_n)
    stages_d = min(epi_tiles, 2)
    # ReuseSmem: matching C/D element width and > 8 bits (bf16/fp16/fp32).
    reuse_smem = has_source_c and int(bytes_c * 8) > 8
    if reuse_smem:
        stages_c = max(min(epi_tiles, 4), stages_d + 1)
        return _subtile_smem_bytes(epi_m, epi_n, bytes_c, stages_c)
    stages_c = min(epi_tiles, 4) if has_source_c else 0
    smem_c = _subtile_smem_bytes(epi_m, epi_n, bytes_c, stages_c) if has_source_c else 0.0
    smem_d = _subtile_smem_bytes(epi_m, epi_n, bytes_c, stages_d)
    return smem_c + smem_d


def estimate_smem_total(
    tile_m: float,
    tile_n: float,
    tile_k: float,
    stages: float,
    bytes_a: float,
    bytes_b: float,
    bytes_c: float,
    kernel_schedule: str,
    epilogue_schedule: str,
) -> float:
    """Analytic smem footprint for featurization (EpilogueTile subtiles, not full CTA tile)."""
    smem_mainloop = stages * (tile_m * tile_k * bytes_a + tile_n * tile_k * bytes_b)
    smem_epi = estimate_epilogue_smem_bytes(tile_m, tile_n, bytes_c, epilogue_schedule)
    if smem_epi == 0.0:
        return smem_mainloop
    is_coop = "Cooperative" in kernel_schedule
    is_pingpong = "Pingpong" in kernel_schedule
    if is_coop or is_pingpong:
        return smem_mainloop + smem_epi
    return max(smem_mainloop, smem_epi)


# ── Validity rules  ──────
def is_valid_config(config: dict) -> bool:
    """Return True iff this config combination is valid for SM90 Hopper."""
    prec = config["precisions"]
    tile_m, tile_n, tile_k = config["tile_shape_mnk"]
    cluster_m, cluster_n, cluster_k = config["cluster_shape_mnk"]
    kernel_sched = config["kernel_schedule"]
    scheduler = config["tile_scheduler"]
    epilogue_sched = config["epilogue_schedule"]
    stages = config["stage_count"]
    _, layout_a, layout_b = config["layouts"]

    # CUTLASS SM90 warp-specialized mainloops require Stages >= 2 (StageCount<0/1 fails nvcc).
    if stages < 2:
        return False

    bytes_a = TYPE_BYTES[prec[1]]
    bytes_b = TYPE_BYTES[prec[2]]
    is_8bit = bytes_a == 1

    # Rule 1: Stream-K requires Cooperative mainloop (CUTLASS SM90 StreamKScheduler).
    if "StreamK" in scheduler and kernel_sched != "cutlass::gemm::KernelTmaWarpSpecializedCooperative":
        return False

    # Rule 2: Cooperative mainloop requires tile_m divisible by 128.
    # Cooperative splits the tile across 2 warpgroups: WG0 owns rows 0..tile_m/2-1,
    # WG1 owns rows tile_m/2..tile_m-1. Each half must be 64-row aligned (one WGMMA
    # warpgroup register tile), so tile_m/2 must be a multiple of 64, i.e. tile_m
    # must be a multiple of 128. tile_m=192 fails (192/2=96, not 64-aligned).
    if "Cooperative" in kernel_sched and tile_m % 128 != 0:
        return False

    # Rule 3: Mainloop / epilogue schedule pairing (CUTLASS CollectiveBuilder).
    #   • TmaWarpSpecializedCooperative epilogue requires cooperative mainloop.
    #   • Cooperative mainloop pairs with TmaWarpSpecializedCooperative or NoSmem
    #     (not plain TmaWarpSpecialized — see cutlass_reference cooperative_epi_nosmem).
    #   • NoSmem is valid with WS, Pingpong, and Cooperative mainloops.
    is_coop_kernel = "Cooperative" in kernel_sched
    is_coop_epi = "Cooperative" in epilogue_sched
    is_nosmem = "NoSmem" in epilogue_sched
    is_plain_tma = not is_nosmem and not is_coop_epi
    if is_coop_epi and not is_coop_kernel:
        return False
    if is_coop_kernel and is_plain_tma:
        return False

    # Rule 4: 8-bit types have strict WGMMA and TMA 128-byte swizzle constraints.
    # FP8 K-atom is 32 elements = 32 bytes. TMA's 128-byte swizzle requires
    # tile_k * 1B == 128 bytes, so tile_k must be >= 128.
    # WGMMA on SM90 requires A's K-dimension to be contiguous (RowMajor).
    if is_8bit:
        if tile_k < 128:
            return False
        if layout_a == "cutlass::layout::ColumnMajor":
            return False

    # Rule 5 (FP8): SmemLayoutAtom alignment — when tile_n is a multiple of 128,
    # the CollectiveBuilder selects a stricter combined smem layout that also requires
    # tile_m to be a multiple of 128.
    if is_8bit and tile_n % 128 == 0 and tile_m % 128 != 0:
        return False

    # Rule 7: 227 KiB Hopper cap — reject if tile smem + 2 KiB barrier reserve exceeds limit.
    bytes_c = TYPE_BYTES[prec[3]]
    smem_total = estimate_smem_total(
        tile_m, tile_n, tile_k, stages,
        bytes_a, bytes_b, bytes_c,
        kernel_sched, epilogue_sched,
    )
    if smem_total + SMEM_BARRIER_RESERVE_BYTES > HOPPER_SMEM_LIMIT_BYTES:
        return False

    # Rule 8: FP8 + B=RowMajor (TMA-transpose path) with non-power-of-2 tile_n.
    # The CUTE swizzle composition for multi-CTA clusters requires cluster_m to be
    # strictly less than (tile_n & -tile_n) / 4, where (tile_n & -tile_n) is the
    # highest power of 2 dividing tile_n. When this is violated, nvcc fails deep
    # inside cute::detail::composition_impl. Power-of-2 tile_n values are immune
    # because their layout decomposes cleanly at every granularity.
    # Derived empirically: catches all 1,196 observed failures, zero false positives.
    if is_8bit and layout_b == "cutlass::layout::RowMajor":
        if (tile_n & (tile_n - 1)) != 0:  # tile_n is not a power of 2
            if cluster_m * 4 >= (tile_n & -tile_n):
                return False

    # Rule 9: TMA box K-extent <= 256 (cute asserts smem_box_shape[1] <= 1<<8).
    # K is the gmem-outer box dim for A=ColumnMajor / B=RowMajor, multicast splits it
    # (cluster_n for A, cluster_m for B). Over-budget compiles but SIGABRTs at launch.
    if layout_a == "cutlass::layout::ColumnMajor" and tile_k > 256 * cluster_n:
        return False
    if layout_b == "cutlass::layout::RowMajor" and tile_k > 256 * cluster_m:
        return False

    return True


def build_config_entry(config: dict) -> dict:
    """Flatten a raw search-space config into the kernel-template dict format.
    """
    prec = config["precisions"]
    tile_m, tile_n, tile_k = config["tile_shape_mnk"]
    cluster_m, cluster_n, cluster_k = config["cluster_shape_mnk"]
    kernel_sched = config["kernel_schedule"]
    scheduler = config["tile_scheduler"]
    stages = config["stage_count"]
    epilogue_sched = config["epilogue_schedule"]
    _, layout_a, layout_b = config["layouts"]

    bytes_a = TYPE_BYTES[prec[1]]
    bytes_b = TYPE_BYTES[prec[2]]
    bytes_c = TYPE_BYTES[prec[3]]
    align_a = 128 // bytes_a
    align_b = 128 // bytes_b
    align_c = 128 // bytes_c

    layout_str = f"{LAYOUT_SHORT[layout_a]}{LAYOUT_SHORT[layout_b]}n"
    name = (
        f"cutlass3x_sm90_tensorop"
        f"_{TYPE_SHORT_NAME[prec[1]]}_{TYPE_SHORT_NAME[prec[2]]}"
        f"_{TYPE_SHORT_NAME[prec[4]]}_{TYPE_SHORT_NAME[prec[3]]}_{TYPE_SHORT_NAME[prec[3]]}"
        f"_{tile_m}x{tile_n}x{tile_k}_{cluster_m}x{cluster_n}x{cluster_k}"
        f"_{stages}_{layout_str}_align{max(align_a, align_b)}"
        f"{TILE_SCHEDULER_SUFFIX[scheduler]}{KERNEL_SCHEDULE_SUFFIX[kernel_sched]}"
        f"{EPILOGUE_SCHEDULE_SUFFIX[epilogue_sched]}"
    )

    return {
        "name": name,
        "cutlass_type_a": prec[1],
        "cutlass_type_b": prec[2],
        "cutlass_type_c": prec[3],
        "cutlass_type_acc": prec[4],
        "layout_a": layout_a,
        "layout_b": layout_b,
        "alignment_a": align_a,
        "alignment_b": align_b,
        "alignment_c": align_c,
        "op_class": OP_CLASS,
        "tile_m": tile_m,
        "tile_n": tile_n,
        "tile_k": tile_k,
        "cluster_m": cluster_m,
        "cluster_n": cluster_n,
        "cluster_k": cluster_k,
        "kernel_schedule": kernel_sched,
        "stages": stages,
        "epilogue_schedule": epilogue_sched,
        "scheduler": scheduler,
    }


DEFAULT_DTYPE = "bf16"


def generate_search_space(dtypes=None, layout_names=None) -> list[dict]:
    """Build the full list of valid config-entry dicts.

    With no filters this returns all valid BF16 configs (~208k with NoSmem epilogue).
    Pass ``dtypes=["fp32"]`` / ``["fp8_e4m3"]`` for other precision sweeps.
    ``layout_names`` restricts layout pairs (e.g. ``["TN"]`` for FP32/FP8).
    """
    precisions = [PRECISIONS[d] for d in (dtypes if dtypes is not None else [DEFAULT_DTYPE])]
    layouts = [LAYOUTS[name] for name in (layout_names or LAYOUTS)]
    axes = {
        "precisions": precisions,
        "tile_shape_mnk": TILE_SHAPES_MNK,
        "cluster_shape_mnk": CLUSTER_SHAPES_MNK,
        "kernel_schedule": KERNEL_SCHEDULES,
        "stage_count": STAGE_COUNTS,
        "epilogue_schedule": EPILOGUE_SCHEDULES,
        "tile_scheduler": TILE_SCHEDULERS,
        "layouts": layouts,
    }
    return [
        build_config_entry(config)
        for config in (dict(zip(axes.keys(), values)) for values in itertools.product(*axes.values()))
        if is_valid_config(config)
    ]


def enumerate_candidates(M: int, N: int, K: int, layout: str, dtype: str = "bf16") -> list[dict]:
    """Inference-time boundary: valid candidate configs for one problem.

    ``layout`` is the two-char A/B combo ("TN", "TT", "NN", "NT").  ``dtype``
    defaults to "bf16" — the only datatype the shipped model covers.  (M, N, K)
    are accepted for symmetry with the selector / future shape-dependent pruning;
    the SM90 validity rules are shape-independent, so they do not filter here.
    """
    if dtype not in PRECISIONS:
        raise ValueError(f"unknown dtype {dtype!r}; known: {sorted(PRECISIONS)}")
    if layout not in LAYOUTS:
        raise ValueError(f"unknown layout {layout!r}; known: {sorted(LAYOUTS)}")
    return generate_search_space(dtypes=[dtype], layout_names=[layout])
