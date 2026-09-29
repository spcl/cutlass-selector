#!/usr/bin/env python3
"""
translate.py — Turn an nvMMH recommendation into runnable autotuner kernel configs.

nvMMH emits a tile + cluster (+ stages, split_k, raster, swizzle) but NO kernel /
epilogue schedule. The real CUTLASS SM90 workflow handles this by *enumerating* all
valid schedule variants for the tile and letting the profiler pick the best measured
one (see media/docs/cpp/heuristics.md and sm90_utils.get_valid_schedules). We mirror
that: each recommendation fans out into the valid (kernel_schedule, epilogue_schedule,
tile_scheduler) variants in our search space, each with:

  * stage_count from nvMMH's `stages` field (our template needs an explicit StageCount,
    not 0 — nvMMH often returns 0 on SM90; we default to 4 in that case)
  * raster_order / swizzle_size / splits carried as *runtime* benchmark args
    (raster from nvMMH cta_order: 0 = AlongM, 1 = AlongN; splits used by Stream-K only)

Schedule variants that violate config_space.is_valid_config are skipped (e.g.
cooperative mainloop with tile_m < 128). nvMMH's tile/cluster are still honoured.
No GPU required.
"""

import sys
from pathlib import Path

# src/autotuner/config_space.py is the single source of truth for the kernel config format.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "autotuner"))
from config_space import STAGE_COUNTS, build_config_entry, is_valid_config  # noqa: E402

# bf16 precision tuple in the autotuner's raw search-space format:
# (name_prefix, type_a, type_b, type_c_and_d, type_acc)
BF16_PRECISION = (
    "BF16_F32_BF16",
    "cutlass::bfloat16_t",
    "cutlass::bfloat16_t",
    "cutlass::bfloat16_t",
    "float",
)

WS = "cutlass::gemm::KernelTmaWarpSpecialized"
PINGPONG = "cutlass::gemm::KernelTmaWarpSpecializedPingpong"
COOPERATIVE = "cutlass::gemm::KernelTmaWarpSpecializedCooperative"
EPI_NOSMEM = "cutlass::epilogue::NoSmemWarpSpecialized"
EPI_TMA = "cutlass::epilogue::TmaWarpSpecialized"
EPI_TMA_COOPERATIVE = "cutlass::epilogue::TmaWarpSpecializedCooperative"
PERSISTENT = "cutlass::gemm::PersistentScheduler"
STREAM_K = "cutlass::gemm::StreamKScheduler"

SCHEDULE_VARIANTS = [
    (WS,          EPI_NOSMEM,          PERSISTENT),
    (WS,          EPI_TMA,             PERSISTENT),
    (PINGPONG,    EPI_NOSMEM,          PERSISTENT),
    (PINGPONG,    EPI_TMA,             PERSISTENT),
    (COOPERATIVE, EPI_NOSMEM,          PERSISTENT),
    (COOPERATIVE, EPI_TMA_COOPERATIVE, PERSISTENT),
    (COOPERATIVE, EPI_NOSMEM,          STREAM_K),
    (COOPERATIVE, EPI_TMA_COOPERATIVE, STREAM_K),
]


def _stage_count(rec: dict) -> int:
    """Map nvMMH stages to an explicit hopper_template StageCount value (>= 2)."""
    s = int(rec.get("stages") or 0)
    if s in STAGE_COUNTS:
        return s
    # nvMMH returns 0 on SM90; CUTLASS requires StageCount >= 2.
    return 4


def materialize(rec: dict, layout_a: str, layout_b: str,
                precisions: tuple = BF16_PRECISION) -> list[dict]:
    """Expand one nvMMH recommendation into its valid runnable variants.

    Returns a list of dicts, each:
        {
          "config":       <build_config_entry dict, ready for the template/registry>,
          "raster_order": int,   # 0 AlongM, 1 AlongN (passed to benchmark)
          "swizzle_size": int,
          "splits":       int,   # Stream-K split count (ignored by Persistent)
        }
    Valid schedule variants for nvMMH's tile/cluster are returned (deduplicated by name).
    """
    layouts = ("cutlass::layout::ColumnMajor", layout_a, layout_b)  # [0] (output) ignored by builder
    tile = (rec["cta_tile_m"], rec["cta_tile_n"], rec["cta_tile_k"])
    cluster = (rec["cluster_m"], rec["cluster_n"], 1)
    stages = _stage_count(rec)

    variants: list[dict] = []
    seen: set[str] = set()
    for kernel_sched, epilogue_sched, scheduler in SCHEDULE_VARIANTS:
        raw = dict(
            precisions=precisions,
            tile_shape_mnk=tile,
            cluster_shape_mnk=cluster,
            kernel_schedule=kernel_sched,
            tile_scheduler=scheduler,
            stage_count=stages,
            epilogue_schedule=epilogue_sched,
            layouts=layouts,
        )
        if not is_valid_config(raw):
            continue
        cfg = build_config_entry(raw)
        if cfg["name"] in seen:
            continue
        seen.add(cfg["name"])
        variants.append({
            "config":       cfg,
            "raster_order": rec["cta_order"],
            "swizzle_size": rec["swizzle_factor"],
            "splits":       rec["split_k"],
        })
    return variants


if __name__ == "__main__":
    # Show the materialized variants for nvMMH's top picks on a couple of shapes.
    from query import NvmmhInterface, layout_enum, precision_string

    iface = NvmmhInterface(gpu="H100_SXM")
    prec = precision_string("cutlass::bfloat16_t", "cutlass::bfloat16_t", "float", "cutlass::bfloat16_t")
    la, lb = "cutlass::layout::RowMajor", "cutlass::layout::ColumnMajor"  # TN
    lay = layout_enum(la, lb, iface.nvmmh)

    for (M, N, K) in [(4096, 4096, 4096), (256, 256, 8192)]:
        print(f"\n=== {M}x{N}x{K} (TN) ===")
        for rec in iface.recommend(M, N, K, prec, lay, top_k=3):
            vs = materialize(rec, la, lb)
            tile = f"{rec['cta_tile_m']}x{rec['cta_tile_n']}x{rec['cta_tile_k']}"
            print(f"  rank #{rec['rank']} tile={tile} clus={rec['cluster_m']}x{rec['cluster_n']} "
                  f"split_k={rec['split_k']} raster={rec['cta_order']} swz={rec['swizzle_factor']} "
                  f"-> {len(vs)} variant(s)")
            for v in vs:
                sched = v["config"]["scheduler"].split("::")[-1]
                ks = v["config"]["kernel_schedule"].split("::")[-1].replace("KernelTma", "")
                print(f"        {sched:18s} {ks:24s} splits={v['splits']}")
    iface.close()
