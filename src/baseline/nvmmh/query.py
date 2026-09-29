#!/usr/bin/env python3
"""
query.py — Thin wrapper around NVIDIA's nvMatmulHeuristics (nvMMH).

nvMMH is a purely analytical, CPU-side model. Given a GEMM problem it returns a
ranked list of kernel configurations, each with an estimated runtime. This module
exposes a single helper, `recommend()`, that returns those configs as plain dicts
with the ranking preserved (rank 1 = nvMMH's top pick).

We only care about bf16 for the baseline, but the precision/layout helpers are
general. No GPU is required to run this module.
"""

import inspect

# ── cuBLAS single-char precision codes used by nvMMH ──────────────────────────
_DTYPE_CUBLAS = {
    "cutlass::half_t":        "H",
    "cutlass::bfloat16_t":    "T",
    "float":                  "S",
    "cutlass::tfloat32_t":    "S",
    "cutlass::float_e4m3_t":  "Q",
    "cutlass::float_e5m2_t":  "R",
    "int8_t":                 "B",
    "int32_t":                "I",
}


def precision_string(type_a: str, type_b: str, type_acc: str, type_d: str) -> str:
    """Build the nvMMH precision string from CUTLASS dtype strings.

    Non-FP8: "<a><acc><d>" (3 chars). FP8: "<a><b><c><acc><d>" (5 chars).
    For bf16 this is "TST".
    """
    a = _DTYPE_CUBLAS[type_a]
    if a.lower() != "q":
        return a + _DTYPE_CUBLAS[type_acc] + _DTYPE_CUBLAS[type_d]
    # FP8 path (unused for the bf16 baseline, kept for completeness).
    return (a + _DTYPE_CUBLAS[type_b] + _DTYPE_CUBLAS[type_d]
            + _DTYPE_CUBLAS[type_acc] + _DTYPE_CUBLAS[type_d])


def layout_enum(layout_a: str, layout_b: str, nvmmh):
    """Map CUTLASS layout strings to NvMatmulHeuristicsMatmulLayout.

    C/D is always ColumnMajor in our template, so the layout is "<a><b>_COL_MAJOR".
    """
    ta = "T" if layout_a.endswith("RowMajor") else "N"
    tb = "T" if layout_b.endswith("RowMajor") else "N"
    return nvmmh.NvMatmulHeuristicsMatmulLayout[f"{ta}{tb}_COL_MAJOR"]


class NvmmhInterface:
    """Initialized nvMMH handle (analytical, CPU only).

    Handles the pre-0.1.0.28 ("legacy") API as well as the newer hardware-descriptor
    API, matching cutlass_library/heuristics_provider.py.
    """

    def __init__(self, gpu: str | None = None):
        import nvMatmulHeuristics as nvmmh
        self.nvmmh = nvmmh
        self.gpu = gpu

        init_params = set(inspect.signature(nvmmh.NvMatmulHeuristicsInterfaceEx.__init__).parameters)
        self._legacy = "load_discovery_implicitly" in init_params

        kwargs = dict(
            backend=nvmmh.NvMatmulHeuristicsTarget.CUTLASS3,
            flags=nvmmh.NvMatmulHeuristicsFlags.PERF_MODEL_BASED_AUTO_TUNING,
        )
        if self._legacy:
            kwargs["gpu"] = nvmmh.NvMatmulHeuristicsNvidiaGpu[gpu] if gpu else None
            kwargs["load_discovery_implicitly"] = True

        self.lh = nvmmh.NvMatmulHeuristicsInterfaceEx(**kwargs)

        self.hw_desc = None
        if not self._legacy and gpu:
            self.hw_desc = self.lh.createHardwareDescriptor()
            self.lh.setHardwarePredefinedGpu(self.hw_desc, nvmmh.NvMatmulHeuristicsNvidiaGpu[gpu])

        self.backend = self.lh.createBackend(nvmmh.NvMatmulHeuristicsTarget.CUTLASS3)

    def recommend(self, M: int, N: int, K: int, precision: str, layout, top_k: int) -> list[dict]:
        """Return up to `top_k` ranked configs for a problem.

        Each dict carries nvMMH's raw fields plus `rank` (1-based) and
        `estimated_runtime_s`. Returns [] if nvMMH declines the problem.
        """
        if self._legacy:
            problem = self.lh.makeNvMatmulHeuristicsProblem(M, N, K, layout, 1)
            kwargs = dict(precision=precision)
        else:
            problem = self.lh.makeNvMatmulHeuristicsProblem((M, N, K), layout, 1)
            kwargs = dict(precision=precision, hardware_descriptor=self.hw_desc)

        try:
            results = self.lh.getEx(problem, top_k, self.backend, **kwargs)
        except Exception:
            return []

        configs = []
        for rank, r in enumerate(results, start=1):
            k = r["kernel"]
            configs.append({
                "rank":                 rank,
                "estimated_runtime_s":  r.get("runtime", 0.0),
                "cta_tile_m":           k.cta_tile_m,
                "cta_tile_n":           k.cta_tile_n,
                "cta_tile_k":           k.cta_tile_k,
                "cluster_m":            k.cluster_m,
                "cluster_n":            k.cluster_n,
                "stages":               k.stages,
                "split_k":              k.split_k,
                "cta_order":            k.cta_order,        # 0 = along_m, 1 = along_n
                "swizzle_factor":       k.swizzle_factor,
            })
        return configs

    def close(self):
        """Release the nvMMH backend and hardware descriptor. Safe to call twice."""
        try:
            if getattr(self, "backend", None):
                self.lh.destroyBackend(self.backend)
                self.backend = None
            if getattr(self, "hw_desc", None):
                self.lh.destroyHardwareDescriptor(self.hw_desc)
                self.hw_desc = None
        except Exception:
            pass


if __name__ == "__main__":
    # Smoke test: print nvMMH's top-5 for a couple of bf16 problems.
    iface = NvmmhInterface(gpu="H100_SXM")
    prec = precision_string("cutlass::bfloat16_t", "cutlass::bfloat16_t", "float", "cutlass::bfloat16_t")
    lay = layout_enum("cutlass::layout::RowMajor", "cutlass::layout::ColumnMajor", iface.nvmmh)
    for (M, N, K) in [(4096, 4096, 4096), (256, 256, 8192)]:
        print(f"\n{M}x{N}x{K}  precision={prec}")
        for c in iface.recommend(M, N, K, prec, lay, top_k=5):
            print(f"  #{c['rank']}  {c['cta_tile_m']}x{c['cta_tile_n']}x{c['cta_tile_k']} "
                  f"clus={c['cluster_m']}x{c['cluster_n']} stages={c['stages']} "
                  f"split_k={c['split_k']} raster={c['cta_order']} swz={c['swizzle_factor']} "
                  f"est={c['estimated_runtime_s']*1e3:.4f}ms")
    iface.close()
