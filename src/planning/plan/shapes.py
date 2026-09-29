#!/usr/bin/env python3
"""
shapes.py — Generate a training shape grid for proxy-guided sweeps.

Usage:
    python src/planning/plan.py shapes
    python src/planning/plan.py shapes --dry-run
    python src/planning/plan.py shapes --no-pool-check
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from itertools import product
from pathlib import Path

from repo_paths import PLANS, SRC

SEED = 42

HULL_MIN, HULL_MAX = 32, 16384

# Proxy constants (mirrored from plan/write.py)
N_SMS      = 132
MIN_TILE_M = 64
MIN_TILE_N = 16

# K values used by plan/write.py — cost estimate must match these exactly.
K_DEFAULT            = 2000
K_LARGE              = 1000
K_BOUNDARY_EXTREME   = 3000
LARGE_THRESHOLD = 6144   # all dims ≥ this → K_LARGE
POOL_WARN       = 3000   # warn if effective proxy pool < this for any layout

# GH200 transition constants (match src/model/features.py)
L2_BYTES = 50e6
PEAK_TC_BF16 = 1530e6 * 4 * N_SMS * 1024  # ≈827.2 TFLOP/s (BF16/FP16 dense TC)
DRAM_BW = 4.0e12
BF16_BYTES = 2
# Roofline ridge: AI* [FLOP/byte] where compute-bound ↔ memory-bound balance.
AI_RIDGE = PEAK_TC_BF16 / DRAM_BW
TARGET_SHAPES = 593  # 298 interior + 32 tense + 172 boundary + 91 anchor

# ── Eval shapes (excluded from the training grid) ─────────────────────────────
EVAL_SHAPES = [
    (2048,2048,2048),(4096,4096,4096),(64,64,64),(128,128,128),(256,256,256),(512,512,512),
    (32,128,4096),(2048,128,4096),(4096,128,4096),(12288,128,4096),
    (256,4096,4096),(256,12288,4096),(64,12288,4096),
    (2048,2048,128),(4096,11008,4096),(256,256,8192),(512,3072,768),
]
EVAL_SET = frozenset(EVAL_SHAPES)

ANCHORS       = [64, 128, 256, 512, 1024, 1536, 2048, 3072, 4096, 6144, 8192]
LARGE_ANCHORS = [6144, 7168, 8192, 9216, 10240, 12288]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _regime(M, N, K):
    if M >= 6144 and N >= 6144 and K >= 6144:
        return "large"
    if M / N >= 4:
        return "tall"
    if N / M >= 4:
        return "wide"
    if K <= min(M, N) / 4:
        return "skinny-K"
    return "square"


def _entry(M, N, K, layer):
    return {"M": M, "N": N, "K": K, "regime": _regime(M, N, K), "layer": layer}


def _k_used(shape: dict) -> int:
    """Mirror src/planning/plan/write.py::_effective_k (default args)."""
    M, N, K = shape["M"], shape["N"], shape["K"]
    if M >= LARGE_THRESHOLD and N >= LARGE_THRESHOLD and K >= LARGE_THRESHOLD:
        return K_LARGE
    if shape.get("layer") == "tense":
        return K_BOUNDARY_EXTREME
    if shape.get("layer") == "boundary" and (K <= 128 or K >= 10240):
        return K_BOUNDARY_EXTREME
    return K_DEFAULT


def _snap(v):
    return max(HULL_MIN, min(HULL_MAX, round(v / 32) * 32))


# ── Layer generators (structure unchanged from original) ──────────────────────

def _ab_bytes(M, N, K, ba=BF16_BYTES, bb=BF16_BYTES):
    return ba * M * K + bb * K * N


def _arith_intensity(M, N, K, ba=BF16_BYTES, bb=BF16_BYTES, bc=BF16_BYTES):
    return (2.0 * M * N * K) / (ba * M * K + bb * K * N + bc * M * N)


def interior_layer(rng):
    # regime derived via _regime(M,N,K) — _entry() is the single source of truth
    """Shapes drawn from the anchor grid with near-square aspect ratios (the interior of the space)."""
    def sample(cands, n):
        rng.shuffle(cands)
        return [_entry(M, N, K, "interior")
                for M, N, K in cands[:n] if (M, N, K) not in EVAL_SET]

    A = ANCHORS
    shapes = []
    cands = [(M, N, K) for M, N, K in product(A, A, A)
             if 0.5 <= M / N <= 2.0 and 0.5 <= K / M <= 2.0]
    shapes += sample(cands, 53)   # was 85 (−32 raw → ~298 kept after dedup)
    cands = [(M, N, K) for M, N, K in product(A, A, A) if M / N >= 4.0]
    shapes += sample(cands, 70)
    cands = [(M, N, K) for M, N, K in product(A, A, A) if N / M >= 4.0]
    shapes += sample(cands, 70)
    cands = [(M, N, K) for M, N, K in product(A, A, A)
             if K <= min(M, N) / 4 and K >= 64]
    shapes += sample(cands, 70)
    cands = list(product(LARGE_ANCHORS, LARGE_ANCHORS, LARGE_ANCHORS))
    shapes += sample(cands, 45)
    return shapes


def _solve_k_for_intensity(M, N, target_ai, ba=BF16_BYTES, bb=BF16_BYTES, bc=BF16_BYTES):
    """Solve K so _arith_intensity(M,N,K) ≈ target_ai."""
    # 2MNK = target_ai * (ba*MK + bb*KN + bc*MN)
    # → K*(MN - target_ai*ba*M/bc - target_ai*bb*N/bc) = target_ai*M*N  (divide by bc)
    denom = M * N - target_ai * (ba * M + bb * N) / bc
    if denom <= 0:
        return None
    k = target_ai * M * N / denom
    if k < HULL_MIN or k > HULL_MAX:
        return None
    return _snap(k)


def _tense_candidates():
    """Enumerate transition-near shapes; caller picks 32 unique best."""
    cands: list[tuple[float, str, int, int, int]] = []

    def ridge_score(M, N, K):
        return abs(_arith_intensity(M, N, K) - AI_RIDGE) / AI_RIDGE

    def l2_score(M, N, K):
        return abs(_ab_bytes(M, N, K) - L2_BYTES) / L2_BYTES

    # Square ridge: AI = L/3 when M=N=K
    AI_RIDGE * 3
    for L in range(HULL_MIN, HULL_MAX + 1, 32):
        cands.append((ridge_score(L, L, L), "ridge", L, L, L))

    # Square L2: ab_bytes = 4 L^2
    for L in range(HULL_MIN, HULL_MAX + 1, 32):
        cands.append((l2_score(L, L, L), "l2", L, L, L))

    # Rectangular ridge: fix M,N, solve K
    mn_grid = [
        (4096, 1024), (1024, 4096), (3072, 768), (768, 3072),
        (2048, 2048), (8192, 2048), (2048, 8192), (6144, 1536),
        (1536, 6144), (5120, 1280), (1280, 5120),
    ]
    for M, N in mn_grid:
        for scale in (0.92, 0.96, 1.0, 1.04, 1.08):
            k = _solve_k_for_intensity(M, N, AI_RIDGE * scale)
            if k is not None:
                cands.append((ridge_score(M, N, k), "ridge", M, N, k))

    # Rectangular L2: ab = 2K(M+N); scan K and solve for M≈N
    for K in range(512, 8193, 256):
        for target in (0.90 * L2_BYTES, 0.95 * L2_BYTES, 1.0 * L2_BYTES,
                       1.05 * L2_BYTES, 1.10 * L2_BYTES):
            # 2K(2L) = 4KL = target → L = target/(4K)
            L = target / (2 * BF16_BYTES * K)
            L = _snap(L)
            if L < HULL_MIN:
                continue
            for ratio in (1.0, 2.0, 0.5, 4.0, 0.25):
                M = _snap(L * (ratio ** 0.5))
                N = _snap(L / (ratio ** 0.5))
                if M < HULL_MIN or N < HULL_MIN:
                    continue
                cands.append((l2_score(M, N, K), "l2", M, N, K))

    return cands


def tense_layer():
    """32 shapes at hardware transition points: roofline ridge and L2 cliff (~50 MB).

    Ridge: AI ≈ PEAK_TC / DRAM_BW (FLOP/byte) — standard roofline knee.
    L2: operand working set ab_bytes = ba*MK + bb*KN straddles L2_BYTES.
    """
    all_cands = _tense_candidates()
    ridge_cands = sorted((s, M, N, K) for s, kind, M, N, K in all_cands if kind == "ridge")
    l2_cands = sorted((s, M, N, K) for s, kind, M, N, K in all_cands if kind == "l2")

    shapes: list[dict] = []
    seen: set[tuple[int, int, int]] = set()

    def add_from(pool, limit):
        added = 0
        for _score, M, N, K in pool:
            if added >= limit:
                break
            key = (M, N, K)
            if key in EVAL_SET or key in seen:
                continue
            seen.add(key)
            shapes.append(_entry(M, N, K, "tense"))
            added += 1

    add_from(ridge_cands, 16)
    add_from(l2_cands, 16)
    if len(shapes) < 32:
        add_from(ridge_cands, 32 - len(shapes))
    if len(shapes) < 32:
        add_from(l2_cands, 32 - len(shapes))

    if len(shapes) < 32:
        raise RuntimeError(f"tense_layer produced {len(shapes)} shapes, expected 32")

    return shapes[:32]


def boundary_layer(rng):
    """Shapes at the edges of the space: very tall, very wide and skinny-K problems."""
    shapes = []

    def add(cands, n=30):
        rng.shuffle(cands)
        for M, N, K in cands[:n]:
            if (M, N, K) not in EVAL_SET:
                shapes.append(_entry(M, N, K, "boundary"))

    add([(M, N, K)
         for M in [10240, 12288, 16384]
         for N in [64, 128, 256, 512, 1024, 2048, 4096, 8192]
         for K in [1024, 2048, 4096, 8192]], n=35)
    add([(M, N, K)
         for M in [32, 48, 64]
         for N in [8192, 12288, 16384]
         for K in [512, 1024, 2048, 4096, 8192]], n=30)
    add([(M, N, K)
         for N in [32, 48, 64]
         for M in [8192, 12288, 16384]
         for K in [512, 1024, 2048, 4096, 8192]], n=30)
    add([(M, N, K)
         for K in [10240, 16384]
         for M in [256, 512, 1024, 2048, 4096]
         for N in [256, 512, 1024, 2048, 4096]], n=30)
    add([(M, N, K)
         for K in [32, 64, 128]
         for M in [256, 512, 1024, 2048, 4096, 8192]
         for N in [256, 512, 1024, 2048, 4096, 8192]
         if K <= min(M, N) / 4], n=30)
    for M, N, K in product([8192, 12288, 16384], [32, 48, 64], [512, 1024, 2048, 4096]):
        if (M, N, K) not in EVAL_SET:
            shapes.append(_entry(M, N, K, "boundary"))
    for M, N, K in product([32, 48, 64], [8192, 12288, 16384], [512, 1024, 2048, 4096]):
        if (M, N, K) not in EVAL_SET:
            shapes.append(_entry(M, N, K, "boundary"))
    return shapes


def anchor_layer():
    """Axis-aligned ×0.5 / ×2 neighbors of each eval shape, snapped to nearest 32."""
    shapes = []
    seen = set()
    for M, N, K in EVAL_SHAPES:
        for nm, nn, nk in [
            (_snap(M * 2), N, K), (_snap(M // 2), N, K),
            (M, _snap(N * 2), K), (M, _snap(N // 2), K),
            (M, N, _snap(K * 2)), (M, N, _snap(K // 2)),
            (_snap(M * 1.5), N, K), (M, _snap(N * 1.5), K),
        ]:
            if (nm, nn, nk) in EVAL_SET or (nm, nn, nk) in seen:
                continue
            if nm == M and nn == N and nk == K:
                continue
            seen.add((nm, nn, nk))
            shapes.append(_entry(nm, nn, nk, "anchor"))
    return shapes


# ── Dedup with transparency ───────────────────────────────────────────────────

def deduplicate(all_shapes):
    """Dedup tense > interior > boundary > anchor. Returns (result, dropped_counter)."""
    seen: set[tuple] = set()
    result = []
    dropped: Counter = Counter()
    for e in all_shapes:
        key = (e["M"], e["N"], e["K"])
        if key in EVAL_SET:
            continue
        if key in seen:
            dropped[e["layer"]] += 1
        else:
            seen.add(key)
            result.append(e)
    return result, dropped


# ── Eval-neighbor verification ────────────────────────────────────────────────

def _lower_edge(v):
    """v/2 falls below hull_min → no grid shape can provide a smaller neighbor."""
    return v < 2 * HULL_MIN  # v < 64: v/2 < 32

def _upper_edge(v):
    """v is at or above hull_max → no grid shape can be larger."""
    return v >= HULL_MAX


def verify_neighbors(result):
    """
    For each eval shape, classify as INTERPOLATION / EDGE / ISOLATED.
    Returns (rows, any_isolated).
    rows: list of dicts with shape, n_neighbors, axis info, status, suggestions.
    """
    grid = [(e["M"], e["N"], e["K"]) for e in result]

    rows = []
    any_isolated = False

    for M, N, K in EVAL_SHAPES:
        # Collect all grid neighbors (within 2× on ALL axes simultaneously)
        neighbors = [
            (gm, gn, gk) for gm, gn, gk in grid
            if (M / 2 <= gm <= 2 * M) and
               (N / 2 <= gn <= 2 * N) and
               (K / 2 <= gk <= 2 * K)
        ]
        n_nbrs = len(neighbors)

        axis_info = {}
        suggestions = []

        for axis, val, nbrs_vals in [
            ("M", M, [gm for gm, _, _ in neighbors]),
            ("N", N, [gn for _, gn, _ in neighbors]),
            ("K", K, [gk for _, _, gk in neighbors]),
        ]:
            le = _lower_edge(val)
            ue = _upper_edge(val)
            has_smaller = any(v < val for v in nbrs_vals)
            has_larger  = any(v > val for v in nbrs_vals)

            needs_smaller = not le
            needs_larger  = not ue
            bracketed = (not needs_smaller or has_smaller) and (not needs_larger or has_larger)

            axis_info[axis] = {
                "lower_edge": le, "upper_edge": ue,
                "has_smaller": has_smaller, "has_larger": has_larger,
                "bracketed": bracketed,
            }

            if needs_smaller and not has_smaller:
                suggestions.append(f"add {axis}={_snap(val // 2)} (smaller {axis} neighbor)")
            if needs_larger and not has_larger:
                suggestions.append(f"add {axis}={_snap(val * 2)} (larger {axis} neighbor)")

        at_edge    = any(i["lower_edge"] or i["upper_edge"] for i in axis_info.values())
        all(i["bracketed"] for i in axis_info.values())
        non_edge_all_ok = all(
            i["bracketed"]
            for i in axis_info.values()
            if not (i["lower_edge"] or i["upper_edge"])
        )

        if not non_edge_all_ok:
            status = "ISOLATED"
            any_isolated = True
        elif at_edge:
            status = "EDGE"
        else:
            status = "INTERPOLATION"

        rows.append({
            "shape": (M, N, K),
            "n_neighbors": n_nbrs,
            "axis": axis_info,
            "status": status,
            "suggestions": suggestions,
        })

    return rows, any_isolated


# ── Valid-config pool check ───────────────────────────────────────────────────

def pool_check(result, warn_threshold=POOL_WARN):
    """
    For each (M, N) in the grid, count configs with proxy score > -1 per layout.
    Returns list of (shape_entry, {layout: count}) sorted by min-layout count ascending.
    Requires numpy + generate_search_space (torch must be available).
    """
    try:
        import numpy as np
    except ImportError:
        print("  [pool check skipped — numpy not available]")
        return []

    try:
        import sys as _sys
        _sys.path.insert(0, str(SRC / "autotuner"))
        from config_space import generate_search_space
    except Exception as e:
        print(f"  [pool check skipped — could not import generate_search_space: {e}]")
        return []

    BF16 = "cutlass::bfloat16_t"
    LAYOUTS = {
        "TN": ("cutlass::layout::RowMajor",    "cutlass::layout::ColumnMajor"),
        "TT": ("cutlass::layout::RowMajor",    "cutlass::layout::RowMajor"),
        "NN": ("cutlass::layout::ColumnMajor", "cutlass::layout::ColumnMajor"),
        "NT": ("cutlass::layout::ColumnMajor", "cutlass::layout::RowMajor"),
    }

    print("  Loading BF16 config pool for pool check...", end=" ", flush=True)
    all_cfgs = generate_search_space()
    bf16_by_layout = {}
    for lname, (la, lb) in LAYOUTS.items():
        pool = [c for c in all_cfgs
                if c["cutlass_type_a"] == BF16 and c["layout_a"] == la and c["layout_b"] == lb]
        tm = np.array([c["tile_m"] for c in pool])
        tn = np.array([c["tile_n"] for c in pool])
        bf16_by_layout[lname] = (len(pool), tm, tn)

    total = sum(v[0] for v in bf16_by_layout.values())
    per_layout = {ln: v[0] for ln, v in bf16_by_layout.items()}
    print(f"{total:,} total  {per_layout}")

    # Compute effective (non-guard-rejected) pool per unique (M, N)
    unique_mn = {(e["M"], e["N"]): e for e in result}
    # cache: (M, N) -> {layout: count}
    mn_counts = {}
    for (M, N), _ in unique_mn.items():
        counts = {}
        for lname, (_, tm, tn) in bf16_by_layout.items():
            guard = np.ones(len(tm), dtype=bool)
            guard &= ~((M >= MIN_TILE_M) & (tm > 2 * M))
            guard &= ~((N >= MIN_TILE_N) & (tn > 2 * N))
            counts[lname] = int(guard.sum())
        mn_counts[(M, N)] = counts

    # Annotate each shape
    shape_pools = []
    for e in result:
        counts = mn_counts[(e["M"], e["N"])]
        k = _k_used(e)
        min_count = min(counts.values())
        shape_pools.append((e, counts, k, min_count))

    # Sort by min_count ascending (worst first)
    shape_pools.sort(key=lambda x: x[3])
    return shape_pools


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=PLANS / "shapes.json")
    parser.add_argument("--dry-run", action="store_true",
                        help="Run all checks and print counts; do not write file")
    parser.add_argument("--no-pool-check", action="store_true",
                        help="Skip the valid-config pool scan (requires torch)")
    args = parser.parse_args()

    # ── Generate layers ───────────────────────────────────────────────────────
    rng = random.Random(SEED)
    int_shapes = interior_layer(rng)
    tense_shapes = tense_layer()
    bnd_shapes = boundary_layer(rng)
    anc_shapes = anchor_layer()

    raw_counts = {
        "interior": len(int_shapes),
        "tense": len(tense_shapes),
        "boundary": len(bnd_shapes),
        "anchor": len(anc_shapes),
    }

    result, dropped = deduplicate(tense_shapes + int_shapes + bnd_shapes + anc_shapes)

    if len(result) > TARGET_SHAPES:
        trim = len(result) - TARGET_SHAPES
        kept, removed = [], 0
        for e in result:
            if e["layer"] == "anchor" and removed < trim:
                removed += 1
                dropped["anchor"] += 1
                continue
            kept.append(e)
        result = kept
        if len(result) > TARGET_SHAPES:
            raise RuntimeError(
                f"Could not trim to {TARGET_SHAPES}: {len(result)} shapes remain"
            )

    lc = Counter(e["layer"]  for e in result)
    rc = Counter(e["regime"] for e in result)

    # ── Header ────────────────────────────────────────────────────────────────
    print(f"Run 1 shape grid  (seed={SEED})")
    print("=" * 60)
    print(f"HULL: [{HULL_MIN}, {HULL_MAX}]^3")
    print(f"  Shapes AT {HULL_MAX} on any axis cannot be bracketed above.")
    print("  This box is frozen as the interpolation region for all later runs.")
    print()

    # ── Layer counts with dedup transparency ──────────────────────────────────
    print("Layer generation:")
    for layer in ["interior", "tense", "boundary", "anchor"]:
        gen = raw_counts[layer]
        drop = dropped[layer]
        kept = lc[layer]
        print(f"  {layer:<10} generated={gen:>4}  dropped_by_dedup={drop:>3}  kept={kept:>4}")
    print()
    print(f"  Total shapes : {len(result)}   × 4 layouts = {len(result) * 4} problems")
    print()

    # ── Regime breakdown ──────────────────────────────────────────────────────
    print("Regime breakdown:")
    for regime in ["square", "tall", "wide", "skinny-K", "large"]:
        k = K_LARGE if regime == "large" else K_DEFAULT
        print(f"  {regime:<10} : {rc[regime]:>4}   K_used={k}")
    print()

    # ── Regime/K sample (3 per regime) ───────────────────────────────────────
    print("Regime sample (3 per regime) — spot-check K assignment:")
    by_regime: dict[str, list] = {}
    for e in result:
        by_regime.setdefault(e["regime"], []).append(e)
    for regime in ["square", "tall", "wide", "skinny-K", "large"]:
        samples = by_regime.get(regime, [])[:3]
        for e in samples:
            k = _k_used(e)
            print(f"  {regime:<10}  ({e['M']:>6},{e['N']:>6},{e['K']:>6})  K_used={k}  [{e['layer']}]")
    print()

    # ── Cost estimate ─────────────────────────────────────────────────────────
    # K values here must match plan/write.py --k-default / --k-large.
    COST_P75 = {"large": 0.755, "square": 0.093, "wide": 0.111, "tall": 0.061, "skinny-K": 0.056}
    gpu_s = 0.0
    for e in result:
        k = _k_used(e)
        gpu_s += COST_P75.get(e["regime"], 0.093) * k * 4  # 4 layouts
    print(f"Cost estimate (K_default={K_DEFAULT}, K_large={K_LARGE}, 4 layouts, p75):")
    print(f"  {gpu_s / 3600:.1f} GPU-hours  (before 15% crash headroom → target <82h)")
    print()

    # ── Valid-config pool check ───────────────────────────────────────────────
    if not args.no_pool_check:
        print("Valid-config pool check:")
        pools = pool_check(result, warn_threshold=POOL_WARN)
        if pools:
            warned = [(e, counts, k, mn) for e, counts, k, mn in pools if mn < POOL_WARN]
            print("  Worst 10 shapes by min-layout effective pool (proxy-guard applied):")
            for e, counts, k, mn in pools[:10]:
                flag = " *** BELOW WARN" if mn < POOL_WARN else ""
                cs = "  ".join(f"{ln}={v:,}" for ln, v in sorted(counts.items()))
                print(f"    ({e['M']:>6},{e['N']:>6},{e['K']:>6})  K={k}  {cs}  min={mn:,}{flag}")
            if warned:
                print(f"\n  *** WARNING: {len(warned)} shape(s) have effective pool < {POOL_WARN} for some layout.")
                print("      The proxy shortlist will be truncated for these shapes.")
            else:
                print(f"\n  All shapes have effective pool >= {POOL_WARN} for all layouts. ✓")
        print()

    # ── Eval-neighbor verification (most important) ───────────────────────────
    print("Eval-neighbor verification:")
    print("  Neighbor = grid shape within 2× on ALL of M, N, K simultaneously.")
    print("  INTERPOLATION = bracketed (smaller+larger neighbor) on all non-edge axes")
    print("  EDGE          = at hull boundary on some axis (expected for extreme shapes)")
    print("  ISOLATED      = unbracketed on a non-edge axis  ← anchor layer failure")
    print()

    rows, any_isolated = verify_neighbors(result)

    status_counts = Counter(r["status"] for r in rows)

    # Header
    print(f"  {'Shape':<28}  {'nbrs':>4}  M  N  K  Status")
    print(f"  {'-'*28}  {'----':>4}  -  -  -  ------")
    for r in rows:
        M, N, K = r["shape"]
        ai = r["axis"]

        def axis_sym(a):
            info = ai[a]
            if info["lower_edge"] or info["upper_edge"]:
                return "○"  # hull edge
            return "✓" if info["bracketed"] else "✗"

        flag = "  ← ISOLATED — ANCHOR LAYER FAILED" if r["status"] == "ISOLATED" else ""
        print(f"  ({M:>6},{N:>6},{K:>6})  {r['n_neighbors']:>4}  {axis_sym('M')}  {axis_sym('N')}  {axis_sym('K')}  {r['status']}{flag}")

        if r["suggestions"]:
            for s in r["suggestions"]:
                print(f"    → {s}")
        if r["status"] == "EDGE":
            edge_axes = [a for a, i in ai.items() if i["lower_edge"] or i["upper_edge"]]
            print(f"    (edge on {', '.join(edge_axes)})")

    print()
    print(f"  Summary: {status_counts['INTERPOLATION']} INTERPOLATION  "
          f"{status_counts['EDGE']} EDGE  "
          f"{status_counts.get('ISOLATED', 0)} ISOLATED")

    if any_isolated:
        print()
        print("  ERROR: one or more eval shapes are ISOLATED.")
        print("  Add the suggested neighbor shapes before freezing the grid.")

    # ── Write (or not) ────────────────────────────────────────────────────────
    print()
    if args.dry_run:
        print("[dry-run — no file written]")
    elif any_isolated:
        print("[file NOT written — fix ISOLATED shapes first]")
    else:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2))
        print(f"Written → {args.out}")

    if any_isolated:
        sys.exit(1)


if __name__ == "__main__":
    main()
