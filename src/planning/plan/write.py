#!/usr/bin/env python3
"""
write.py — Create a named sweep plan in the registry.

Usage:
    python src/planning/plan.py write \\
        --tag sweep --shapes plans/shapes.json \\
        [--db path/to/autotuner.db] \\
        [--k-default 2000] [--k-large 1000] [--k-top-frac 0.75]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from repo_paths import REPO_ROOT, SRC

SCRIPT_DIR = Path(__file__).resolve().parent
AUTOTUNER_DIR = SRC / "autotuner"
sys.path.insert(0, str(AUTOTUNER_DIR))

from config_space import PRECISIONS, generate_search_space
from registry import SQLiteRegistry

# ── Proxy constants ────────────────────────────────────────────────────────────
N_SMS = 132
MIN_TILE_M = 64
MIN_TILE_N = 16
LARGE_THRESHOLD = 6144

# p75 cost per config (seconds) per regime — for the in-script budget estimate
COST_P75 = {
    "large": 0.755,
    "square": 0.093,
    "wide": 0.111,
    "tall": 0.061,
    "skinny-K": 0.056,
}

WAVE_EFF_MODES = ("old_we", "new_we")

LAYOUTS = {
    "TN": ("cutlass::layout::RowMajor", "cutlass::layout::ColumnMajor"),
    "TT": ("cutlass::layout::RowMajor", "cutlass::layout::RowMajor"),
    "NN": ("cutlass::layout::ColumnMajor", "cutlass::layout::ColumnMajor"),
    "NT": ("cutlass::layout::ColumnMajor", "cutlass::layout::RowMajor"),
}


# ── K selection ───────────────────────────────────────────────────────────────


def _effective_k(shape: dict, args) -> int:
    """K to use for this shape's shortlist, in priority order:
    0. --exhaustive                            → all valid configs (no cap)
    1. is_large (all dims ≥ 6144)            → k_large   (proxy expensive, cap depth)
    2. tense layer                           → k_boundary_extreme (transition points)
    3. boundary + K-extreme (K≤128 or K≥10240) → k_boundary_extreme (proxy blind here)
    4. default                                → k_default
    """
    if args.exhaustive:
        return 10**9
    M, N, K = shape["M"], shape["N"], shape["K"]
    if M >= LARGE_THRESHOLD and N >= LARGE_THRESHOLD and K >= LARGE_THRESHOLD:
        return args.k_large
    if shape.get("layer") == "tense":
        return args.k_boundary_extreme
    if shape.get("layer") == "boundary" and (K <= 128 or K >= 10240):
        return args.k_boundary_extreme
    return args.k_default


# ── Scoring ────────────────────────────────────────────────────────────────────


def _build_arrays(configs: list[dict]) -> dict:
    tm = np.array([c["tile_m"] for c in configs], dtype=np.float64)
    tn = np.array([c["tile_n"] for c in configs], dtype=np.float64)
    tk = np.array([c["tile_k"] for c in configs], dtype=np.float64)
    cm = np.array([c["cluster_m"] for c in configs], dtype=np.float64)
    cn = np.array([c["cluster_n"] for c in configs], dtype=np.float64)
    s = np.array([c["stages"] for c in configs], dtype=np.float64)
    return {
        "tm": tm,
        "tn": tn,
        "cm": cm,
        "cn": cn,
        "s": s,
        "tile_factor": np.sqrt(tm * tn * tk),
    }


def _wave_eff(nc: np.ndarray, cm: np.ndarray, cn: np.ndarray, mode: str) -> np.ndarray:
    """Wave-efficiency term for the proxy score.

    old_we: last-wave fraction  nc / (num_waves · N_SMS · cluster_m · cluster_n)
    new_we: average across waves  (num_waves − 1 + old_we) / num_waves
            — full waves count as 1.0, so multi-wave configs score higher.
    """
    if mode not in WAVE_EFF_MODES:
        raise ValueError(f"unknown wave-eff mode {mode!r}; choose {WAVE_EFF_MODES}")
    w = nc / N_SMS
    num_waves = np.ceil(w)
    we_last = np.where(w > 0, nc / (num_waves * N_SMS * cm * cn), 0.0)
    if mode == "old_we":
        return we_last
    return np.where(num_waves > 0, (num_waves - 1 + we_last) / num_waves, 0.0)


def _score_all(M: int, N: int, arrs: dict, wave_eff_mode: str = "new_we") -> np.ndarray:
    """Proxy score for all configs at problem (M, N).

    score = wave_eff · stages · sqrt(tile_m · tile_n · tile_k)

    wave_eff from _wave_eff() — see WAVE_EFF_MODES.  Guard-rejected → -1.
    """
    tm, tn, cm, cn, s = arrs["tm"], arrs["tn"], arrs["cm"], arrs["cn"], arrs["s"]
    nc = np.ceil(np.ceil(M / tm) * np.ceil(N / tn) / (cm * cn))
    we = _wave_eff(nc, cm, cn, wave_eff_mode)
    scores = np.sqrt(we * s) * arrs["tile_factor"]
    guard = np.ones(len(tm), dtype=bool)
    guard &= ~((M >= MIN_TILE_M) & (tm > 2 * M))
    guard &= ~((N >= MIN_TILE_N) & (tn > 2 * N))
    return np.where(guard, scores, -1.0)


def _rng_seed(tag: str, M: int, N: int, K: int, layout: str) -> int:
    """Stable RNG seed for stratified sampling, independent of PYTHONHASHSEED."""
    key = f"{tag}:{M}:{N}:{K}:{layout}".encode()
    return int(hashlib.md5(key).hexdigest()[:8], 16)


def _shortlist(
    M: int,
    N: int,
    K: int,
    pool: list[dict],
    arrs: dict,
    k_used: int,
    k_top_frac: float,
    tag: str,
    layout: str,
    wave_eff_mode: str = "new_we",
) -> tuple[list[tuple[int, int, str]], int, bool]:
    """
    Build the shortlist for one (shape, layout).

    Returns:
        entries   — list of (pool_idx, proxy_rank, sampling_method)
        valid_count — number of configs with score > -1
        truncated   — True if valid_count < k_used
    """
    scores = _score_all(M, N, arrs, wave_eff_mode)
    valid_mask = scores > -1.0
    valid_idx = np.where(valid_mask)[0]
    valid_count = len(valid_idx)

    if valid_count == 0:
        return [], 0, True

    # Sort valid configs by score descending
    order = np.argsort(scores[valid_idx])[::-1]
    valid_sorted = valid_idx[order]  # pool indices, rank-1 = valid_sorted[0]

    truncated = valid_count < k_used
    if truncated:
        # Exhaustive coverage of the small pool — tag all as proxy_top
        return (
            [(int(i), r + 1, "proxy_top") for r, i in enumerate(valid_sorted)],
            valid_count,
            True,
        )

    k_top = round(k_used * k_top_frac)
    k_bad = k_used - k_top

    entries: list[tuple[int, int, str]] = [
        (int(valid_sorted[r]), r + 1, "proxy_top") for r in range(k_top)
    ]

    # Stratified sample from remaining valid pool (ranks k_top+1 … valid_count)
    remaining = valid_sorted[k_top:]  # still score-sorted descending

    if k_bad > 0 and len(remaining) > 0:
        rng = np.random.default_rng(seed=_rng_seed(tag, M, N, K, layout))
        n_buckets = min(10, len(remaining))
        buckets = np.array_split(remaining, n_buckets)

        # Distribute k_bad proportionally; first (k_bad % n_buckets) buckets get +1
        per_base = k_bad // n_buckets
        remainder = k_bad % n_buckets
        sampled = 0

        for d, bucket in enumerate(buckets):
            if sampled >= k_bad:
                break
            n_samp = min(per_base + (1 if d < remainder else 0), len(bucket))
            if n_samp <= 0:
                continue
            chosen = rng.choice(bucket, size=n_samp, replace=False)
            for pool_idx in chosen:
                # proxy_rank = position in full valid_sorted array (1-indexed)
                rank_in_valid = k_top + int(np.where(remaining == pool_idx)[0][0]) + 1
                entries.append((int(pool_idx), rank_in_valid, "stratified_bad"))
            sampled += n_samp

    return entries, valid_count, False


# ── Verifications ──────────────────────────────────────────────────────────────


def verify_regime_consistency(shapes: list[dict]) -> int:
    """VERIFY 4: check stored regime vs recomputed is_large. Return mismatch count."""
    mismatches = 0
    for s in shapes:
        M, N, K = s["M"], s["N"], s["K"]
        computed_large = (
            M >= LARGE_THRESHOLD and N >= LARGE_THRESHOLD and K >= LARGE_THRESHOLD
        )
        stored_large = s.get("regime") == "large"
        if computed_large != stored_large:
            print(
                f"  MISMATCH  ({M},{N},{K})  stored={s.get('regime')}  "
                f"computed_large={computed_large}"
            )
            mismatches += 1
    return mismatches


def verify_name_uniqueness(pool_by_layout: dict) -> bool:
    """VERIFY 7: confirm layout is encoded in config names (pools are disjoint)."""
    names_per = {
        ln: {c["name"] for c in pool} for ln, (pool, _) in pool_by_layout.items()
    }
    all_names = [n for ns in names_per.values() for n in ns]
    unique = len(set(all_names))
    total = len(all_names)
    if unique != total:
        print(
            f"  *** COLLISION: {total - unique} names appear in multiple layout pools!"
        )
        return False
    n_layouts = len(pool_by_layout)
    print(
        f"  Layout pools are disjoint ({unique:,} unique names across {n_layouts} layout(s)). ✓"
    )
    return True


def verify_schedule_diversity(
    shapes: list[dict],
    pool_by_layout: dict,
    k_used: int,
    k_top_frac: float,
    tag: str,
    wave_eff_mode: str,
    sample_layout: str = "TN",
) -> None:
    """VERIFY 5: schedule distribution in the top-K for a sample shape+layout."""
    if sample_layout not in pool_by_layout:
        print(f"  Skipped (layout {sample_layout} not in plan).")
        return
    # Pick first square-ish shape
    sample = next((s for s in shapes if s.get("regime") == "square"), shapes[0])
    M, N, K = sample["M"], sample["N"], sample["K"]
    lname = sample_layout
    pool, arrs = pool_by_layout[lname]
    entries, valid_count, trunc = _shortlist(
        M,
        N,
        K,
        pool,
        arrs,
        k_used,
        k_top_frac,
        tag,
        lname,
        wave_eff_mode,
    )

    sched_count: Counter = Counter()
    for pool_idx, rank, method in entries:
        if method == "proxy_top":
            sched_count[pool[pool_idx]["kernel_schedule"].split("::")[-1]] += 1

    print(
        f"  Sample shape ({M},{N},{K}) layout={lname}  top-{sum(sched_count.values())} proxy_top:"
    )
    total = sum(sched_count.values())
    for sched, cnt in sorted(sched_count.items(), key=lambda x: -x[1]):
        pct = 100 * cnt / total if total else 0
        flag = "  ← MISSING" if cnt == 0 else ""
        print(f"    {sched:<45} {cnt:>5}  ({pct:4.1f}%){flag}")
    missing = [s for s, c in sched_count.items() if c == 0]
    if not sched_count:
        print("    (no proxy_top entries)")
    elif missing:
        print(f"  WARNING: schedules absent from proxy_top: {missing}")
    else:
        print("  All schedule variants present. ✓")


def verify_k_blindness(shapes: list[dict]) -> None:
    """VERIFY 6: shapes sharing (M,N) but differing in K get identical shortlists."""
    mn_to_shapes: dict[tuple, list] = defaultdict(list)
    for s in shapes:
        mn_to_shapes[(s["M"], s["N"])].append(s)

    multi_k = {mn: ss for mn, ss in mn_to_shapes.items() if len(ss) > 1}
    n_boundary = sum(
        1 for ss in multi_k.values() for s in ss if s.get("layer") == "boundary"
    )

    print(f"  Distinct (M,N) pairs with multiple K values: {len(multi_k)}")
    print(
        f"  Shapes in those pairs: {sum(len(v) for v in multi_k.values())} "
        f"({n_boundary} boundary)"
    )

    if multi_k:
        # Show the K-extreme boundary cases (tiny-K and large-K)
        k_extreme = sorted(
            [
                (mn, ss)
                for mn, ss in multi_k.items()
                if any(s.get("layer") == "boundary" for s in ss)
            ],
            key=lambda x: min(s["K"] for s in x[1]),
        )[:8]
        if k_extreme:
            print("  Sample boundary (M,N) pairs sharing a shortlist:")
            for (M, N), ss in k_extreme:
                ks = sorted(s["K"] for s in ss)
                layers = [s.get("layer", "?") for s in ss]
                print(f"    ({M:>6},{N:>6})  K={ks}  layers={layers}")
        print()
        print("  Recommendation: the proxy ignores K, so K-extreme boundary shapes")
        print(
            f"  ({n_boundary} of them) get the same shortlist as their (M,N) neighbors."
        )
        print("  Consider bumping K for boundary shapes where layer='boundary' and")
        print("  K is very small (≤128) or very large (≥10240) — those are exactly")
        print("  the shapes where proxy ranking is least trustworthy.")


# ── Main ──────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Create a sweep plan in eval_plan")
    parser.add_argument("--tag", required=True, help="Plan tag, e.g. 'sweep'")
    parser.add_argument(
        "--shapes", required=True, type=Path, help="Shapes JSON from plan.py shapes"
    )
    parser.add_argument(
        "--dtype",
        default="bf16",
        choices=sorted(PRECISIONS),
        help="Precision key from src/autotuner/config_space.py PRECISIONS",
    )
    parser.add_argument(
        "--layouts",
        default="TN,TT,NN,NT",
        help="Comma-separated layout names (FP32/FP8: use TN only)",
    )
    parser.add_argument(
        "--max-shapes",
        type=int,
        default=None,
        help="Use only the first N shapes (debug / smoke plans)",
    )
    parser.add_argument("--db", type=Path, default=REPO_ROOT / "build_cache" / "autotuner.db")
    parser.add_argument(
        "--k-default",
        type=int,
        default=2000,
        help="Shortlist size for standard regimes",
    )
    parser.add_argument(
        "--k-large",
        type=int,
        default=1000,
        help="Shortlist size for large-compute regime (all dims ≥6144)",
    )
    parser.add_argument(
        "--k-boundary-extreme",
        type=int,
        default=3000,
        help="Shortlist size for boundary shapes with K≤128 or K≥10240",
    )
    parser.add_argument(
        "--k-top-frac",
        type=float,
        default=0.75,
        help="Fraction of K from proxy top (rest = stratified random from remainder)",
    )
    parser.add_argument(
        "--wave-eff",
        default=os.environ.get("CKS_WAVE_EFF", "new_we"),
        choices=WAVE_EFF_MODES,
        help="Proxy wave-eff term: old_we=last-wave only, new_we=avg over waves (default: new_we or CKS_WAVE_EFF)",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Count and verify; do not write to DB"
    )
    parser.add_argument(
        "--exhaustive",
        action="store_true",
        help="Include every valid config per shape×layout (oracle / held-out eval set)",
    )
    args = parser.parse_args()

    layout_names = [x.strip() for x in args.layouts.split(",") if x.strip()]
    unknown = [ln for ln in layout_names if ln not in LAYOUTS]
    if unknown:
        parser.error(f"unknown layout(s): {unknown}; known: {sorted(LAYOUTS)}")

    shapes = json.loads(args.shapes.read_text())
    if args.max_shapes is not None:
        shapes = shapes[: args.max_shapes]
    operand_type = PRECISIONS[args.dtype][1]

    def _split(k):
        top = round(k * args.k_top_frac)
        return top, k - top

    print(f"Loaded {len(shapes)} base shapes from {args.shapes}")
    print(f"Tag      : {args.tag}")
    print(f"Dtype    : {args.dtype} ({operand_type})")
    print(f"Layouts  : {','.join(layout_names)}")
    print(f"Wave eff : {args.wave_eff}")
    print(f"Exhaustive: {args.exhaustive}")
    print(f"DB       : {args.db}")
    if args.exhaustive:
        print("K split  : exhaustive (all valid configs per shape×layout)")
    else:
        print(
            f"K split  : default={args.k_default} (top={_split(args.k_default)[0]}+bad={_split(args.k_default)[1]})  "
            f"large={args.k_large} (top={_split(args.k_large)[0]}+bad={_split(args.k_large)[1]})  "
            f"boundary_extreme={args.k_boundary_extreme} (top={_split(args.k_boundary_extreme)[0]}+bad={_split(args.k_boundary_extreme)[1]})  "
            f"frac={args.k_top_frac}"
        )

    n_extreme = sum(
        1
        for s in shapes
        if s.get("layer") == "boundary"
        and (s["K"] <= 128 or s["K"] >= 10240)
        and not (
            s["M"] >= LARGE_THRESHOLD
            and s["N"] >= LARGE_THRESHOLD
            and s["K"] >= LARGE_THRESHOLD
        )
    )
    print(
        f"Boundary-extreme shapes (layer=boundary, K≤128 or K≥10240, not large): {n_extreme}"
    )

    n_layouts = len(layout_names)
    gpu_s = sum(
        COST_P75.get(s.get("regime", "square"), 0.093)
        * _effective_k(s, args)
        * n_layouts
        for s in shapes
    )
    print(
        f"Est. GPU-h : {gpu_s / 3600:.1f}  (p75, {n_layouts} layout(s), before 15% headroom)"
    )

    # ── Build dtype pool per layout ─────────────────────────────────────────────
    print(f"\nBuilding {args.dtype} config pool...", end=" ", flush=True)
    all_configs = generate_search_space(dtypes=[args.dtype], layout_names=layout_names)
    pool_by_layout: dict[str, tuple[list[dict], dict]] = {}
    for lname in layout_names:
        la, lb = LAYOUTS[lname]
        pool = [
            c
            for c in all_configs
            if c["cutlass_type_a"] == operand_type
            and c["layout_a"] == la
            and c["layout_b"] == lb
        ]
        pool_by_layout[lname] = (pool, _build_arrays(pool))
    total_pool = sum(len(v[0]) for v in pool_by_layout.values())
    print(
        f"{total_pool:,} total  "
        + "  ".join(f"{ln}={len(pool_by_layout[ln][0]):,}" for ln in layout_names)
    )

    # ── VERIFY 4: regime consistency ───────────────────────────────────────────
    print("\n── VERIFY 4: Regime/K consistency ──────────────────────────────────")
    mismatches = verify_regime_consistency(shapes)
    if mismatches == 0:
        print(
            f"  All {len(shapes)} shapes: stored regime agrees with recomputed is_large. ✓"
        )
    else:
        print(
            f"  {mismatches} MISMATCH(ES) — stored regime and recomputed is_large disagree."
        )

    # ── VERIFY 7: name uniqueness ──────────────────────────────────────────────
    print("\n── VERIFY 7: Config-name uniqueness ─────────────────────────────────")
    verify_name_uniqueness(pool_by_layout)

    # ── VERIFY 6: K-blindness ─────────────────────────────────────────────────
    print("\n── VERIFY 6: K-blindness ────────────────────────────────────────────")
    verify_k_blindness(shapes)

    # ── Score and build plan ───────────────────────────────────────────────────
    print("\n── Scoring (shape × layout) ─────────────────────────────────────────")
    plan_entries: list[dict] = []
    plan_configs: list[dict] = []
    seen_configs: set[str] = set()
    truncations: list[str] = []
    method_counts: Counter = Counter()

    for shape in shapes:
        M, N, K = shape["M"], shape["N"], shape["K"]
        regime = shape.get("regime", "square")
        layer = shape.get("layer")
        k_used = _effective_k(shape, args)

        for lname, (pool, arrs) in pool_by_layout.items():
            if not pool:
                continue

            entries, valid_count, truncated = _shortlist(
                M,
                N,
                K,
                pool,
                arrs,
                k_used,
                args.k_top_frac,
                args.tag,
                lname,
                args.wave_eff,
            )

            if truncated:
                truncations.append(
                    f"  ({M:>6},{N:>6},{K:>6}) {lname}  valid={valid_count}  K={k_used}  "
                    f"→ truncated to {len(entries)}"
                )

            for pool_idx, proxy_rank, method in entries:
                cfg = pool[pool_idx]
                plan_entries.append(
                    {
                        "name": cfg["name"],
                        "M": M,
                        "N": N,
                        "K": K,
                        "tag": args.tag,
                        "proxy_rank": proxy_rank,
                        "k_used": k_used,
                        "regime": regime,
                        "layer": layer,
                        "sampling_method": method,
                        "layout": lname,
                    }
                )
                method_counts[method] += 1
                if cfg["name"] not in seen_configs:
                    seen_configs.add(cfg["name"])
                    plan_configs.append(cfg)

    # ── VERIFY 5: schedule diversity ───────────────────────────────────────────
    print("\n── VERIFY 5: Schedule diversity in proxy_top ────────────────────────")
    verify_schedule_diversity(
        shapes,
        pool_by_layout,
        args.k_default,
        args.k_top_frac,
        args.tag,
        args.wave_eff,
        sample_layout=layout_names[0],
    )

    # ── Summary ────────────────────────────────────────────────────────────────
    print("\n── Plan summary ─────────────────────────────────────────────────────")
    print(f"  Total entries    : {len(plan_entries):,}")
    print(f"    proxy_top      : {method_counts['proxy_top']:,}")
    print(f"    stratified_bad : {method_counts['stratified_bad']:,}")
    print(f"  Unique configs   : {len(plan_configs):,}")
    print(f"  Truncated pools  : {len(truncations)}")
    if truncations:
        for t in truncations:
            print(t)

    if args.dry_run:
        print("\n[dry-run — nothing written to DB]")
        return

    registry = SQLiteRegistry(args.db)
    print("\nRegistering configs...", end=" ", flush=True)
    registry.register(plan_configs)
    print("done")
    deleted = registry.plan_delete_tag(args.tag)
    print(f"Cleared {deleted:,} existing eval_plan rows for tag '{args.tag}'")
    print("Writing eval_plan entries...", flush=True)
    registry.plan_put(plan_entries)
    print("done")
    print(f"\nPlan '{args.tag}' ready. Next:")
    print(
        f"  python src/autotuner/scheduler.py --tag {args.tag} --num-gpus <N> --phase compile,eval"
    )


if __name__ == "__main__":
    main()
