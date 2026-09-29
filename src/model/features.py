#!/usr/bin/env python3
"""
Feature matrix builder for the CUTLASS SM90 GEMM selection model.

Implements the hardware-aware features (groups G0-G7) as a *pure* vectorized
function featurize(df), with zero dependence on any measured/NCU column, so
the identical function scores arbitrary valid configs at inference time.

Pipeline:
    load_split(conn, ...) -> raw rows (config params + label + provenance)
    featurize(df)         -> + G0-G7 feature columns
    add_labels(df)        -> + group_best_tflops, y_norm, relevance_grade, rank
    -> concat train+eval with a `split` column, write parquet (+ manifest json)

Audits (--self-test) validate the feature formulas against ncu_runs:
    restream_factor vs NCU l2_read_sectors (L2 absorbs re-streaming, not DRAM),
    smem_total vs is_valid_config, resident_panel vs measured l2_hit_rate, etc.

Usage:
    python src/model/features.py --db autotuner.db --out features.parquet
    python src/model/features.py --db train.db --train-tags bf16_final \
        --eval-db eval.db --eval-tag bf16_eval --out features.parquet
    python src/model/features.py --db autotuner.db --self-test
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------
# Hardware constants (GH200 / SM90a). Parameters of ratio features, not features.
# ----------------------------------------------------------------------
NUM_SMS = 132
SMEM_CAP = 232448  # 227 KB; matches scheduler HOPPER_SMEM_LIMIT_BYTES
L2_BYTES = 50e6  # L2 phase-transition knee from NCU profiling
# Dense tensor-core peak: 1530 MHz × 4 FMA/SM/cycle × NUM_SMS × 1024 FLOPs/MMA
PEAK_TC_BF16 = 1530e6 * 4 * NUM_SMS * 1024  # ≈827.2 TFLOP/s (BF16/FP16)
DRAM_BW = 4.0e12  # GH200 HBM3 B/s (ratio scale only)
REG_FILE_PER_SM = 65536
MAX_CTA_PER_SM = 32
GRADE_MAX = 31  # cap on relevance grade (NDCG gain stability)

# CUTLASS dtype name -> operand byte width (BF16/FP16=2, FP32=4, FP8=1).
_DTYPE_BYTES = {
    "bf16": 2,
    "bfloat16": 2,
    "bfloat16_t": 2,
    "cutlass::bfloat16_t": 2,
    "f16": 2,
    "half": 2,
    "half_t": 2,
    "cutlass::half_t": 2,
    "f32": 4,
    "float": 4,
    "cutlass::float": 4,
    "tf32": 4,
    "tfloat32_t": 4,
    "cutlass::tfloat32_t": 4,
    "e4m3": 1,
    "float_e4m3_t": 1,
    "cutlass::float_e4m3_t": 1,
    "e5m2": 1,
    "float_e5m2_t": 1,
    "cutlass::float_e5m2_t": 1,
}


def _peak_tc_flops(operand_bytes: np.ndarray, peak_bf16: float) -> np.ndarray:
    """Tensor-core peak FLOP/s by operand width (GH200 SM90 ratios).

    FP8 ~2× BF16; BF16/FP16 = ``peak_bf16``; FP32 GEMM uses TF32 MMA ~0.5× BF16.
    """
    return np.where(
        operand_bytes <= 1,
        peak_bf16 * 2.0,
        np.where(operand_bytes >= 4, peak_bf16 * 0.5, peak_bf16),
    )


def _dtype_bytes(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip().str.lower()
    out = s.map(_DTYPE_BYTES)
    if out.isna().any():
        bad = sorted(s[out.isna()].unique())
        raise ValueError(f"Unknown CUTLASS dtype name(s); extend _DTYPE_BYTES: {bad}")
    return out.astype("int64")


def _sched_class(kernel_schedule: pd.Series) -> pd.Series:
    """ws | pingpong | cooperative, from the kernel_schedule string."""
    k = kernel_schedule.astype(str).str.lower()
    cls = np.where(k.str.contains("cooperative"), "cooperative", np.where(k.str.contains("pingpong"), "pingpong", "ws"))
    return pd.Series(cls, index=kernel_schedule.index)


# ----------------------------------------------------------------------
# The pure feature function. Operates on a DataFrame, returns a new one with
# all G0-G7 feature columns added. No label/measured inputs anywhere.
# ----------------------------------------------------------------------
def featurize(df: pd.DataFrame) -> pd.DataFrame:
    """Add the analytic feature columns to a frame of (config, M, N, K) rows.

    Pure function of the configuration and problem: it reads no measured column, so the
    same code scores arbitrary valid configs at inference time.
    """
    d = df.copy()
    M, N, K = d["M"].astype("float64"), d["N"].astype("float64"), d["K"].astype("float64")
    tm, tn, tk = d["tile_m"].astype("float64"), d["tile_n"].astype("float64"), d["tile_k"].astype("float64")
    stages = d["stages"].astype("float64")
    cm, cn = d["cluster_m"].astype("float64"), d["cluster_n"].astype("float64")

    ba = _dtype_bytes(d["cutlass_type_a"]).astype("float64")
    bb = _dtype_bytes(d["cutlass_type_b"]).astype("float64")
    bc = _dtype_bytes(d["cutlass_type_c"]).astype("float64")
    peak_tc = _peak_tc_flops(ba, PEAK_TC_BF16)

    sched = _sched_class(d["kernel_schedule"])
    is_coop = sched == "cooperative"
    is_pingpong = sched == "pingpong"
    threads_per_block = np.where(is_coop | is_pingpong, 384.0, 256.0)
    consumer_wgs = np.where(is_coop, 2.0, 1.0)  # Pingpong overlaps epilogue, not mainloop -> 1 here (G5)

    # ---- shared derived quantities ----
    tiles_m = np.ceil(M / tm)
    tiles_n = np.ceil(N / tn)
    num_tiles = tiles_m * tiles_n
    k_iters = np.ceil(K / tk)
    smem_stage = tm * tk * ba + tn * tk * bb
    smem_mainloop = stages * smem_stage
    is_nosmem = d["epilogue_schedule"].astype(str).str.contains("NoSmem")
    is_coop_epi = d["epilogue_schedule"].astype(str).str.contains("Cooperative")
    epi_m = np.where(is_coop_epi, np.minimum(128.0, tm), np.minimum(64.0, tm))
    n_perf = np.where((bc == 1) & (np.mod(tn, 64) == 0), 64.0, 32.0)
    tn_i = tn.astype(np.int64)
    n_perf_i = np.minimum(n_perf, tn).astype(np.int64)
    epi_n = np.where(is_coop_epi, np.gcd(np.minimum(32, tn_i), tn_i), np.gcd(n_perf_i, tn_i))
    epi_tiles = np.ceil(tm / epi_m) * np.ceil(tn / epi_n)
    stages_d = np.minimum(epi_tiles, 2.0)
    reuse_smem = bc * 8.0 > 8.0
    stages_c_reuse = np.maximum(np.minimum(epi_tiles, 4.0), stages_d + 1.0)
    stages_c_noreuse = np.minimum(epi_tiles, 4.0)
    per_stage = epi_m * epi_n * bc
    per_stage_aligned = np.floor((per_stage + 127.0) / 128.0) * 128.0
    smem_epi_reuse = per_stage_aligned * stages_c_reuse
    smem_epi_noreuse = per_stage_aligned * stages_c_noreuse + per_stage_aligned * stages_d
    smem_epi = np.where(reuse_smem, smem_epi_reuse, smem_epi_noreuse)
    smem_total = np.where(
        is_nosmem,
        smem_mainloop,
        np.where(is_coop | is_pingpong, smem_mainloop + smem_epi, np.maximum(smem_mainloop, smem_epi)),
    )  # WS TMA = union; NoSmem = mainloop only (G4)

    # ---- G4 first (owns blocks_per_sm, consumed by G1/G5) ----
    blocks_per_sm = np.minimum(np.floor(SMEM_CAP / smem_total), MAX_CTA_PER_SM)
    reg_pressure_proxy = (tm * tn) / threads_per_block
    reg_limited_blocks_per_sm = np.floor(REG_FILE_PER_SM / (reg_pressure_proxy * threads_per_block))
    true_blocks_per_sm = np.maximum(1.0, np.minimum(blocks_per_sm, reg_limited_blocks_per_sm))

    d["smem_total"] = smem_total
    d["smem_frac"] = smem_total / SMEM_CAP
    d["blocks_per_sm"] = blocks_per_sm
    d["reg_pressure_proxy"] = reg_pressure_proxy
    d["reg_limited_blocks_per_sm"] = reg_limited_blocks_per_sm
    d["true_blocks_per_sm"] = true_blocks_per_sm
    d["bytes_per_stage"] = smem_stage

    # ---- G1 grid decomposition (uses true_blocks_per_sm) ----
    slots = NUM_SMS * true_blocks_per_sm
    sm_subscription = num_tiles / slots
    last_wave_eff = num_tiles / (np.ceil(sm_subscription) * slots)
    d["sm_subscription"] = sm_subscription
    d["last_wave_eff"] = last_wave_eff
    d["tiles_m"] = tiles_m
    d["tiles_n"] = tiles_n
    d["k_iters"] = k_iters
    d["m_quant_waste"] = (tiles_m * tm - M) / M
    d["n_quant_waste"] = (tiles_n * tn - N) / N

    # ---- G2 roofline / intensity ----
    d["mainloop_compute_intensity"] = (2.0 * tm * tn) / (tm * ba + tn * bb)
    d["tile_aspect_ratio"] = np.log2(tm / tn)
    d["problem_arith_intensity"] = (2.0 * M * N * K) / (ba * M * K + bb * K * N + bc * M * N)

    # ---- G3 memory traffic & cache ----
    ab_bytes = ba * M * K + bb * K * N
    d["restream_factor"] = (tiles_n * ba * M * K + tiles_m * bb * K * N) / ab_bytes
    d["working_set_vs_L2"] = ab_bytes / L2_BYTES
    d["fits_L2"] = (ab_bytes < L2_BYTES).astype("int64")
    d["resident_panel_vs_L2"] = np.minimum(ba * tm * K, bb * tn * K) / L2_BYTES

    # ---- G5 pipeline balance ----
    consumer_time = (2.0 * tm * tn * tk) / (consumer_wgs * (peak_tc / NUM_SMS))
    producer_time = smem_stage / DRAM_BW
    d["producer_consumer_ratio"] = consumer_time / producer_time
    d["pipeline_fill_frac"] = np.minimum(stages, k_iters) / stages
    d["stage_amortization"] = k_iters / stages
    d["stages_vs_smem"] = stages * d["smem_frac"]  # product for MLP; stages & smem_frac also present

    # ---- G6 scheduler / StreamK ----
    d["streamk_applicability"] = np.maximum(0.0, 1.0 - last_wave_eff)
    d["is_streamk"] = d["scheduler"].astype(str).str.lower().str.contains("streamk").astype("int64")

    # ---- G7 cluster topology (cluster_grid_quant deferred to v2) ----
    d["cluster_size"] = cm * cn
    d["cluster_m_f"] = cm
    d["cluster_n_f"] = cn
    d["cluster_fits_m"] = tiles_m / cm
    d["cluster_fits_n"] = tiles_n / cn
    d["cluster_overshoots"] = ((cm > tiles_m) | (cn > tiles_n)).astype("int64")

    # ---- G0 scaffolding ----
    d["log2_M"] = np.log2(M)
    d["log2_N"] = np.log2(N)
    d["log2_K"] = np.log2(K)
    d["sched_class"] = sched

    # Deterministic categorical encoding: pin the levels so train-time and
    # inference-time .cat.codes are identical under enable_categorical. Without
    # this, a single-(shape,layout) inference frame would re-derive different
    # codes (e.g. a constant `layout` column -> code 0 instead of its true code).
    for _col, _levels in CATEGORY_LEVELS.items():
        if _col in d.columns:
            _vals = d[_col].astype(str)
            # Unknown values -> NaN (missing) before constructing, so codes are
            # stable and pandas doesn't warn about out-of-category entries.
            d[_col] = pd.Categorical(_vals.where(_vals.isin(_levels)), categories=_levels)
    return d


# Columns the model consumes. Categoricals one-hot in the MLP / enable_categorical in XGB.
NUMERIC_FEATURES = [
    "log2_M",
    "log2_N",
    "log2_K",
    "tile_m",
    "tile_n",
    "tile_k",
    "stages",
    "cluster_m_f",
    "cluster_n_f",
    "sm_subscription",
    "last_wave_eff",
    "tiles_m",
    "tiles_n",
    "k_iters",
    "m_quant_waste",
    "n_quant_waste",
    "mainloop_compute_intensity",
    "tile_aspect_ratio",
    "problem_arith_intensity",
    "restream_factor",
    "working_set_vs_L2",
    "fits_L2",
    "resident_panel_vs_L2",
    "smem_total",
    "smem_frac",
    "blocks_per_sm",
    "reg_pressure_proxy",
    "reg_limited_blocks_per_sm",
    "true_blocks_per_sm",
    "bytes_per_stage",
    "producer_consumer_ratio",
    "pipeline_fill_frac",
    "stage_amortization",
    "stages_vs_smem",
    "streamk_applicability",
    "is_streamk",
    "cluster_size",
    "cluster_fits_m",
    "cluster_fits_n",
    "cluster_overshoots",
]
CATEGORICAL_FEATURES = ["kernel_schedule", "epilogue_schedule", "scheduler", "sched_class", "layout"]

# Ablation baseline: the problem and the config exactly as written, with nothing
# derived from a hardware constant (NUM_SMS, SMEM_CAP, L2_BYTES, PEAK_TC_BF16,
# DRAM_BW, REG_FILE_PER_SM). The categoricals are shared by both sets.
STRUCTURAL_NUMERIC = [
    "log2_M",
    "log2_N",
    "log2_K",
    "tile_m",
    "tile_n",
    "tile_k",
    "stages",
    "cluster_m_f",
    "cluster_n_f",
]

FEATURE_SETS = ("full", "structural")


def feature_columns(manifest: dict, feature_set: str = "full") -> tuple[list[str], list[str]]:
    """(numeric, categorical) column names for a feature set, validated against a manifest."""
    if feature_set not in FEATURE_SETS:
        raise ValueError(f"unknown feature set {feature_set!r}; expected one of {FEATURE_SETS}")
    numeric = list(manifest["numeric_features"])
    categorical = list(manifest["categorical_features"])
    if feature_set == "structural":
        missing = [c for c in STRUCTURAL_NUMERIC if c not in numeric]
        if missing:
            raise ValueError(f"manifest is missing structural features: {missing}")
        numeric = [c for c in STRUCTURAL_NUMERIC]
    return numeric, categorical

# Frozen categorical level orderings — the single source of truth for the
# .cat.codes the model is trained and served on. The order matches the historical
# astype('category') (sorted unique) so a retrain reproduces the prior encoding.
CATEGORY_LEVELS = {
    "kernel_schedule": [
        "cutlass::gemm::KernelTmaWarpSpecialized",
        "cutlass::gemm::KernelTmaWarpSpecializedCooperative",
        "cutlass::gemm::KernelTmaWarpSpecializedPingpong",
    ],
    "epilogue_schedule": [
        "cutlass::epilogue::NoSmemWarpSpecialized",
        "cutlass::epilogue::TmaWarpSpecialized",
        "cutlass::epilogue::TmaWarpSpecializedCooperative",
    ],
    "scheduler": [
        "cutlass::gemm::PersistentScheduler",
        "cutlass::gemm::StreamKScheduler",
    ],
    "sched_class": ["cooperative", "pingpong", "ws"],
    "layout": ["NN", "NT", "TN", "TT"],
}


# ----------------------------------------------------------------------
# Labeling
# ----------------------------------------------------------------------
def add_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Label rows within their (M, N, K, layout) group.

    Adds group_id, group_best_tflops, y_norm (throughput relative to the group best),
    rank_in_group and relevance_grade: configs whose throughput gap is within measurement
    noise share a grade.
    """
    d = df.copy().reset_index(drop=True)
    gcols = ["M", "N", "K", "layout_a", "layout_b"]
    d["group_id"] = d.groupby(gcols, sort=False).ngroup()
    d["group_best_tflops"] = d.groupby("group_id")["mean_tflops"].transform("max")
    d["y_norm"] = d["mean_tflops"] / d["group_best_tflops"]
    d["rank_in_group"] = d.groupby("group_id")["mean_tflops"].rank(ascending=False, method="min")

    means = d["mean_tflops"].to_numpy()
    stds = np.nan_to_num(d["std_tflops"].to_numpy(), nan=0.0)
    grades = np.empty(len(d), dtype="int64")
    for pos in d.groupby("group_id").indices.values():
        grades[pos] = _grades_array(means[pos], stds[pos])
    d["relevance_grade"] = grades
    return d


def _grades_array(means: np.ndarray, stds: np.ndarray) -> np.ndarray:
    """Tie-collapsed grades for one group. Sort desc, merge configs whose gap
    < 2*sqrt(stdi^2+stdj^2) into one band; grade = GRADE_MAX - band (floored at 0)."""
    if len(means) == 1:
        return np.array([GRADE_MAX], dtype="int64")
    order = means.argsort()[::-1]
    m, s = means[order], stds[order]
    new_band = (m[:-1] - m[1:] >= 2.0 * np.sqrt(s[:-1] ** 2 + s[1:] ** 2)).astype(int)
    band = np.concatenate([[0], np.cumsum(new_band)])
    grade = np.clip(GRADE_MAX - band, 0, GRADE_MAX)
    out = np.empty(len(means), dtype="int64")
    out[order] = grade
    return out


# ----------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------
_CFG = (
    "tile_m,tile_n,tile_k,cluster_m,cluster_n,stages,kernel_schedule,"
    "epilogue_schedule,scheduler,cutlass_type_a,cutlass_type_b,cutlass_type_c,"
    "layout_a,layout_b"
)


def load_train(conn, tags: list) -> pd.DataFrame:
    """Successful runs under the given tags, joined with their configs and plan provenance, as the training split."""
    placeholders = ",".join("?" * len(tags))
    df = pd.read_sql(
        f"""
        SELECT r.name, r.M, r.N, r.K, r.mean_tflops, r.std_tflops, {_CFG},
               p.sampling_method, p.layer, p.regime, p.proxy_rank, p.layout
        FROM runs r
        JOIN configs c ON r.name = c.name
        LEFT JOIN eval_plan p
          ON p.name=r.name AND p.M=r.M AND p.N=r.N AND p.K=r.K AND p.tag=r.tag
        WHERE r.status='success' AND r.tag IN ({placeholders}) AND r.mean_tflops > 0
    """,
        conn,
        params=list(tags),
    )
    df["split"] = "train"
    return df


_DTYPE_SQL = {
    "bf16": (
        "c.cutlass_type_a = 'cutlass::bfloat16_t' "
        "AND c.cutlass_type_b = 'cutlass::bfloat16_t'"
    ),
    "fp32": (
        "c.cutlass_type_a = 'float' "
        "AND c.cutlass_type_b = 'float'"
    ),
    "fp8_e4m3": (
        "c.cutlass_type_a = 'cutlass::float_e4m3_t' "
        "AND c.cutlass_type_b = 'cutlass::float_e4m3_t'"
    ),
}


def _dtype_filter(dtype: str) -> str:
    key = dtype.strip().lower().replace("-", "_")
    if key not in _DTYPE_SQL:
        raise ValueError(f"unknown --dtype {dtype!r}; choose from {list(_DTYPE_SQL)}")
    return _DTYPE_SQL[key]


def holdout_eval_from_train(train: pd.DataFrame, frac: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split train rows into train/eval by (M,N,K,layout) group — for dtypes without an oracle DB."""
    if not (0.0 < frac < 1.0):
        raise ValueError("--holdout-frac must be in (0, 1)")
    gcols = ["M", "N", "K", "layout_a", "layout_b"]
    keys = train[gcols].drop_duplicates().reset_index(drop=True)
    rng = np.random.default_rng(seed)
    n_hold = max(1, int(round(len(keys) * frac)))
    hold_idx = rng.choice(len(keys), size=n_hold, replace=False)
    hold_set = set(map(tuple, keys.iloc[hold_idx].to_numpy()))
    is_hold = [tuple(x) in hold_set for x in train[gcols].to_numpy()]
    ev = train[is_hold].copy()
    tr = train[~np.array(is_hold)].copy()
    ev["split"] = "eval"
    tr["split"] = "train"
    return tr, ev


def load_eval(conn, min_configs: int, eval_tag: str | None = None, dtype: str = "bf16") -> pd.DataFrame:
    """Near-exhaustive eval groups.

    Default (eval_tag=None): runs with tag IS NULL in a merged registry DB.
    Tagged (e.g. bf16_eval): exhaustive eval_plan shapes joined to runs.
    """
    dtype_sql = _dtype_filter(dtype)
    if eval_tag is None:
        grp = pd.read_sql(
            f"""
            SELECT r.M, r.N, r.K, c.layout_a, c.layout_b, COUNT(*) n
            FROM runs r JOIN configs c ON r.name=c.name
            WHERE r.status='success' AND r.mean_tflops > 0
              AND r.tag IS NULL
              AND {dtype_sql}
            GROUP BY r.M, r.N, r.K, c.layout_a, c.layout_b
            HAVING COUNT(*) >= ?
        """,
            conn,
            params=(min_configs,),
        )
        if grp.empty:
            return grp
        keys = set(map(tuple, grp[["M", "N", "K", "layout_a", "layout_b"]].to_numpy()))
        df = pd.read_sql(
            f"""
            SELECT r.name, r.M, r.N, r.K, r.mean_tflops, r.std_tflops, {_CFG}
            FROM runs r JOIN configs c ON r.name=c.name
            WHERE r.status='success' AND r.mean_tflops > 0
              AND r.tag IS NULL
              AND {dtype_sql}
        """,
            conn,
        )
        mask = [tuple(x) in keys for x in df[["M", "N", "K", "layout_a", "layout_b"]].to_numpy()]
        df = df[mask].copy()
        for col in ("sampling_method", "layer", "regime", "proxy_rank", "layout"):
            df[col] = pd.NA
        df["split"] = "eval"
        return df

    grp = pd.read_sql(
        f"""
        SELECT r.M, r.N, r.K, c.layout_a, c.layout_b, COUNT(*) n
        FROM runs r
        JOIN configs c ON r.name=c.name
        JOIN eval_plan p
          ON p.name=r.name AND p.M=r.M AND p.N=r.N AND p.K=r.K
        WHERE r.status='success' AND r.mean_tflops > 0
          AND p.tag = ?
          AND {dtype_sql}
        GROUP BY r.M, r.N, r.K, c.layout_a, c.layout_b
        HAVING COUNT(*) >= ?
    """,
        conn,
        params=(eval_tag, min_configs),
    )
    if grp.empty:
        return grp
    keys = set(map(tuple, grp[["M", "N", "K", "layout_a", "layout_b"]].to_numpy()))
    df = pd.read_sql(
        f"""
        SELECT r.name, r.M, r.N, r.K, r.mean_tflops, r.std_tflops, {_CFG},
               p.sampling_method, p.layer, p.regime, p.proxy_rank, p.layout
        FROM runs r
        JOIN configs c ON r.name=c.name
        JOIN eval_plan p
          ON p.name=r.name AND p.M=r.M AND p.N=r.N AND p.K=r.K
        WHERE r.status='success' AND r.mean_tflops > 0
          AND p.tag = ?
          AND {dtype_sql}
    """,
        conn,
        params=(eval_tag,),
    )
    mask = [tuple(x) in keys for x in df[["M", "N", "K", "layout_a", "layout_b"]].to_numpy()]
    df = df[mask].copy()
    df["split"] = "eval"
    return df


def _layout_label(df: pd.DataFrame) -> pd.Series:
    code = {"rowmajor": "T", "columnmajor": "N", "row": "T", "column": "N"}
    a = df["layout_a"].astype(str).str.lower().str.replace("cutlass::layout::", "", regex=False).map(code).fillna("?")
    b = df["layout_b"].astype(str).str.lower().str.replace("cutlass::layout::", "", regex=False).map(code).fillna("?")
    return a + b


# ----------------------------------------------------------------------
# Self-tests: feature formulas vs measured NCU columns
# ----------------------------------------------------------------------
def self_test(conn) -> int:
    """Audit pure features against measured ncu_runs via WITHIN-SHAPE Spearman
    (the meaningful lens for a within-shape ranker), BF16-only."""
    print("== Feature audits vs ncu_runs (BF16, within-shape Spearman) ==")
    ncu = pd.read_sql(
        """
        SELECT n.M,n.N,n.K, n.l2_read_sectors, n.l2_hit_rate, n.registers_per_thread,
               n.smem_dynamic_bytes,
               c.tile_m,c.tile_n,c.tile_k,c.cluster_m,c.cluster_n,c.stages,
               c.kernel_schedule,c.epilogue_schedule,c.scheduler,
               c.cutlass_type_a,c.cutlass_type_b,c.cutlass_type_c,c.layout_a,c.layout_b
        FROM ncu_runs n JOIN configs c ON n.name=c.name
        WHERE n.status='success'
    """,
        conn,
    )
    if ncu.empty:
        print("  (no NCU rows; skipping)")
        return 0
    ncu = ncu[_dtype_bytes(ncu["cutlass_type_a"]) == 2].copy()  # BF16 only
    f = featurize(ncu)
    nshapes = f.groupby(["M", "N", "K"]).ngroups
    print(f"  {len(f)} BF16 profiles across {nshapes} shapes")

    def ws_spearman(feat, meas):
        cors = []
        for _, g in f.groupby(["M", "N", "K"]):
            if len(g) >= 5 and g[meas].nunique() > 2 and g[feat].nunique() > 1:
                cors.append(g[feat].corr(g[meas], method="spearman"))
        return (float(np.nanmean(cors)) if cors else float("nan")), len(cors)

    checks = [
        (
            "restream_factor",
            "l2_read_sectors",
            "+",
            "more re-streaming -> more L2 reads (re-streaming is L2-absorbed, not DRAM)",
        ),
        ("resident_panel_vs_L2", "l2_hit_rate", "-", "bigger panel/L2 -> worse residence -> lower hit"),
        ("reg_pressure_proxy", "registers_per_thread", "+", "bigger tile -> more registers"),
    ]
    rc = 0
    for feat, meas, sign, why in checks:
        rho, n = ws_spearman(feat, meas)
        good = (rho > 0) if sign == "+" else (rho < 0)
        rc += 0 if (good or np.isnan(rho)) else 1
        print(
            f"  {feat:22} vs {meas:21} rho={rho:+.2f} / {n} shapes  expect {sign}  {'OK' if good else 'CHECK'}  ({why})"
        )

    v = f[f.smem_dynamic_bytes > 0]
    if not v.empty:
        med = ((v.smem_total - v.smem_dynamic_bytes).abs() / v.smem_dynamic_bytes).median()
        print(
            f"  {'smem_total':22} vs {'smem_dynamic_bytes':21} rel_err={med:.1%}            "
            f"{'OK' if med < 0.20 else 'CHECK'}"
        )
    print(f"\n  audits complete ({rc} sign mismatches).")
    return 0


# ----------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True, help="SQLite DB for training runs")
    ap.add_argument(
        "--eval-db",
        default=None,
        help="optional separate DB for exhaustive eval (e.g. autotuner_bf16_eval.db)",
    )
    ap.add_argument("--train-tags", nargs="+", default=["sweep"],
                    metavar="TAG", help="eval_plan / runs tag(s) to use as training data")
    ap.add_argument("--eval-min-configs", type=int, default=3000)
    ap.add_argument(
        "--eval-tag",
        default=None,
        help="eval_plan tag for exhaustive eval (e.g. bf16_eval); default tag IS NULL",
    )
    ap.add_argument(
        "--dtype",
        default="bf16",
        choices=["bf16", "fp32", "fp8_e4m3"],
        help="dtype filter for load_eval (train rows come from --train-tags)",
    )
    ap.add_argument(
        "--holdout-frac",
        type=float,
        default=0.0,
        help="if no eval rows loaded, hold out this fraction of train groups for eval",
    )
    ap.add_argument("--holdout-seed", type=int, default=42)
    ap.add_argument("--out", default="features.parquet")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    conn = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    eval_conn = (
        sqlite3.connect(f"file:{args.eval_db}?mode=ro", uri=True) if args.eval_db else conn
    )
    try:
        if args.self_test:
            return self_test(conn)

        train = load_train(conn, args.train_tags)
        ev = load_eval(eval_conn, args.eval_min_configs, eval_tag=args.eval_tag, dtype=args.dtype)
        if ev.empty and args.holdout_frac > 0:
            train, ev = holdout_eval_from_train(train, args.holdout_frac, args.holdout_seed)
            print(
                f"holdout split: train={len(train):,} rows  eval={len(ev):,} rows  "
                f"(frac={args.holdout_frac}, seed={args.holdout_seed})"
            )
        else:
            print(f"loaded: train={len(train):,} rows  eval={len(ev):,} rows")

        # No (shape,layout) on both sides.
        if not ev.empty:
            tk = set(map(tuple, train[["M", "N", "K", "layout_a", "layout_b"]].to_numpy()))
            ek = set(map(tuple, ev[["M", "N", "K", "layout_a", "layout_b"]].to_numpy()))
            overlap = tk & ek
            if overlap:
                print(f"  dropping {len(overlap)} (shape,layout) groups present in BOTH splits from train")
                mask = [tuple(x) not in overlap for x in train[["M", "N", "K", "layout_a", "layout_b"]].to_numpy()]
                train = train[mask].copy()

        df = pd.concat([train, ev], ignore_index=True)
        df["layout"] = np.where(df["layout"].notna(), df["layout"], _layout_label(df))
        df = featurize(df)
        # Overlaps removed above, so every group is entirely train OR entirely eval;
        # labeling once keeps group_id globally unique. group_best is the sampled best
        # on train groups and the true oracle on the near-exhaustive eval groups.
        df = add_labels(df)

        if not ev.empty:
            n_eval_groups = df[df.split == "eval"]["group_id"].nunique()
            print(f"  eval groups (true-oracle): {n_eval_groups}")

        manifest = {
            "numeric_features": NUMERIC_FEATURES,
            "categorical_features": CATEGORICAL_FEATURES,
            "label_cols": [
                "mean_tflops",
                "std_tflops",
                "group_best_tflops",
                "y_norm",
                "relevance_grade",
                "rank_in_group",
            ],
            "group_key": ["M", "N", "K", "layout_a", "layout_b"],
            "provenance": ["sampling_method", "layer", "regime", "proxy_rank", "split"],
            "hw_constants": {
                "NUM_SMS": NUM_SMS,
                "SMEM_CAP": SMEM_CAP,
                "L2_BYTES": L2_BYTES,
                "PEAK_TC_BF16": PEAK_TC_BF16,
                "DRAM_BW": DRAM_BW,
                "REG_FILE_PER_SM": REG_FILE_PER_SM,
                "GRADE_MAX": GRADE_MAX,
            },
            "n_rows": len(df),
            "n_groups": int(df["group_id"].nunique()),
            "sources": {
                "train_db": args.db,
                "train_tags": list(args.train_tags),
                "eval_db": args.eval_db or args.db,
                "eval_tag": args.eval_tag,
                "dtype": args.dtype,
                "eval_min_configs": args.eval_min_configs,
            },
        }
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(args.out, index=False)
        with open(args.out.rsplit(".", 1)[0] + ".manifest.json", "w") as fh:
            json.dump(manifest, fh, indent=2)
        print(f"wrote {args.out}  ({len(df):,} rows, {manifest['n_groups']:,} groups)")
        print(f"  features: {len(NUMERIC_FEATURES)} numeric + {len(CATEGORICAL_FEATURES)} categorical")
        return 0
    finally:
        if eval_conn is not conn:
            eval_conn.close()
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
