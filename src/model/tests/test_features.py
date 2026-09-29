"""Offline unit tests for the feature pipeline. No DB required.

Validates the G0-G7 formulas against the NCU section-5/6 reference numbers and
checks the labeling (tie-collapsed grades) and that featurize is label-free.
Run with:  python -m pytest model/tests -q
"""

import numpy as np
import pandas as pd
import pytest
from features import NUMERIC_FEATURES, add_labels, featurize


def _row(
    M,
    N,
    K,
    tm,
    tn,
    tk,
    st,
    cm,
    cn,
    sched="KernelTmaWarpSpecialized",
    scheduler="PersistentScheduler",
    epi="cutlass::epilogue::TmaWarpSpecialized",
    dt="bf16",
    tflops=100.0,
    std=0.5,
):
    return dict(
        M=M,
        N=N,
        K=K,
        tile_m=tm,
        tile_n=tn,
        tile_k=tk,
        stages=st,
        cluster_m=cm,
        cluster_n=cn,
        kernel_schedule=sched,
        epilogue_schedule=epi,
        scheduler=scheduler,
        cutlass_type_a=dt,
        cutlass_type_b=dt,
        cutlass_type_c=dt,
        layout_a="RowMajor",
        layout_b="ColumnMajor",
        mean_tflops=tflops,
        std_tflops=std,
    )


def _f(rows):
    return featurize(pd.DataFrame(rows))


def test_compute_intensity_matches_design_doc():
    f = _f(
        [_row(4096, 4096, 4096, 64, tn, 64, 4, 1, 1) for tn in (64, 128, 256)]
        + [_row(4096, 4096, 4096, 128, 128, 64, 4, 1, 1)]
    )
    g = f.set_index(["tile_m", "tile_n"]).mainloop_compute_intensity
    assert g.loc[(64, 64)] == pytest.approx(32.0)
    assert g.loc[(64, 128)] == pytest.approx(42.667, abs=1e-2)
    assert g.loc[(64, 256)] == pytest.approx(51.2, abs=1e-2)
    assert g.loc[(128, 128)] == pytest.approx(64.0)


def test_smem_total_winning_template():
    # 64x128x64, 4 stages, WS, BF16 -> 98304 B (the 97 KB sweet spot).
    f = _f([_row(4096, 4096, 4096, 64, 128, 64, 4, 1, 1)])
    assert f.smem_total.iloc[0] == 98304
    assert f.blocks_per_sm.iloc[0] == 2


def test_nosmem_smem_is_mainloop_only():
    # Pingpong + TMA epilogue stages C/D in smem; NoSmem does not.
    tma_pp = _f(
        [_row(4096, 4096, 4096, 128, 128, 64, 4, 1, 1, sched="KernelTmaWarpSpecializedPingpong")]
    ).smem_total.iloc[0]
    nosmem_pp = _f(
        [
            _row(
                4096,
                4096,
                4096,
                128,
                128,
                64,
                4,
                1,
                1,
                sched="KernelTmaWarpSpecializedPingpong",
                epi="cutlass::epilogue::NoSmemWarpSpecialized",
            )
        ]
    ).smem_total.iloc[0]
    assert nosmem_pp < tma_pp
    assert nosmem_pp == 4 * (128 * 64 * 2 + 128 * 64 * 2)  # mainloop only


def test_cooperative_smem_is_sum_and_larger_than_ws():
    ws = _f([_row(4096, 4096, 4096, 128, 128, 64, 4, 1, 1)]).smem_total.iloc[0]
    coop = _f(
        [_row(4096, 4096, 4096, 128, 128, 64, 4, 1, 1, sched="KernelTmaWarpSpecializedCooperative")]
    ).smem_total.iloc[0]
    assert coop > ws  # union (max) for WS vs sum for Cooperative


def test_restream_factor_is_no_reuse_upper_bound():
    # No-L2-reuse bound for 64x128 cubes: 24/48/96/144x. Must be monotone in shape.
    f = _f([_row(s, s, s, 64, 128, 64, 4, 1, 1) for s in (2048, 4096, 8192, 12288)])
    ests = f.sort_values("M").restream_factor.to_numpy()
    np.testing.assert_allclose(ests, [24, 48, 96, 144], rtol=1e-6)
    assert np.all(np.diff(ests) > 0)
    assert np.all(ests >= 1.0)


def test_fits_L2_flips_at_knee():
    f = _f([_row(2048, 2048, 2048, 64, 128, 64, 4, 1, 1), _row(4096, 4096, 4096, 64, 128, 64, 4, 1, 1)]).sort_values(
        "M"
    )
    assert f.fits_L2.tolist() == [1, 0]  # 2048 resident, 4096 spills


def test_streamk_and_cluster_flags():
    f = _f(
        [
            _row(
                4096,
                4096,
                4096,
                128,
                128,
                64,
                4,
                1,
                1,
                sched="KernelTmaWarpSpecializedCooperative",
                scheduler="StreamKScheduler",
            )
        ]
    )
    assert f.is_streamk.iloc[0] == 1
    assert f.sched_class.iloc[0] == "cooperative"
    # 1x1 cluster on a big shape never overshoots the grid.
    assert f.cluster_overshoots.iloc[0] == 0


def test_all_numeric_features_present_and_finite():
    f = _f([_row(4096, 4096, 4096, 64, 128, 64, 4, 1, 1)])
    for col in NUMERIC_FEATURES:
        assert col in f.columns, f"missing feature {col}"
    vals = f[NUMERIC_FEATURES].to_numpy(dtype="float64")
    assert np.isfinite(vals).all()


def test_fp32_and_fp8_features_finite():
    """Operand byte widths and peak-TC scaling must be defined for all sweep dtypes."""
    bf16 = _f([_row(4096, 4096, 4096, 64, 128, 64, 4, 1, 1, dt="cutlass::bfloat16_t")])
    fp32 = _f([_row(4096, 4096, 4096, 64, 128, 64, 4, 1, 1, dt="float")])
    fp8 = _f([_row(4096, 4096, 4096, 64, 128, 64, 4, 1, 1, dt="cutlass::float_e4m3_t")])
    for label, frame in (("bf16", bf16), ("fp32", fp32), ("fp8", fp8)):
        vals = frame[NUMERIC_FEATURES].to_numpy(dtype="float64")
        assert np.isfinite(vals).all(), label


def test_smem_scales_with_operand_bytes():
    bf16 = _f([_row(4096, 4096, 4096, 64, 128, 64, 4, 1, 1, dt="cutlass::bfloat16_t")]).smem_total.iloc[0]
    fp32 = _f([_row(4096, 4096, 4096, 64, 128, 64, 4, 1, 1, dt="float")]).smem_total.iloc[0]
    fp8 = _f([_row(4096, 4096, 4096, 64, 128, 64, 4, 1, 1, dt="cutlass::float_e4m3_t")]).smem_total.iloc[0]
    assert fp32 == pytest.approx(2 * bf16)
    assert fp8 == pytest.approx(0.5 * bf16)


def test_problem_arith_intensity_scales_with_operand_bytes():
    bf16 = _f([_row(4096, 4096, 4096, 64, 128, 64, 4, 1, 1, dt="cutlass::bfloat16_t")]).problem_arith_intensity.iloc[0]
    fp32 = _f([_row(4096, 4096, 4096, 64, 128, 64, 4, 1, 1, dt="float")]).problem_arith_intensity.iloc[0]
    fp8 = _f([_row(4096, 4096, 4096, 64, 128, 64, 4, 1, 1, dt="cutlass::float_e4m3_t")]).problem_arith_intensity.iloc[0]
    assert fp8 > bf16 > fp32
    assert bf16 == pytest.approx(2 * fp32)
    assert fp8 == pytest.approx(4 * fp32)


def test_featurize_is_label_free():
    # featurize must run without any measured/label column present.
    row = _row(4096, 4096, 4096, 64, 128, 64, 4, 1, 1)
    for k in ("mean_tflops", "std_tflops"):
        row.pop(k)
    f = featurize(pd.DataFrame([row]))
    assert set(NUMERIC_FEATURES).issubset(f.columns)


def test_tie_collapsed_grades():
    g = (
        add_labels(
            _f(
                [
                    _row(4096, 4096, 4096, 64, 128, 64, 4, 1, 1, tflops=500.0, std=2.0),
                    _row(4096, 4096, 4096, 64, 128, 64, 3, 1, 1, tflops=499.0, std=2.0),  # tie with top
                    _row(4096, 4096, 4096, 64, 64, 64, 4, 1, 1, tflops=480.0, std=2.0),  # new band
                    _row(4096, 4096, 4096, 64, 64, 256, 4, 1, 1, tflops=40.0, std=1.0),  # far down
                ]
            )
        )
        .sort_values("mean_tflops", ascending=False)
        .reset_index(drop=True)
    )
    assert g.relevance_grade.iloc[0] == 31
    assert g.relevance_grade.iloc[0] == g.relevance_grade.iloc[1]  # 500/499 tie
    assert g.relevance_grade.iloc[2] < g.relevance_grade.iloc[1]  # 480 steps down
    assert (g.y_norm <= 1.0).all() and g.y_norm.iloc[0] == pytest.approx(1.0)
