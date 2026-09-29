"""Offline unit tests for the shared evaluation harness (src/model/harness.py)."""

import numpy as np
import pandas as pd
import pytest
from harness import (
    base_shape_id,
    baselines,
    make_relevance,
    ndcg_at_k,
    ndcg_per_group,
    regime,
    select_and_regret,
    summarize,
)


def test_regime_matches_generator():
    assert regime(8192, 8192, 8192) == "large"
    assert regime(8192, 1024, 1024) == "tall"  # M/N = 8 >= 4
    assert regime(1024, 8192, 1024) == "wide"  # N/M = 8 >= 4
    assert regime(2048, 2048, 256) == "skinny-K"  # 256 <= 2048/4
    assert regime(2048, 2048, 2048) == "square"


def test_make_relevance_top_focused():
    # Default REL_TOP_BANDS=2 (tuned, Optuna trial #202): top band -> 2, then 1, rest 0.
    g = np.array([31, 30, 29, 28, 27, 10, 0])
    np.testing.assert_array_equal(make_relevance(g, 31), [2, 1, 0, 0, 0, 0, 0])


def _eval_df(groups):
    """groups: list of (M,N,K, [y_norms]). best_tflops fixed at 500."""
    rows = []
    for gid, (M, N, K, ynorms) in enumerate(groups):
        for r, yn in enumerate(sorted(ynorms, reverse=True), 1):
            rows.append(
                dict(
                    group_id=gid,
                    M=M,
                    N=N,
                    K=K,
                    name=f"cfg{r}",
                    mean_tflops=500.0 * yn,
                    group_best_tflops=500.0,
                    y_norm=yn,
                    rank_in_group=r,
                )
            )
    return pd.DataFrame(rows)


def test_select_and_regret_picks_argmax():
    df = _eval_df([(2048, 2048, 2048, [1.0, 0.97, 0.90, 0.40]), (4096, 1024, 1024, [1.0, 0.95, 0.80, 0.30])])
    perg = select_and_regret(df, df["y_norm"].to_numpy())  # perfect scorer
    assert (perg["regret"] == 0.0).all()
    assert perg["top1"].sum() == 2 and perg["within5pct"].sum() == 2
    assert set(perg["regime"]) == {"square", "tall"}


def test_select_and_regret_bad_scorer():
    # 6 configs/group so the worst is genuinely outside the top 5.
    df = _eval_df(
        [(2048, 2048, 2048, [1.0, 0.9, 0.8, 0.7, 0.6, 0.4]), (4096, 1024, 1024, [1.0, 0.9, 0.8, 0.7, 0.6, 0.4])]
    )
    perg = select_and_regret(df, (-df["y_norm"]).to_numpy())  # picks the worst
    assert perg["regret"].to_numpy() == pytest.approx([0.6, 0.6])
    assert perg["top1"].sum() == 0 and perg["top5"].sum() == 0


def test_baselines_random_pick():
    df = _eval_df(
        [
            (2048, 2048, 2048, [1.0, 0.97, 0.90, 0.40]),  # mean 0.8175
            (4096, 1024, 1024, [1.0, 0.50, 0.40, 0.30]),
        ]
    )  # mean 0.55
    b = baselines(df)
    assert b["random_pick"]["regret_mean"] == pytest.approx(1 - np.mean([0.8175, 0.55]), abs=1e-6)
    # static_best_config: cfg1 (y_norm 1.0 in both) is the oracle everywhere -> regret 0
    assert b["static_best_config"]["regret_mean"] == pytest.approx(0.0)


def test_random_pick_sampled_and_expected_per_group():
    from harness import random_pick_expected_per_group, random_pick_sampled_per_group

    df = _eval_df([(2048, 2048, 2048, [1.0, 0.80, 0.60, 0.40])])
    expected = random_pick_expected_per_group(df)
    assert len(expected) == 1
    assert expected["regret"].iloc[0] == pytest.approx(0.30, abs=1e-6)
    sampled = random_pick_sampled_per_group(df, seed=0)
    assert len(sampled) == 1
    assert sampled["regret"].iloc[0] in {0.0, 0.2, 0.4, 0.6}


def test_base_shape_id_groups_layouts_together():
    df = pd.DataFrame({"M": [2048, 2048, 4096], "N": [2048, 2048, 4096], "K": [2048, 2048, 4096]})
    ids = base_shape_id(df)
    assert ids[0] == ids[1] and ids[0] != ids[2]


def test_ndcg_perfect_and_reversed_ranking():
    gains = np.array([1.0, 0.9, 0.8, 0.5])
    assert ndcg_at_k(gains, gains, 4) == pytest.approx(1.0)
    worst = ndcg_at_k(gains, -gains, 4)
    assert 0.0 < worst < 1.0


def test_ndcg_at_1_equals_one_minus_regret():
    """Linear y_norm gain makes NDCG@1 exactly the selected config's y_norm."""
    df = _eval_df([(2048, 2048, 2048, [1.0, 0.9, 0.8, 0.4]), (4096, 1024, 1024, [1.0, 0.7, 0.6, 0.3])])
    scores = (-df["y_norm"]).to_numpy()  # picks the worst in each group
    perg = select_and_regret(df, scores)
    np.testing.assert_allclose(perg["ndcg@1"].to_numpy(), 1.0 - perg["regret"].to_numpy())


def test_ndcg_k_larger_than_group_is_clamped():
    gains = np.array([1.0, 0.5])
    assert ndcg_at_k(gains, gains, 10) == pytest.approx(ndcg_at_k(gains, gains, 2))


def test_ndcg_all_zero_gains_is_zero():
    gains = np.zeros(4)
    assert ndcg_at_k(gains, np.arange(4.0), 4) == 0.0


def test_ndcg_per_group_index_and_columns():
    df = _eval_df([(2048, 2048, 2048, [1.0, 0.9]), (4096, 1024, 1024, [1.0, 0.5])])
    nd = ndcg_per_group(df, df["y_norm"].to_numpy(), ks=(1, 5))
    assert list(nd.columns) == ["ndcg@1", "ndcg@5"]
    assert set(nd.index) == {0, 1}
    np.testing.assert_allclose(nd.to_numpy(), 1.0)


def test_ndcg_columns_align_with_their_group():
    """A model that is perfect on one group and inverted on the other must not swap rows."""
    df = _eval_df([(2048, 2048, 2048, [1.0, 0.9, 0.2]), (4096, 1024, 1024, [1.0, 0.9, 0.2])])
    y = df["y_norm"].to_numpy()
    scores = np.where(df["group_id"].to_numpy() == 0, y, -y)
    perg = select_and_regret(df, scores).set_index("group_id")
    assert perg.loc[0, "ndcg@1"] == pytest.approx(1.0)
    assert perg.loc[1, "ndcg@1"] == pytest.approx(0.2)


def test_summarize_reports_new_bands():
    df = _eval_df([(2048, 2048, 2048, [1.0, 0.995, 0.93, 0.5])])
    perg = select_and_regret(df, np.array([0.0, 1.0, 0.0, 0.0]))  # picks y_norm=0.995
    s = summarize(perg, "t")
    assert s["within1pct"] == 1.0
    assert s["within5pct"] == 1.0 and s["within10pct"] == 1.0
    assert "regret_p90" in s and "ndcg@10" in s


def _train_df(n_shapes=40, layouts=("NN", "NT", "TN", "TT")):
    """n_shapes base shapes spanning several regimes, 4 layouts each."""
    rows = []
    for i in range(n_shapes):
        M, N, K = (
            [(2048, 2048, 2048), (8192, 1024, 1024), (1024, 8192, 1024),
             (2048, 2048, 256), (8192, 8192, 8192)][i % 5]
        )
        M += i  # distinct shapes, same regime family
        for lay in layouts:
            rows.append(dict(M=M, N=N, K=K, layout=lay, split="train", y_norm=0.5))
    return pd.DataFrame(rows)


def test_split_keeps_layouts_of_a_shape_together():
    from harness import grouped_train_val_split
    df = _train_df()
    tr, va = grouped_train_val_split(df, val_frac=0.25, seed=0)
    tr_shapes = set(map(tuple, tr[["M", "N", "K"]].drop_duplicates().to_numpy()))
    va_shapes = set(map(tuple, va[["M", "N", "K"]].drop_duplicates().to_numpy()))
    assert not (tr_shapes & va_shapes)
    assert tr_shapes | va_shapes == set(map(tuple, df[["M", "N", "K"]].drop_duplicates().to_numpy()))
    # every validation shape brought all four of its layouts
    assert (va.groupby(["M", "N", "K"]).size() == 4).all()
    assert len(tr) + len(va) == len(df)


def test_split_is_deterministic_and_seed_sensitive():
    from harness import grouped_train_val_split
    df = _train_df()
    a = grouped_train_val_split(df, seed=1)[1]
    b = grouped_train_val_split(df, seed=1)[1]
    c = grouped_train_val_split(df, seed=2)[1]
    pd.testing.assert_frame_equal(a, b)
    assert not a.equals(c)


def test_split_stratifies_by_regime():
    from harness import grouped_train_val_split
    df = _train_df(n_shapes=50)
    _, va = grouped_train_val_split(df, val_frac=0.2, seed=0, stratify=True)
    va_regimes = {regime(m, n, k) for m, n, k in va[["M", "N", "K"]].drop_duplicates().to_numpy()}
    all_regimes = {regime(m, n, k) for m, n, k in df[["M", "N", "K"]].drop_duplicates().to_numpy()}
    assert va_regimes == all_regimes


def test_split_rejects_eval_rows():
    from harness import grouped_train_val_split
    df = _train_df()
    df.loc[df.index[:4], "split"] = "eval"
    with pytest.raises(ValueError, match="training rows only"):
        grouped_train_val_split(df)


def test_split_rejects_bad_fraction():
    from harness import grouped_train_val_split
    with pytest.raises(ValueError, match="val_frac"):
        grouped_train_val_split(_train_df(), val_frac=0.0)


def test_split_never_empties_a_side():
    """A tiny val_frac still yields one shape per stratum, never an empty side."""
    from harness import grouped_train_val_split
    df = _train_df(n_shapes=10)  # 2 shapes per regime
    tr, va = grouped_train_val_split(df, val_frac=0.01, seed=0)
    assert len(tr) and len(va)
    assert va.groupby(["M", "N", "K"]).ngroups == 5  # one per regime


def test_split_raises_when_every_stratum_is_a_singleton():
    from harness import grouped_train_val_split
    df = _train_df(n_shapes=5)  # 1 shape per regime -> nothing can be spared
    with pytest.raises(ValueError, match="empty validation set"):
        grouped_train_val_split(df, val_frac=0.01, seed=0)
