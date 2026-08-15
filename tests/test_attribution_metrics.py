"""Regression tests for the corrected attribution metrics.

The headline test is `test_antialigned_map_is_not_penalised`: it constructs a map that is
PERFECTLY clip-specific but ANTI-aligned with overlap -- the physically correct behaviour
for a quantity that is unrecoverable under two-talker overlap -- and asserts the corrected
statistic scores it POSITIVE. The old one-sided statistic scored exactly this case
NEGATIVE, which is what produced the false "maps are not clip-specific" verdict.
"""
import numpy as np
import pytest

from src.eval.attribution_metrics import (
    auc,
    iid_entropy_null,
    normalised_entropy,
    score_all,
    score_feature,
)


def _clips(n=40, T=48, seed=0):
    """Synthetic clips, each with its OWN random contiguous overlap span."""
    rng = np.random.default_rng(seed)
    ov = np.zeros((n, T), np.float32)
    for i in range(n):
        s = rng.integers(0, T // 2)
        e = s + rng.integers(T // 6, T // 2)
        ov[i, s:min(e, T)] = 1.0
    return ov


# --------------------------------------------------------------------- auc


def test_auc_perfect_and_chance():
    lab = np.array([0, 0, 1, 1], float)
    assert auc(np.array([0.0, 0.1, 0.9, 1.0]), lab) == pytest.approx(1.0)
    assert auc(np.array([1.0, 0.9, 0.1, 0.0]), lab) == pytest.approx(0.0)


def test_auc_constant_map_is_exactly_chance():
    """Ties must be rank-averaged, or a flat map scores 0 or 1 instead of 0.5."""
    assert auc(np.ones(8), np.array([0, 0, 0, 0, 1, 1, 1, 1], float)) == pytest.approx(0.5)


def test_auc_degenerate_labels_are_nan():
    # clean clips have an all-zero overlap mask -> AUC undefined, must NOT be silently 0.5
    assert np.isnan(auc(np.arange(8.0), np.zeros(8)))


# ------------------------------------------------------------------ entropy


def test_entropy_flat_is_one_spike_is_low():
    assert normalised_entropy(np.ones(32)) == pytest.approx(1.0)
    spike = np.zeros(32)
    spike[3] = 1.0
    assert normalised_entropy(spike) < 0.05


def test_iid_null_is_well_below_one():
    """An UNSTRUCTURED map does not score 1.0. Comparing raw entropy against 1.0 is what
    made every real map look 'flat'."""
    null = iid_entropy_null(29)
    assert 0.85 < null < 0.96


# ------------------------------------------- the sign bug (headline regression)


def test_antialigned_map_is_not_penalised():
    ov = _clips()
    # map = INVERSE of this clip's own overlap: perfectly clip-specific, anti-aligned.
    contrib = (1.0 - ov) + 1e-3
    r = score_feature(contrib, ov, n_others=16, n_perm=4, seed=1)

    assert r["spec_oneside_BUGGY"] < -0.1, "old statistic should penalise it (documents the bug)"
    assert r["spec_twoside"] > 0.1, "corrected statistic must reward clip-specificity"
    assert r["spec_twoside"] > r["spec_perm_null"] + 0.05, "must clear the permutation null"
    assert r["dev_corr"] < -0.5, "signed correlation must expose the anti-alignment"


def test_aligned_map_also_scores_positive():
    """Both polarities are clip-specific; the corrected statistic must be blind to sign."""
    ov = _clips()
    r = score_feature(ov + 1e-3, ov, n_others=16, n_perm=4, seed=2)
    assert r["spec_twoside"] > 0.1
    assert r["dev_corr"] > 0.5


def test_population_average_map_scores_near_zero():
    """The actual failure mode we were testing for: every clip emitting the SAME map."""
    ov = _clips()
    avg = np.tile(ov.mean(axis=0), (ov.shape[0], 1)) + 1e-3
    r = score_feature(avg, ov, n_others=16, n_perm=4, seed=3)
    assert abs(r["spec_twoside"]) < 0.06, "a population-average map must NOT look specific"


def test_random_map_scores_near_zero():
    ov = _clips()
    rng = np.random.default_rng(7)
    r = score_feature(rng.random(ov.shape).astype(np.float32), ov,
                      n_others=16, n_perm=4, seed=4)
    assert abs(r["spec_twoside"]) < 0.06


# ------------------------------------------------------------ bookkeeping


def test_clean_clips_counted_not_silently_dropped():
    """400 captured clips were really ~200 scored: the all-zero-mask clean twins are
    undefined for AUC. That must show up as n_degenerate, not vanish."""
    ov = _clips(n=20)
    ov = np.concatenate([ov, np.zeros_like(ov)])  # 20 mixtures + 20 clean
    r = score_feature(np.abs(np.random.default_rng(0).standard_normal(ov.shape)).astype(np.float32),
                      ov, n_others=8, n_perm=2, seed=5)
    assert r["n_scored"] == 20
    assert r["n_degenerate"] == 20


def test_oracle_row_is_the_ceiling():
    """Pairwise masks overlap heavily, so a PERFECT map scores ~0.25, not 1.0. Reporting
    specificity without this ceiling makes a good map look mediocre."""
    ov = _clips()
    out = score_all(np.stack([ov, 1.0 - ov], axis=1), ov, ["aligned", "anti"],
                    n_others=16, n_perm=4, seed=6)
    ceil = out["__ORACLE__"]["spec_twoside"]
    assert 0.1 < ceil < 0.6, f"oracle ceiling {ceil} should be well below 1.0"
    for f in ("aligned", "anti"):
        assert out[f]["spec_twoside"] <= ceil + 1e-6
