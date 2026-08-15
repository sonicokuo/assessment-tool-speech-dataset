"""Unit tests for src/eval/faithfulness_metrics.py.

Pure numpy — runs without torch. Known-value cases target the exact naive-wrong
traps the audit flagged (CCC 1/n moments, CRPS 1/sqrt(pi) constant + sigma->0,
E-AURC oracle ordering, AUGRC generalized-risk identity, DM cluster-robust).
"""

import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from eval.faithfulness_metrics import (  # noqa: E402
    augrc,
    aurc,
    bias,
    bland_altman,
    ccc,
    crps_gaussian,
    digit_drift,
    dm_paired,
    e_aurc,
    holm_correction,
    norm_cdf,
    risk_coverage_curve,
    sigma_from_logvar,
    sigma_raw,
)


# ---- E3 bias --------------------------------------------------------------
def test_bias_sign_and_value():
    assert bias([2.0, 4.0], [1.0, 1.0]) == pytest.approx(2.0)   # over-reports
    assert bias([0.0, 0.0], [1.0, 3.0]) == pytest.approx(-2.0)  # under-reports
    assert bias([], []) is None
    # nan pairs dropped
    assert bias([1.0, float("nan"), 3.0], [0.0, 5.0, 1.0]) == pytest.approx(1.5)


# ---- E4 CCC ---------------------------------------------------------------
def test_ccc_perfect_agreement_is_one():
    x = [1.0, 2.0, 3.0, 4.0]
    assert ccc(x, x) == pytest.approx(1.0)


def test_ccc_penalizes_scale_shift_below_pearson():
    # y = 2x: Pearson=1 but CCC<1 because of the scale mismatch (SRCC's blind spot).
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y = 2.0 * x
    c = ccc(y.tolist(), x.tolist())
    assert c is not None and c < 1.0
    # explicit Lin value: 2*cov/(vx+vy+(my-mx)^2)
    mx, my = x.mean(), y.mean()
    vx = np.mean((x - mx) ** 2); vy = np.mean((y - my) ** 2)
    sxy = np.mean((x - mx) * (y - my))
    assert c == pytest.approx(2 * sxy / (vx + vy + (my - mx) ** 2))


def test_ccc_uses_1_over_n_not_np_cov():
    # If someone used np.cov (ddof=1) for sxy but 1/n for variances, CCC would drift.
    # Perfect-agreement must be exactly 1.0 regardless of n.
    for n in (2, 3, 7):
        x = list(range(n))
        assert ccc(x, x) == pytest.approx(1.0)


# ---- E5/E6/E7 risk-coverage family ---------------------------------------
def test_risk_coverage_orders_by_confidence():
    losses = [0.0, 1.0, 2.0]
    conf = [10.0, 5.0, 1.0]     # already best-first
    rc = risk_coverage_curve(losses, conf)
    assert list(rc["coverage"]) == pytest.approx([1 / 3, 2 / 3, 1.0])
    assert list(rc["selective_risk"]) == pytest.approx([0.0, 0.5, 1.0])


def test_aurc_prefers_good_confidence():
    losses = [0.0, 0.0, 3.0, 3.0]
    good = [4.0, 3.0, 2.0, 1.0]     # confident where loss is low
    bad = [1.0, 2.0, 3.0, 4.0]      # confident where loss is high
    assert aurc(losses, good) < aurc(losses, bad)


def test_e_aurc_nonneg_and_zero_on_oracle():
    losses = [0.0, 1.0, 4.0, 9.0]
    # oracle ordering => confidence = -loss => E-AURC exactly 0
    assert e_aurc(losses, [-l for l in losses]) == pytest.approx(0.0, abs=1e-12)
    # any ordering >= 0
    assert e_aurc(losses, [1.0, 2.0, 3.0, 4.0]) >= -1e-12


def test_augrc_equals_generalized_risk_identity():
    rng = np.random.default_rng(0)
    losses = rng.random(20)
    conf = rng.random(20)
    # AUGRC == (1/n) sum_k (k/n) * R(k)
    rc = risk_coverage_curve(losses, conf)
    k = np.arange(1, rc["n"] + 1)
    manual = np.mean((k / rc["n"]) * rc["selective_risk"])
    assert augrc(losses, conf) == pytest.approx(manual)


# ---- E9 CRPS --------------------------------------------------------------
def test_crps_constant_is_one_over_sqrt_pi():
    # z=0, sigma=1: CRPS = 2*phi(0) - 1/sqrt(pi) = sqrt(2/pi) - 1/sqrt(pi)
    expected = math.sqrt(2.0 / math.pi) - 1.0 / math.sqrt(math.pi)
    assert crps_gaussian(0.0, 1.0, 0.0) == pytest.approx(expected)
    assert expected == pytest.approx(0.23369, abs=1e-4)   # NOT the 1/(2 sqrt pi) value


def test_crps_degenerates_to_abs_error_as_sigma_to_zero():
    assert crps_gaussian(5.0, 1e-9, 7.0) == pytest.approx(2.0, abs=1e-6)
    assert crps_gaussian(5.0, 0.0, 7.0) == pytest.approx(2.0)


def test_crps_lower_for_calibrated_sigma():
    # a confident-correct prediction should beat an over-hedged one and an over-confident-wrong one
    correct_sharp = crps_gaussian(5.0, 0.3, 5.0)
    over_hedged = crps_gaussian(5.0, 5.0, 5.0)
    conf_wrong = crps_gaussian(5.0, 0.3, 9.0)
    assert correct_sharp < over_hedged
    assert correct_sharp < conf_wrong


# ---- E11 Diebold-Mariano --------------------------------------------------
def test_dm_zero_when_identical():
    r = dm_paired([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    assert r["dbar"] == pytest.approx(0.0)
    assert r["z"] is None or r["z"] == pytest.approx(0.0)


def test_dm_negative_z_when_a_better():
    a = [0.1, 0.2, 0.15, 0.25, 0.1]   # A lower loss
    b = [0.9, 1.0, 0.8, 1.1, 0.95]
    r = dm_paired(a, b)
    assert r["dbar"] < 0 and r["z"] < 0 and r["p"] < 0.05


def test_dm_cluster_robust_widens_variance():
    # two speakers, each with correlated differentials -> clustered SE > iid SE
    a = [0.1, 0.1, 0.1, 0.9, 0.9, 0.9]
    b = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
    spk = ["s1", "s1", "s1", "s2", "s2", "s2"]
    iid = dm_paired(a, b)
    clustered = dm_paired(a, b, clusters=spk)
    # clustered variance is larger => |z| smaller (more conservative)
    assert abs(clustered["z"]) < abs(iid["z"])


def test_holm_correction_monotone_and_fwer():
    pv = {"f1": 0.001, "f2": 0.02, "f3": 0.6}
    out = holm_correction(pv, alpha=0.05)
    # smallest p * m; monotone non-decreasing adjusted p
    assert out["f1"]["p_adj"] == pytest.approx(0.003)
    assert out["f1"]["reject"] and not out["f3"]["reject"]
    assert out["f2"]["p_adj"] >= out["f1"]["p_adj"]


# ---- E12 digit-drift ------------------------------------------------------
def test_digit_drift():
    assert digit_drift([30.5, 8.0], [30.0, 9.0]) == pytest.approx(0.75)
    assert digit_drift([30.5], [30.0], scale=0.5) == pytest.approx(1.0)


# ---- E10 Bland-Altman -----------------------------------------------------
def test_bland_altman_loa():
    p = [1.1, 2.2, 2.9, 4.1]
    g = [1.0, 2.0, 3.0, 4.0]
    ba = bland_altman(p, g)
    assert ba["n"] == 4
    assert ba["mean_diff"] == pytest.approx(np.mean(np.array(p) - np.array(g)))
    assert ba["loa_low"] < ba["mean_diff"] < ba["loa_high"]


# ---- sigma-units ----------------------------------------------------------
def test_sigma_units_convention():
    # log_var=0 => normalized sigma=1; raw sigma = scale
    assert sigma_from_logvar(0.0) == pytest.approx(1.0)
    assert sigma_raw(0.0, 15.0) == pytest.approx(15.0)
    assert sigma_raw(math.log(4.0), 2.0) == pytest.approx(4.0)   # exp(0.5*ln4)=2, *2=4


def test_norm_cdf_sanity():
    assert norm_cdf(0.0) == pytest.approx(0.5)
    assert norm_cdf(1.96) == pytest.approx(0.975, abs=1e-3)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
