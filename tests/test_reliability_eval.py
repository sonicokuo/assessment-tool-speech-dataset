"""Tests for the risk-coverage / AURC reliability eval (deliverable requirement (b)).

The thesis-proof artifact: when the per-feature uncertainty (σ) correlates with error,
abstaining on highest-σ first must RAISE precision on the answered subset, so the
model's AURC beats random abstention. Also covers the degenerate / edge cases.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from reliability_eval import (  # noqa: E402
    risk_coverage_curve,
    aurc,
    random_abstention_curve,
    risk_coverage_report,
)


def test_perfect_detector_has_low_risk_until_forced_to_answer_wrong():
    """If uncertainty perfectly ranks errors last, the answered prefix is error-free
    until the wrong items are forced in: risk stays 0 over the correct prefix."""
    # 6 correct (low unc) then 4 wrong (high unc).
    unc = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    correct = np.array([1, 1, 1, 1, 1, 1, 0, 0, 0, 0], dtype=float)
    cov, risk = risk_coverage_curve(unc, correct)
    # Over the first 6 (all correct) risk is exactly 0.
    assert np.allclose(risk[:6], 0.0)
    # At full coverage risk = base error rate = 0.4.
    assert risk[-1] == pytest.approx(0.4)
    assert cov[-1] == pytest.approx(1.0)


def test_high_sigma_correlates_with_error_beats_random():
    """The headline property: σ correlated with error → AURC_model < AURC_random."""
    rng = np.random.default_rng(7)
    n = 500
    unc = rng.random(n)
    # higher uncertainty → more likely wrong.
    correct = (rng.random(n) > unc).astype(float)
    rep = risk_coverage_report(unc, correct)
    assert rep["aurc_model"] < rep["aurc_random"], "uncertainty ordering should beat random"
    assert rep["aurc_gain_vs_random"] > 0.0
    assert rep["n"] == n
    # always-answer risk equals the base error rate.
    assert rep["always_answer_risk"] == pytest.approx(1.0 - correct.mean())


def test_uninformative_uncertainty_matches_random():
    """If uncertainty is independent of correctness, the model AURC should be close to
    the random-abstention AURC (no systematic gain)."""
    rng = np.random.default_rng(3)
    n = 4000
    correct = (rng.random(n) > 0.4).astype(float)  # ~60% correct
    unc = rng.random(n)  # independent of correctness
    rep = risk_coverage_report(unc, correct)
    # Gain should be small in magnitude (no real signal). Tolerance accounts for the
    # noisy small-coverage tail of the curve.
    assert abs(rep["aurc_gain_vs_random"]) < 0.05


def test_random_abstention_curve_is_flat_at_base_error_rate():
    correct = np.array([1, 0, 1, 1, 0, 1, 0, 1], dtype=float)
    cov, risk = random_abstention_curve(correct)
    base = 1.0 - correct.mean()
    assert np.allclose(risk, base)
    assert cov[-1] == pytest.approx(1.0)
    # AURC of a flat curve over coverage in [1/n, 1] equals base * (1 - 1/n).
    expected_area = base * (cov[-1] - cov[0])
    assert aurc(cov, risk) == pytest.approx(expected_area, rel=1e-6)


def test_aurc_single_point_is_zero():
    cov, risk = risk_coverage_curve([0.5], [1.0])
    assert aurc(cov, risk) == 0.0


def test_anti_correlated_uncertainty_is_worse_than_random():
    """If uncertainty is INVERSELY related to error (the worst detector), abstaining by
    σ removes the CORRECT items first and AURC is worse than random."""
    # low unc on the wrong items, high unc on the correct items.
    unc = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9])
    correct = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=float)
    rep = risk_coverage_report(unc, correct)
    assert rep["aurc_model"] > rep["aurc_random"]
    assert rep["aurc_gain_vs_random"] < 0.0


def test_empty_input_raises():
    with pytest.raises(ValueError):
        risk_coverage_curve([], [])
    with pytest.raises(ValueError):
        risk_coverage_report([], [])


def test_mismatched_lengths_raise():
    with pytest.raises(ValueError):
        risk_coverage_curve([0.1, 0.2], [1.0])


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
