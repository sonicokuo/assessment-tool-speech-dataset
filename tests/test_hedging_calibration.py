"""Tests for src/hedging_calibration.py — pure, no torch."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from hedging_calibration import (  # noqa: E402
    extract_clip_signals,
    reliability_by_bin,
    risk_coverage_curve,
    selective_improvement,
    is_monotone_nonincreasing,
)


def _entry(ov, f0_correct, f0_err=1.0):
    return {
        "filename": "x.wav",
        "per_feature": [
            {"feature": "overlap_ratio", "actual": ov, "correct": True},
            {"feature": "f0_mean", "actual": 150.0, "error": f0_err, "correct": f0_correct},
        ],
    }


# ── extract ──────────────────────────────────────────────────────────────────
def test_extract_signals_basic():
    s = extract_clip_signals(_entry(0.3, True, 2.0))
    assert s["overlap_ratio"] == 0.3 and s["correct"] is True and s["error"] == 2.0


def test_extract_none_when_feature_missing():
    e = {"per_feature": [{"feature": "snr", "actual": 10.0, "correct": True}]}
    assert extract_clip_signals(e) is None              # no overlap_ratio, no f0_mean


# ── reliability_by_bin: accuracy falls as overlap rises ──────────────────────
def _synthetic_signals():
    # 40 clips: low-overlap clips mostly correct, high-overlap clips mostly wrong
    sigs = []
    for i in range(20):
        sigs.append({"filename": f"lo{i}", "overlap_ratio": 0.1, "error": 2.0,
                     "correct": i < 18})          # 18/20 correct at low overlap
    for i in range(20):
        sigs.append({"filename": f"hi{i}", "overlap_ratio": 0.9, "error": 40.0,
                     "correct": i < 4})           # 4/20 correct at high overlap
    return sigs


def test_reliability_by_bin_monotone():
    rows = reliability_by_bin(_synthetic_signals(), [0.0, 0.5, 1.01])
    assert rows[0]["n"] == 20 and rows[1]["n"] == 20
    assert rows[0]["accuracy"] > rows[1]["accuracy"]    # accuracy falls with overlap
    assert is_monotone_nonincreasing(rows, "accuracy")


def test_reliability_empty_bin_safe():
    rows = reliability_by_bin([], [0.0, 0.5, 1.0])
    assert all(r["n"] == 0 and r["accuracy"] == 0.0 for r in rows)


# ── risk-coverage: precision rises as we abstain on high-overlap ─────────────
def test_risk_coverage_precision_drops_with_coverage():
    curve = risk_coverage_curve(_synthetic_signals(), steps=10)
    assert curve[0]["coverage"] == 0.1 and curve[-1]["coverage"] == 1.0
    # lowest-coverage (only the least-overlap clips asserted) precision must beat
    # full-coverage precision — the calibrated-hedge signature.
    assert curve[0]["precision"] > curve[-1]["precision"]
    assert selective_improvement(curve) > 0.0


def test_risk_coverage_empty():
    assert risk_coverage_curve([], steps=5) == []
    assert selective_improvement([]) == 0.0


def test_uncalibrated_signal_no_improvement():
    # correctness independent of overlap → abstaining buys ~nothing.
    sigs = [{"filename": str(i), "overlap_ratio": (i % 10) / 10.0,
             "error": 1.0, "correct": (i % 2 == 0)} for i in range(40)]
    curve = risk_coverage_curve(sigs, steps=10)
    assert abs(selective_improvement(curve)) <= 0.5     # not a strong improvement


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
