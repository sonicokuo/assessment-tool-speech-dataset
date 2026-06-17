"""Tests for src/f0_clean.py — well-posed F0 from non-overlapped voiced frames.

Pure numpy (no Praat), runs anywhere.
"""
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from f0_clean import (  # noqa: E402
    parse_overlap_windows_samples,
    f0_stats_voiced_nonoverlap,
)


# ── window parsing ───────────────────────────────────────────────────────────
def test_parse_windows_samples_to_seconds():
    # 16000-1.0s, 32000-2.0s at sr=16000
    w = parse_overlap_windows_samples("16000-32000;48000-64000", sr=16000)
    assert w == [(1.0, 2.0), (3.0, 4.0)]


def test_parse_windows_empty_and_malformed():
    assert parse_overlap_windows_samples("") == []
    assert parse_overlap_windows_samples("garbage;1-x;5-3") == []   # 5-3 not increasing


# ── core masking ─────────────────────────────────────────────────────────────
def test_no_overlap_uses_all_voiced():
    f0 = [150.0, 152.0, 0.0, 148.0]            # one unvoiced (0)
    t = [0.0, 0.1, 0.2, 0.3]
    r = f0_stats_voiced_nonoverlap(f0, t, [])
    assert r["n_clean_voiced"] == 3
    assert abs(r["f0_mean_hz"] - 150.0) < 0.5
    assert r["clean_voiced_frac"] == 1.0


def test_unvoiced_frames_excluded():
    f0 = [0.0, 0.0, 200.0, 200.0]
    t = [0.0, 0.1, 0.2, 0.3]
    r = f0_stats_voiced_nonoverlap(f0, t, [])
    assert r["n_clean_voiced"] == 2 and abs(r["f0_mean_hz"] - 200.0) < 1e-6


def test_overlap_window_is_half_open():
    f0 = [150.0, 150.0, 150.0, 150.0]
    t = [1.0, 1.5, 2.0, 2.5]
    # window [1.0, 2.0): excludes t=1.0 and t=1.5, KEEPS t=2.0 (end is exclusive)
    r = f0_stats_voiced_nonoverlap(f0, t, [(1.0, 2.0)])
    assert r["n_clean_voiced"] == 2   # t=2.0 and t=2.5


def test_excluding_overlap_changes_estimate():
    # THE point: overlap frames carry octave-error pitch (~2x). Excluding them
    # moves the mean from a skewed mixture value back to the true ~150 Hz.
    f0 = [150.0, 150.0, 150.0, 300.0, 300.0, 300.0]   # last 3 are overlapped octave errors
    t = [0.0, 0.1, 0.2, 1.0, 1.1, 1.2]
    full = f0_stats_voiced_nonoverlap(f0, t, [])
    clean = f0_stats_voiced_nonoverlap(f0, t, [(0.95, 1.25)])
    assert abs(full["f0_mean_hz"] - 225.0) < 1.0     # mixture mean skewed high
    assert abs(clean["f0_mean_hz"] - 150.0) < 1e-6   # clean mean recovered
    assert clean["n_clean_voiced"] == 3


def test_too_few_clean_frames_returns_nan():
    f0 = [150.0, 150.0, 150.0]
    t = [0.0, 0.1, 0.2]
    r = f0_stats_voiced_nonoverlap(f0, t, [(-1.0, 10.0)])   # all excluded
    assert math.isnan(r["f0_mean_hz"]) and math.isnan(r["f0_sd_hz"])
    assert r["n_clean_voiced"] == 0


def test_clean_voiced_frac_reported():
    f0 = [150.0, 150.0, 150.0, 150.0]
    t = [0.0, 0.1, 1.0, 1.1]
    r = f0_stats_voiced_nonoverlap(f0, t, [(0.95, 1.25)])   # excludes last 2 of 4 voiced
    assert r["n_clean_voiced"] == 2
    assert r["clean_voiced_frac"] == 0.5


def test_length_mismatch_raises():
    try:
        f0_stats_voiced_nonoverlap([1.0, 2.0], [0.0], [])
        assert False, "expected ValueError"
    except ValueError:
        pass


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
