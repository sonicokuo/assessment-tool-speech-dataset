"""Tests for src/grounding_metrics.py — pure numpy."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from grounding_metrics import (  # noqa: E402
    time_mass,
    freq_mass,
    time_concentration_ratio,
    freq_band_concentration_ratio,
)

NF = 8


def _grid(time_profile, freq_profile):
    """Build a flat (T_p*NF) map as the outer product of a time and freq profile."""
    t = np.asarray(time_profile, dtype=float)
    f = np.asarray(freq_profile, dtype=float)
    return np.outer(t, f).ravel().tolist()


# ── marginals ────────────────────────────────────────────────────────────────
def test_time_mass_normalized_and_localized():
    # all mass in time-bin 2 of 4
    flat = _grid([0, 0, 1, 0], [1] * NF)
    tm = time_mass(flat, NF)
    assert abs(tm.sum() - 1.0) < 1e-9
    assert tm[2] > 0.99


def test_freq_mass_localized_low_band():
    # all mass in freq rows 0-1 (low band)
    fp = [1, 1, 0, 0, 0, 0, 0, 0]
    fm = freq_mass(_grid([1, 1, 1], fp), NF)
    assert abs(fm.sum() - 1.0) < 1e-9
    assert fm[0] + fm[1] > 0.99


# ── time concentration ───────────────────────────────────────────────────────
def test_time_concentration_grounded():
    # 10 time bins over 1.0 s; attention all in bins covering [0.4,0.6)
    tp = [0.0] * 10
    tp[4] = tp[5] = 1.0
    r = time_concentration_ratio(_grid(tp, [1] * NF), [(0.4, 0.6)], duration=1.0, n_freq=NF)
    assert r["ratio"] > 4.0          # mass ~1.0 in a 0.2-of-1.0 window → ratio ~5
    assert abs(r["frac_time"] - 0.2) < 1e-6


def test_time_concentration_uniform_is_about_one():
    r = time_concentration_ratio(_grid([1] * 10, [1] * NF), [(0.0, 0.5)],
                                 duration=1.0, n_freq=NF)
    assert 0.8 < r["ratio"] < 1.2    # uniform attention → ~chance


def test_time_concentration_avoids_region_below_one():
    tp = [1.0] * 10
    tp[0] = tp[1] = 0.0              # attention avoids the first 0.2 s
    r = time_concentration_ratio(_grid(tp, [1] * NF), [(0.0, 0.2)],
                                 duration=1.0, n_freq=NF)
    assert r["ratio"] < 0.5


def test_time_concentration_none_cases():
    assert time_concentration_ratio(_grid([1, 1], [1] * NF), [], 1.0, NF) is None
    assert time_concentration_ratio(_grid([1, 1], [1] * NF), [(0, 0.5)], 0.0, NF) is None
    # zero mass
    assert time_concentration_ratio([0.0] * (4 * NF), [(0, 0.5)], 1.0, NF) is None


# ── frequency-band concentration (pitch) ─────────────────────────────────────
def test_freq_band_concentration_pitch_in_low_band():
    # attention concentrated in low freq rows 0-1 → high ratio for that band
    fp = [1, 1, 0, 0, 0, 0, 0, 0]
    r = freq_band_concentration_ratio(_grid([1, 1, 1], fp), band_rows=[0, 1], n_freq=NF)
    assert r["ratio"] > 3.5          # mass ~1 in 2/8 band → ratio ~4
    assert abs(r["frac_band"] - 0.25) < 1e-9


def test_freq_band_uniform_is_about_one():
    r = freq_band_concentration_ratio(_grid([1, 1], [1] * NF), band_rows=[0, 1], n_freq=NF)
    assert 0.8 < r["ratio"] < 1.2


def test_freq_band_none_cases():
    assert freq_band_concentration_ratio([0.0] * (3 * NF), [0, 1], NF) is None
    assert freq_band_concentration_ratio(_grid([1, 1], [1] * NF), [], NF) is None
    # out-of-range band rows filtered out → None
    assert freq_band_concentration_ratio(_grid([1, 1], [1] * NF), [99, 100], NF) is None


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
