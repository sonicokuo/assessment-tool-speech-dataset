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
    iou_time,
    pointing_game,
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
    # genuinely-undefined inputs (no windows / no duration) still return None.
    assert time_concentration_ratio(_grid([1, 1], [1] * NF), [], 1.0, NF) is None
    assert time_concentration_ratio(_grid([1, 1], [1] * NF), [(0, 0.5)], 0.0, NF) is None


def test_time_concentration_zero_mass_is_uniform_not_none():
    # A collapsed / all-zero map (e.g. a fully-OFF bottleneck keep-mask) localizes
    # nothing. It must NOT return null: it is scored as the maximally uninformative
    # (uniform) map, so the concentration ratio is ~1.0 (no better than uniform).
    r = time_concentration_ratio([0.0] * (4 * NF), [(0, 0.5)], 1.0, NF)
    assert r is not None
    assert 0.8 < r["ratio"] < 1.2


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
    # empty band (no valid rows) is genuinely undefined → None.
    assert freq_band_concentration_ratio(_grid([1, 1], [1] * NF), [], NF) is None
    # out-of-range band rows filtered out → None
    assert freq_band_concentration_ratio(_grid([1, 1], [1] * NF), [99, 100], NF) is None


def test_freq_band_zero_mass_is_uniform_not_none():
    # A zero-mass map falls back to a uniform frequency distribution: a 2/8 band
    # then carries ~its fractional mass, so ratio ~1.0 (uninformative), not None.
    r = freq_band_concentration_ratio([0.0] * (3 * NF), [0, 1], NF)
    assert r is not None
    assert 0.8 < r["ratio"] < 1.2


# ── time-axis IoU (bottleneck keep-mask vs oracle spans) ─────────────────────
def test_iou_time_planted_band_is_one():
    # planted overlap mass in bins 4-5 of 10 over 1.0 s → window [0.4,0.6)
    tp = [0.0] * 10
    tp[4] = tp[5] = 1.0
    r = iou_time(_grid(tp, [1] * NF), [(0.4, 0.6)], duration=1.0, n_freq=NF, thresh=0.0)
    assert r is not None and r["iou"] > 0.99
    assert r["n_pred"] == 2 and r["n_gt"] == 2


def test_iou_time_disjoint_is_zero():
    tp = [0.0] * 10
    tp[0] = tp[1] = 1.0                     # mass at the start
    r = iou_time(_grid(tp, [1] * NF), [(0.8, 1.0)], duration=1.0, n_freq=NF, thresh=0.0)
    assert r is not None and r["iou"] == 0.0


def test_iou_time_median_threshold_default():
    # a diffuse-but-peaked map; median threshold keeps the above-median bins.
    tp = [0.1] * 10
    tp[5] = 1.0
    r = iou_time(_grid(tp, [1] * NF), [(0.5, 0.6)], duration=1.0, n_freq=NF)  # median
    assert r is not None and 0.0 <= r["iou"] <= 1.0


def test_iou_time_none_cases():
    # no windows / no duration are genuinely undefined → None.
    assert iou_time(_grid([1, 1], [1] * NF), [], 1.0, NF) is None
    assert iou_time(_grid([1, 1], [1] * NF), [(0, 0.5)], 0.0, NF) is None


def test_iou_time_zero_mass_scores_low_not_none():
    # A collapsed map is scored, not nulled. Under the uniform fallback the per-bin
    # mass is constant, so a strict > median threshold keeps nothing → a low IoU
    # (0.0 here), never None. The threshold-free soft IoU is the robust companion.
    r = iou_time([0.0] * (4 * NF), [(0, 0.5)], 1.0, NF)
    assert r is not None
    assert r["iou"] == 0.0


def test_soft_iou_time_collapsed_map_is_low_not_none():
    from grounding_metrics import soft_iou_time
    # uniform/collapsed map → soft IoU near the gt-fraction floor, always a number.
    r = soft_iou_time([0.0] * (10 * NF), [(0.0, 0.5)], 1.0, NF)
    assert r is not None
    assert 0.0 < r["soft_iou"] < 0.6
    # a perfectly concentrated map on the GT bins → soft IoU ~1.
    tp = [0.0] * 10
    tp[0] = tp[1] = tp[2] = tp[3] = tp[4] = 1.0   # mass on the first half = GT window
    r2 = soft_iou_time(_grid(tp, [1] * NF), [(0.0, 0.5)], 1.0, NF)
    assert r2 is not None and r2["soft_iou"] > 0.95
    # no windows → None.
    assert soft_iou_time(_grid(tp, [1] * NF), [], 1.0, NF) is None


# ── pointing game (argmax bin inside a window?) ──────────────────────────────
def test_pointing_game_hit_and_miss():
    tp = [0.0] * 10
    tp[4] = tp[5] = 1.0
    hit = pointing_game(_grid(tp, [1] * NF), [(0.4, 0.6)], duration=1.0, n_freq=NF)
    assert hit is not None and hit["hit"] is True
    tp2 = [1.0] * 10
    tp2[4] = tp2[5] = 0.0
    miss = pointing_game(_grid(tp2, [1] * NF), [(0.4, 0.6)], duration=1.0, n_freq=NF)
    assert miss is not None and miss["hit"] is False


def test_pointing_game_none_cases():
    # no windows → genuinely undefined → None.
    assert pointing_game(_grid([1, 1], [1] * NF), [], 1.0, NF) is None


def test_pointing_game_zero_mass_returns_dict_not_none():
    # A collapsed map falls back to uniform; argmax is deterministic (bin 0) so the
    # metric still returns a result dict (a number), never None.
    r = pointing_game([0.0] * (3 * NF), [(0, 0.5)], 1.0, NF)
    assert r is not None
    assert "hit" in r and isinstance(r["hit"], bool)


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
