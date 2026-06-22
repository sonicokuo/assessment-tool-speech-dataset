"""Unit tests for src/snr_maps.py — oracle dense SNR supervision targets.

Pure numpy, no torch/Praat, so these run anywhere. They assert the documented
target semantics:
  * snr_timeline is HIGH where only s1 plays, LOW where only s2 plays, ~0 at equal.
  * the clamp is [-30, 40] dB.
  * s1_active flags only s1-bearing frames (silence is masked).
  * irm_map ~1 where s1 dominates a band, ~0 where s2 dominates, ~0.5 at equal.
  * grid sizes match the WavLM 50 Hz / BEATs T_p conventions.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from snr_maps import (  # noqa: E402
    snr_timeline_from_stems,
    irm_map_from_stems,
    snr_maps_from_stems,
    wavlm_n_frames,
    beats_t_p,
    SNR_CLAMP_DB,
    F_BINS_DEFAULT,
)

SR = 16000


def _tone(freq, n, amp=0.5, sr=SR):
    t = np.arange(n) / sr
    return amp * np.sin(2 * np.pi * freq * t)


def _scenario():
    """5 s clip. s1 (200 Hz) plays 0-2.5 and 3.5-5; s2 (3000 Hz) plays 1.5-3.5.
    Regions: [0,1.5] s1-only, [1.5,2.5] overlap, [2.5,3.5] s2-only, [3.5,5] s1-only.
    """
    n = int(5.0 * SR)
    s1 = np.zeros(n)
    s1[: int(2.5 * SR)] = _tone(200, int(2.5 * SR))
    s1[int(3.5 * SR):] = _tone(200, n - int(3.5 * SR))
    s2 = np.zeros(n)
    seg = int(3.5 * SR) - int(1.5 * SR)
    s2[int(1.5 * SR): int(3.5 * SR)] = _tone(3000, seg)
    return s1, s2, n


# ── frame-grid helpers ───────────────────────────────────────────────────────
def test_wavlm_frame_count():
    assert wavlm_n_frames(int(5.0 * SR)) == 250          # 5 s * 50 Hz
    assert wavlm_n_frames(320) == 1
    assert wavlm_n_frames(319) == 0


def test_beats_t_p_5s():
    # spec_encoder: T_p ~= 31 for a 5 s clip
    assert beats_t_p(int(5.0 * SR)) == 31


# ── (1) SNR timeline ─────────────────────────────────────────────────────────
def test_timeline_length_is_wavlm_frames():
    s1, s2, n = _scenario()
    tl = snr_timeline_from_stems(s1, s2)
    assert tl["snr_timeline"].shape[0] == wavlm_n_frames(n) == 250


def test_timeline_tracks_overlap_regions():
    s1, s2, n = _scenario()
    snr = snr_timeline_from_stems(s1, s2)["snr_timeline"]

    def med(a, b):  # seconds → 50 Hz frames
        return float(np.median(snr[int(a * 50): int(b * 50)]))

    s1_only = med(0.2, 1.4)
    overlap = med(1.6, 2.4)
    s2_only = med(2.6, 3.4)
    assert s1_only > 30.0                 # clean target → near +40 clamp
    assert abs(overlap) < 5.0             # equal tones → ~0 dB
    assert s2_only < -25.0                # target silent → near -30 clamp
    assert s1_only > overlap > s2_only    # monotone ordering


def test_timeline_clamp_bounds():
    s1, s2, _ = _scenario()
    snr = snr_timeline_from_stems(s1, s2)["snr_timeline"]
    assert snr.min() >= SNR_CLAMP_DB[0] - 1e-4
    assert snr.max() <= SNR_CLAMP_DB[1] + 1e-4


def test_s1_active_masks_silence():
    s1, s2, _ = _scenario()
    out = snr_timeline_from_stems(s1, s2)
    act = out["s1_active"]
    # s1 silent in [2.5, 3.5] → inactive; speaking at 0.5 → active
    assert not act[int(2.8 * 50)]
    assert act[int(0.5 * 50)]
    assert act[int(4.5 * 50)]


def test_timeline_force_n_frames():
    s1, s2, _ = _scenario()
    out = snr_timeline_from_stems(s1, s2, n_frames=240)
    assert out["snr_timeline"].shape[0] == 240
    assert out["s1_active"].shape[0] == 240
    out2 = snr_timeline_from_stems(s1, s2, n_frames=260)   # pad
    assert out2["snr_timeline"].shape[0] == 260
    assert not out2["s1_active"][255]                      # pad frames inactive


def test_timeline_empty_input():
    out = snr_timeline_from_stems(np.zeros(0), np.zeros(0))
    assert out["snr_timeline"].shape[0] == 0


# ── (2) IRM map ──────────────────────────────────────────────────────────────
def test_irm_grid_shape_default_beats():
    s1, s2, n = _scenario()
    irm = irm_map_from_stems(s1, s2)
    assert irm["f_bins"] == F_BINS_DEFAULT == 8
    assert irm["t_p"] == beats_t_p(n) == 31
    assert irm["irm_map"].shape == (31, 8)


def test_irm_range_and_active_mask_shapes():
    s1, s2, _ = _scenario()
    irm = irm_map_from_stems(s1, s2)
    m = irm["irm_map"]
    assert m.min() >= 0.0 and m.max() <= 1.0
    assert irm["irm_active"].shape == m.shape
    assert irm["irm_energy"].shape == m.shape


def test_irm_s1_dominant_band_is_one():
    s1, s2, n = _scenario()
    irm = irm_map_from_stems(s1, s2)
    m, tp = irm["irm_map"], irm["t_p"]
    # s1 is 200 Hz → lowest band (band 0). In the s1-only region it must be ~1.
    t = int(0.5 / 5.0 * tp)
    assert m[t, 0] > 0.9


def test_irm_s2_only_region_is_zero():
    s1, s2, n = _scenario()
    irm = irm_map_from_stems(s1, s2)
    m, tp = irm["irm_map"], irm["t_p"]
    t = int(3.0 / 5.0 * tp)               # s2-only window
    assert m[t].max() < 0.1               # target absent everywhere → IRM ~0


def test_irm_equal_energy_is_half():
    # two tones in the SAME band, equal amplitude → IRM = sqrt(0.5) ~= 0.707
    n = int(2.0 * SR)
    a = _tone(500, n, amp=0.5)
    b = _tone(520, n, amp=0.5)            # near 500 Hz → same low band
    irm = irm_map_from_stems(a, b, f_bins=8)
    m = irm["irm_map"]
    band0 = m[:, 0]
    assert 0.55 < float(band0.mean()) < 0.85


def test_irm_custom_f_bins_and_t_p():
    s1, s2, _ = _scenario()
    irm = irm_map_from_stems(s1, s2, t_p=20, f_bins=4)
    assert irm["irm_map"].shape == (20, 4)


def test_irm_short_clip_does_not_crash():
    s1 = _tone(200, 1000)
    s2 = _tone(3000, 1000)
    irm = irm_map_from_stems(s1, s2, t_p=8, f_bins=8)
    assert irm["irm_map"].shape == (8, 8)
    assert np.isfinite(irm["irm_map"]).all()


# ── combined convenience ─────────────────────────────────────────────────────
def test_snr_maps_from_stems_bundle():
    s1, s2, n = _scenario()
    out = snr_maps_from_stems(s1, s2)
    assert out["T"] == wavlm_n_frames(n)
    assert out["snr_timeline"].shape[0] == out["T"]
    assert out["irm_map"].shape == (out["t_p"], out["f_bins"])
    assert out["irm_active"].shape == out["irm_map"].shape


def test_silent_interferer_caps_high():
    # s2 all zero → very clean → timeline pinned at +40 clamp on active frames
    s1 = _tone(200, int(2.0 * SR))
    s2 = np.zeros(int(2.0 * SR))
    out = snr_timeline_from_stems(s1, s2)
    act = out["s1_active"]
    assert float(out["snr_timeline"][act].min()) > 35.0


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
