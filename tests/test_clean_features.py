"""Tests for src/clean_features.py — clean-stem GT for recoverable features.

Only the SNR math is pure-numpy and tested here (SRMR / Praat passes need their
heavy deps and a compute node, so they are smoke-only and skipped if absent).
"""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from clean_features import clean_snr_db, _ratio_to_db  # noqa: E402


def _tone(freq, sr, n, amp):
    t = np.arange(n, dtype=float)
    return amp * np.sin(2 * np.pi * freq * t / sr)


# ── _ratio_to_db ────────────────────────────────────────────────────────────
def test_ratio_to_db_basic():
    # 10x power -> 10 dB
    assert _ratio_to_db(10.0, 1.0, 1e-12) == 10.0
    # equal power -> 0 dB
    assert _ratio_to_db(1.0, 1.0, 1e-12) == 0.0


def test_ratio_to_db_silent_interferer_caps():
    # interferer below eps -> capped at +60 dB (very clean), never +inf
    assert _ratio_to_db(1.0, 0.0, 1e-12) == 60.0


# ── clean_snr_db ────────────────────────────────────────────────────────────
def test_snr_target_10x_louder_is_20db():
    sr, n = 16000, 16000
    s1 = _tone(200, sr, n, 0.5)   # 10x amplitude -> 100x power -> 20 dB
    s2 = _tone(150, sr, n, 0.05)
    assert abs(clean_snr_db(s1, s2, sr) - 20.0) < 0.5


def test_snr_equal_power_is_zero():
    sr, n = 16000, 16000
    s1 = _tone(200, sr, n, 0.5)
    s2 = _tone(150, sr, n, 0.5)   # same power
    assert abs(clean_snr_db(s1, s2, sr)) < 0.5


def test_snr_silent_interferer_caps_at_60():
    sr, n = 16000, 16000
    s1 = _tone(200, sr, n, 0.5)
    s2 = np.zeros(n)
    assert clean_snr_db(s1, s2, sr) == 60.0


def test_snr_excludes_target_silence():
    """The whole point: interferer energy that lands in the TARGET's silence must
    NOT drag the SNR down, because the metric is over the target's active frames.

    This is what makes it better-posed than P90/P10-on-mix, which would conflate
    the interferer-in-silence energy into the floor.
    """
    sr = 16000
    # 1s of target speech, then 1s of target silence
    s1 = np.concatenate([_tone(200, sr, sr, 0.5), np.zeros(sr)])
    # interferer ONLY in the target's silent second
    s2 = np.concatenate([np.zeros(sr), _tone(150, sr, sr, 0.5)])
    snr = clean_snr_db(s1, s2, sr)
    # interferer is essentially absent while the target talks -> high SNR
    assert snr > 12.0


def test_snr_empty_target_is_nan():
    sr = 16000
    assert math.isnan(clean_snr_db(np.zeros(sr), _tone(150, sr, sr, 0.1), sr))


def test_snr_length_mismatch_truncates():
    sr = 16000
    s1 = _tone(200, sr, sr, 0.5)
    s2 = _tone(150, sr, sr // 2, 0.5)   # half length
    v = clean_snr_db(s1, s2, sr)
    assert not math.isnan(v)
