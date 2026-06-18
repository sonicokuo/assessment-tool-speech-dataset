"""
Tests for scripts/augment_noise.py — the controlled-synthesis noise augmenter.

These are pure-numpy tests (no audio files, no soundfile). They synthesize
speech/noise signals and prove the core paper claim:

    when we mix at a target SNR by construction, the injected SNR is EXACT.

The SNR-exactness test is non-negotiable: if it fails, the mixing math is wrong.
"""

import os
import sys

import numpy as np
import pytest

# make scripts/ importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from augment_noise import (  # noqa: E402
    augment_feature_row,
    fit_noise_length,
    mix_at_snr,
    noise_scale_for_snr,
    power,
    snr_db,
)


# ──────────────────────────────────────────────────────────────────────────────
# 1. SNR exactness — THE correctness proof
# ──────────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("target", [-5.0, 0.0, 10.0, 20.0])
def test_mix_achieves_target_snr(target):
    """After scaling, speech-vs-scaled-noise SNR must equal the target exactly."""
    rng = np.random.default_rng(0)
    t = np.linspace(0, 1, 16000, endpoint=False)
    speech = 0.5 * np.sin(2 * np.pi * 220 * t)        # synthetic "speech" tone
    noise = rng.standard_normal(16000) * 0.3          # synthetic noise

    _mix, info = mix_at_snr(speech, noise, target_snr_db=target)

    # The returned scaled noise vs returned speech must realize the target SNR.
    assert info["realized_snr_db"] == pytest.approx(target, abs=1e-6)
    # And computed directly from the scale function on the fitted noise:
    noise_fit = fit_noise_length(noise, speech.size)
    alpha = noise_scale_for_snr(speech, noise_fit, target)
    assert snr_db(speech, alpha * noise_fit) == pytest.approx(target, abs=1e-6)


@pytest.mark.parametrize("target", [-10.0, -5.0, 0.0, 5.0, 10.0, 20.0, 30.0, 40.0])
def test_mix_snr_exact_full_grid(target):
    """Sweep a wide SNR grid (incl. the range Libri2Mix lacks) — all exact."""
    rng = np.random.default_rng(42)
    speech = rng.standard_normal(8000)
    noise = rng.standard_normal(8000)
    _mix, info = mix_at_snr(speech, noise, target_snr_db=target)
    assert info["realized_snr_db"] == pytest.approx(target, abs=1e-6)


def test_snr_db_definition():
    """snr_db must match 10*log10(P_s/P_n) on a hand-computable case."""
    speech = np.array([1.0, -1.0, 1.0, -1.0])      # power 1.0
    noise = np.array([0.1, -0.1, 0.1, -0.1])       # power 0.01
    # 10*log10(1/0.01) = 10*log10(100) = 20 dB
    assert snr_db(speech, noise) == pytest.approx(20.0, abs=1e-9)
    assert power(speech) == pytest.approx(1.0, abs=1e-12)
    assert power(noise) == pytest.approx(0.01, abs=1e-12)


# ──────────────────────────────────────────────────────────────────────────────
# 2. Exact-GT feature row: only SNR (and filename) change
# ──────────────────────────────────────────────────────────────────────────────
def _clean_row():
    # mirrors src/feature_extractor_mix.py COLUMN_ORDER plus extra trailing cols
    return {
        "filename": "1089-134686-0000_121-127105-0031.wav",
        "filepath": "/data/test/mix_clean/1089-134686-0000_121-127105-0031.wav",
        "duration_sec": "4.275",
        "sample_rate_hz": "16000",
        "snr_db": "18.71",
        "silence_ratio": "0.12",
        "overlap_ratio": "0.78",
        "overlap_segments": "[(0.5, 1.2)]",
        "srmr": "5.34",
        "f0_mean_hz": "121.4",
        "f0_sd_hz": "22.1",
        "praat_speaking_rate_syl_sec": "4.2",
        "praat_pause_count": "3",
        "overlap_ratio_vad": "0.80",  # extra trailing column from real CSV
    }


def test_augment_feature_row_overwrites_only_snr():
    clean = _clean_row()
    out = augment_feature_row(clean, target_snr_db=5.0)

    # SNR overwritten to the injected target, 2 decimals, as string
    assert out["snr_db"] == "5.00"
    # filename made unique
    assert out["filename"] != clean["filename"]
    assert "augN" in out["filename"]
    assert out["filename"].endswith(".wav")

    # every OTHER feature column unchanged
    for key in clean:
        if key in ("snr_db", "filename"):
            continue
        assert out[key] == clean[key], f"column {key} should be unchanged"

    # original row not mutated
    assert clean["snr_db"] == "18.71"
    assert clean["filename"] == "1089-134686-0000_121-127105-0031.wav"


def test_augment_feature_row_rounds_and_signs():
    clean = _clean_row()
    assert augment_feature_row(clean, 12.345)["snr_db"] == "12.34" or \
        augment_feature_row(clean, 12.345)["snr_db"] == "12.35"  # banker's rounding tolerant
    neg = augment_feature_row(clean, -5.0)
    assert neg["snr_db"] == "-5.00"
    # distinct SNRs produce distinct filenames (so multiple augments don't collide)
    f_a = augment_feature_row(clean, 0.0)["filename"]
    f_b = augment_feature_row(clean, 10.0)["filename"]
    assert f_a != f_b


# ──────────────────────────────────────────────────────────────────────────────
# 3. Noise length handling: tile or crop to speech length
# ──────────────────────────────────────────────────────────────────────────────
def test_noise_tiled_or_cropped_to_speech_length():
    rng = np.random.default_rng(7)
    speech = rng.standard_normal(10000)

    short_noise = rng.standard_normal(3000)   # shorter -> tiled
    long_noise = rng.standard_normal(50000)   # longer  -> cropped

    mix_short, info_short = mix_at_snr(speech, short_noise, 10.0)
    mix_long, info_long = mix_at_snr(speech, long_noise, 10.0)

    assert mix_short.shape[0] == speech.shape[0]
    assert mix_long.shape[0] == speech.shape[0]

    # fit_noise_length itself
    assert fit_noise_length(short_noise, 10000).shape[0] == 10000
    assert fit_noise_length(long_noise, 10000).shape[0] == 10000
    # SNR still exact regardless of tiling/cropping. Use the returned (possibly
    # clip-normed) speech/scaled_noise from info, NOT (mix - original speech):
    # if clip-norm fired, the original speech is no longer the mix's speech part.
    assert snr_db(info_short["speech"], info_short["scaled_noise"]) == pytest.approx(10.0, abs=1e-6)
    assert snr_db(info_long["speech"], info_long["scaled_noise"]) == pytest.approx(10.0, abs=1e-6)


# ──────────────────────────────────────────────────────────────────────────────
# 4. Edge cases: clipping guard preserves SNR; length always preserved
# ──────────────────────────────────────────────────────────────────────────────
def test_clipping_guard_preserves_snr_and_caps_peak():
    """A loud mix is normalized down, but the injected SNR stays exact."""
    rng = np.random.default_rng(3)
    speech = 5.0 * rng.standard_normal(4000)   # large amplitude -> mix will clip
    noise = rng.standard_normal(4000)

    mix, info = mix_at_snr(speech, noise, target_snr_db=0.0, prevent_clipping=True)

    # peak capped at 1.0
    assert np.max(np.abs(mix)) <= 1.0 + 1e-9
    # clip-norm actually fired
    assert info["clip_norm"] < 1.0
    # SNR preserved exactly despite the peak-normalization (same factor on both)
    assert info["realized_snr_db"] == pytest.approx(0.0, abs=1e-6)
    # mix equals returned (clip-normed) speech + scaled noise
    np.testing.assert_allclose(mix, info["speech"] + info["scaled_noise"], atol=1e-9)


def test_no_clip_leaves_signal_untouched():
    """A quiet mix is NOT amplified (we only normalize down)."""
    speech = 0.1 * np.ones(1000)
    noise = 0.1 * np.ones(1000)
    mix, info = mix_at_snr(speech, noise, target_snr_db=0.0, prevent_clipping=True)
    assert info["clip_norm"] == 1.0
    assert np.max(np.abs(mix)) < 1.0


def test_empty_speech_raises():
    with pytest.raises(ValueError):
        mix_at_snr(np.array([]), np.array([1.0, 2.0]), 0.0)
