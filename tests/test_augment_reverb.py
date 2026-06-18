"""
Tests for scripts/augment_reverb.py — the controlled-synthesis REVERB augmenter.

These are pure-numpy/scipy tests. They use NO audio files, NO soundfile, and NO
pyroomacoustics (pra_rir is never called here — it is import-lazy and exercised only
on the cluster). The RT60 math is validated on a pure-numpy synthetic RIR.

The key proof is test_measure_rt60_recovers_synthetic: building an exponential RIR
for a target RT60 and measuring it back must recover that RT60. If it fails, the
tau↔RT60 derivation or the Schroeder fit is wrong — fix the math, not the tolerance.
"""

import os
import sys

import numpy as np
import pytest

# make scripts/ importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from augment_reverb import (  # noqa: E402
    RT60_COLUMN,
    REVERB_AFFECTED_COLUMNS,
    augment_feature_row,
    measure_rt60,
    reverberate,
    synth_exp_rir,
)


# ──────────────────────────────────────────────────────────────────────────────
# 1. RT60 control — THE correctness proof
# ──────────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("target_rt60", [0.3, 0.6, 1.0])
def test_measure_rt60_recovers_synthetic(target_rt60):
    """
    Build a synthetic exponential RIR for a known RT60, measure it back via
    Schroeder backward integration, and require the recovered RT60 within 15%
    relative of the target.

    Why 15% and not tighter: the synthetic RIR is white noise * an exponential
    envelope, so the squared-and-integrated EDC is the deterministic exponential
    decay PLUS the variance of a finite white-noise realization. That ripple makes
    the dB curve wiggle around the ideal straight line, so a least-squares fit over
    the [-5,-35] dB window lands a few percent off the analytic slope (and varies
    with the noise draw). 15% comfortably covers that stochastic fit error while
    still being far tighter than a "wrong by 2x" tau-derivation bug would produce
    (a factor-of-2 tau error shows up as ~100% RT60 error). It proves the control is
    real without pretending the Schroeder fit is exact.
    """
    sr = 16000
    rir = synth_exp_rir(target_rt60, sr)
    est = measure_rt60(rir, sr)
    assert np.isfinite(est), f"measure_rt60 returned non-finite for rt60={target_rt60}"
    rel_err = abs(est - target_rt60) / target_rt60
    assert rel_err <= 0.15, (
        f"target_rt60={target_rt60}s recovered as {est:.4f}s "
        f"(rel err {rel_err:.1%} > 15%) — check tau↔RT60 or Schroeder fit"
    )


def test_measure_rt60_monotone_in_target():
    """A longer target RT60 must measure back as a longer RT60 (ordering preserved)."""
    sr = 16000
    estimates = [measure_rt60(synth_exp_rir(rt, sr), sr) for rt in (0.3, 0.6, 1.0)]
    assert estimates[0] < estimates[1] < estimates[2]


def test_measure_rt60_degenerate_inputs():
    """Too-short or zero IRs return NaN rather than raising or returning garbage."""
    assert np.isnan(measure_rt60(np.zeros(4), 16000))      # too short
    assert np.isnan(measure_rt60(np.zeros(2048), 16000))   # no energy
    assert np.isnan(measure_rt60(np.ones(2048), 0))        # bad sr


# ──────────────────────────────────────────────────────────────────────────────
# 2. reverberate: length preservation
# ──────────────────────────────────────────────────────────────────────────────
def test_reverberate_preserves_length():
    """Output length must equal input length for various RIR lengths."""
    rng = np.random.default_rng(0)
    speech = rng.standard_normal(16000) * 0.2
    for rt60 in (0.3, 0.8, 1.2):
        rir = synth_exp_rir(rt60, 16000)
        out = reverberate(speech, rir)
        assert out.shape[0] == speech.shape[0], f"length changed at rt60={rt60}"


def test_reverberate_short_rir():
    """A trivially short RIR (single spike) returns the input length unchanged."""
    rng = np.random.default_rng(1)
    speech = rng.standard_normal(8000) * 0.2
    rir = np.array([1.0])  # identity RIR
    out = reverberate(speech, rir)
    assert out.shape[0] == speech.shape[0]
    # Identity RIR (single unit spike at index 0) returns the speech unchanged.
    np.testing.assert_allclose(out, speech, atol=1e-9)


# ──────────────────────────────────────────────────────────────────────────────
# 3. reverberate: clipping guard
# ──────────────────────────────────────────────────────────────────────────────
def test_reverberate_clipping_guard():
    """Loud input is normalized to peak <= 1; quiet input is NOT amplified."""
    sr = 16000
    rir = synth_exp_rir(0.6, sr)

    # Loud input -> convolution likely exceeds 1.0 -> must be capped.
    rng = np.random.default_rng(2)
    loud = 5.0 * rng.standard_normal(16000)
    out_loud = reverberate(loud, rir)
    assert np.max(np.abs(out_loud)) <= 1.0 + 1e-9

    # No-amplification check. The synth RIR has large total gain (white noise *
    # envelope), so any non-trivial input convolved with it can exceed 1.0; that is
    # genuine and correctly capped. To isolate the "never scaled UP" guarantee we use
    # an attenuating RIR (single sub-unit spike) whose convolution output is already
    # below 1.0 — the guard must leave it untouched, NOT amplify it to fill [-1,1].
    quiet = 0.01 * rng.standard_normal(16000)
    atten_rir = np.array([0.5])  # scales input by 0.5, output peak well under 1
    out_quiet = reverberate(quiet, atten_rir)
    expected_peak = 0.5 * float(np.max(np.abs(quiet)))
    assert np.max(np.abs(out_quiet)) <= 1.0 + 1e-9
    # Left as-is: peak equals the un-normalized 0.5*input peak (no upward scaling).
    assert np.max(np.abs(out_quiet)) == pytest.approx(expected_peak, abs=1e-9)
    assert expected_peak < 0.5  # sanity: it really was quiet


# ──────────────────────────────────────────────────────────────────────────────
# 4. Partial-GT feature row: rt60 set, srmr blanked, filename uniquified
# ──────────────────────────────────────────────────────────────────────────────
def _clean_row():
    # mirrors src/feature_extractor_mix.py COLUMN_ORDER plus an extra trailing col.
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
        "overlap_ratio_vad": "0.80",
    }


def test_augment_feature_row():
    clean = _clean_row()
    out = augment_feature_row(clean, target_rt60=0.5)

    # rt60 column set to the injected target (3 decimals, as string).
    assert out[RT60_COLUMN] == "0.500"

    # filename uniquified with the _augR<ms> tag, extension preserved.
    assert out["filename"] != clean["filename"]
    assert "augR" in out["filename"]
    assert out["filename"].endswith(".wav")

    # SRMR-handling choice: reverb-affected columns BLANKED for re-extraction.
    for col in REVERB_AFFECTED_COLUMNS:
        assert out[col] == "", f"{col} should be blanked for downstream re-extraction"
    assert "srmr" in REVERB_AFFECTED_COLUMNS  # srmr is the headline reverb-affected col

    # F0 / timing columns passed through unchanged (documented linear-reverb assumption).
    for key in ("f0_mean_hz", "f0_sd_hz", "praat_speaking_rate_syl_sec", "praat_pause_count"):
        assert out[key] == clean[key], f"{key} should pass through unchanged"

    # original row not mutated.
    assert clean[RT60_COLUMN] == clean.get(RT60_COLUMN) if RT60_COLUMN in clean else True
    assert clean["srmr"] == "5.34"
    assert clean["filename"] == "1089-134686-0000_121-127105-0031.wav"


def test_augment_feature_row_adds_rt60_when_absent():
    """The clean CSV has no rt60 column; augment_feature_row must ADD it."""
    clean = _clean_row()
    assert RT60_COLUMN not in clean
    out = augment_feature_row(clean, 0.8)
    assert RT60_COLUMN in out
    assert out[RT60_COLUMN] == "0.800"


def test_augment_feature_row_distinct_rt60_distinct_filenames():
    """Distinct RT60 grid points must yield distinct filenames (no collisions)."""
    clean = _clean_row()
    names = {augment_feature_row(clean, rt)["filename"] for rt in (0.3, 0.5, 0.8, 1.2)}
    assert len(names) == 4
