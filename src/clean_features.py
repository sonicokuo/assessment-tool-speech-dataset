"""clean_features.py — well-posed GT for the RECOVERABLE features, computed from
the CLEAN s1 stem (and the s2 interferer) instead of the 2-speaker mixture.

WHY
---
`feature_extractor_mix.py` measures snr / srmr / speaking_rate / pause_count /
pause_rate on the 2-speaker MIXTURE (mix_clean). Libri2Mix is ~78% overlapped, so
several of these GT numbers are corrupted by the second talker:

  * SNR   — `estimate_snr` does 10*log10(P90/P10) of the per-frame energy envelope
            of the MIX. With two talkers the "noise floor" (P10) is no longer the
            recording's noise floor; it is whatever the quieter talker is doing.
            The statistic measures the energy *dynamic range of two overlaid
            voices*, not a signal-to-noise ratio. It is not even defined relative
            to a target speaker.
  * SRMR  — a single-channel reverberation metric. On a sum of two reverberant
            voices the modulation spectrum is the superposition of two talkers'
            modulations, so the ratio no longer reflects the room.
  * speaking_rate / pause_count / pause_rate — Praat finds silences via an
            intensity-threshold TextGrid. In a 78%-overlapped mixture the second
            talker fills the first talker's pauses, so the mixture has almost no
            silences: ~41% of mix clips get pause_rate = 0 spuriously, and the
            syllable-nuclei rate is inflated by the interferer's syllables.

The clean s1 stem is a single speaker with real silences, so every one of these
is well-posed on it. overlap_ratio stays from the VAD-on-stems oracle (already
correct) and is NOT recomputed here.

KEY FACT (verified on PSC, Libri2Mix wav16k/min):
    mix_clean = s1 + s2   (max|mix-(s1+s2)| = 3e-5, corr(mix-s1, s2) = 1.0)
so the interferer seen by the s1 target is EXACTLY s2, and a target-vs-interferer
SNR is exactly defined as 10*log10(||s1||^2 / ||s2||^2).

This module keeps the math pure (numpy only) so it is unit-testable without Praat
or SRMRpy; the Praat / VERSA wrappers import lazily.

SRMRpy NOTE: the compiled SRMRpy extension uses CPU instructions that fault on the
Bridges2 LOGIN nodes ("Illegal instruction (core dumped)"). Run the SRMR pass on a
COMPUTE node (interact / sbatch), not the login node. `--skip_srmr` lets the rest
of the features be computed anywhere.
"""
from __future__ import annotations

import numpy as np


# ── SNR from the clean stem + interferer ─────────────────────────────────────
def clean_snr_db(
    s1: np.ndarray,
    interferer: np.ndarray,
    sr: int = 16000,
    frame_ms: float = 30.0,
    active_db_below_peak: float = 40.0,
    eps: float = 1e-12,
) -> float:
    """Target-vs-interferer SNR in dB, measured over the TARGET's active frames.

    Definition
    ----------
        SNR = 10 * log10( mean_t in A ||s1[t]||^2  /  mean_t in A ||interferer[t]||^2 )

    where A is the set of short frames in which the TARGET speaker s1 is active
    (frame RMS within `active_db_below_peak` dB of the clip's peak frame RMS).
    Restricting to the target's active frames is what makes this a SNR rather than
    a silence-dominated energy ratio: it answers "while s1 is talking, how much
    louder is s1 than the competing source", which is exactly the quantity a
    listener experiences and the quantity a separation/quality model should report.

    For Libri2Mix mix_clean the interferer is s2 (mix = s1 + s2, verified), so this
    is a true target-vs-interferer ratio. For a genuinely clean single-speaker clip
    (interferer all-zero) the ratio is +inf; we cap it at +60 dB so it stays finite
    and the verbalizer can still emit a number ("very high SNR").

    This is strictly better-posed than 10*log10(P90/P10) on the mixture because (a)
    the numerator/denominator are the actual target and the actual competing
    energy, not two percentiles of a blended envelope, and (b) it is defined
    relative to a specific target speaker.

    Returns NaN only if the target has no active frames at all.
    """
    s1 = np.asarray(s1, dtype=np.float64).ravel()
    interferer = np.asarray(interferer, dtype=np.float64).ravel()
    n = min(s1.shape[0], interferer.shape[0])
    if n == 0:
        return float("nan")
    s1, interferer = s1[:n], interferer[:n]

    flen = max(1, int(round(sr * frame_ms / 1000.0)))
    nf = n // flen
    if nf < 1:
        # clip shorter than one frame: fall back to whole-clip energies
        sig = float(np.mean(s1 ** 2))
        noi = float(np.mean(interferer ** 2))
        if sig <= eps:
            return float("nan")
        return _ratio_to_db(sig, noi, eps)

    s1f = s1[: nf * flen].reshape(nf, flen)
    inf = interferer[: nf * flen].reshape(nf, flen)
    s1_pow = (s1f ** 2).mean(axis=1)
    in_pow = (inf ** 2).mean(axis=1)

    peak = float(s1_pow.max())
    if peak <= eps:
        return float("nan")
    # active = frames whose power is within active_db_below_peak dB of the peak
    thresh = peak * (10.0 ** (-active_db_below_peak / 10.0))
    active = s1_pow >= thresh
    if not active.any():
        active = np.ones_like(s1_pow, dtype=bool)

    sig = float(s1_pow[active].mean())
    noi = float(in_pow[active].mean())
    return _ratio_to_db(sig, noi, eps)


def _ratio_to_db(sig: float, noi: float, eps: float, cap_db: float = 60.0) -> float:
    if noi <= eps:
        return cap_db  # interferer silent during target speech -> very clean
    val = 10.0 * np.log10(sig / noi)
    val = float(np.clip(val, -cap_db, cap_db))
    return round(val, 2)


# ── lazy Praat / VERSA wrappers (single-speaker stem, well-posed) ────────────
def clean_srmr(s1_wav_path: str, srmr_config: dict | None = None) -> float:
    """SRMR of the clean s1 stem. Single-speaker -> well-posed reverberation.

    Imports VERSA lazily. NOTE: SRMRpy faults on Bridges2 login nodes; run on a
    compute node. Returns NaN if VERSA is unavailable or the metric fails.
    """
    cfg = srmr_config or {"max_cf": 128, "fast": True, "norm": False}
    try:
        import soundfile as sf
        from versa.utterance_metrics.srmr import srmr_metric
        audio, sr = sf.read(s1_wav_path)
        if getattr(audio, "ndim", 1) > 1:
            audio = audio.mean(axis=1)
        score = srmr_metric(
            audio, sr,
            n_cochlear_filters=cfg.get("n_cochlear_filters", 23),
            low_freq=cfg.get("low_freq", 125),
            min_cf=cfg.get("min_cf", 4),
            max_cf=cfg.get("max_cf", 128),
            fast=cfg.get("fast", True),
            norm=cfg.get("norm", False),
        )
        return round(float(score["srmr"]), 4)
    except Exception as e:  # noqa: BLE001
        print(f"  [WARNING] clean SRMR failed on {s1_wav_path}: {e}")
        return float("nan")


def clean_rate_and_pauses(s1_wav_path: str, min_pause_dur: float = 0.3) -> dict:
    """speaking_rate / pause_count / pause_rate from the clean s1 stem.

    Reuses feature_extractor_mix's exact Praat passes (same thresholds/algorithm),
    just pointed at the single-speaker stem where silences are real. Returns the
    SFS-scored subset plus the auxiliary pause stats for completeness.
    """
    out = {
        "praat_speaking_rate_syl_sec": float("nan"),
        "praat_articulation_rate_syl_sec": float("nan"),
        "praat_pause_count": float("nan"),
        "praat_pause_rate_per_min": float("nan"),
        "praat_mean_pause_dur_sec": float("nan"),
        "praat_total_pause_dur_sec": float("nan"),
        "praat_pause_to_speech_ratio": float("nan"),
    }
    try:
        import feature_extractor_mix as fx  # lazy: pulls torch/torchaudio
        out.update(fx.compute_praat_speaking_rate(s1_wav_path))
        out.update(fx.compute_praat_pause_patterns(s1_wav_path, min_pause_dur=min_pause_dur))
    except Exception as e:  # noqa: BLE001
        print(f"  [WARNING] clean rate/pause failed on {s1_wav_path}: {e}")
    return out
