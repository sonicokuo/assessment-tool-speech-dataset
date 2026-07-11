"""f0_clean.py — well-posed F0 ground truth from non-overlapped voiced frames.

WHY
---
`feature_extractor_mix.compute_f0_variation` runs Praat pitch tracking on the
2-speaker MIXTURE (mix_clean). Libri2Mix is ~78% overlapped, and a pitch tracker
on mixed speech locks onto whichever speaker dominates a frame, producing
octave jumps and track-switches. So the F0 ground truth (f0_mean_hz / f0_sd_hz)
that SFS scores is itself ill-posed under overlap — which is the main reason
f0_mean / f0_sd are the worst-scoring features (≈0.15 / 0.19 F1). Restricting
the statistics to VOICED frames OUTSIDE the overlap windows makes the reference
well-defined (it is the pitch of the single active speaker), without dropping
the feature.

The overlap windows come from VAD on the separated s1/s2 stems (already produced
by feature_extractor_mix.compute_overlap_vad and stored as overlap_segments_vad,
a ';'-joined list of '<start>-<end>' SAMPLE indices), so no new model is needed.

This module keeps the masking logic pure (numpy only) so it is unit-testable
without Praat; the Praat wrapper imports parselmouth lazily.
"""
from __future__ import annotations

import numpy as np


def parse_overlap_windows_samples(raw: str, sr: int = 16000) -> list[tuple[float, float]]:
    """'a-b;c-d' (sample indices) -> [(start_s, end_s), ...] in seconds."""
    out: list[tuple[float, float]] = []
    for seg in (raw or "").split(";"):
        seg = seg.strip()
        if not seg:
            continue
        try:
            a, b = seg.split("-", 1)
            s0, s1 = int(a) / sr, int(b) / sr
        except (ValueError, IndexError):
            continue
        if s1 > s0:
            out.append((s0, s1))
    return out


def f0_stats_voiced_nonoverlap(
    f0_values,
    frame_times,
    overlap_windows: list[tuple[float, float]],
) -> dict:
    """Mean / SD of F0 over VOICED frames OUTSIDE every overlap window.

    Args:
        f0_values:   per-frame F0 in Hz, 0 (or <=0) marking unvoiced frames.
        frame_times: center time (seconds) of each frame; same length as f0_values.
        overlap_windows: list of (start_s, end_s) windows to EXCLUDE (half-open
            [start, end): a frame at exactly `start` is excluded, at `end` kept).

    Returns dict: f0_mean_hz, f0_sd_hz (NaN if < 2 clean voiced frames),
    n_clean_voiced, clean_voiced_frac (clean voiced / all voiced).
    """
    f0 = np.asarray(f0_values, dtype=float).ravel()
    t = np.asarray(frame_times, dtype=float).ravel()
    if f0.shape != t.shape:
        raise ValueError(f"f0_values ({f0.shape}) and frame_times ({t.shape}) must align")

    voiced = f0 > 0.0
    in_overlap = np.zeros_like(voiced, dtype=bool)
    for (s, e) in overlap_windows:
        in_overlap |= (t >= s) & (t < e)
    keep = voiced & ~in_overlap

    clean = f0[keep]
    n_clean = int(keep.sum())
    n_voiced = int(voiced.sum())
    frac = round(n_clean / n_voiced, 4) if n_voiced else 0.0
    if n_clean < 2:
        return {
            "f0_mean_hz": float("nan"),
            "f0_sd_hz": float("nan"),
            "n_clean_voiced": n_clean,
            "clean_voiced_frac": frac,
        }
    return {
        "f0_mean_hz": round(float(np.mean(clean)), 2),
        "f0_sd_hz": round(float(np.std(clean, ddof=1)), 2),
        "n_clean_voiced": n_clean,
        "clean_voiced_frac": frac,
    }


def compute_f0_variation_clean(
    wav_path: str,
    overlap_windows: list[tuple[float, float]],
    min_pitch: float = 75.0,
    max_pitch: float = 500.0,
    time_step: float = 0.01,
) -> dict:
    """Praat pitch on `wav_path`, then F0 stats over non-overlapped voiced frames.

    `wav_path` is still the mixture (we have one audio file), but the windows
    carve out the overlapped spans so the statistics describe the single active
    speaker. parselmouth is imported lazily so this module stays import-light.
    """
    import parselmouth  # lazy: heavy, PSC-only

    snd = parselmouth.Sound(wav_path)
    pitch = snd.to_pitch(time_step=time_step, pitch_floor=min_pitch, pitch_ceiling=max_pitch)
    f0_values = pitch.selected_array["frequency"]
    frame_times = pitch.xs()
    return f0_stats_voiced_nonoverlap(f0_values, frame_times, overlap_windows)
