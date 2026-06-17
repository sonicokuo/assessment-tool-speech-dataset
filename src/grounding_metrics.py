"""grounding_metrics.py — quantify whether section attention is grounded.

Generalizes the overlap-attention concentration in scripts/attention_gt_alignment.py
to (a) ANY time-windowed attribute (overlap, pauses) and (b) a FREQUENCY-band
attribute (pitch), which is the axis unique to a 2D time-frequency map and that a
1D temporal map / a CV bounding box cannot express.

The section attention map is a flat vector of length T_p * n_freq, row-major with
time as the outer axis (index = t * n_freq + f), n_freq = BEATs freq patches (8).
All functions are pure numpy so they are unit-testable without a model.

Concentration ratio convention (both axes):
    ratio = (attention mass inside the region) / (fraction the region occupies)
  > 1  attention concentrates in the region more than uniform  → grounded
  ≈ 1  no better than uniform
  < 1  attention avoids the region
"""
from __future__ import annotations

import numpy as np

F_P_DEFAULT = 8   # BEATs frequency patches (128 mel / 16 patch)


def _grid(alpha_flat, n_freq: int) -> np.ndarray:
    a = np.asarray(alpha_flat, dtype=float).ravel()
    t_p = a.size // n_freq
    if t_p == 0:
        raise ValueError(f"alpha of size {a.size} too short for n_freq={n_freq}")
    return a[: t_p * n_freq].reshape(t_p, n_freq)


def time_mass(alpha_flat, n_freq: int = F_P_DEFAULT) -> np.ndarray:
    """Per-time-bin attention mass (marginalized over frequency), normalized to sum 1."""
    m = _grid(alpha_flat, n_freq).mean(axis=1)
    s = m.sum()
    return m / s if s > 0 else m


def freq_mass(alpha_flat, n_freq: int = F_P_DEFAULT) -> np.ndarray:
    """Per-frequency-band attention mass (marginalized over time), normalized to sum 1."""
    m = _grid(alpha_flat, n_freq).mean(axis=0)
    s = m.sum()
    return m / s if s > 0 else m


def time_concentration_ratio(
    alpha_flat,
    windows: list[tuple[float, float]],
    duration: float,
    n_freq: int = F_P_DEFAULT,
) -> dict | None:
    """Concentration of attention in time `windows` (overlap or pause spans).

    Returns {mass_in, frac_time, ratio, n_bins} or None when there are no windows,
    no duration, or zero attention mass. A bin counts as in-window if it overlaps
    any window (half-open intersection).
    """
    if not windows or duration <= 0:
        return None
    tm = time_mass(alpha_flat, n_freq)
    if tm.sum() <= 0:
        return None
    t_p = len(tm)
    bin_dur = duration / t_p
    mass_in = 0.0
    for i in range(t_p):
        t0, t1 = i * bin_dur, (i + 1) * bin_dur
        if any(t0 < e and t1 > s for s, e in windows):
            mass_in += tm[i]
    win_dur = sum(e - s for s, e in windows)
    frac = min(win_dur / duration, 1.0)
    if frac <= 0:
        return None
    return {"mass_in": float(mass_in), "frac_time": float(frac),
            "ratio": float(mass_in / frac), "n_bins": t_p}


def freq_band_concentration_ratio(
    alpha_flat,
    band_rows: list[int],
    n_freq: int = F_P_DEFAULT,
) -> dict | None:
    """Concentration of attention in a FREQUENCY band (e.g. pitch -> low mel rows).

    band_rows are frequency-patch indices in [0, n_freq). For the pitch section we
    expect attention to concentrate in the low band where the fundamental and its
    first harmonics live; a ratio > 1 there, and ≈ 1 for non-pitch sections, is
    the frequency-axis grounding result. Returns {mass_in, frac_band, ratio} or
    None on zero mass / empty band.
    """
    fm = freq_mass(alpha_flat, n_freq)
    if fm.sum() <= 0:
        return None
    band = sorted({r for r in band_rows if 0 <= r < n_freq})
    if not band:
        return None
    mass_in = float(sum(fm[r] for r in band))
    frac = len(band) / n_freq
    return {"mass_in": mass_in, "frac_band": frac, "ratio": float(mass_in / frac),
            "n_freq": n_freq}
