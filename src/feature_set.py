"""Canonical 8-feature list for AQUA-NL B-full multi-task training and the aux regression head.

Single source of truth for:
  - The numerical target string used in B-full's forward A ("snr=15.66 srmr=4.5 ...").
  - The (B, 8) scalar tensor + mask used by the aux regression head's MSE.

These 8 features are the scalar entries inside src/section_tags.py's catalog —
one per <f_*> open tag, excluding the overlap_segments span set (which isn't a
scalar). Order MUST stay synced with the scalar entries of FEATURE_TAGS in
section_tags.py.

Update history:
  - 2026-05-11: trimmed from 13 → 7 (drop f0_sd, jitter, shimmer, srmr,
    articulation_rate, pause_rate; keep duration + hnr).
  - 2026-05-12: realigned to the section catalog (8 features). Reverb (srmr),
    pitch SD (f0_sd), and pause_rate are restored because each has its own
    section in the EMNLP design and needs scalar supervision. duration and hnr
    are dropped — duration is an intro sentence outside any section; hnr was
    grouped under voice_quality which we cut to avoid a redundant attention
    figure with pitch.
"""

from __future__ import annotations

import math

import torch


# (short_name, csv_column, format_string)
# Order matches the scalar tags in src/section_tags.py::FEATURE_TAGS.
SUPERVISED_FEATURES: list[tuple[str, str, str]] = [
    ("snr",               "snr_db",                          "{:.2f}"),
    ("srmr",              "srmr",                            "{:.4f}"),
    ("f0_mean",           "f0_mean_hz",                      "{:.2f}"),
    ("f0_sd",             "f0_sd_hz",                        "{:.2f}"),
    ("speaking_rate",     "praat_speaking_rate_syl_sec",     "{:.3f}"),
    ("pause_count",       "praat_pause_count",               "{:d}"),
    ("pause_rate",        "praat_pause_rate_per_min",        "{:.3f}"),
    ("overlap_ratio",     "overlap_ratio",                   "{:.4f}"),
]

N_FEATURES: int = len(SUPERVISED_FEATURES)  # 8


# Per-feature scales used to NORMALIZE the auxiliary-head MSE.
# Without normalization, F0 (typical magnitude ~150 Hz) dominates the squared-error
# sum 1000x over features like overlap_ratio (~0.5). Each scale is roughly the typical
# absolute value of that feature on real Libri2Mix clips; dividing (pred - gt) by the
# scale gives a unit-free relative-error term so every feature contributes ~equally.
# Order MUST match SUPERVISED_FEATURES.
FEATURE_SCALES: tuple[float, ...] = (
    5.0,    # snr  (dB, typical ~15)
    2.0,    # srmr  (typical ~5)
    50.0,   # f0_mean  (Hz, typical ~150)
    20.0,   # f0_sd  (Hz, typical ~40)
    2.0,    # speaking_rate  (syl/sec, typical ~6)
    3.0,    # pause_count  (count, typical ~3)
    5.0,    # pause_rate  (per min, typical ~10)
    0.3,    # overlap_ratio  (typical ~0.5)
)
assert len(FEATURE_SCALES) == N_FEATURES, "FEATURE_SCALES length must match SUPERVISED_FEATURES"

# Features that are *integers in nature* — pause_count is the only one in the
# trimmed catalog. Used by build_nums_target to cast before formatting.
_INT_FEATURES = {"pause_count"}

# Features whose 0.0 value is a *genuine zero*, not a "missing" signal.
# A clip with no pauses really has pause_count=0; a clip with no overlap really
# has overlap_ratio=0.0. These should NOT be replaced with "na" when zero-valued.
_GENUINE_ZERO_FEATURES = {
    "overlap_ratio", "pause_count", "pause_rate",
}


def _is_missing(val) -> bool:
    """NaN / None / empty string / 'nan'-like strings → treated as missing."""
    if val is None:
        return True
    if isinstance(val, str):
        s = val.strip().lower()
        return s in ("", "nan", "n/a", "na", "none")
    if isinstance(val, float):
        return math.isnan(val)
    return False


def _to_float(val):
    """Coerce a CSV cell value to float; returns float('nan') if missing/unparseable."""
    if _is_missing(val):
        return float("nan")
    try:
        return float(val)
    except (TypeError, ValueError):
        return float("nan")


def build_nums_target(row: dict) -> str:
    """Build the bare-numbers training target for B-full's forward A.

    Args:
        row: dict mapping CSV column name → raw value (string or already-parsed float).

    Returns:
        Fixed-order space-separated string like
            "snr=15.66 hnr=8.34 f0_mean=152.46 f0_sd=53.18 ... duration=10.435"
        with "na" substituted for missing measurements (e.g. silent clips have no F0).

    Note:
        - Integer-typed features (pause_count) are formatted without decimals.
        - "Genuine zero" features (overlap_ratio, pause_count, pause_rate) are emitted as
          their numeric value (0.0000 / 0 / 0.000) rather than "na" when zero, because
          zero is a real measurement for those.
    """
    parts: list[str] = []
    for short_name, csv_col, fmt in SUPERVISED_FEATURES:
        raw = row.get(csv_col)
        val = _to_float(raw)
        if math.isnan(val) and short_name not in _GENUINE_ZERO_FEATURES:
            parts.append(f"{short_name}=na")
        else:
            if math.isnan(val):
                # Genuine-zero feature is missing in CSV → still emit 0 (extremely rare)
                val = 0.0
            if short_name in _INT_FEATURES:
                parts.append(f"{short_name}={int(round(val))}")
            else:
                parts.append(f"{short_name}={fmt.format(val)}")
    return " ".join(parts)


def extract_scalars(row: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract the (13,) scalar tensor + (13,) bool mask for the aux regression head.

    Returns:
        scalars: float32 tensor of shape (13,) — values, with 0.0 substituted for missing.
        mask:    bool tensor of shape (13,) — True where the original CSV value was present.

    The mask is used by compute_loss to zero out MSE contribution from missing slots,
    so the aux head isn't penalized for "this clip has no F0".
    """
    scalars = torch.zeros(N_FEATURES, dtype=torch.float32)
    mask = torch.zeros(N_FEATURES, dtype=torch.bool)
    for i, (short_name, csv_col, _fmt) in enumerate(SUPERVISED_FEATURES):
        raw = row.get(csv_col)
        val = _to_float(raw)
        if math.isnan(val):
            if short_name in _GENUINE_ZERO_FEATURES:
                scalars[i] = 0.0
                mask[i] = True   # genuine zero is a real measurement
            else:
                scalars[i] = 0.0
                mask[i] = False  # measurement was missing
        else:
            scalars[i] = val
            mask[i] = True
    return scalars, mask
