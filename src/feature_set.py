"""Canonical 12-feature list for the new-project multi-task training and the aux regression head.

Single source of truth for:
  - The numerical target string used in forward A ("snr=15.66 srmr=4.5 ...").
  - The (B, 12) scalar tensor + mask used by the aux regression head's MSE.

The order matches the canonical descriptions builder
(scripts/build_canonical_descriptions.py), which emits the 12 features in this
fixed sequence: snr, srmr, hnr, f0_mean, f0_sd, jitter, shimmer, speaking_rate,
articulation_rate, pause_count, pause_rate, overlap_ratio.

Update history:
  - 2026-05-11: trimmed from 13 → 7 (drop f0_sd, jitter, shimmer, srmr,
    articulation_rate, pause_rate; keep duration + hnr).
  - 2026-05-12: realigned to the section catalog (8 features). Reverb (srmr),
    pitch SD (f0_sd), and pause_rate are restored because each has its own
    section in the EMNLP design and needs scalar supervision. duration and hnr
    are dropped — duration is an intro sentence outside any section; hnr was
    grouped under voice_quality which we cut to avoid a redundant attention
    figure with pitch.
  - 2026-06-24: EXPANDED 8 → 12 for the new full-feature run. Re-added hnr,
    jitter, shimmer, articulation_rate to match the canonical 12-feature
    descriptions (data/descriptions_canonical_{train,dev,test}.json). The order
    is realigned to the canonical builder: voice-quality scalars (hnr, jitter,
    shimmer) sit next to pitch (f0_mean, f0_sd); articulation_rate next to
    speaking_rate. The aux head dim (adapter.N_AUX_FEATURES) reads N_FEATURES so
    it auto-tracks this list.
"""

from __future__ import annotations

import math

import torch


# (short_name, csv_column, format_string)
# Order matches the canonical descriptions builder (scripts/build_canonical_descriptions.py):
#   snr, srmr, hnr, f0_mean, f0_sd, jitter, shimmer, speaking_rate,
#   articulation_rate, pause_count, pause_rate, overlap_ratio.
SUPERVISED_FEATURES: list[tuple[str, str, str]] = [
    ("snr",               "snr_db",                          "{:.2f}"),
    ("srmr",              "srmr",                            "{:.4f}"),
    ("hnr",               "hnr_db",                          "{:.2f}"),
    ("f0_mean",           "f0_mean_hz",                      "{:.2f}"),
    ("f0_sd",             "f0_sd_hz",                        "{:.2f}"),
    ("jitter",            "jitter_local_pct",                "{:.2f}"),
    ("shimmer",           "shimmer_pct",                     "{:.2f}"),
    ("speaking_rate",     "praat_speaking_rate_syl_sec",     "{:.3f}"),
    ("articulation_rate", "praat_articulation_rate_syl_sec", "{:.3f}"),
    ("pause_count",       "praat_pause_count",               "{:d}"),
    ("pause_rate",        "praat_pause_rate_per_min",        "{:.3f}"),
    ("overlap_ratio",     "overlap_ratio",                   "{:.4f}"),
]

N_FEATURES: int = len(SUPERVISED_FEATURES)  # 12


# Per-feature scales used to NORMALIZE the auxiliary-head MSE.
# Without normalization, F0 (typical magnitude ~150 Hz) dominates the squared-error
# sum 1000x over features like overlap_ratio (~0.5). Each scale is roughly the typical
# absolute value of that feature on real Libri2Mix clips; dividing (pred - gt) by the
# scale gives a unit-free relative-error term so every feature contributes ~equally.
# Order MUST match SUPERVISED_FEATURES.
FEATURE_SCALES: tuple[float, ...] = (
    5.0,    # snr  (dB, typical ~15)
    2.0,    # srmr  (typical ~5)
    20.0,   # hnr  (dB, typical ~15-20)
    50.0,   # f0_mean  (Hz, typical ~150)
    20.0,   # f0_sd  (Hz, typical ~40)
    2.0,    # jitter  (pct, typical ~1-2)
    8.0,    # shimmer  (pct, typical ~6-8)
    2.0,    # speaking_rate  (syl/sec, typical ~6)
    6.0,    # articulation_rate  (syl/sec, typical ~6)
    3.0,    # pause_count  (count, typical ~3)
    5.0,    # pause_rate  (per min, typical ~10)
    0.3,    # overlap_ratio  (typical ~0.5)
)
assert len(FEATURE_SCALES) == N_FEATURES, "FEATURE_SCALES length must match SUPERVISED_FEATURES"


# ── Observability classification (paper pivot: "Observability-Aware Description") ──
# Which features the signal can physically support a number for. On a 2-speaker mix,
# single-speaker pitch (f0_mean / f0_sd) is unrecoverable — pitch is the confirmed
# ill-posed case. SRMR is NOT ill-posed (it does not recover on a clean stem either),
# so it is treated as recoverable. These sets drive the reliability/abstention head's
# evaluation: the ILL_POSED features are where the model should band/abstain under
# overlap, and the RECOVERABLE features are the ones SFS keeps scoring as numbers.
#
# Membership is by short_name (col 0 of SUPERVISED_FEATURES). Anything not listed in
# ILL_POSED is recoverable by default. These are LABELS only — they do not change any
# training math unless an experiment explicitly consumes them (e.g. risk-coverage
# stratification), so adding them is a no-op for existing runs.
# 2026-06-24: extended to the 12-feature set. The voice-quality scalars (hnr,
# jitter, shimmer) join pitch (f0_mean, f0_sd) as ILL_POSED: they are derived from
# a single speaker's periodicity/cycle-to-cycle perturbation, which a 2-speaker mix
# corrupts the same way it corrupts pitch — the model should abstain on them under
# overlap. articulation_rate joins speaking_rate as RECOVERABLE (rate/timing
# features survive mixing well enough to keep emitting a number).
RECOVERABLE_FEATURES: frozenset[str] = frozenset({
    "snr", "srmr", "speaking_rate", "articulation_rate",
    "pause_count", "pause_rate", "overlap_ratio",
})
ILL_POSED_UNDER_OVERLAP_FEATURES: frozenset[str] = frozenset({
    "f0_mean", "f0_sd", "jitter", "shimmer", "hnr",
})

# Per-feature index lookups, in SUPERVISED_FEATURES order, for the reliability head /
# risk-coverage eval. Sanity: every short_name is classified exactly once.
FEATURE_NAMES: tuple[str, ...] = tuple(name for name, _csv, _fmt in SUPERVISED_FEATURES)
assert (RECOVERABLE_FEATURES | ILL_POSED_UNDER_OVERLAP_FEATURES) == frozenset(FEATURE_NAMES), (
    "every supervised feature must be classified recoverable XOR ill-posed"
)
assert not (RECOVERABLE_FEATURES & ILL_POSED_UNDER_OVERLAP_FEATURES), (
    "a feature cannot be both recoverable and ill-posed"
)


def recoverable_mask() -> "torch.Tensor":
    """(N_FEATURES,) bool tensor, True where the feature is recoverable from the mix.

    Useful as a constant abstention prior or for stratifying the risk-coverage curve
    by observability class. Order matches SUPERVISED_FEATURES.
    """
    return torch.tensor(
        [name in RECOVERABLE_FEATURES for name in FEATURE_NAMES], dtype=torch.bool,
    )

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
    """Extract the (N_FEATURES,) scalar tensor + (N_FEATURES,) bool mask for the aux head.

    Returns:
        scalars: float32 tensor of shape (N_FEATURES,) — values, 0.0 substituted for missing.
        mask:    bool tensor of shape (N_FEATURES,) — True where the CSV value was present.

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
