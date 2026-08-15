"""Canonical 11-feature list for the new-project multi-task training and the aux regression head.

Single source of truth for:
  - The numerical target string used in forward A ("snr=15.66 srmr=4.5 ...").
  - The (B, 11) scalar tensor + mask used by the aux regression head's MSE.

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

import csv
import math
import os

import torch


# (short_name, csv_column, format_string)
# Order matches the canonical descriptions builder (scripts/build_canonical_descriptions.py):
#   snr, srmr, hnr, f0_mean, f0_sd, jitter, shimmer, speaking_rate,
#
# NUMERIC SURFACE FORM (2026-07-16, F15): every format is a CONSTANT-WIDTH two-decimal
# "{:.2f}". Mixed decimal widths (srmr "3.2667", jitter/shimmer "1.7458" vs snr "15.66")
# break digit-position/magnitude alignment for a digit-by-digit tokenizer — the same
# fractional digit lands at a different token position per feature, diluting the
# audio→digit gradient (Singh & Strouse, arXiv:2402.14903; R4 in the 2026-07-13
# training-method memo). This changes TRAINING TARGETS: it takes effect at the next
# dataset rebuild + retrain, and has no effect on already-built targets or running jobs.
# pause_count keeps integer emission via _INT_FEATURES (its fmt entry is never used).
SUPERVISED_FEATURES: list[tuple[str, str, str]] = [
    ("snr",               "snr_db",                          "{:.2f}"),
    ("srmr",              "srmr",                            "{:.2f}"),
    ("f0_mean",           "f0_mean_hz",                      "{:.2f}"),
    ("f0_sd",             "f0_sd_hz",                        "{:.2f}"),
    ("speaking_rate",     "praat_speaking_rate_syl_sec",     "{:.2f}"),
    ("pause_count",       "praat_pause_count",               "{:.2f}"),  # int-cast wins (see _INT_FEATURES)
    ("pause_rate",        "praat_pause_rate_per_min",        "{:.2f}"),
    ("overlap_ratio",     "overlap_ratio",                   "{:.2f}"),
    ("jitter",            "jitter_local_pct",                "{:.2f}"),
    ("shimmer",           "shimmer",                         "{:.2f}"),
    ("hnr",               "hnr",                             "{:.2f}"),
]

N_FEATURES: int = len(SUPERVISED_FEATURES)  # 11 (voice patch 2026-06-24 dropped articulation_rate)


# Per-feature scales used to NORMALIZE the auxiliary-head MSE / heteroscedastic NLL.
# Without normalization, F0 (typical magnitude ~150 Hz) dominates the squared-error
# sum 1000x over features like overlap_ratio (~0.5). The correct normalizer is each
# feature's DISPERSION (so (pred-gt)/scale is a unit-free z-score-like error and every
# feature contributes ~equally), NOT its magnitude. These are the robust dispersion
# 1.4826*MAD measured on the corrected merged CSV (train-100, mix+s1clean, 2026-07-11);
# the earlier magnitude-based scales were miscalibrated to the corrected distribution
# (effective aux-loss weights ranged ~0.33x-4x across features — snr/f0/hnr were
# over-weighted, pause_count under-weighted). Order MUST match SUPERVISED_FEATURES.
FEATURE_SCALES: tuple[float, ...] = (
    15.0,   # snr  (dB; robust dispersion 1.4826*MAD ≈ 14.8; was 5.0 → 3x under-scaled)
    1.6,    # srmr  (≈ 1.56; was 2.0)
    44.0,   # f0_mean  (Hz; ≈ 44.2; was 50.0)
    23.0,   # f0_sd  (Hz; ≈ 22.7; was 20.0)
    0.75,   # speaking_rate  (syl/sec; ≈ 0.75; was 2.0 → 2.7x over-scaled)
    1.5,    # pause_count  (≈ 1.48; was 3.0)
    8.0,    # pause_rate  (per min; ≈ 7.92; was 5.0)
    0.35,   # overlap_ratio  (bimodal s1clean=0 / mix≈0.8; robust spread ≈ 0.35; was 0.3)
    0.6,    # jitter  (local %; ≈ 0.60; was 1.0)
    2.2,    # shimmer  (≈ 2.17; was 3.0)
    3.0,    # hnr  (dB; ≈ 3.01)
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
# a single speaker's periodicity/cycle-to-cycle perturbation, which a 2-speaker mix
# corrupts the same way it corrupts pitch — the model should abstain on them under
# overlap. articulation_rate joins speaking_rate as RECOVERABLE (rate/timing
# features survive mixing well enough to keep emitting a number).
RECOVERABLE_FEATURES: frozenset[str] = frozenset({
    "snr", "srmr", "speaking_rate", "pause_count", "pause_rate", "overlap_ratio",
})
ILL_POSED_UNDER_OVERLAP_FEATURES: frozenset[str] = frozenset({
    "f0_mean", "f0_sd", "jitter", "shimmer", "hnr", })

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


# ── F11: clean-f0 GT fail-loud validation (2026-07-15 audit risk 10) ─────────
# SUPERVISED_FEATURES maps ("f0_mean", "f0_mean_hz") and ("f0_sd", "f0_sd_hz") —
# noisy-NAMED (mixture-measured) columns. They are only safe as GT because the
# shipped CSVs were clean-substituted IN PLACE (scripts/make_clean_f0_csv.py
# overwrites f0_*_hz with clean-frame values, writing *_cleanf0.csv / *_cleangt.csv).
# One wrong --features_csv silently trains/evals on noisy mixture f0. These helpers
# let a call site fail loud instead.
#
# STRICT_CLEAN_F0: opt-in module flag. Default False = behavior byte-identical to
# before. When a dataset owner sets it True (data.feature_set.STRICT_CLEAN_F0 = True),
# build_nums_target / extract_scalars refuse to run until validate_f0_source() has
# been called once for the features CSV feeding the run.
STRICT_CLEAN_F0: bool = False
_F0_SOURCE_VALIDATED: bool = False

# Sentinel marker column convention: a CSV column named `f0_clean_substituted` whose
# value is 1 declares "f0_*_hz in this file hold clean-frame values".
_F0_SENTINEL_COLUMN = "f0_clean_substituted"
# Filename convention already in use: features_aug_train_cleangt.csv, dev_cleanf0.csv.
_F0_CLEAN_NAME_MARKERS = ("cleanf0", "cleangt")
_F0_CLEAN_NAMED_COLUMNS = ("f0_mean_hz_clean", "f0_sd_hz_clean")
_F0_NOISY_NAMED_COLUMNS = ("f0_mean_hz", "f0_sd_hz")


def assert_clean_f0_csv(csv_path_or_rows, filename: str | None = None) -> None:
    """Fail loud if a features CSV's f0 columns may hold noisy MIXTURE values.

    2026-07-15 audit risk 10: f0_mean_hz / f0_sd_hz are noisy-named columns that are
    only clean because make_clean_f0_csv.py substituted them in place; a config that
    points at a raw features CSV silently trains on ill-posed mixture f0.

    Accepts either a CSV path (str / os.PathLike) or already-loaded rows (a dict row
    or an iterable of dict rows; pass `filename` to enable the name-convention check).

    Passes when ANY of these hold:
      (a) explicitly clean-NAMED columns exist (f0_mean_hz_clean / f0_sd_hz_clean);
      (b) the sentinel column `f0_clean_substituted` exists with value 1;
      (c) the CSV filename contains 'cleanf0' or 'cleangt' (existing convention);
      (d) the CSV has no f0_mean_hz / f0_sd_hz column at all (no f0 GT to poison).

    Otherwise raises RuntimeError naming the file and the risk.
    """
    first_row: dict | None = None
    if isinstance(csv_path_or_rows, (str, os.PathLike)):
        path = os.fspath(csv_path_or_rows)
        if filename is None:
            filename = os.path.basename(path)
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            columns = list(reader.fieldnames or [])
            first_row = next(iter(reader), None)
    elif isinstance(csv_path_or_rows, dict):
        first_row = csv_path_or_rows
        columns = list(first_row.keys())
    else:
        rows = list(csv_path_or_rows)
        first_row = rows[0] if rows else None
        columns = list(first_row.keys()) if first_row else []

    # (d) no noisy-named f0 columns at all → nothing to validate.
    if not any(c in columns for c in _F0_NOISY_NAMED_COLUMNS):
        return
    # (a) explicitly clean-named columns present.
    if any(c in columns for c in _F0_CLEAN_NAMED_COLUMNS):
        return
    # (b) sentinel marker column with value 1 (column presence alone suffices when
    # the file has a header but no data rows).
    if _F0_SENTINEL_COLUMN in columns:
        if first_row is None:
            return
        sentinel = str(first_row.get(_F0_SENTINEL_COLUMN, "")).strip()
        if sentinel in ("1", "1.0", "true", "True"):
            return
    # (c) filename convention.
    if filename is not None and any(
        marker in os.path.basename(filename).lower() for marker in _F0_CLEAN_NAME_MARKERS
    ):
        return

    name = filename if filename is not None else "<in-memory rows>"
    raise RuntimeError(
        f"clean-f0 GT check FAILED for {name!r}: this CSV has noisy-NAMED f0 columns "
        f"(f0_mean_hz/f0_sd_hz) with no evidence of clean-frame substitution — no "
        f"f0_mean_hz_clean/f0_sd_hz_clean columns, no {_F0_SENTINEL_COLUMN}=1 sentinel "
        f"column, and the filename does not contain 'cleanf0'/'cleangt'. Using it as GT "
        f"silently trains/scores on ill-posed MIXTURE f0 (2026-07-15 audit risk 10). "
        f"Regenerate it with scripts/make_clean_f0_csv.py (or add the sentinel column) "
        f"before pointing --features_csv at it."
    )


def assert_supervised_columns_exist(fieldnames, *, strict: bool = True) -> list:
    """Fail loudly when a SUPERVISED_FEATURES csv_col is absent from the CSV header.

    THE SAME CLASS OF BUG THAT ERASED hnr AND shimmer FROM EVERY TRAINING TARGET
    (2026-07-28). `scripts/build_canonical_descriptions.py` asked for "hnr_db" /
    "shimmer_pct" while features_corrected_merged/*.csv had been regenerated as "hnr" /
    "shimmer". `row.get()` returns None identically for "column renamed" and "measurement
    missing", so the features vanished from 39,800 targets without one warning.

    This is the aux-head/nums-target side of the same lookup. A rename here would silently
    zero the MSE/NLL supervision for that feature and mask it out of the presence mask, so
    the abstention head would train on a feature it never actually sees -- indistinguishable
    from "this clip has no measurement". Better to stop the run than to burn a node on it.

    Escape hatch for legitimately partial CSVs (cross-domain sets such as AMI genuinely
    lack the temporal columns): set AQUA_ALLOW_MISSING_FEATURE_COLS=1, which downgrades
    this to a printed warning naming exactly which features go unsupervised.

    Returns the list of missing columns (empty when clean).
    """
    have = set(fieldnames or ())
    missing = [(name, col) for name, col, _fmt in SUPERVISED_FEATURES if col not in have]
    if not missing:
        return []
    detail = ", ".join(f"{n} -> {c!r}" for n, c in missing)
    if not strict or os.environ.get("AQUA_ALLOW_MISSING_FEATURE_COLS") == "1":
        print(f"[feature_set][WARNING] CSV header lacks {len(missing)} supervised column(s): "
              f"{detail}. These features will be UNSUPERVISED (masked out) for this run.")
        return [c for _n, c in missing]
    raise RuntimeError(
        f"features_csv is missing {len(missing)} SUPERVISED_FEATURES column(s): {detail}\n"
        f"CSV header: {sorted(have)}\n"
        "A renamed column is indistinguishable from a missing measurement, so these "
        "features would train with silently empty supervision (this is exactly how hnr and "
        "shimmer were lost from every target on 2026-07-28). Fix the column name in "
        "SUPERVISED_FEATURES, or set AQUA_ALLOW_MISSING_FEATURE_COLS=1 if the CSV is "
        "genuinely partial (e.g. a cross-domain set without the temporal features)."
    )


def validate_f0_source(csv_path_or_rows, filename: str | None = None) -> None:
    """One-shot call site for the dataset: run assert_clean_f0_csv and record success
    so the STRICT_CLEAN_F0 gate in build_nums_target / extract_scalars is satisfied."""
    global _F0_SOURCE_VALIDATED
    assert_clean_f0_csv(csv_path_or_rows, filename=filename)
    _F0_SOURCE_VALIDATED = True


def _strict_clean_f0_gate() -> None:
    """No-op unless STRICT_CLEAN_F0 is set (default off → behavior identical)."""
    if STRICT_CLEAN_F0 and not _F0_SOURCE_VALIDATED:
        raise RuntimeError(
            "feature_set.STRICT_CLEAN_F0 is enabled but validate_f0_source(features_csv) "
            "was never called — refusing to build f0 GT from a possibly-noisy CSV "
            "(f0_mean_hz/f0_sd_hz are mixture-named columns; 2026-07-15 audit risk 10)."
        )


# Overlap threshold at/above which the ill-posed (speaker-intrinsic) features are HEDGED
# (abstained). MUST match the canonical prose builder's abstain rule
# (scripts/build_canonical_descriptions.py: overlap_ratio >= 0.5) so the prose, the nums-CE
# target, and the aux-MSE target all agree on which features are abstained (the B3 fix for the
# "hedge-only-on-prose" bug). The NLL deliberately does NOT use this — see hedge_mask().
HEDGE_OVERLAP_TAU: float = 0.5


def _hedged_features(row: dict) -> frozenset[str]:
    """Set of feature short-names abstained for this clip: the ILL_POSED features when
    overlap_ratio >= HEDGE_OVERLAP_TAU, else empty. Recovered from the CSV with the same
    deterministic rule the canonical builder used — NEVER parsed from the prose."""
    ov = _to_float(row.get("overlap_ratio"))
    if math.isnan(ov) or ov < HEDGE_OVERLAP_TAU:
        return frozenset()
    return ILL_POSED_UNDER_OVERLAP_FEATURES


def hedge_mask(row: dict) -> "torch.Tensor":
    """(N_FEATURES,) bool tensor — True where the feature is HEDGED for this clip
    (ill-posed AND overlap_ratio >= HEDGE_OVERLAP_TAU). NEVER True for a recoverable/counting
    feature. Used to build the point-estimate (aux-MSE + nums-CE) mask so those channels stop
    fitting mix-noisy values on hedged clips. The heteroscedastic-NLL / sigma head does NOT use
    this mask (it keeps the presence mask so it still sees the hard pairs and learns high sigma
    there — the observability signal). Order matches SUPERVISED_FEATURES."""
    hedged = _hedged_features(row)
    return torch.tensor([name in hedged for name in FEATURE_NAMES], dtype=torch.bool)


def build_nums_target(row: dict, hedge: bool = False) -> str:
    """Build the bare-numbers training target for B-full's forward A.

    Args:
        row: dict mapping CSV column name → raw value (string or already-parsed float).
        hedge: when True, emit "na" for the ill-posed features on high-overlap clips
            (overlap_ratio >= HEDGE_OVERLAP_TAU), so the nums-CE target agrees with the hedged
            prose instead of training the mix-noisy F0/voice values (B3 fix). Recoverable
            features are always emitted. Default False = byte-identical to before.

    Returns:
        Fixed-order space-separated string like
            "snr=15.66 hnr=8.34 f0_mean=152.46 f0_sd=53.18 ..."
        with "na" substituted for missing measurements (e.g. silent clips have no F0) and,
        when hedge=True, for the abstained ill-posed features under heavy overlap.
        All non-integer values use the constant two-decimal surface form (F15).

    Note:
        - Integer-typed features (pause_count) are formatted without decimals.
        - "Genuine zero" features (overlap_ratio, pause_count, pause_rate) are emitted as
          their numeric value (0.00 / 0 / 0.00) rather than "na" when zero, because
          zero is a real measurement for those.
    """
    _strict_clean_f0_gate()
    hedged = _hedged_features(row) if hedge else frozenset()
    parts: list[str] = []
    for short_name, csv_col, fmt in SUPERVISED_FEATURES:
        if short_name in hedged:
            parts.append(f"{short_name}=na")
            continue
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
    _strict_clean_f0_gate()
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
