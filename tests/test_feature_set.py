"""Tests for src/feature_set.py — the canonical 12-feature list used by multi-task + aux head.

Verifies:
  - SUPERVISED_FEATURES has exactly 12 entries in the canonical-builder order.
  - build_nums_target produces fixed-order output, "na" for missing, integer formatting for pause_count.
  - extract_scalars returns (12,) tensors with correct mask handling.
  - The recoverable / ill-posed observability sets partition all 12 features (XOR).
  - Round-trip: feed build_nums_target output to ClaimParser, recover values within tolerance.
"""

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from feature_set import (
    FEATURE_NAMES,
    FEATURE_SCALES,
    ILL_POSED_UNDER_OVERLAP_FEATURES,
    N_FEATURES,
    RECOVERABLE_FEATURES,
    SUPERVISED_FEATURES,
    build_nums_target,
    extract_scalars,
)


# A representative complete row (all 12 features present). duration_sec is in the CSV
# but is NOT a supervised feature — it must be ignored.
COMPLETE_ROW = {
    "snr_db": 15.66,
    "srmr": 5.1569,
    "hnr_db": 8.34,
    "f0_mean_hz": 152.46,
    "f0_sd_hz": 53.18,
    "jitter_local_pct": 2.7732,
    "shimmer_pct": 14.1259,
    "praat_speaking_rate_syl_sec": 5.61,
    "praat_articulation_rate_syl_sec": 6.94,
    "praat_pause_count": 1,
    "praat_pause_rate_per_min": 5.317,
    "overlap_ratio": 0.7928,
    # In the CSV but NOT a supervised feature — must NOT appear in the nums target.
    "duration_sec": 10.695,
}

# A heavily-overlapped clip: pitch + voice-quality features are unmeasurable (abstain),
# rate/pause features still present.
SILENT_ROW = {
    "snr_db": 22.10,
    "srmr": 4.5,
    "hnr_db": float("nan"),
    "f0_mean_hz": float("nan"),
    "f0_sd_hz": float("nan"),
    "jitter_local_pct": float("nan"),
    "shimmer_pct": float("nan"),
    "praat_speaking_rate_syl_sec": float("nan"),
    "praat_articulation_rate_syl_sec": float("nan"),
    "praat_pause_count": 0,             # genuine zero — no pauses
    "praat_pause_rate_per_min": 0.0,    # genuine zero
    "overlap_ratio": 0.0,               # genuine zero — no overlap
}


CANONICAL_ORDER = [
    "snr", "srmr", "hnr", "f0_mean", "f0_sd", "jitter", "shimmer",
    "speaking_rate", "articulation_rate", "pause_count", "pause_rate", "overlap_ratio",
]


def test_canonical_list_has_12_entries():
    assert N_FEATURES == 12
    assert len(SUPERVISED_FEATURES) == 12
    short_names = [f[0] for f in SUPERVISED_FEATURES]
    assert short_names == CANONICAL_ORDER


def test_feature_scales_length_matches():
    assert len(FEATURE_SCALES) == N_FEATURES == 12


def test_canonical_order_is_stable():
    # First feature is snr; last is overlap_ratio (matches the canonical builder).
    assert SUPERVISED_FEATURES[0][0] == "snr"
    assert SUPERVISED_FEATURES[-1][0] == "overlap_ratio"
    # The 4 re-added features sit at their canonical positions.
    assert SUPERVISED_FEATURES[2][0] == "hnr"
    assert SUPERVISED_FEATURES[5][0] == "jitter"
    assert SUPERVISED_FEATURES[6][0] == "shimmer"
    assert SUPERVISED_FEATURES[8][0] == "articulation_rate"


def test_observability_sets_partition_all_features():
    """recoverable XOR ill-posed must exactly cover the 12 features (mandate)."""
    names = frozenset(FEATURE_NAMES)
    assert len(names) == 12
    # Union covers everything, intersection is empty (XOR).
    assert (RECOVERABLE_FEATURES | ILL_POSED_UNDER_OVERLAP_FEATURES) == names
    assert not (RECOVERABLE_FEATURES & ILL_POSED_UNDER_OVERLAP_FEATURES)
    # The exact mandated membership.
    assert RECOVERABLE_FEATURES == frozenset({
        "snr", "srmr", "speaking_rate", "articulation_rate",
        "pause_count", "pause_rate", "overlap_ratio",
    })
    assert ILL_POSED_UNDER_OVERLAP_FEATURES == frozenset({
        "f0_mean", "f0_sd", "jitter", "shimmer", "hnr",
    })


def test_build_nums_target_complete_row():
    out = build_nums_target(COMPLETE_ROW)
    print(f"\nCOMPLETE: {out}")
    # All 12 slots present, fixed order, including the 4 re-added features.
    assert "snr=15.66" in out
    assert "srmr=5.1569" in out
    assert "hnr=8.34" in out
    assert "f0_mean=152.46" in out
    assert "f0_sd=53.18" in out
    assert "jitter=2.77" in out
    assert "shimmer=14.13" in out
    assert "speaking_rate=5.610" in out
    assert "articulation_rate=6.940" in out
    assert "pause_count=1" in out          # integer, no decimal
    assert "pause_rate=5.317" in out
    assert "overlap_ratio=0.7928" in out
    # duration is in the CSV but NOT a supervised feature.
    assert "duration=" not in out
    # Order check
    parts = out.split()
    assert parts[0].startswith("snr="), f"first slot must be snr; got {parts[0]}"
    assert parts[-1].startswith("overlap_ratio="), f"last slot must be overlap_ratio; got {parts[-1]}"


def test_build_nums_target_silent_row_uses_na():
    out = build_nums_target(SILENT_ROW)
    print(f"\nSILENT: {out}")
    # Pitch + voice-quality features are unmeasurable under heavy overlap → "na"
    assert "hnr=na" in out
    assert "f0_mean=na" in out
    assert "f0_sd=na" in out
    assert "jitter=na" in out
    assert "shimmer=na" in out
    assert "speaking_rate=na" in out
    assert "articulation_rate=na" in out
    # Genuine zeros are NOT na
    assert "overlap_ratio=0.0000" in out
    assert "pause_count=0" in out
    assert "pause_rate=0.000" in out


def test_build_nums_target_fixed_order_across_rows():
    # Every row must produce exactly 12 slots in the same order
    out_a = build_nums_target(COMPLETE_ROW).split()
    out_b = build_nums_target(SILENT_ROW).split()
    assert len(out_a) == 12
    assert len(out_b) == 12
    keys_a = [s.split("=")[0] for s in out_a]
    keys_b = [s.split("=")[0] for s in out_b]
    assert keys_a == keys_b   # same order, regardless of value content
    assert keys_a == CANONICAL_ORDER


def test_extract_scalars_complete_row():
    scalars, mask = extract_scalars(COMPLETE_ROW)
    assert scalars.shape == (12,)
    assert mask.shape == (12,)
    assert scalars.dtype == torch.float32
    assert mask.dtype == torch.bool
    # All 12 present → mask all True
    assert mask.all().item()
    # Spot-check values at known indices (matches the order in SUPERVISED_FEATURES)
    assert abs(scalars[0].item() - 15.66) < 1e-4    # snr
    assert abs(scalars[1].item() - 5.1569) < 1e-4   # srmr
    assert abs(scalars[2].item() - 8.34) < 1e-4     # hnr
    assert abs(scalars[9].item() - 1.0) < 1e-4      # pause_count
    assert abs(scalars[11].item() - 0.7928) < 1e-4  # overlap_ratio


def test_extract_scalars_silent_row_mask():
    scalars, mask = extract_scalars(SILENT_ROW)
    # SNR + SRMR present → mask True at those indices.
    for short_name in ("snr", "srmr"):
        idx = next(i for i, (s, _, _) in enumerate(SUPERVISED_FEATURES) if s == short_name)
        assert mask[idx].item() is True
    # Pitch + voice-quality + rate missing → mask False (all the ill-posed features)
    for short_name in ("hnr", "f0_mean", "f0_sd", "jitter", "shimmer",
                       "speaking_rate", "articulation_rate"):
        idx = next(i for i, (s, _, _) in enumerate(SUPERVISED_FEATURES) if s == short_name)
        assert mask[idx].item() is False, f"{short_name} should be masked-out"
        # And the scalar value at masked positions is 0.0 (safe placeholder)
        assert scalars[idx].item() == 0.0
    # Genuine-zero features → mask True
    for short_name in ("overlap_ratio", "pause_count", "pause_rate"):
        idx = next(i for i, (s, _, _) in enumerate(SUPERVISED_FEATURES) if s == short_name)
        assert mask[idx].item() is True, f"{short_name}=0 is genuine, mask should be True"


def test_round_trip_with_claim_parser():
    """Feed prose containing the 8 features through ClaimParser, recover values."""
    from sfs import ClaimParser

    prose = (
        f"The SNR is {COMPLETE_ROW['snr_db']} dB. "
        f"The SRMR is {COMPLETE_ROW['srmr']}. "
        f"The F0 mean is {COMPLETE_ROW['f0_mean_hz']} Hz. "
        f"The F0 SD is {COMPLETE_ROW['f0_sd_hz']} Hz. "
        f"The speaking rate is {COMPLETE_ROW['praat_speaking_rate_syl_sec']} syl/sec. "
        f"The pause count is {COMPLETE_ROW['praat_pause_count']}. "
        f"The pause rate is {COMPLETE_ROW['praat_pause_rate_per_min']} per min. "
        f"The overlap ratio is {COMPLETE_ROW['overlap_ratio']}."
    )
    claims = ClaimParser().parse(prose)
    parsed = {c.feature: c.value for c in claims}
    print(f"\nROUND-TRIP parsed: {parsed}")
    # Spot-check a few features survive the round trip
    assert "snr" in parsed and abs(parsed["snr"] - 15.66) < 1e-3
    assert "srmr" in parsed and abs(parsed["srmr"] - 5.1569) < 1e-3
    assert "overlap_ratio" in parsed and abs(parsed["overlap_ratio"] - 0.7928) < 1e-3


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v", "-s"])
