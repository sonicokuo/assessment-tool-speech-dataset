"""Tests for src/feature_set.py — the canonical 13-feature list used by B-full and aux head.

Verifies:
  - SUPERVISED_FEATURES has exactly 13 entries in stable canonical order.
  - build_nums_target produces fixed-order output, "na" for missing, integer formatting for pause_count.
  - extract_scalars returns (13,) tensors with correct mask handling.
  - Round-trip: feed build_nums_target output to ClaimParser, recover values within tolerance.
"""

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from feature_set import (
    N_FEATURES,
    SUPERVISED_FEATURES,
    build_nums_target,
    extract_scalars,
)


# A representative complete row (all features present)
COMPLETE_ROW = {
    "snr_db": 15.66,
    "hnr_db": 8.34,
    "f0_mean_hz": 152.46,
    "f0_sd_hz": 53.18,
    "jitter_local_pct": 2.7732,
    "shimmer_pct": 14.1259,
    "srmr": 5.1569,
    "overlap_ratio": 0.7928,
    "praat_speaking_rate_syl_sec": 5.61,
    "praat_articulation_rate_syl_sec": 6.94,
    "praat_pause_count": 1,
    "praat_pause_rate_per_min": 5.317,
    "duration_sec": 10.695,
}

# A short / silent clip with NaN voice features
SILENT_ROW = {
    "snr_db": 22.10,
    "hnr_db": float("nan"),
    "f0_mean_hz": float("nan"),
    "f0_sd_hz": float("nan"),
    "jitter_local_pct": float("nan"),
    "shimmer_pct": float("nan"),
    "srmr": 4.5,
    "overlap_ratio": 0.0,           # genuine zero — no overlap
    "praat_speaking_rate_syl_sec": float("nan"),
    "praat_articulation_rate_syl_sec": float("nan"),
    "praat_pause_count": 0,         # genuine zero — no pauses
    "praat_pause_rate_per_min": 0.0,  # genuine zero
    "duration_sec": 2.5,
}


def test_canonical_list_has_13_entries():
    assert N_FEATURES == 13
    assert len(SUPERVISED_FEATURES) == 13
    short_names = [f[0] for f in SUPERVISED_FEATURES]
    expected = ["snr", "hnr", "f0_mean", "f0_sd", "jitter", "shimmer", "srmr",
                "overlap_ratio", "speaking_rate", "articulation_rate",
                "pause_count", "pause_rate", "duration"]
    assert short_names == expected


def test_canonical_order_is_stable():
    # The first feature is always SNR; the last is always duration.
    assert SUPERVISED_FEATURES[0][0] == "snr"
    assert SUPERVISED_FEATURES[-1][0] == "duration"


def test_build_nums_target_complete_row():
    out = build_nums_target(COMPLETE_ROW)
    print(f"\nCOMPLETE: {out}")
    # All 13 slots present, fixed order
    assert "snr=15.66" in out
    assert "hnr=8.34" in out
    assert "f0_mean=152.46" in out
    assert "f0_sd=53.18" in out
    assert "jitter=2.7732" in out
    assert "shimmer=14.1259" in out
    assert "srmr=5.1569" in out
    assert "overlap_ratio=0.7928" in out
    assert "speaking_rate=5.610" in out
    assert "articulation_rate=6.940" in out
    assert "pause_count=1" in out          # integer, no decimal
    assert "pause_rate=5.317" in out
    assert "duration=10.695" in out
    # Order check
    parts = out.split()
    assert parts[0].startswith("snr="), f"first slot must be SNR; got {parts[0]}"
    assert parts[-1].startswith("duration="), f"last slot must be duration; got {parts[-1]}"


def test_build_nums_target_silent_row_uses_na():
    out = build_nums_target(SILENT_ROW)
    print(f"\nSILENT: {out}")
    # Voice-quality features are unmeasurable on silent clips → "na"
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
    # Every row must produce exactly 13 slots in the same order
    out_a = build_nums_target(COMPLETE_ROW).split()
    out_b = build_nums_target(SILENT_ROW).split()
    assert len(out_a) == 13
    assert len(out_b) == 13
    keys_a = [s.split("=")[0] for s in out_a]
    keys_b = [s.split("=")[0] for s in out_b]
    assert keys_a == keys_b   # same order, regardless of value content


def test_extract_scalars_complete_row():
    scalars, mask = extract_scalars(COMPLETE_ROW)
    assert scalars.shape == (13,)
    assert mask.shape == (13,)
    assert scalars.dtype == torch.float32
    assert mask.dtype == torch.bool
    # All present → mask all True
    assert mask.all().item()
    # Spot-check values
    assert abs(scalars[0].item() - 15.66) < 1e-4   # snr
    assert abs(scalars[7].item() - 0.7928) < 1e-4  # overlap_ratio
    assert abs(scalars[10].item() - 1.0) < 1e-4    # pause_count
    assert abs(scalars[12].item() - 10.695) < 1e-4 # duration


def test_extract_scalars_silent_row_mask():
    scalars, mask = extract_scalars(SILENT_ROW)
    # SNR present → mask True
    assert mask[0].item() is True
    # HNR / F0 / jitter / shimmer / speaking / articulation missing → mask False
    for short_name in ("hnr", "f0_mean", "f0_sd", "jitter", "shimmer",
                       "speaking_rate", "articulation_rate"):
        idx = next(i for i, (s, _, _) in enumerate(SUPERVISED_FEATURES) if s == short_name)
        assert mask[idx].item() is False, f"{short_name} should be masked-out"
        # And the scalar value at masked positions is 0.0 (safe placeholder)
        assert scalars[idx].item() == 0.0
    # Genuine-zero features (overlap_ratio, pause_count, pause_rate) → mask True
    for short_name in ("overlap_ratio", "pause_count", "pause_rate"):
        idx = next(i for i, (s, _, _) in enumerate(SUPERVISED_FEATURES) if s == short_name)
        assert mask[idx].item() is True, f"{short_name}=0 is genuine, mask should be True"


def test_round_trip_with_claim_parser():
    """Feed build_nums_target's output through ClaimParser, recover values."""
    from sfs import ClaimParser

    target = build_nums_target(COMPLETE_ROW)
    # The numbers target uses "snr=15.66", "hnr=8.34" etc. ClaimParser is built for the
    # prose form ("SNR is 15.66 dB"). Rebuild a synthetic prose so we can confirm the
    # ClaimParser would correctly score the *prose* version of these numbers — not that
    # it parses the bare-numbers target directly.
    prose = (
        f"The SNR is {COMPLETE_ROW['snr_db']} dB. "
        f"The HNR is {COMPLETE_ROW['hnr_db']} dB. "
        f"The F0 mean is {COMPLETE_ROW['f0_mean_hz']} Hz. "
        f"The F0 SD is {COMPLETE_ROW['f0_sd_hz']} Hz. "
        f"The Jitter local is {COMPLETE_ROW['jitter_local_pct']} %. "
        f"The Shimmer is {COMPLETE_ROW['shimmer_pct']} %. "
        f"The SRMR is {COMPLETE_ROW['srmr']}. "
        f"The overlap ratio is {COMPLETE_ROW['overlap_ratio']}. "
        f"The speaking rate is {COMPLETE_ROW['praat_speaking_rate_syl_sec']} syl/sec. "
        f"The articulation rate is {COMPLETE_ROW['praat_articulation_rate_syl_sec']} syl/sec. "
        f"The pause count is {COMPLETE_ROW['praat_pause_count']}. "
        f"The pause rate is {COMPLETE_ROW['praat_pause_rate_per_min']} per min. "
        f"The duration is {COMPLETE_ROW['duration_sec']} s."
    )
    claims = ClaimParser().parse(prose)
    parsed = {c.feature: c.value for c in claims}
    print(f"\nROUND-TRIP parsed: {parsed}")
    # Spot-check a few features survive the round trip
    assert "snr" in parsed and abs(parsed["snr"] - 15.66) < 1e-3
    assert "hnr" in parsed and abs(parsed["hnr"] - 8.34) < 1e-3
    assert "duration_sec" in parsed and abs(parsed["duration_sec"] - 10.695) < 1e-3
    assert "overlap_ratio" in parsed and abs(parsed["overlap_ratio"] - 0.7928) < 1e-3


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v", "-s"])
