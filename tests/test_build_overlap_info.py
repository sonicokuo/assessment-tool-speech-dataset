"""Unit tests for preprocess.build_overlap_info — focused on defensive clamping
against the pre-fix pyannote OOB issue.

Run from repo root:
    python -m pytest tests/test_build_overlap_info.py -v
"""
from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from preprocess import build_overlap_info, WAVLM_HOP_SAMPLES


def test_empty_returns_zeros():
    info, segs = build_overlap_info("", 0.0, T=10, sample_rate=16000)
    assert info.shape == (10, 4)
    assert torch.allclose(info, torch.zeros(10, 4))
    assert segs == []


def test_fully_oob_segment_dropped():
    # Clip is 200 frames * 320 samples = 64000 samples = 4.0 s.
    # Segment 80000-120000 (5.0-7.5 s) is fully past the clip end.
    info, segs = build_overlap_info("80000-120000", 0.0, T=200, sample_rate=16000)
    assert torch.allclose(info, torch.zeros(200, 4)), "expected zero overlap_info"
    assert segs == [], "fully-OOB segment must not appear in segments_sec"


def test_partial_oob_segment_clamped():
    # Clip 200 frames -> 64000 samples = 4.0 s. Segment 56000-80000 (3.5-5.0s)
    # has its end past the clip. Expect: clamped to 56000-64000 (3.5-4.0s),
    # duration_s should be 0.5, segments_sec should be (3.5, 4.0).
    info, segs = build_overlap_info("56000-80000", 0.0, T=200, sample_rate=16000)
    assert segs == [(3.5, 4.0)], f"got {segs}"
    # col 1 (segment_duration_s) should be 0.5 inside the segment, NOT 1.5.
    in_segment_rows = info[info[:, 0] > 0]
    assert in_segment_rows.shape[0] > 0
    durations = in_segment_rows[:, 1].unique().tolist()
    assert durations == [0.5], f"col1 not clamped: got durations {durations}"


def test_reversed_segment_swapped():
    # 48000-16000 (3.0s-1.0s reversed) should be swapped to 16000-48000.
    info, segs = build_overlap_info("48000-16000", 0.0, T=200, sample_rate=16000)
    assert segs == [(1.0, 3.0)], f"got {segs}"


def test_valid_in_clip_segment_unchanged():
    # 16000-48000 = 1.0-3.0s in a 4.0s clip. No clamping should happen.
    info, segs = build_overlap_info("16000-48000", 0.0, T=200, sample_rate=16000)
    assert segs == [(1.0, 3.0)], f"got {segs}"
    in_segment_rows = info[info[:, 0] > 0]
    durations = in_segment_rows[:, 1].unique().tolist()
    assert durations == [2.0], f"got durations {durations}"


def test_multi_segment_mixed_valid_oob():
    # 16000-32000 valid (1.0-2.0s), 80000-120000 fully OOB on a 4.0s clip.
    info, segs = build_overlap_info("16000-32000;80000-120000", 0.0,
                                     T=200, sample_rate=16000)
    assert segs == [(1.0, 2.0)], f"got {segs}"


def test_col2_ramps_correctly_in_clamped_segment():
    # Clamped 3.5-4.0s -> 25 frames (frames 175 to 200). col 2 should go 0->1.
    info, _ = build_overlap_info("56000-80000", 0.0, T=200, sample_rate=16000)
    in_segment_mask = info[:, 0] > 0
    # Earliest in-segment frame should have col2 == 0 (or very close).
    in_segment_indices = torch.where(in_segment_mask)[0]
    assert len(in_segment_indices) > 0
    assert info[in_segment_indices[0], 2].item() == 0.0
    assert info[in_segment_indices[-1], 2].item() == 1.0


def test_col0_is_one_inside_segment_zero_outside():
    info, _ = build_overlap_info("16000-32000", 0.0, T=200, sample_rate=16000)
    col0 = info[:, 0]
    # Inside frames 50-100 should be 1, outside should be 0.
    assert col0[50:100].sum().item() > 0
    assert col0[0:30].sum().item() == 0
    assert col0[110:200].sum().item() == 0
