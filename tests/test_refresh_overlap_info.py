"""Tests for scripts/refresh_overlap_info.py.

Each test builds a synthetic .pt + CSV pair under a tmp_path, runs the
refresh, and asserts the documented invariants (clamp behavior, key
preservation, idempotency, missing-CSV error handling, empty-segments
zeroing).

Run from the repo root:
    python -m pytest tests/test_refresh_overlap_info.py -v
"""
from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

import torch

# conftest.py adds src/. We need scripts/ too for refresh_overlap_info.
HERE = Path(__file__).parent.resolve()
sys.path.insert(0, str(HERE.parent / "scripts"))

from refresh_overlap_info import run_full           # noqa: E402
from preprocess import build_overlap_info, WAVLM_HOP_SAMPLES  # noqa: E402


SR = 16000
HOP = WAVLM_HOP_SAMPLES   # 320


# ---------------------------------------------------------------------------
# Helpers — fabricate inputs the tests need
# ---------------------------------------------------------------------------
def _write_csv(path: Path, rows: list[dict]) -> None:
    """Write a minimal features CSV with just the columns refresh reads."""
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["filename", "overlap_segments", "overlap_ratio"])
        for r in rows:
            w.writerow([r["filename"], r["overlap_segments"], r["overlap_ratio"]])


def _make_pt(path: Path, T: int, overlap_info: torch.Tensor, *,
             beats_patches: torch.Tensor | None = None,
             beats_grid_meta: dict | None = None,
             filename: str = "abc.wav",
             audio_features: torch.Tensor | None = None,
             overlap_segments: list | None = None) -> None:
    """Create a .pt with the same shape as src/preprocess.py outputs.

    Extra keys (beats_patches, beats_grid_meta) are optional; when supplied
    they should survive the refresh byte-identical.
    """
    cached = {
        "audio_features": audio_features if audio_features is not None
                          else torch.zeros(T, 1024),
        "overlap_info": overlap_info,
        "overlap_segments": overlap_segments if overlap_segments is not None else [],
        "filename": filename,
    }
    if beats_patches is not None:
        cached["beats_patches"] = beats_patches
    if beats_grid_meta is not None:
        cached["beats_grid_meta"] = beats_grid_meta
    torch.save(cached, path)


def _bad_old_format_overlap_info(segs_str: str, T: int) -> torch.Tensor:
    """Reproduce the pre-de49d6a (un-clamped) build_overlap_info to fabricate
    a 'stale' overlap_info tensor with the inflated segment_duration_s the
    old code wrote. Frame-range was already clamped to T via min(T, ...), so
    only col 1 (= (e_samp - s_samp) / SR) takes the unclamped value.
    """
    info = torch.zeros(T, 4)
    if not segs_str:
        return info
    for seg in segs_str.split(";"):
        seg = seg.strip()
        if not seg or "-" not in seg:
            continue
        s_samp, e_samp = (int(x) for x in seg.split("-"))
        if e_samp < s_samp:
            s_samp, e_samp = e_samp, s_samp
        f_start = max(0, s_samp // HOP)
        f_end = min(T, (e_samp + HOP - 1) // HOP)
        if f_end <= f_start:
            continue
        n_frames = f_end - f_start
        duration_s = (e_samp - s_samp) / SR    # unclamped — that IS the bug
        info[f_start:f_end, 0] = 1.0
        info[f_start:f_end, 1] = duration_s
        info[f_start:f_end, 2] = torch.linspace(0.0, 1.0, n_frames)
    return info


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def test_oob_partial_clamp(tmp_path):
    """Partial-OOB pyannote segment: 56000-80000 in a T=200 (= 64000 samples =
    4.0 s) clip. The end (80000 = 5.0 s) is 1.0 s past the clip.
    OLD col 1 = (80000 - 56000) / 16000 = 1.5 s   (inflated)
    NEW col 1 = (64000 - 56000) / 16000 = 0.5 s   (clamped)
    """
    T = 200
    pt_path = tmp_path / "clip.pt"
    bad_info = _bad_old_format_overlap_info("56000-80000", T)
    # Sanity on the fabricated bad state:
    assert abs(bad_info[175:200, 1].max().item() - 1.5) < 1e-6, \
        "test setup invariant: bad_info col1 in OOB frames should be 1.5"
    _make_pt(pt_path, T, bad_info, filename="clip.wav")

    csv_path = tmp_path / "feat.csv"
    _write_csv(csv_path, [
        {"filename": "clip.wav", "overlap_segments": "56000-80000",
         "overlap_ratio": "0.1"},
    ])
    rc = run_full(tmp_path, csv_path)
    assert rc == 0

    out = torch.load(pt_path, weights_only=False)
    col1 = out["overlap_info"][:, 1]
    # All nonzero col1 values should now sit at the clamped duration 0.5 s.
    nonzero = col1[col1 != 0]
    assert nonzero.numel() > 0, "post-refresh: some frames should still be marked overlap"
    assert torch.all(torch.isclose(nonzero, torch.full_like(nonzero, 0.5))), \
        f"col1 nonzero values should all be 0.5, got distinct values: {sorted(set(nonzero.tolist()))}"
    # And no value past the clip duration should remain (col1 was the issue).
    assert col1.max().item() <= 4.0 + 1e-6


def test_oob_full_drop(tmp_path):
    """Fully-OOB pyannote segment: 80000-120000 (5.0-7.5 s) against a T=200
    (4.0 s) clip. The whole segment is past the clip end → overlap_info
    should be all zeros, overlap_segments should be [].
    """
    T = 200
    pt_path = tmp_path / "clip.pt"
    # Deliberately bogus pre-state — pretend an older buggy run wrote garbage.
    bogus_info = torch.full((T, 4), 0.5)
    _make_pt(pt_path, T, bogus_info, filename="clip.wav",
             overlap_segments=[(5.0, 7.5)])

    csv_path = tmp_path / "feat.csv"
    _write_csv(csv_path, [
        {"filename": "clip.wav", "overlap_segments": "80000-120000",
         "overlap_ratio": "0.1"},
    ])
    rc = run_full(tmp_path, csv_path)
    assert rc == 0

    out = torch.load(pt_path, weights_only=False)
    assert torch.all(out["overlap_info"] == 0), \
        "fully-OOB segment should refresh to all-zero overlap_info"
    assert out["overlap_segments"] == [], \
        f"fully-OOB: segments_sec should be empty list, got {out['overlap_segments']}"


def test_idempotent(tmp_path):
    """Two consecutive refreshes must produce a byte-identical file.

    Round 1 may rewrite the file (the .pt was created with a 'wrong' state
    if any). Round 2 must observe new == old and skip the write entirely,
    leaving the post-round-1 file untouched.
    """
    T = 200
    pt_path = tmp_path / "clip.pt"
    # Start with a clean, post-clamp overlap_info — refresh should be a no-op.
    clean_info, clean_segs = build_overlap_info("16000-32000", 0.1, T, sample_rate=SR)
    _make_pt(pt_path, T, clean_info, filename="clip.wav",
             overlap_segments=clean_segs)
    csv_path = tmp_path / "feat.csv"
    _write_csv(csv_path, [
        {"filename": "clip.wav", "overlap_segments": "16000-32000",
         "overlap_ratio": "0.1"},
    ])

    rc1 = run_full(tmp_path, csv_path)
    assert rc1 == 0
    bytes_after_first = pt_path.read_bytes()

    rc2 = run_full(tmp_path, csv_path)
    assert rc2 == 0
    bytes_after_second = pt_path.read_bytes()

    assert bytes_after_first == bytes_after_second, \
        "second refresh must be a no-op; .pt bytes should be identical"


def test_keys_preserved(tmp_path):
    """audio_features, beats_patches, beats_grid_meta, filename must survive
    byte-identical. The key set itself must be unchanged."""
    T = 200
    pt_path = tmp_path / "clip.pt"
    af = torch.randn(T, 1024)
    bp = torch.randn(248, 768)
    bm = {"n_patches": 248, "d_patch": 768, "time_dim": 31,
          "freq_dim": 8, "backend": "fbank"}
    bad_info = _bad_old_format_overlap_info("56000-80000", T)
    _make_pt(pt_path, T, bad_info,
             beats_patches=bp, beats_grid_meta=bm,
             filename="clip.wav", audio_features=af)

    csv_path = tmp_path / "feat.csv"
    _write_csv(csv_path, [
        {"filename": "clip.wav", "overlap_segments": "56000-80000",
         "overlap_ratio": "0.1"},
    ])
    rc = run_full(tmp_path, csv_path)
    assert rc == 0

    out = torch.load(pt_path, weights_only=False)
    # The key set is unchanged.
    assert sorted(out.keys()) == sorted([
        "audio_features", "overlap_info", "overlap_segments",
        "filename", "beats_patches", "beats_grid_meta",
    ])
    # The 'untouched' keys' contents are byte-identical to the originals.
    assert torch.equal(out["audio_features"], af)
    assert torch.equal(out["beats_patches"], bp)
    assert out["beats_grid_meta"] == bm
    assert out["filename"] == "clip.wav"
    # And the only mutated key actually changed.
    assert not torch.equal(out["overlap_info"], bad_info)


def test_missing_csv_row_is_error(tmp_path):
    """A .pt with no matching CSV row must NOT be modified, and the run
    must exit with a nonzero return code."""
    T = 200
    pt_path = tmp_path / "ghost.pt"
    bad_info = _bad_old_format_overlap_info("56000-80000", T)
    _make_pt(pt_path, T, bad_info, filename="ghost.wav")
    bytes_before = pt_path.read_bytes()

    csv_path = tmp_path / "feat.csv"
    _write_csv(csv_path, [
        {"filename": "other.wav", "overlap_segments": "0-1000",
         "overlap_ratio": "0.1"},
    ])

    rc = run_full(tmp_path, csv_path)
    assert rc != 0, "rc should be nonzero when any .pt has no matching CSV row"
    bytes_after = pt_path.read_bytes()
    assert bytes_after == bytes_before, "ghost.pt must not be touched"


def test_empty_segments_means_zero_overlap_info(tmp_path):
    """CSV row with overlap_segments='' → refreshed overlap_info is all-zero."""
    T = 200
    pt_path = tmp_path / "clip.pt"
    # Start with deliberately bogus non-zero data to confirm it gets cleared.
    bogus_info = torch.full((T, 4), 0.99)
    _make_pt(pt_path, T, bogus_info, filename="clip.wav",
             overlap_segments=[(99.0, 99.5)])

    csv_path = tmp_path / "feat.csv"
    _write_csv(csv_path, [
        {"filename": "clip.wav", "overlap_segments": "",
         "overlap_ratio": "0.0"},
    ])
    rc = run_full(tmp_path, csv_path)
    assert rc == 0

    out = torch.load(pt_path, weights_only=False)
    assert torch.all(out["overlap_info"] == 0)
    assert out["overlap_segments"] == []


if __name__ == "__main__":
    import subprocess
    raise SystemExit(subprocess.call([sys.executable, "-m", "pytest", __file__, "-v"]))
