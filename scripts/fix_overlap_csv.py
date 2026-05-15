#!/usr/bin/env python3
"""Re-derive overlap_segments + overlap_ratio from Silero VAD on s1/s2 stems
and write the result to NEW columns (overlap_segments_vad, overlap_ratio_vad)
alongside the existing pyannote-derived columns.

Why this exists
---------------
The Libri2Mix feature CSVs at $SHARED/data/features_pyannote/*.csv had their
overlap columns produced by `compute_overlap_pyannote` on the mix. Two
consequences:

1. Pyannote's sliding window over short clips produced segments past
   duration_sec (e.g. <r>5.0-7.5s</r> ranges in a 3.92 s clip) because the
   per-frame post-processing never clamped endpoints to clip duration.

2. The architectural intent (per CLAUDE.md and the verbalizer comments) is:
   - overlap_info channels for the model INPUT come from pyannote-on-mix
     (cross-domain swap-in)
   - the description GT for SFS encodes VAD-on-s1/s2 values (oracle)
   But both were being read from the same column, so the GT became
   pyannote-derived and circular.

This script keeps the original columns intact (preserves the input
distribution) and adds VAD-derived columns next to them so the description
builder can prefer the oracle values without disturbing preprocess.py.

Usage
-----
    python scripts/fix_overlap_csv.py \
        --csv            $SHARED/data/features_pyannote/dev.csv \
        --libri2mix_root $SHARED/data/Libri2Mix/Libri2Mix/wav16k/min/dev

Each split takes the matching libri2mix_root (contains s1/, s2/ subdirs).
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))


def load_silero():
    # Lazy imports keep the pure helpers (merge_adjacent, overlap_for_pair)
    # testable without torch/soundfile installed locally.
    import torch
    vad_model, utils = torch.hub.load(
        repo_or_dir="snakers4/silero-vad",
        model="silero_vad",
        force_reload=False,
    )
    get_speech_timestamps, _, _, *_ = utils
    return vad_model, get_speech_timestamps


def overlap_for_pair(wav1, wav2, sr, vad_model, get_speech_timestamps,
                     min_overlap_sec: float = 0.1):
    """Intersect Silero speech segments from wav1 and wav2; return list of
    (start_sample, end_sample) tuples. Bounded by min(len1, len2) by
    construction (max(s1.start, s2.start) <= min(s1.end, s2.end) <= min(len1, len2))."""
    import torch
    min_overlap_samples = int(min_overlap_sec * sr)
    t1 = torch.from_numpy(wav1).float()
    t2 = torch.from_numpy(wav2).float()
    segs1 = get_speech_timestamps(t1, vad_model, sampling_rate=sr, return_seconds=False)
    segs2 = get_speech_timestamps(t2, vad_model, sampling_rate=sr, return_seconds=False)
    overlaps = []
    for s1 in segs1:
        for s2 in segs2:
            a = max(s1["start"], s2["start"])
            b = min(s1["end"],   s2["end"])
            if b - a < min_overlap_samples:
                continue
            overlaps.append((a, b))
    overlaps.sort()
    return overlaps


def merge_adjacent(segs, min_gap_samples: int = 0):
    """Merge segments with gap <= min_gap_samples. Returns the merged list."""
    if not segs:
        return segs
    out = [list(segs[0])]
    for a, b in segs[1:]:
        if a - out[-1][1] <= min_gap_samples:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [(a, b) for a, b in out]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", required=True, help="features CSV to patch in place")
    p.add_argument("--libri2mix_root", required=True,
                   help="split dir containing s1/ and s2/ subdirs")
    p.add_argument("--min_overlap_sec", type=float, default=0.1,
                   help="ignore overlap intervals shorter than this (default: 0.1)")
    p.add_argument("--merge_gap_sec", type=float, default=0.0,
                   help="merge adjacent overlap segments with gap <= this (default: 0)")
    p.add_argument("--sample_rate", type=int, default=16000,
                   help="expected sample rate (default: 16000)")
    p.add_argument("--limit", type=int, default=None,
                   help="process at most N rows (smoke test)")
    args = p.parse_args()

    # Heavy deps live here (only needed for the actual CSV-rewrite path).
    import soundfile as sf  # noqa: F401  -- used below

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"ERROR: {csv_path} does not exist", file=sys.stderr)
        return 2

    s1_dir = Path(args.libri2mix_root) / "s1"
    s2_dir = Path(args.libri2mix_root) / "s2"
    if not s1_dir.is_dir() or not s2_dir.is_dir():
        print(f"ERROR: {args.libri2mix_root} must contain s1/ and s2/ subdirs",
              file=sys.stderr)
        return 2

    print(f"loading Silero VAD ...")
    vad_model, get_speech_timestamps = load_silero()
    print("ok")

    sr = args.sample_rate
    min_gap = int(args.merge_gap_sec * sr)

    with csv_path.open() as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])

    for col in ("overlap_segments_vad", "overlap_ratio_vad"):
        if col not in fieldnames:
            fieldnames.append(col)

    n_total = len(rows) if args.limit is None else min(len(rows), args.limit)
    n_done = n_missing = n_with_overlap = 0
    for i, row in enumerate(rows[:n_total]):
        fname = row.get("filename", "").strip()
        if not fname:
            row["overlap_segments_vad"] = ""
            row["overlap_ratio_vad"] = ""
            n_missing += 1
            continue
        s1p = s1_dir / fname
        s2p = s2_dir / fname
        if not s1p.exists() or not s2p.exists():
            row["overlap_segments_vad"] = ""
            row["overlap_ratio_vad"] = ""
            n_missing += 1
            if i < 5:
                print(f"  [warn] missing stems for {fname}")
            continue
        try:
            wav1, sr1 = sf.read(s1p, dtype="float32")
            wav2, sr2 = sf.read(s2p, dtype="float32")
        except Exception as e:
            row["overlap_segments_vad"] = ""
            row["overlap_ratio_vad"] = ""
            n_missing += 1
            print(f"  [warn] read failed for {fname}: {e}")
            continue
        if wav1.ndim > 1:
            wav1 = wav1.mean(axis=1)
        if wav2.ndim > 1:
            wav2 = wav2.mean(axis=1)
        if sr1 != sr or sr2 != sr:
            print(f"  [warn] sample-rate mismatch on {fname}: s1={sr1} s2={sr2}")
        mix_samples = min(len(wav1), len(wav2))
        if mix_samples == 0:
            row["overlap_segments_vad"] = ""
            row["overlap_ratio_vad"] = "0.0000"
            n_done += 1
            continue
        segs = overlap_for_pair(wav1, wav2, sr, vad_model, get_speech_timestamps,
                                min_overlap_sec=args.min_overlap_sec)
        if min_gap > 0:
            segs = merge_adjacent(segs, min_gap_samples=min_gap)
        total_overlap = sum(b - a for a, b in segs)
        ratio = total_overlap / mix_samples
        row["overlap_segments_vad"] = ";".join(f"{a}-{b}" for a, b in segs)
        row["overlap_ratio_vad"] = f"{ratio:.4f}"
        n_done += 1
        if segs:
            n_with_overlap += 1
        if (i + 1) % 250 == 0 or i + 1 == n_total:
            print(f"  [{i+1}/{n_total}] done={n_done} missing={n_missing} "
                  f"with_overlap={n_with_overlap}")

    tmp = csv_path.with_suffix(csv_path.suffix + ".tmp")
    with tmp.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})
    tmp.replace(csv_path)

    print()
    print(f"wrote {csv_path}")
    print(f"  total rows seen   : {len(rows)}")
    print(f"  processed         : {n_done}")
    print(f"  missing stems     : {n_missing}")
    print(f"  with VAD overlap  : {n_with_overlap}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
