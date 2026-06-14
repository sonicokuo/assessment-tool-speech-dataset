"""Compute ORACLE overlap ground truth for AMI clips from manual annotations.

AMI has no clean source stems, so the Libri2Mix overlap GT route (Silero VAD on
s1/s2 in scripts/fix_overlap_csv.py) does not apply. Instead we use AMI's manual
segment annotations: for each clip window [B,E] of one speaker, the union of times
where >= 2 DISTINCT speakers are simultaneously active (across ALL utterances of
that meeting) is the overlapped-speech region.

This is genuinely independent of the model's pyannote-on-clip INPUT (written by
src/feature_extractor.py into overlap_segments/overlap_ratio), so it avoids the
trivial-copy leakage the Libri2Mix VAD/pyannote split was designed to prevent.

Writes the canonical GT columns the description builder reads:
  overlap_ratio_vad      float, fraction of clip with >=2 speakers active
  overlap_segments_vad   ";"-joined start-end SAMPLE-INDEX pairs (sr=16000),
                         relative to clip start (the format _clean_overlap_segments
                         and src/preprocess.py expect)

Usage:
  python scripts/compute_ami_overlap_gt.py \
    --features_csv $SHARED/data/features_ami/test.csv \
    --manifest     $SHARED/data/ami_sdm/manifest_test.csv \
    --segments_csv $SHARED/data/ami_sdm/segments_test.csv
"""

import argparse
import csv
import os
import sys
from collections import defaultdict

SR = 16000


def load_segments(path: str) -> dict:
    """meeting_id -> list of (begin, end, speaker)."""
    segs = defaultdict(list)
    with open(path) as f:
        for r in csv.DictReader(f):
            segs[r["meeting_id"]].append(
                (float(r["begin_time"]), float(r["end_time"]), r["speaker_id"])
            )
    return segs


def overlap_regions(clip_b: float, clip_e: float, meeting_segs: list) -> list:
    """Return list of (start_sec, end_sec) within [clip_b, clip_e] where >= 2
    distinct speakers are simultaneously active."""
    # Segments intersecting the clip window, clipped to it.
    intervals = []
    for b, e, spk in meeting_segs:
        a = max(b, clip_b)
        z = min(e, clip_e)
        if z > a:
            intervals.append((a, z, spk))
    if len(intervals) < 2:
        return []

    # Boundary sweep: between consecutive boundaries the active speaker set is
    # constant; flag sub-intervals with >= 2 distinct speakers, then merge.
    bounds = sorted({clip_b, clip_e}
                    | {a for a, _, _ in intervals}
                    | {z for _, z, _ in intervals})
    regions = []
    for i in range(len(bounds) - 1):
        lo, hi = bounds[i], bounds[i + 1]
        if hi <= lo:
            continue
        mid = 0.5 * (lo + hi)
        active = {spk for a, z, spk in intervals if a <= mid < z}
        if len(active) >= 2:
            if regions and abs(regions[-1][1] - lo) < 1e-9:
                regions[-1] = (regions[-1][0], hi)   # merge adjacent
            else:
                regions.append((lo, hi))
    return regions


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features_csv", required=True,
                    help="feature CSV from src/feature_extractor.py (modified in place "
                         "unless --output given)")
    ap.add_argument("--manifest", required=True,
                    help="manifest_*.csv from prepare_ami_sdm.py (clip->meeting+window)")
    ap.add_argument("--segments_csv", required=True,
                    help="segments_test.csv from prepare_ami_sdm.py (ALL utterances)")
    ap.add_argument("--output", default=None, help="default: overwrite --features_csv")
    args = ap.parse_args()

    segs = load_segments(args.segments_csv)
    manifest = {}
    with open(args.manifest) as f:
        for r in csv.DictReader(f):
            manifest[r["filename"]] = (r["meeting_id"], float(r["begin_time"]),
                                       float(r["end_time"]))

    with open(args.features_csv) as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames)
        rows = list(reader)
    for col in ("overlap_segments_vad", "overlap_ratio_vad"):
        if col not in fieldnames:
            fieldnames.append(col)

    n_done = n_missing = n_with_overlap = 0
    for row in rows:
        fn = row["filename"]
        if fn not in manifest:
            row["overlap_segments_vad"] = ""
            row["overlap_ratio_vad"] = ""
            n_missing += 1
            continue
        meeting, B, E = manifest[fn]
        dur = E - B
        regions = overlap_regions(B, E, segs.get(meeting, []))
        total = sum(z - a for a, z in regions)
        ratio = round(total / dur, 4) if dur > 0 else 0.0
        seg_str = ";".join(
            f"{int(round((a - B) * SR))}-{int(round((z - B) * SR))}" for a, z in regions
        )
        row["overlap_ratio_vad"] = ratio
        row["overlap_segments_vad"] = seg_str
        n_done += 1
        if ratio > 0:
            n_with_overlap += 1

    out_path = args.output or args.features_csv
    tmp = out_path + ".tmp"
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, out_path)

    print(f"wrote {out_path}")
    print(f"  clips with oracle overlap GT : {n_done}")
    print(f"  clips with overlap_ratio > 0 : {n_with_overlap}")
    if n_missing:
        print(f"  [WARN] {n_missing} feature rows had no manifest entry (left blank)")


if __name__ == "__main__":
    sys.exit(main())
