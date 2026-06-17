#!/usr/bin/env python3
"""Compute well-posed (clean-frame) F0 for every clip in a features CSV.

Reads a features CSV (filename + overlap_segments_vad in sample indices) and the
mix_clean audio dir, runs Praat pitch on each clip but restricts the F0 mean/SD
to VOICED frames OUTSIDE the VAD overlap windows (src/f0_clean), and writes a
JSON lookup {filename: {f0_mean_hz, f0_sd_hz, clean_voiced_frac, n_clean}}.

This is the reference fix for the F0-GT problem (R4): the default extractor F0 is
on the 2-speaker mixture and ill-posed under overlap. Used to (a) re-score an
existing run against well-posed F0 and (b) rebuild descriptions for retraining.

Usage:
  python scripts/compute_clean_f0.py \
    --features_csv $SHARED/data/features_pyannote/test.csv \
    --audio_dir    $SHARED/data/Libri2Mix/Libri2Mix/wav16k/min/test/mix_clean \
    --output       $SHARED/data/clean_f0_test.json   [--limit N]
"""
import argparse
import csv
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from f0_clean import compute_f0_variation_clean, parse_overlap_windows_samples  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features_csv", required=True)
    ap.add_argument("--audio_dir", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--sr", type=int, default=16000)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--flush_every", type=int, default=200)
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.features_csv)))
    col = "overlap_segments_vad" if (rows and "overlap_segments_vad" in rows[0]) else "overlap_segments"
    if args.limit:
        rows = rows[: args.limit]

    # resume: keep any already-computed entries
    out = {}
    if os.path.exists(args.output):
        try:
            out = json.load(open(args.output))
        except Exception:
            out = {}

    n_done = 0
    for i, r in enumerate(rows):
        fn = (r.get("filename") or "").strip()
        if not fn or fn in out:
            continue
        wav = os.path.join(args.audio_dir, fn)
        if not os.path.exists(wav):
            continue
        win = parse_overlap_windows_samples(r.get(col) or "", args.sr)
        try:
            cf = compute_f0_variation_clean(wav, win)
        except Exception as e:
            cf = {"f0_mean_hz": float("nan"), "f0_sd_hz": float("nan"),
                  "clean_voiced_frac": 0.0, "n_clean_voiced": 0, "error": str(e)}
        out[fn] = cf
        n_done += 1
        if n_done % args.flush_every == 0:
            tmp = args.output + ".tmp"
            json.dump(out, open(tmp, "w"))
            os.replace(tmp, args.output)
            print(f"  {n_done} computed ({i+1}/{len(rows)} scanned)", flush=True)

    tmp = args.output + ".tmp"
    json.dump(out, open(tmp, "w"))
    os.replace(tmp, args.output)
    print(f"done: {len(out)} clips with clean F0 -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
