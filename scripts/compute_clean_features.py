#!/usr/bin/env python3
"""Compute well-posed (clean-stem) GT for the RECOVERABLE features for every clip
in a features CSV, mirroring scripts/compute_clean_f0.py.

For each clip it reads the CLEAN s1 stem (and s2 interferer) from the Libri2Mix
split and recomputes the features that the 2-speaker mixture corrupts:

  snr_db                         <- clean_features.clean_snr_db(s1, s2)   [target-vs-interferer]
  srmr                           <- clean_features.clean_srmr(s1)         [single-speaker]
  praat_speaking_rate_syl_sec    \
  praat_articulation_rate_syl_sec |
  praat_pause_count               > clean_features.clean_rate_and_pauses(s1)
  praat_pause_rate_per_min        |   [real silences on the single-speaker stem]
  praat_mean_pause_dur_sec        |
  praat_total_pause_dur_sec       |
  praat_pause_to_speech_ratio    /

overlap_ratio is NOT touched (the VAD-on-stems oracle in the CSV is already
correct). F0 is NOT touched (handled by compute_clean_f0.py).

Output: a JSON lookup {filename: {snr_db, srmr, praat_*}} analogous to the
clean_f0_*.json files, consumed by make_clean_features_csv.py to splice into a
single clean-GT features CSV.

The s1/s2 stems live next to mix_clean in the Libri2Mix tree, e.g.
  .../Libri2Mix/wav16k/min/<split>/{mix_clean,s1,s2}/<filename>.wav
Pass --stems_root pointing at the split dir that CONTAINS s1/ and s2/ (same dir
you would pass to feature_extractor_mix as --libri2mix_root).

SRMRpy NOTE: the compiled SRMRpy extension FAULTS ("Illegal instruction") on the
Bridges2 LOGIN nodes. Run the SRMR pass on a COMPUTE node, or pass --skip_srmr to
compute everything else on the login node and fill SRMR in a later compute-node
pass (resume merges into the same JSON).

Usage:
  python scripts/compute_clean_features.py \
    --features_csv $SHARED/data/features_pyannote/test.csv \
    --stems_root   $SHARED/data/Libri2Mix/Libri2Mix/wav16k/min/test \
    --output       $SHARED/data/clean_features_test.json   [--limit N] [--skip_srmr]
"""
import argparse
import csv
import json
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import soundfile as sf  # noqa: E402
from clean_features import clean_snr_db, clean_srmr, clean_rate_and_pauses  # noqa: E402


def _read_mono(path: str):
    a, sr = sf.read(path, dtype="float32")
    if getattr(a, "ndim", 1) > 1:
        a = a.mean(axis=1)
    return a, sr


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--features_csv", required=True)
    ap.add_argument("--stems_root", required=True,
                    help="split dir containing s1/ and s2/ subdirs (== feature_extractor_mix --libri2mix_root)")
    ap.add_argument("--output", required=True)
    ap.add_argument("--sr", type=int, default=16000)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--flush_every", type=int, default=200)
    ap.add_argument("--skip_srmr", action="store_true",
                    help="skip the SRMR pass (run it later on a compute node; SRMRpy faults on login nodes)")
    ap.add_argument("--min_pause_dur", type=float, default=0.3)
    args = ap.parse_args()

    s1_dir = os.path.join(args.stems_root, "s1")
    s2_dir = os.path.join(args.stems_root, "s2")
    if not os.path.isdir(s1_dir):
        print(f"ERROR: s1 dir not found: {s1_dir}", file=sys.stderr)
        return 2
    has_s2 = os.path.isdir(s2_dir)
    if not has_s2:
        print(f"[warn] s2 dir not found ({s2_dir}); SNR will be left unmeasured "
              f"(no interferer available)", file=sys.stderr)

    rows = list(csv.DictReader(open(args.features_csv)))
    if args.limit:
        rows = rows[: args.limit]

    # resume: keep any already-computed entries (so a login-node no-SRMR pass and a
    # later compute-node SRMR pass merge into the same file)
    out = {}
    if os.path.exists(args.output):
        try:
            out = json.load(open(args.output))
        except Exception:
            out = {}

    n_done = n_missing_stem = 0
    for i, r in enumerate(rows):
        fn = (r.get("filename") or "").strip()
        if not fn:
            continue
        prev = out.get(fn, {})
        # If a full entry already exists (and we're not back-filling SRMR), skip.
        need_srmr = (not args.skip_srmr) and (prev.get("srmr") is None)
        have_rest = all(k in prev for k in ("snr_db", "praat_pause_count"))
        if have_rest and not need_srmr:
            continue

        s1_path = os.path.join(s1_dir, fn)
        if not os.path.exists(s1_path):
            n_missing_stem += 1
            continue
        rec = dict(prev)
        try:
            s1, sr = _read_mono(s1_path)
        except Exception as e:  # noqa: BLE001
            print(f"  [WARNING] could not read s1 {fn}: {e}")
            continue

        if not have_rest:
            # SNR from target-vs-interferer (needs s2)
            if has_s2 and os.path.exists(os.path.join(s2_dir, fn)):
                try:
                    s2, _ = _read_mono(os.path.join(s2_dir, fn))
                    rec["snr_db"] = clean_snr_db(s1, s2, sr=sr)
                except Exception as e:  # noqa: BLE001
                    print(f"  [WARNING] SNR failed on {fn}: {e}")
                    rec["snr_db"] = float("nan")
            else:
                rec["snr_db"] = float("nan")
                rec["snr_no_interferer"] = True
            # rate + pauses on the clean stem
            rec.update(clean_rate_and_pauses(s1_path, min_pause_dur=args.min_pause_dur))

        if need_srmr:
            rec["srmr"] = clean_srmr(s1_path)

        out[fn] = rec
        n_done += 1
        if n_done % args.flush_every == 0:
            tmp = args.output + ".tmp"
            json.dump(out, open(tmp, "w"))
            os.replace(tmp, args.output)
            print(f"  {n_done} computed ({i+1}/{len(rows)} scanned)", flush=True)

    tmp = args.output + ".tmp"
    json.dump(out, open(tmp, "w"))
    os.replace(tmp, args.output)
    print(f"done: {len(out)} clips with clean features -> {args.output}  "
          f"(computed {n_done}, missing-stem {n_missing_stem}, "
          f"srmr={'skipped' if args.skip_srmr else 'included'})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
