"""Resumable wrapper around src/feature_extractor.py for large clip sets.

feature_extractor.py accumulates every record in memory and writes the CSV once
at the very end, so a session timeout / SIGKILL mid-run loses all work and it
re-does everything on restart. For the full AMI SDM test set (~3.6k clips, a
multi-hour pyannote+Praat+SRMR pass on a shared node) that risk is unacceptable.

This wrapper reuses the EXACT extraction functions (so feature values are
identical to feature_extractor.py), but:
  - loads the pyannote + SRMR models ONCE,
  - writes the output CSV atomically every --checkpoint_every clips, and
  - skips clips already present in an existing output CSV (resume).

Usage (offline on a compute node, models pre-cached):
  HF_HUB_OFFLINE=1 python scripts/extract_features_resumable.py \
    --audio_dir  $SHARED/data/ami_sdm/wav/test \
    --output     $SHARED/data/features_ami/test.csv
"""

import argparse
import glob
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from feature_extractor import (  # noqa: E402
    COLUMN_ORDER,
    SUPPORTED_EXTENSIONS,
    extract_features,
    load_overlap_pipeline,
    load_srmr_model,
)


def write_csv(records: list, out_path: str):
    df = pd.DataFrame(records)
    extra = [c for c in df.columns if c not in COLUMN_ORDER]
    df = df[[c for c in COLUMN_ORDER if c in df.columns] + extra]
    tmp = out_path + ".tmp"
    df.to_csv(tmp, index=False)
    os.replace(tmp, out_path)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--audio_dir", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--checkpoint_every", type=int, default=100)
    ap.add_argument("--srmr_max_cf", type=int, default=128)
    ap.add_argument("--no_overlap", action="store_true")
    ap.add_argument("--no_srmr", action="store_true")
    args = ap.parse_args()

    audio_files = sorted(
        p for ext in SUPPORTED_EXTENSIONS
        for p in glob.glob(os.path.join(args.audio_dir, f"**/*{ext}"), recursive=True)
    )
    if not audio_files:
        print(f"[ERROR] no audio in {args.audio_dir}", file=sys.stderr)
        return 2
    print(f"Found {len(audio_files)} audio files in {args.audio_dir}")

    # Resume: load any rows already extracted.
    records = []
    done = set()
    if os.path.exists(args.output):
        old = pd.read_csv(args.output)
        records = old.to_dict("records")
        done = set(old["filename"].astype(str))
        print(f"[resume] {len(done)} clips already in {args.output}; will skip them")

    todo = [p for p in audio_files if os.path.basename(p) not in done]
    print(f"{len(todo)} clips to extract")
    if not todo:
        print("nothing to do.")
        return 0

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    overlap_pipeline = None if args.no_overlap else load_overlap_pipeline()
    srmr_model = None if args.no_srmr else load_srmr_model(
        {"max_cf": args.srmr_max_cf, "fast": True, "norm": False})

    for i, wav_path in enumerate(todo):
        records.append(extract_features(wav_path, overlap_pipeline, srmr_model))
        if (i + 1) % args.checkpoint_every == 0:
            write_csv(records, args.output)
            print(f"[ckpt] {i + 1}/{len(todo)} extracted ({len(records)} total rows)", flush=True)

    write_csv(records, args.output)
    print(f"Done. wrote {len(records)} rows to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
