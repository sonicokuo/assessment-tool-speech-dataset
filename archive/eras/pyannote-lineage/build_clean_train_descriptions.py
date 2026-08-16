#!/usr/bin/env python3
"""Build no-hedge descriptions for clean single-speaker (overlap=0) training clips.

The main builder (build_descriptions_deterministic --all) expects the three
Libri2Mix split CSVs. The clean s1-stem training set is a single CSV, so this
standalone builder runs build_description over it with the headline flags
(--untagged --no-overlap-segments --no-duration). Because these clips have no
overlap (overlap_ratio is nan/empty), the overlap section AND the F0-unreliable
hedge are omitted -> the model sees "clean speech -> no hedge", which is the
missing half of the calibrated-hedging signal.

Usage:
  python scripts/build_clean_train_descriptions.py \
    --features_csv $SHARED/data/features_clean_train.csv \
    --output       $SHARED/data/descriptions_clean_train.json
"""
import argparse
import csv
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_descriptions_deterministic import build_description  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features_csv", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--tag_suffix", default="",
                    help="optional suffix appended to each stem key to avoid "
                         "collisions with Libri2Mix stems (usually unneeded — "
                         "clean s1 names are unique)")
    ap.add_argument("--tagged", action="store_true",
                    help="emit tagged (<sec_*>/<f_*>) descriptions for the "
                         "section/2D-map path (default: untagged plain prose)")
    args = ap.parse_args()

    out = {}
    n_hedge = n_empty = 0
    for row in csv.DictReader(open(args.features_csv)):
        fname = (row.get("filename") or "").strip()
        stem = os.path.splitext(fname)[0]
        if not stem:
            continue
        text = build_description(row, None, untagged=not args.tagged,
                                 drop_overlap_segments=True, drop_duration=True)
        if not text:
            n_empty += 1
            continue
        if "unreliable" in text.lower():
            n_hedge += 1
        out[stem + args.tag_suffix] = text

    tmp = args.output + ".tmp"
    json.dump(out, open(tmp, "w"), ensure_ascii=False, indent=2)
    os.replace(tmp, args.output)
    print(f"wrote {args.output}")
    print(f"  entries        : {len(out)}")
    print(f"  with-hedge      : {n_hedge}  (should be ~0 for clean clips)")
    print(f"  empty (skipped) : {n_empty}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
