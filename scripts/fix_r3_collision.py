#!/usr/bin/env python3
"""Fix the R3 clean-train collision: clean s1 stems are byte-identical to the
Libri2Mix mixture stems, so the merge/link silently dropped the clean clips and
the clean targets overwrote the mixture targets. This suffixes every clean-clip
identifier with `_s1clean` in all FOUR places the dataset keys on:

  1. clean features CSV `filename` column   (features-CSV / aux-head lookup)
  2. clean description key                   (build from the suffixed CSV)
  3. clean .pt on-disk name                  (stem -> description lookup, dataset.py:55)
  4. clean .pt internal "filename" field     (features-CSV lookup, dataset.py:56)

After this, assemble_r3_r4_data.py merges/links with no collision -> 27800 train.

Usage (run on the GPU node, CPU-only):
  python scripts/fix_r3_collision.py \
    --in_csv  $SHARED/data/features_clean_train.csv \
    --in_pt   $SHARED/data/processed_clean_train \
    --out_csv $SHARED/data/features_clean_train_s1clean.csv \
    --out_pt  $SHARED/data/processed_clean_train_s1clean \
    --suffix  _s1clean
"""
import argparse
import csv
import os

import torch


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", required=True)
    ap.add_argument("--in_pt", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--out_pt", required=True)
    ap.add_argument("--suffix", default="_s1clean")
    args = ap.parse_args()
    sfx = args.suffix

    # 1+2. suffix the features CSV filename (stem + suffix + original ext)
    rows = list(csv.DictReader(open(args.in_csv)))
    cols = rows[0].keys() if rows else []
    n_csv = 0
    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(cols))
        w.writeheader()
        for r in rows:
            fn = r.get("filename", "")
            stem, ext = os.path.splitext(fn)
            if stem:
                r["filename"] = stem + sfx + ext
                w.writerow(r)
                n_csv += 1
    print(f"[1/2] wrote suffixed CSV: {n_csv} rows -> {args.out_csv}")

    # 3+4. rename .pt on-disk + rewrite internal "filename"
    os.makedirs(args.out_pt, exist_ok=True)
    pts = [f for f in os.listdir(args.in_pt) if f.endswith(".pt")]
    n_pt = 0
    for fn in pts:
        stem, _ = os.path.splitext(fn)
        dst = os.path.join(args.out_pt, stem + sfx + ".pt")
        if os.path.exists(dst):
            n_pt += 1
            continue
        cached = torch.load(os.path.join(args.in_pt, fn), weights_only=False)
        # internal filename -> suffixed .wav (matches the suffixed CSV)
        old = cached.get("filename", fn)
        ostem, oext = os.path.splitext(old)
        cached["filename"] = ostem + sfx + (oext or ".wav")
        torch.save(cached, dst)
        n_pt += 1
        if n_pt % 1000 == 0:
            print(f"  rewrote {n_pt}/{len(pts)} .pt", flush=True)
    print(f"[3/4] rewrote+renamed {n_pt} .pt -> {args.out_pt}")
    print("Next: build_clean_train_descriptions.py on the suffixed CSV, then "
          "assemble_r3_r4_data.py with the suffixed clean sources.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
