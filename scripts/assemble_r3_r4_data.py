#!/usr/bin/env python3
"""Assemble the R3+R4 joint retrain data: merge clean-F0 Libri2Mix + clean-train.

Three sub-commands (run after feature-extraction, description-build, and WavLM
preprocess of the clean s1 clips):

  merge-desc   descriptions_cleanf0.json (+ descriptions_clean_train.json)
               -> descriptions_cleanf0_r3.json   (Libri2Mix clean-F0 + clean no-hedge)

  merge-csv    features_pyannote/train-100.csv (+ features_clean_train.csv)
               -> features_cleanf0_r3_train.csv  (union of columns, for aux-head GT)

  link-dir     symlink processed_pyannote/train/*.pt + processed_clean_train/*.pt
               into <out>/train ; symlink processed_pyannote/{val,test} into <out>/{val,test}

So train.py --data_dir <out> --descriptions_path descriptions_cleanf0_r3.json
            --features_csv.train features_cleanf0_r3_train.csv  trains on both.
"""
import argparse
import csv
import json
import os
import sys


def merge_desc(args):
    out = {}
    for p in args.inputs:
        out.update(json.loads(open(p).read()))
    json.dump(out, open(args.output, "w"), ensure_ascii=False, indent=2)
    print(f"merge-desc: {len(out)} entries -> {args.output}")


def merge_csv(args):
    rows, cols = [], []
    seen = set()
    for p in args.inputs:
        with open(p) as f:
            r = csv.DictReader(f)
            for c in r.fieldnames or []:
                if c not in seen:
                    seen.add(c); cols.append(c)
            for row in r:
                rows.append(row)
    with open(args.output, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow({c: row.get(c, "") for c in cols})
    print(f"merge-csv: {len(rows)} rows, {len(cols)} cols -> {args.output}")


def link_dir(args):
    out = args.output
    for sub in ("train", "val", "test"):
        os.makedirs(os.path.join(out, sub), exist_ok=True)
    n = 0
    for src in args.train_dirs:
        for fn in os.listdir(src):
            if fn.endswith(".pt"):
                dst = os.path.join(out, "train", fn)
                if not os.path.exists(dst):
                    os.symlink(os.path.abspath(os.path.join(src, fn)), dst)
                    n += 1
    # val/test point at the Libri2Mix processed dirs (overlap eval set)
    for sub, src in (("val", args.val_dir), ("test", args.test_dir)):
        if not src:
            continue
        for fn in os.listdir(src):
            if fn.endswith(".pt"):
                dst = os.path.join(out, sub, fn)
                if not os.path.exists(dst):
                    os.symlink(os.path.abspath(os.path.join(src, fn)), dst)
    print(f"link-dir: linked {n} train .pt into {out}/train  "
          f"({[os.path.basename(d) for d in args.train_dirs]}); val/test linked")


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("merge-desc"); a.add_argument("--inputs", nargs="+", required=True)
    a.add_argument("--output", required=True); a.set_defaults(fn=merge_desc)
    b = sub.add_parser("merge-csv"); b.add_argument("--inputs", nargs="+", required=True)
    b.add_argument("--output", required=True); b.set_defaults(fn=merge_csv)
    c = sub.add_parser("link-dir"); c.add_argument("--train_dirs", nargs="+", required=True)
    c.add_argument("--val_dir", default=None); c.add_argument("--test_dir", default=None)
    c.add_argument("--output", required=True); c.set_defaults(fn=link_dir)
    args = ap.parse_args()
    args.fn(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
