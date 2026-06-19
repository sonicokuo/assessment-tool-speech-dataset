#!/usr/bin/env python3
"""Write a clean-F0 copy of a features CSV for CORRECT SFS scoring.

SFS grades F0 against the GT in the eval features CSV, but the shipped
features_pyannote/{dev,test}.csv have F0 measured on the 2-speaker MIXTURE
(ill-posed under overlap). A clean-F0-trained model predicts clean-frame F0, so
scoring it against the mixture GT undercounts f0_mean/f0_sd. This substitutes the
well-posed clean-frame F0 (from compute_clean_f0.py) into f0_mean_hz/f0_sd_hz so
the eval answer key matches what the model was trained to predict.

On clips whose clean F0 is UNDEFINED (no non-overlap voiced frames), f0 is blanked
so SFS does not score it for that clip (it is genuinely unmeasurable -> the model
should hedge, not be graded).

Usage:
  python scripts/make_clean_f0_csv.py --in_csv features_pyannote/dev.csv \
    --clean_f0 clean_f0_dev.json --out_csv features_pyannote/dev_cleanf0.csv
"""
import argparse
import csv
import json
import math


def _undef(x):
    return x is None or (isinstance(x, float) and math.isnan(x))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", required=True)
    ap.add_argument("--clean_f0", required=True)
    ap.add_argument("--out_csv", required=True)
    args = ap.parse_args()

    clean = json.loads(open(args.clean_f0).read())
    rows = list(csv.DictReader(open(args.in_csv)))
    cols = list(rows[0].keys()) if rows else []
    n_sub = n_blank = 0
    for r in rows:
        cf = clean.get((r.get("filename") or "").strip())
        if cf is None:
            continue
        cm, cs = cf.get("f0_mean_hz"), cf.get("f0_sd_hz")
        if _undef(cm):
            if "f0_mean_hz" in r:
                r["f0_mean_hz"] = ""
            if "f0_sd_hz" in r:
                r["f0_sd_hz"] = ""
            n_blank += 1
        else:
            if "f0_mean_hz" in r:
                r["f0_mean_hz"] = f"{round(float(cm), 2)}"
            if "f0_sd_hz" in r and not _undef(cs):
                r["f0_sd_hz"] = f"{round(float(cs), 2)}"
            n_sub += 1

    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {args.out_csv}: {len(rows)} rows  (clean-F0 substituted {n_sub}, "
          f"blanked-unmeasurable {n_blank})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
