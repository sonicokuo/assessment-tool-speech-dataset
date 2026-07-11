"""score_ami_transfer.py — score AMI cross-domain transfer against the extractor CSV as GT.

Cross-domain eval on AMI-SDM (far-field, real overlap). Ground truth = the DSP feature_extractor CSV
run on the AMI audio (measured snr/srmr/speaking_rate/pause_count/pause_rate + overlap_ratio). This
avoids needing an observability-prose GT: the model's generated prose is parsed with the SAME
ClaimParser used in-domain, and the parsed numbers are Spearman-correlated against the CSV values.

f0 stays HEDGED on real overlap (no clean stems exist for AMI), so f0 is NOT SRCC-scored here; instead
we report the f0 assert/hedge rate stratified by overlap quartile — the abstention behaviour on real
meetings, which is the on-message cross-domain result.

Usage:
  python scripts/score_ami_transfer.py --gt_csv data/features_ami_sdm_test.csv \
    --results .../inference_results.json --label M3b
"""
import argparse
import json
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, "scripts")
sys.path.insert(0, "src")
from score_matched_test import parse_feats  # noqa: E402

ROBUST = ["snr", "srmr", "speaking_rate", "pause_count", "pause_rate"]
CSVCOL = {
    "snr": "snr_db",
    "srmr": "srmr",
    "speaking_rate": "praat_speaking_rate_syl_sec",
    "pause_count": "praat_pause_count",
    "pause_rate": "praat_pause_rate_per_min",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt_csv", required=True)
    ap.add_argument("--results", required=True, help="inference_results.json: [{filename, generated}]")
    ap.add_argument("--label", default="model")
    a = ap.parse_args()

    gt = pd.read_csv(a.gt_csv)
    gt["key"] = gt["filename"].astype(str).str.replace(r"\.wav$", "", regex=True)
    gt = gt.set_index("key")
    preds = json.load(open(a.results))

    pairs = {f: [] for f in ROBUST}
    f0_assert, f0_overlap = [], []
    matched = 0
    for r in preds:
        key = str(r["filename"]).replace(".wav", "")
        if key not in gt.index:
            continue
        matched += 1
        pf = parse_feats(r.get("generated", ""))
        row = gt.loc[key]
        for f in ROBUST:
            g = row[CSVCOL[f]]
            p = pf.get(f)
            if p is not None and pd.notna(g):
                pairs[f].append((float(p), float(g)))
        ov = row.get("overlap_ratio", 0.0)
        ov = 0.0 if pd.isna(ov) else float(ov)
        f0_assert.append(pf.get("f0_mean") is not None)
        f0_overlap.append(ov)

    print(f"=== {a.label} AMI-SDM cross-domain transfer (matched {matched}/{len(preds)}) ===")
    srccs = []
    for f in ROBUST:
        xy = pairs[f]
        if len(xy) >= 10:
            x = [p for p, _ in xy]
            y = [g for _, g in xy]
            rho = spearmanr(x, y).correlation
            srccs.append(rho)
            print(f"  {f:14s} SRCC={rho:+.3f}  n={len(xy)}")
        else:
            print(f"  {f:14s} n={len(xy)} (skip)")
    if srccs:
        print(f"  >> mean SRCC over {len(srccs)} robust = {np.nanmean(srccs):.3f}")

    # f0 abstention behaviour vs overlap (on-message: no clean f0 GT on AMI)
    fa = np.array(f0_assert, dtype=float)
    ov = np.array(f0_overlap, dtype=float)
    print(f"  f0 assert rate overall: {fa.mean():.2f}  (hedge {1-fa.mean():.2f}), n={len(fa)}")
    if len(fa) >= 40:
        # split into overlap bins: none / low / high
        bins = [(-0.01, 0.0, "no-overlap"), (0.0, 0.2, "low"), (0.2, 1.01, "high")]
        for lo, hi, name in bins:
            m = (ov > lo) & (ov <= hi)
            if m.sum() >= 5:
                print(f"    overlap {name:10s} (n={int(m.sum()):4d}): f0 assert {fa[m].mean():.2f} / hedge {1-fa[m].mean():.2f}")


if __name__ == "__main__":
    main()
