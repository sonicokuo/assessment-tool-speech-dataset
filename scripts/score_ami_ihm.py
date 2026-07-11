"""Re-score AMI cross-domain with CLEAN IHM-derived GT for the intrinsic features.

Model input was SDM (far-field). We keep SDM GT for the channel features (snr, srmr, which
describe the recording condition) and switch to IHM GT for the intrinsic features
(speaking_rate, pauses, f0 -- which describe the speaker and can only be measured reliably on
the clean headset channel). This removes the reverb-artifact that made the SDM-GT counting
columns uninterpretable.

Usage:
  python scripts/score_ami_ihm.py \
    --ihm_gt_csv $SHARED/data/features_ami_ihm_matched.csv \
    --sdm_gt_csv $SHARED/data/features_ami_sdm_test.csv \
    --results    $SHARED/checkpoints_ami_eval/M3b_sdm_merged.json --label M3b
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

# feature -> GT csv column; split by which GT source is valid
SDM_FEATS = {"snr": "snr_db", "srmr": "srmr"}                       # channel props: keep SDM GT
IHM_FEATS = {"speaking_rate": "praat_speaking_rate_syl_sec",         # intrinsic: clean IHM GT
             "pause_count": "praat_pause_count",
             "pause_rate": "praat_pause_rate_per_min",
             "f0_mean": "f0_mean_hz"}


def load_gt(path):
    d = pd.read_csv(path)
    d["key"] = d["filename"].astype(str).str.replace(r"\.wav$", "", regex=True)
    return d.set_index("key")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ihm_gt_csv", required=True)
    ap.add_argument("--sdm_gt_csv", required=True)
    ap.add_argument("--results", required=True, help="inference_results.json [{filename, generated}]")
    ap.add_argument("--label", default="model")
    a = ap.parse_args()

    ihm = load_gt(a.ihm_gt_csv)
    sdm = load_gt(a.sdm_gt_csv)
    preds = json.load(open(a.results))

    pairs = {f: [] for f in list(SDM_FEATS) + list(IHM_FEATS)}
    ihm_key_hits = 0
    for r in preds:
        key = str(r["filename"]).replace(".wav", "")
        pf = parse_feats(r.get("generated", ""))
        in_ihm = key in ihm.index
        if in_ihm:
            ihm_key_hits += 1
        for f, col in SDM_FEATS.items():
            if key in sdm.index and pf.get(f) is not None and pd.notna(sdm.loc[key, col]):
                pairs[f].append((float(pf[f]), float(sdm.loc[key, col])))
        for f, col in IHM_FEATS.items():
            if in_ihm and pf.get(f) is not None and pd.notna(ihm.loc[key, col]):
                pairs[f].append((float(pf[f]), float(ihm.loc[key, col])))

    print(f"=== {a.label} AMI cross-domain (IHM-clean GT for intrinsic feats) ===")
    print(f"    IHM GT covers {ihm_key_hits}/{len(preds)} predicted clips")
    srccs_all, srccs_ihm = [], []
    for f in list(SDM_FEATS) + list(IHM_FEATS):
        xy = pairs[f]
        src = "SDM" if f in SDM_FEATS else "IHM"
        if len(xy) >= 10:
            rho = spearmanr([x for x, _ in xy], [y for _, y in xy]).correlation
            print(f"  {f:14s} SRCC={rho:+.3f}  n={len(xy):4d}  [{src}-GT]")
            srccs_all.append(rho)
            if f in IHM_FEATS:
                srccs_ihm.append(rho)
        else:
            print(f"  {f:14s} n={len(xy)} (skip)")
    if srccs_all:
        print(f"  >> mean SRCC (5 feats, snr/srmr=SDM + spk/pauses/f0=IHM) = {np.nanmean(srccs_all):.3f}")
    if srccs_ihm:
        print(f"  >> intrinsic-only mean SRCC (IHM GT: spk_rate/pause_count/pause_rate/f0) = {np.nanmean(srccs_ihm):.3f}")
    # f0-when-asserted highlight
    f0xy = pairs.get("f0_mean", [])
    if len(f0xy) >= 10:
        rho = spearmanr([x for x, _ in f0xy], [y for _, y in f0xy]).correlation
        print(f"  >> f0-when-asserted vs clean IHM f0: SRCC={rho:+.3f} n={len(f0xy)}")


if __name__ == "__main__":
    main()
