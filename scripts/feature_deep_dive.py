"""Per-feature deep dive for a (results_json, gt_csv) pair.

For every parseable feature: coverage, emitted-value distribution, GT distribution, self-SRCC
(emitted vs its own GT), and the ENTANGLEMENT row (emitted feature vs EVERY numeric GT column,
to detect proxy/confound features, e.g. an emitted 'srmr' that tracks GT f0 more than GT srmr).

Flags per feature:
  EMIT-COLLAPSE  emitted values near-constant (<=max(5,3% of n) distinct)  -> degeneration
  GT-LOWVAR      GT itself near-constant (<=8 distinct)                     -> SRCC uninformative
  ENTANGLED>X    |SRCC vs some OTHER GT feature| exceeds self-SRCC by >0.05 -> confound/proxy

Usage:
  python scripts/feature_deep_dive.py --results <inference_results.json> --gt_csv <features.csv> --label NAME
"""
import argparse
import json
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr  # noqa: F401  (kept for parity; corr uses pandas)

sys.path.insert(0, "scripts")
sys.path.insert(0, "src")
from score_matched_test import parse_feats  # noqa: E402

# parse_feats key -> candidate GT column names (first present is used)
FEATURE_COLS = {
    "snr": ["snr_db", "snr"],
    "srmr": ["srmr", "srmr_db"],
    "f0_mean": ["f0_mean_hz", "f0_mean", "f0_mean_praat_hz"],
    "f0_sd": ["f0_sd_hz", "f0_std_hz", "f0_sd"],
    "hnr": ["hnr_db", "hnr"],
    "jitter": ["jitter_local_pct", "jitter_pct", "jitter_local", "jitter"],
    "shimmer": ["shimmer_local_pct", "shimmer_pct", "shimmer_local", "shimmer"],
    "speaking_rate": ["praat_speaking_rate_syl_sec", "speaking_rate_syl_sec", "speaking_rate"],
    "articulation_rate": ["praat_articulation_rate_syl_sec", "articulation_rate_syl_sec", "articulation_rate"],
    "pause_count": ["praat_pause_count", "pause_count"],
    "pause_rate": ["praat_pause_rate_per_min", "pause_rate_per_min", "pause_rate"],
    "overlap_ratio": ["overlap_ratio"],
    "duration_sec": ["duration_sec", "duration"],
}


def dist(a):
    a = np.asarray([x for x in a if x == x], float)
    if len(a) == 0:
        return "n=0"
    return (f"n={len(a)} dist={len(np.unique(np.round(a, 3)))} "
            f"std={a.std():.3f} [{a.min():.2f}/{np.median(a):.2f}/{a.max():.2f}]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--gt_csv", required=True)
    ap.add_argument("--label", default="")
    a = ap.parse_args()

    gt = pd.read_csv(a.gt_csv)
    gt["key"] = gt["filename"].astype(str).str.replace(r"\.wav$", "", regex=True)
    gt = gt.drop_duplicates("key").set_index("key")
    numeric_gt = [c for c in gt.columns if pd.api.types.is_numeric_dtype(gt[c])]
    preds = json.load(open(a.results))

    col = {f: next((c for c in cands if c in gt.columns), None) for f, cands in FEATURE_COLS.items()}
    present = [f for f in FEATURE_COLS if col[f]]

    # aligned emitted frame merged with numeric GT
    erows = []
    N = 0
    for r in preds:
        k = str(r["filename"]).replace(".wav", "")
        if k not in gt.index:
            continue
        N += 1
        pf = parse_feats(r.get("generated", ""))
        erows.append({"key": k, **{"emit_" + f: pf.get(f) for f in present}})
    edf = pd.DataFrame(erows)
    gsub = gt.reset_index()[["key"] + numeric_gt].rename(columns={c: "gt_" + c for c in numeric_gt})
    m = edf.merge(gsub, on="key")
    corr = m.corr(method="spearman", numeric_only=True)

    print(f"=== {a.label}  (n={N} matched clips; GT cols: {len(numeric_gt)}) ===")
    print(f"{'feature':17s}{'cov%':>5s}{'selfSRCC':>9s}   {'emitted dist':38s} {'GT dist':30s}  entangle(bestOtherGT)")
    for f in present:
        ecol, gcol = "emit_" + f, "gt_" + col[f]
        emit_vals = m[ecol].dropna().values
        cov = len(emit_vals) / max(N, 1)
        gt_vals = m[gcol].dropna().values
        self_srcc = corr.loc[ecol, gcol] if ecol in corr.index and gcol in corr.columns else float("nan")
        # entanglement: emitted f vs every OTHER gt col
        others = {}
        if ecol in corr.index:
            for gc in numeric_gt:
                gk = "gt_" + gc
                if gk == gcol or gk not in corr.columns:
                    continue
                v = corr.loc[ecol, gk]
                if v == v:
                    others[gc] = v
        bo = max(others.items(), key=lambda kv: abs(kv[1])) if others else None
        bo_s = f"{bo[0]}:{bo[1]:+.2f}" if bo else "-"
        print(f"{f:17s}{cov*100:4.0f}%{self_srcc:9.3f}   {dist(emit_vals):38s} {dist(gt_vals):30s}  {bo_s}")
        flags = []
        if len(emit_vals) >= 20 and len(np.unique(np.round(emit_vals, 2))) <= max(5, 0.03 * len(emit_vals)):
            flags.append("EMIT-COLLAPSE")
        if len(gt_vals) >= 20 and len(np.unique(np.round(gt_vals, 3))) <= 8:
            flags.append("GT-LOWVAR")
        if bo and self_srcc == self_srcc and abs(bo[1]) > abs(self_srcc) + 0.05:
            flags.append(f"ENTANGLED>{bo[0]}")
        if flags:
            print(f"{'':17s}   >> FLAGS: {flags}")


if __name__ == "__main__":
    main()
