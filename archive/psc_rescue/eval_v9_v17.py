#!/usr/bin/env python
"""Trustworthy v9-vs-v17 SFS eval with bootstrap CIs.

Reuses scoring/bootstrap helpers from scripts/eval_trustworthy.py. Recomputes SFS
per-clip from each model's inference_results.json with the COMMITTED src/sfs.py
(GT = parse(target) restricted to TOLERANCES keys). Paired comparison is run on the
intersection of completed filenames (v17 may be partial). Labels are v9 / v17.
"""
import argparse, json, os, sys
sys.path.insert(0, "/ocean/projects/cis260125p/shared/assessment-tool-redirect/src")
sys.path.insert(0, "/ocean/projects/cis260125p/shared/assessment-tool-redirect/scripts")
import numpy as np
from sfs import HybridClaimParser, SFSScorer
import eval_trustworthy as ET   # score_clip, bootstrap_ci, per_feature_bootstrap, aggregate_block, is_degenerate, fmt_ci, load_and_score

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v9", required=True)
    ap.add_argument("--v17", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--score_overlap_spans", action="store_true")
    args = ap.parse_args()

    parser = HybridClaimParser(); scorer = SFSScorer()
    B = args.bootstrap

    print(f"[load] v9  <- {args.v9}")
    _, v9_res, v9_by = ET.load_and_score(args.v9, parser, scorer, args.score_overlap_spans)
    print(f"[load] v17 <- {args.v17}")
    v17_exists = os.path.exists(args.v17)
    if v17_exists:
        _, v17_res, v17_by = ET.load_and_score(args.v17, parser, scorer, args.score_overlap_spans)
    else:
        v17_res, v17_by = [], {}

    print(f"[bootstrap] B={B}")
    v9_agg = ET.aggregate_block(v9_res, B, args.seed)
    summary = {"bootstrap_B": B, "seed": args.seed,
               "score_overlap_spans": args.score_overlap_spans,
               "v9": v9_agg, "v9_n": len(v9_res)}

    print("\n" + "="*86)
    print(f"TRUSTWORTHY SFS EVAL  (point [95% bootstrap CI], B={B})")
    print("="*86)
    print(f"\n{'model':6} {'n':>5}  {'SFS-F1 [95% CI]':30} {'precision':26} {'recall':26} {'degen':22}")
    print(f"{'v9':6} {v9_agg['n']:>5}  {ET.fmt_ci(v9_agg['f1']):30} {ET.fmt_ci(v9_agg['precision']):26} "
          f"{ET.fmt_ci(v9_agg['recall']):26} {ET.fmt_ci(v9_agg['degen_rate']):22}")

    if v17_exists:
        v17_agg = ET.aggregate_block(v17_res, B, args.seed+1000)
        summary["v17"] = v17_agg; summary["v17_n"] = len(v17_res)
        print(f"{'v17':6} {v17_agg['n']:>5}  {ET.fmt_ci(v17_agg['f1']):30} {ET.fmt_ci(v17_agg['precision']):26} "
              f"{ET.fmt_ci(v17_agg['recall']):26} {ET.fmt_ci(v17_agg['degen_rate']):22}")

    print(f"\nPER-FEATURE SFS accuracy (v9)  [fraction of made claims within tolerance]")
    print(f"  {'feature':18} {'acc [95% CI]':34} {'n_claims':>9}")
    for ft, pf in sorted(v9_agg["per_feature"].items()):
        lo, hi = pf["ci"]
        print(f"  {ft:18} {pf['accuracy']:.4f} [{lo:.4f}, {hi:.4f}]   {pf['n_claims']:>9}")

    if v17_exists:
        print(f"\nPER-FEATURE SFS accuracy (v17)")
        print(f"  {'feature':18} {'acc [95% CI]':34} {'n_claims':>9}")
        for ft, pf in sorted(v17_agg["per_feature"].items()):
            lo, hi = pf["ci"]
            print(f"  {ft:18} {pf['accuracy']:.4f} [{lo:.4f}, {hi:.4f}]   {pf['n_claims']:>9}")

        # paired on intersection
        common = sorted(set(v9_by) & set(v17_by))
        diffs = np.array([v17_by[fn]["f1"] - v9_by[fn]["f1"] for fn in common])
        p, lo, hi = ET.bootstrap_ci(diffs, B, args.seed+2000)
        excl = (lo > 0) or (hi < 0)
        paired = {"n_common": len(common), "mean_diff_v17_minus_v9": [float(p),float(lo),float(hi)],
                  "frac_v17_better": float((diffs>0).mean()), "frac_v9_better": float((diffs<0).mean()),
                  "frac_tie": float((diffs==0).mean()), "ci_excludes_zero": bool(excl)}
        summary["paired_v17_minus_v9"] = paired
        print(f"\nPAIRED  v17 - v9  per-clip F1 diff  (n_common={len(common)}, intersection of completed clips)")
        print(f"  mean diff: {p:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]")
        verdict = ('v17 RELIABLY better' if (excl and p>0) else ('v9 reliably better' if (excl and p<0) else 'within noise (CI straddles 0)'))
        print(f"  CI excludes 0: {excl}  ->  {verdict}")
        print(f"  fraction clips v17>v9: {paired['frac_v17_better']:.3f}  v9>v17: {paired['frac_v9_better']:.3f}  tie: {paired['frac_tie']:.3f}")
    else:
        print("\n[v17 inference_results.json not present yet — paired comparison skipped]")

    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[saved] {args.out}")

if __name__ == "__main__":
    main()
