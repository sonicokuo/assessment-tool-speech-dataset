"""score_inference_vs_clean.py — band-free SRCC of generated numeric claims vs the
INDEPENDENT clean-stem ground truth (NOT the mix-derived target the model trains on).

This is the only trustworthy headline metric (see overnight-report-2026-06-26.md and the
v21repro sanity audit): the in-training val/srcc_mean_reliable scores predictions against
the MIX observability target (correlates only ~0.58 with clean), so it overstates grounding.
Re-scoring the model's PARSED claims against clean_features_{split}.json + clean_f0_{split}.json
is what reproduces v21's audited 0.732.

Input: an inference_results.json written by src/inference.py — a list of per-clip dicts,
each with `filename` plus EITHER `per_feature` (list of {feature, claimed, actual, ...}) OR
`claims` (list of [feature, value]). Both forms are handled.

Usage:
    python scripts/score_inference_vs_clean.py \
        --inference_results checkpoints/<run>/inference_results.json \
        --clean_features data/clean_features_test.json \
        --clean_f0       data/clean_f0_test.json
"""
import argparse
import json

# model feature name -> (field in the clean GT file, which file: "features" | "f0")
FEATURE_MAP = {
    "srmr":             ("srmr",                           "features"),
    "speaking_rate":    ("praat_speaking_rate_syl_sec",    "features"),
    "articulation_rate":("praat_articulation_rate_syl_sec","features"),
    "pause_count":      ("praat_pause_count",              "features"),
    "pause_rate":       ("praat_pause_rate_per_min",       "features"),
    "snr":              ("snr_db",                         "features"),
    "f0_mean":          ("f0_mean_hz",                     "f0"),
    "f0_sd":            ("f0_sd_hz",                        "f0"),
}
# overlap_ratio is a mix-only property (no clean-stem GT) -> excluded here by construction.
# snr-scalar GT is degenerate (near-constant) -> reported but excluded from the headline mean.
DEGENERATE = {"snr"}


def _extract_pred(clip):
    """Return {feature: claimed_value} for one inference_results clip (handles both schemas)."""
    out = {}
    pf = clip.get("per_feature")
    if isinstance(pf, list):
        for e in pf:
            if isinstance(e, dict) and "feature" in e and e.get("claimed") is not None:
                out.setdefault(e["feature"], e["claimed"])
    for item in (clip.get("claims") or []):
        # claims are [feature, value] pairs
        try:
            f, v = item
        except (ValueError, TypeError):
            continue
        out.setdefault(f, v)
    return out


def _spearman(a, b):
    """Spearman rho without a hard scipy dependency (rank-transform + Pearson)."""
    n = len(a)
    if n < 3:
        return float("nan")

    def ranks(xs):
        order = sorted(range(n), key=lambda i: xs[i])
        r = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and xs[order[j + 1]] == xs[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0  # average rank for ties (1-based)
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r

    ra, rb = ranks(a), ranks(b)
    ma = sum(ra) / n
    mb = sum(rb) / n
    cov = sum((ra[i] - ma) * (rb[i] - mb) for i in range(n))
    va = sum((x - ma) ** 2 for x in ra) ** 0.5
    vb = sum((x - mb) ** 2 for x in rb) ** 0.5
    if va == 0 or vb == 0:
        return float("nan")
    return cov / (va * vb)


def score_vs_clean(results, clean_features, clean_f0, min_pairs=10):
    """Return {feature: {"srcc": float, "n": int}} + summary keys.

    results: list of inference_results clips. clean_features/clean_f0: dicts keyed by filename.
    """
    srcs = {"features": clean_features, "f0": clean_f0}
    # gather paired (pred, clean_gt) per feature
    cols = {}
    for clip in results:
        fn = clip.get("filename", "")
        pred = _extract_pred(clip)
        for feat, (field, which) in FEATURE_MAP.items():
            pv = pred.get(feat)
            if pv is None:
                continue
            g = srcs[which].get(fn) or srcs[which].get(fn.replace(".wav", ""))
            cg = g.get(field) if isinstance(g, dict) else None
            if cg is None:
                continue
            try:
                pv = float(pv)
                cg = float(cg)
            except (TypeError, ValueError):
                continue
            cols.setdefault(feat, ([], []))
            cols[feat][0].append(pv)
            cols[feat][1].append(cg)

    per_feature = {}
    reliable = []
    for feat, (a, b) in cols.items():
        if len(a) < min_pairs:
            per_feature[feat] = {"srcc": float("nan"), "n": len(a)}
            continue
        rho = _spearman(a, b)
        per_feature[feat] = {"srcc": rho, "n": len(a)}
        if feat not in DEGENERATE and rho == rho:  # not NaN
            reliable.append(rho)

    mean_reliable = sum(reliable) / len(reliable) if reliable else float("nan")
    return {
        "per_feature": per_feature,
        "mean_excl_snr": mean_reliable,
        "n_reliable_features": len(reliable),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inference_results", required=True)
    ap.add_argument("--clean_features", required=True)
    ap.add_argument("--clean_f0", required=True)
    ap.add_argument("--min_pairs", type=int, default=10)
    args = ap.parse_args()

    results = json.load(open(args.inference_results))
    clean_features = json.load(open(args.clean_features))
    clean_f0 = json.load(open(args.clean_f0))
    out = score_vs_clean(results, clean_features, clean_f0, args.min_pairs)

    print(f"{'feature':16s} {'vs-clean SRCC':>13s}  {'n':>5s}")
    for feat in ["srmr", "f0_mean", "speaking_rate", "articulation_rate",
                 "pause_count", "pause_rate", "f0_sd", "snr"]:
        if feat in out["per_feature"]:
            pf = out["per_feature"][feat]
            tag = "  (excl: degenerate)" if feat in DEGENERATE else ""
            print(f"{feat:16s} {pf['srcc']:13.3f}  {pf['n']:5d}{tag}")
    print(f"\nMEAN-excl-snr (headline) = {out['mean_excl_snr']:.3f}  "
          f"over {out['n_reliable_features']} reliable features   (v21 = 0.732)")


if __name__ == "__main__":
    main()
