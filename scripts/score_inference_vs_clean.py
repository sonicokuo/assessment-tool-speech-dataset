"""score_inference_vs_clean.py — band-free SRCC of generated numeric claims vs the
INDEPENDENT clean-stem ground truth (NOT the mix-derived target the model trains on).

This is the only trustworthy headline metric (see overnight-report-2026-06-26.md and the
v21repro sanity audit): the in-training val/srcc_mean_reliable scores predictions against
the MIX observability target (correlates only ~0.58 with clean), so it overstates grounding.
Re-scoring the model's PARSED claims against clean_features_{split}.json + clean_f0_{split}.json
is what reproduces v21's audited 0.732.

HEADLINE FREEZE (fix F8, 2026-07-15 metric-consistency audit): the headline
`mean_reliable` is the mean SRCC over EXACTLY the frozen ROBUST5 set imported from
src/eval/selection_metric.HEADLINE_FEATURES — the same set the val selector uses —
so val selection and test reporting can never diverge again. f0_mean / f0_sd are
still scored per-feature but live in a separate "ill-posed / abstention panel";
they are NEVER averaged into the headline. The stale articulation_rate row (dropped
from the supervised set 2026-06-24) was deleted from FEATURE_MAP entirely.

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
import os
import sys

# Path-insert like the other scripts so the frozen headline set is IMPORTED from its
# single source of truth (src/eval/selection_metric.py), never duplicated here (F8).
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from eval.selection_metric import HEADLINE_FEATURES  # noqa: E402

# model feature name -> (field in the clean GT file, which file: "features" | "f0")
# The first block is exactly HEADLINE_FEATURES (asserted below). The f0_* rows are
# PANEL-ONLY: per-feature stats are still computed and printed, but under the
# "ill-posed / abstention panel" — never inside mean_reliable (f0 was diagnosed
# ill-posed / mode-collapsing on the mix, 2026-07-13).
FEATURE_MAP = {
    # ── headline (frozen ROBUST5) ────────────────────────────────────────────
    "snr":              ("snr_db",                         "features"),
    "srmr":             ("srmr",                           "features"),
    "speaking_rate":    ("praat_speaking_rate_syl_sec",    "features"),
    "pause_count":      ("praat_pause_count",              "features"),
    "pause_rate":       ("praat_pause_rate_per_min",       "features"),
    # ── ill-posed / abstention panel (NEVER in the headline mean) ────────────
    "f0_mean":          ("f0_mean_hz",                     "f0"),
    "f0_sd":            ("f0_sd_hz",                       "f0"),
}
# overlap_ratio is a mix-only property (no clean-stem GT) -> excluded by construction.
# articulation_rate was DELETED 2026-07-15 (F8): dropped from the supervised feature
# set on 2026-06-24, its stale row here was silently averaged into the old headline.
PANEL_FEATURES = tuple(f for f in FEATURE_MAP if f not in HEADLINE_FEATURES)
assert set(HEADLINE_FEATURES) <= set(FEATURE_MAP), (
    "every frozen headline feature needs a clean-GT mapping in FEATURE_MAP"
)


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
    """Score parsed claims against clean GT; return per-feature stats + frozen headline.

    results: list of inference_results clips. clean_features/clean_f0: dicts keyed by filename.

    Returns:
        per_feature: {feat: {"srcc": float|None, "n": int, "n_emitted": int,
                             "coverage": float}} for EVERY feature in FEATURE_MAP —
            a feature emitted on 0 clips still appears (coverage 0.0, srcc None).
            srcc is None when undefined (< min_pairs pairs, or zero rank variance).
        mean_reliable:          THE headline — mean SRCC over exactly HEADLINE_FEATURES
                                (frozen ROBUST5; snr IN, f0/overlap_ratio/articulation_rate OUT).
        mean_reliable_no_snr:   the same mean without snr — the SNR-circularity check,
                                made a standard secondary line (F8).
        mean_coverage_reliable: mean coverage over ALL headline features (headline-level
                                coverage; defined even when a feature's SRCC is not).
        n_reliable_features:    headline features contributing a defined SRCC (max 5).

    Coverage (F10) = n_emitted / n_clips: the fraction of clips whose generation
    contained a parseable numeric claim for the feature. Rationale (2026-07-15 audit
    risk 3): SRCC pairs form only where a claim parses, so under asymmetric
    degeneration two models are silently scored on different clip subsets — coverage
    must always be visible next to every SRCC.
    """
    srcs = {"features": clean_features, "f0": clean_f0}
    total = len(results)
    # gather paired (pred, clean_gt) per feature; count emissions independently of GT
    cols = {feat: ([], []) for feat in FEATURE_MAP}
    n_emitted = {feat: 0 for feat in FEATURE_MAP}
    for clip in results:
        fn = clip.get("filename", "")
        pred = _extract_pred(clip)
        for feat, (field, which) in FEATURE_MAP.items():
            pv = pred.get(feat)
            if pv is None:
                continue
            try:
                pv = float(pv)
            except (TypeError, ValueError):
                continue
            n_emitted[feat] += 1
            g = srcs[which].get(fn) or srcs[which].get(fn.replace(".wav", ""))
            cg = g.get(field) if isinstance(g, dict) else None
            if cg is None:
                continue
            try:
                cg = float(cg)
            except (TypeError, ValueError):
                continue
            cols[feat][0].append(pv)
            cols[feat][1].append(cg)

    per_feature = {}
    for feat in FEATURE_MAP:
        a, b = cols[feat]
        n = len(a)
        rho = _spearman(a, b) if n >= min_pairs else float("nan")
        per_feature[feat] = {
            "srcc": None if rho != rho else rho,  # NaN -> None (undefined, JSON-safe)
            "n": n,
            "n_emitted": n_emitted[feat],
            "coverage": (n_emitted[feat] / total) if total else 0.0,
        }

    # F8 frozen headline: mean SRCC over exactly HEADLINE_FEATURES — the same set
    # the val selector optimizes, so selection and reporting cannot diverge.
    head = [per_feature[f]["srcc"] for f in HEADLINE_FEATURES
            if per_feature[f]["srcc"] is not None]
    head_no_snr = [per_feature[f]["srcc"] for f in HEADLINE_FEATURES
                   if f != "snr" and per_feature[f]["srcc"] is not None]
    head_cov = [per_feature[f]["coverage"] for f in HEADLINE_FEATURES]
    return {
        "per_feature": per_feature,
        "mean_reliable": sum(head) / len(head) if head else float("nan"),
        "mean_reliable_no_snr": (sum(head_no_snr) / len(head_no_snr)
                                 if head_no_snr else float("nan")),
        "mean_coverage_reliable": sum(head_cov) / len(head_cov) if head_cov else 0.0,
        "n_reliable_features": len(head),
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

    def _row(feat):
        pf = out["per_feature"][feat]
        s = f"{pf['srcc']:13.3f}" if pf["srcc"] is not None else f"{'-':>13s}"
        print(f"{feat:16s} {s}  {pf['coverage']:8.3f}  {pf['n']:5d}")

    print(f"{'feature':16s} {'vs-clean SRCC':>13s}  {'coverage':>8s}  {'n':>5s}")
    print("── headline (frozen ROBUST5 = selection_metric.HEADLINE_FEATURES) ──")
    for feat in HEADLINE_FEATURES:
        _row(feat)
    print("── ill-posed / abstention panel (NEVER in the headline mean) ──")
    for feat in PANEL_FEATURES:
        _row(feat)

    print(f"\nHEADLINE mean SRCC (ROBUST5 {', '.join(HEADLINE_FEATURES)}; snr INCLUDED) "
          f"= {out['mean_reliable']:.3f}  over {out['n_reliable_features']}/"
          f"{len(HEADLINE_FEATURES)} features")
    print(f"  secondary: no-snr mean (SNR-circularity check) = "
          f"{out['mean_reliable_no_snr']:.3f}")
    print(f"  headline mean coverage (ROBUST5)               = "
          f"{out['mean_coverage_reliable']:.3f}")


if __name__ == "__main__":
    main()
