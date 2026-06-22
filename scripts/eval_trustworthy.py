"""Statistically trustworthy full-test-set SFS evaluation with bootstrap CIs.

Replaces the noisy 32-clip val gauge. Recomputes SFS per-clip from each model's
inference_results.json (generated text + target text) using the COMMITTED
src/sfs.py parser+scorer, so every number reflects the corrected metric
(overlap_span recall gating + combined-F0 phrasing parser). Then attaches a
nonparametric bootstrap 95% CI to every headline number, and does a PAIRED
v17-vs-v14 comparison on the identical clips.

GT source: features.json is absent on PSC, so inference itself fell back to
parsing the target prose for SFS ground truth. We reproduce that exactly here:
GT = {claim.feature: claim.value for claim in parse(target)} restricted by the
scorer to TOLERANCES keys. This is the honest path — GT is naturally limited to
parseable, scorable features.

Usage:
    python scripts/eval_trustworthy.py \
        --v17 /ocean/.../checkpoints/v17_decoupled/inference_results.json \
        --v14 /ocean/.../checkpoints/v14_aug/inference_results.json \
        --out /ocean/.../checkpoints/v17_decoupled/trustworthy_eval.json \
        --bootstrap 2000 --seed 0
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import numpy as np

from sfs import HybridClaimParser, SFSScorer


# ── Degeneration detection ──────────────────────────────────
def is_degenerate(text: str) -> bool:
    """Flag a generation as degenerate using HIGH-CONFIDENCE signals only.

    Mirrors the failures the n=32 val gauge could not see: EOS-spam, token loops,
    off-topic hallucination drift. Deliberately conservative — the SFS template
    ("The X is V. The Y is W. ...") legitimately reuses function/unit words, so a
    naive global unique-word ratio over-flags healthy long outputs (verified on
    v14: a 247-word valid description sits at 0.19 unique-ratio). We therefore use:

      - empty / whitespace-only, OR
      - >5% non-ASCII characters (Qwen emits CJK on collapse), OR
      - any single word repeated >=8 times consecutively, OR
      - a long n-gram LOOP: some 8-word window recurs >=3 times (catches both the
        "The F0 mean is 197.40 Hz ... The F0 mean is 197.40 Hz" duplication and the
        off-topic "...whiteboard. ...whiteboard." hallucination loop) without
        penalizing the normal one-claim-per-sentence template.
    """
    if not text or not text.strip():
        return True
    non_ascii = sum(1 for c in text if ord(c) > 127)
    if len(text) > 0 and non_ascii / len(text) > 0.05:
        return True
    words = text.split()
    if len(words) >= 20:
        # max consecutive single-word repeat
        run = max_run = 1
        for i in range(1, len(words)):
            if words[i] == words[i - 1]:
                run += 1
                max_run = max(max_run, run)
            else:
                run = 1
        if max_run >= 8:
            return True
        # Repeated long n-gram LOOP, restricted to ADJACENT stutters. A true
        # decode loop repeats an 8-word span back-to-back ("The overlap ratio is
        # 0.515. The overlap ratio is 0.515. ..." / off-topic "...whiteboard..."
        # restatements). We must NOT flag the model's legitimate per-segment hedge
        # ("F0 and formant estimates are unreliable during overlap ...") which
        # recurs once per overlap segment and is therefore DISTRIBUTED, not
        # adjacent. Condition: some 8-gram has two occurrences whose start indices
        # are <= 12 words apart (i.e. the span repeats within itself + a short
        # gap), happening >= 3 times total.
        n = 8
        from collections import defaultdict
        positions = defaultdict(list)
        for i in range(len(words) - n + 1):
            positions[tuple(words[i:i + n])].append(i)
        for gram, idxs in positions.items():
            if len(idxs) < 3:
                continue
            # count how many of the occurrences are part of an adjacent run
            adjacent = sum(1 for a, b in zip(idxs, idxs[1:]) if (b - a) <= 12)
            if adjacent >= 2:   # >=2 adjacent gaps => >=3 occurrences stuttering
                return True
    return False


# ── Per-clip scoring (committed sfs.py) ─────────────────────
def score_clip(parser: HybridClaimParser, scorer: SFSScorer, generated: str, target: str,
               score_overlap_spans: bool, overlap_segments=None) -> dict:
    """Recompute SFS for one clip from generated + target text.

    Returns dict with precision/recall/f1, per_feature list, and degeneracy flag.
    GT = parse(target); scorer restricts the recall denominator to TOLERANCES keys.
    """
    gt_claims = parser.parse(target or "")
    ground_truth = {c.feature: c.value for c in gt_claims}

    # Match inference.py: only add overlap_segments to GT when the model is
    # trained to emit spans (score_overlap_spans). Untagged v14/v17 set this
    # False, so spans never enter the recall denominator.
    if score_overlap_spans and overlap_segments and "overlap_segments" not in ground_truth:
        ground_truth["overlap_segments"] = overlap_segments

    claims = parser.parse(generated or "")
    res = scorer.score(claims, ground_truth)
    return {
        "precision": res["precision"],
        "recall": res["recall"],
        "f1": res["f1"],
        "per_feature": res["per_feature"],
        "degenerate": is_degenerate(generated or ""),
        "has_gt": bool(set(ground_truth) & set(scorer.TOLERANCES) or ground_truth.get("overlap_segments")),
    }


# ── Bootstrap ───────────────────────────────────────────────
def bootstrap_ci(values, B=2000, seed=0, agg=np.mean):
    """Nonparametric bootstrap 95% CI of an aggregate over per-clip values.

    values: 1-D array of per-clip scalars (e.g. per-clip F1).
    Returns (point, lo, hi) where point = agg(values) and [lo,hi] is the
    2.5/97.5 percentile of the bootstrap distribution of agg.
    """
    values = np.asarray(values, dtype=float)
    n = len(values)
    if n == 0:
        return (float("nan"), float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(B, n))
    boot = agg(values[idx], axis=1)
    point = float(agg(values))
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return (point, float(lo), float(hi))


def per_feature_bootstrap(clip_results, feature, B=2000, seed=0):
    """Per-feature SFS accuracy = fraction of MADE claims (for that feature)
    that are within tolerance, with a bootstrap CI resampling at the CLIP level
    (so clips that make the claim multiple times stay coupled).

    We resample clips; for each bootstrap draw we recompute
    (total correct claims for the feature) / (total claims for the feature)
    over the resampled clips. This respects clip-level dependence, unlike
    resampling individual claims.
    """
    # Per clip: (n_correct_for_feature, n_total_for_feature)
    per_clip = []
    for r in clip_results:
        c = t = 0
        for f in r["per_feature"]:
            if f["feature"] == feature:
                t += 1
                if f["correct"]:
                    c += 1
        per_clip.append((c, t))
    per_clip = np.asarray(per_clip, dtype=float)  # (n_clips, 2)
    total_claims = int(per_clip[:, 1].sum())
    if total_claims == 0:
        return None
    point = per_clip[:, 0].sum() / total_claims

    rng = np.random.default_rng(seed)
    n = len(per_clip)
    idx = rng.integers(0, n, size=(B, n))
    drawn = per_clip[idx]                      # (B, n, 2)
    num = drawn[:, :, 0].sum(axis=1)
    den = drawn[:, :, 1].sum(axis=1)
    den = np.where(den == 0, np.nan, den)
    boot = num / den
    lo, hi = np.nanpercentile(boot, [2.5, 97.5])
    return {"accuracy": float(point), "n_claims": total_claims, "ci": [float(lo), float(hi)]}


# ── Driver ──────────────────────────────────────────────────
def load_and_score(path, parser, scorer, score_overlap_spans):
    with open(path) as f:
        entries = json.load(f)
    by_file = {}
    results = []
    for e in entries:
        gen = e.get("generated", "")
        tgt = e.get("target", "")
        r = score_clip(parser, scorer, gen, tgt, score_overlap_spans,
                       overlap_segments=e.get("overlap_segments"))
        r["filename"] = e.get("filename")
        results.append(r)
        by_file[e.get("filename")] = r
    return entries, results, by_file


def aggregate_block(results, B, seed):
    f1 = [r["f1"] for r in results]
    pr = [r["precision"] for r in results]
    rc = [r["recall"] for r in results]
    degen = [1.0 if r["degenerate"] else 0.0 for r in results]
    out = {
        "n": len(results),
        "f1": bootstrap_ci(f1, B, seed),
        "precision": bootstrap_ci(pr, B, seed + 1),
        "recall": bootstrap_ci(rc, B, seed + 2),
        "degen_rate": bootstrap_ci(degen, B, seed + 3),
    }
    # per-feature
    feats = set()
    for r in results:
        for f in r["per_feature"]:
            feats.add(f["feature"])
    out["per_feature"] = {}
    for i, ft in enumerate(sorted(feats)):
        pf = per_feature_bootstrap(results, ft, B, seed + 100 + i)
        if pf is not None:
            out["per_feature"][ft] = pf
    return out


def fmt_ci(triple):
    p, lo, hi = triple
    return f"{p:.4f} [{lo:.4f}, {hi:.4f}]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v17", required=True)
    ap.add_argument("--v14", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--score_overlap_spans", action="store_true",
                    help="Add overlap_segments to GT denominator. Default off "
                         "(matches untagged v14/v17 score_overlap_spans=false).")
    args = ap.parse_args()

    parser = HybridClaimParser()
    scorer = SFSScorer()

    print(f"[load] v17 ← {args.v17}")
    _, v17_res, v17_by = load_and_score(args.v17, parser, scorer, args.score_overlap_spans)
    print(f"[load] v14 ← {args.v14}")
    _, v14_res, v14_by = load_and_score(args.v14, parser, scorer, args.score_overlap_spans)

    B = args.bootstrap
    print(f"[bootstrap] B={B}")
    v17_agg = aggregate_block(v17_res, B, args.seed)
    v14_agg = aggregate_block(v14_res, B, args.seed + 1000)

    # ── PAIRED comparison on identical clips ──
    common = sorted(set(v17_by) & set(v14_by))
    diffs = np.array([v17_by[fn]["f1"] - v14_by[fn]["f1"] for fn in common])
    paired = {
        "n_common": len(common),
        "mean_diff": bootstrap_ci(diffs, B, args.seed + 2000),
        "frac_v17_better": float((diffs > 0).mean()),
        "frac_v14_better": float((diffs < 0).mean()),
        "frac_tie": float((diffs == 0).mean()),
    }

    summary = {
        "bootstrap_B": B,
        "seed": args.seed,
        "score_overlap_spans": args.score_overlap_spans,
        "v17": v17_agg,
        "v14": v14_agg,
        "paired_v17_minus_v14": paired,
    }
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)

    # ── Print tables ──
    print("\n" + "=" * 78)
    print("TRUSTWORTHY SFS EVAL  (point [95% bootstrap CI], n clips, B={})".format(B))
    print("=" * 78)
    print(f"\n{'model':6} {'n':>5}  {'SFS-F1 [95% CI]':28} {'precision':24} {'recall':24} {'degen':22}")
    for name, agg in [("v17", v17_agg), ("v14", v14_agg)]:
        print(f"{name:6} {agg['n']:>5}  {fmt_ci(agg['f1']):28} {fmt_ci(agg['precision']):24} "
              f"{fmt_ci(agg['recall']):24} {fmt_ci(agg['degen_rate']):22}")

    print(f"\nPER-FEATURE SFS accuracy (v17)  [fraction of made claims within tolerance]")
    print(f"  {'feature':18} {'acc [95% CI]':30} {'n_claims':>9}")
    for ft, pf in sorted(v17_agg["per_feature"].items()):
        lo, hi = pf["ci"]
        print(f"  {ft:18} {pf['accuracy']:.4f} [{lo:.4f}, {hi:.4f}]   {pf['n_claims']:>9}")

    print(f"\nPER-FEATURE SFS accuracy (v14)")
    print(f"  {'feature':18} {'acc [95% CI]':30} {'n_claims':>9}")
    for ft, pf in sorted(v14_agg["per_feature"].items()):
        lo, hi = pf["ci"]
        print(f"  {ft:18} {pf['accuracy']:.4f} [{lo:.4f}, {hi:.4f}]   {pf['n_claims']:>9}")

    print(f"\nPAIRED  v17 − v14  per-clip F1 diff  (n_common={paired['n_common']})")
    p, lo, hi = paired["mean_diff"]
    excludes_zero = (lo > 0) or (hi < 0)
    print(f"  mean diff: {p:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]")
    print(f"  CI excludes 0: {excludes_zero}  →  "
          f"{'v17 RELIABLY better' if (excludes_zero and p > 0) else ('v14 reliably better' if (excludes_zero and p < 0) else 'within noise (CI straddles 0)')}")
    print(f"  fraction clips v17>v14: {paired['frac_v17_better']:.3f}  "
          f"v14>v17: {paired['frac_v14_better']:.3f}  tie: {paired['frac_tie']:.3f}")
    print(f"\n[saved] {args.out}")


if __name__ == "__main__":
    main()
