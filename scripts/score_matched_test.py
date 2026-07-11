"""score_matched_test.py — §6 matched-TEST faithfulness scorer.

Scores an inference_results.json (list of {filename, generated, ...}) against the clean
GT encoded in the observability target prose (parsed with the SAME ClaimParser, so GT and
prediction are measured identically).  Reports per-feature Spearman SRCC, the headline mean
over the 5 robust features, and a per-condition split (overlap / clean / noise via filename
suffix).  f0 is scored only where the GT states it (the observability target withholds f0
under overlap, so f0-SRCC is naturally restricted to assert conditions — no overlap penalty).

Usage:
    python scripts/score_matched_test.py GT.json PRED_A.json [PRED_B.json ...]
"""
import json
import sys

sys.path.insert(0, "src")
import eval.sfs as sfs  # noqa: E402

try:
    from scipy.stats import spearmanr
    def srcc(x, y):
        return float(spearmanr(x, y).correlation)
except Exception:  # manual Spearman fallback
    def _rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        for rank, i in enumerate(order):
            r[i] = rank
        return r
    def srcc(x, y):
        rx, ry = _rank(x), _rank(y)
        n = len(x)
        mx = sum(rx) / n
        my = sum(ry) / n
        num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
        dx = sum((a - mx) ** 2 for a in rx) ** 0.5
        dy = sum((b - my) ** 2 for b in ry) ** 0.5
        return num / (dx * dy) if dx > 0 and dy > 0 else float("nan")

FEATS = ["snr", "srmr", "f0_mean", "speaking_rate", "pause_count", "pause_rate", "overlap_ratio"]
ROBUST = ["srmr", "snr", "speaking_rate", "pause_count", "pause_rate"]

_P = sfs.ClaimParser()


def parse_feats(text):
    out = {}
    for c in _P.parse(text or ""):
        if c.feature not in out:            # first claim per feature
            out[c.feature] = c.value
    return out


def load_preds(path):
    d = json.load(open(path))
    recs = d if isinstance(d, list) else list(d.values())
    out = {}
    for r in recs:
        stem = str(r.get("filename", "")).replace(".wav", "")
        out[stem] = parse_feats(r.get("generated", ""))
    return out


def cond(stem):
    if "_s1clean" in stem:
        return "clean"
    if "_aug" in stem:
        return "noise"
    return "overlap"


def score(gtf, preds, name, restrict_stems=None):
    print(f"=== {name}" + (f" [{restrict_stems[1]}]" if restrict_stems else "") + " ===")
    stems = gtf.keys() if restrict_stems is None else restrict_stems[0]
    per = {}
    for f in FEATS:
        xs, ys = [], []
        for s in stems:
            gv = gtf.get(s, {})
            if f in gv and s in preds and f in preds[s]:
                xs.append(preds[s][f]); ys.append(gv[f])
        if len(xs) >= 10:
            per[f] = (srcc(xs, ys), len(xs))
            print(f"  {f:14s} SRCC={per[f][0]:+.3f}  n={len(xs)}")
        else:
            print(f"  {f:14s} n={len(xs)} (skip)")
    rob = [per[f][0] for f in ROBUST if f in per]
    mean = sum(rob) / len(rob) if rob else float("nan")
    print(f"  >> mean SRCC over 5 robust = {mean:.4f}")
    return per, mean


def main():
    gt_path, pred_paths = sys.argv[1], sys.argv[2:]
    gt = json.load(open(gt_path))
    gtf = {k: parse_feats(v) for k, v in gt.items()}
    from collections import Counter
    print("condition counts:", dict(Counter(cond(s) for s in gtf)))
    means = {}
    for pp in pred_paths:
        name = pp.split("/")[-2]
        preds = load_preds(pp)
        _, m = score(gtf, preds, name)
        means[name] = m
        # per-condition (only if that condition has clips)
        by = {}
        for s in gtf:
            by.setdefault(cond(s), []).append(s)
        for c, ss in by.items():
            if len(ss) >= 30:
                score(gtf, preds, name, restrict_stems=(ss, c))
        print()
    if len(means) >= 2:
        ks = list(means)
        print(f"HEADLINE matched-TEST srcc_robust: "
              + " | ".join(f"{k}={means[k]:.4f}" for k in ks))


if __name__ == "__main__":
    main()
