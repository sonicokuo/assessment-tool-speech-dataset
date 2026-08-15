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
import statistics
import sys


def _boot_ci(xs, ys, fn, n_boot=1000, seed=0, alpha=0.05):
    """Percentile bootstrap CI over CLIPS for any paired statistic.

    Required because differences of 0.002-0.01 have been treated as results in this project
    (the 2L depth arm "won" by +0.0017, inside a 0.017 seed spread). A headline without an
    interval invites exactly that error. Resamples clips with replacement; seeded so the
    reported interval is reproducible.
    """
    import random as _r
    rng = _r.Random(seed)
    n = len(xs)
    if n < 20:
        return (float("nan"), float("nan"))
    stats = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        bx = [xs[i] for i in idx]
        by = [ys[i] for i in idx]
        try:
            v = fn(bx, by)
        except Exception:
            continue
        if v == v:
            stats.append(v)
    if not stats:
        return (float("nan"), float("nan"))
    stats.sort()
    lo = stats[int((alpha / 2) * len(stats))]
    hi = stats[int((1 - alpha / 2) * len(stats)) - 1]
    return (lo, hi)


def _stdev(v):
    """Population sd of the GT, the nMAE denominator. 0.0 for degenerate input so the
    caller emits nan rather than dividing by zero."""
    return statistics.pstdev(v) if len(v) > 1 else 0.0

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


def load_measured(csv_path):
    """{stem: {feature: value}} for every feature the INSTRUMENT actually measured.

    This is the cov_meas denominator: it includes overlap clips, where the target
    deliberately abstains, so it can distinguish "withheld" from "never measurable".
    Returns {} when no CSV is supplied, in which case cov_meas prints nan.
    """
    if not csv_path:
        return {}
    import csv as _csv
    import os as _os
    sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "src"))
    from data.feature_set import SUPERVISED_FEATURES
    cols = {n: c for n, c, _ in SUPERVISED_FEATURES}
    out = {}
    for row in _csv.DictReader(open(csv_path)):
        stem = (row.get("filename") or "").rsplit(".", 1)[0]
        d = {}
        for feat, col in cols.items():
            v = row.get(col)
            if v not in (None, "", "nan"):
                try:
                    fv = float(v)
                    if fv == fv:
                        d[feat] = fv
                except ValueError:
                    pass
        out[stem] = d
    return out


def score(gtf, preds, name, restrict_stems=None, meas=None):
    """Band-free scoring: SRCC + nMAE + COVERAGE, which is the full metric the
    2026-06-23 band-retirement decision specified.

    Reporting SRCC alone was a real hole, for two reasons:

    nMAE  — SRCC is RANK-ONLY and therefore blind to systematic bias. A model whose
            predictions are uniformly 1.5x too large ranks perfectly and scores a high
            SRCC. That is not hypothetical: the aux-head pooling bug (2026-07-27) inflated
            every prediction ~1.5x above the training mean and SRCC never flagged it.
            For a paper claiming NUMERIC faithfulness, "are the numbers right" and "are
            they correctly ordered" are different questions and both need answering.
            nMAE = mean|pred-gt| / sd(gt), so 1.0 = as bad as predicting the GT mean.

    coverage — SRCC on the subset the model chose to talk about is not comparable across
            features. f0 at 48% coverage and snr at 100% are different quantities, and a
            model that emitted only on easy clips would look BETTER. With abstention in
            the story, coverage is load-bearing, not a footnote. Never report a bare SRCC.
    """
    meas = meas or {}
    print(f"=== {name}" + (f" [{restrict_stems[1]}]" if restrict_stems else "") + " ===")
    stems = list(gtf.keys()) if restrict_stems is None else restrict_stems[0]
    per = {}
    # TWO coverage denominators, because they answer different questions and only the
    # second is the abstention story:
    #   cov_gt   = emitted / (clips where the TARGET states this feature).  The target
    #              abstains under overlap too, so this measures FIDELITY TO THE ABSTENTION
    #              RULE. It sits at ~1.0 and is NOT evidence of independent judgement.
    #   cov_meas = emitted / (clips where the PHYSICAL MEASUREMENT exists at all, i.e. the
    #              feature CSV has a value).  This is the true emission rate: ~0.5 for the
    #              ill-posed features, because the model withholds them under overlap.
    # Reporting only cov_gt would let "the model abstains" look like "the model always
    # speaks", which is the opposite of the claim.
    # FLOOR = nMAE of the CONSTANT predictor (always emit the GT mean). This is the
    # cheapest question a reviewer can ask -- "is 0.746 actually better than guessing the
    # average?" -- and without it every nMAE is uninterpretable. For a Gaussian the floor
    # is E|X-mu|/sd = sqrt(2/pi) ~ 0.798, so an nMAE near 0.8 means the feature is NOT
    # being predicted at all. Report nMAE and floor side by side, never nMAE alone.
    print(f"  {'feature':<14}{'SRCC':>8}{'  95% CI':>17}{'nMAE':>8}{'floor':>8}{'gain':>7}"
          f"{'cov_gt':>8}{'cov_meas':>10}{'n':>7}")
    for f in FEATS:
        xs, ys = [], []
        n_gt = 0
        for s in stems:
            gv = gtf.get(s, {})
            # Coverage denominator = clips the model was actually ASKED about (present in
            # preds) AND where GT for this feature exists. Using all of gtf instead would
            # divide by the whole train+dev+test description file and report ~0.14 for a
            # feature emitted on every evaluated clip.
            if f not in gv or s not in preds:
                continue
            n_gt += 1
            if f in preds[s]:
                xs.append(preds[s][f]); ys.append(gv[f])
        cov = (len(xs) / n_gt) if n_gt else 0.0
        n_meas = sum(1 for s in stems if s in preds and s in meas and f in meas[s])
        cov_m = (len(xs) / n_meas) if n_meas else float("nan")
        if len(xs) >= 10:
            sd = _stdev(ys)
            nmae = (sum(abs(a - b) for a, b in zip(xs, ys)) / len(xs) / sd) if sd > 1e-9 else float("nan")
            mu = sum(ys) / len(ys)
            floor = (sum(abs(mu - b) for b in ys) / len(ys) / sd) if sd > 1e-9 else float("nan")
            gain = (1.0 - nmae / floor) if floor > 1e-9 else float("nan")
            lo, hi = _boot_ci(xs, ys, srcc)
            per[f] = (srcc(xs, ys), len(xs), nmae, cov, cov_m, floor)
            print(f"  {f:<14}{per[f][0]:>+8.3f} [{lo:+.3f},{hi:+.3f}]{nmae:>8.3f}"
                  f"{floor:>8.3f}{gain:>+6.0%}{cov:>8.3f}{cov_m:>10.3f}{len(xs):>7}")
        else:
            print(f"  {f:<14}{'--':>8}{'--':>8}{'--':>8}{'--':>7}"
                  f"{cov:>8.3f}{cov_m:>10.3f}{len(xs):>7}  (skip, n<10)")
    rob = [per[f][0] for f in ROBUST if f in per]
    mean = sum(rob) / len(rob) if rob else float("nan")
    rob_nmae = [per[f][2] for f in ROBUST if f in per and per[f][2] == per[f][2]]
    rob_cov = [per[f][3] for f in ROBUST if f in per]
    rob_covm = [per[f][4] for f in ROBUST if f in per and per[f][4] == per[f][4]]
    rob_floor = [per[f][5] for f in ROBUST if f in per and per[f][5] == per[f][5]]
    print(f"  >> mean over 5 robust: SRCC {mean:.4f}"
          f" | nMAE {(sum(rob_nmae)/len(rob_nmae) if rob_nmae else float('nan')):.4f}"
          f" | floor {(sum(rob_floor)/len(rob_floor) if rob_floor else float('nan')):.4f}"
          f" | cov_gt {(sum(rob_cov)/len(rob_cov) if rob_cov else float('nan')):.4f}"
          f" | cov_meas {(sum(rob_covm)/len(rob_covm) if rob_covm else float('nan')):.4f}")
    return per, mean


def load_gt_from_csv(csv_path):
    """GT straight from the INSTRUMENT CSV, bypassing the target prose entirely.

    Why this exists: the default path parses GT out of the target text with the SAME
    ClaimParser used on the prediction. That is self-referential -- it measures agreement
    with an LLM-verbalized rendering of the instrument, and it rewards reproducing the
    training distribution's phrasing. Scoring against the CSV asks the stricter and more
    honest question: does the emitted number match what the INSTRUMENT measured?
    Any gap between the two scorings is itself a finding worth reporting.
    """
    import csv as _csv
    import os as _os
    sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "src"))
    from data.feature_set import SUPERVISED_FEATURES
    cols = {n: c for n, c, _ in SUPERVISED_FEATURES}
    out = {}
    for row in _csv.DictReader(open(csv_path)):
        stem = (row.get("filename") or "").rsplit(".", 1)[0]
        d = {}
        for feat, col in cols.items():
            v = row.get(col)
            if v not in (None, "", "nan"):
                try:
                    fv = float(v)
                    if fv == fv:
                        d[feat] = fv
                except ValueError:
                    pass
        out[stem] = d
    return out


def main():
    argv = sys.argv[1:]
    feat_csv = None
    # 2026-08-03: instrument-CSV GT is now the DEFAULT. Scoring against the target prose
    # runs BOTH sides through the same ClaimParser and additionally routes GT through the
    # LLM verbalizer, so parser quirks cancel invisibly and any value the verbalizer
    # mangled becomes "truth" the model is rewarded for reproducing. Measured: the two
    # scorings agree exactly (snr .942 / srmr .851 / f0 .359 either way), so the round-trip
    # is clean -- but the CSV path is the one that states the claim we actually make, and
    # it costs nothing. --gt_from_prose keeps the old path for reporting that agreement.
    gt_from_prose = "--gt_from_prose" in argv
    if gt_from_prose:
        argv.remove("--gt_from_prose")
    gt_from_csv = not gt_from_prose
    if "--gt_from_csv" in argv:
        argv.remove("--gt_from_csv")      # accepted as a no-op for back-compat
    if "--features_csv" in argv:
        i = argv.index("--features_csv")
        feat_csv = argv[i + 1]
        argv = argv[:i] + argv[i + 2:]
    gt_path, pred_paths = argv[0], argv[1:]
    meas = load_measured(feat_csv)
    if meas:
        print(f"[cov_meas] measurement map loaded for {len(meas)} clips from {feat_csv}")
    if gt_from_csv:
        if not feat_csv:
            raise SystemExit(
                "instrument-CSV GT is the default and requires --features_csv.\n"
                "Pass --features_csv <split>.csv, or --gt_from_prose to score against the "
                "target text instead (self-referential; see the note in main())."
            )
        gtf = load_gt_from_csv(feat_csv)
        print(f"[GT] INSTRUMENT CSV ({len(gtf)} clips) — parser used on predictions ONLY, "
              f"so this is not self-referential")
    else:
        gt = json.load(open(gt_path))
        gtf = {k: parse_feats(v) for k, v in gt.items()}
        print(f"[GT] parsed from target prose ({len(gtf)} clips) — SELF-REFERENTIAL, "
              f"same parser on GT and prediction; use --gt_from_csv for the strict scoring")
    from collections import Counter
    print("condition counts:", dict(Counter(cond(s) for s in gtf)))
    means = {}
    for pp in pred_paths:
        name = pp.split("/")[-2]
        preds = load_preds(pp)
        _, m = score(gtf, preds, name, meas=meas)
        means[name] = m
        # per-condition (only if that condition has clips)
        by = {}
        for s in gtf:
            by.setdefault(cond(s), []).append(s)
        for c, ss in by.items():
            if len(ss) >= 30:
                score(gtf, preds, name, restrict_stems=(ss, c), meas=meas)
        print()
    if len(means) >= 2:
        ks = list(means)
        print(f"HEADLINE matched-TEST srcc_robust: "
              + " | ".join(f"{k}={means[k]:.4f}" for k in ks))


if __name__ == "__main__":
    main()
