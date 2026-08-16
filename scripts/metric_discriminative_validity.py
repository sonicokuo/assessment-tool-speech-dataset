#!/usr/bin/env python3
"""metric_discriminative_validity.py — does the instrument metric catch failures that
BLEU / ROUGE-L / BERTScore / a numeric-extraction strawman miss?

WHY THIS EXISTS
Contribution I ("instrument-grounded numeric-claim faithfulness") is an EVALUATION
PROTOCOL. A protocol earns a paper only by demonstrating DISCRIMINATIVE VALIDITY:
it must move where the incumbent metrics do not, and be invariant where they move
spuriously. That is a two-sided claim and needs a two-sided experiment.

METHODOLOGICAL PRECEDENT (verified 2026-08-15 from the arXiv abstracts):
  Sai, Dixit, Sheth, Mohan & Khapra, "Perturbation CheckLists for Evaluating NLG
      Evaluation Metrics", EMNLP 2021, arXiv:2109.05771 — perturb the output so that
      quality changes along ONE criterion only; 25 metrics x 6 tasks x 18 criteria.
  Kryscinski, McCann, Xiong & Socher, arXiv:1910.12840 — factual-consistency data
      built by RULE-BASED TRANSFORMATIONS of source sentences.
  Pagnoni, Balachandran & Tsvetkov, FRANK, NAACL 2021, arXiv:2104.13346 — a typology
      of factual errors, used to benchmark factuality metrics.
  Laban, Schnabel, Bennett & Hearst, SummaC, TACL 2021, arXiv:2111.09525.

⚠️ THE NAIVE fw-vs-fw2 EXPERIMENT ARGUES AGAINST US — MEASURED, DO NOT RUN IT THAT WAY.
Scoring both arms against a FIXED COMPLETE reference, BLEU separates them by +25 sBLEU
(paired CI [+25.11,+25.18] at n=600 in the synthetic replica). Four missing clauses are
a large surface change and n-gram metrics see it. Two valid forms instead:
  (A) SHARED-CLAUSE RESTRICTION — score every text metric on only the clauses BOTH arms
      emit. Coverage is then matched by construction, the text metrics are matched, and
      the ONLY moving quantity is the instrument panel on the repaired features.
  (B) OWN-REFERENCE — score each arm against the reference it was TRAINED on, which is
      what train.py's val block and inference.py actually logged. The broken arm scores
      HIGHER. This is the real-world failure and it is documented in our own wandb.

Usage
  # synthetic ladder (no checkpoints needed, CPU, seconds)
  python scripts/metric_discriminative_validity.py ladder --features_csv test.csv -n 600

  # real paired generations (fw vs fw2)
  python scripts/metric_discriminative_validity.py paired \\
      --gen_a $SHARED/temperature_redecode.json      --ref_a $SHARED/descriptions_corrected_fw.json  \\
      --gen_b $SHARED/temperature_redecode_fw2.json  --ref_b $SHARED/descriptions_corrected_fw2.json \\
      --features_csv $SHARED/features_corrected_merged/test.csv

Deps: sacrebleu, rouge-score, scipy; bert-score optional (--bertscore).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import statistics
import sys
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import eval.sfs as sfs  # noqa: E402

# ── canonical clause order + surface form, per build_canonical_descriptions.py:237-262 ──
CLAUSES = [
    ("snr",               "snr_db",                          "The SNR is {} dB."),
    ("srmr",              "srmr",                            "The SRMR is {}."),
    ("hnr",               "hnr",                             "The HNR is {} dB."),
    ("f0_mean",           "f0_mean_hz",                      "The F0 mean is {} Hz."),
    ("f0_sd",             "f0_sd_hz",                        "The F0 standard deviation SD is {} Hz."),
    ("jitter",            "jitter_local_pct",                "The jitter is {} percent."),
    ("shimmer",           "shimmer",                         "The shimmer is {} percent."),
    ("speaking_rate",     "praat_speaking_rate_syl_sec",     "The speaking rate is {} syl/sec."),
    ("articulation_rate", "praat_articulation_rate_syl_sec", "The articulation rate is {} syl/sec."),
    ("pause_count",       "praat_pause_count",               "The pause count is {}."),
    ("pause_rate",        "praat_pause_rate_per_min",        "The pause rate is {} per min."),
    ("overlap_ratio",     "overlap_ratio",                   "The overlap ratio is {}."),
]
# legacy CSV spellings (the pre-2026-07-28 columns) accepted as fallbacks
ALT_COL = {"hnr": "hnr_db", "shimmer": "shimmer_pct"}

ALL_FEATS = [c[0] for c in CLAUSES]
INT_FEATS = frozenset({"pause_count"})
DEAD4 = ("hnr", "f0_mean", "f0_sd", "shimmer")          # the fw target-bug casualties
ILL5 = ("hnr", "f0_mean", "f0_sd", "jitter", "shimmer")  # ILL_POSED_UNDER_OVERLAP
HEDGE_SENT = ("Because the speakers overlap heavily, the F0 mean, F0 standard deviation SD, "
              "jitter, shimmer and HNR cannot be reliably estimated and are not reported.")

# FLUENCY-AXIS control renderings: different surface, identical numbers, parser-compatible.
PARA = {
    "snr": "A signal-to-noise ratio SNR of {} dB was measured.",
    "srmr": "We obtain a reverberation score SRMR of {}.",
    "hnr": "The harmonics-to-noise ratio HNR of {} dB characterises the voice.",
    "f0_mean": "Mean pitch of {} Hz is observed across voiced frames.",
    "f0_sd": "Across the utterance the F0 SD is {} Hz.",
    "jitter": "Cycle-to-cycle period variation, jitter of {} percent, is present.",
    "shimmer": "Amplitude variation, shimmer of {} percent, is present.",
    "speaking_rate": "Delivery proceeds at a speaking rate of {} syl/sec.",
    "articulation_rate": "Excluding pauses, the articulation rate of {} syl/sec applies.",
    "pause_count": "The talker produces {} pauses.",
    "pause_rate": "The pause rate is {} per min.",
    "overlap_ratio": "Concurrent talk covers the clip with a ratio of {}.",
}

NUMRE = re.compile(r"-?\d+\.?\d*")
_P = sfs.ClaimParser()
_A = sfs.AbstentionDetector()


# ── metric backends (fail soft, exactly like src/eval/text_metrics.py) ──────────
def _bleu_corpus(h, r):
    import sacrebleu
    return sacrebleu.corpus_bleu(h, [r]).score


def _bleu_sent(h, r):
    import sacrebleu
    return [sacrebleu.sentence_bleu(a, [b]).score for a, b in zip(h, r)]


def _rouge_l(h, r):
    from rouge_score import rouge_scorer
    sc = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    return [sc.score(b, a)["rougeL"].fmeasure for a, b in zip(h, r)]


def mask_numbers(t):
    """FLUENCY CONTROL: strip every numeral. A number-only corruption leaves this at 100."""
    return NUMRE.sub("<NUM>", t)


def numset_f1(hyp, ref):
    """THE STRAWMAN a reviewer will propose: extract every numeral, compare as multisets.
    Blind to feature/value mis-binding by construction (measured: 0.877 under FEATURE-SWAP)."""
    h, r = NUMRE.findall(hyp), NUMRE.findall(ref)
    if not h or not r:
        return 0.0
    ch, cr = Counter(h), Counter(r)
    inter = sum((ch & cr).values())
    p, rc = inter / sum(ch.values()), inter / sum(cr.values())
    return 2 * p * rc / (p + rc) if (p + rc) else 0.0


# ── instrument metric ──────────────────────────────────────────────────────────
def parse_feats(text):
    out = {}
    for c in _P.parse(text or ""):
        if c.feature not in out:
            out[c.feature] = c.value
    return out


def instrument(preds, gts, feats=ALL_FEATS):
    """band-free SRCC + nMAE + coverage + constant-predictor floor, per feature.

    THREE-WAY COVERAGE, which score_matched_test.py does not currently do: a claim is
    ASSERTED, HEDGED (AbstentionDetector fires) or SILENTLY OMITTED. Conflating the last
    two is the defect the unit-swap row exposes.
    """
    from scipy.stats import spearmanr
    res = {}
    for f in feats:
        xs, ys, n_gt = [], [], 0
        for p, g in zip(preds, gts):
            if f not in g:
                continue
            n_gt += 1
            if f in p:
                xs.append(p[f]); ys.append(g[f])
        if not n_gt:
            continue
        cov = len(xs) / n_gt
        if len(xs) >= 10:
            sd = statistics.pstdev(ys)
            mu = sum(ys) / len(ys)
            nmae = sum(abs(a - b) for a, b in zip(xs, ys)) / len(xs) / sd if sd > 1e-9 else float("nan")
            floor = sum(abs(mu - b) for b in ys) / len(ys) / sd if sd > 1e-9 else float("nan")
            sr = float(spearmanr(xs, ys).correlation)
        else:
            nmae = floor = sr = float("nan")
        res[f] = {"srcc": sr, "nmae": nmae, "cov": cov, "floor": floor, "n": len(xs)}
    return res


def panel(ins, feats):
    sr = [ins[f]["srcc"] for f in feats if f in ins and ins[f]["srcc"] == ins[f]["srcc"]]
    nm = [ins[f]["nmae"] for f in feats if f in ins and ins[f]["nmae"] == ins[f]["nmae"]]
    cv = [ins[f]["cov"] for f in feats if f in ins]
    return ((sum(sr) / len(sr)) if sr else float("nan"),
            (sum(nm) / len(nm)) if nm else float("nan"),
            (sum(cv) / len(cv)) if cv else float("nan"))


# ── rendering + corruption operators ───────────────────────────────────────────
def fmt(feat, v):
    return f"{int(round(v))}" if feat in INT_FEATS else f"{v:.2f}"


def render(vals, para=False, drop=(), unit_swap=(), hedge=False):
    out = []
    for feat, _col, tmpl in CLAUSES:
        if feat in drop or feat not in vals:
            continue
        s = (PARA[feat] if para else tmpl).format(fmt(feat, vals[feat]))
        if feat in unit_swap:
            s = (s.replace(" dB.", " percent.").replace(" Hz.", " percent.")
                  .replace(" syl/sec.", " per min."))
        out.append(s)
    if hedge:
        out.append(HEDGE_SENT)
    return " ".join(out)


def load_rows(csv_path, n):
    rows = []
    for r in csv.DictReader(open(csv_path)):
        d, ok = {"__stem": (r.get("filename") or "").rsplit(".", 1)[0]}, True
        for feat, col, _ in CLAUSES:
            v = r.get(col)
            if v in (None, "", "nan"):
                v = r.get(ALT_COL.get(feat, ""))
            try:
                x = float(v)
                if x != x:
                    raise ValueError
                d[feat] = x
            except (TypeError, ValueError):
                ok = False
                break
        if ok:
            rows.append(d)
        if n and len(rows) >= n:
            break
    return rows


def op_noise(rows, feats, mult, rng):
    sds = {f: statistics.pstdev([d[f] for d in rows]) for f in feats}
    out = []
    for d in rows:
        nd = dict(d)
        for f in feats:
            nd[f] = d[f] + rng.gauss(0, mult * sds[f])
            if f in INT_FEATS:
                nd[f] = max(0.0, nd[f])
        out.append(nd)
    return out


def op_digit_perm(rows, feats, rng):
    """same digit tokens, different value — the corruption n-gram metrics reward."""
    out = []
    for d in rows:
        nd = dict(d)
        for f in feats:
            s = fmt(f, d[f])
            dg = [c for c in s if c.isdigit()]
            rng.shuffle(dg)
            it = iter(dg)
            try:
                nd[f] = float("".join(next(it) if c.isdigit() else c for c in s))
            except Exception:
                pass
        out.append(nd)
    return out


def op_shuffle(rows, feats, rng):
    """destroy clip-specificity, preserve every marginal distribution exactly."""
    out = [dict(d) for d in rows]
    for f in feats:
        col = [d[f] for d in rows]
        rng.shuffle(col)
        for i, d in enumerate(out):
            d[f] = col[i]
    return out


def op_feature_swap(rows, feats, rng):
    """right numbers, wrong slots. Defeats naive numeric extraction by construction."""
    out = []
    for d in rows:
        perm = list(feats)
        rng.shuffle(perm)
        nd = dict(d)
        for src, dst in zip(feats, perm):
            nd[dst] = d[src]
        out.append(nd)
    return out


def op_constant(rows, feats):
    mus = {f: sum(d[f] for d in rows) / len(rows) for f in feats}
    return [dict(d, **mus) for d in rows]


def paired_boot(a, b, n_boot=10000, seed=0, groups=None):
    """paired percentile bootstrap on mean(a)-mean(b); `groups` enables twin-clustering."""
    rng = random.Random(seed)
    d = [x - y for x, y in zip(a, b)]
    if groups is None:
        idxs = [[i] for i in range(len(d))]
    else:
        byg = {}
        for i, g in enumerate(groups):
            byg.setdefault(g, []).append(i)
        idxs = list(byg.values())
    m = sum(d) / len(d)
    s = []
    for _ in range(n_boot):
        pick = [idxs[rng.randrange(len(idxs))] for _ in range(len(idxs))]
        flat = [i for c in pick for i in c]
        s.append(sum(d[i] for i in flat) / len(flat))
    s.sort()
    return m, s[int(0.025 * n_boot)], s[int(0.975 * n_boot)]


def tost(a, b, margin, **kw):
    """equivalence test — the correct form for a BLINDNESS (null) claim.
    EQUIVALENT iff the whole CI lies inside +/- margin."""
    m, lo, hi = paired_boot(a, b, **kw)
    return m, lo, hi, (lo > -margin and hi < margin)


def score_all(hyps, refs, gts, bertscore=False, feats=ALL_FEATS):
    row = {"BLEU": _bleu_corpus(hyps, refs)}
    sb = _bleu_sent(hyps, refs)
    rl = _rouge_l(hyps, refs)
    row["sBLEU"] = sum(sb) / len(sb)
    row["ROUGE_L"] = sum(rl) / len(rl)
    row["maskBLEU"] = _bleu_corpus([mask_numbers(h) for h in hyps],
                                   [mask_numbers(r) for r in refs])
    row["numF1"] = sum(numset_f1(h, r) for h, r in zip(hyps, refs)) / len(hyps)
    if bertscore:
        from bert_score import score as _bs
        _, _, f1 = _bs(hyps, refs, lang="en", rescale_with_baseline=True, verbose=False)
        row["BERTScore"] = float(f1.mean())
    ins = instrument([parse_feats(h) for h in hyps], gts, feats)
    s, n, c = panel(ins, feats)
    row.update({"SRCC": s, "nMAE": n, "cov": c})
    return row, sb, rl, ins


HDR = ["BLEU", "sBLEU", "ROUGE_L", "BERTScore", "maskBLEU", "numF1", "SRCC", "nMAE", "cov"]


def show(name, row):
    cells = "".join(f"{row[k]:>10.4f}" if k in row else f"{'--':>10}" for k in HDR)
    print(f"  {name:<26}{cells}")


def cmd_ladder(a):
    rng = random.Random(a.seed)
    rows = load_rows(a.features_csv, a.n)
    refs = [render(d) for d in rows]
    gts = [parse_feats(r) for r in refs]
    missing = {f: sum(1 for g in gts if f not in g) for f in ALL_FEATS}
    bad = {k: v for k, v in missing.items() if v}
    print(f"[data] n={len(rows)}  parser round-trip failures on the reference: {bad or 'none'}")
    print(f"[note] a non-empty dict above is a PARSER BUG and invalidates every row below.\n")

    systems = [
        ("PERFECT (upper bound)",       [dict(d) for d in rows], {}),
        *[(f"noise {m:>4}sd all", op_noise(rows, ALL_FEATS, m, rng), {})
          for m in (0.05, 0.10, 0.25, 0.50, 1.00, 2.00, 3.00)],
        ("digit-permute all",           op_digit_perm(rows, ALL_FEATS, rng), {}),
        ("FEATURE-SWAP (right nums)",   op_feature_swap(rows, ALL_FEATS, rng), {}),
        ("shuffle across clips",        op_shuffle(rows, ALL_FEATS, rng), {}),
        ("constant predictor",          op_constant(rows, ALL_FEATS), {}),
        ("SIGN FLIP snr",               [dict(d, snr=-d["snr"]) for d in rows], {}),
        ("UNIT SWAP on 4",              [dict(d) for d in rows], {"unit_swap": DEAD4}),
        ("OMIT 1 (hnr)",                [dict(d) for d in rows], {"drop": ("hnr",)}),
        ("OMIT 4 (= the fw bug)",       [dict(d) for d in rows], {"drop": DEAD4}),
        ("OMIT 8 of 12",                [dict(d) for d in rows], {"drop": tuple(ALL_FEATS[2:10])}),
        ("PARAPHRASE (nums exact)",     [dict(d) for d in rows], {"para": True}),
        ("PARAPHRASE + shuffle",        op_shuffle(rows, ALL_FEATS, rng), {"para": True}),
    ]
    print("=== CORRUPTION LADDER: text metrics vs the instrument ===")
    print(f"  {'system':<26}" + "".join(f"{h:>10}" for h in HDR))
    rec = []
    for nm, vals, kw in systems:
        hyps = [render(v, **kw) for v in vals]
        row, _sb, _rl, _ins = score_all(hyps, refs, gts, bertscore=a.bertscore)
        show(nm, row)
        rec.append((nm, row))

    # ordering agreement over the population that all emit 12 numeric clauses
    from scipy.stats import kendalltau
    numeric = [r for r in rec if not r[0].startswith(("OMIT", "PARAPHRASE", "UNIT", "PERFECT"))]
    print("\n=== ORDERING AGREEMENT (the headline blindness statistic) ===")
    print("  population = systems that all emit 12 numeric clauses, so coverage is matched")
    S = [r[1]["SRCC"] for r in numeric]
    keep = [i for i, v in enumerate(S) if v == v]
    for k in ("BLEU", "ROUGE_L", "BERTScore", "numF1", "maskBLEU"):
        if k not in numeric[0][1]:
            continue
        M = [numeric[i][1][k] for i in keep]
        Sk = [S[i] for i in keep]
        inv = tot = 0
        for i in range(len(M)):
            for j in range(i + 1, len(M)):
                if abs(Sk[i] - Sk[j]) < 1e-6:
                    continue
                tot += 1
                inv += (Sk[i] - Sk[j]) * (M[i] - M[j]) < 0
        print(f"  {k:<10} Kendall tau vs SRCC {kendalltau(M, Sk).correlation:+.3f}   "
              f"pairwise inversions {inv}/{tot} = {100*inv/max(tot,1):.1f}%")

    # abstention vs fabrication on high-overlap clips
    hi = [d for d in rows if d.get("overlap_ratio", 0) >= 0.5]
    if len(hi) >= 30:
        print(f"\n=== ABSTENTION vs FABRICATION, {len(hi)} clips at overlap >= 0.5 ===")
        rh = [render(d) for d in hi]
        gh = [parse_feats(x) for x in rh]
        for nm, hyps in (("ABSTAINS (correct)", [render(d, drop=ILL5, hedge=True) for d in hi]),
                         ("FABRICATES (fluent)", [render(v) for v in op_shuffle(hi, ILL5, rng)])):
            row, _, _, ins = score_all(hyps, rh, gh, bertscore=a.bertscore, feats=list(ILL5))
            show(nm, row)
        print("  a text metric that prefers the fabricator cannot be used to evaluate abstention.")


def _load_gen(path):
    """accepts inference_results.json (list of {filename,generated}) or
    temperature_redecode.json (list of {clip, gen_T0.0})."""
    d = json.load(open(path))
    recs = d if isinstance(d, list) else list(d.values())
    out = {}
    for r in recs:
        stem = str(r.get("filename") or r.get("clip") or "").replace(".wav", "").replace(".pt", "")
        g = r.get("generated")
        if g is None:
            for k in r:
                if k.startswith("gen_T"):
                    g = r[k]
                    break
        if stem and g:
            out[stem] = g
    return out


def cmd_paired(a):
    ga, gb = _load_gen(a.gen_a), _load_gen(a.gen_b)
    ra, rb = json.load(open(a.ref_a)), json.load(open(a.ref_b))
    stems = sorted(set(ga) & set(gb) & set(ra) & set(rb))
    print(f"[paired] arm A n={len(ga)}  arm B n={len(gb)}  intersection n={len(stems)}")
    if len(stems) < 100:
        print("[warn] n<100. A BLINDNESS claim is an EQUIVALENCE claim and needs a tight CI;\n"
              "       budget n>=500 paired clips before asserting any null.")
    HA = [ga[s] for s in stems]
    HB = [gb[s] for s in stems]
    RA = [ra[s] for s in stems]
    RB = [rb[s] for s in stems]
    gtsB = [parse_feats(x) for x in RB]          # complete reference defines GT

    print("\n=== FORM 0 (NAIVE — measured to ARGUE AGAINST US; reported for completeness) ===")
    print(f"  {'arm':<26}" + "".join(f"{h:>10}" for h in HDR))
    rowA, sbA, rlA, insA = score_all(HA, RB, gtsB, bertscore=a.bertscore)
    rowB, sbB, rlB, insB = score_all(HB, RB, gtsB, bertscore=a.bertscore)
    show("A vs complete ref", rowA)
    show("B vs complete ref", rowB)
    m, lo, hi = paired_boot(sbB, sbA)
    print(f"  paired sBLEU delta B-A = {m:+.2f} [{lo:+.2f},{hi:+.2f}]"
          f"  -> a large positive here means the surface change is visible; NOT our claim")

    print("\n=== FORM A (SHARED-CLAUSE RESTRICTION — the valid natural experiment) ===")
    emitted_A = {f for f in ALL_FEATS
                 if sum(1 for h in HA if f in parse_feats(h)) > 0.5 * len(HA)}
    shared = [f for f in ALL_FEATS if f in emitted_A]
    dropped = [f for f in ALL_FEATS if f not in shared]
    print(f"  clauses arm A emits: {sorted(shared)}")
    print(f"  clauses arm A never emits (the natural lesion): {sorted(dropped)}")
    if dropped:
        refS = [render(parse_feats(r), drop=tuple(dropped)) for r in RB]
        HBs = [render(parse_feats(h), drop=tuple(dropped)) for h in HB]
        HAs = [render(parse_feats(h), drop=tuple(dropped)) for h in HA]
        print(f"  {'arm (shared clauses only)':<26}" + "".join(f"{h:>10}" for h in HDR))
        rA, sbA2, rlA2, _ = score_all(HAs, refS, gtsB, bertscore=a.bertscore, feats=shared)
        rB, sbB2, rlB2, _ = score_all(HBs, refS, gtsB, bertscore=a.bertscore, feats=shared)
        show("A", rA)
        show("B", rB)
        for nm, xa, xb, marg in (("sBLEU", sbB2, sbA2, a.margin_bleu),
                                 ("ROUGE_L", rlB2, rlA2, a.margin_rouge)):
            m, lo, hi, eq = tost(xb, xa, marg)
            print(f"  TOST {nm:<8} delta {m:+.4f} [{lo:+.4f},{hi:+.4f}] margin +/-{marg} -> "
                  f"{'EQUIVALENT (blind)' if eq else 'NOT equivalent'}")
        print("  instrument panel on the DROPPED features (the whole effect):")
        for nm, H in (("A", HA), ("B", HB)):
            ins = instrument([parse_feats(h) for h in H], gtsB, dropped)
            s, n, c = panel(ins, dropped)
            print(f"    arm {nm}: coverage {c:.4f}  SRCC {s:.4f}  nMAE {n:.4f}")

    print("\n=== FORM B (OWN-REFERENCE — what train.py/inference.py actually logged) ===")
    print(f"  {'arm vs its OWN ref':<26}" + "".join(f"{h:>10}" for h in HDR))
    gtsA = [parse_feats(x) for x in RA]
    rA2, sbA3, rlA3, _ = score_all(HA, RA, gtsA, bertscore=a.bertscore)
    show("A vs ref A", rA2)
    show("B vs ref B", rowB)
    m, lo, hi = paired_boot(sbA3, sbB)
    print(f"  paired sBLEU delta A-B = {m:+.2f} [{lo:+.2f},{hi:+.2f}]"
          f"  {'<<< THE BROKEN ARM SCORES HIGHER' if lo > 0 else ''}")
    print("  instrument metric on the SAME generations, complete 12-feature panel:")
    for nm, ins in (("A", insA), ("B", insB)):
        s, n, c = panel(ins, ALL_FEATS)
        print(f"    arm {nm}: coverage {c:.4f}  SRCC {s:.4f}  nMAE {n:.4f}")

    print("\n=== FLUENCY CONTROL (rules out 'the checkpoints just differ in fluency') ===")
    for nm, H in (("A", HA), ("B", HB)):
        mb = _bleu_corpus([mask_numbers(x) for x in H], [mask_numbers(x) for x in RB])
        ln = sum(len(x.split()) for x in H) / len(H)
        ttr = sum(len(set(x.split())) / max(len(x.split()), 1) for x in H) / len(H)
        print(f"  arm {nm}: masked-number BLEU {mb:6.2f}   mean length {ln:6.1f} tok   TTR {ttr:.3f}")
    print("  masked-number BLEU equal => the arms are fluency-matched and every text-metric")
    print("  difference is attributable to numeric content, not to prose quality.")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    L = sub.add_parser("ladder")
    L.add_argument("--features_csv", required=True)
    L.add_argument("-n", type=int, default=600)
    L.add_argument("--seed", type=int, default=0)
    L.add_argument("--bertscore", action="store_true")
    L.set_defaults(fn=cmd_ladder)
    P = sub.add_parser("paired")
    P.add_argument("--gen_a", required=True, help="the DEFECTIVE arm (fw)")
    P.add_argument("--ref_a", required=True, help="the reference arm A was TRAINED on")
    P.add_argument("--gen_b", required=True, help="the REPAIRED arm (fw2)")
    P.add_argument("--ref_b", required=True, help="the reference arm B was trained on (complete)")
    P.add_argument("--features_csv", default=None)
    P.add_argument("--bertscore", action="store_true")
    P.add_argument("--margin_bleu", type=float, default=1.0)
    P.add_argument("--margin_rouge", type=float, default=0.01)
    P.set_defaults(fn=cmd_paired)
    a = ap.parse_args()
    return a.fn(a) or 0


if __name__ == "__main__":
    raise SystemExit(main())
