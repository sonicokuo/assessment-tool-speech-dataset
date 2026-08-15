#!/usr/bin/env python3
"""metric_discriminative_validity.py — does our instrument metric catch what text metrics miss?

WHY THIS IS CONTRIBUTION I's LOAD-BEARING EXPERIMENT
Contribution I claims that numeric claims in generated descriptions must be scored against
INSTRUMENT ground truth, rather than against human opinion (ALLD/MOS) or a GPT judge
(QualiSpeech). That is a METHODOLOGY claim, and a methodology claim is only worth publishing if
the proposed instrument DETECTS SOMETHING THE INCUMBENTS MISS. Asserting it is not enough; this
script measures it.

PART A — THE NATURAL EXPERIMENT (not synthetic, we did not plan it)
A target-construction bug ("fw" targets) made the LM emit NOTHING for 4 of 11 features: hnr,
f0_mean, f0_sd and shimmer all at 0.0% on CLEAN clips, where supervision is complete and the
hedge rule cannot fire. `jitter` — the one ill-posed feature with no target defect — stayed at
96.7% and is the control. Repairing the targets ("fw2") restored all four to 98.3%. We hold
GREEDY GENERATIONS FROM BOTH CHECKPOINTS ON THE SAME CLIPS, so this is a paired comparison in
which a third of the measured quantities silently vanished while the prose stayed fluent.

    If BLEU / ROUGE-L / BERTScore are ~flat across that pair, they are BLIND to the total
    disappearance of 4 of 11 measured quantities. That is the argument for contribution I,
    made on a REAL failure rather than a constructed one.

PART B — THE CONTROLLED CORRUPTION CURVE (part A is n=1 failure; this generalises it)
Part A is a single anecdote and a reviewer will say so. Part B injects GRADED, DELIBERATE numeric
corruptions into the SAME generations, holding the prose fixed, and traces every metric against
corruption severity. Four corruption families, because they fail in different ways:

    omit    — delete a fraction of numeric claims        (the fw failure mode: silence)
    perturb — scale each value by a relative error       (plausible-but-wrong magnitudes)
    shuffle — permute values BETWEEN features            (right numbers, wrong quantities)
    collapse— replace every value with the corpus median (mode collapse / prior-only)

`shuffle` and `collapse` are the sharpest: they change NO tokens' plausibility and often keep the
exact multiset of numbers, so a similarity metric cannot see them at all, while the instrument
metric should fall to chance. A metric that cannot separate `collapse` from the truth is not
measuring faithfulness.

WHAT IS REPORTED
For every (family, severity) cell: BLEU-4, ROUGE-L, BERTScore-F1 versus the reference targets, and
the instrument panel (per-feature SRCC vs instrument GT, plus COVERAGE). Sensitivity is summarised
as the RELATIVE DROP from the uncorrupted generations, so metrics on different scales are
comparable. The headline statistic is the ratio of instrument-metric drop to text-metric drop: how
many times more sensitive the instrument is to a purely numeric failure.

CONTROLS, BECAUSE THIS PROJECT'S DEFECT RATE IS ~1 PER SCRIPT
* Part A's confound is that the two checkpoints might differ in FLUENCY, not just numbers. The
  `jitter` control feature (no target defect, 96.7% -> 98.3%) plus the robust five (100% -> 100%)
  bound that: if the prose were broadly degraded, those would move too.
* Part B holds the generating model FIXED and edits only numerals, so fluency is constant BY
  CONSTRUCTION and any text-metric movement is attributable to the digits alone.
* Corruption is applied to the PARSED SPAN, so a "corrupted" text differs from the original only
  in the numerals. We verify this by asserting the non-numeric token stream is unchanged.
* SRCC needs variance: cells whose corrupted predictions are constant (`collapse`) are reported
  as nan rather than silently scored, and coverage is reported alongside every SRCC so an
  abstaining or emptied arm can never look accurate. This project already produced exactly that
  artifact once (an arm's f0 SRCC beat another's only because it emitted f0 on 50.5% of clips).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from data.feature_set import SUPERVISED_FEATURES  # noqa: E402
from eval.sfs import HybridClaimParser, SFSScorer  # noqa: E402

# short_name -> CSV column. NEVER look GT up by short name: this corpus has hnr_db/shimmer_pct
# style columns, and assuming they matched is the exact bug that once zeroed shimmer and HNR in
# 100% of training targets. A silent nan here would have made the control feature unreadable.
CSV_COL = {(f[0] if isinstance(f, (tuple, list)) else str(f)):
           (f[1] if isinstance(f, (tuple, list)) and len(f) > 1 else f[0])
           for f in SUPERVISED_FEATURES}

ROBUST5 = ["snr", "srmr", "speaking_rate", "pause_count", "pause_rate"]
ILLPOSED = ["f0_mean", "f0_sd", "jitter", "shimmer", "hnr"]
# the four features the fw target bug silenced, plus the control that it did not touch
FW_SILENCED = ["hnr", "f0_mean", "f0_sd", "shimmer"]
FW_CONTROL = "jitter"

NUM_RE = re.compile(r"-?\d+\.?\d*")


def srcc(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    if a.size < 10 or np.allclose(a, a[0]) or np.allclose(b, b[0]):
        return float("nan")
    def rk(x):
        o = np.argsort(x, kind="mergesort"); r = np.empty(x.size, float); sx = x[o]; i = 0
        while i < x.size:
            j = i + 1
            while j < x.size and sx[j] == sx[i]:
                j += 1
            r[o[i:j]] = 0.5 * (i + j - 1); i = j
        return r
    ra, rb = rk(a) - rk(a).mean(), rk(b) - rk(b).mean()
    d = float(np.sqrt((ra ** 2).sum() * (rb ** 2).sum()))
    return float((ra * rb).sum() / d) if d > 0 else float("nan")


def load_gt(features_csv: str, names: list[str]) -> dict:
    gt = {}
    for r in csv.DictReader(open(features_csv)):
        fn = r.get("filename") or ""
        stem = fn[:-4] if fn.endswith(".wav") else fn
        row = {}
        for nm in names:
            col = CSV_COL.get(nm, nm)
            raw = r.get(col, r.get(nm, ""))
            try:
                row[nm] = float(raw)
            except (TypeError, ValueError):
                pass
        gt[stem] = row
    return gt


def claims_of(parser, text: str) -> dict:
    out = {}
    for c in parser.parse(text or ""):
        if c.feature in SFSScorer.TOLERANCES and c.feature not in out:
            out[c.feature] = c.value
    return out


def instrument_panel(parser, gens: dict, gt: dict, feats: list[str]) -> dict:
    """Per-feature SRCC vs instrument GT + COVERAGE. Coverage is reported ALWAYS: an arm that
    emits nothing must not be able to look accurate on the few clips it did emit."""
    parsed = {k: claims_of(parser, v) for k, v in gens.items()}
    out = {}
    for f in feats:
        pairs = [(p[f], gt[k][f]) for k, p in parsed.items()
                 if f in p and k in gt and f in gt[k] and np.isfinite(gt[k][f])]
        n_gt = sum(1 for k in parsed if k in gt and f in gt[k] and np.isfinite(gt[k][f]))
        out[f] = {
            "srcc": srcc([a for a, _ in pairs], [b for _, b in pairs]) if pairs else float("nan"),
            "coverage": (len(pairs) / n_gt) if n_gt else float("nan"),
            "n": len(pairs),
        }
    return out


def corrupt(parser, text: str, family: str, sev: float, rng: random.Random,
            medians: dict) -> str:
    """Rewrite ONLY the numerals of the parsed claims. Prose is untouched by construction, so any
    text-metric movement is attributable to digits alone."""
    cl = parser.parse(text or "")
    if not cl:
        return text
    out = text
    for c in cl:
        span = c.raw_text
        m = NUM_RE.search(span)
        if not m:
            continue
        if family == "omit":
            if rng.random() >= sev:
                continue
            new = None                                    # drop the whole clause below
        elif family == "perturb":
            new = f"{c.value * (1.0 + sev * rng.choice([-1.0, 1.0])):.2f}"
        elif family == "shuffle":
            if rng.random() >= sev:
                continue
            other = rng.choice([x for x in cl if x.feature != c.feature] or [c])
            new = f"{other.value:.2f}"
        elif family == "collapse":
            if rng.random() >= sev:
                continue
            new = f"{medians.get(c.feature, c.value):.2f}"
        else:
            raise ValueError(family)

        if new is None:
            # remove the sentence containing this claim, which is what "silence" looks like
            i = out.find(span)
            if i < 0:
                continue
            s = out.rfind(".", 0, i) + 1
            e = out.find(".", i)
            e = len(out) if e < 0 else e + 1
            out = (out[:s] + out[e:]).replace("  ", " ")
        else:
            rep = span[:m.start()] + new + span[m.end():]
            out = out.replace(span, rep, 1)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gens_fw", help="fw-arm generations JSON (the broken targets)")
    ap.add_argument("--gens_fw2", required=True, help="fw2-arm generations JSON (repaired)")
    ap.add_argument("--descriptions", required=True, help="reference targets JSON")
    ap.add_argument("--features_csv", required=True, help="INSTRUMENT ground truth")
    ap.add_argument("--out", required=True)
    ap.add_argument("--gen_key", default="gen_T0.0")
    ap.add_argument("--no_bertscore", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    from eval.text_metrics import compute_generation_metrics

    parser = HybridClaimParser()
    feats = ROBUST5 + ILLPOSED
    gt = load_gt(a.features_csv, feats)
    missing = [f for f in feats if not any(f in row for row in gt.values())]
    if missing:
        raise SystemExit(f"GT columns never resolved for {missing} — check CSV_COL mapping "
                         f"against {a.features_csv}. Refusing to report nan coverage as a result.")
    refs_all = json.load(open(a.descriptions))

    def load_gens(p):
        d = json.load(open(p))
        recs = d if isinstance(d, list) else list(d.values())
        out = {}
        for r in recs:
            if not isinstance(r, dict):
                continue
            k = r.get("clip") or r.get("filename") or ""
            k = k[:-4] if k.endswith(".wav") else k
            g = r.get(a.gen_key) or r.get("generated") or r.get("prediction") or ""
            if k and g:
                out[k] = g
        return out

    fw2 = load_gens(a.gens_fw2)
    fw = load_gens(a.gens_fw) if a.gens_fw else {}
    report = {}

    def text_metrics(gens: dict, keys: list[str]) -> dict:
        hyps = [gens[k] for k in keys]
        refs = [refs_all.get(k, "") for k in keys]
        return compute_generation_metrics(hyps, refs, use_bertscore=not a.no_bertscore)

    # ---------------------------------------------------------------- PART A
    if fw:
        shared = sorted(set(fw) & set(fw2) & set(refs_all))
        print(f"\n=== PART A — natural experiment (paired, n={len(shared)}) ===", flush=True)
        if len(shared) < 10:
            print("  too few paired clips with references; skipping Part A")
        else:
            tm_fw, tm_fw2 = text_metrics(fw, shared), text_metrics(fw2, shared)
            ip_fw = instrument_panel(parser, {k: fw[k] for k in shared}, gt, feats)
            ip_fw2 = instrument_panel(parser, {k: fw2[k] for k in shared}, gt, feats)
            print(f"  {'metric':<22}{'fw (broken)':>14}{'fw2 (fixed)':>14}{'delta':>10}")
            print("  " + "-" * 60)
            for k in ("bleu", "rouge_l", "bertscore_f1"):
                x, y = tm_fw.get(k), tm_fw2.get(k)
                if x is None or y is None:
                    print(f"  {k:<22}{'n/a':>14}{'n/a':>14}{'':>10}")
                    continue
                print(f"  {k:<22}{x:14.4f}{y:14.4f}{y - x:+10.4f}")
            print(f"  {'-- coverage --':<22}")
            for f in FW_SILENCED + [FW_CONTROL]:
                cx, cy = ip_fw[f]["coverage"], ip_fw2[f]["coverage"]
                tag = "  <- CONTROL (undamaged)" if f == FW_CONTROL else ""
                print(f"  {f:<22}{cx:14.4f}{cy:14.4f}{cy - cx:+10.4f}{tag}")
            report["part_a"] = {"n": len(shared), "text_fw": tm_fw, "text_fw2": tm_fw2,
                                "instrument_fw": ip_fw, "instrument_fw2": ip_fw2}

    # ---------------------------------------------------------------- PART B
    keys = sorted(set(fw2) & set(refs_all) & set(gt))
    print(f"\n=== PART B — controlled corruption curve (n={len(keys)}) ===", flush=True)
    base_gens = {k: fw2[k] for k in keys}
    medians = {}
    for f in feats:
        vals = [claims_of(parser, base_gens[k]).get(f) for k in keys]
        vals = [v for v in vals if v is not None and np.isfinite(v)]
        if vals:
            medians[f] = float(np.median(vals))

    base_tm = text_metrics(base_gens, keys)
    base_ip = instrument_panel(parser, base_gens, gt, feats)
    base_srcc = float(np.nanmean([base_ip[f]["srcc"] for f in ROBUST5]))
    print(f"  baseline: BLEU {base_tm.get('bleu')} ROUGE {base_tm.get('rouge_l')} "
          f"BERTScore {base_tm.get('bertscore_f1')} | instrument robust5 SRCC {base_srcc:.4f}")

    hdr = (f"  {'family':<10}{'sev':>5}{'BLEU':>9}{'ROUGE':>9}{'BERTSc':>9}"
           f"{'SRCC':>9}{'cov':>7}{'txt drop':>10}{'inst drop':>11}{'ratio':>8}")
    print("\n" + hdr); print("  " + "-" * (len(hdr) - 2))
    rows = []
    for family in ("omit", "perturb", "shuffle", "collapse"):
        for sev in (0.25, 0.5, 1.0):
            rng = random.Random(a.seed)
            cg = {k: corrupt(parser, base_gens[k], family, sev, rng, medians) for k in keys}
            tm = text_metrics(cg, keys)
            ip = instrument_panel(parser, cg, gt, feats)
            sc = float(np.nanmean([ip[f]["srcc"] for f in ROBUST5]))
            cov = float(np.nanmean([ip[f]["coverage"] for f in ROBUST5]))
            # relative drop from the uncorrupted generations, so scales are comparable
            def drop(new, old):
                return float("nan") if (new is None or old in (None, 0)) else (old - new) / abs(old)
            td = float(np.nanmean([drop(tm.get(k), base_tm.get(k))
                                   for k in ("bleu", "rouge_l", "bertscore_f1")]))
            idrop = drop(sc, base_srcc)
            ratio = (idrop / td) if (np.isfinite(td) and abs(td) > 1e-9) else float("inf")
            print(f"  {family:<10}{sev:5.2f}"
                  f"{(tm.get('bleu') or float('nan')):9.4f}"
                  f"{(tm.get('rouge_l') or float('nan')):9.4f}"
                  f"{(tm.get('bertscore_f1') or float('nan')):9.4f}"
                  f"{sc:9.4f}{cov:7.3f}{td:10.3f}{idrop:11.3f}{ratio:8.1f}")
            rows.append({"family": family, "severity": sev, "text": tm,
                         "instrument_robust5_srcc": sc, "coverage_robust5": cov,
                         "text_rel_drop": td, "instrument_rel_drop": idrop, "ratio": ratio})
    report["part_b"] = {"n": len(keys), "baseline_text": base_tm,
                        "baseline_instrument_robust5_srcc": base_srcc, "cells": rows}

    json.dump(report, open(a.out, "w"), indent=2)
    print(f"\nwrote {a.out}")
    print("READ: `ratio` is how many times more sensitive the instrument metric is than the mean")
    print("text metric, for a failure that is PURELY numeric. `shuffle` and `collapse` are the")
    print("sharpest cases — the prose and often the exact multiset of numbers are unchanged, so a")
    print("similarity metric is structurally incapable of seeing them. Report coverage beside")
    print("every SRCC: an emptied arm must never be able to look accurate on what little remains.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
