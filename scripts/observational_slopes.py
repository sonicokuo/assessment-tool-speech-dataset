#!/usr/bin/env python3
"""observational_slopes.py — b_obs, the benchmark the causal probe must be scored against.

THE FIFTH TYPE ERROR THIS FIXES
`pitch_intervention.py` benchmarks its partial derivatives against +1 -- the response of a
PERFECT estimator. But the aux head is MSE/NLL-trained, so its optimum is E[y | audio], and the
regression slope of E[y|x] on y is NOT 1: by the tower property Cov(E[y|x], y) = Var(E[y|x]),
so the slope is Var(E[y|x])/Var(y) ~ R^2 (regression dilution, Spearman 1904; shrinkage of
predictions, Copas JRSS-B 45, 1983).

A PERFECTLY GROUNDED estimator with Pearson r ~ 0.31 is therefore EXPECTED to show a causal
slope near 0.10, not 1.0. Scoring against 1 makes an accuracy limit look like a grounding
failure -- the same type mismatch as the previous four errors, and again in the pessimistic
direction.

So the correct benchmark for the interventional slope is the OBSERVATIONAL slope

    b_obs = Cov(yhat, y) / Var(y)

on natural test clips. The comparison interventional-vs-observational is the textbook
operationalisation of "is this association causal":
  * b_int ~ b_obs   -> the association is exactly as causal as it is predictive
  * b_int >  b_obs  -> the model responds to the intervention MORE than its cross-sectional
                       accuracy predicts; the low SRCC is calibration/bias, not deafness
  * b_int <  b_obs  -> the observed association is partly CONFOUNDED (e.g. by speaker
                       covariates) and does not survive intervention -- a grounding finding

Also emits the WITHIN-SPEAKER slope, because the intervention is a within-clip contrast and the
matching observational quantity must remove between-speaker variation.

AND IT SETTLES A UNIT QUESTION the probe depends on: if the aux head emitted normalised values
while GT deltas are in raw Hz, every interventional slope would be off by that feature's scale.
A raw-unit head gives b_obs of order R^2; a z-scored head would show a slope smaller by the
feature's standard deviation. The printed `slope_raw` vs `corr` pair makes this visible.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from data.feature_set import SUPERVISED_FEATURES  # noqa: E402


def speaker_of(stem: str) -> str:
    m = re.match(r"(\d+)-", stem)
    return m.group(1) if m else stem


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True, help="inference_results.json (has aux_mean)")
    ap.add_argument("--features_csv", required=True)
    a = ap.parse_args()

    names = [f[0] if isinstance(f, (tuple, list)) else str(f) for f in SUPERVISED_FEATURES]
    cols = [f[1] if isinstance(f, (tuple, list)) and len(f) > 1 else f[0] for f in SUPERVISED_FEATURES]

    gt: dict[str, dict[str, float]] = {}
    for r in csv.DictReader(open(a.features_csv)):
        fn = r.get("filename") or ""
        stem = fn[:-4] if fn.endswith(".wav") else fn
        row = {}
        for nm, cl in zip(names, cols):
            v = r.get(cl, r.get(nm, ""))
            try:
                row[nm] = float(v)
            except (TypeError, ValueError):
                pass
        gt[stem] = row

    res = json.load(open(a.results))
    Y: dict[str, list[float]] = {n: [] for n in names}
    P: dict[str, list[float]] = {n: [] for n in names}
    S: dict[str, list[str]] = {n: [] for n in names}
    n_noaux = 0
    for r in res:
        fn = r.get("filename") or ""
        stem = fn[:-4] if fn.endswith(".wav") else fn
        aux = r.get("aux_mean")
        if not aux:
            n_noaux += 1
            continue
        base = stem[:-8] if stem.endswith("_s1clean") else stem   # clean twins share GT rows
        g = gt.get(stem) or gt.get(base) or {}
        for i, nm in enumerate(names):
            if nm in g and i < len(aux) and np.isfinite(g[nm]) and np.isfinite(aux[i]):
                Y[nm].append(g[nm]); P[nm].append(float(aux[i])); S[nm].append(speaker_of(stem))

    print(f"clips={len(res)}  without aux_mean={n_noaux}\n")
    print(f"{'feature':<15}{'n':>6}{'corr':>8}{'slope_obs':>11}{'within_spk':>12}"
          f"{'sd(yhat)':>10}{'sd(y)':>10}{'mean(yhat)':>12}{'mean(y)':>10}")
    print("-" * 94)
    out = {}
    for nm in names:
        y, p = np.asarray(Y[nm], float), np.asarray(P[nm], float)
        if y.size < 50 or y.std() < 1e-9 or p.std() < 1e-9:
            print(f"{nm:<15}{y.size:>6}   insufficient")
            continue
        r_pearson = float(np.corrcoef(p, y)[0, 1])
        b = float(np.cov(p, y, ddof=1)[0, 1] / np.var(y, ddof=1))
        # within-speaker: centre both series per speaker, which is the contrast the
        # intervention actually makes (a change within one clip, not across speakers)
        spk = np.asarray(S[nm])
        yc, pc = y.astype(float).copy(), p.astype(float).copy()
        for u in np.unique(spk):
            m = spk == u
            if m.sum() >= 2:
                yc[m] -= yc[m].mean(); pc[m] -= pc[m].mean()
            else:
                yc[m] = 0.0; pc[m] = 0.0
        bw = (float(np.cov(pc, yc, ddof=1)[0, 1] / np.var(yc, ddof=1))
              if np.var(yc, ddof=1) > 1e-12 else float("nan"))
        print(f"{nm:<15}{y.size:>6}{r_pearson:>8.3f}{b:>11.4f}{bw:>12.4f}"
              f"{p.std():>10.2f}{y.std():>10.2f}{p.mean():>12.2f}{y.mean():>10.2f}")
        out[nm] = {"n": int(y.size), "corr": r_pearson, "b_obs": b, "b_within": bw}

    print("\nREAD: the causal probe's partial derivatives must be compared to b_obs (or")
    print("b_within), NOT to +1. b_int > b_obs means the intervention moves the model MORE")
    print("than its accuracy predicts -> the association is causal, and low SRCC is")
    print("calibration. b_int < b_obs means the observational association is confounded.")
    print("\nUNITS CHECK: if mean(yhat) and sd(yhat) are on the same scale as mean(y)/sd(y),")
    print("the head emits RAW units and the interventional slopes need no rescaling.")
    json.dump(out, open("observational_slopes.json", "w"), indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
