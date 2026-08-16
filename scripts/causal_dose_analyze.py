#!/usr/bin/env python3
"""causal_dose_analyze.py — read out the dose-response causal test.

GATE FIRST: if the overlap fraction does not FALL with alpha, the intervention did not
do what the design assumes and the SENSITIVE arm is invalid — exactly the failure that
silently killed the earlier f0 arm (|dGT| stuck at 0.00). Check the intervention before
reading any verdict.

INVARIANT features (speaking_rate, pause_count, pause_rate, jitter, shimmer, hnr)
  GT is fixed on the clean s1 stem. Per clip we take spearman(alpha, |yhat(alpha) - GT|).
  NEGATIVE rho = error falls as interference is removed = the model measures s1 from the
  signal. Monotonicity across 5 doses is far stronger than a 2-point contrast, and needs
  no clean control window (the data has none: the best available control window still
  averaged 62.6% overlap).

SENSITIVE features (f0_mean, f0_sd, srmr)
  GT MOVES with alpha, recomputed by the instrument on the edited audio. Endpoint is the
  paired regression of d_yhat on d_GT against alpha=0. A faithful model tracks the new
  truth.
"""
from __future__ import annotations

import argparse
import csv
import json

import numpy as np

INVARIANT = ("speaking_rate", "pause_count", "pause_rate", "jitter", "shimmer", "hnr")
SENSITIVE = ("f0_mean", "f0_sd", "srmr")
# Column names DERIVED from the single source of truth, never retyped. The KEY SELECTION stays
# explicit because it is meaningful (these are the features this test measures); only the
# short-name -> CSV-column mapping is inherited, so a rename in feature_set.py cannot leave this
# file silently pointing at a column that no longer exists. That failure mode is why hnr and
# shimmer were valued in 0 of 39,800 targets.
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "src"))
from data.feature_set import SUPERVISED_FEATURES as _SF  # noqa: E402
_CANON = {(f[0] if isinstance(f, (tuple, list)) else str(f)):
          (f[1] if isinstance(f, (tuple, list)) and len(f) > 1 else f[0]) for f in _SF}

CSV_COL = {k: _CANON[k] for k in
           ("speaking_rate", "pause_count", "pause_rate", "jitter", "shimmer", "hnr")}


def _rank(x):
    o = np.argsort(x, kind="mergesort")
    r = np.empty(x.size, float)
    s = x[o]
    i = 0
    while i < x.size:
        j = i + 1
        while j < x.size and s[j] == s[i]:
            j += 1
        r[o[i:j]] = 0.5 * (i + j - 1)
        i = j
    return r


def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.size < 3 or np.allclose(a, a[0]) or np.allclose(b, b[0]):
        return float("nan")
    ra, rb = _rank(a) - _rank(a).mean(), _rank(b) - _rank(b).mean()
    d = float(np.sqrt((ra ** 2).sum() * (rb ** 2).sum()))
    return float((ra * rb).sum() / d) if d > 0 else float("nan")


def boot(v, n=4000, seed=0):
    v = np.asarray([x for x in v if np.isfinite(x)], float)
    if v.size < 5:
        return float("nan"), float("nan"), float("nan"), v.size
    r = np.random.default_rng(seed)
    m = np.array([r.choice(v, v.size, replace=True).mean() for _ in range(n)])
    return float(v.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5)), v.size


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--features_csv", required=True)
    a = ap.parse_args()

    gt = {r["filename"]: r for r in csv.DictReader(open(a.features_csv))}
    rows = json.load(open(a.results))
    alphas = sorted({float(k) for r in rows for k in r["doses"]})
    print(f"n clips = {len(rows)}   alphas = {alphas}\n")

    # ---------------- GATE: did the intervention actually work? ----------------
    print("=== GATE — overlap fraction by dose (must FALL; else the SENSITIVE arm is invalid) ===")
    ok_gate = True
    prev = None
    for al in alphas:
        v = [r["doses"][str(al)].get("ovl_frac") for r in rows if str(al) in r["doses"]]
        v = np.array([x for x in v if x is not None and np.isfinite(x)], float)
        m = float(v.mean()) if v.size else float("nan")
        flag = ""
        if prev is not None and np.isfinite(m) and m > prev + 1e-6:
            flag, ok_gate = "  <-- NOT falling", False
        print(f"    alpha={al:<5} mean overlap fraction = {m:.4f}{flag}")
        prev = m
    print(f"    gate: {'PASS' if ok_gate else 'FAIL — sensitive arm not interpretable'}\n")

    # ---------------- INVARIANT: does error fall monotonically? ----------------
    print("=== INVARIANT FEATURES — GT fixed; per-clip spearman(alpha, |err|) ===")
    print("    NEGATIVE rho = error FALLS as interference is removed = grounding")
    print(f"    {'feature':<15}{'rho(alpha,|err|)':>26}   {'mean |err| by dose':<44} verdict")
    for f in INVARIANT:
        col = CSV_COL[f]
        rhos, by_dose = [], {al: [] for al in alphas}
        for r in rows:
            g = gt.get(r["filename"], {}).get(col)
            try:
                g = float(g)
            except (TypeError, ValueError):
                continue
            xs, es = [], []
            for al in alphas:
                d = r["doses"].get(str(al))
                if not d:
                    continue
                p = d["pred"].get(f)
                if p is None or not np.isfinite(p):
                    continue
                xs.append(al); es.append(abs(p - g)); by_dose[al].append(abs(p - g))
            if len(xs) >= 3:
                rhos.append(spearman(np.array(xs), np.array(es)))
        m, lo, hi, n = boot(rhos)
        if not np.isfinite(m):
            continue
        v = ("GROUNDED (error falls monotonically)" if hi < 0
             else "reads interferer (error RISES)" if lo > 0 else "flat")
        cells = " ".join(f"{np.mean(by_dose[al]):.3f}" if by_dose[al] else " nan" for al in alphas)
        print(f"    {f:<15}{m:+.4f} [{lo:+.4f},{hi:+.4f}]   {cells:<44} {v}   n={n}")

    # ---------------- SENSITIVE: does yhat track the moved GT? ----------------
    print("\n=== SENSITIVE FEATURES — GT moves; paired regression of d_yhat on d_GT ===")
    key = {"f0_mean": "gt_f0_mean", "f0_sd": "gt_f0_sd", "srmr": "gt_srmr"}
    a0 = str(alphas[0])
    for f in SENSITIVE:
        X, Y = [], []
        for r in rows:
            b = r["doses"].get(a0)
            if not b:
                continue
            g0, p0 = b.get(key[f]), b["pred"].get(f)
            if g0 is None or p0 is None or not np.isfinite(g0):
                continue
            for al in alphas[1:]:
                d = r["doses"].get(str(al))
                if not d:
                    continue
                g1, p1 = d.get(key[f]), d["pred"].get(f)
                if g1 is None or p1 is None or not np.isfinite(g1) or not np.isfinite(p1):
                    continue
                X.append(g1 - g0); Y.append(p1 - p0)
        if len(X) < 20:
            print(f"    {f:<9} n={len(X)} too few")
            continue
        X, Y = np.array(X), np.array(Y)
        if X.std() < 1e-9:
            print(f"    {f:<9} dGT CONSTANT — the intervention did not move the truth")
            continue
        slope = float(np.polyfit(X, Y, 1)[0])
        rp = float(np.corrcoef(X, Y)[0, 1])
        rng = np.random.default_rng(0)
        bs = [np.corrcoef(X[i], Y[i])[0, 1] for i in
              (rng.integers(0, len(X), len(X)) for _ in range(2000)) if X[i].std() > 1e-9]
        lo, hi = (np.percentile(bs, [2.5, 97.5]) if bs else (float("nan"),) * 2)
        v = "TRACKS GT" if lo > 0 else ("ANTI-tracks" if hi < 0 else "no relation")
        print(f"    {f:<9} n={len(X):<5} slope={slope:+.3f}  r={rp:+.3f} [{lo:+.3f},{hi:+.3f}]  "
              f"|dGT| median={np.median(np.abs(X)):.3f}   {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
