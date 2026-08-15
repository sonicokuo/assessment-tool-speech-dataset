#!/usr/bin/env python3
"""causal_remix_analyze.py — read out the corrected causal test.

TWO ENDPOINTS, chosen by GT provenance (verified from source):

CLEAN-STEM FEATURES (srmr, speaking_rate, pause_count, pause_rate, jitter, shimmer, hnr)
  GT is measured on the s1 stem, so attenuating s2 leaves the TRUTH fixed while giving
  the model a CLEANER VIEW of it. Endpoint is therefore the change in ERROR, not the
  change in prediction:

      d_err = |yhat' - GT| - |yhat - GT|

  d_err < 0 in HIGH, and more negative than in LOW  -> the model measures s1 from local
  evidence and interference was degrading it: GROUNDING.
  d_err ~ 0        -> not using the newly-clean evidence.
  d_err > 0        -> the model reads the INTERFERER as signal for an s1 property.
  HIGH ~ LOW       -> responds to global change, not to evidence: no localisation.

F0 (f0_mean, f0_sd)
  GT is measured on the MIXTURE outside overlap, so it MOVES. Endpoint is the paired
  regression of d_yhat on d_GT: a faithful model tracks the new truth. Slope near 1 and
  positive correlation = tracking. This is the continuous endpoint that replaces the
  binary win-rate (n=120 binary had CI half-width ~0.09; 0.55 vs 0.50 needed n~780).

The HIGH-vs-LOW contrast is the paired control: attenuating s2 inside the most-overlapped
window removes real interference; doing it inside the least-overlapped window removes
almost nothing.
"""
from __future__ import annotations

import argparse
import csv
import json

import numpy as np

# srmr GT is recomputed on the DEGRADED MIXTURE (regenerate_corrected.py:113), so it is
# NOT invariant under an s2 edit. Scoring it against a fixed GT produced a spurious
# "reads the interferer" verdict — the same stale-GT error that invalidated the f0 arm.
CLEAN_STEM = ("speaking_rate", "pause_count", "pause_rate", "jitter", "shimmer", "hnr")
MIXTURE = ("f0_mean", "f0_sd", "srmr")
CSV_COL = {"srmr": "srmr", "speaking_rate": "praat_speaking_rate_syl_sec",
           "pause_count": "praat_pause_count", "pause_rate": "praat_pause_rate_per_min",
           "jitter": "jitter_local_pct", "shimmer": "shimmer", "hnr": "hnr",
           "f0_mean": "f0_mean_hz", "f0_sd": "f0_sd_hz"}


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
    print(f"n clips = {len(rows)}\n")

    print("=== CLEAN-STEM FEATURES — GT is FIXED; endpoint = change in ERROR ===")
    print("    d_err < 0 means the prediction moved TOWARD the truth when interference was removed")
    print(f"    {'feature':<15}{'d_err HIGH':>26}{'d_err LOW':>26}   verdict")
    verdicts = {}
    for f in CLEAN_STEM:
        col = CSV_COL[f]
        de = {"high": [], "low": []}
        for r in rows:
            g = gt.get(r["filename"], {}).get(col)
            try:
                g = float(g)
            except (TypeError, ValueError):
                continue
            b = r.get("base", {}).get(f)
            if b is None:
                continue
            for tag in ("high", "low"):
                d = r.get(f"d_{tag}", {}).get(f)
                if d is None:
                    continue
                de[tag].append(abs(b + d - g) - abs(b - g))
        hm, hl, hh, nh = boot(de["high"])
        lm, ll, lh, nl = boot(de["low"])
        if not np.isfinite(hm):
            continue
        # HIGH vs LOW must be a PAIRED test on the same clips. Comparing two independent
        # means (hm < lm) called speaking_rate "GROUNDED" on a difference of 0.002 —
        # far inside either CI. The paired difference is the actual statistic.
        pair = [h - l for h, l in zip(de["high"], de["low"]) if np.isfinite(h) and np.isfinite(l)]
        pm, pl, ph, npair = boot(pair)
        if hl > 0:
            v = "READS INTERFERER (error RISES)"
        elif hh < 0 and ph < 0:
            v = "GROUNDED (error drops, HIGH>LOW paired)"
        elif hh < 0 and pl > 0:
            v = "BACKWARDS (error drops MORE in the control)"
        elif hh < 0:
            v = "error drops, but HIGH=LOW (no localisation)"
        else:
            v = "flat (no effect detected)"
        verdicts[f] = v
        print(f"    {f:<15}{hm:+.4f} [{hl:+.4f},{hh:+.4f}]{lm:+.4f} [{ll:+.4f},{lh:+.4f}]"
              f"{pm:+.4f} [{pl:+.4f},{ph:+.4f}]   {v}")

    print("\n=== F0 — GT MOVES; endpoint = paired regression of d_yhat on d_GT ===")
    print("    a faithful model TRACKS the new truth: slope>0, positive correlation")
    for f in MIXTURE:
        for tag in ("high", "low"):
            X, Y = [], []
            for r in rows:
                dg = r.get(f"dgt_{tag}", {}).get(f)
                dy = r.get(f"d_{tag}", {}).get(f)
                if dg is None or dy is None or not np.isfinite(dg) or not np.isfinite(dy):
                    continue
                X.append(dg); Y.append(dy)
            if len(X) < 20:
                print(f"    {f:<9} {tag:<5} n={len(X)} too few")
                continue
            X, Y = np.array(X), np.array(Y)
            if X.std() < 1e-9:
                print(f"    {f:<9} {tag:<5} dGT is constant — intervention did not move the truth")
                continue
            slope = float(np.polyfit(X, Y, 1)[0])
            r_p = float(np.corrcoef(X, Y)[0, 1])
            rng = np.random.default_rng(0)
            bs = []
            for _ in range(2000):
                i = rng.integers(0, len(X), len(X))
                if X[i].std() > 1e-9:
                    bs.append(np.corrcoef(X[i], Y[i])[0, 1])
            lo, hi = np.percentile(bs, [2.5, 97.5]) if bs else (float("nan"),) * 2
            tracks = "TRACKS GT" if lo > 0 else ("anti-tracks" if hi < 0 else "no relation")
            print(f"    {f:<9} {tag:<5} n={len(X):<4} slope={slope:+.3f}  r={r_p:+.3f} "
                  f"[{lo:+.3f},{hi:+.3f}]  |dGT| median={np.median(np.abs(X)):.2f}   {tracks}")

    print("\n=== SUMMARY ===")
    if verdicts:
        n_g = sum("GROUNDED" in v for v in verdicts.values())
        n_b = sum("READS INTERFERER" in v for v in verdicts.values())
        n_f = sum("flat" in v for v in verdicts.values())
        print(f"    clean-stem features: {n_g} grounded / {n_b} read-interferer / {n_f} flat "
              f"(of {len(verdicts)})")
        print("    NOTE: a null here is only interpretable once the MODEL-LEVEL POSITIVE CONTROL")
        print("    (P2) has shown this pipeline can detect a known effect.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
