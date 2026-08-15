#!/usr/bin/env python3
"""target_coverage_audit.py — which features do the TRAINING TARGETS actually state?

WHY. §1.49 recorded that the model silently drops f0_mean, f0_sd, shimmer and hnr on clean
clips while sparing jitter, and attributed it to COVERAGE COLLAPSE (a model failure) on the
strength of one measurement: f0 is present in 98.8% of clean `fw2` targets. **hnr, shimmer and
jitter were never checked** — I verified one feature and generalised to four.

That matters because of a striking coincidence a red-team audit surfaced: the dropped set is
EXACTLY the union of the two historical target-bug victim sets, and jitter is the sole escapee
of both. `build_canonical_descriptions.py:139-143` says so verbatim:

    BUGFIX 2026-07-28: these read "hnr_db" / "shimmer_pct", which do NOT exist in
    features_corrected_merged/*.csv (the columns are "hnr" / "shimmer") ... jitter escaped
    only because "jitter_local_pct" happens to be spelled correctly.

So there are two incompatible explanations and this script separates them:
  * targets STATE the feature and the model omits it  -> COVERAGE COLLAPSE (model failure,
    a reportable finding about conditional-abstention training)
  * targets DO NOT state it                            -> the model never had supervision, and
    the "collapse" for that feature is a DATA defect wearing a model's clothes

The arm trained from scratch (`resume_from: null`), so weight lineage is excluded; only the
target content can carry the historical bug forward.
"""
from __future__ import annotations

import argparse
import json
import re

# Match the canonical clause openers emitted by build_canonical_descriptions.py.
PATTERNS = [
    ("snr",           r"SNR is"),
    ("srmr",          r"SRMR is"),
    ("hnr",           r"HNR is"),
    ("f0_mean",       r"F0 mean is"),
    ("f0_sd",         r"F0 standard deviation"),
    ("jitter",        r"jitter is"),
    ("shimmer",       r"shimmer is"),
    ("speaking_rate", r"speaking rate is"),
    ("pause_count",   r"pause count is"),
    ("pause_rate",    r"pause rate is"),
    ("overlap_ratio", r"overlap ratio is"),
]
HEDGE = re.compile(r"cannot be reliably estimated|not reported", re.I)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", required=True)
    a = ap.parse_args()

    d = json.load(open(a.targets))
    pats = [(n, re.compile(p, re.I)) for n, p in PATTERNS]
    clean = [v for k, v in d.items() if k.endswith("_s1clean")]
    mixed = [v for k, v in d.items() if not k.endswith("_s1clean")]

    print(f"targets={len(d)}  clean={len(clean)}  mixtures={len(mixed)}\n")
    print(f"{'feature':<16}{'CLEAN tgt%':>12}{'MIX tgt%':>11}   verdict")
    print("-" * 62)
    for n, p in pats:
        c = 100.0 * sum(1 for v in clean if p.search(v)) / max(len(clean), 1)
        m = 100.0 * sum(1 for v in mixed if p.search(v)) / max(len(mixed), 1)
        verdict = ""
        if c < 5.0:
            verdict = "<<< ABSENT FROM TARGETS — not a model failure"
        elif c > 90.0 and m < 5.0:
            verdict = "conditional (intended)"
        elif c > 90.0 and m > 90.0:
            verdict = "always stated"
        print(f"{n:<16}{c:11.1f}%{m:10.1f}%   {verdict}")

    hc = 100.0 * sum(1 for v in clean if HEDGE.search(v)) / max(len(clean), 1)
    hm = 100.0 * sum(1 for v in mixed if HEDGE.search(v)) / max(len(mixed), 1)
    print(f"\n{'hedge phrase':<16}{hc:11.1f}%{hm:10.1f}%")
    print("\nREAD: a feature ABSENT from clean targets was never supervised there, so the "
          "model omitting it is correct behaviour, NOT coverage collapse. Only features the "
          "targets DO state can support the §1.49 finding.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
