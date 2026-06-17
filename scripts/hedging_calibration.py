#!/usr/bin/env python3
"""Measure whether overlap-aware hedging is calibrated to actual F0 error.

Consumes an inference_results.json (per-clip per_feature carries claimed/actual/
error/correct; the overlap_ratio feature's `actual` is the GT overlap) and
reports, for a target feature (default f0_mean):
  - reliability-by-overlap-bin (accuracy should fall as overlap rises),
  - a risk-coverage curve (precision on asserted claims as we abstain on the
    most-overlapped clips),
  - selective improvement (precision at low coverage minus at full coverage).

Usage:
  python scripts/hedging_calibration.py \
    --inference_results $SHARED/checkpoints/<run>/inference_results.json \
    --feature f0_mean
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from hedging_calibration import (  # noqa: E402
    extract_clip_signals,
    reliability_by_bin,
    risk_coverage_curve,
    selective_improvement,
    is_monotone_nonincreasing,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inference_results", required=True)
    ap.add_argument("--feature", default="f0_mean")
    ap.add_argument("--bins", default="0,0.25,0.5,0.75,1.01")
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--output_md", default=None)
    args = ap.parse_args()

    data = json.loads(open(args.inference_results).read())
    signals = [s for s in (extract_clip_signals(e, args.feature) for e in data) if s]

    edges = [float(x) for x in args.bins.split(",")]
    rel = reliability_by_bin(signals, edges)
    curve = risk_coverage_curve(signals, steps=args.steps)
    sel = selective_improvement(curve)
    mono = is_monotone_nonincreasing(rel, "accuracy")

    lines = [
        f"# Hedging calibration — feature `{args.feature}`",
        "",
        f"N clips with both overlap_ratio and {args.feature}: {len(signals)}",
        "",
        "## Reliability by overlap bin (accuracy should fall as overlap rises)",
        "| overlap range | n | accuracy | mean |err| |",
        "|---|---:|---:|---:|",
    ]
    for r in rel:
        lines.append(f"| [{r['lo']:.2f}, {r['hi']:.2f}) | {r['n']} | {r['accuracy']:.3f} | "
                     f"{'' if r['mean_abs_error'] is None else f'{r['mean_abs_error']:.2f}'} |")
    lines += [
        "",
        f"accuracy monotonically non-increasing with overlap: **{mono}**",
        "",
        "## Risk-coverage (assert lowest-overlap fraction, abstain on the rest)",
        "| coverage | n asserted | precision |",
        "|---:|---:|---:|",
    ]
    for c in curve:
        lines.append(f"| {c['coverage']:.2f} | {c['n_asserted']} | {c['precision']:.3f} |")
    lines += [
        "",
        f"**selective improvement** (precision@min-coverage − precision@full): **{sel:+.3f}**",
        "",
        "A positive selective improvement + accuracy falling with overlap means the "
        "model's reliability under overlap is calibrated: abstaining (hedging) on "
        "the most-overlapped clips raises precision on the claims it still asserts.",
    ]
    out = "\n".join(lines)
    print(out)
    if args.output_md:
        open(args.output_md, "w").write(out)
        print(f"\nWrote {args.output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
