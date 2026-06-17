#!/usr/bin/env python3
"""Re-score a run's F0 against well-posed (clean-frame) F0, and report the
overlap-reliability story for the hedging contribution.

The per_feature f0_mean in inference_results.json was scored against the MIXTURE
F0 (ill-posed under overlap). This rewrites each clip's f0_mean entry against the
clean-frame F0 (from scripts/compute_clean_f0.py): updates actual/error/correct,
or marks the clip f0-UNMEASURABLE when the clean F0 is undefined (too few
non-overlapped voiced frames — exactly the heavily-overlapped clips where the
model SHOULD hedge).

Outputs a rescored inference_results.json (feed it to scripts/hedging_calibration.py)
and prints a reliability-by-overlap-bin table: as overlap rises, the fraction of
clips whose F0 is unmeasurable rises AND the F0 error on the measurable clips
rises — i.e. F0 reliability falls with overlap, so the overlap-driven hedge is
calibrated.

Usage:
  python scripts/rescore_clean_f0.py \
    --inference_results <run>/inference_results.json \
    --clean_f0 $SHARED/data/clean_f0_test.json \
    --tolerance 5 --output <run>/inference_results_cleanf0.json
"""
import argparse
import json
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))


def _is_nan(x):
    return x is None or (isinstance(x, float) and math.isnan(x))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inference_results", required=True)
    ap.add_argument("--clean_f0", required=True)
    ap.add_argument("--tolerance", type=float, default=5.0)
    ap.add_argument("--output", default=None)
    ap.add_argument("--bins", default="0,0.25,0.5,0.75,1.01")
    args = ap.parse_args()

    data = json.loads(open(args.inference_results).read())
    clean = json.loads(open(args.clean_f0).read())

    edges = [float(x) for x in args.bins.split(",")]
    # per-bin accumulators
    bin_stats = [{"lo": lo, "hi": hi, "n": 0, "unmeasurable": 0,
                  "err_sum": 0.0, "err_n": 0, "correct": 0}
                 for lo, hi in zip(edges[:-1], edges[1:])]

    n_updated = n_unmeasurable = 0
    for e in data:
        pf = {f.get("feature"): f for f in e.get("per_feature", []) if isinstance(f, dict)}
        f0 = pf.get("f0_mean")
        ov = pf.get("overlap_ratio")
        cf = clean.get(e.get("filename"))
        if f0 is None or ov is None or cf is None:
            continue
        ov_val = ov.get("actual")
        if ov_val is None:
            continue
        clean_mean = cf.get("f0_mean_hz")
        # locate the bin
        b = next((bs for bs in bin_stats if bs["lo"] <= ov_val < bs["hi"]), None)
        if b is None:
            continue
        b["n"] += 1

        if _is_nan(clean_mean):
            # F0 is genuinely undefined here (too overlapped) -> unmeasurable.
            f0["actual"] = None
            f0["error"] = None
            f0["correct"] = None
            f0["unmeasurable"] = True
            n_unmeasurable += 1
            b["unmeasurable"] += 1
        else:
            claimed = f0.get("claimed")
            if claimed is None:
                continue
            err = abs(float(claimed) - float(clean_mean))
            f0["actual"] = round(float(clean_mean), 2)
            f0["error"] = err
            f0["tolerance"] = args.tolerance
            f0["correct"] = bool(err <= args.tolerance)
            f0.pop("unmeasurable", None)
            n_updated += 1
            b["err_sum"] += err
            b["err_n"] += 1
            if f0["correct"]:
                b["correct"] += 1

    out = args.output or args.inference_results.replace(".json", "_cleanf0.json")
    json.dump(data, open(out, "w"))

    print(f"Re-scored F0 against clean-frame GT (tolerance ±{args.tolerance} Hz)")
    print(f"  measurable updated: {n_updated}   unmeasurable (clean F0 undefined): {n_unmeasurable}")
    print(f"  wrote {out}\n")
    print("Reliability by overlap bin (F0 reliability should FALL as overlap rises):")
    print(f"  {'overlap':14} {'n':>5} {'%unmeasurable':>14} {'F0 acc(meas)':>13} {'mean|err|':>10}")
    for b in bin_stats:
        n = b["n"]
        unm = (100 * b["unmeasurable"] / n) if n else 0.0
        acc = (b["correct"] / b["err_n"]) if b["err_n"] else float("nan")
        mae = (b["err_sum"] / b["err_n"]) if b["err_n"] else float("nan")
        rng = f"[{b['lo']:.2f},{b['hi']:.2f})"
        print(f"  {rng:14} {n:5d} {unm:13.1f}% {acc:13.3f} {mae:10.2f}")
    print("\nFeed the rescored file to scripts/hedging_calibration.py for the "
          "risk-coverage curve on the measurable clips.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
