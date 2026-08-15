"""Calibrate + VALIDATE the Mondrian abstention map on a real sigma capture.

This is the runner that turns `src/eval/mondrian_srlc.py` (pure, unit-tested, node-free)
into the paper's abstention number. It does the one thing the unit tests cannot: check
that the finite-sample guarantee actually HOLDS on data the calibrator never saw.

    calibrate on split A   ->   thresholds lambda_{f,g}
    apply   to split B     ->   realized selective risk must be <= alpha

Reporting held-out risk is the whole point. Calibrating and reporting coverage on the
same clips is the standard way to publish a vacuous guarantee, so the two halves are
disjoint and the split is deterministic (seeded, filename-sorted) for reproducibility.

Inputs
  --capture       sigma_capture_*.json : {filename: {aux_mean: [...], aux_log_var: [...]}}
                  written by sigma_capture.py (adapter-only forward pass, no LM).
  --features_csv  the SAME split's feature CSV that supplied training GT for the aux head.

Units (audit-critical, see mondrian_srlc docstring)
  sigma is consumed NORMALIZED: the ReliabilityHead learns log_var in FEATURE_SCALES
  units, so sigma_f = exp(0.5 * log_var_f) is already normalized. The error is raw
  |pred - gt| and mondrian_calibrate divides it by S_f internally. Do not pre-normalize
  the error here or it gets divided twice.

  k is FROZEN first-principles, never tuned: the miscoverage event is
  |pred - gt| > k * S_f, with S_f the robust dispersion in feature_set.FEATURE_SCALES.
  k = 1.0 means "off by more than one dispersion unit". Tuning k to make a cell pass
  would be exactly the circularity the band-free SFS decision retired.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from data.feature_set import (  # noqa: E402
    FEATURE_NAMES,
    FEATURE_SCALES,
    ILL_POSED_UNDER_OVERLAP_FEATURES,
    SUPERVISED_FEATURES,
)
from eval.mondrian_srlc import (  # noqa: E402
    DEFAULT_OVERLAP_EDGES,
    DEFAULT_OVERLAP_LABELS,
    mondrian_calibrate,
    mondrian_calibrate_split,
    overlap_bin,
)

CSV_COL = {name: col for name, col, _ in SUPERVISED_FEATURES}


def _fnum(v):
    if v is None or v == "" or v == "nan":
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def build_records(capture: dict, gt_rows: list[dict]) -> list[dict]:
    """Join capture <-> GT on filename into mondrian_calibrate's record format."""
    gt_by_name = {r["filename"]: r for r in gt_rows}
    records, skipped = [], 0
    for fname, cap in capture.items():
        row = gt_by_name.get(fname)
        if row is None:
            skipped += 1
            continue
        mean = cap.get("aux_mean")
        logv = cap.get("aux_log_var")
        if not mean or not logv or len(mean) != len(FEATURE_NAMES):
            skipped += 1
            continue
        ov = _fnum(row.get("overlap_ratio"))
        rec = {"filename": fname, "overlap_ratio": 0.0 if ov is None else ov}
        for i, f in enumerate(FEATURE_NAMES):
            gt = _fnum(row.get(CSV_COL.get(f, "")))
            if gt is None:
                continue
            rec[f"sigma_{f}"] = math.exp(0.5 * float(logv[i]))   # already normalized
            rec[f"err_{f}"] = abs(float(mean[i]) - gt)           # RAW; normalized downstream
        records.append(rec)
    if skipped:
        print(f"[warn] skipped {skipped} capture entries with no GT row / malformed payload")
    return records


def assert_head_is_trained(records: list[dict]) -> None:
    """Refuse to publish numbers off an untrained / default-off head.

    A never-trained reliability head emits a constant log_var, so sigma carries no
    ranking information and every 'guarantee' is an artifact of the fixed threshold.
    """
    for f in sorted(ILL_POSED_UNDER_OVERLAP_FEATURES):
        vals = [r[f"sigma_{f}"] for r in records if f"sigma_{f}" in r]
        if len(vals) < 2:
            continue
        if max(vals) - min(vals) < 1e-6:
            raise SystemExit(
                f"[FATAL] sigma_{f} is constant across {len(vals)} clips "
                "-> the reliability head is untrained (or was dropped at load). "
                "Calibrating this yields a meaningless guarantee. Refusing to run."
            )


def split_records(records: list[dict], calib_frac: float, seed: int):
    """Deterministic disjoint split. Sorted first so the result never depends on dict order."""
    ordered = sorted(records, key=lambda r: r["filename"])
    rng = random.Random(seed)
    rng.shuffle(ordered)
    n_cal = int(round(len(ordered) * calib_frac))
    return ordered[:n_cal], ordered[n_cal:]


def evaluate_heldout(cal: dict, records: list[dict], k) -> dict:
    """Apply calibrated thresholds to unseen clips; measure realized coverage + risk."""
    scale = dict(zip(FEATURE_NAMES, FEATURE_SCALES))
    edges, labels = DEFAULT_OVERLAP_EDGES, DEFAULT_OVERLAP_LABELS
    out: dict[tuple[str, str], dict] = {}
    for (f, g), cell in cal["cells"].items():
        kf = k[f] if isinstance(k, dict) else float(k)
        n_seen = n_emit = n_fail = 0
        for r in records:
            if overlap_bin(r["overlap_ratio"], edges, labels) != g:
                continue
            sig, err = r.get(f"sigma_{f}"), r.get(f"err_{f}")
            if sig is None or err is None:
                continue
            n_seen += 1
            if cell.get("feasible") and sig <= cell["threshold"]:
                n_emit += 1
                if err / scale[f] > kf:
                    n_fail += 1
        out[(f, g)] = {
            "n": n_seen,
            "coverage": (n_emit / n_seen) if n_seen else 0.0,
            "risk": (n_fail / n_emit) if n_emit else None,
            "n_emit": n_emit,
            "n_fail": n_fail,
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture", required=True)
    ap.add_argument("--features_csv", required=True)
    ap.add_argument("--capture_cal", default=None,
                    help="Capture for the CALIBRATION split (e.g. dev). When given, thresholds "
                         "are picked here and --capture is used ENTIRELY for certification. "
                         "This is the correct design: calibration must not consume test data, "
                         "and splitting test in half doubles the Clopper-Pearson width.")
    ap.add_argument("--features_csv_cal", default=None,
                    help="GT CSV matching --capture_cal (required with it).")
    ap.add_argument("--out", default=None, help="write the calibrated map + validation as JSON")
    ap.add_argument("--alpha", type=float, default=0.1, help="target selective risk")
    ap.add_argument("--delta", type=float, default=0.05, help="FWER over all cells")
    ap.add_argument("--k", type=float, default=1.0, help="FROZEN miscoverage multiple of S_f")
    ap.add_argument("--calib_frac", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--method", choices=("split", "grid"), default="split",
                    help="split: pick threshold on half, certify on the disjoint half (one "
                         "binomial test, delta over cells only) -- tighter, and the default. "
                         "grid: single-sample scan, delta Bonferroni'd over every candidate "
                         "threshold as well -- valid but markedly more conservative.")
    args = ap.parse_args()

    capture = json.load(open(args.capture))
    gt_rows = list(csv.DictReader(open(args.features_csv)))
    records = build_records(capture, gt_rows)
    print(f"joined {len(records)} clips ({len(capture)} capture / {len(gt_rows)} GT rows)")
    assert_head_is_trained(records)

    if args.capture_cal:
        if not args.features_csv_cal:
            raise SystemExit("--capture_cal requires --features_csv_cal")
        cal_rows = list(csv.DictReader(open(args.features_csv_cal)))
        calib = build_records(json.load(open(args.capture_cal)), cal_rows)
        heldout = records
        assert_head_is_trained(calib)
        # Disjointness is by construction (different splits); assert it rather than trust it.
        overlap_names = {r["filename"] for r in calib} & {r["filename"] for r in heldout}
        if overlap_names:
            raise SystemExit(f"calibration and certification sets share {len(overlap_names)} "
                             "filenames — the guarantee would be invalid")
        print(f"calibration set {len(calib)} clips (separate split) | certification set "
              f"{len(heldout)} clips — disjoint by construction")
    else:
        calib, heldout = split_records(records, args.calib_frac, args.seed)
        print(f"calibration split {len(calib)} clips | held-out split {len(heldout)} clips "
              f"(seed {args.seed}, disjoint)")

    if args.method == "split":
        # Threshold chosen on `calib`, certified on the disjoint `heldout`: one binomial
        # test per cell, so delta is Bonferroni'd over cells only. The grid method has to
        # correct over every candidate threshold too, which at n~3000 costs enough UCB
        # width to reject cells whose true selective risk is comfortably under alpha.
        cal = mondrian_calibrate_split(
            calib, heldout,
            feature_names=FEATURE_NAMES,
            scales=FEATURE_SCALES,
            ill_posed_features=ILL_POSED_UNDER_OVERLAP_FEATURES,
            alpha=args.alpha, delta=args.delta, k=args.k,
        )
    else:
        cal = mondrian_calibrate(
            calib,
            feature_names=FEATURE_NAMES,
            scales=FEATURE_SCALES,
            ill_posed_features=ILL_POSED_UNDER_OVERLAP_FEATURES,
            alpha=args.alpha, delta=args.delta, k=args.k,
        )
    val = evaluate_heldout(cal, heldout, args.k)

    p = cal["params"]
    print(f"\nalpha={args.alpha}  delta={args.delta} (delta_cell={p['delta_cell']:.2e} over "
          f"{p['n_cells']} cells)  k={args.k} (frozen)")
    print(f"always-emit (recoverable, never hedged): {', '.join(cal['always_emit'])}\n")
    split = args.method == "split"
    tail = "   (cov/risk below are ON the certification half)" if split else \
           "   |  held-out validation"
    hdr = (f"{'feature':<10}{'bin':<9}{'n':>7}{'thresh':>9}{'cov':>8}{'risk':>8}{'UCB':>8}"
           f"  {'verdict':<9}")
    print(hdr + tail)
    print("-" * (len(hdr) + 8))
    n_pass = n_tested = 0
    for (f, g) in sorted(cal["cells"]):
        c, v = cal["cells"][(f, g)], val[(f, g)]
        n_show = c.get("n", c.get("n_cert", 0))
        if not c["feasible"]:
            reason = "abstain" if c.get("ucb") is None else "UCB>alpha"
            print(f"{f:<10}{g:<9}{n_show:>7}{'INFEAS':>9}{c.get('coverage', 0.0):>8.3f}"
                  f"{(c['emp_risk'] or 0):>8.3f}{(c['ucb'] or float('nan')):>8.3f}  {reason:<9}")
            continue
        thr = "inf" if math.isinf(c["threshold"]) else f"{c['threshold']:.3f}"
        n_tested += 1
        # In split mode the UCB IS the guarantee (threshold is independent of the cert half).
        # In grid mode we additionally check realized risk on the untouched held-out half.
        ok = (c["ucb"] <= args.alpha) if split else (v["risk"] is not None and v["risk"] <= args.alpha)
        n_pass += int(ok)
        shown_risk = c["emp_risk"] if split else (v["risk"] if v["risk"] is not None else float("nan"))
        shown_cov = c["coverage"] if split else v["coverage"]
        print(f"{f:<10}{g:<9}{n_show:>7}{thr:>9}{shown_cov:>8.3f}{shown_risk:>8.3f}"
              f"{c['ucb']:>8.3f}  {('CERTIFIED' if ok else 'FAILED'):<9}")

    if split:
        print(f"\ncertified: {n_pass}/{len(cal['cells'])} cells emit with a finite-sample "
              f"guarantee (selective risk <= {args.alpha} w.p. >= {1-args.delta:.2f}).")
        print("  Threshold picked on the disjoint half, so the UCB is a valid guarantee, not a fit.")
    else:
        print(f"\nheld-out guarantee: {n_pass}/{n_tested} emitting cells satisfy risk <= alpha")
        if n_tested and n_pass < n_tested:
            print("  NOTE: a violated cell is a real finding, not a bug — report it. With "
                  f"delta={args.delta} we expect a violation at most ~{args.delta:.0%} of the time.")

    if args.out:
        payload = {
            "params": {**p, "k": args.k, "seed": args.seed, "calib_frac": args.calib_frac,
                       "capture": os.path.abspath(args.capture),
                       "features_csv": os.path.abspath(args.features_csv)},
            "always_emit": cal["always_emit"],
            "cells": [{"feature": f, "bin": g, **cal["cells"][(f, g)],
                       "heldout": val[(f, g)]} for (f, g) in sorted(cal["cells"])],
        }
        with open(args.out, "w") as fh:
            json.dump(payload, fh, indent=2)
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
