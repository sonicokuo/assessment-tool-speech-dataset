"""hedging_calibration.py — is overlap-aware hedging CALIBRATED to actual error?

The overlap-aware hedging contribution claims the model's numbers stay faithful
precisely where the signal estimate is reliable, and that it hedges where it is
not. That is only a real result if hedging tracks ACTUAL error. This module
turns it into a measurement on inference_results.json:

  - Reliability-by-bin: bin clips by ground-truth overlap_ratio; show that F0
    correctness FALLS as overlap rises (so overlap is a valid reliability signal).
  - Risk-coverage: order clips by overlap_ratio (the implicit hedge signal) and,
    as we ABSTAIN on the most-overlapped clips, measure precision on the F0
    claims we still assert. If precision rises as coverage drops, the hedge
    signal is calibrated and selective generation buys real faithfulness.

Pure functions (no torch); the CLI at scripts/hedging_calibration.py feeds them
inference_results.json (per-clip per_feature carries claimed/actual/error/correct
and the overlap_ratio feature's `actual` is the GT overlap).
"""
from __future__ import annotations


def extract_clip_signals(entry: dict, feature: str = "f0_mean") -> dict | None:
    """Pull (overlap_ratio GT, target-feature correctness/error) from one entry.

    Returns None if the clip lacks either the overlap_ratio or the target feature
    in its per_feature list (so it cannot enter the calibration).
    """
    pf = {f.get("feature"): f for f in entry.get("per_feature", []) if isinstance(f, dict)}
    ov = pf.get("overlap_ratio")
    tg = pf.get(feature)
    if ov is None or tg is None:
        return None
    if ov.get("actual") is None or tg.get("correct") is None:
        return None
    return {
        "filename": entry.get("filename"),
        "overlap_ratio": float(ov["actual"]),
        "error": tg.get("error"),
        "correct": bool(tg["correct"]),
    }


def reliability_by_bin(signals: list[dict], bin_edges: list[float]) -> list[dict]:
    """Mean correctness (and |error|) per overlap_ratio bin.

    bin_edges e.g. [0, 0.25, 0.5, 0.75, 1.01]. Returns one row per bin with n,
    overlap range, accuracy (mean correct), and mean |error| (over clips with a
    numeric error). A calibrated reliability signal yields monotonically falling
    accuracy as overlap rises.
    """
    rows = []
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        members = [s for s in signals if lo <= s["overlap_ratio"] < hi]
        n = len(members)
        acc = (sum(1 for s in members if s["correct"]) / n) if n else 0.0
        errs = [abs(s["error"]) for s in members if isinstance(s.get("error"), (int, float))]
        rows.append({
            "lo": lo, "hi": hi, "n": n,
            "accuracy": round(acc, 4),
            "mean_abs_error": round(sum(errs) / len(errs), 4) if errs else None,
        })
    return rows


def risk_coverage_curve(signals: list[dict], steps: int = 10) -> list[dict]:
    """Selective-prediction curve using overlap_ratio as the (ascending) risk key.

    Sort clips by overlap_ratio ascending (least overlap = most reliable). At each
    coverage c, ASSERT the lowest-overlap fraction c and ABSTAIN on the rest;
    precision is mean correctness over the asserted set. A calibrated hedge signal
    gives precision that is highest at low coverage and falls toward full coverage.

    Returns rows of {coverage, n_asserted, precision}.
    """
    if not signals:
        return []
    ordered = sorted(signals, key=lambda s: s["overlap_ratio"])
    n = len(ordered)
    rows = []
    for k in range(1, steps + 1):
        cov = k / steps
        m = max(1, round(cov * n))
        asserted = ordered[:m]
        prec = sum(1 for s in asserted if s["correct"]) / len(asserted)
        rows.append({"coverage": round(cov, 3), "n_asserted": len(asserted),
                     "precision": round(prec, 4)})
    return rows


def selective_improvement(curve: list[dict]) -> float:
    """precision at the lowest coverage minus precision at full coverage.

    Positive => abstaining on high-overlap clips raises precision on the asserted
    F0 claims => the hedge signal is calibrated and selective generation helps.
    """
    if not curve:
        return 0.0
    return round(curve[0]["precision"] - curve[-1]["precision"], 4)


def is_monotone_nonincreasing(rows: list[dict], key: str = "accuracy",
                              tol: float = 1e-9) -> bool:
    """True if `key` is (weakly) non-increasing across populated bins — the
    signature of a calibrated reliability signal (accuracy falls with overlap)."""
    vals = [r[key] for r in rows if r["n"] > 0 and r.get(key) is not None]
    return all(a + tol >= b for a, b in zip(vals[:-1], vals[1:]))
