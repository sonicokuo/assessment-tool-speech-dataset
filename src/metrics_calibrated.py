"""Calibrated / field-standard metrics for AQUA-NL speech-feature descriptions.

Two additions on top of the absolute-tolerance SFS in `src/sfs.py`:

1. CORRELATION + ERROR metrics (SQUIM / UTMOS / VoiceMOS convention).
   For each scorable feature we collect, across a result set, the pairs
   (model's parsed numeric claim, ground-truth value) and report:
       SRCC  Spearman rank correlation
       PCC   Pearson correlation
       MAE   mean absolute error
       n     number of paired (asserted-and-GT) clips
   These are the standard numbers a reviewer expects for "how well does the
   model's predicted X track the true X", and they do NOT depend on a tolerance
   threshold (unlike binary SFS). They complement SFS rather than replace it.

2. RELATIVE / tolerance-CALIBRATED SFS.
   The legacy SFS uses a single absolute tolerance per feature (`±2 dB` SNR,
   `±5 Hz` F0). That under-credits high-magnitude clips (a 2 dB miss on a 35 dB
   SNR is proportionally tiny) and over-credits low-magnitude ones. The
   calibrated path uses a per-feature tolerance

       tol(f, gt) = max(abs_tol[f], rel_frac[f] * |gt|)

   so the band widens proportionally with the GT magnitude while never falling
   below a sensible floor. `abs_tol` defaults to `SFSScorer.TOLERANCES`; the old
   absolute path stays available (rel_frac=0 reproduces it exactly).

Both helpers consume the SAME `inference_results.json` schema the rest of the
pipeline writes: a list of {"filename", "generated", "target", ...}. Ground
truth comes from parsing the `target` string with the same HybridClaimParser
the scorer uses, so GT is naturally restricted to parseable, scorable features
(no denominator inflation — see the recall note in src/sfs.py).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

try:  # package-relative when imported as src.metrics_calibrated
    from .sfs import Claim, HybridClaimParser, AbstentionDetector, SFSScorer
except ImportError:  # flat import when src/ is on sys.path (matches sfs.py style)
    from sfs import Claim, HybridClaimParser, AbstentionDetector, SFSScorer


# ── Per-feature relative-tolerance fractions ────────────────────────────────
# tol = max(abs_tol, rel_frac * |gt|). rel_frac chosen per feature from the
# scale of typical SP measurement disagreement. abs_tol comes from
# SFSScorer.TOLERANCES (the floor). A feature absent here falls back to
# DEFAULT_REL_FRAC.
DEFAULT_REL_FRAC = 0.10
REL_FRAC = {
    "snr": 0.10,            # ±10% of the SNR value
    "hnr": 0.10,
    "f0_mean": 0.05,        # ±5% of pitch
    "f0_sd": 0.10,
    "f0_std": 0.10,
    "srmr": 0.10,
    "speaking_rate": 0.10,
    "articulation_rate": 0.10,
    "pause_rate": 0.15,
    "pause_count": 0.0,     # discrete count: keep the ±1 absolute floor only
    "overlap_ratio": 0.10,
    "jitter": 0.15,
    "shimmer": 0.15,
    "spectral_tilt": 0.15,
    "vot": 0.15,
    "rt60": 0.15,
    "f1_mean": 0.05,
    "f2_mean": 0.05,
    "f3_mean": 0.05,
    "f4_mean": 0.05,
}


def relative_tolerance(feature: str, gt_value: float,
                       abs_tol: dict[str, float] | None = None,
                       rel_frac: dict[str, float] | None = None) -> float:
    """tol(f, gt) = max(abs_tol[f], rel_frac[f] * |gt|).

    With rel_frac[f] == 0 this collapses to the absolute tolerance (back-compat).
    """
    abs_tol = abs_tol if abs_tol is not None else SFSScorer.TOLERANCES
    rel_frac = rel_frac if rel_frac is not None else REL_FRAC
    floor = abs_tol.get(feature, 0.0)
    frac = rel_frac.get(feature, DEFAULT_REL_FRAC)
    return max(floor, frac * abs(gt_value))


# ── Lightweight correlation primitives (no scipy dependency) ────────────────
def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs)


def pearson(xs: list[float], ys: list[float]) -> float | None:
    """Pearson product-moment correlation. None if undefined (n<2 or zero
    variance on either axis)."""
    n = len(xs)
    if n < 2 or n != len(ys):
        return None
    mx, my = _mean(xs), _mean(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    if sxx <= 0 or syy <= 0:
        return None
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    return sxy / math.sqrt(sxx * syy)


def _rankdata(xs: list[float]) -> list[float]:
    """Average-rank tie handling (Spearman convention)."""
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0  # 1-based average rank
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def spearman(xs: list[float], ys: list[float]) -> float | None:
    """Spearman rank correlation (Pearson on average ranks). None if undefined."""
    n = len(xs)
    if n < 2 or n != len(ys):
        return None
    return pearson(_rankdata(xs), _rankdata(ys))


def mae(xs: list[float], ys: list[float]) -> float | None:
    if not xs or len(xs) != len(ys):
        return None
    return _mean([abs(a - b) for a, b in zip(xs, ys)])


# ── Result-set aggregation ──────────────────────────────────────────────────
@dataclass
class FeatureCorr:
    feature: str
    n: int
    srcc: float | None
    pcc: float | None
    mae: float | None
    pred: list[float]
    gt: list[float]


def _scalar_claims(text: str, parser: HybridClaimParser) -> dict[str, float]:
    """First numeric claim per scorable scalar feature (skip overlap spans)."""
    out: dict[str, float] = {}
    for c in parser.parse(text):
        if c.feature in ("overlap_start", "overlap_end"):
            continue
        if c.feature in SFSScorer.TOLERANCES and c.feature not in out:
            out[c.feature] = c.value
    return out


def collect_pairs(
    results: list[dict],
    parser: HybridClaimParser | None = None,
) -> dict[str, tuple[list[float], list[float]]]:
    """Walk an inference_results list and collect, per feature, the paired
    (predicted, ground-truth) numeric values.

    A pair is added for a (clip, feature) only when BOTH the generated text and
    the target text assert a number for that feature (so correlation is computed
    over the intersection — clips where the model abstained or omitted do not
    contribute a pair, which is the SQUIM/UTMOS convention of correlating
    co-present predictions).

    Each result entry needs `generated` and `target` strings.
    """
    parser = parser or HybridClaimParser()
    pred_by: dict[str, list[float]] = {}
    gt_by: dict[str, list[float]] = {}
    for entry in results:
        gen = entry.get("generated") or entry.get("generated_clean") or ""
        tgt = entry.get("target") or ""
        if not tgt:
            continue
        pred = _scalar_claims(gen, parser)
        gt = _scalar_claims(tgt, parser)
        for f, gv in gt.items():
            if f in pred:
                pred_by.setdefault(f, []).append(pred[f])
                gt_by.setdefault(f, []).append(gv)
    return {f: (pred_by[f], gt_by[f]) for f in gt_by}


def correlation_report(
    results: list[dict],
    parser: HybridClaimParser | None = None,
) -> dict:
    """Per-feature {SRCC, PCC, MAE, n} + an aggregate (n-weighted MAE, and the
    mean of the defined per-feature correlations).

    Returns:
        {
          "per_feature": {feat: {"srcc":.., "pcc":.., "mae":.., "n":..}, ...},
          "aggregate":   {"mean_srcc":.., "mean_pcc":.., "weighted_mae":..,
                          "n_features":.., "n_pairs":..},
        }
    """
    pairs = collect_pairs(results, parser)
    per_feature: dict[str, dict] = {}
    for f, (pred, gt) in sorted(pairs.items()):
        per_feature[f] = {
            "srcc": spearman(pred, gt),
            "pcc": pearson(pred, gt),
            "mae": mae(pred, gt),
            "n": len(pred),
        }

    srccs = [v["srcc"] for v in per_feature.values() if v["srcc"] is not None]
    pccs = [v["pcc"] for v in per_feature.values() if v["pcc"] is not None]
    total_n = sum(v["n"] for v in per_feature.values())
    w_mae = (
        sum(v["mae"] * v["n"] for v in per_feature.values() if v["mae"] is not None)
        / total_n
        if total_n else None
    )
    aggregate = {
        "mean_srcc": _mean(srccs) if srccs else None,
        "mean_pcc": _mean(pccs) if pccs else None,
        "weighted_mae": w_mae,
        "n_features": len(per_feature),
        "n_pairs": total_n,
    }
    return {"per_feature": per_feature, "aggregate": aggregate}


# ── Calibrated (relative-tolerance) SFS over a result set ───────────────────
def calibrated_sfs_report(
    results: list[dict],
    parser: HybridClaimParser | None = None,
    rel_frac: dict[str, float] | None = None,
    abs_tol: dict[str, float] | None = None,
) -> dict:
    """Per-feature and aggregate SFS-precision using the relative/calibrated
    tolerance band tol = max(abs_tol, rel_frac*|gt|).

    GT comes from parsing each entry's `target`. For each (clip, feature) where
    the target asserts a number and the generation asserts a number, the claim
    is "correct" iff |pred - gt| <= tol(feature, gt). rel_frac=0 for every
    feature reproduces the legacy absolute path exactly.

    Returns:
        {
          "per_feature": {feat: {"precision":.., "n_asserted":.., "n_correct":..}},
          "aggregate":   {"precision":.., "n_asserted":.., "n_correct":..},
        }
    """
    parser = parser or HybridClaimParser()
    abs_tol = abs_tol if abs_tol is not None else SFSScorer.TOLERANCES
    rel_frac = rel_frac if rel_frac is not None else REL_FRAC

    n_asserted_by: dict[str, int] = {}
    n_correct_by: dict[str, int] = {}
    for entry in results:
        gen = entry.get("generated") or entry.get("generated_clean") or ""
        tgt = entry.get("target") or ""
        if not tgt:
            continue
        pred = _scalar_claims(gen, parser)
        gt = _scalar_claims(tgt, parser)
        for f, pv in pred.items():
            if f not in gt:
                continue  # no GT to score against
            n_asserted_by[f] = n_asserted_by.get(f, 0) + 1
            tol = relative_tolerance(f, gt[f], abs_tol, rel_frac)
            if abs(pv - gt[f]) <= tol:
                n_correct_by[f] = n_correct_by.get(f, 0) + 1

    per_feature: dict[str, dict] = {}
    for f in sorted(n_asserted_by):
        na = n_asserted_by[f]
        nc = n_correct_by.get(f, 0)
        per_feature[f] = {
            "precision": nc / na if na else 0.0,
            "n_asserted": na,
            "n_correct": nc,
        }
    tot_a = sum(n_asserted_by.values())
    tot_c = sum(n_correct_by.values())
    aggregate = {
        "precision": tot_c / tot_a if tot_a else 0.0,
        "n_asserted": tot_a,
        "n_correct": tot_c,
    }
    return {"per_feature": per_feature, "aggregate": aggregate}
