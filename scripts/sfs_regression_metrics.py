#!/usr/bin/env python3
"""sfs_regression_metrics.py — band-free SFS scoring: MAE / NMAE / CCC / SRCC.

WHY THIS EXISTS
---------------
SFS-precision is `mean(|pred - gt| <= tolerance)` — a HARD THRESHOLD on a
continuous error. The tolerance band is a free parameter: tighten it and
precision drops, loosen it and precision rises, monotonically. So precision
conflates "how accurate is the model" with "how wide did I set the band", and
two runs with different bands are not comparable. (Worse: the bands in
SFSScorer.TOLERANCES were partly chosen to absorb the GROUND-TRUTH estimator's
own noise — see the rt60/pause_count comments there — so precision partly
measures "did you beat Praat's noise floor", not "are you right".)

This script keeps the EXACT SAME parser + the EXACT SAME (prediction, GT) pairs
that SFS already produced, but scores them with continuous, band-free statistics:

    MAE   mean absolute error           — accuracy in the feature's native unit
    NMAE  MAE / std(GT)                  — unit-free, so it's comparable across
                                           features (you cannot average raw MAE of
                                           150-Hz F0 with 0.05 overlap_ratio)
    CCC   Lin's concordance corr coef    — agreement with the y=x line; penalises
                                           BOTH poor correlation AND scale/offset
                                           bias. The single best "faithfulness"
                                           number per feature.
    PCC   Pearson corr                   — reported only to expose bias: if PCC is
                                           high but CCC is low, the model TRACKS
                                           the signal but is systematically biased.
    SRCC  Spearman rank corr            — does the model rank clips correctly;
                                           robust to outliers + monotone nonlinearity.
    coverage  emission rate             — fraction of applicable clips where the
                                           model actually emitted a parseable value.
                                           REQUIRED alongside the above: MAE/CCC are
                                           only computable on emitted claims, so a
                                           model that stays silent on hard clips
                                           would otherwise look artificially good.

NO RE-INFERENCE, NO RE-PARSE
----------------------------
src/inference.py already writes `per_feature` into inference_results.json — a list
of {feature, claimed, actual, error, tolerance, correct} per scored claim. That is
exactly the matched (prediction, GT) pairs. We just aggregate them differently.
SFS-F1 is still printed as a legacy column so you can see the two side by side.

Usage
-----
    python scripts/sfs_regression_metrics.py \
        --results     $SHARED/checkpoints/v13_section_warmup/inference_results.json \
        --features_csv $SHARED/data/features_pyannote/test.csv \
        --output_md   $SHARED/checkpoints/v13_section_warmup/regression_metrics.md

`--features_csv` is OPTIONAL and only used to compute exact coverage (the count of
clips for which each feature is genuinely present in GT). Without it, coverage is
reported against the total clip count, which under-states it for rare features
like overlap_ratio.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

# Stdlib-only (numpy not assumed on every login node). All stats are tiny 1-D.

# Some feature names appear under two spellings depending on the verbalizer wording
# ("SD" vs "std"); collapse them so the table doesn't split one feature into two.
FEATURE_ALIASES = {
    "f0_std": "f0_mean_sd",   # keep distinct from f0_mean below; see _canon
    "f0_sd": "f0_mean_sd",
}

# Map a scorer feature name → the CSV column that holds its GT (for coverage).
# Mirrors feature_set.SUPERVISED_FEATURES; extend if you score more features.
CSV_COL = {
    "snr": "snr_db",
    "srmr": "srmr",
    "f0_mean": "f0_mean_hz",
    "f0_mean_sd": "f0_sd_hz",
    "speaking_rate": "praat_speaking_rate_syl_sec",
    "pause_count": "praat_pause_count",
    "pause_rate": "praat_pause_rate_per_min",
    "overlap_ratio": "overlap_ratio",
}


def _canon(feature: str) -> str:
    """Canonicalise a feature name (collapse SD/std spellings)."""
    return FEATURE_ALIASES.get(feature, feature)


# ── tiny pure-Python statistics ──────────────────────────────────────────────
def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs)


def _var(xs: list[float]) -> float:
    m = _mean(xs)
    return sum((x - m) ** 2 for x in xs) / len(xs)


def _std(xs: list[float]) -> float:
    return math.sqrt(_var(xs))


def _cov(xs: list[float], ys: list[float]) -> float:
    mx, my = _mean(xs), _mean(ys)
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / len(xs)


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    sx, sy = _std(xs), _std(ys)
    if sx == 0 or sy == 0:
        return None   # undefined when one side is constant
    return _cov(xs, ys) / (sx * sy)


def _ccc(xs: list[float], ys: list[float]) -> float | None:
    """Lin's concordance correlation. xs=pred, ys=gt.

    CCC = 2*cov / (var_x + var_y + (mean_x - mean_y)^2).
    1.0 = perfect agreement with the y=x line; penalises bias, unlike Pearson.
    """
    vx, vy = _var(xs), _var(ys)
    denom = vx + vy + (_mean(xs) - _mean(ys)) ** 2
    if denom == 0:
        return None
    return 2 * _cov(xs, ys) / denom


def _avg_ranks(xs: list[float]) -> list[float]:
    """Average ranks (ties share the mean of their rank positions)."""
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0   # 1-based average rank over the tie block
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def _spearman(xs: list[float], ys: list[float]) -> float | None:
    """Spearman rank correlation = Pearson on average ranks."""
    return _pearson(_avg_ranks(xs), _avg_ranks(ys))


# ── load + aggregate ─────────────────────────────────────────────────────────
def collect_pairs(results_path: Path) -> dict[str, dict[str, list[float]]]:
    """Walk inference_results.json → {feature: {"pred": [...], "gt": [...]}}.

    Reads the per_feature breakdown that SFS already stored (claimed vs actual).
    """
    with open(results_path) as f:
        entries = json.load(f)

    pairs: dict[str, dict[str, list[float]]] = {}
    for e in entries:
        for pf in e.get("per_feature", []):
            feat = _canon(pf["feature"])
            try:
                pred = float(pf["claimed"])
                gt = float(pf["actual"])
            except (TypeError, ValueError, KeyError):
                continue
            if not (math.isfinite(pred) and math.isfinite(gt)):
                continue
            d = pairs.setdefault(feat, {"pred": [], "gt": []})
            d["pred"].append(pred)
            d["gt"].append(gt)
    return pairs, len(entries)


def coverage_denominator(features_csv: Path | None) -> dict[str, int]:
    """For each feature, count clips whose CSV cell is present (non-missing).

    This is the # of clips for which the feature is genuinely applicable —
    the correct denominator for coverage. Returns {} if no CSV given.
    """
    if features_csv is None:
        return {}
    denom: dict[str, int] = {feat: 0 for feat in CSV_COL}
    with open(features_csv) as f:
        for row in csv.DictReader(f):
            for feat, col in CSV_COL.items():
                raw = (row.get(col) or "").strip().lower()
                if raw not in ("", "nan", "n/a", "na", "none"):
                    denom[feat] += 1
    return denom


# ── reporting ────────────────────────────────────────────────────────────────
def _fmt(v: float | None, nd: int = 3) -> str:
    return "  n/a" if v is None else f"{v:.{nd}f}"


def build_report(
    pairs: dict[str, dict[str, list[float]]],
    n_clips: int,
    cov_denom: dict[str, int],
    min_pairs: int,
) -> str:
    header = (
        f"{'feature':<16} {'n':>5} {'cover':>6} "
        f"{'MAE':>9} {'NMAE':>7} {'CCC':>7} {'PCC':>7} {'SRCC':>7}"
    )
    lines = [header, "-" * len(header)]
    # Stable, catalog-ish order; unknown features appended alphabetically.
    order = list(CSV_COL.keys())
    for feat in sorted(pairs):
        if feat not in order:
            order.append(feat)

    for feat in order:
        if feat not in pairs:
            continue
        pred, gt = pairs[feat]["pred"], pairs[feat]["gt"]
        n = len(pred)
        denom = cov_denom.get(feat, n_clips) or n_clips
        cover = n / denom if denom else float("nan")
        if n < min_pairs:
            lines.append(f"{feat:<16} {n:>5} {cover:>6.2f}   (too few pairs)")
            continue
        mae = _mean([abs(p - g) for p, g in zip(pred, gt)])
        gtstd = _std(gt)
        nmae = mae / gtstd if gtstd > 0 else None
        lines.append(
            f"{feat:<16} {n:>5} {cover:>6.2f} "
            f"{mae:>9.3f} {_fmt(nmae):>7} {_fmt(_ccc(pred, gt)):>7} "
            f"{_fmt(_pearson(pred, gt)):>7} {_fmt(_spearman(pred, gt)):>7}"
        )

    note = (
        "\nReading guide:\n"
        "  cover  emission rate (n matched pairs / n clips with that GT). Low cover\n"
        "         means the model stays silent on this feature; MAE/CCC below are\n"
        "         then only over the clips it chose to answer, so read them together.\n"
        "  MAE    native units (Hz, dB, syl/s, ...). Do NOT average across rows.\n"
        "  NMAE   MAE / std(GT): unit-free, comparable across features. <1 = better\n"
        "         than always predicting the mean.\n"
        "  CCC    agreement with y=x (accuracy + bias). The headline faithfulness #.\n"
        "  PCC    correlation only. PCC high but CCC low => tracks but biased.\n"
        "  SRCC   rank agreement; robust to outliers.\n"
    )
    return "\n".join(lines) + "\n" + note


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", type=Path, required=True,
                   help="inference_results.json produced by src/inference.py")
    p.add_argument("--features_csv", type=Path, default=None,
                   help="optional: GT CSV, for exact coverage denominators")
    p.add_argument("--output_md", type=Path, default=None,
                   help="optional: also write the table to this markdown file")
    p.add_argument("--min_pairs", type=int, default=5,
                   help="skip stats for features with fewer matched pairs (default 5)")
    args = p.parse_args()

    if not args.results.exists():
        sys.exit(f"results file not found: {args.results}")

    pairs, n_clips = collect_pairs(args.results)
    if not pairs:
        sys.exit("no per_feature pairs found — was this inference run with GT "
                 "(features CSV / target text)? Nothing to score.")
    cov_denom = coverage_denominator(args.features_csv)
    report = build_report(pairs, n_clips, cov_denom, args.min_pairs)

    print(f"\nBand-free SFS regression metrics  ({n_clips} clips)\n")
    print(report)

    if args.output_md:
        args.output_md.write_text(
            f"# Band-free SFS regression metrics ({n_clips} clips)\n\n"
            f"```\n{report}\n```\n"
        )
        print(f"\n[written] {args.output_md}")


if __name__ == "__main__":
    main()
