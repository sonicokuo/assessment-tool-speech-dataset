"""selection_metric.py — band-free, lower-variance checkpoint-selection primitives.

WHY THIS EXISTS
---------------
`train.py` historically selected `best.pt` on `val_sfs_f1`. The research memo
(.claude/research/findings/research-target-style-and-selection.md, Part B) shows
that axis is unsafe for ranking checkpoints:

  * SFS-F1 SATURATES (~0.95): when many checkpoints sit near the ceiling, the
    between-checkpoint variance collapses toward the sampling-noise floor and the
    argmax is dominated by noise (Signal-and-Noise, Heineman 2508.13144). A
    CONTINUOUS metric (SRCC / nMAE) raises the eval SNR.
  * Computed on a tiny (32-clip) val subset, a Spearman has a Fisher-z half-width
    of ~+-0.20 at n=40 (Schonbrodt & Perugini 2013; Bonett & Wright 2000), wider
    than the gap between adjacent checkpoints — winner's curse.

The robust protocol the memo endorses, implemented here as PURE FUNCTIONS (no
torch / model code in the metric path, except the optional weight-averaging
helper) so they are CPU-unit-testable:

  1. band_free_val_scores  — parse each generated description, join to clean GT,
     compute per-feature SRCC (rank corr), nMAE (MAE / std(gt)), and coverage
     (fraction of clips that emit the claim) over the canonical 12-feature set.
     Band-free: no tolerance threshold, so it does not saturate.
  2. composite_score       — a single scalar for selection:
        mean_SRCC(reliable, non-degenerate)  -  lam_nmae * mean_nMAE
     with a HARD BLEU/ROUGE-L FLUENCY FLOOR (returns -inf below the floor so a
     numerically-good but degenerate-prose checkpoint can never win). `snr` is
     excluded per the memo (its scalar SRCC sits ~0 on Libri2Mix and only adds
     variance); features with too few paired clips are also excluded.
  3. ema                   — exponential moving average to DENOISE the selection
     signal across eval steps.
  4. avg_state_dicts       — SWA / model-soup parameter averaging over the last
     k above-floor checkpoints, for selecting a lower-variance final weight set
     (Izmailov UAI 2018; Wortsman ICML 2022; Signal-and-Noise +2.4% decision acc).

The metric primitives (Spearman, MAE) are reused from `metrics_calibrated`
(band-free, no scipy) so SRCC/nMAE here are byte-identical to the rest of the
calibrated-metrics pipeline.
"""
from __future__ import annotations

import math
from typing import Iterable, Sequence

try:  # package-relative when imported as src.selection_metric
    from .metrics_calibrated import spearman, mae
    from .sfs import HybridClaimParser, SFSScorer
    from .feature_set import (
        FEATURE_NAMES,
        RECOVERABLE_FEATURES,
        ILL_POSED_UNDER_OVERLAP_FEATURES,
    )
except ImportError:  # flat import when src/ is on sys.path (matches sfs.py style)
    from metrics_calibrated import spearman, mae
    from sfs import HybridClaimParser, SFSScorer
    from feature_set import (
        FEATURE_NAMES,
        RECOVERABLE_FEATURES,
        ILL_POSED_UNDER_OVERLAP_FEATURES,
    )


# The 12 canonical features SRCC/nMAE/coverage are computed over. Single source
# of truth = feature_set.FEATURE_NAMES so this auto-tracks the supervised set.
SELECTION_FEATURES: tuple[str, ...] = tuple(FEATURE_NAMES)

# Features whose scalar SRCC is degenerate-by-construction on Libri2Mix and must
# be dropped from the composite (the memo's "exclude snr-scalar ~0"). SNR has
# almost no clip-to-clip rank spread on the clean-mix targets, so its SRCC is a
# noise term. Kept as a named set (not hard-coded into composite_score) so it is
# auditable and overridable.
DEGENERATE_SELECTION_FEATURES: frozenset[str] = frozenset({"snr"})

# Minimum paired (pred, gt) clips before a feature's SRCC/nMAE is trusted in the
# composite. Below this the rank correlation is pure sampling noise.
DEFAULT_MIN_PAIRS: int = 5


def _scalar_claims(text: str, parser: HybridClaimParser) -> dict[str, float]:
    """First numeric claim per scorable scalar feature in `text` (skip overlap
    spans). Mirrors metrics_calibrated._scalar_claims so the two stay aligned,
    but restricts the key space to the 12-feature selection set.
    """
    out: dict[str, float] = {}
    for c in parser.parse(text):
        if c.feature in ("overlap_start", "overlap_end"):
            continue
        if c.feature in SELECTION_FEATURES and c.feature not in out:
            out[c.feature] = c.value
    return out


def _std(xs: Sequence[float]) -> float:
    """Population-style sample std (ddof=1). 0.0 if fewer than 2 points."""
    n = len(xs)
    if n < 2:
        return 0.0
    m = sum(xs) / n
    var = sum((x - m) ** 2 for x in xs) / (n - 1)
    return math.sqrt(var)


def band_free_val_scores(
    generated_texts: Sequence[str],
    filenames: Sequence[str],
    clean_gt: dict[str, dict[str, float]],
    *,
    parser: HybridClaimParser | None = None,
    features: Sequence[str] = SELECTION_FEATURES,
) -> dict:
    """Band-free per-feature SRCC / nMAE / coverage over a val generation set.

    Args:
        generated_texts: one generated description per clip.
        filenames:       per-clip key into `clean_gt` (same length / order as
                         `generated_texts`). A trailing extension on the filename
                         is tolerated (e.g. "clip_0001.wav" matches a GT key of
                         "clip_0001"), matching how the rest of the pipeline keys.
        clean_gt:        {clip_key: {feature: gt_value, ...}} — the clean-stem
                         ground-truth scalars. Features absent for a clip are
                         simply not paired.
        parser:          HybridClaimParser (one is built if omitted).
        features:        feature set to score (defaults to the canonical 12).

    For each (clip, feature) we add a (predicted, gt) PAIR only when BOTH the
    generated text asserts a number AND the clip's GT has that feature (the
    SQUIM/UTMOS co-present convention; abstained/omitted clips do not contribute
    a pair). Per feature:

        srcc     = Spearman rank corr over the paired (pred, gt)  (None if <2
                   pairs or zero variance)
        nmae     = MAE(pred, gt) / std(gt_over_pairs)  (band-free normalized MAE;
                   None if <2 pairs or zero GT spread)
        coverage = (# clips that emitted a parseable claim for this feature AND
                    had GT for it) / (# clips that had GT for it)
        n        = number of paired clips

    Returns:
        {
          "per_feature": {feat: {"srcc":.., "nmae":.., "coverage":.., "n":..,
                                 "n_gt":.., "n_emitted":..}, ...},
          "n_clips": int,
        }
    """
    parser = parser or HybridClaimParser()
    if len(generated_texts) != len(filenames):
        raise ValueError(
            f"generated_texts ({len(generated_texts)}) and filenames "
            f"({len(filenames)}) must be aligned"
        )

    # Normalize GT keys once: allow both "stem" and "stem.wav" lookups.
    def _gt_for(fname: str) -> dict[str, float] | None:
        if fname in clean_gt:
            return clean_gt[fname]
        stem = fname.rsplit(".", 1)[0]
        return clean_gt.get(stem)

    feats = tuple(features)
    pred_by: dict[str, list[float]] = {f: [] for f in feats}
    gt_by: dict[str, list[float]] = {f: [] for f in feats}
    n_gt_by: dict[str, int] = {f: 0 for f in feats}       # clips that HAVE gt[f]
    n_emit_by: dict[str, int] = {f: 0 for f in feats}     # of those, # emitting a claim

    for text, fname in zip(generated_texts, filenames):
        gt = _gt_for(fname)
        if not gt:
            continue
        claims = _scalar_claims(text or "", parser)
        for f in feats:
            if f not in gt:
                continue
            gv = gt[f]
            if gv is None or (isinstance(gv, float) and math.isnan(gv)):
                continue
            n_gt_by[f] += 1
            if f in claims:
                n_emit_by[f] += 1
                pred_by[f].append(float(claims[f]))
                gt_by[f].append(float(gv))

    per_feature: dict[str, dict] = {}
    for f in feats:
        preds, gts = pred_by[f], gt_by[f]
        n = len(preds)
        srcc = spearman(preds, gts) if n >= 2 else None
        m = mae(preds, gts) if n >= 1 else None
        gt_std = _std(gts)
        nmae = (m / gt_std) if (m is not None and gt_std > 0.0) else None
        coverage = (n_emit_by[f] / n_gt_by[f]) if n_gt_by[f] else 0.0
        per_feature[f] = {
            "srcc": srcc,
            "nmae": nmae,
            "coverage": coverage,
            "n": n,
            "n_gt": n_gt_by[f],
            "n_emitted": n_emit_by[f],
        }

    return {"per_feature": per_feature, "n_clips": len(generated_texts)}


def composite_score(
    per_feature: dict[str, dict],
    reliable_features: Iterable[str],
    *,
    lam_nmae: float = 0.5,
    bleu: float | None = None,
    bleu_floor: float | None = None,
    min_pairs: int = DEFAULT_MIN_PAIRS,
    degenerate_features: Iterable[str] = DEGENERATE_SELECTION_FEATURES,
) -> float:
    """Single selection scalar from a band_free_val_scores `per_feature` dict.

        composite = mean_SRCC(usable reliable features) - lam_nmae * mean_nMAE

    HARD FLUENCY FLOOR: if `bleu_floor` is set and `bleu` is below it (or `bleu`
    is None when a floor is required), returns -inf so the checkpoint can never
    be selected (a numerically-good but degenerate-prose epoch is rejected).

    A feature contributes to the means only if it is:
      * in `reliable_features` (the recoverable set; ill-posed features under
        overlap are excluded — the model is meant to hedge there, not be ranked
        on its number),
      * NOT in `degenerate_features` (snr-scalar by default; ~0 SRCC, pure
        variance per the memo),
      * NON-DEGENERATE in the data: has >= `min_pairs` paired clips and a defined
        SRCC (>=2 pairs, non-zero variance).

    The nMAE mean is taken over the SAME usable feature set (only those with a
    defined nmae). If NO feature is usable (e.g. the model emitted nothing
    parseable), the SRCC mean is 0.0 and the nMAE penalty is 0.0, so the
    composite is 0.0 — strictly worse than any checkpoint that produced signal,
    and still gated by the BLEU floor.

    Returns a float (possibly -inf).
    """
    # Hard fluency floor first: a degenerate-prose checkpoint is rejected outright.
    if bleu_floor is not None:
        if bleu is None or bleu < bleu_floor:
            return float("-inf")

    reliable = frozenset(reliable_features)
    degenerate = frozenset(degenerate_features)

    srccs: list[float] = []
    nmaes: list[float] = []
    for f, stats in per_feature.items():
        if f not in reliable or f in degenerate:
            continue
        if stats.get("n", 0) < min_pairs:
            continue
        s = stats.get("srcc")
        if s is not None:
            srccs.append(s)
        nm = stats.get("nmae")
        if nm is not None:
            nmaes.append(nm)

    mean_srcc = (sum(srccs) / len(srccs)) if srccs else 0.0
    mean_nmae = (sum(nmaes) / len(nmaes)) if nmaes else 0.0
    return mean_srcc - lam_nmae * mean_nmae


def ema(prev: float | None, new: float, beta: float = 0.7) -> float:
    """Exponential moving average of a scalar selection signal.

        ema_t = beta * ema_{t-1} + (1 - beta) * new

    `beta` is the weight on the HISTORY (higher beta = more smoothing / slower
    response). On the first call (prev is None) the EMA is the new value itself,
    so no warm-up bias is introduced. NaN `new` is passed through unchanged from
    `prev` (a failed eval should not corrupt the running average).
    """
    if not (0.0 <= beta < 1.0):
        raise ValueError(f"ema beta must be in [0,1), got {beta}")
    if new != new:  # nan guard
        return prev if prev is not None else new
    if prev is None:
        return float(new)
    return beta * prev + (1.0 - beta) * new


def avg_state_dicts(state_dicts: Sequence[dict]):
    """Parameter-averaged (SWA / model-soup) state dict over `state_dicts`.

    Returns a NEW state dict whose every tensor value is the elementwise mean of
    the corresponding tensors across the inputs. Use it to build a lower-variance
    FINAL checkpoint by averaging the weights of the last k epochs that cleared
    the fluency floor, instead of taking the single noisy argmax (Izmailov UAI
    2018; Wortsman ICML 2022; Signal-and-Noise reports +2.4% decision accuracy
    from averaging over final checkpoints).

    Contract:
      * all inputs must share the same key set (raises ValueError otherwise).
      * tensor-valued entries are averaged in float (then cast back to the first
        dict's dtype, so an int/bool buffer stays its type — averaging is applied
        in a higher-precision accumulator to avoid overflow/rounding drift).
      * non-tensor / non-averageable entries (e.g. a stray Python scalar) are
        copied from the FIRST state dict unchanged.

    torch is imported lazily so the rest of this module (and its CPU unit tests
    for the pure metric functions) does not depend on torch being installed.
    """
    if not state_dicts:
        raise ValueError("avg_state_dicts needs at least one state dict")
    if len(state_dicts) == 1:
        return dict(state_dicts[0])

    import torch  # lazy — only the weight-averaging path needs torch

    keys = set(state_dicts[0].keys())
    for i, sd in enumerate(state_dicts[1:], start=1):
        if set(sd.keys()) != keys:
            raise ValueError(
                f"state_dicts[{i}] key set differs from state_dicts[0]; "
                "cannot average mismatched checkpoints"
            )

    n = len(state_dicts)
    out: dict = {}
    for k in state_dicts[0].keys():
        v0 = state_dicts[0][k]
        if torch.is_tensor(v0):
            acc = state_dicts[0][k].detach().to(torch.float64).clone()
            for sd in state_dicts[1:]:
                acc += sd[k].detach().to(torch.float64)
            acc /= n
            out[k] = acc.to(v0.dtype)
        else:
            # Non-tensor entry (rare in a slim adapter/LoRA dict): keep first.
            out[k] = v0
    return out
