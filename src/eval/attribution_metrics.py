"""Corrected attribution metrics for per-feature evidence maps.

SUPERSEDES the scratchpad `attribution_analyze.py`, which reported a FALSE NEGATIVE
("maps are flat and not clip-specific") through two defects in the metrics themselves.
Full record: `.claude/research/EXPLAINABILITY_2026-08-07.md`.

DEFECT 1 -- the specificity sign bug (the one that mattered).
    old:  spec = AUC_own - mean(AUC_others)                        # ONE-SIDED
A map that is clip-specifically ANTI-aligned -- concentrating on its own clip's
NON-overlapped frames -- is the PHYSICALLY CORRECT behaviour for f0/f0_sd/jitter/
shimmer/hnr, because those quantities are unrecoverable under two-talker overlap.
The one-sided statistic scores exactly that behaviour NEGATIVE. Corrected:
    new:  spec = |AUC_own - 0.5| - mean_j |AUC_j - 0.5|            # TWO-SIDED
Under the correction every feature flips from negative to positive, landing at
40-55% of the oracle ceiling.

DEFECT 2 -- entropy measured VALUE MAGNITUDE, not flatness.
Head biases are ~0, so per-frame z_t carries the broadcast clip-level value at every
t, and entropy of |z| is driven to 1 by arithmetic (rho(concentration, AC/DC) = -0.993).
Corrected by (a) scoring the DEMEANED map, and (b) referencing the iid-Gaussian null
at the same token count instead of against 1.0. NOTE the resulting number is still
near-useless as a quality signal: for GLOBAL features (snr, speaking_rate) a broad map
is PHYSICALLY CORRECT, so low entropy is not a goal. Never optimise it.

Reported alongside:
  * a PERMUTATION NULL (map shuffled in time) for every specificity value;
  * the ORACLE ceiling (each clip's own GT mask used as its map), because pairwise
    overlap masks agree on ~75% of frames and the ceiling is ~0.25, not 1.0;
  * the SIGNED deviation correlation against the CONTINUOUS overlap fraction, which
    is the cleanest single statistic -- it needs no AUC and keeps the sign;
  * n_scored, because all-zero-mask (clean) clips are undefined for AUC and were
    silently dropped by the old script (400 captured -> ~200 actually scored).
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "auc",
    "normalised_entropy",
    "iid_entropy_null",
    "score_feature",
    "score_all",
]


def auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Rank AUC of `scores` against binary `labels`; 0.5 = chance, NaN if degenerate."""
    m = np.isfinite(scores) & np.isfinite(labels)
    s = scores[m]
    l = (labels[m] > 0.5).astype(int)
    n1 = int(l.sum())
    n0 = len(l) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), float)
    ranks[order] = np.arange(1, len(s) + 1)
    # average ranks over ties so constant maps score exactly 0.5 rather than 0 or 1
    _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    if (cnt > 1).any():
        sums = np.bincount(inv, weights=ranks)
        ranks = (sums / cnt)[inv]
    return float((ranks[l == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def normalised_entropy(v: np.ndarray) -> float:
    """Shannon entropy of |v| as a distribution over time, normalised to [0,1]; 1 = flat."""
    a = np.abs(v)
    a = a[np.isfinite(a)]
    if len(a) < 2 or a.sum() <= 0:
        return np.nan
    p = a / a.sum()
    return float(-(p * np.log(p + 1e-12)).sum() / np.log(len(p)))


_NULL_CACHE: dict[int, float] = {}


def iid_entropy_null(n: int, draws: int = 400, seed: int = 0) -> float:
    """Expected normalised entropy of |iid Gaussian| at length n.

    This is the reference an UNSTRUCTURED map achieves. At n~29 it is ~0.92, not 1.0,
    so comparing raw entropy against 1.0 systematically overstates flatness.
    """
    if n < 2:
        return np.nan
    if n not in _NULL_CACHE:
        rng = np.random.default_rng(seed + n)
        vals = [normalised_entropy(rng.standard_normal(n)) for _ in range(draws)]
        _NULL_CACHE[n] = float(np.nanmean(vals))
    return _NULL_CACHE[n]


def boot_ci(
    vals: list[float], n_boot: int = 2000, alpha: float = 0.05, seed: int = 0
) -> tuple[float, float]:
    """Percentile bootstrap CI over CLIPS for a per-clip statistic.

    The oracle ceiling and every specificity value are means over ~200 mixtures, so they
    carry real sampling error. Reporting "52% of ceiling" from two point estimates hides
    error in BOTH the numerator and the denominator — this makes it visible.
    """
    v = np.asarray([x for x in vals if np.isfinite(x)], float)
    if len(v) < 8:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(v), size=(n_boot, len(v)))
    means = v[idx].mean(axis=1)
    return (float(np.percentile(means, 100 * alpha / 2)),
            float(np.percentile(means, 100 * (1 - alpha / 2))))


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    m = np.isfinite(a) & np.isfinite(b)
    a, b = a[m], b[m]
    if len(a) < 4 or a.std() < 1e-9 or b.std() < 1e-9:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def score_feature(
    contrib: np.ndarray,
    overlap: np.ndarray,
    n_others: int = 32,
    n_perm: int = 10,
    seed: int = 0,
) -> dict:
    """Score one feature's maps across all clips.

    contrib : (n_clip, T) exact per-token contributions, may contain NaN padding
    overlap : (n_clip, T) per-frame overlap in [0,1] (continuous preferred over binary)
    """
    rng = np.random.default_rng(seed)
    n_clip = contrib.shape[0]

    ent_raw, ent_dev, ent_null = [], [], []
    align, spec_1s, spec_2s, spec_perm, dev_r = [], [], [], [], []
    n_scored = 0
    n_degenerate = 0

    # pre-extract valid slices once
    slices = []
    for i in range(n_clip):
        c, o = contrib[i], overlap[i]
        m = np.isfinite(c) & np.isfinite(o)
        slices.append((c[m], o[m]))

    for i in range(n_clip):
        c, o = slices[i]
        if len(c) < 8 or not np.isfinite(c).any():
            continue
        a_abs = np.abs(c)
        dev = c - c.mean()

        e_raw = normalised_entropy(a_abs)
        e_dev = normalised_entropy(dev)
        if np.isfinite(e_raw):
            ent_raw.append(e_raw)
            ent_null.append(iid_entropy_null(len(c)))
        if np.isfinite(e_dev):
            ent_dev.append(e_dev)

        # signed deviation correlation vs CONTINUOUS overlap -- keeps the sign,
        # needs no binarisation, and is defined even when the mask is not binary
        r = _pearson(dev, o)
        if np.isfinite(r):
            dev_r.append(r)

        # AUC-based stats need a non-degenerate binary mask; clean clips (all-zero
        # overlap) are UNDEFINED here. Count them rather than dropping them silently.
        a_own = auc(a_abs, o)
        if not np.isfinite(a_own):
            n_degenerate += 1
            continue
        n_scored += 1
        align.append(a_own)

        others = []
        for j in rng.choice(n_clip, size=min(n_others, n_clip), replace=False):
            if j == i:
                continue
            oj = slices[j][1]
            k = min(len(oj), len(a_abs))
            if k < 8:
                continue
            aj = auc(a_abs[:k], oj[:k])
            if np.isfinite(aj):
                others.append(aj)
        if not others:
            continue
        others = np.asarray(others)
        spec_1s.append(a_own - others.mean())                                  # OLD, buggy
        spec_2s.append(abs(a_own - 0.5) - np.abs(others - 0.5).mean())         # CORRECTED

        # permutation null: same map, time order destroyed. A real clip-specific map
        # must beat this; a map that merely has the right marginal will not.
        perms = []
        for _ in range(n_perm):
            sh = rng.permutation(a_abs)
            a_sh = auc(sh, o)
            if not np.isfinite(a_sh):
                continue
            oth = [auc(sh[: min(len(slices[j][1]), len(sh))],
                       slices[j][1][: min(len(slices[j][1]), len(sh))])
                   for j in rng.choice(n_clip, size=min(8, n_clip), replace=False)]
            oth = np.asarray([x for x in oth if np.isfinite(x)])
            if len(oth):
                perms.append(abs(a_sh - 0.5) - np.abs(oth - 0.5).mean())
        if perms:
            spec_perm.append(float(np.mean(perms)))

    f = lambda x: float(np.nanmean(x)) if len(x) else np.nan
    # CIs over CLIPS. Without these, "X% of the oracle ceiling" hides sampling error in BOTH
    # the numerator and the denominator -- the ceiling is itself a ~200-clip mean.
    ci_spec = boot_ci(spec_2s)
    ci_dev = boot_ci(dev_r)
    return {
        "n_scored": n_scored,
        "n_degenerate": n_degenerate,
        "spec_twoside_ci": ci_spec,
        "dev_corr_ci": ci_dev,
        "ent_raw": f(ent_raw),
        "ent_dev": f(ent_dev),
        "ent_null": f(ent_null),
        "align": f(align),
        "spec_oneside_BUGGY": f(spec_1s),
        "spec_twoside": f(spec_2s),
        "spec_perm_null": f(spec_perm),
        "dev_corr": f(dev_r),
        "dev_corr_frac_strong": (
            float(np.mean(np.abs(dev_r) > 0.3)) if len(dev_r) else np.nan
        ),
    }


def score_all(contrib: np.ndarray, overlap: np.ndarray, features: list[str], **kw) -> dict:
    """Score every feature, plus the ORACLE row (own GT mask used as the map).

    The oracle is essential context: pairwise overlap masks agree on ~75% of frames,
    so a perfect map scores ~0.25 on two-sided specificity, NOT 1.0. Reporting a raw
    specificity without this ceiling makes a good map look mediocre.
    """
    out = {f: score_feature(contrib[:, i], overlap, **kw) for i, f in enumerate(features)}
    out["__ORACLE__"] = score_feature(overlap.copy(), overlap, **kw)
    return out
