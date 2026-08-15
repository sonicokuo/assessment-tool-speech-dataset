"""Band-free faithfulness + selective-prediction metrics for AQUA-NL.

These score EMITTED numeric claims against MEASURED ground truth. There is NO
tolerance band anywhere here (the tolerance/precision "SFS" metric is retired):
everything is rank correlation, scale-normalized error, agreement, a proper score,
or a risk-coverage summary.

UNITS CONVENTION (load-bearing — audit finding "SIGMA-UNITS"):
The ReliabilityHead learns log_var in NORMALIZED (per-FEATURE_SCALES) units, while
its mean is in RAW units. Every uncertainty-consuming metric here (risk-coverage,
AURC, CRPS) MUST operate in normalized units:
    y_n = y / S_f,  mu_n = mu / S_f,  sigma_n = exp(0.5 * log_var)
    raw sigma = sigma_n * S_f
Callers pass already-normalized (loss, sigma) or a per-feature scale S_f so the
functions can normalize. Mixing raw and normalized units silently corrupts AURC/CRPS.

This module is pure numpy/math (no torch, no repo imports) so it is unit-testable
without a GPU. Formulas are the audited-correct ones with the naive-wrong traps
avoided (see per-function notes). Primitives (norm_cdf, Holm) are reimplemented
inline to match src/eval/metrics_calibrated.py rather than importing it (keeps this
module torch-free for local testing).
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

_SQRT2 = math.sqrt(2.0)
_SQRT_PI = math.sqrt(math.pi)
_INV_SQRT_2PI = 1.0 / math.sqrt(2.0 * math.pi)


# ---------------------------------------------------------------------------
# tiny primitives (match metrics_calibrated.py; inlined to stay torch-free)
# ---------------------------------------------------------------------------
def norm_cdf(z: float) -> float:
    """Standard normal CDF Phi(z) via erf."""
    return 0.5 * (1.0 + math.erf(z / _SQRT2))


def norm_pdf(z: float) -> float:
    """Standard normal PDF phi(z)."""
    return _INV_SQRT_2PI * math.exp(-0.5 * z * z)


def _clean_pairs(preds: Sequence[float], gts: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    """Drop pairs where either side is nan/inf. Returns float64 arrays."""
    p = np.asarray(preds, dtype=np.float64)
    g = np.asarray(gts, dtype=np.float64)
    if p.shape != g.shape:
        raise ValueError(f"preds/gts shape mismatch: {p.shape} vs {g.shape}")
    ok = np.isfinite(p) & np.isfinite(g)
    return p[ok], g[ok]


# ---------------------------------------------------------------------------
# E3 — bias (signed mean error)
# ---------------------------------------------------------------------------
def bias(preds: Sequence[float], gts: Sequence[float]) -> float | None:
    """Signed mean error = mean(pred - gt). Positive => model over-reports.

    Conditioned on emitted (pred present) + GT present, same selection as SRCC.
    """
    p, g = _clean_pairs(preds, gts)
    if p.size == 0:
        return None
    return float(np.mean(p - g))


# ---------------------------------------------------------------------------
# E4 — CCC (Lin's concordance correlation coefficient)
# ---------------------------------------------------------------------------
def ccc(preds: Sequence[float], gts: Sequence[float]) -> float | None:
    """Lin (1989) concordance: 2*s_xy / (s_x^2 + s_y^2 + (mx - my)^2).

    Penalizes scale/location shift AND scatter (fixes SRCC's blind spot: a model
    that reports 2x the true value every clip gets SRCC 1.0 but low CCC).

    TRAP AVOIDED: all moments use explicit 1/n (Lin's biased convention). Do NOT
    mix np.cov (ddof=1) with a hand 1/n variance — that silently biases CCC.
    """
    x, y = _clean_pairs(preds, gts)
    n = x.size
    if n < 2:
        return None
    mx, my = x.mean(), y.mean()
    vx = np.mean((x - mx) ** 2)          # 1/n
    vy = np.mean((y - my) ** 2)          # 1/n
    sxy = np.mean((x - mx) * (y - my))   # 1/n
    denom = vx + vy + (mx - my) ** 2
    if denom == 0.0:
        return None
    return float(2.0 * sxy / denom)


# ---------------------------------------------------------------------------
# E10 — Bland-Altman (Krouwer variant: difference vs reference)
# ---------------------------------------------------------------------------
def bland_altman(preds: Sequence[float], gts: Sequence[float]) -> dict:
    """Agreement plot data. Classic BA plots diff vs MEAN of the two methods;
    since GT is a *reference instrument*, we use the Krouwer (2008) variant:
    difference (pred - gt) vs the reference (gt). Returns the arrays + bias and
    +/-1.96 SD limits of agreement. (Cite the variant in the paper.)
    """
    p, g = _clean_pairs(preds, gts)
    if p.size == 0:
        return {"n": 0}
    diff = p - g
    md = float(np.mean(diff))
    sd = float(np.std(diff, ddof=1)) if p.size > 1 else 0.0
    return {
        "n": int(p.size),
        "reference": g,          # x-axis (Krouwer)
        "difference": diff,      # y-axis
        "mean_diff": md,
        "loa_low": md - 1.96 * sd,
        "loa_high": md + 1.96 * sd,
    }


# ---------------------------------------------------------------------------
# E5/E6/E7 — risk-coverage curve, AURC, E-AURC, AUGRC
# ---------------------------------------------------------------------------
# All operate on (loss, confidence) where higher confidence => emit first.
# For the sigma-head: confidence = -sigma_n; loss = |pred - gt| / S_f (NORMALIZED,
# FIXED scale so risk is comparable across coverage/models — audit E2 caveat).
def _sorted_losses(losses: Sequence[float], confidences: Sequence[float]) -> np.ndarray:
    ls = np.asarray(losses, dtype=np.float64)
    cs = np.asarray(confidences, dtype=np.float64)
    if ls.shape != cs.shape:
        raise ValueError("losses/confidences shape mismatch")
    ok = np.isfinite(ls) & np.isfinite(cs)
    ls, cs = ls[ok], cs[ok]
    # descending confidence; stable so ties keep input order
    order = np.argsort(-cs, kind="stable")
    return ls[order]


def risk_coverage_curve(losses: Sequence[float], confidences: Sequence[float]) -> dict:
    """Sweep the abstention threshold. Emit the k most-confident, k=1..n.
    Returns coverage = k/n and selective_risk R(k) = mean loss of top-k.
    """
    s = _sorted_losses(losses, confidences)
    n = s.size
    if n == 0:
        return {"n": 0, "coverage": np.array([]), "selective_risk": np.array([])}
    cum = np.cumsum(s)
    k = np.arange(1, n + 1)
    return {
        "n": int(n),
        "coverage": k / n,
        "selective_risk": cum / k,          # R(k) = (1/k) sum top-k
    }


def aurc(losses: Sequence[float], confidences: Sequence[float]) -> float | None:
    """Area under the risk-coverage curve = (1/n) sum_k R(k),
    R(k) = (1/k) sum of the k lowest-loss-if-confidence-ordered.
    Matches metrics_calibrated.aurc (works for continuous losses).
    """
    rc = risk_coverage_curve(losses, confidences)
    if rc["n"] == 0:
        return None
    return float(np.mean(rc["selective_risk"]))


def e_aurc(losses: Sequence[float], confidences: Sequence[float]) -> float | None:
    """Excess AURC = AURC(model ordering) - AURC(oracle ordering). >= 0.
    Oracle orders by ascending true loss (confidence = -loss). 0 => the model's
    confidence ordering is loss-optimal. TRAP: oracle is best ORDERING of THESE
    losses, not zero risk.
    """
    a = aurc(losses, confidences)
    ls = np.asarray(losses, dtype=np.float64)
    ok = np.isfinite(ls)
    a_star = aurc(ls[ok], -ls[ok])
    if a is None or a_star is None:
        return None
    return float(a - a_star)


def augrc(losses: Sequence[float], confidences: Sequence[float]) -> float | None:
    """Area under the GENERALIZED risk-coverage curve (Traub 2024).
    Generalized risk GR(k) = coverage * selective_risk = (k/n) * R(k)
                           = (1/n) sum of top-k losses.
    AUGRC = (1/n) sum_k GR(k). Robust to AURC's over-weighting of high-confidence
    failures. Equivalent form asserted in tests: (1/n) sum_k (k/n) R(k).
    """
    s = _sorted_losses(losses, confidences)
    n = s.size
    if n == 0:
        return None
    cum = np.cumsum(s)          # cum[k-1] = sum of top-k losses
    gr = cum / n                # GR(k) = (1/n) sum top-k
    return float(np.mean(gr))


# ---------------------------------------------------------------------------
# E9 — Gaussian closed-form CRPS
# ---------------------------------------------------------------------------
def crps_gaussian(mu: float, sigma: float, y: float) -> float:
    """CRPS(N(mu, sigma^2), y) = sigma * [ z(2 Phi(z) - 1) + 2 phi(z) - 1/sqrt(pi) ],
    z = (y - mu)/sigma. Negatively oriented (lower is better). As sigma -> 0 it
    degenerates to |y - mu|.

    TRAPS AVOIDED: (1) the constant is 1/sqrt(pi) ~= 0.5642 (NOT 1/(2 sqrt(pi)));
    (2) sigma must be in the SAME units as (y - mu); (3) sigma <= 0 handled.

    NOTE: closed-form Gaussian CRPS is misspecified for small-support integers
    (pause_count) — use as a secondary column only, never a headline.
    """
    if not (math.isfinite(mu) and math.isfinite(sigma) and math.isfinite(y)):
        return float("nan")
    if sigma <= 0.0:
        return abs(y - mu)
    z = (y - mu) / sigma
    return float(sigma * (z * (2.0 * norm_cdf(z) - 1.0) + 2.0 * norm_pdf(z) - 1.0 / _SQRT_PI))


# ---------------------------------------------------------------------------
# E11 — Diebold-Mariano-style paired loss-differential test (cluster-robust)
# ---------------------------------------------------------------------------
def dm_paired(
    losses_a: Sequence[float],
    losses_b: Sequence[float],
    clusters: Sequence | None = None,
) -> dict:
    """Paired loss-differential test for "model A beats model B" per feature.
    d_i = L_i^A - L_i^B (use L = |pred - gt|/S_f, fixed scale). Reports z and a
    two-sided p. Negative z => A has lower loss (A better).

    CLUSTER-ROBUST (audit E11 risk): Libri2Mix test clips SHARE SPEAKERS, so
    per-clip d_i are cluster-dependent; a naive paired test understates variance
    and inflates significance. Pass `clusters` (e.g. speaker id per clip) to use a
    cluster-robust variance of dbar: Var = (1/n^2) sum_g (sum_{i in g}(d_i-dbar))^2.
    Without clusters, falls back to the i.i.d. paired variance s^2/n (lag-0 DM).
    """
    a = np.asarray(losses_a, dtype=np.float64)
    b = np.asarray(losses_b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError("losses_a/losses_b shape mismatch")
    ok = np.isfinite(a) & np.isfinite(b)
    d = (a - b)[ok]
    n = d.size
    if n < 2:
        return {"n": int(n), "dbar": None, "z": None, "p": None}
    dbar = float(np.mean(d))
    if clusters is not None:
        cl = np.asarray(clusters)[ok]
        resid = d - dbar
        var = 0.0
        for g in np.unique(cl):
            sg = float(np.sum(resid[cl == g]))
            var += sg * sg
        var /= (n * n)
    else:
        var = float(np.var(d, ddof=1)) / n
    if var <= 0.0:
        return {"n": int(n), "dbar": dbar, "z": None, "p": None}
    z = dbar / math.sqrt(var)
    p = 2.0 * (1.0 - norm_cdf(abs(z)))
    return {"n": int(n), "dbar": dbar, "z": float(z), "p": float(p)}


def holm_correction(pvals: dict[str, float], alpha: float = 0.05) -> dict[str, dict]:
    """Holm-Bonferroni step-down FWER control over the per-feature DM p-values.
    Returns {name: {p, p_adj, reject}}. Matches metrics_calibrated.holm_correction.
    """
    items = [(k, v) for k, v in pvals.items() if v is not None and math.isfinite(v)]
    m = len(items)
    out: dict[str, dict] = {k: {"p": v, "p_adj": None, "reject": False} for k, v in pvals.items()}
    if m == 0:
        return out
    items.sort(key=lambda kv: kv[1])
    prev_adj = 0.0
    for rank, (k, p) in enumerate(items):
        p_adj = min(1.0, (m - rank) * p)
        p_adj = max(p_adj, prev_adj)   # enforce monotone non-decreasing
        prev_adj = p_adj
        out[k]["p_adj"] = p_adj
        out[k]["reject"] = p_adj <= alpha
    return out


# ---------------------------------------------------------------------------
# E12 — digit-drift  (needs aux_mean logged at inference; metric is node-free)
# ---------------------------------------------------------------------------
def digit_drift(emitted: Sequence[float], aux_mean: Sequence[float],
                scale: float | None = None) -> float | None:
    """Mean |emitted_number - aux_head_mean|, optionally /scale (normalized).
    Quantifies the digit-tokenization dilution: how far the LM's written number
    drifts from the model's own scalar estimate. Feed the free-decode arm.
    """
    e, a = _clean_pairs(emitted, aux_mean)
    if e.size == 0:
        return None
    drift = np.abs(e - a)
    if scale:
        drift = drift / float(scale)
    return float(np.mean(drift))


# ---------------------------------------------------------------------------
# sigma-units helpers (audit SIGMA-UNITS convention)
# ---------------------------------------------------------------------------
def sigma_from_logvar(log_var: float) -> float:
    """sigma in NORMALIZED units."""
    return math.exp(0.5 * log_var)


def sigma_raw(log_var: float, scale: float) -> float:
    """sigma in RAW (per-feature) units = normalized sigma * FEATURE_SCALES[f]."""
    return math.exp(0.5 * log_var) * scale
