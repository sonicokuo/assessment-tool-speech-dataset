"""Risk-coverage / AURC evaluation for the reliability (abstention) head.

This is the thesis-proof artifact for "Observability-Aware Speech-Feature
Description". The reliability head emits a per-(clip, feature) σ = exp(0.5·log σ²);
σ is the model's own "this number is unreliable" score. The claim is that abstaining
on the highest-σ predictions first removes the WRONG predictions first, so accuracy on
the answered subset rises as coverage drops. We quantify that with a risk-coverage
curve and its area under the risk-coverage curve (AURC), versus two baselines:

  - random abstention  : abstain in a random order (σ carries no information).
  - always-answer      : coverage fixed at 1.0; the operating point with no abstention.

CONVENTIONS
-----------
We work with one flat array of items, each an (uncertainty, correct) pair, where
`correct ∈ {0,1}` is SFS correctness for that scored claim (1 = the generated number
matched GT within the SFS tolerance). Risk = error rate = 1 - precision on the
ANSWERED subset. We sweep a coverage grid: at coverage c we answer the c-fraction of
items with the LOWEST uncertainty (most confident) and abstain on the rest.

  - `risk_coverage_curve(uncertainty, correct)` -> (coverage[], risk[]) sorted by
    ascending uncertainty (confident-first). coverage runs from 1/N up to 1.0.
  - `aurc(coverage, risk)` -> scalar area under the risk-coverage curve (trapezoid).
    Lower is better: a perfectly-ordered detector pushes all errors to the
    high-uncertainty tail, so risk stays ~0 until forced to answer wrong items.
  - `risk_coverage_report(uncertainty, correct)` -> dict bundling the model curve, the
    random-abstention curve (closed form: risk == base error rate at every coverage),
    the always-answer point, the three AURCs, and an `aurc_gain_vs_random`.

Everything is pure numpy so the verifier can run it on CPU with synthetic inputs; the
training/inference side computes σ from the reliability head and `correct` from the
SFS scorer, then calls these.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


def _as_1d(uncertainty: Sequence[float], correct: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    u = np.asarray(uncertainty, dtype=np.float64).reshape(-1)
    c = np.asarray(correct, dtype=np.float64).reshape(-1)
    if u.shape != c.shape:
        raise ValueError(f"uncertainty {u.shape} and correct {c.shape} must be the same length")
    if u.size == 0:
        raise ValueError("need at least one item for a risk-coverage curve")
    return u, c


def risk_coverage_curve(
    uncertainty: Sequence[float],
    correct: Sequence[float],
) -> tuple[np.ndarray, np.ndarray]:
    """Confident-first risk-coverage curve.

    Sort items by ASCENDING uncertainty (answer the most-confident first). At each
    prefix of length k (k = 1..N) we have answered the k most-confident items; the
    coverage is k/N and the risk is the error rate (1 - mean correctness) over those k.

    Args:
        uncertainty: (N,) per-item abstention score (higher = less reliable, e.g. σ).
        correct:     (N,) per-item correctness in {0,1} (1 = SFS-correct).

    Returns:
        coverage: (N,) increasing from 1/N to 1.0.
        risk:     (N,) error rate over the answered prefix at each coverage.
    """
    u, c = _as_1d(uncertainty, correct)
    n = u.size
    # Ascending uncertainty → most-confident answered first. Stable sort so ties keep
    # input order (deterministic).
    order = np.argsort(u, kind="stable")
    correct_sorted = c[order]
    # Cumulative correctness over the answered prefix.
    cum_correct = np.cumsum(correct_sorted)
    k = np.arange(1, n + 1, dtype=np.float64)
    coverage = k / n
    risk = 1.0 - (cum_correct / k)
    return coverage, risk


def aurc(coverage: np.ndarray, risk: np.ndarray) -> float:
    """Area under the risk-coverage curve (trapezoidal). Lower is better.

    Integrates risk over coverage. With a single point the area is 0 by definition.
    """
    coverage = np.asarray(coverage, dtype=np.float64).reshape(-1)
    risk = np.asarray(risk, dtype=np.float64).reshape(-1)
    if coverage.size < 2:
        return 0.0
    # np.trapezoid (numpy>=2.0) supersedes the deprecated np.trapz; fall back for
    # older numpy so this runs on both the local env and PSC.
    _trap = getattr(np, "trapezoid", None) or np.trapz
    return float(_trap(risk, coverage))


def random_abstention_curve(
    correct: Sequence[float],
) -> tuple[np.ndarray, np.ndarray]:
    """Closed-form risk-coverage curve for RANDOM abstention.

    If we abstain in a random (uninformative) order, the EXPECTED error rate over any
    randomly-chosen answered subset equals the base error rate of the whole set, at
    every coverage. So the expected risk is flat at (1 - mean correctness). We return
    it on the same coverage grid as `risk_coverage_curve` for a fair AURC comparison.
    """
    c = np.asarray(correct, dtype=np.float64).reshape(-1)
    if c.size == 0:
        raise ValueError("need at least one item")
    n = c.size
    base_risk = 1.0 - float(c.mean())
    coverage = np.arange(1, n + 1, dtype=np.float64) / n
    risk = np.full(n, base_risk, dtype=np.float64)
    return coverage, risk


def risk_coverage_report(
    uncertainty: Sequence[float],
    correct: Sequence[float],
) -> dict:
    """Full risk-coverage report: model curve + baselines + AURCs.

    Returns a dict with:
        coverage              : (N,) coverage grid (list of float).
        risk_model            : (N,) model risk, confident-first by uncertainty.
        risk_random           : (N,) random-abstention risk (flat at base error rate).
        aurc_model            : float, area under the model risk-coverage curve.
        aurc_random           : float, area under the random-abstention curve.
        aurc_gain_vs_random   : float, aurc_random - aurc_model (POSITIVE = the
                                uncertainty ordering beats random; bigger is better).
        always_answer_risk    : float, error rate with no abstention (coverage 1.0).
        base_error_rate       : float, same as always_answer_risk (alias).
        n                     : int, number of items.

    A working abstention signal has aurc_model < aurc_random, i.e.
    aurc_gain_vs_random > 0.
    """
    u, c = _as_1d(uncertainty, correct)
    cov_m, risk_m = risk_coverage_curve(u, c)
    cov_r, risk_r = random_abstention_curve(c)
    a_model = aurc(cov_m, risk_m)
    a_random = aurc(cov_r, risk_r)
    always_answer = 1.0 - float(c.mean())
    return {
        "coverage": cov_m.tolist(),
        "risk_model": risk_m.tolist(),
        "risk_random": risk_r.tolist(),
        "aurc_model": a_model,
        "aurc_random": a_random,
        "aurc_gain_vs_random": a_random - a_model,
        "always_answer_risk": always_answer,
        "base_error_rate": always_answer,
        "n": int(u.size),
    }
