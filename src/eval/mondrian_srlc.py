"""Mondrian selective-risk-control for the observability-aware abstention head.

Turns the hand-set overlap threshold into a per-(feature x overlap-bin) finite-sample
guarantee. For each cell we choose the largest emission region (highest coverage) whose
UPPER confidence bound on the selective miscoverage risk stays <= alpha:

    emit claim f on clip i  iff  sigma_f(i) <= lambda_{f,g(i)}
    guarantee (per cell, w.p. >= 1 - delta):  P( normalized_error_f > k_f | emitted ) <= alpha

CORRECTNESS GUARDRAILS (from the 2026-07-12 audit):
- eps_f is FIRST-PRINCIPLES and FROZEN before any calibration: the miscoverage event is
  |pred - gt| > k_f * S_f, i.e. in NORMALIZED units normalized_error > k_f. k_f is a fixed
  multiple of the robust dispersion (a JND-like tolerance), NOT tuned to any correlation.
  This keeps the guarantee's epsilon consistent with the band-free / no-circular-tuning
  decision (there is one paper sentence to write reconciling the two).
- sigma is consumed in NORMALIZED units (ReliabilityHead learns log_var per FEATURE_SCALES).
- COUNTING / recoverable features are OUTSIDE the taxonomy: their GT is mix-measured, so
  overlap does not make them ill-posed; they are always emitted, never hedged.
- Only the CODE is node-free. A calibrated map is meaningful ONLY on a sigma-trained
  checkpoint (H2). Calibrating off the default-off / untrained head yields garbage; callers
  must not report numbers pre-H2.

Method: RCPS / Learn-then-Test fixed-sequence (Angelopoulos CRC 2024; Bates RCPS 2021;
Angelopoulos LTT 2021). The miscoverage loss is BINARY (1{normalized_error > k}), so the
tight, exact upper confidence bound is the Clopper-Pearson binomial UCB (Hoeffding is far too
loose at low empirical risk and would force spurious abstention on well-calibrated cells). Per
cell we scan every emission prefix (candidate threshold) and take the LARGEST whose CP UCB
<= alpha; because the UCB is U-shaped in the prefix size, we Bonferroni-correct delta over the
within-cell threshold grid (delta_grid = delta_cell / n_cell) rather than a fixed-sequence stop.
FWER across cells is a further Bonferroni: delta_cell = delta / n_active_cells. Pareto Testing
(Laufer-Goldshtein 2023) is the tighter alternative to both Bonferronis, noted as future work.
No torch (uses numpy + scipy) so it is unit-testable without a GPU.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
from scipy.stats import beta as _beta

# default overlap bins: none / partial / heavy. Right-closed on the last edge.
DEFAULT_OVERLAP_EDGES: tuple[float, ...] = (0.0, 1e-9, 0.5, 1.0 + 1e-9)
DEFAULT_OVERLAP_LABELS: tuple[str, ...] = ("none", "partial", "heavy")


def overlap_bin(overlap_ratio: float,
                edges: Sequence[float] = DEFAULT_OVERLAP_EDGES,
                labels: Sequence[str] = DEFAULT_OVERLAP_LABELS) -> str:
    """Map an overlap ratio to a discrete Mondrian bin label.
    Bin j covers [edges[j], edges[j+1]). overlap==0 -> 'none' (via the 1e-9 second edge).
    """
    if not math.isfinite(overlap_ratio):
        return labels[0]
    r = min(max(overlap_ratio, 0.0), 1.0)
    for j in range(len(labels)):
        if edges[j] <= r < edges[j + 1]:
            return labels[j]
    return labels[-1]


def _binom_ucb(n_fail: int, n: int, delta: float) -> float:
    """Clopper-Pearson (exact binomial) upper (1-delta) confidence bound on the failure
    probability given n_fail failures in n trials. Tight at low empirical risk:
    with 0 failures it is 1 - delta**(1/n), vs Hoeffding's much larger sqrt(ln(1/delta)/2n).
    """
    if n <= 0:
        return 1.0
    if n_fail >= n:
        return 1.0
    return float(_beta.ppf(1.0 - delta, n_fail + 1, n - n_fail))


def crc_cell(sigmas: Sequence[float],
             norm_errors: Sequence[float],
             alpha: float,
             delta: float,
             k: float) -> dict:
    """Calibrate one cell. Loss L_i = 1{ norm_error_i > k } (frozen first-principles k).
    Emit the most-confident (lowest sigma) clips; find the largest emission prefix whose
    Hoeffding UCB on the selective risk is <= alpha (fixed-sequence RCPS/LTT).

    Returns:
      feasible: bool  (False => "abstains by construction": no lambda achieves the risk)
      threshold: float sigma cutoff (np.inf if all emitted; None if infeasible)
      coverage, emp_risk, ucb, n
    """
    s = np.asarray(sigmas, dtype=np.float64)
    e = np.asarray(norm_errors, dtype=np.float64)
    ok = np.isfinite(s) & np.isfinite(e)
    s, e = s[ok], e[ok]
    n = s.size
    if n == 0:
        return {"feasible": False, "threshold": None, "coverage": 0.0,
                "emp_risk": None, "ucb": None, "n": 0}
    order = np.argsort(s, kind="stable")          # ascending sigma = most confident first
    L = (e[order] > k).astype(np.float64)
    s_sorted = s[order]
    cum = np.cumsum(L)                             # cum[kk-1] = #failures in top-kk
    # LTT over the within-cell threshold grid (each prefix kk = one candidate threshold).
    # The CP-UCB(kk) is U-shaped in kk (loose at tiny kk, tight mid, rises with failures),
    # so a fixed-sequence break is invalid; scan ALL candidates and take the largest that
    # passes, Bonferroni-correcting delta over the n candidate thresholds (union bound).
    delta_grid = delta / n
    best = 0
    for kk in range(1, n + 1):
        n_fail = int(cum[kk - 1])
        if _binom_ucb(n_fail, kk, delta_grid) <= alpha:
            best = kk                              # keep the LARGEST feasible (no break)
    if best == 0:
        return {"feasible": False, "threshold": None, "coverage": 0.0,
                "emp_risk": float(L.mean()), "ucb": None, "n": int(n)}
    thr = float("inf") if best == n else float(s_sorted[best - 1])
    n_fail = int(cum[best - 1])
    return {
        "feasible": True,
        "threshold": thr,
        "coverage": best / n,
        "emp_risk": float(n_fail / best),
        "ucb": _binom_ucb(n_fail, best, delta_grid),
        "n": int(n),
    }


def split_certify_cell(sigmas_pick: Sequence[float],
                       errs_pick: Sequence[float],
                       sigmas_cert: Sequence[float],
                       errs_cert: Sequence[float],
                       alpha: float,
                       delta: float,
                       k: float,
                       transfer: str = "quantile") -> dict:
    """Split-conformal variant of `crc_cell`: PICK the threshold on one sample, CERTIFY on a
    disjoint one. Strictly tighter than crc_cell's grid scan, at no cost in rigor.

    crc_cell must Bonferroni-correct delta over every candidate threshold it scans (n_cell of
    them), because it uses the same data to choose AND to bound. That costs roughly a factor
    of z(delta/n)/z(delta) in the UCB width -- at n=3000 it is the difference between
    certifying a cell whose true selective risk is 0.065 and rejecting it.

    Here the threshold is a function of the PICK half only, so it is fixed and independent
    w.r.t. the CERT half. One binomial test then suffices, and delta needs Bonferroni over
    CELLS only. This is the standard split-conformal / hold-out selective-risk argument
    (Angelopoulos CRC 2024 sec. "split"; Bates RCPS 2021).

    Selection rule on the pick half: the largest emission prefix whose pick-half UCB is
    <= alpha. Selecting on the point estimate instead lands the proposal exactly AT alpha,
    which leaves the certification step no sampling-error margin and rejects cells that are
    genuinely under alpha (observed: jitter/none, true risk 0.065, proposed at risk 0.096,
    certified UCB 0.127 > 0.1). Proposing against the UCB builds the margin in. This half
    never certifies, so its bound needs no multiplicity correction -- alpha and k are
    untouched, so this is a rule change, not a tuning knob.
    """
    sp = np.asarray(sigmas_pick, dtype=np.float64)
    ep = np.asarray(errs_pick, dtype=np.float64)
    ok = np.isfinite(sp) & np.isfinite(ep)
    sp, ep = sp[ok], ep[ok]
    sc_ = np.asarray(sigmas_cert, dtype=np.float64)
    ec = np.asarray(errs_cert, dtype=np.float64)
    ok2 = np.isfinite(sc_) & np.isfinite(ec)
    sc_, ec = sc_[ok2], ec[ok2]
    if sp.size == 0 or sc_.size == 0:
        return {"feasible": False, "threshold": None, "coverage": 0.0, "emp_risk": None,
                "ucb": None, "n_pick": int(sp.size), "n_cert": int(sc_.size)}

    order = np.argsort(sp, kind="stable")
    L = (ep[order] > k).astype(np.float64)
    s_sorted = sp[order]
    cum = np.cumsum(L)
    # Choose the prefix that MAXIMIZES COVERAGE SUBJECT TO the certification step succeeding.
    # "Largest prefix clearing alpha" is the wrong objective: it spends the entire risk budget
    # on coverage, so the cert-half UCB lands just over alpha and the cell is lost (observed:
    # jitter/none proposed at 33.6% coverage / risk 0.081 -> cert UCB 0.112, while the 18.7%
    # prefix at risk 0.054 would have certified at 0.084). The pick half knows n_cert, so it
    # can predict the cert UCB for each candidate and keep the largest that is projected to pass.
    n_cert = sc_.size
    best = 0
    for kk in range(1, sp.size + 1):
        risk_hat = cum[kk - 1] / kk
        n_emit_hat = max(1, int(round(n_cert * kk / sp.size)))
        proj = _binom_ucb(int(round(risk_hat * n_emit_hat)), n_emit_hat, delta)
        if proj <= alpha:
            best = kk
    if best == 0:
        return {"feasible": False, "threshold": None, "coverage": 0.0,
                "emp_risk": float(L.mean()), "ucb": None,
                "n_pick": int(sp.size), "n_cert": int(sc_.size)}
    thr = float("inf") if best == sp.size else float(s_sorted[best - 1])

    # TRANSFER MODE. sigma is not perfectly scale-stable across splits (measured dev->test:
    # jitter/none mean 0.796 -> 0.815), so an ABSOLUTE threshold picked on the calibration
    # split lands at the wrong coverage on the certification split (19% intended -> 13.6%
    # actual), shrinking the emitted sample and inflating the CP width. Transferring the
    # QUANTILE instead keeps the intended coverage. The emission rule stays label-independent
    # -- it reads only sigma, never the error -- so the selective-risk bound still holds; the
    # rank-level dependence on the cert sample is the standard Mondrian selection argument.
    if transfer == "quantile":
        q = best / sp.size
        n_take = max(1, int(round(q * sc_.size)))
        order_c = np.argsort(sc_, kind="stable")
        keep = order_c[:n_take]
        emitted = np.zeros(sc_.size, dtype=bool)
        emitted[keep] = True
        thr = float(sc_[order_c[n_take - 1]])
    else:
        emitted = sc_ <= thr
    n_e = int(emitted.sum())
    if n_e == 0:
        return {"feasible": False, "threshold": thr, "coverage": 0.0, "emp_risk": None,
                "ucb": None, "n_pick": int(sp.size), "n_cert": int(sc_.size)}
    n_f = int((ec[emitted] / 1.0 > k).sum())
    ucb = _binom_ucb(n_f, n_e, delta)                   # ONE test: no grid Bonferroni
    return {
        "feasible": bool(ucb <= alpha),
        "threshold": thr,
        "coverage": n_e / sc_.size,
        "emp_risk": n_f / n_e,
        "ucb": float(ucb),
        "n_pick": int(sp.size),
        "n_cert": int(sc_.size),
    }


def mondrian_calibrate(
    records: Sequence[dict],
    *,
    feature_names: Sequence[str],
    scales: Sequence[float],
    ill_posed_features: frozenset,
    alpha: float = 0.1,
    delta: float = 0.05,
    k: float | dict = 1.0,
    edges: Sequence[float] = DEFAULT_OVERLAP_EDGES,
    labels: Sequence[str] = DEFAULT_OVERLAP_LABELS,
) -> dict:
    """Calibrate a per-(feature x overlap-bin) abstention map.

    records: per-clip dicts with keys:
        'overlap_ratio': float
        for each ill-posed feature f: 'sigma_<f>' (NORMALIZED sigma) and
                                      'err_<f>'  (raw |pred-gt|; normalized here by S_f)
      (missing sigma/err for a clip-feature just drops that clip from that cell.)

    Only ILL_POSED features get cells; recoverable/counting features are recorded as
    'always_emit' (never hedged). k is a fixed multiple of dispersion (float, or per-feature
    dict); the miscoverage event is |pred-gt|/S_f > k. delta is split Bonferroni across the
    active cells.

    Returns { 'cells': {(f,g): crc_cell(...)}, 'always_emit': [recoverable feats], 'params': {...} }.
    """
    name_to_scale = {f: float(s) for f, s in zip(feature_names, scales)}
    active_feats = [f for f in feature_names if f in ill_posed_features]
    always_emit = [f for f in feature_names if f not in ill_posed_features]

    # collect per-cell (sigma, norm_error) then Bonferroni-split delta by #active cells
    buckets: dict[tuple[str, str], list[tuple[float, float]]] = {}
    for rec in records:
        g = overlap_bin(rec.get("overlap_ratio", 0.0), edges, labels)
        for f in active_feats:
            sig = rec.get(f"sigma_{f}")
            err = rec.get(f"err_{f}")
            if sig is None or err is None:
                continue
            if not (math.isfinite(sig) and math.isfinite(err)):
                continue
            ne = err / name_to_scale[f]                 # normalize the error
            buckets.setdefault((f, g), []).append((float(sig), float(ne)))

    n_cells = max(len(buckets), 1)
    delta_cell = delta / n_cells
    cells: dict[tuple[str, str], dict] = {}
    for (f, g), pairs in buckets.items():
        arr = np.asarray(pairs, dtype=np.float64)
        kf = k[f] if isinstance(k, dict) else float(k)
        cells[(f, g)] = crc_cell(arr[:, 0], arr[:, 1], alpha=alpha, delta=delta_cell, k=kf)
        cells[(f, g)]["k"] = kf

    return {
        "cells": cells,
        "always_emit": always_emit,
        "params": {"alpha": alpha, "delta": delta, "delta_cell": delta_cell,
                   "n_cells": n_cells, "edges": list(edges), "labels": list(labels)},
    }


def _bucket(records, active_feats, name_to_scale, edges, labels):
    buckets: dict[tuple[str, str], list[tuple[float, float]]] = {}
    for rec in records:
        g = overlap_bin(rec.get("overlap_ratio", 0.0), edges, labels)
        for f in active_feats:
            sig, err = rec.get(f"sigma_{f}"), rec.get(f"err_{f}")
            if sig is None or err is None:
                continue
            if not (math.isfinite(sig) and math.isfinite(err)):
                continue
            buckets.setdefault((f, g), []).append((float(sig), err / name_to_scale[f]))
    return buckets


def mondrian_calibrate_split(
    pick_records: Sequence[dict],
    cert_records: Sequence[dict],
    *,
    feature_names: Sequence[str],
    scales: Sequence[float],
    ill_posed_features: frozenset,
    alpha: float = 0.1,
    delta: float = 0.05,
    k: float | dict = 1.0,
    transfer: str = "quantile",
    edges: Sequence[float] = DEFAULT_OVERLAP_EDGES,
    labels: Sequence[str] = DEFAULT_OVERLAP_LABELS,
) -> dict:
    """Split-conformal Mondrian calibration: choose thresholds on `pick_records`, certify on
    the disjoint `cert_records`. See `split_certify_cell` for why this is tighter than the
    single-sample grid scan in `mondrian_calibrate` while remaining a valid finite-sample
    guarantee. delta is Bonferroni-split across CELLS only (not across thresholds).
    """
    name_to_scale = {f: float(s) for f, s in zip(feature_names, scales)}
    active_feats = [f for f in feature_names if f in ill_posed_features]
    always_emit = [f for f in feature_names if f not in ill_posed_features]

    bp = _bucket(pick_records, active_feats, name_to_scale, edges, labels)
    bc = _bucket(cert_records, active_feats, name_to_scale, edges, labels)
    keys = sorted(set(bp) | set(bc))
    delta_cell = delta / max(len(keys), 1)

    cells: dict[tuple[str, str], dict] = {}
    for key in keys:
        f = key[0]
        kf = k[f] if isinstance(k, dict) else float(k)
        p = np.asarray(bp.get(key, []), dtype=np.float64).reshape(-1, 2)
        c = np.asarray(bc.get(key, []), dtype=np.float64).reshape(-1, 2)
        cells[key] = split_certify_cell(p[:, 0], p[:, 1], c[:, 0], c[:, 1],
                                        alpha=alpha, delta=delta_cell, k=kf, transfer=transfer)
        cells[key]["k"] = kf
    return {
        "cells": cells,
        "always_emit": always_emit,
        "params": {"alpha": alpha, "delta": delta, "delta_cell": delta_cell,
                   "n_cells": len(keys), "edges": list(edges), "labels": list(labels),
                   "method": "split", "transfer": transfer},
    }


def should_emit(cal: dict, feature: str, sigma_norm: float, overlap_ratio: float,
                edges: Sequence[float] = DEFAULT_OVERLAP_EDGES,
                labels: Sequence[str] = DEFAULT_OVERLAP_LABELS) -> bool:
    """Apply a calibrated map at inference. Recoverable features always emit; ill-posed
    features emit iff sigma <= the cell threshold; an infeasible/absent cell abstains.
    """
    if feature in cal["always_emit"]:
        return True
    g = overlap_bin(overlap_ratio, edges, labels)
    cell = cal["cells"].get((feature, g))
    if cell is None or not cell.get("feasible"):
        return False
    return sigma_norm <= cell["threshold"]
