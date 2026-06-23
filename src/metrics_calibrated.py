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

2. TIER-3 COVERAGE-GUARANTEED SFS (the DEFAULT meaning of SFS).
   The legacy SFS uses a single absolute tolerance per feature (`±2 dB` SNR,
   `±5 Hz` F0). That under-credits high-magnitude clips (a 2 dB miss on a 35 dB
   SNR is proportionally tiny) and over-credits low-magnitude ones. Tier-3
   derives the band from the MEASURED GT-noise model:

       tau_f(|gt|) = JND_f + k(alpha, family_f) * sigma_f(|gt|)

   where sigma_f is the measured per-estimator noise scale, family_f is chosen
   per feature by its measured tail (heavy-tailed -> distribution-free VP /
   Chebyshev, light -> two-sided Gaussian), and HETEROSCEDASTICITY is folded
   into sigma itself: sigma_f(|gt|) = max(sigma_const, rel_sigma*|gt|). The
   magnitude-dependence therefore comes from sigma, NOT a separate hand-set
   rel_frac. The old `rel_frac` term in ToleranceConfig is retained ONLY as a
   deprecated back-compat alias; new configs are built by
   `build_coverage_tolerance_config_from_models`, which emits the rel_frac that
   exactly reproduces k*rel_sigma so the magnitude-dependence is sourced from
   the noise model. The legacy absolute path stays available as a labeled
   compatibility shim (SFSScorer(legacy=True)).

Both helpers consume the SAME `inference_results.json` schema the rest of the
pipeline writes: a list of {"filename", "generated", "target", ...}. Ground
truth comes from parsing the `target` string with the same HybridClaimParser
the scorer uses, so GT is naturally restricted to parseable, scorable features
(no denominator inflation — see the recall note in src/sfs.py).
"""

from __future__ import annotations

import math
import random
import statistics as _st
from dataclasses import dataclass, field

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


# ═════════════════════════════════════════════════════════════════════════════
# FORMAL FRAMEWORK (T1-T5): coverage-guaranteed tolerances, observability
# ceiling, identifiability, bootstrap CIs + multiple-comparison correction.
#
# All of this is ADDITIVE: nothing here is invoked by the legacy SFS path, so
# the no-op contract of `SFSScorer(tol_config=None)` is untouched. The pieces:
#
#   normal_quantile / norm_cdf      Phi^{-1}, Phi (no scipy)
#   estimate_noise_model            sigma_f, bias, kurtosis from two GT sources
#                                   (Lemma 1.5: sigma = median|D| / 0.9539)
#   coverage_tolerance              T1: tau_f = JND_f + k(alpha)*sigma_f, with
#                                   gaussian / chebyshev / vp factor families
#   build_coverage_tolerance_config a ToleranceConfig DERIVED from {sigma,JND}
#   p_max_ceiling                   T3a: P_max = 2*Phi(tau/sigma) - 1
#   identifiability                 T3b: SCORABLE vs UNIDENTIFIABLE on this GT
#                                   (Fano channel-capacity threshold)
#   bootstrap_precision_ci          T4/T5: seeded, reproducible CI for precision
#   bootstrap_skill_ci              seeded CI + paired model-vs-baseline test
#   holm_correction / bh_correction multiple-comparison across features
# ═════════════════════════════════════════════════════════════════════════════

# Lemma 1.5 constant: for D = eps1 - eps2 (two independent same-sigma errors),
# median|D| = sqrt(2)*sigma*Phi^{-1}(0.75) = 0.95387...*sigma under Gaussianity.
_MEDIAN_ABS_TO_SIGMA = math.sqrt(2.0) * 0.6744897501960817  # 0.9538725...


def norm_cdf(z: float) -> float:
    """Standard normal CDF Phi(z) via erf (no scipy)."""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def normal_quantile(p: float) -> float:
    """Inverse standard normal CDF Phi^{-1}(p), p in (0,1).

    Acklam's rational approximation; abs error < ~1.15e-9 over (0,1). Used for
    the z_{1-alpha} tolerance factor and the Lemma-1.5 / T3 ceiling math so the
    framework has no scipy dependency (matching the rest of this module).
    """
    if not (0.0 < p < 1.0):
        raise ValueError(f"normal_quantile needs p in (0,1), got {p}")
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
                ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = p - 0.5
    r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
           (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


# ── Noise model (Lemma 1.5) ──────────────────────────────────────────────────
@dataclass
class NoiseModel:
    """Empirical measurement-error model for one feature: g_obs = g_true + eps.

    Estimated from two independent extractors on the SAME clips (clean-stem
    oracle vs mixture estimate). `d = g_a - g_b` is the per-clip disagreement.
    """

    feature: str
    n: int
    bias: float            # mean(d) — signed; eps zero-mean is REJECTED if != 0
    sigma_random: float    # std(d - bias) / sqrt(2)  (per-estimator sigma)
    sigma_total: float     # RMS(d) / sqrt(2)         (incl. bias contribution)
    robust_sigma: float    # 1.4826*MAD(d) / sqrt(2)  (heavy-tail-robust)
    median_abs_diff: float # median|d|  (the heuristic GT-noise floor)
    ex_kurtosis: float     # excess kurtosis of d  (>1 => heavy tailed)
    dynamic_range: float   # p2.5..p97.5 span of g_a (identifiability denom)
    rel_sigma: float = 0.0       # robust per-estimator sigma of the RELATIVE
                                 # disagreement d/|gt| (heteroscedastic scale).
                                 # 0.0 => homoscedastic (no magnitude growth).
    heteroscedastic: bool = False  # set when the data shows magnitude-growth of
                                   # the disagreement (corr(|d|,|gt|) above thr).

    def sigma(self, robust: bool = True) -> float:
        """Single sigma_f to feed the tolerance/ceiling math. Robust (MAD) by
        default because every paired SFS feature is heavy-tailed (octave F0
        errors), so the plain std overstates the Gaussian core."""
        return self.robust_sigma if robust else self.sigma_random

    def sigma_at(self, gt_value: float, robust: bool = True) -> float:
        """Magnitude-DEPENDENT sigma_f(|gt|) for a heteroscedastic feature.

        Tier-3 standardizes magnitude-dependence INTO sigma rather than a
        separate rel_frac knob: where the GT disagreement grows with the value
        (heteroscedastic), the per-estimator scale is

            sigma_f(|gt|) = max( sigma_homoscedastic , rel_sigma * |gt| )

        so the coverage band tau = JND + k*sigma_f(|gt|) widens with magnitude
        THROUGH sigma. A homoscedastic feature (rel_sigma == 0 or not flagged)
        returns the constant `sigma()` exactly, so nothing changes for those.
        This replaces the old ad-hoc `rel_frac * |gt|` term as the mechanism for
        heteroscedasticity (rel_frac is retained only as a deprecated no-op
        alias for back-compat)."""
        base = self.sigma(robust=robust)
        if not self.heteroscedastic or self.rel_sigma <= 0.0:
            return base
        return max(base, self.rel_sigma * abs(gt_value))


def _percentile(sorted_xs: list[float], p: float) -> float:
    if not sorted_xs:
        return float("nan")
    i = p / 100.0 * (len(sorted_xs) - 1)
    lo, hi = math.floor(i), math.ceil(i)
    if lo == hi:
        return sorted_xs[lo]
    return sorted_xs[lo] + (sorted_xs[hi] - sorted_xs[lo]) * (i - lo)


def estimate_noise_model(feature: str,
                         vals_a: list[float],
                         vals_b: list[float],
                         independent: bool = True,
                         hetero_corr_threshold: float = 0.2) -> NoiseModel:
    """Fit a NoiseModel for `feature` from two estimators' paired values.

    vals_a, vals_b are aligned per-clip (drop unpaired before calling). Under
    the additive model with two INDEPENDENT same-sigma estimators,
    Var(d) = 2*sigma^2 (Lemma 1.5), so per-estimator sigma = std(d)/sqrt(2) and
    median|d| = 0.9539*sigma. If `independent=False` (e.g. both read the same
    acoustic content, so the disagreement is dominated by a definitional bias
    not independent noise), the sqrt(2) split is skipped and sigma is read as
    the full std(d) — a conservative (larger) band.

    HETEROSCEDASTICITY (Tier-3): we also estimate a RELATIVE per-estimator scale
    `rel_sigma` = robust MAD of the per-clip relative disagreement d/|gt_ref|
    (gt_ref = mean of the two estimates), divided by the same sqrt(2) split. A
    feature is flagged `heteroscedastic` when Spearman corr(|d|, |gt_ref|)
    exceeds `hetero_corr_threshold` AND rel_sigma > 0. Downstream,
    `NoiseModel.sigma_at(|gt|)` then widens the band with magnitude THROUGH
    sigma (= max(const sigma, rel_sigma*|gt|)), which is how Tier-3 captures
    magnitude-dependence instead of the deprecated rel_frac term.

    Returns a NoiseModel with bias, sigma (random/total/robust), excess
    kurtosis, rel_sigma + heteroscedastic flag, and the dynamic range of
    `vals_a` for identifiability.
    """
    if len(vals_a) != len(vals_b):
        raise ValueError("vals_a and vals_b must be aligned (same length)")
    d = [a - b for a, b in zip(vals_a, vals_b)]
    n = len(d)
    if n == 0:
        return NoiseModel(feature, 0, 0.0, 0.0, 0.0, 0.0, 0.0, float("nan"), 0.0)
    bias = sum(d) / n
    if n > 1:
        var = sum((x - bias) ** 2 for x in d) / (n - 1)
        std = math.sqrt(var)
    else:
        std = 0.0
    rms = math.sqrt(sum(x * x for x in d) / n)
    med = _st.median(d)
    mad = _st.median([abs(x - med) for x in d]) * 1.4826
    median_abs = _st.median([abs(x) for x in d])
    # excess kurtosis
    if n > 1 and std > 0:
        m2 = sum((x - bias) ** 2 for x in d) / n
        ex_kurt = sum((x - bias) ** 4 for x in d) / n / (m2 * m2) - 3.0
    else:
        ex_kurt = float("nan")
    scale = math.sqrt(2.0) if independent else 1.0
    sa = sorted(vals_a)
    dyn = _percentile(sa, 97.5) - _percentile(sa, 2.5)

    # ── heteroscedastic (relative) scale ─────────────────────────────────────
    # reference magnitude per clip = mean of the two estimates (symmetric, and
    # never zero unless both are zero). relative disagreement r = d / |gt_ref|.
    gt_ref = [0.5 * (a + b) for a, b in zip(vals_a, vals_b)]
    rels = [di / abs(g) for di, g in zip(d, gt_ref) if abs(g) > 1e-9]
    if len(rels) > 1:
        rmed = _st.median(rels)
        rel_mad = _st.median([abs(x - rmed) for x in rels]) * 1.4826
        rel_sigma = rel_mad / scale
    else:
        rel_sigma = 0.0
    # heteroscedasticity test: does |d| grow with |gt_ref|? (Spearman, robust)
    abs_d = [abs(x) for x in d]
    abs_g = [abs(x) for x in gt_ref]
    sp = spearman(abs_d, abs_g)
    heteroscedastic = (sp is not None and sp > hetero_corr_threshold
                       and rel_sigma > 0.0)

    return NoiseModel(
        feature=feature, n=n, bias=bias,
        sigma_random=std / scale,
        sigma_total=rms / scale,
        robust_sigma=mad / scale,
        median_abs_diff=median_abs,
        ex_kurtosis=ex_kurt,
        dynamic_range=dyn,
        rel_sigma=rel_sigma,
        heteroscedastic=heteroscedastic,
    )


# ── T1: coverage-guaranteed tolerance ────────────────────────────────────────
def _tolerance_factor(alpha: float, family: str) -> float:
    """k in tau = JND + k*sigma, giving a TRUE two-sided false-reject <= alpha.

    The acceptance event is |delta - eps| <= tau where eps is the GT noise and
    |delta| <= JND for a faithful claim. The worst-case faithful claim sits at
    delta = +-JND, so the eps that pushes the OBSERVED error outside tau can
    arrive from EITHER tail. A one-sided z_{1-alpha} factor only bounds ONE
    tail at alpha and lets the opposite tail re-enter, so the true worst-case
    false-reject is up to 2*alpha (the documented T1 defect, now pinned FIXED in
    tests/test_formal_sfs.py::TestT1WorstCaseCoverageFixed). The fix is to
    split alpha across both tails:

    gaussian   z_{1-alpha/2} = Phi^{-1}(1-alpha/2)         (Thm 1.2a, fixed)
               -> worst-case false-reject <= alpha at EVERY faithful delta.
    chebyshev  1/sqrt(alpha)                               (Thm 1.3, dist-free)
    vp         (2/3)/sqrt(alpha)  for alpha <= 1/6          (Thm 1.4, unimodal)

    Chebyshev and VP are ALREADY two-sided (they bound P(|eps| > k*sigma)
    symmetrically), so they are UNCHANGED. Only the Gaussian family carried the
    one-sided bug; it now uses the alpha/2 quantile. Heavy-tailed features
    (excess kurtosis > 1 per estimate_noise_model) should select chebyshev / vp
    regardless, since the Gaussian quantile under-covers their tails.
    """
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0,1), got {alpha}")
    if family == "gaussian":
        # two-sided: split alpha across both tails -> z_{1-alpha/2}
        return normal_quantile(1.0 - alpha / 2.0)
    if family == "chebyshev":
        return 1.0 / math.sqrt(alpha)
    if family == "vp":
        if alpha > 1.0 / 6.0:
            raise ValueError("VP factor requires alpha <= 1/6")
        return (2.0 / 3.0) / math.sqrt(alpha)
    raise ValueError(f"unknown tolerance family {family!r}")


def coverage_tolerance(sigma_f: float, jnd_f: float, alpha: float = 0.05,
                       family: str = "gaussian") -> float:
    """T1 band tau_f = JND_f + k(alpha, family) * sigma_f.

    A truly faithful claim (|claim - truth| <= JND_f) is accepted with prob
    >= 1 - alpha against this band, two-sided (Thm 1.2a / 1.3 / 1.4). The
    Gaussian factor is z_{1-alpha/2} (two-sided), so the worst-case faithful
    claim at delta = +-JND is still accepted with prob >= 1 - alpha. The band
    is DERIVED from the measured noise scale, not hand-tuned.
    """
    return jnd_f + _tolerance_factor(alpha, family) * sigma_f


def select_family(ex_kurtosis: float, kurt_threshold: float = 1.0,
                  unimodal: bool = True) -> str:
    """Pick the distribution-free band a feature's measured tail demands.

    estimate_noise_model reports excess kurtosis of the GT disagreement. When
    it exceeds `kurt_threshold` (every paired SFS feature does, F0 worst at
    ~6.5) the Gaussian z-quantile under-covers the tail, so we fall back to a
    distribution-free factor: Vysochanskii-Petunin if the error law is unimodal
    (tighter, valid for any unimodal law), else Chebyshev (no shape assumption).
    A light-tailed feature (exKurt <= threshold) keeps the two-sided Gaussian
    band. nan kurtosis (degenerate / n<2) defaults to the conservative
    distribution-free choice.
    """
    if ex_kurtosis != ex_kurtosis:  # nan
        return "vp" if unimodal else "chebyshev"
    if ex_kurtosis > kurt_threshold:
        return "vp" if unimodal else "chebyshev"
    return "gaussian"


def _one_sided_factor(alpha: float, family: str) -> float:
    """The ONE-SIDED tail factor: gaussian z_{1-alpha} = Phi^{-1}(1-alpha).

    Distinct from `_tolerance_factor`, whose Gaussian arm is two-sided
    (z_{1-alpha/2}) because the coverage band must hold against worst-case
    faithful claims at delta=+-JND. The separation / Fano budget below is a sum
    of two genuinely ONE-SIDED guarantees (a false-reject tail at one boundary,
    a false-accept tail at the other), so it keeps the one-sided z_{1-alpha}.
    Chebyshev / VP are symmetric and so coincide for both factors.
    """
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0,1), got {alpha}")
    if family == "gaussian":
        return normal_quantile(1.0 - alpha)
    return _tolerance_factor(alpha, family)


def identifiability_budget(alpha: float, beta: float, family: str = "gaussian") -> float:
    """The factor (z_{1-alpha}+z_{1-beta}) [gaussian] in the separation
    condition Delta_f - JND_f >= budget * sigma_f (Thm 1.2b). Both guarantees
    (false-reject <= alpha, false-accept <= beta) hold iff sigma_f is below
    (Delta_f - JND_f)/budget. These are ONE-SIDED tails (one per boundary), so
    this uses the one-sided z_{1-alpha}, NOT the two-sided coverage factor."""
    return _one_sided_factor(alpha, family) + _one_sided_factor(beta, family)


def build_coverage_tolerance_config(
    sigma: dict[str, float],
    jnd: dict[str, float],
    alpha: float = 0.05,
    family: str = "gaussian",
    rel_frac: dict[str, float] | None = None,
):
    """DERIVE a ToleranceConfig whose abs_floor[f] = coverage_tolerance(...).

    This is the constructor the spec asks for: tolerances come out of
    {sigma_f, JND_f, alpha}, never hand-set. Returns a `sfs.ToleranceConfig`
    (abs_floor only by default; pass rel_frac to add a relative term on top).
    Features absent from `sigma` are simply omitted (they fall back to the
    legacy TOLERANCES floor inside the scorer).
    """
    try:  # package-relative
        from .sfs import ToleranceConfig
    except ImportError:  # flat
        from sfs import ToleranceConfig
    floors = {
        f: coverage_tolerance(sigma[f], jnd.get(f, 0.0), alpha, family)
        for f in sigma
    }
    return ToleranceConfig(abs_floor=floors, rel_frac=dict(rel_frac or {}))


def build_coverage_tolerance_config_from_models(
    models: dict,                       # feature -> NoiseModel
    jnd: dict[str, float],
    alpha: float = 0.05,
    family: str | None = None,          # None => per-feature select_family(tail)
    robust: bool = True,
    kurt_threshold: float = 1.0,
):
    """DERIVE a Tier-3 ToleranceConfig directly from measured NoiseModels.

    For each feature the band is tau_f(|gt|) = JND_f + k(alpha, fam_f)*sigma_f(|gt|):
      * fam_f is chosen per feature by `select_family(model.ex_kurtosis)` when
        `family is None` (heavy-tailed -> VP/Chebyshev, light -> Gaussian), so the
        distribution-free band is used exactly where the tail demands it.
      * magnitude-dependence is folded INTO sigma: when a model is
        `heteroscedastic`, sigma_f(|gt|) = max(sigma_const, rel_sigma*|gt|), so
        the ToleranceConfig gets abs_floor = JND + k*sigma_const AND
        rel_frac = k*rel_sigma. Then the scorer's
        max(abs_floor, rel_frac*|gt|) reproduces JND + k*max(sigma_const,
        rel_sigma*|gt|) in BOTH regimes (the JND is folded into abs_floor; for a
        heteroscedastic feature it sits inside the small-magnitude floor). For a
        homoscedastic feature rel_frac is 0 and the band is the constant floor.

    This is the Tier-3 standardization: magnitude-dependence comes from sigma
    (the relative noise scale), NOT a separately hand-set rel_frac.
    """
    try:  # package-relative
        from .sfs import ToleranceConfig
    except ImportError:  # flat
        from sfs import ToleranceConfig
    abs_floor: dict[str, float] = {}
    rel_frac: dict[str, float] = {}
    for f, m in models.items():
        if m is None:
            continue
        fam = family if family is not None else select_family(
            m.ex_kurtosis, kurt_threshold=kurt_threshold)
        k = _tolerance_factor(alpha, fam)
        sig_const = m.sigma(robust=robust)
        abs_floor[f] = jnd.get(f, 0.0) + k * sig_const
        if m.heteroscedastic and m.rel_sigma > 0.0:
            rel_frac[f] = k * m.rel_sigma
    return ToleranceConfig(abs_floor=abs_floor, rel_frac=rel_frac)


# ── T3a: observability precision ceiling ─────────────────────────────────────
def p_max_ceiling(sigma_f: float, tau_f: float) -> float:
    """T3a Gaussian ceiling: P_max = 2*Phi(tau/sigma) - 1.

    The MAX probability ANY estimator (incl. the oracle emitting g_true) has of
    being accepted against a noisy GT with this band. Reported precision can
    never exceed this; precision == P_max means the residual misses are entirely
    GT noise, not model error. sigma_f -> 0 gives 1.0 (noiseless GT)."""
    if sigma_f <= 0:
        return 1.0
    return 2.0 * norm_cdf(tau_f / sigma_f) - 1.0


# ── T3b: identifiability (Fano channel-capacity threshold) ───────────────────
@dataclass
class Identifiability:
    feature: str
    sigma_f: float
    jnd_f: float
    dynamic_range: float
    delta_f: float           # effect size used for the cell width
    channel_capacity: float  # C_f = 0.5*log2(1 + R^2/(12 sigma^2)) bits
    n_cells: int             # M_f = ceil(R / (2 Delta))
    log2_cells: float        # log2(M_f)
    fano_err_lb: float       # T3b min decode-error lower bound
    scorable: bool           # C_f >= log2(M_f) -> identifiable on this GT


def identifiability(feature: str, sigma_f: float, jnd_f: float,
                    dynamic_range: float, delta_f: float | None = None) -> Identifiability:
    """Classify a feature as SCORABLE vs UNIDENTIFIABLE-on-this-GT (T3b).

    Channel model: input g_true constrained to a range R=dynamic_range, additive
    noise variance sigma_f^2, capacity C_f = 0.5*log2(1 + R^2/(12 sigma^2)) bits
    (range-limited Gaussian channel, uniform-input worst case). Discretize R into
    M_f = ceil(R / 2*Delta) cells of half-width Delta (Delta defaults to
    max(JND, sigma) — the finest distinction the noise permits). Fano:
        P_err >= 1 - (C_f + 1)/log2(M_f).
    A feature is scorable iff C_f >= log2(M_f) (the latent cell is recoverable
    through the noisy GT). f0_mean/f0_sd under heavy overlap are the canonical
    UNIDENTIFIABLE instances (matches ABSTAINABLE_FEATURES)."""
    delta = delta_f if delta_f is not None else max(jnd_f, sigma_f)
    R = max(dynamic_range, 0.0)
    if sigma_f <= 0:
        cap = float("inf")
    else:
        cap = 0.5 * math.log2(1.0 + (R * R / 12.0) / (sigma_f * sigma_f))
    M = max(1, math.ceil(R / (2.0 * delta))) if delta > 0 else 1
    log2M = math.log2(M) if M > 1 else 0.0
    if log2M <= 0:
        fano_lb = 0.0
        scorable = True
    else:
        fano_lb = max(0.0, 1.0 - (cap + 1.0) / log2M)
        # Scorable iff Fano does NOT force a positive decode-error floor, i.e.
        # the channel can carry log2(M_f) bits up to the +1 Fano slack
        # (C_f + 1 >= log2 M_f). When that fails, fano_lb > 0 and the latent
        # cell is provably unrecoverable through the noisy GT (T3b).
        scorable = fano_lb <= 0.0
    return Identifiability(
        feature=feature, sigma_f=sigma_f, jnd_f=jnd_f,
        dynamic_range=R, delta_f=delta,
        channel_capacity=cap, n_cells=M, log2_cells=log2M,
        fano_err_lb=fano_lb, scorable=scorable,
    )


# ── T4 / T5: bootstrap CIs + paired significance + multiple-comparison ───────
def _precision_of(preds: list[float], gts: list[float],
                  feature: str, tol_fn) -> float:
    if not preds:
        return 0.0
    return sum(1 for p, g in zip(preds, gts)
               if abs(p - g) <= tol_fn(feature, g)) / len(preds)


def bootstrap_precision_ci(preds: list[float], gts: list[float], feature: str,
                           tol_fn, n_boot: int = 2000, alpha: float = 0.05,
                           seed: int = 0) -> dict:
    """Seeded, reproducible percentile-bootstrap CI for per-feature precision.

    tol_fn(feature, gt) -> tolerance (pass SFSScorer(...)._tolerance, or a
    closure over coverage_tolerance). Returns {point, lo, hi, n, n_boot, alpha}.
    Same seed => identical interval (verified in tests)."""
    n = len(preds)
    point = _precision_of(preds, gts, feature, tol_fn)
    if n == 0:
        return {"point": 0.0, "lo": 0.0, "hi": 0.0, "n": 0,
                "n_boot": n_boot, "alpha": alpha}
    rng = random.Random(seed)
    stats = []
    idx = range(n)
    for _ in range(n_boot):
        sample = [rng.randrange(n) for _ in idx]
        nc = sum(1 for i in sample
                 if abs(preds[i] - gts[i]) <= tol_fn(feature, gts[i]))
        stats.append(nc / n)
    stats.sort()
    lo = stats[max(0, int((alpha / 2) * n_boot))]
    hi = stats[min(n_boot - 1, int((1 - alpha / 2) * n_boot))]
    return {"point": point, "lo": lo, "hi": hi, "n": n,
            "n_boot": n_boot, "alpha": alpha}


def bootstrap_skill_ci(preds: list[float], gts: list[float], feature: str,
                       tol_fn, baseline_value: float,
                       n_boot: int = 2000, alpha: float = 0.05,
                       seed: int = 0) -> dict:
    """Seeded bootstrap CI for SKILL = model_precision - baseline_precision and a
    PAIRED test of skill > 0 (model beats the constant baseline).

    On each bootstrap resample BOTH the model and the constant baseline are
    re-scored on the IDENTICAL resampled clips (paired), so the skill CI removes
    the shared base-rate variance (T2/T5). `p_value` is the one-sided bootstrap
    probability that skill <= 0. Returns point/lo/hi/p_value/n."""
    n = len(preds)
    if n == 0:
        return {"point": 0.0, "lo": 0.0, "hi": 0.0, "p_value": 1.0,
                "n": 0, "n_boot": n_boot, "alpha": alpha}
    mp = _precision_of(preds, gts, feature, tol_fn)
    bp = sum(1 for g in gts
             if abs(baseline_value - g) <= tol_fn(feature, g)) / n
    point = mp - bp
    rng = random.Random(seed)
    stats = []
    n_le0 = 0
    for _ in range(n_boot):
        sample = [rng.randrange(n) for _ in range(n)]
        mc = bc = 0
        for i in sample:
            tol = tol_fn(feature, gts[i])
            if abs(preds[i] - gts[i]) <= tol:
                mc += 1
            if abs(baseline_value - gts[i]) <= tol:
                bc += 1
        s = (mc - bc) / n
        stats.append(s)
        if s <= 0:
            n_le0 += 1
    stats.sort()
    lo = stats[max(0, int((alpha / 2) * n_boot))]
    hi = stats[min(n_boot - 1, int((1 - alpha / 2) * n_boot))]
    return {"point": point, "lo": lo, "hi": hi,
            "p_value": n_le0 / n_boot, "n": n,
            "n_boot": n_boot, "alpha": alpha}


def holm_correction(pvals: dict[str, float], alpha: float = 0.05) -> dict:
    """Holm-Bonferroni step-down across features. Returns
    {feature: {"p": .., "p_adj": .., "reject": bool}}. Strong FWER control."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    out: dict[str, dict] = {}
    prev_adj = 0.0
    still_rejecting = True
    for rank, (f, p) in enumerate(items):
        adj = min(1.0, (m - rank) * p)
        adj = max(adj, prev_adj)  # enforce monotone non-decreasing adjusted p
        prev_adj = adj
        if not (still_rejecting and adj <= alpha):
            still_rejecting = False
        out[f] = {"p": p, "p_adj": adj, "reject": still_rejecting}
    return out


def bh_correction(pvals: dict[str, float], alpha: float = 0.05) -> dict:
    """Benjamini-Hochberg step-up (FDR control). Returns
    {feature: {"p": .., "p_adj": .., "reject": bool}}."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    out: dict[str, dict] = {}
    # adjusted p (monotone from the largest rank down)
    adj_sorted = [0.0] * m
    running_min = 1.0
    for rank in range(m - 1, -1, -1):
        f, p = items[rank]
        adj = min(1.0, p * m / (rank + 1))
        running_min = min(running_min, adj)
        adj_sorted[rank] = running_min
    # find largest k with p_(k) <= (k/m)*alpha
    max_reject_rank = -1
    for rank, (f, p) in enumerate(items):
        if p <= (rank + 1) / m * alpha:
            max_reject_rank = rank
    for rank, (f, p) in enumerate(items):
        out[f] = {"p": p, "p_adj": adj_sorted[rank],
                  "reject": rank <= max_reject_rank}
    return out


# ── T4: AURC selective-risk helpers + finite-sample Hoeffding half-width ──────
def aurc(confidences: list[float], losses: list[float]) -> float:
    """Area under the risk-coverage curve (T4a). Sort by DESCENDING confidence,
    accumulate the running mean loss over the most-confident k, average over
    coverage. losses are 0/1 (claim wrong=1). Lower AURC = better ordering."""
    n = len(confidences)
    if n == 0:
        return 0.0
    order = sorted(range(n), key=lambda i: confidences[i], reverse=True)
    run = 0.0
    total = 0.0
    for k, i in enumerate(order, start=1):
        run += losses[i]
        total += run / k
    return total / n


def hoeffding_halfwidth(n: int, delta: float = 0.05, n_curves: int = 2) -> float:
    """T4b leading term: 2*sqrt(log((2*n_curves)/delta)/(2n)). For the AURC
    gain over random (two curves), at n=18000, delta=0.05 this is ~0.022."""
    if n <= 0:
        return float("inf")
    return 2.0 * math.sqrt(math.log((2 * n_curves) / delta) / (2.0 * n))
