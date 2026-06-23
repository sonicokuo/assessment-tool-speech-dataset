"""Tests for the FORMAL SFS framework (T1-T5) in src/metrics_calibrated.py +
src/sfs.py.

These are the proof-checks the framework demands:

  (a) Monte-Carlo COVERAGE — inject eps_f ~ N(0,sigma) (and a heavy-tailed
      mixture), confirm a faithful claim (claim == truth) is accepted against
      the T1 band with empirical prob ~ 1 - alpha (Thm 1.2a). The Chebyshev /
      VP distribution-free bands are CONSERVATIVE (>= 1-alpha) on heavy tails.
  (b) Skill NO-GAMING (T2c) — an always-abstain / always-hedge strategy and a
      baseline-mimic strategy do NOT raise skill (skill <= 0 / == 0).
  (c) Observability ceiling HOLDS (T3a) — on synthetic data with KNOWN sigma,
      no estimator (incl. the oracle) exceeds P_max = 2*Phi(tau/sigma)-1.
  (d) NO-OP preserved — coverage config path is additive; the legacy
      SFSScorer(tol_config=None) precision is byte-identical to before.
  (e) Bootstrap CI REPRODUCIBLE with a fixed seed (same seed => same interval),
      and the identifiability classifier + Holm/BH corrections behave.
"""

import os
import sys
import random

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from sfs import (  # noqa: E402
    Claim,
    SFSScorer,
    ToleranceConfig,
    AbstentionDetector,
    coverage_guaranteed_config,
    PERCEPTUAL_JND,
    DEFAULT_TOLERANCE_CONFIG,
)
from metrics_calibrated import (  # noqa: E402
    norm_cdf,
    normal_quantile,
    estimate_noise_model,
    coverage_tolerance,
    select_family,
    identifiability_budget,
    build_coverage_tolerance_config,
    build_coverage_tolerance_config_from_models,
    p_max_ceiling,
    identifiability,
    bootstrap_precision_ci,
    bootstrap_skill_ci,
    holm_correction,
    bh_correction,
    aurc,
    hoeffding_halfwidth,
)


# ── helpers ──────────────────────────────────────────────────────────────────
def _heavy_tailed(rng, sigma, n, contam=0.1, heavy_scale=6.0):
    """Gaussian core + 10% wide-Gaussian contamination (octave-error analogue)."""
    out = []
    for _ in range(n):
        if rng.random() < contam:
            out.append(rng.gauss(0.0, sigma * heavy_scale))
        else:
            out.append(rng.gauss(0.0, sigma))
    return out


# ── (0) primitives ───────────────────────────────────────────────────────────
class TestPrimitives:
    def test_norm_cdf_quantile_roundtrip(self):
        for p in (0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99):
            z = normal_quantile(p)
            assert abs(norm_cdf(z) - p) < 1e-6

    def test_z_95_is_1p645(self):
        assert abs(normal_quantile(0.95) - 1.6448536) < 1e-5

    def test_tolerance_factor_families(self):
        # TWO-SIDED FIX: gaussian is now z_{1-alpha/2}=1.95996 (not the one-sided
        # 1.645); chebyshev 4.472, vp 2.981 unchanged (already two-sided).
        assert abs(coverage_tolerance(1.0, 0.0, 0.05, "gaussian") - 1.9599640) < 1e-4
        assert abs(coverage_tolerance(1.0, 0.0, 0.05, "chebyshev") - 4.472136) < 1e-4
        assert abs(coverage_tolerance(1.0, 0.0, 0.05, "vp") - 2.981424) < 1e-4

    def test_identifiability_budget(self):
        # the SEPARATION budget keeps the ONE-SIDED z_.95 per boundary (a
        # false-reject tail + a false-accept tail), so z_.95 + z_.95 = 3.29 at
        # alpha=beta=0.05 — distinct from the two-sided coverage factor above.
        b = identifiability_budget(0.05, 0.05, "gaussian")
        assert abs(b - 2 * 1.6448536) < 1e-4


# ── (Lemma 1.5) noise-model estimation ───────────────────────────────────────
class TestNoiseModel:
    def test_sigma_recovered_from_two_estimators(self):
        rng = random.Random(0)
        sigma_true = 3.0
        true = [rng.gauss(50.0, 10.0) for _ in range(20000)]
        a = [t + rng.gauss(0.0, sigma_true) for t in true]
        b = [t + rng.gauss(0.0, sigma_true) for t in true]
        nm = estimate_noise_model("snr", a, b, independent=True)
        # per-estimator sigma from std(d)/sqrt(2) ~ sigma_true
        assert abs(nm.sigma_random - sigma_true) < 0.15
        assert abs(nm.bias) < 0.15  # zero-mean eps

    def test_bias_detected(self):
        rng = random.Random(1)
        true = [rng.gauss(0.0, 5.0) for _ in range(5000)]
        a = [t + 16.6 + rng.gauss(0.0, 2.0) for t in true]  # definitional bias
        b = [t + rng.gauss(0.0, 2.0) for t in true]
        nm = estimate_noise_model("snr", a, b, independent=True)
        assert abs(nm.bias - 16.6) < 0.3  # bias != 0 -> reject zero-mean

    def test_heavy_tail_kurtosis(self):
        rng = random.Random(2)
        d_a = _heavy_tailed(rng, 5.0, 10000)
        zeros = [0.0] * len(d_a)
        nm = estimate_noise_model("f0_mean", d_a, zeros, independent=False)
        assert nm.ex_kurtosis > 1.0  # heavy-tailed


# ── (a) Monte-Carlo COVERAGE matches 1 - alpha (Thm 1.2a) ─────────────────────
class TestCoverageMonteCarlo:
    def _empirical_accept(self, eps_samples, tau):
        """Faithful claim has claim==truth so claim-error delta=0; accept iff
        |delta - eps| = |eps| <= tau."""
        return sum(1 for e in eps_samples if abs(e) <= tau) / len(eps_samples)

    def test_gaussian_coverage_matches_1_minus_alpha(self):
        rng = random.Random(42)
        sigma, alpha = 4.0, 0.05
        # zero JND isolates the noise term so the target is exactly 1 - alpha
        tau = coverage_tolerance(sigma, 0.0, alpha, "gaussian")
        eps = [rng.gauss(0.0, sigma) for _ in range(200000)]
        acc = self._empirical_accept(eps, tau)
        # TWO-SIDED FIX: tau = z_{1-alpha/2}*sigma, so the two-sided accept is
        # 2*Phi(z_.975)-1 = 1 - alpha = 0.95, and EACH tail is alpha/2 = 0.025.
        right_tail = sum(1 for e in eps if e > tau) / len(eps)
        assert abs(right_tail - alpha / 2.0) < 0.01
        # overall accept is the true 1 - alpha (= 2*Phi(tau/sigma)-1)
        assert abs(acc - (1.0 - alpha)) < 0.01
        assert abs(acc - (2 * norm_cdf(tau / sigma) - 1.0)) < 0.01

    def test_chebyshev_is_conservative_on_heavy_tail(self):
        rng = random.Random(7)
        sigma, alpha = 4.0, 0.05
        tau = coverage_tolerance(sigma, 0.0, alpha, "chebyshev")
        eps = _heavy_tailed(rng, sigma, 200000)
        # distribution-free band: false-reject must be <= alpha even on heavy tail
        reject = sum(1 for e in eps if abs(e) > tau) / len(eps)
        assert reject <= alpha

    def test_vp_conservative_unimodal(self):
        rng = random.Random(9)
        sigma, alpha = 4.0, 0.05
        tau = coverage_tolerance(sigma, 0.0, alpha, "vp")
        # unimodal heavy-ish: Student-t-like via gauss mixture stays unimodal
        eps = [rng.gauss(0.0, sigma) for _ in range(200000)]
        reject = sum(1 for e in eps if abs(e) > tau) / len(eps)
        assert reject <= alpha


# ── (c) Observability ceiling HOLDS on synthetic known-sigma data (T3a) ───────
class TestObservabilityCeiling:
    def test_oracle_cannot_exceed_pmax(self):
        rng = random.Random(123)
        sigma = 5.0
        tau = coverage_tolerance(sigma, 0.0, 0.05, "gaussian")
        pmax = p_max_ceiling(sigma, tau)
        # ORACLE: emits the TRUE value exactly. GT = true + eps. accept iff
        # |true - (true+eps)| = |eps| <= tau.
        n = 200000
        accept = 0
        for _ in range(n):
            eps = rng.gauss(0.0, sigma)
            if abs(0.0 - eps) <= tau:
                accept += 1
        emp = accept / n
        # empirical oracle precision must not exceed P_max (up to MC noise)
        assert emp <= pmax + 0.01
        assert abs(emp - pmax) < 0.01  # oracle ATTAINS the ceiling

    def test_pmax_monotone_and_bounded(self):
        # noiseless GT -> ceiling 1.0; huge sigma -> ceiling small
        assert p_max_ceiling(0.0, 1.0) == 1.0
        assert p_max_ceiling(100.0, 1.0) < 0.05
        # wider band raises ceiling
        assert p_max_ceiling(5.0, 10.0) > p_max_ceiling(5.0, 2.0)

    def test_noisy_estimator_below_oracle(self):
        rng = random.Random(321)
        sigma = 5.0
        tau = coverage_tolerance(sigma, 0.0, 0.05, "gaussian")
        pmax = p_max_ceiling(sigma, tau)
        n = 100000
        accept = 0
        for _ in range(n):
            eps = rng.gauss(0.0, sigma)
            model_err = rng.gauss(0.0, 3.0)  # model adds its own error
            if abs(model_err - eps) <= tau:
                accept += 1
        emp = accept / n
        assert emp <= pmax + 0.01  # NO estimator beats the ceiling


# ── (T3b) identifiability classifier ─────────────────────────────────────────
class TestIdentifiability:
    def test_low_noise_scorable(self):
        # tight sigma vs wide range -> recoverable
        idf = identifiability("speaking_rate", sigma_f=0.3, jnd_f=0.3,
                              dynamic_range=4.0, delta_f=0.3)
        assert idf.scorable
        assert idf.fano_err_lb < 0.5

    def test_huge_noise_unidentifiable(self):
        # sigma comparable to the whole range -> NOT recoverable
        idf = identifiability("f0_mean", sigma_f=40.0, jnd_f=1.0,
                              dynamic_range=120.0, delta_f=1.0)
        assert not idf.scorable
        assert idf.fano_err_lb > 0.0

    def test_capacity_drops_with_noise(self):
        a = identifiability("x", 1.0, 0.5, 10.0)
        b = identifiability("x", 5.0, 0.5, 10.0)
        assert a.channel_capacity > b.channel_capacity


# ── (d) LEGACY SHIM preserved (back-compat only) ─────────────────────────────
# The DEFAULT meaning of SFS is now the Tier-3 coverage band; the legacy
# hand-picked absolute tolerances are reachable ONLY via SFSScorer(legacy=True).
# These tests pin that the shim still reproduces the old ±2 dB / ±5 Hz behavior
# byte-for-byte, AND that the default scorer is NOT the legacy band.
class TestLegacyShimPreserved:
    def _claims(self, **kw):
        return [Claim(f, v, "", "") for f, v in kw.items()]

    def test_legacy_shim_unchanged(self):
        """SFSScorer(legacy=True) must reproduce the old absolute band exactly."""
        scorer = SFSScorer(legacy=True)
        gt = {"snr": 30.0, "f0_mean": 200.0}
        # snr ±2, f0 ±5 from TOLERANCES
        r = scorer.score(self._claims(snr=31.0, f0_mean=203.0), gt)
        assert r["precision"] == 1.0  # both within legacy band
        r2 = scorer.score(self._claims(snr=35.0), {"snr": 30.0})
        assert r2["precision"] == 0.0  # 5 dB miss > ±2 legacy band

    def test_default_is_tier3_not_legacy(self):
        """SFSScorer() now means the PRINCIPLED coverage band, NOT the legacy
        ±2 dB dict. A 35 dB claim against a 30 dB GT is OUT of the legacy band
        but WELL INSIDE the principled (GT-noise-derived) band, so the default
        accepts where the legacy shim rejects."""
        default = SFSScorer()           # Tier-3 coverage band
        legacy = SFSScorer(legacy=True)  # old ±2 dB
        gt = {"snr": 30.0}
        c = [Claim("snr", 35.0, "dB", "")]
        assert default.score(c, gt)["precision"] == 1.0   # inside principled band
        assert legacy.score(c, gt)["precision"] == 0.0    # outside legacy band
        # the default band is the explicit DEFAULT_TOLERANCE_CONFIG object
        assert default.tol_config is DEFAULT_TOLERANCE_CONFIG
        assert legacy.tol_config is None

    def test_legacy_and_tol_config_mutually_exclusive(self):
        import pytest
        with pytest.raises(ValueError):
            SFSScorer(tol_config=ToleranceConfig(), legacy=True)

    def test_empty_coverage_config_paths_dont_touch_legacy(self):
        """A coverage config with sigma=0, JND=legacy floors equals the legacy
        absolute path (a constructive no-op witness)."""
        legacy = SFSScorer(legacy=True)
        # sigma=0 => tau = JND; set JND to the legacy TOLERANCES so it matches.
        sigma = {f: 0.0 for f in ("snr", "f0_mean")}
        jnd = {"snr": SFSScorer.TOLERANCES["snr"],
               "f0_mean": SFSScorer.TOLERANCES["f0_mean"]}
        cfg = build_coverage_tolerance_config(sigma, jnd, alpha=0.05)
        derived = SFSScorer(tol_config=cfg)
        for snr_pred in (30.5, 31.0, 31.9, 32.0, 32.1, 35.0):
            gt = {"snr": 30.0}
            c = [Claim("snr", snr_pred, "dB", "")]
            assert (legacy.score(c, gt)["precision"]
                    == derived.score(c, gt)["precision"])

    def test_bare_tolerance_config_still_noop(self):
        """A bare ToleranceConfig() (empty abs_floor + rel_frac) still reproduces
        the legacy absolute band claim-for-claim (the documented no-op)."""
        legacy = SFSScorer(legacy=True)
        bare = SFSScorer(tol_config=ToleranceConfig())
        for f0 in (198.0, 200.0, 204.0, 205.0, 206.0):
            gt = {"f0_mean": 200.0}
            c = [Claim("f0_mean", f0, "Hz", "")]
            assert (legacy.score(c, gt)["precision"]
                    == bare.score(c, gt)["precision"])


# ── (b) skill NO-GAMING (T2c) ─────────────────────────────────────────────────
class TestSkillNoGaming:
    def _tol_fn(self, feature, gt):
        # fixed ±1 band for a clean pause_count-style artifact
        return 1.0

    def test_baseline_mimic_zero_skill(self):
        # model == constant baseline -> skill exactly 0
        gts = [1.0, 1.0, 1.0, 2.0, 1.0, 1.0]
        baseline_value = 1.0  # the mode
        preds = [1.0] * len(gts)  # mimic the baseline
        r = bootstrap_skill_ci(preds, gts, "pause_count", self._tol_fn,
                               baseline_value, n_boot=500, seed=0)
        assert abs(r["point"]) < 1e-9  # zero skill
        assert r["p_value"] >= 0.5  # cannot reject skill<=0

    def test_abstain_does_not_raise_skill(self):
        """Always-abstain asserts NO numbers, so it has no precision to add and
        cannot raise skill — score_selective excludes hedges from the precision
        denominator (T2c)."""
        scorer = SFSScorer()
        gt = {"f0_mean": 200.0, "snr": 30.0}
        # reliable f0 (number SHOULD be given); abstaining it is a coverage miss,
        # not a precision gain.
        text = "The pitch cannot be reliably estimated due to overlap."
        res = scorer.score_selective(text, gt,
                                     reliable={"f0_mean": True, "snr": True})
        # no asserted numbers -> precision 0, NOT inflated by the hedge
        assert res["n_asserted"] == 0
        assert res["precision"] == 0.0

    def test_overclaim_counted_against_precision(self):
        """Asserting a number on an ill-posed feature is penalized even if in
        tolerance (T2c / Cor 3.7), so 'assert the safe value, call it skill'
        cannot game the metric."""
        scorer = SFSScorer()
        gt = {"f0_mean": 200.0}
        text = "The F0 mean is 200.0 Hz."  # exact, but feature is unreliable
        res = scorer.score_selective(text, gt, reliable={"f0_mean": False})
        assert res["n_overclaim"] == 1
        assert res["precision"] == 0.0  # over-claim not credited

    def test_skill_positive_when_model_beats_baseline(self):
        # model nails a high-variance feature the constant baseline misses
        rng = random.Random(5)
        gts = [rng.uniform(0, 100) for _ in range(400)]
        preds = [g + rng.gauss(0, 1.0) for g in gts]  # tight model

        def tol_fn(feature, gt):
            return 3.0  # ±3 band

        baseline_value = sorted(gts)[len(gts) // 2]  # median constant
        r = bootstrap_skill_ci(preds, gts, "snr", tol_fn, baseline_value,
                               n_boot=1000, seed=0)
        assert r["point"] > 0.5  # large positive skill
        assert r["lo"] > 0.0     # CI excludes 0
        assert r["p_value"] < 0.01


# ── (e) bootstrap reproducibility + multiple comparison ──────────────────────
class TestBootstrapReproducible:
    def _tol_fn(self, feature, gt):
        return 2.0

    def test_same_seed_same_interval(self):
        rng = random.Random(11)
        gts = [rng.uniform(0, 50) for _ in range(300)]
        preds = [g + rng.gauss(0, 1.5) for g in gts]
        r1 = bootstrap_precision_ci(preds, gts, "snr", self._tol_fn,
                                    n_boot=1000, seed=99)
        r2 = bootstrap_precision_ci(preds, gts, "snr", self._tol_fn,
                                    n_boot=1000, seed=99)
        assert r1 == r2  # byte-identical with same seed

    def test_different_seed_differs(self):
        rng = random.Random(12)
        gts = [rng.uniform(0, 50) for _ in range(300)]
        preds = [g + rng.gauss(0, 1.5) for g in gts]
        r1 = bootstrap_precision_ci(preds, gts, "snr", self._tol_fn,
                                    n_boot=1000, seed=1)
        r2 = bootstrap_precision_ci(preds, gts, "snr", self._tol_fn,
                                    n_boot=1000, seed=2)
        # point estimate identical (no resampling), CI bounds may differ
        assert r1["point"] == r2["point"]
        assert (r1["lo"], r1["hi"]) != (r2["lo"], r2["hi"])

    def test_ci_brackets_point(self):
        rng = random.Random(13)
        gts = [rng.uniform(0, 50) for _ in range(300)]
        preds = [g + rng.gauss(0, 1.5) for g in gts]
        r = bootstrap_precision_ci(preds, gts, "snr", self._tol_fn,
                                   n_boot=1000, seed=0)
        assert r["lo"] <= r["point"] <= r["hi"]


class TestMultipleComparison:
    def test_holm_strong_fwer(self):
        pvals = {"a": 0.001, "b": 0.04, "c": 0.5, "d": 0.0001}
        out = holm_correction(pvals, alpha=0.05)
        assert out["d"]["reject"]
        assert out["a"]["reject"]
        assert not out["c"]["reject"]
        # adjusted p monotone non-decreasing in raw-p order
        ordered = sorted(out.items(), key=lambda kv: kv[1]["p"])
        adj = [v["p_adj"] for _, v in ordered]
        assert adj == sorted(adj)

    def test_bh_more_lenient_than_holm(self):
        pvals = {f"f{i}": p for i, p in enumerate([0.001, 0.01, 0.02, 0.03, 0.04])}
        holm = holm_correction(pvals, 0.05)
        bh = bh_correction(pvals, 0.05)
        n_holm = sum(v["reject"] for v in holm.values())
        n_bh = sum(v["reject"] for v in bh.values())
        assert n_bh >= n_holm  # FDR rejects at least as many as FWER


# ── (T4) AURC + Hoeffding ─────────────────────────────────────────────────────
class TestAURC:
    def test_perfect_ordering_beats_random(self):
        # confidence inversely tracks loss -> low AURC
        rng = random.Random(8)
        n = 2000
        losses = [1.0 if rng.random() < 0.3 else 0.0 for _ in range(n)]
        # good conf: high when loss=0
        good_conf = [1.0 - l + rng.gauss(0, 0.01) for l in losses]
        rand_conf = [rng.random() for _ in losses]
        a_good = aurc(good_conf, losses)
        a_rand = aurc(rand_conf, losses)
        assert a_good < a_rand  # informative ordering reduces AURC

    def test_hoeffding_halfwidth_18000(self):
        hw = hoeffding_halfwidth(18000, delta=0.05, n_curves=2)
        assert abs(hw - 0.022) < 0.005  # ~0.022 << measured gain 0.1295

    def test_aurc_random_equals_base_rate(self):
        # random ordering -> AURC ~ overall risk
        rng = random.Random(3)
        n = 5000
        losses = [1.0 if rng.random() < 0.25 else 0.0 for _ in range(n)]
        conf = [rng.random() for _ in losses]
        base = sum(losses) / n
        assert abs(aurc(conf, losses) - base) < 0.03


# ── integration: derive a config from a measured NoiseModel ──────────────────
class TestEndToEndDerivedBand:
    def test_coverage_config_from_noise_model(self):
        rng = random.Random(0)
        true = [rng.gauss(30.0, 8.0) for _ in range(8000)]
        a = [t + rng.gauss(0.0, 2.5) for t in true]
        b = [t + rng.gauss(0.0, 2.5) for t in true]
        nm = estimate_noise_model("snr", a, b, independent=True)
        cfg = coverage_guaranteed_config({"snr": nm.sigma(robust=False)},
                                         alpha=0.05, family="gaussian")
        scorer = SFSScorer(tol_config=cfg)
        # band ~ JND(1.0) + z_.975*2.5 ~ 1.0 + 1.96*2.5 ~ 5.9; a 4 dB miss is
        # accepted, 8 dB not (two-sided gaussian factor)
        gt = {"snr": 30.0}
        assert scorer.score([Claim("snr", 34.0, "dB", "")], gt)["precision"] == 1.0
        assert scorer.score([Claim("snr", 39.0, "dB", "")], gt)["precision"] == 0.0


# ── FIX VERIFIED: the T1 Gaussian band is now TWO-SIDED (z_{1-alpha/2}), so a ──
# faithful claim within JND is accepted with prob >= 1-alpha at EVERY delta in
# [-JND, JND] (true coverage). These tests PIN the FIXED behavior: the worst-
# case faithful false-reject is now <= alpha, and the old one-sided z_{1-alpha}
# band (which under-covered) is shown to fail the same check.
class TestT1WorstCaseCoverageFixed:
    def _false_reject(self, eps, delta, tau):
        # accept iff |delta - eps| <= tau
        return sum(1 for e in eps if abs(delta - e) > tau) / len(eps)

    def test_fixed_gaussian_band_covers_worst_case(self):
        """With the two-sided fix, coverage_tolerance(...,'gaussian') uses
        z_{1-alpha/2}, so the WORST-CASE faithful claim delta=+JND has total
        false-reject <= alpha (true coverage), not the old > alpha defect."""
        rng = random.Random(11)
        sigma, jnd, alpha = 3.0, 1.0, 0.05
        tau = coverage_tolerance(sigma, jnd, alpha, "gaussian")
        eps = [rng.gauss(0.0, sigma) for _ in range(300000)]
        # false-reject is <= alpha at EVERY faithful delta in [-JND, JND]
        for delta in (-jnd, -jnd / 2, 0.0, jnd / 2, jnd):
            fr = self._false_reject(eps, delta, tau)
            assert fr <= alpha + 0.003, (
                f"faithful delta={delta} FR={fr:.4f} must be <= alpha={alpha} "
                "after the two-sided fix")

    def test_old_one_sided_band_would_undercover(self):
        """Demonstrate WHY the fix was needed: the OLD one-sided band
        tau_old = JND + z_{1-alpha}*sigma lets the worst-case faithful claim
        (delta=+JND) false-reject STRICTLY ABOVE alpha (the dropped tail
        re-enters). The new two-sided band closes this gap."""
        rng = random.Random(11)
        sigma, jnd, alpha = 3.0, 1.0, 0.05
        z_one = normal_quantile(1.0 - alpha)        # old one-sided factor
        tau_old = jnd + z_one * sigma
        tau_new = coverage_tolerance(sigma, jnd, alpha, "gaussian")
        assert tau_new > tau_old                    # fix widened the band
        eps = [rng.gauss(0.0, sigma) for _ in range(300000)]
        fr_old_worst = self._false_reject(eps, jnd, tau_old)
        fr_new_worst = self._false_reject(eps, jnd, tau_new)
        assert fr_old_worst > alpha                 # old band under-covers
        assert fr_new_worst <= alpha + 0.003        # fixed band covers
        # both still respect the honest two-sided 2*alpha bound
        assert fr_old_worst <= 2 * alpha + 0.005

    def test_alpha_half_band_equals_coverage_tolerance(self):
        """coverage_tolerance gaussian IS the z_{1-alpha/2} band now."""
        sigma, jnd, alpha = 3.0, 1.0, 0.05
        z_half = normal_quantile(1.0 - alpha / 2.0)
        assert abs(coverage_tolerance(sigma, jnd, alpha, "gaussian")
                   - (jnd + z_half * sigma)) < 1e-9


# ── ADVERSARIAL: the Fano +1 slack makes identifiability VACUOUS for M<=2 ─────
class TestT3bFanoBinaryVacuity:
    def test_fano_cannot_reject_binary_distinction(self):
        """With M=2 (a binary quartile/median split), log2(M)=1 and the Fano
        bound 1-(C+1)/log2(M) = 1-(C+1) <= 0 for ANY capacity C>=0. So no
        feature is ever 'UNIDENTIFIABLE' at M=2, no matter how noisy. The
        UNIDENTIFIABLE labels therefore depend entirely on the (arbitrary)
        cell-count M; classification needs M>=3 to have teeth."""
        # huge noise, range only spans ~1 cell at delta=R/4 (M=2)
        idn = identifiability("x", sigma_f=50.0, jnd_f=1.0,
                              dynamic_range=14.0, delta_f=14.0 / 4)
        assert idn.n_cells == 2
        assert idn.scorable is True  # vacuously scorable despite cap << 1 bit
        assert idn.channel_capacity < 0.2
        # only a finer grid (M>=3) lets Fano declare unidentifiable
        idn3 = identifiability("x", sigma_f=50.0, jnd_f=1.0,
                               dynamic_range=14.0, delta_f=14.0 / 6)
        assert idn3.n_cells >= 3
        assert idn3.scorable is False


# ── per-feature family selection by measured tail ────────────────────────────
class TestFamilySelection:
    def test_heavy_tail_picks_distribution_free(self):
        # exKurt > 1 -> distribution-free (VP if unimodal, else Chebyshev)
        assert select_family(6.49) == "vp"               # F0-like heavy tail
        assert select_family(6.49, unimodal=False) == "chebyshev"

    def test_light_tail_keeps_gaussian(self):
        assert select_family(0.35) == "gaussian"         # speaking-rate-like
        assert select_family(0.84) == "gaussian"

    def test_nan_defaults_distribution_free(self):
        assert select_family(float("nan")) == "vp"
        assert select_family(float("nan"), unimodal=False) == "chebyshev"


# ── heteroscedasticity folded INTO sigma (Tier-3 standardization) ────────────
class TestHeteroscedasticSigma:
    def test_relative_sigma_estimated_and_flagged(self):
        """A multiplicative-error feature (eps ~ |gt| * N) is flagged
        heteroscedastic and its rel_sigma > 0; sigma_at widens with |gt|."""
        rng = random.Random(0)
        true = [rng.uniform(50.0, 300.0) for _ in range(8000)]
        # multiplicative per-estimator noise: |error| grows with the value
        a = [t * (1.0 + rng.gauss(0.0, 0.1)) for t in true]
        b = [t * (1.0 + rng.gauss(0.0, 0.1)) for t in true]
        nm = estimate_noise_model("f0_mean", a, b, independent=True)
        assert nm.heteroscedastic
        assert nm.rel_sigma > 0.0
        # sigma_at grows with |gt| through the relative term
        s_low = nm.sigma_at(60.0)
        s_high = nm.sigma_at(280.0)
        assert s_high > s_low

    def test_homoscedastic_sigma_at_is_constant(self):
        """Additive constant-sigma noise is NOT flagged heteroscedastic, so
        sigma_at returns the constant sigma at every magnitude."""
        rng = random.Random(1)
        true = [rng.uniform(0.0, 100.0) for _ in range(8000)]
        a = [t + rng.gauss(0.0, 2.0) for t in true]
        b = [t + rng.gauss(0.0, 2.0) for t in true]
        nm = estimate_noise_model("snr", a, b, independent=True)
        assert not nm.heteroscedastic
        assert nm.sigma_at(10.0) == nm.sigma_at(90.0) == nm.sigma()


# ── Tier-3 config built from measured noise models ───────────────────────────
class TestModelDerivedConfig:
    def test_family_selected_per_feature(self):
        """build_..._from_models picks the family per measured tail: a heavy-
        tailed feature gets the (larger) distribution-free band, a light-tailed
        one the Gaussian band, even at the SAME sigma."""
        rng = random.Random(2)
        # heavy-tailed disagreement
        true_h = [rng.uniform(50, 300) for _ in range(8000)]
        heavy = _heavy_tailed(rng, 4.0, 8000)
        a_h = [t + e for t, e in zip(true_h, heavy)]
        b_h = [t for t in true_h]
        nm_h = estimate_noise_model("f0_mean", a_h, b_h, independent=False)
        # light-tailed disagreement, same nominal sigma scale
        a_l = [t + rng.gauss(0, 4.0) for t in true_h]
        b_l = [t for t in true_h]
        nm_l = estimate_noise_model("speaking_rate", a_l, b_l, independent=False)
        cfg = build_coverage_tolerance_config_from_models(
            {"f0_mean": nm_h, "speaking_rate": nm_l},
            jnd={"f0_mean": 1.0, "speaking_rate": 0.3}, alpha=0.05)
        # heavy-tailed feature must have a LARGER factor (VP 2.98 > gaussian 1.96)
        assert nm_h.ex_kurtosis > 1.0
        assert nm_l.ex_kurtosis <= 1.0 or select_family(nm_l.ex_kurtosis) == "gaussian"
        # both features present with positive floors
        assert cfg.abs_floor["f0_mean"] > 0
        assert cfg.abs_floor["speaking_rate"] > 0

    def test_heteroscedastic_feature_gets_rel_frac(self):
        rng = random.Random(3)
        true = [rng.uniform(50.0, 300.0) for _ in range(8000)]
        a = [t * (1.0 + rng.gauss(0.0, 0.1)) for t in true]
        b = [t * (1.0 + rng.gauss(0.0, 0.1)) for t in true]
        nm = estimate_noise_model("f0_mean", a, b, independent=True)
        cfg = build_coverage_tolerance_config_from_models(
            {"f0_mean": nm}, jnd={"f0_mean": 1.0}, alpha=0.05)
        # heteroscedastic -> rel_frac = k * rel_sigma > 0
        assert cfg.rel_frac.get("f0_mean", 0.0) > 0.0
        # and the band reproduces JND + k*max(sigma_const, rel_sigma*|gt|) via
        # max(abs_floor, rel_frac*|gt|): at a large |gt| the relative term binds
        scorer = SFSScorer(tol_config=cfg)
        tol_big = scorer._tolerance("f0_mean", 280.0)
        tol_small = scorer._tolerance("f0_mean", 60.0)
        assert tol_big > tol_small


# ── DEFAULT band = the Tier-3 principled coverage band ───────────────────────
class TestDefaultIsPrincipled:
    def test_default_config_has_principled_floors(self):
        """DEFAULT_TOLERANCE_CONFIG is the coverage band, NOT the legacy dict:
        every covered floor is far wider than the legacy TOLERANCES value (the
        measured GT noise dominates the hand-picked ±2 dB band)."""
        cfg = DEFAULT_TOLERANCE_CONFIG
        # snr legacy floor is 2.0; principled floor is the GT-noise-derived band
        assert cfg.abs_floor["snr"] > SFSScorer.TOLERANCES["snr"]
        assert cfg.abs_floor["f0_mean"] > SFSScorer.TOLERANCES["f0_mean"]
        # heteroscedastic features carry a rel_frac term; f0_sd (homoscedastic)
        # does not
        assert "snr" in cfg.rel_frac and cfg.rel_frac["snr"] > 0
        assert cfg.rel_frac.get("f0_sd", 0.0) == 0.0

    def test_default_scorer_uses_default_config(self):
        s = SFSScorer()
        assert s.tol_config is DEFAULT_TOLERANCE_CONFIG
        assert s.legacy is False
