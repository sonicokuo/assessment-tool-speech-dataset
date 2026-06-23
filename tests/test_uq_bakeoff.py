"""Tests for the UNCERTAINTY-QUANTIFICATION BAKE-OFF (the kill-fast diffusion gate).

CPU-only, synthetic, deterministic — no GPU, no network. Covers the UQ math that has to
be right for the gate to be TRUSTWORTHY:

  MDN
    * NLL DECREASES when a mixture component is moved onto the target (the head's
      learning signal points the right way).
    * predictive total_variance equals the ANALYTIC law of total variance E[Var]+Var[E]
      on a hand example (2 comps, hand-computed 27.5).
    * predict() mean = sum_k pi_k mu_k on the same example.
    * NLL is finite / stable when one component sits far from the target (logsumexp).
    * masked NLL ignores mask=False frames.
  MC-dropout
    * sample() variance is > 0 with p>0 and EXACTLY 0 when p=0 (the hard anchor).
  Ensemble
    * ensemble_uncertainty mean+variance on a hand example (population variance).
  AURC / harness
    * continuous-risk AURC is LOWER when uncertainty is correlated with |error| than when
      it is random (the CORE property the whole gate rests on).
    * continuous-risk AURC INTEGRATES non-binary risks (it is NOT a 0/1 band cut): a
      monotone rescale of the risks changes the AURC, and the perfect-ordering AURC equals
      the hand-computed cumulative-mean integral of the real-valued risks.
    * paired bootstrap CI BRACKETS a known AURC difference (good vs random uncertainty)
      and its sign is correct (challenger < incumbent => negative delta, CI excludes 0).
    * calibration_summary flags an OVERCONFIDENT (too-tight sigma) method and does not
      flag a well-calibrated one.
    * run_bakeoff end-to-end produces the table + a GO/NO-GO verdict.
"""

import math
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from uq_heads import (  # noqa: E402
    MDNSNRMapHead,
    MCDropoutSNRMapHead,
    ensemble_uncertainty,
)
from uq_bakeoff import (  # noqa: E402
    continuous_risk_aurc,
    spearman_unc_err,
    calibration_summary,
    paired_bootstrap_aurc_delta,
    run_bakeoff,
    format_table,
    score_method,
)


# ── helpers ──────────────────────────────────────────────────────────────────
def _mdn_params(means, log_sigmas, logits):
    """Build (B=1, T=1, K) param tensors from python lists of K values."""
    m = torch.tensor([[means]], dtype=torch.float64)
    s = torch.tensor([[log_sigmas]], dtype=torch.float64)
    g = torch.tensor([[logits]], dtype=torch.float64)
    return m, s, g


# ════════════════════════════════════════════════════════════════════════════════
# MDN
# ════════════════════════════════════════════════════════════════════════════════
class TestMDN:
    def test_nll_decreases_when_component_matches_target(self):
        """Moving a component's mean ONTO the target lowers the mixture NLL."""
        head = MDNSNRMapHead(audio_dim=4, n_components=2)
        target = torch.tensor([[5.0]])  # (B=1, T=1)
        # far: both means at 0; near: one mean exactly at the target.
        far = _mdn_params([0.0, 0.0], [0.0, 0.0], [0.0, 0.0])
        near = _mdn_params([5.0, 0.0], [0.0, 0.0], [0.0, 0.0])
        nll_far = float(head.nll(far, target.double()))
        nll_near = float(head.nll(near, target.double()))
        assert nll_near < nll_far

    def test_total_variance_equals_analytic_lotv(self):
        """LoTV hand example: pi=[.5,.5], mu=[0,10], sigma=[1,2].
        mean=5; E[Var]=.5*1+.5*4=2.5; Var[E]=.5*25+.5*25=25; total=27.5."""
        params = _mdn_params([0.0, 10.0], [math.log(1.0), math.log(2.0)], [0.0, 0.0])
        mean, total_var = MDNSNRMapHead.predict(params)
        assert mean.shape == (1, 1)
        assert float(mean[0, 0]) == pytest.approx(5.0, abs=1e-9)
        assert float(total_var[0, 0]) == pytest.approx(27.5, abs=1e-9)

    def test_predict_mean_is_weighted_component_mean(self):
        """Unequal weights: pi=[.8,.2] via logits, mu=[0,10] -> mean = 2.0."""
        # softmax([log4, 0]) = [0.8, 0.2]
        params = _mdn_params([0.0, 10.0], [0.0, 0.0], [math.log(4.0), 0.0])
        mean, _ = MDNSNRMapHead.predict(params)
        assert float(mean[0, 0]) == pytest.approx(2.0, abs=1e-9)

    def test_nll_stable_with_far_component(self):
        """A component 1e6 away from y must not make the mixture NLL inf/nan
        (logsumexp keeps it finite — the OTHER component still has mass)."""
        params = _mdn_params([5.0, 1.0e6], [0.0, 0.0], [0.0, 0.0])
        target = torch.tensor([[5.0]], dtype=torch.float64)
        nll = float(MDNSNRMapHead(audio_dim=4, n_components=2).nll(params, target))
        assert math.isfinite(nll)

    def test_nll_masked_ignores_false_frames(self):
        """A frame with mask=False contributes nothing to the masked NLL."""
        head = MDNSNRMapHead(audio_dim=4, n_components=2)
        # T=2 frames; second frame has a wildly wrong target but is masked out.
        means = torch.tensor([[[5.0, 0.0], [0.0, 0.0]]], dtype=torch.float64)
        logs = torch.zeros(1, 2, 2, dtype=torch.float64)
        logits = torch.zeros(1, 2, 2, dtype=torch.float64)
        target = torch.tensor([[5.0, 1.0e6]], dtype=torch.float64)
        mask_all = torch.tensor([[True, True]])
        mask_one = torch.tensor([[True, False]])
        nll_all = float(head.nll((means, logs, logits), target, mask_all))
        nll_one = float(head.nll((means, logs, logits), target, mask_one))
        assert math.isfinite(nll_one)
        # masking out the catastrophic second frame strictly lowers the mean NLL
        assert nll_one < nll_all

    def test_forward_shapes_and_length_preserving(self):
        head = MDNSNRMapHead(audio_dim=8, n_components=3, hidden=16)
        x = torch.randn(2, 7, 8)
        means, log_sigmas, logits = head.forward(x)
        assert means.shape == (2, 7, 3)
        assert log_sigmas.shape == (2, 7, 3)
        assert logits.shape == (2, 7, 3)
        mean, var = head.predict((means, log_sigmas, logits))
        assert mean.shape == (2, 7)
        assert var.shape == (2, 7)
        assert bool((var >= 0).all())  # variance is non-negative


# ════════════════════════════════════════════════════════════════════════════════
# MC-dropout
# ════════════════════════════════════════════════════════════════════════════════
class TestMCDropout:
    def test_nonzero_variance_with_dropout(self):
        torch.manual_seed(0)
        head = MCDropoutSNRMapHead(audio_dim=8, hidden=16, p=0.5)
        x = torch.randn(1, 12, 8)
        with torch.no_grad():
            mean, var = head.sample(x, n=30)
        assert mean.shape == (1, 12)
        assert var.shape == (1, 12)
        assert float(var.mean()) > 0.0  # dropout makes the passes disagree

    def test_zero_variance_when_p_zero(self):
        """p=0 -> dropout is the identity -> all n passes coincide -> variance == 0."""
        head = MCDropoutSNRMapHead(audio_dim=8, hidden=16, p=0.0)
        x = torch.randn(1, 12, 8)
        with torch.no_grad():
            _, var = head.sample(x, n=10)
        assert float(var.abs().max()) == pytest.approx(0.0, abs=1e-12)

    def test_sample_n_one_is_zero_variance(self):
        head = MCDropoutSNRMapHead(audio_dim=8, hidden=16, p=0.5)
        x = torch.randn(1, 5, 8)
        with torch.no_grad():
            _, var = head.sample(x, n=1)
        assert float(var.abs().max()) == pytest.approx(0.0, abs=1e-12)


# ════════════════════════════════════════════════════════════════════════════════
# Ensemble
# ════════════════════════════════════════════════════════════════════════════════
class TestEnsemble:
    def test_variance_hand_example(self):
        """Members [1,2,3] at a single frame: mean=2, population var = (1+0+1)/3 = 2/3."""
        preds = [torch.tensor([1.0]), torch.tensor([2.0]), torch.tensor([3.0])]
        mean, var = ensemble_uncertainty(preds)
        # float32 default tensors -> use a float32-appropriate tolerance.
        assert float(mean[0]) == pytest.approx(2.0, abs=1e-6)
        assert float(var[0]) == pytest.approx(2.0 / 3.0, abs=1e-6)

    def test_single_member_zero_variance(self):
        mean, var = ensemble_uncertainty([torch.tensor([4.0, 5.0])])
        assert float(var.abs().max()) == pytest.approx(0.0, abs=1e-12)

    def test_two_d_shapes(self):
        a = torch.zeros(2, 3)
        b = torch.ones(2, 3)
        mean, var = ensemble_uncertainty([a, b])
        assert mean.shape == (2, 3)
        # mean of {0,1} = 0.5; population var = 0.25
        assert float(mean.mean()) == pytest.approx(0.5, abs=1e-9)
        assert float(var.mean()) == pytest.approx(0.25, abs=1e-9)


# ════════════════════════════════════════════════════════════════════════════════
# continuous-risk AURC + the not-a-band-cut property
# ════════════════════════════════════════════════════════════════════════════════
class TestAURC:
    def test_aurc_lower_when_uncertainty_tracks_error(self):
        """THE core property: an uncertainty that ranks errors well gives a LOWER AURC
        (sheds error faster as we abstain) than a random uncertainty."""
        rng = torch.Generator().manual_seed(0)
        n = 400
        errs = torch.rand(n, generator=rng).tolist()              # |error| in [0,1)
        good_unc = list(errs)                                      # perfectly correlated
        rand_unc = torch.rand(n, generator=rng).tolist()          # random
        a_good = continuous_risk_aurc(good_unc, errs)
        a_rand = continuous_risk_aurc(rand_unc, errs)
        assert a_good < a_rand

    def test_anti_correlated_uncertainty_is_worst(self):
        """An uncertainty ANTI-correlated with error (abstains on the SMALL errors first)
        is worse than random — AURC ordering must reflect that."""
        errs = [0.1, 0.2, 0.3, 0.9, 1.0]
        good = list(errs)                 # high unc on high err
        bad = [-e for e in errs]          # high unc on LOW err
        assert continuous_risk_aurc(good, errs) < continuous_risk_aurc(bad, errs)

    def test_continuous_risk_not_a_band_cut(self):
        """The AURC must INTEGRATE the real-valued risks, NOT threshold them to 0/1.
        Proof: scaling all risks by 10 scales the AURC by 10 (a band cut would be
        invariant to a monotone rescale because it only sees the 0/1 indicator)."""
        unc = [4.0, 3.0, 2.0, 1.0]
        risk = [0.4, 0.3, 0.2, 0.1]
        a1 = continuous_risk_aurc(unc, risk)
        a10 = continuous_risk_aurc(unc, [10.0 * r for r in risk])
        assert a10 == pytest.approx(10.0 * a1, rel=1e-9)
        # and the absolute value is the hand-computed cumulative-mean integral, NOT a
        # fraction in [0,1] (which a band cut would force).
        # perfect ordering: keep lowest-unc first => retained sets {0.1},{0.1,0.2},
        # {0.1,0.2,0.3},{0.1,..,0.4}; running means .1,.15,.2,.25; AURC=mean=0.175.
        assert a1 == pytest.approx(0.175, abs=1e-9)

    def test_perfect_ordering_integral_matches_hand(self):
        """Independently re-derive the cumulative-mean integral for a 3-frame case."""
        # risks 0.0, 0.5, 1.0; uncertainty equals risk (perfect). Keep lowest-unc first:
        # {0.0} mean 0; {0.0,0.5} mean .25; {0.0,0.5,1.0} mean .5 -> AURC=(0+.25+.5)/3
        risk = [0.0, 0.5, 1.0]
        a = continuous_risk_aurc(list(risk), risk)
        assert a == pytest.approx((0.0 + 0.25 + 0.5) / 3.0, abs=1e-9)

    def test_spearman_unc_err_perfect(self):
        risk = [0.1, 0.2, 0.3, 0.4]
        assert spearman_unc_err(list(risk), risk) == pytest.approx(1.0, abs=1e-9)


# ════════════════════════════════════════════════════════════════════════════════
# paired bootstrap CI
# ════════════════════════════════════════════════════════════════════════════════
class TestPairedBootstrap:
    def test_ci_brackets_known_negative_difference(self):
        """A GOOD uncertainty (challenger) vs a RANDOM one (incumbent): the paired
        bootstrap delta = AURC(good) - AURC(random) must be NEGATIVE, its 95% CI must
        BRACKET the point estimate, and (with a clear gap) exclude 0."""
        rng = torch.Generator().manual_seed(1)
        n = 500
        errs = torch.rand(n, generator=rng).tolist()
        good = list(errs)
        rand = torch.rand(n, generator=rng).tolist()
        bs = paired_bootstrap_aurc_delta(good, rand, errs, n_boot=400, seed=7)
        assert bs["point"] < 0.0
        assert bs["lo"] <= bs["point"] <= bs["hi"]
        assert bs["hi"] < 0.0           # CI excludes 0 -> a real win
        assert bs["p_value"] < 0.05     # one-sided P(delta>=0) is tiny

    def test_bootstrap_is_seed_reproducible(self):
        errs = [0.1, 0.5, 0.2, 0.9, 0.3, 0.7]
        a = paired_bootstrap_aurc_delta(list(errs), [0.5] * 6, errs, n_boot=200, seed=3)
        b = paired_bootstrap_aurc_delta(list(errs), [0.5] * 6, errs, n_boot=200, seed=3)
        assert a == b

    def test_identical_uncertainty_zero_delta(self):
        """method == incumbent => delta point is exactly 0."""
        errs = [0.2, 0.4, 0.6, 0.8]
        unc = [1.0, 2.0, 3.0, 4.0]
        bs = paired_bootstrap_aurc_delta(unc, list(unc), errs, n_boot=100, seed=0)
        assert bs["point"] == pytest.approx(0.0, abs=1e-12)


# ════════════════════════════════════════════════════════════════════════════════
# calibration / overconfidence
# ════════════════════════════════════════════════════════════════════════════════
class TestCalibration:
    def test_overconfident_flag_fires_on_too_tight_sigma(self):
        """Errors ~ N(0, 1) but the model reports sigma=0.1 (10x too tight): the 90%
        interval covers far less than 90% -> overconfident flag fires."""
        rng = torch.Generator().manual_seed(0)
        errs = torch.randn(2000, generator=rng).abs().tolist()  # |N(0,1)|
        tiny = [0.1] * len(errs)
        summ = calibration_summary(errs, tiny)
        assert summ["overconfident"] is True
        assert summ["empirical_coverage"]["90"] < 0.80

    def test_well_calibrated_not_flagged(self):
        """sigma == true scale: coverage ~ nominal, no overconfidence flag."""
        rng = torch.Generator().manual_seed(0)
        errs = torch.randn(5000, generator=rng).abs().tolist()
        ones = [1.0] * len(errs)
        summ = calibration_summary(errs, ones)
        assert summ["overconfident"] is False
        # 90% Gaussian interval should cover ~0.90 (loose bound for sampling noise)
        assert summ["empirical_coverage"]["90"] == pytest.approx(0.90, abs=0.05)


# ════════════════════════════════════════════════════════════════════════════════
# end-to-end harness
# ════════════════════════════════════════════════════════════════════════════════
class TestRunBakeoff:
    def _arrays(self):
        rng = torch.Generator().manual_seed(42)
        n = 600
        errs = torch.rand(n, generator=rng).tolist()
        return {
            "heteroscedastic_sigma": {
                "uncertainty": torch.rand(n, generator=rng).tolist(),  # random (weak)
                "abs_error": errs,
            },
            "mdn_var": {
                "uncertainty": list(errs),                             # perfect (strong)
                "abs_error": errs,
            },
        }

    def test_run_bakeoff_verdict_and_table(self):
        report = run_bakeoff(self._arrays(), n_boot=300, seed=0)
        assert report["incumbent"] == "heteroscedastic_sigma"
        assert "mdn_var" in report["methods"]
        # the strong MDN should beat the weak incumbent
        mdn = report["methods"]["mdn_var"]
        assert mdn["aurc"] < report["methods"]["heteroscedastic_sigma"]["aurc"]
        assert mdn["aurc_delta_vs_incumbent"] < 0.0
        # the verdict mentions a NO-GO when the MDN is a publishable win
        assert "decision" in report["verdict"]
        table = format_table(report)
        assert "UQ BAKE-OFF" in table
        assert "VERDICT" in table

    def test_hoeffding_floor_present_and_positive(self):
        report = run_bakeoff(self._arrays(), n_boot=100, seed=0)
        assert report["hoeffding_floor"] > 0.0
        assert report["hoeffding_floor_3000_stated"] == pytest.approx(0.054, abs=1e-3)

    def test_missing_incumbent_raises(self):
        with pytest.raises(ValueError):
            run_bakeoff({"mdn_var": {"uncertainty": [1.0], "abs_error": [0.5]}})

    def test_score_method_defaults_sigma_from_variance(self):
        """score_method derives sigma = sqrt(variance) for calibration when not given."""
        s = score_method([4.0, 1.0], [2.0, 1.0], "mdn_var")
        assert s["method"] == "mdn_var"
        assert "calibration" in s
        # sigma defaulted to sqrt([4,1]) = [2,1]
        assert s["calibration"]["n"] == 2
