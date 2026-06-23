"""Tests for the PRINCIPLED SFS scorer.

Covers the three additions in src/sfs.py (+ the metrics_calibrated wiring):

  1. RELATIVE tolerance through SFSScorer(tol_config=ToleranceConfig(...)):
       - rel_frac == 0 (and no config) reproduces the legacy absolute precision
         EXACTLY (true no-op), per-claim and aggregate.
       - tol(f, gt) widens with |gt| once rel_frac > 0.
  2. BASELINE-RELATIVE SFS skill = precision - mode/median-baseline precision,
     scored under the SAME tolerance (hand-checked tiny case; pause_count
     artifact shrinks under baseline subtraction).
  3. TOLERANCE SWEEP: raw precision is MONOTONE non-decreasing in the multiplier.

The pause_count artifact: GT in {1,1,1,2}, ±1 tolerance. A constant mode
predictor (predict 1) is within ±1 of every GT, so baseline precision == 1.0; a
model that is also ~always within ±1 has near-zero SKILL even at ~1.0 raw
precision.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from sfs import (  # noqa: E402
    Claim,
    SFSScorer,
    ToleranceConfig,
    baseline_relative_sfs,
    tolerance_sweep,
    _baseline_predictor,
)
from metrics_calibrated import relative_tolerance  # noqa: E402


# ── 1. relative tolerance / no-op ────────────────────────────────────────────
class TestRelativeToleranceNoOp:
    def _claims(self, snr=None, f0=None):
        cs = []
        if snr is not None:
            cs.append(Claim("snr", snr, "dB", ""))
        if f0 is not None:
            cs.append(Claim("f0_mean", f0, "Hz", ""))
        return cs

    def test_no_config_is_legacy(self):
        """Default scorer (no tol_config) == the original TOLERANCES path."""
        scorer = SFSScorer()
        # snr tol = 2.0: 31.0 vs gt 30.0 -> err 1.0 <= 2.0 correct;
        # 35.0 vs gt 30.0 -> err 5.0 > 2.0 wrong.
        gt = {"snr": 30.0}
        assert scorer.score(self._claims(snr=31.0), gt)["precision"] == 1.0
        assert scorer.score(self._claims(snr=35.0), gt)["precision"] == 0.0

    def test_rel_frac_zero_reproduces_absolute_exactly(self):
        """A ToleranceConfig with empty rel_frac is a TRUE no-op vs the default
        scorer on a battery of features/values (per-claim correctness identical)."""
        legacy = SFSScorer()
        principled = SFSScorer(tol_config=ToleranceConfig())  # no rel_frac, no floors
        cases = [
            ({"snr": 30.0}, [Claim("snr", v, "dB", "") for v in (28.0, 31.9, 32.1, 35.0)]),
            ({"f0_mean": 200.0}, [Claim("f0_mean", v, "Hz", "") for v in (195.0, 205.1, 210.0)]),
            ({"pause_count": 1.0}, [Claim("pause_count", v, "", "") for v in (0.0, 1.0, 2.0, 3.0)]),
            ({"shimmer": 10.0}, [Claim("shimmer", v, "%", "") for v in (9.5, 10.4, 10.6)]),
        ]
        for gt, claims in cases:
            for c in claims:
                a = legacy.score([c], gt)
                b = principled.score([c], gt)
                assert a["precision"] == b["precision"], (gt, c.value, a, b)
                assert a["n_correct"] == b["n_correct"]

    def test_explicit_rel_frac_zero_table_is_noop(self):
        """rel_frac dict with explicit 0.0 entries also reproduces absolute."""
        legacy = SFSScorer()
        cfg = ToleranceConfig(rel_frac={"snr": 0.0, "f0_mean": 0.0})
        principled = SFSScorer(tol_config=cfg)
        gt = {"snr": 30.0}
        for v in (28.0, 31.9, 32.1, 35.0):
            c = [Claim("snr", v, "dB", "")]
            assert legacy.score(c, gt)["precision"] == principled.score(c, gt)["precision"]

    def test_relative_tol_widens_with_gt(self):
        """rel_frac > 0 makes the band scale with |gt|: a miss that was wrong
        under the absolute floor becomes correct at high GT magnitude."""
        # snr abs floor 2.0, rel_frac 0.10. err = 3.0.
        cfg = ToleranceConfig(rel_frac={"snr": 0.10})
        scorer = SFSScorer(tol_config=cfg)
        # gt 10.0 -> tol max(2.0, 1.0) = 2.0 -> err 3.0 WRONG.
        assert scorer.score([Claim("snr", 13.0, "dB", "")], {"snr": 10.0})["precision"] == 0.0
        # gt 40.0 -> tol max(2.0, 4.0) = 4.0 -> err 3.0 CORRECT.
        assert scorer.score([Claim("snr", 43.0, "dB", "")], {"snr": 40.0})["precision"] == 1.0

    def test_scorer_tolerance_reuses_metrics_calibrated(self):
        """SFSScorer._tolerance and metrics_calibrated.relative_tolerance agree
        for the same config (shared implementation, no drift)."""
        cfg = ToleranceConfig(rel_frac={"snr": 0.10})
        scorer = SFSScorer(tol_config=cfg)
        for gt in (5.0, 20.0, 40.0):
            direct = relative_tolerance(
                "snr", gt,
                abs_tol={**SFSScorer.TOLERANCES, "snr": SFSScorer.TOLERANCES["snr"]},
                rel_frac={"snr": 0.10},
            )
            assert abs(scorer._tolerance("snr", gt) - direct) < 1e-12

    def test_toleranceconfig_tolerance_method(self):
        """ToleranceConfig.tolerance() == max(floor, frac*|gt|)."""
        cfg = ToleranceConfig(abs_floor={"snr": 2.0}, rel_frac={"snr": 0.10})
        assert cfg.tolerance("snr", 10.0) == 2.0   # max(2.0, 1.0)
        assert cfg.tolerance("snr", 40.0) == 4.0   # max(2.0, 4.0)


# ── 2. baseline-relative skill ───────────────────────────────────────────────
class TestBaselineRelativeSkill:
    def test_mode_baseline_value(self):
        assert _baseline_predictor([1.0, 1.0, 1.0, 2.0], "mode") == 1.0
        assert _baseline_predictor([0.0, 3.0, 3.0, 3.0], "mode") == 3.0

    def test_median_baseline_value(self):
        assert _baseline_predictor([10.0, 20.0, 30.0], "median") == 20.0

    def test_skill_equals_precision_minus_baseline_handcheck(self):
        """Tiny hand-checked case.

        feature snr (abs tol 2.0). gts = [10, 20, 30], mode baseline = 10
        (each value appears once; _baseline_predictor returns the first modal
        value = 10).
          model preds = [10, 20, 30]  -> all within 2.0 -> precision 1.0
          baseline const = 10         -> |10-10|=0 ok, |10-20|=10 no,
                                         |10-30|=20 no -> baseline 1/3.
          skill = 1.0 - 1/3 = 2/3.
        """
        pairs = {"snr": ([10.0, 20.0, 30.0], [10.0, 20.0, 30.0])}
        rep = baseline_relative_sfs(pairs)  # legacy absolute tol
        pf = rep["per_feature"]["snr"]
        assert pf["precision"] == 1.0
        assert abs(pf["baseline_precision"] - 1.0 / 3.0) < 1e-12
        assert abs(pf["skill"] - 2.0 / 3.0) < 1e-12
        assert pf["baseline_value"] == 10.0
        assert pf["n"] == 3

    def test_pause_count_artifact_shrinks(self):
        """pause_count: GT mostly == 1 with ±1 tolerance. A near-perfect raw
        precision collapses to ~0 SKILL because the mode baseline is also
        within ±1 of every clip."""
        # 9 clips at gt=1, 1 clip at gt=2. Model nails all (pred==gt) -> prec 1.0.
        gts = [1.0] * 9 + [2.0]
        preds = list(gts)  # perfect model
        pairs = {"pause_count": (preds, gts)}
        rep = baseline_relative_sfs(pairs)
        pf = rep["per_feature"]["pause_count"]
        # mode baseline = 1.0. |1-1|=0 ok (x9), |1-2|=1 <=1 ok (x1) -> baseline 1.0.
        assert pf["precision"] == 1.0
        assert pf["baseline_precision"] == 1.0
        assert pf["skill"] == 0.0  # artifact fully exposed: no skill over trivial
        # And the skill is strictly less than raw precision (the headline point).
        assert pf["skill"] < pf["precision"]

    def test_skill_positive_when_model_beats_constant(self):
        """A high-variance feature the model tracks well shows real positive
        skill: the constant baseline cannot cover the spread."""
        gts = [5.0, 15.0, 25.0, 35.0, 45.0]
        preds = list(gts)  # model tracks
        pairs = {"snr": (preds, gts)}
        rep = baseline_relative_sfs(pairs)
        pf = rep["per_feature"]["snr"]
        assert pf["precision"] == 1.0
        assert pf["baseline_precision"] < 0.5  # const within ±2 of at most 1/5
        assert pf["skill"] > 0.5

    def test_aggregate_pools_by_clip_count(self):
        pairs = {
            "snr": ([10.0, 20.0], [10.0, 20.0]),            # 2 clips, prec 1.0
            "pause_count": ([1.0, 1.0, 1.0], [1.0, 1.0, 1.0]),  # 3 clips, prec 1.0
        }
        rep = baseline_relative_sfs(pairs)
        agg = rep["aggregate"]
        assert agg["n"] == 5
        assert agg["precision"] == 1.0
        # baseline: snr const 10 covers 1/2; pause_count const 1 covers 3/3.
        # pooled baseline correct = 1 + 3 = 4 over 5 = 0.8.
        assert abs(agg["baseline_precision"] - 0.8) < 1e-12
        assert abs(agg["skill"] - 0.2) < 1e-12


# ── 4. baseline_kind contract (footgun guard) ────────────────────────────────
class TestBaselineKindContract:
    """`baseline_relative_sfs` defaults to baseline_kind="mode" for EVERY feature.

    The paper analysis uses mode ONLY for the discrete count (pause_count) and
    median for every continuous feature. On a continuous, non-modal distribution
    the two baselines differ, so the default mode-for-all silently produces the
    WRONG skill. These tests pin that contract: callers reproducing the paper
    numbers MUST pass a per-feature {pause_count:"mode", else:"median"} dict.
    """

    def test_mode_and_median_baselines_diverge_on_continuous(self):
        # A distribution where mode (most-frequent value) and median (positional
        # middle) genuinely differ AND the baseline coverage under ±2 differs.
        gts4 = [0.0, 0.0, 100.0, 101.0, 102.0]  # mode 0, median 100
        preds4 = list(gts4)
        p4 = {"snr": (preds4, gts4)}
        b_mode = baseline_relative_sfs(p4, baseline_kind="mode")["per_feature"]["snr"]
        b_med = baseline_relative_sfs(p4, baseline_kind="median")["per_feature"]["snr"]
        # mode const = 0 -> covers the two gt=0 clips (±2) -> 2/5 = 0.4.
        assert abs(b_mode["baseline_value"] - 0.0) < 1e-12
        assert abs(b_mode["baseline_precision"] - 0.4) < 1e-12
        # median const = 100 -> covers gt in {100,101,102}? |100-100|,|100-101|,
        # |100-102| -> 0,1,2 all <=2 -> 3 clips -> 3/5 = 0.6.
        assert abs(b_med["baseline_value"] - 100.0) < 1e-12
        assert abs(b_med["baseline_precision"] - 0.6) < 1e-12
        # The divergence is real: skill under the two baselines differs.
        assert b_mode["skill"] != b_med["skill"]

    def test_per_feature_baseline_kind_dict(self):
        """A dict baseline_kind applies mode to pause_count, median elsewhere —
        the exact invocation the paper / rescore driver uses."""
        pairs = {
            "pause_count": ([1.0] * 9 + [2.0], [1.0] * 9 + [2.0]),
            "snr": ([0.0, 0.0, 100.0, 101.0, 102.0], [0.0, 0.0, 100.0, 101.0, 102.0]),
        }
        kind = {"pause_count": "mode", "snr": "median"}
        rep = baseline_relative_sfs(pairs, baseline_kind=kind)
        assert rep["per_feature"]["pause_count"]["baseline_kind"] == "mode"
        assert rep["per_feature"]["snr"]["baseline_kind"] == "median"
        # pause_count mode baseline = 1.0 (modal); snr median baseline = 100.0.
        assert rep["per_feature"]["pause_count"]["baseline_value"] == 1.0
        assert rep["per_feature"]["snr"]["baseline_value"] == 100.0


# ── 3. tolerance sweep monotonicity ──────────────────────────────────────────
class TestToleranceSweep:
    def test_precision_monotone_in_multiplier(self):
        """For fixed preds/gts, raw precision is NON-DECREASING as the tolerance
        multiplier grows (a wider band can only add corrects)."""
        # snr abs floor 2.0. errors spread so different multipliers flip claims.
        gts = [10.0, 20.0, 30.0, 40.0, 50.0]
        preds = [13.0, 24.0, 38.0, 41.0, 60.0]  # errs 3,4,8,1,10
        pairs = {"snr": (preds, gts)}
        sweep = tolerance_sweep(pairs, multipliers=(0.5, 1.0, 2.0, 4.0))
        precs = [r["precision"] for r in sweep["per_feature"]["snr"]]
        for a, b in zip(precs, precs[1:]):
            assert b >= a, precs
        # widest band should be strictly better than narrowest here.
        assert precs[-1] > precs[0]

    def test_sweep_aggregate_monotone(self):
        gts = [10.0, 20.0, 30.0, 40.0]
        preds = [14.0, 23.0, 36.0, 39.0]  # errs 4,3,6,1
        pairs = {"snr": (preds, gts)}
        sweep = tolerance_sweep(pairs, multipliers=(0.5, 1.0, 2.0, 4.0))
        aprec = [r["precision"] for r in sweep["aggregate"]]
        for a, b in zip(aprec, aprec[1:]):
            assert b >= a, aprec

    def test_sweep_m1_equals_absolute(self):
        """At multiplier 1.0 with the default (absolute) base config, sweep
        precision equals the plain baseline_relative_sfs precision."""
        gts = [10.0, 20.0, 30.0]
        preds = [11.0, 25.0, 30.0]
        pairs = {"snr": (preds, gts)}
        base = baseline_relative_sfs(pairs)["per_feature"]["snr"]["precision"]
        sweep = tolerance_sweep(pairs, multipliers=(1.0,))
        assert sweep["per_feature"]["snr"][0]["precision"] == base

    def test_sweep_carries_skill(self):
        gts = [1.0] * 9 + [2.0]
        preds = list(gts)
        pairs = {"pause_count": (preds, gts)}
        sweep = tolerance_sweep(pairs, multipliers=(0.5, 1.0, 2.0, 4.0))
        by_m = {r["multiplier"]: r for r in sweep["per_feature"]["pause_count"]}
        # The sweep MUST carry a skill field at every multiplier.
        assert all("skill" in r for r in sweep["per_feature"]["pause_count"])
        # pause_count artifact: at the LOOSE end (m>=1, tol>=1) the mode baseline
        # already covers every clip, so skill collapses to 0 — the artifact is
        # exposed exactly where the default tolerance sits.
        assert by_m[1.0]["skill"] == 0.0
        assert by_m[2.0]["skill"] == 0.0
        assert by_m[4.0]["skill"] == 0.0
        # At the TIGHT end (m=0.5, tol=0.5) the baseline can no longer cover the
        # gt=2 clip, so a (perfect) model shows real positive skill. This is the
        # diagnostic value of the sweep: tightening separates skill from artifact.
        assert by_m[0.5]["skill"] > 0.0
