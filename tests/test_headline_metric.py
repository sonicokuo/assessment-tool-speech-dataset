"""Metric-consistency tests for the FROZEN headline (fixes F8/F9/F10, 2026-07-15 audit).

The audit found the VAL selection metric (mean SRCC over 5 robust features) and the
TEST reporting metric (scripts/score_inference_vs_clean.py, which averaged ~8 features
including f0_mean/f0_sd and a stale articulation_rate row) had silently diverged.
These tests pin the fixed contract across all three owners of the number:

  * src/eval/selection_metric.HEADLINE_FEATURES  — the single source of truth (ROBUST5)
  * scripts/score_inference_vs_clean.py          — test-time reporting vs clean GT
  * scripts/bandfree_val_eval.py                 — val_samples NTL-vs-noNTL eval

Cases required by the fix spec:
  (a) mean_reliable uses exactly ROBUST5
  (b) articulation_rate is gone from the reporting scorer
  (c) f0 appears only in the ill-posed / abstention panel, never in the headline
  (d) coverage math is correct, incl. a feature emitted on 0 clips (no crash,
      coverage 0.0, SRCC None)
  (e) the no-snr secondary mean (SNR-circularity check)

Pure-python fixtures (pre-parsed claim dicts; no model, no scipy/numpy needed).
The imports transitively reach data.feature_set, which imports torch at module
level (owned by another workstream), so the module skips cleanly without torch;
it runs in full on PSC.
"""
import importlib.util
import math
import os
import sys

import pytest

if importlib.util.find_spec("torch") is None:
    pytest.skip(
        "eval.selection_metric -> data.feature_set imports torch at module level",
        allow_module_level=True,
    )

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import score_inference_vs_clean as S  # noqa: E402
import bandfree_val_eval as B  # noqa: E402
from eval.selection_metric import HEADLINE_FEATURES  # noqa: E402

ROBUST5 = ("snr", "srmr", "speaking_rate", "pause_count", "pause_rate")


# ─────────────────────────────────────────────────────────────────────────────
# fixture: 25 clips of pre-parsed claims with KNOWN per-feature rank relations
#   snr / srmr / pause_count : claimed tracks GT       -> SRCC +1
#   speaking_rate            : claimed reversed vs GT  -> SRCC -1
#   pause_rate               : emitted on 15/25 clips  -> SRCC +1, coverage 0.6
#   f0_mean (panel)          : claimed reversed vs GT  -> SRCC -1 (must NOT touch headline)
#   f0_sd (panel)            : emitted on 0 clips      -> SRCC None, coverage 0.0
#   articulation_rate        : claimed on EVERY clip, GT present -> must be ignored
# ─────────────────────────────────────────────────────────────────────────────
N = 25


def _fixture():
    results, feats, f0 = [], {}, {}
    for i in range(N):
        fn = f"clip{i}.wav"
        x = float(i)
        claims = [
            {"feature": "snr", "claimed": x},
            {"feature": "srmr", "claimed": x},
            {"feature": "speaking_rate", "claimed": -x},
            {"feature": "pause_count", "claimed": x},
            # stale feature: emitted AND has GT below — a regression to the old
            # FEATURE_MAP would silently pull it back into the mean
            {"feature": "articulation_rate", "claimed": x},
            {"feature": "f0_mean", "claimed": -x},
        ]
        if i < 15:
            claims.append({"feature": "pause_rate", "claimed": x})
        results.append({"filename": fn, "per_feature": claims})
        feats[fn] = {
            "snr_db": x,
            "srmr": x,
            "praat_speaking_rate_syl_sec": x,
            "praat_pause_count": x,
            "praat_pause_rate_per_min": x,
            "praat_articulation_rate_syl_sec": x,
        }
        f0[fn] = {"f0_mean_hz": x, "f0_sd_hz": x}
    return results, feats, f0


@pytest.fixture(scope="module")
def scored():
    results, feats, f0 = _fixture()
    return S.score_vs_clean(results, feats, f0)


# ═══════════════════════════════════════════════════════════════════════════
# the frozen set itself
# ═══════════════════════════════════════════════════════════════════════════
class TestFrozenSetSharedEverywhere:
    def test_canonical_value(self):
        assert tuple(HEADLINE_FEATURES) == ROBUST5

    def test_both_scripts_import_the_same_object(self):
        # not just equal — the scripts must consume selection_metric's constant
        assert S.HEADLINE_FEATURES is HEADLINE_FEATURES
        assert B.HEADLINE_FEATURES is HEADLINE_FEATURES


# ═══════════════════════════════════════════════════════════════════════════
# (a) mean_reliable == mean SRCC over exactly ROBUST5
# ═══════════════════════════════════════════════════════════════════════════
class TestHeadlineIsExactlyRobust5:
    def test_headline_value(self, scored):
        # snr +1, srmr +1, speaking_rate -1, pause_count +1, pause_rate +1
        assert scored["mean_reliable"] == pytest.approx((1 + 1 - 1 + 1 + 1) / 5)
        assert scored["n_reliable_features"] == 5

    def test_headline_equals_mean_over_frozen_set(self, scored):
        vals = [scored["per_feature"][f]["srcc"] for f in ROBUST5]
        assert all(v is not None for v in vals)
        assert scored["mean_reliable"] == pytest.approx(sum(vals) / len(vals))

    def test_f0_would_change_the_mean_if_included(self, scored):
        # guard against silent re-inclusion: f0_mean has SRCC -1 here, so a
        # 6-feature mean would differ from the frozen 5-feature headline
        wrong = ([scored["per_feature"][f]["srcc"] for f in ROBUST5]
                 + [scored["per_feature"]["f0_mean"]["srcc"]])
        assert scored["mean_reliable"] != pytest.approx(sum(wrong) / len(wrong))


# ═══════════════════════════════════════════════════════════════════════════
# (b) articulation_rate is gone
# ═══════════════════════════════════════════════════════════════════════════
class TestArticulationRateDeleted:
    def test_not_in_feature_map(self):
        assert "articulation_rate" not in S.FEATURE_MAP

    def test_not_scored_even_when_emitted_with_gt(self, scored):
        # the fixture emits it on every clip AND provides clean GT for it
        assert "articulation_rate" not in scored["per_feature"]

    def test_not_in_headline_set(self):
        assert "articulation_rate" not in HEADLINE_FEATURES


# ═══════════════════════════════════════════════════════════════════════════
# (c) f0 appears only in the ill-posed / abstention panel
# ═══════════════════════════════════════════════════════════════════════════
class TestF0PanelOnly:
    def test_f0_stats_are_still_computed(self, scored):
        assert scored["per_feature"]["f0_mean"]["srcc"] == pytest.approx(-1.0)
        assert scored["per_feature"]["f0_mean"]["coverage"] == pytest.approx(1.0)

    def test_f0_is_panel_not_headline(self):
        assert "f0_mean" in S.PANEL_FEATURES and "f0_sd" in S.PANEL_FEATURES
        assert "f0_mean" not in HEADLINE_FEATURES
        assert "f0_sd" not in HEADLINE_FEATURES
        # panel and headline partition FEATURE_MAP
        assert set(S.PANEL_FEATURES) | set(HEADLINE_FEATURES) == set(S.FEATURE_MAP)
        assert not set(S.PANEL_FEATURES) & set(HEADLINE_FEATURES)

    def test_bandfree_panel_excludes_headline(self):
        assert not set(B.PANEL_ORDER) & set(HEADLINE_FEATURES)
        for f in ("f0_mean", "f0_sd", "overlap_ratio", "articulation_rate"):
            assert f in B.PANEL_ORDER


# ═══════════════════════════════════════════════════════════════════════════
# (d) coverage math, incl. the 0-emission feature
# ═══════════════════════════════════════════════════════════════════════════
class TestCoverage:
    def test_partial_coverage_fraction(self, scored):
        pf = scored["per_feature"]["pause_rate"]
        assert pf["n_emitted"] == 15
        assert pf["coverage"] == pytest.approx(15 / N)
        assert pf["srcc"] == pytest.approx(1.0)  # SRCC over the 15 paired clips

    def test_zero_emission_feature_no_crash(self, scored):
        pf = scored["per_feature"]["f0_sd"]  # never emitted in the fixture
        assert pf["srcc"] is None
        assert pf["coverage"] == 0.0
        assert pf["n"] == 0 and pf["n_emitted"] == 0

    def test_headline_mean_coverage(self, scored):
        # snr/srmr/speaking_rate/pause_count at 1.0, pause_rate at 0.6
        assert scored["mean_coverage_reliable"] == pytest.approx((4 * 1.0 + 0.6) / 5)

    def test_empty_results_no_crash(self):
        out = S.score_vs_clean([], {}, {})
        assert math.isnan(out["mean_reliable"])
        assert out["mean_coverage_reliable"] == 0.0
        for f in S.FEATURE_MAP:
            assert out["per_feature"][f]["srcc"] is None
            assert out["per_feature"][f]["coverage"] == 0.0


# ═══════════════════════════════════════════════════════════════════════════
# (e) the no-snr secondary mean (SNR-circularity check)
# ═══════════════════════════════════════════════════════════════════════════
class TestNoSnrSecondary:
    def test_no_snr_mean(self, scored):
        # drop snr (+1): mean over srmr +1, speaking_rate -1, pause_count +1,
        # pause_rate +1
        assert scored["mean_reliable_no_snr"] == pytest.approx((1 - 1 + 1 + 1) / 4)

    def test_diverges_from_headline_when_snr_dominates(self, scored):
        assert scored["mean_reliable"] != pytest.approx(scored["mean_reliable_no_snr"])


# ═══════════════════════════════════════════════════════════════════════════
# bandfree_val_eval.headline_summary — same frozen contract, pure aggregation
# ═══════════════════════════════════════════════════════════════════════════
class TestBandfreeHeadlineSummary:
    SCORES = {
        "snr":           {"srcc": 0.9,  "cov": 1.0},
        "srmr":          {"srcc": 0.5,  "cov": 0.8},
        "speaking_rate": {"srcc": 0.7,  "cov": 1.0},
        "pause_count":   {"srcc": None, "cov": 0.0},   # degenerated: 0 emissions
        "pause_rate":    {"srcc": 0.3,  "cov": 0.6},
        "f0_mean":       {"srcc": -0.9, "cov": 1.0},   # panel — must not leak in
        "overlap_ratio": {"srcc": 1.0,  "cov": 1.0},   # panel — must not leak in
    }

    def test_headline_mean_over_defined_robust5(self):
        h = B.headline_summary(self.SCORES)
        assert h["mean_srcc"] == pytest.approx(round((0.9 + 0.5 + 0.7 + 0.3) / 4, 3))
        assert h["n_feats"] == 4

    def test_no_snr_secondary(self):
        h = B.headline_summary(self.SCORES)
        assert h["mean_srcc_no_snr"] == pytest.approx(round((0.5 + 0.7 + 0.3) / 3, 3))

    def test_mean_coverage_counts_collapsed_feature(self):
        # the 0-coverage pause_count stays IN the coverage mean (audit risk 3)
        h = B.headline_summary(self.SCORES)
        assert h["mean_cov"] == pytest.approx(round((1.0 + 0.8 + 1.0 + 0.0 + 0.6) / 5, 3))

    def test_all_undefined_returns_none_not_crash(self):
        h = B.headline_summary({f: {"srcc": None, "cov": 0.0} for f in ROBUST5})
        assert h["mean_srcc"] is None
        assert h["mean_srcc_no_snr"] is None
        assert h["mean_cov"] == 0.0
        assert h["n_feats"] == 0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
