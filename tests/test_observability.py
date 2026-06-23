"""Tests for the observability-aware target builder + selective SFS scoring.

Covers:
  - scripts/build_descriptions_observability.py
      * F0 abstention decision (high-overlap -> hedge, low-overlap -> number)
      * recoverable features always emitted as SFS-parseable numbers
      * clean-frame F0 substitution for the asserted number
      * valid JSON keyed by clip stems (via the description strings)
  - src/sfs.py selective scoring
      * AbstentionDetector recognizes the F0 hedge
      * SFSScorer.score_selective: correct / wrong / abstained / omitted / overclaim
      * a hedge on an unreliable feature is REWARDED, not penalized
      * abstaining on a RELIABLE feature is a coverage miss
      * the OLD SFSScorer.score path is unchanged (back-compat)
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

import build_descriptions_observability as bdo  # noqa: E402
from sfs import (  # noqa: E402
    AbstentionDetector,
    Claim,
    ClaimParser,
    HybridClaimParser,
    SFSScorer,
)


# ── Shared fixtures: a high-overlap and a low-overlap (clean) row ────────────
def _high_overlap_row():
    """A Libri2Mix mixture row: VAD overlap 0.79 -> F0 ill-posed -> abstain."""
    return {
        "filename": "1272-128104-0000_2035-147961-0014.wav",
        "duration_sec": "8.0",
        "sample_rate_hz": "16000",
        "snr_db": "13.17",
        "srmr": "5.3646",
        "f0_mean_hz": "180.0",
        "f0_sd_hz": "40.0",
        "praat_speaking_rate_syl_sec": "6.888",
        "praat_pause_count": "1",
        "praat_pause_rate_per_min": "15.306",
        "overlap_ratio": "0.3333",
        "overlap_segments": "8000-40000",
        "overlap_ratio_vad": "0.7908",
        "overlap_segments_vad": "8000-40000",
    }


def _clean_low_overlap_row():
    """A clean single-speaker row: no overlap columns -> F0 well-posed -> assert."""
    return {
        "filename": "1089-134686-0000_121-127105-0031.wav",
        "duration_sec": "6.0",
        "sample_rate_hz": "16000",
        "snr_db": "26.15",
        "srmr": "4.0477",
        "f0_mean_hz": "96.96",   # mixture F0 in the row
        "f0_sd_hz": "30.0",
        "praat_speaking_rate_syl_sec": "4.312",
        "praat_pause_count": "4",
        "praat_pause_rate_per_min": "23.000",
        # No overlap_ratio / *_vad columns -> treated as overlap 0.0 (clean).
    }


_CLEAN_F0 = {
    # clean-frame (well-posed) F0 for the clean row.
    "1089-134686-0000_121-127105-0031.wav": {
        "f0_mean_hz": 108.17, "f0_sd_hz": 28.07, "clean_voiced_frac": 0.1254,
    },
}


# ── Builder: F0 abstention decision ──────────────────────────────────────────
class TestF0Decision:
    def test_high_overlap_abstains(self):
        row = bdo._prepare_row_for_build(_high_overlap_row(), {"warned": False})
        action, info = bdo.f0_decision(
            row, clean_f0=None, overlap_threshold=0.30, min_clean_voiced_frac=0.05,
        )
        assert action == "abstain"
        assert info["reason"] == "overlap"
        assert info["overlap"] >= 0.30

    def test_low_overlap_asserts(self):
        row = bdo._prepare_row_for_build(_clean_low_overlap_row(), {"warned": False})
        action, info = bdo.f0_decision(
            row, clean_f0=_CLEAN_F0, overlap_threshold=0.30, min_clean_voiced_frac=0.05,
        )
        assert action == "assert"
        # The asserted number is the WELL-POSED clean-frame F0, not the mixture F0.
        assert abs(info["f0_mean"] - 108.17) < 1e-6
        assert abs(info["f0_sd"] - 28.07) < 1e-6

    def test_clean_f0_undefined_abstains(self):
        row = bdo._prepare_row_for_build(_clean_low_overlap_row(), {"warned": False})
        cf = {"1089-134686-0000_121-127105-0031.wav": {"f0_mean_hz": None}}
        action, info = bdo.f0_decision(
            row, clean_f0=cf, overlap_threshold=0.30, min_clean_voiced_frac=0.05,
        )
        assert action == "abstain"
        assert info["reason"] == "undefined"

    def test_too_sparse_clean_frames_abstains(self):
        row = bdo._prepare_row_for_build(_clean_low_overlap_row(), {"warned": False})
        cf = {"1089-134686-0000_121-127105-0031.wav": {
            "f0_mean_hz": 108.17, "f0_sd_hz": 28.07, "clean_voiced_frac": 0.01,
        }}
        action, info = bdo.f0_decision(
            row, clean_f0=cf, overlap_threshold=0.30, min_clean_voiced_frac=0.05,
        )
        assert action == "abstain"
        assert info["reason"] == "undefined"

    def test_threshold_knob_controls_abstention(self):
        # At threshold 0.99, the 0.79-overlap clip is BELOW threshold -> assert.
        row = bdo._prepare_row_for_build(_high_overlap_row(), {"warned": False})
        action, _ = bdo.f0_decision(
            row, clean_f0=None, overlap_threshold=0.99, min_clean_voiced_frac=0.05,
        )
        assert action == "assert"


# ── Builder: description content ─────────────────────────────────────────────
class TestBuilderOutput:
    def _build(self, row, clean_f0=None, threshold=0.30):
        r = dict(row)
        stem = os.path.splitext(r["filename"])[0]
        return bdo.build_observability_description(
            r, stem, clean_f0, threshold, 0.05, {"warned": False},
        )

    def test_high_overlap_has_hedge_no_f0_number(self):
        text = self._build(_high_overlap_row())
        # Hedge present; no F0 number asserted.
        det = AbstentionDetector()
        assert det.detect(text) == {"f0_mean", "f0_sd"}
        claims = ClaimParser().parse(text)
        feats = {c.feature for c in claims}
        assert "f0_mean" not in feats
        assert "f0_sd" not in feats

    def test_low_overlap_has_f0_number_no_hedge(self):
        text = self._build(_clean_low_overlap_row(), clean_f0=_CLEAN_F0)
        det = AbstentionDetector()
        assert det.detect(text) == set()
        claims = ClaimParser().parse(text)
        f0 = {c.feature: c.value for c in claims}
        assert "f0_mean" in f0
        # asserted F0 is the clean-frame value.
        assert abs(f0["f0_mean"] - 108.17) < 1e-6

    def test_recoverable_features_always_numeric_and_parseable(self):
        for row, cf in [
            (_high_overlap_row(), None),
            (_clean_low_overlap_row(), _CLEAN_F0),
        ]:
            text = self._build(row, clean_f0=cf)
            claims = {c.feature: c.value for c in ClaimParser().parse(text)}
            # snr, srmr, speaking_rate, pause_count, pause_rate always present.
            for f in ("snr", "srmr", "speaking_rate", "pause_count", "pause_rate"):
                assert f in claims, f"{f} missing/unparseable in: {text}"
            # snr value round-trips.
            assert abs(claims["snr"] - float(row["snr_db"])) < 0.01

    def test_overlap_ratio_emitted_when_present(self):
        text = self._build(_high_overlap_row())
        claims = {c.feature: c.value for c in ClaimParser().parse(text)}
        assert "overlap_ratio" in claims

    def test_clean_row_omits_overlap_sentence(self):
        # Clean single-speaker row has no overlap_ratio column -> overlap is a
        # selective omission (no overlap sentence), which is correct coverage.
        text = self._build(_clean_low_overlap_row(), clean_f0=_CLEAN_F0)
        claims = {c.feature for c in ClaimParser().parse(text)}
        assert "overlap_ratio" not in claims

    def test_description_is_nonempty_string(self):
        text = self._build(_high_overlap_row())
        assert isinstance(text, str) and len(text) > 0


# ── Abstention detector ──────────────────────────────────────────────────────
class TestAbstentionDetector:
    def setup_method(self):
        self.det = AbstentionDetector()

    def test_detects_overlap_hedge(self):
        t = ("Because the two speakers overlap heavily (0.79 overlap ratio), "
             "the pitch cannot be reliably estimated from the mixture, so no F0 "
             "value is reported.")
        assert self.det.detect(t) == {"f0_mean", "f0_sd"}

    def test_detects_ill_posed_phrasing(self):
        t = ("Given the heavy speaker overlap (0.80 overlap ratio), F0 is "
             "ill-posed on this mixture and is not asserted.")
        assert self.det.detect(t) == {"f0_mean", "f0_sd"}

    def test_detects_too_few_clean_frames(self):
        t = ("Too few clean, non-overlapped voiced frames are available to "
             "estimate pitch, so no F0 value is reported.")
        assert self.det.detect(t) == {"f0_mean", "f0_sd"}

    def test_legacy_unreliable_sentence_detected(self):
        # The OLD builder's trailing sentence is also a (constant) hedge.
        t = "F0 and formant estimates are unreliable during overlap windows."
        assert "f0_mean" in self.det.detect(t)

    def test_no_hedge_when_f0_asserted(self):
        t = "The F0 mean is 150.0 Hz and the F0 standard deviation SD is 40.0 Hz."
        assert self.det.detect(t) == set()

    def test_no_hedge_on_plain_numeric_clip(self):
        t = "The SNR is 20.0 dB. The speaking rate is 6.0 syl/sec."
        assert self.det.detect(t) == set()


# ── Selective scoring ────────────────────────────────────────────────────────
class TestSelectiveScoring:
    def setup_method(self):
        # legacy absolute band: this suite pins ±tol wrong-number penalization
        # (SFS now defaults to the Tier-3 principled band; shim it here).
        self.sc = SFSScorer(legacy=True)

    def test_good_abstention_rewarded(self):
        text = ("The SNR is 13.17 dB, a moderate noise level. The SRMR is 5.36. "
                "Because the two speakers overlap heavily (0.79 overlap ratio), "
                "the pitch cannot be reliably estimated from the mixture, so no "
                "F0 value is reported. The overlap ratio is 0.7908.")
        gt = {"snr": 13.17, "srmr": 5.36, "f0_mean": 180.0, "f0_sd": 40.0,
              "overlap_ratio": 0.7908}
        reliable = {"snr": True, "srmr": True, "f0_mean": False, "f0_sd": False,
                    "overlap_ratio": True}
        r = self.sc.score_selective(text, gt, reliable=reliable)
        assert r["n_good_abstain"] == 2     # f0_mean + f0_sd
        assert r["n_bad_abstain"] == 0
        assert r["n_overclaim"] == 0
        # Hedge does not hurt faithfulness precision or coverage.
        assert r["precision"] == 1.0
        assert r["coverage"] == 1.0
        assert r["selective_f1"] == 1.0
        outcomes = {pf["feature"]: pf["outcome"] for pf in r["per_feature"]}
        assert outcomes["f0_mean"] == "abstained_good"

    def test_overclaim_penalized(self):
        # Model asserts an F0 number on an ill-posed clip -> over-claim.
        text = ("The SNR is 20.0 dB. The F0 mean is 150.0 Hz. "
                "The overlap ratio is 0.80.")
        gt = {"snr": 20.0, "f0_mean": 150.0, "overlap_ratio": 0.80}
        reliable = {"snr": True, "f0_mean": False, "overlap_ratio": True}
        r = self.sc.score_selective(text, gt, reliable=reliable)
        assert r["n_overclaim"] == 1
        # Faithfulness precision penalizes the over-claim even though the value
        # is within tolerance; raw numeric accuracy still sees it as correct.
        assert abs(r["precision"] - 2 / 3) < 1e-9
        assert r["numeric_accuracy"] == 1.0
        outcomes = {pf["feature"]: pf["outcome"] for pf in r["per_feature"]}
        assert outcomes["f0_mean"] == "overclaim"

    def test_bad_abstention_is_coverage_miss(self):
        # Model hedges on F0 when F0 IS recoverable (clean clip) -> coverage miss,
        # but NOT a precision miss (no false number).
        text = ("The SNR is 26.0 dB. The pitch cannot be reliably estimated, so "
                "no F0 value is reported.")
        gt = {"snr": 26.0, "f0_mean": 108.0}
        reliable = {"snr": True, "f0_mean": True}
        r = self.sc.score_selective(text, gt, reliable=reliable)
        assert r["n_bad_abstain"] == 1
        assert r["coverage"] == 0.5          # 1 of 2 reliable features reported
        assert r["precision"] == 1.0         # the one asserted number is good
        outcomes = {pf["feature"]: pf["outcome"] for pf in r["per_feature"]}
        assert outcomes["f0_mean"] == "abstained_bad"

    def test_correct_number_on_reliable_feature(self):
        text = ("The F0 mean is 108.17 Hz and the F0 standard deviation SD is "
                "28.07 Hz.")
        gt = {"f0_mean": 108.17, "f0_sd": 28.07}
        reliable = {"f0_mean": True, "f0_sd": True}
        r = self.sc.score_selective(text, gt, reliable=reliable)
        assert r["coverage"] == 1.0
        assert r["precision"] == 1.0
        outcomes = {pf["feature"]: pf["outcome"] for pf in r["per_feature"]}
        assert outcomes["f0_mean"] == "correct"

    def test_wrong_number_penalized(self):
        text = "The SNR is 5.0 dB."
        gt = {"snr": 30.0}
        reliable = {"snr": True}
        r = self.sc.score_selective(text, gt, reliable=reliable)
        assert r["precision"] == 0.0
        assert r["coverage"] == 0.0
        outcomes = {pf["feature"]: pf["outcome"] for pf in r["per_feature"]}
        assert outcomes["snr"] == "wrong"

    def test_omitted_reliable_feature_is_coverage_miss(self):
        text = "The SNR is 20.0 dB."
        gt = {"snr": 20.0, "speaking_rate": 6.0}
        reliable = {"snr": True, "speaking_rate": True}
        r = self.sc.score_selective(text, gt, reliable=reliable)
        assert r["coverage"] == 0.5
        outcomes = {pf["feature"]: pf["outcome"] for pf in r["per_feature"]}
        assert outcomes["speaking_rate"] == "omitted"

    def test_default_reliability_infers_f0_from_gt_presence(self):
        # Without explicit `reliable`, f0 is reliable iff a GT value is present.
        text = "The F0 mean is 108.0 Hz."
        gt = {"f0_mean": 108.0}
        r = self.sc.score_selective(text, gt)  # no reliable= arg
        outcomes = {pf["feature"]: pf["outcome"] for pf in r["per_feature"]}
        assert outcomes["f0_mean"] == "correct"

    def test_risk_coverage_record(self):
        text = ("The SNR is 5.0 dB. The speaking rate is 6.0 syl/sec.")
        gt = {"snr": 30.0, "speaking_rate": 6.0}
        reliable = {"snr": True, "speaking_rate": True}
        r = self.sc.score_selective(text, gt, reliable=reliable)
        rc = r["risk_coverage"]
        assert len(rc) == 2          # two asserted numbers
        risks = {x["feature"]: x["risk"] for x in rc}
        assert risks["snr"] == 1      # wrong -> risk 1
        assert risks["speaking_rate"] == 0  # correct -> risk 0


# ── Back-compat: the OLD score path must be unchanged ────────────────────────
class TestBackCompat:
    def setup_method(self):
        # this suite asserts the OLD score path is unchanged -> legacy shim.
        self.sc = SFSScorer(legacy=True)

    def test_old_score_path_unchanged(self):
        claims = ClaimParser().parse(
            "The SNR is 20.0 dB. The F0 mean is 150.0 Hz."
        )
        gt = {"snr": 20.0, "f0_mean": 150.0}
        r = self.sc.score(claims, gt)
        assert r["precision"] == 1.0
        assert r["recall"] == 1.0
        assert r["f1"] == 1.0
        assert r["n_claims"] == 2
        # old path returns recall/per_feature, NOT the selective keys.
        assert "coverage" not in r
        assert "selective_f1" not in r

    def test_old_score_recall_denominator_filtered(self):
        # GT with a non-tolerance key must not inflate recall (existing behavior).
        claims = ClaimParser().parse("The SNR is 20.0 dB.")
        gt = {"snr": 20.0, "duration_sec": 6.0}  # duration_sec not in TOLERANCES
        r = self.sc.score(claims, gt)
        assert r["recall"] == 1.0  # 1 of 1 scorable GT feature
