"""Tests for Signal Faithfulness Score (SFS) module."""

from sfs import Claim, ClaimParser, SFSScorer


class TestClaimParser:
    def setup_method(self):
        self.parser = ClaimParser()

    def test_f0_equals(self):
        claims = self.parser.parse("F0 = 187 Hz")
        assert len(claims) == 1
        assert claims[0].feature == "f0_mean"
        assert claims[0].value == 187.0

    def test_f0_with_std(self):
        claims = self.parser.parse("F0 = 187 Hz (σ = 34 Hz)")
        features = {c.feature for c in claims}
        assert "f0_mean" in features
        assert "f0_std" in features

    def test_f0_approx(self):
        claims = self.parser.parse("pitch of approximately 200 Hz")
        assert len(claims) == 1
        assert claims[0].feature == "f0_mean"
        assert claims[0].value == 200.0

    def test_snr(self):
        claims = self.parser.parse("SNR ≈ 28 dB")
        assert len(claims) == 1
        assert claims[0].feature == "snr"
        assert claims[0].value == 28.0

    def test_snr_of_approximately(self):
        claims = self.parser.parse("SNR of approximately 28 dB")
        assert len(claims) == 1
        assert claims[0].feature == "snr"

    def test_rt60(self):
        claims = self.parser.parse("RT60 < 0.15s")
        assert len(claims) == 1
        assert claims[0].feature == "rt60"
        assert claims[0].value == 0.15

    def test_speaking_rate(self):
        claims = self.parser.parse("speaking rate: 7 syl/s")
        assert len(claims) == 1
        assert claims[0].feature == "speaking_rate"
        assert claims[0].value == 7.0

    def test_overlap_span(self):
        claims = self.parser.parse("overlap at 2.3-4.1s")
        features = {c.feature for c in claims}
        assert "overlap_start" in features
        assert "overlap_end" in features
        start = next(c for c in claims if c.feature == "overlap_start")
        end = next(c for c in claims if c.feature == "overlap_end")
        assert start.value == 2.3
        assert end.value == 4.1

    def test_overlapping_speech_from(self):
        claims = self.parser.parse("overlapping speech from 1.5 to 3.2s")
        features = {c.feature for c in claims}
        assert "overlap_start" in features
        assert "overlap_end" in features

    def test_formant(self):
        claims = self.parser.parse("F1 is 542 Hz")
        assert len(claims) == 1
        assert claims[0].feature == "f1_mean"
        assert claims[0].value == 542.0

    def test_jitter(self):
        claims = self.parser.parse("jitter = 1.2%")
        assert len(claims) == 1
        assert claims[0].feature == "jitter"
        assert claims[0].value == 1.2

    def test_shimmer(self):
        claims = self.parser.parse("shimmer = 3.5%")
        assert len(claims) == 1
        assert claims[0].feature == "shimmer"
        assert claims[0].value == 3.5

    def test_hnr(self):
        claims = self.parser.parse("HNR = 15 dB")
        assert len(claims) == 1
        assert claims[0].feature == "hnr"
        assert claims[0].value == 15.0

    def test_sample_rate_khz(self):
        claims = self.parser.parse("sampled at 16 kHz")
        assert len(claims) == 1
        assert claims[0].feature == "sample_rate"
        assert claims[0].value == 16000.0

    def test_spectral_tilt(self):
        claims = self.parser.parse("spectral tilt = -3.5 dB/octave")
        assert len(claims) == 1
        assert claims[0].feature == "spectral_tilt"
        assert claims[0].value == -3.5

    def test_vot(self):
        claims = self.parser.parse("VOT = 25 ms")
        assert len(claims) == 1
        assert claims[0].feature == "vot"
        assert claims[0].value == 25.0

    def test_multiple_claims(self):
        text = "F0 = 187 Hz, SNR ≈ 28 dB, speaking rate: 7 syl/s"
        claims = self.parser.parse(text)
        features = {c.feature for c in claims}
        assert features == {"f0_mean", "snr", "speaking_rate"}

    def test_deduplication(self):
        text = "F0 = 187 Hz and also F0 = 190 Hz"
        claims = self.parser.parse(text)
        f0_claims = [c for c in claims if c.feature == "f0_mean"]
        assert len(f0_claims) == 1
        assert f0_claims[0].value == 187.0

    def test_empty_text(self):
        assert self.parser.parse("") == []

    def test_no_claims(self):
        assert self.parser.parse("The audio quality is good.") == []


class TestSFSScorer:
    def setup_method(self):
        self.scorer = SFSScorer()

    def test_perfect_score(self):
        claims = [Claim("f0_mean", 187.0, "Hz", "F0 = 187 Hz")]
        result = self.scorer.score(claims, {"f0_mean": 187.0})
        assert result["precision"] == 1.0
        assert result["recall"] == 1.0
        assert result["f1"] == 1.0

    def test_within_tolerance(self):
        claims = [Claim("f0_mean", 190.0, "Hz", "F0 = 190 Hz")]
        result = self.scorer.score(claims, {"f0_mean": 187.0})
        assert result["precision"] == 1.0

    def test_outside_tolerance(self):
        claims = [Claim("f0_mean", 200.0, "Hz", "F0 = 200 Hz")]
        result = self.scorer.score(claims, {"f0_mean": 187.0})
        assert result["precision"] == 0.0

    def test_recall_missing_feature(self):
        claims = [Claim("f0_mean", 187.0, "Hz", "F0 = 187 Hz")]
        result = self.scorer.score(claims, {"f0_mean": 187.0, "snr": 28.0})
        assert result["recall"] == 0.5

    def test_overlap_iou_perfect(self):
        claims = [
            Claim("overlap_start", 2.0, "s", ""),
            Claim("overlap_end", 4.0, "s", ""),
        ]
        result = self.scorer.score(claims, {"overlap_segments": [(2.0, 4.0)]})
        assert result["precision"] == 1.0

    def test_overlap_iou_below_threshold(self):
        claims = [
            Claim("overlap_start", 2.0, "s", ""),
            Claim("overlap_end", 6.0, "s", ""),
        ]
        result = self.scorer.score(claims, {"overlap_segments": [(3.0, 5.0)]})
        overlap = next(r for r in result["per_feature"] if r["feature"] == "overlap_span")
        assert not overlap["correct"]

    def test_overlap_best_segment_match(self):
        claims = [
            Claim("overlap_start", 10.0, "s", ""),
            Claim("overlap_end", 12.0, "s", ""),
        ]
        result = self.scorer.score(claims, {"overlap_segments": [(1.0, 3.0), (10.0, 12.0)]})
        overlap = next(r for r in result["per_feature"] if r["feature"] == "overlap_span")
        assert overlap["correct"]

    def test_no_claims_no_gt(self):
        result = self.scorer.score([], {})
        assert result["f1"] == 0.0

    def test_claim_not_in_gt(self):
        claims = [Claim("f0_mean", 187.0, "Hz", "F0 = 187 Hz")]
        result = self.scorer.score(claims, {"snr": 28.0})
        assert result["precision"] == 0.0
