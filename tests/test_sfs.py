"""Tests for Signal Faithfulness Score (SFS) module."""

from sfs import Claim, ClaimParser, HybridClaimParser, SFSScorer, TaggedClaimParser


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

    def test_jitter_parenthetical(self):
        claims = self.parser.parse("jitter (2.28%)")
        assert len(claims) == 1
        assert claims[0].feature == "jitter"
        assert claims[0].value == 2.28

    def test_shimmer_parenthetical(self):
        claims = self.parser.parse("shimmer (10.22%)")
        assert len(claims) == 1
        assert claims[0].feature == "shimmer"
        assert claims[0].value == 10.22

    def test_srmr(self):
        claims = self.parser.parse("SRMR of 4.0478")
        assert len(claims) == 1
        assert claims[0].feature == "srmr"
        assert claims[0].value == 4.0478

    def test_real_description(self):
        desc = "The recording quality is moderate, with an SNR of 26.15 dB and a low SRMR of 4.0478. Voice characteristics are stable, with a good HNR of 10.97 dB, though the jitter (2.28%) and shimmer (10.22%) suggest minor vocal instability. speaking rate of 4.312 syl/sec."
        claims = self.parser.parse(desc)
        features = {c.feature for c in claims}
        assert "snr" in features
        assert "srmr" in features
        assert "hnr" in features
        assert "jitter" in features
        assert "shimmer" in features
        assert "speaking_rate" in features


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


class TestTaggedClaimParser:
    """The tagged parser sees `<f_NAME>…</f>` spans and yields one Claim per
    SFS-scored tag (with overlap_segments expanded to start/end pairs)."""

    def setup_method(self):
        self.parser = TaggedClaimParser()

    def test_single_span(self):
        claims = self.parser.parse("<f_snr>The SNR is 15.10 dB</f>")
        assert len(claims) == 1
        assert claims[0].feature == "snr"
        assert claims[0].value == 15.10

    def test_multiple_spans_in_order(self):
        text = (
            "<f_snr>the SNR is 15.10 dB</f> and "
            "<f_srmr>the SRMR is 5.17</f>. "
            "<f_pause_count>3 pauses</f>."
        )
        claims = self.parser.parse(text)
        features = [c.feature for c in claims]
        assert features == ["snr", "srmr", "pause_count"]

    def test_overlap_segments_expanded_to_pairs(self):
        text = "<f_overlap_segments>Overlap segments are present at 0.5-3.1s, 5.4-7.2s</f>"
        claims = self.parser.parse(text)
        # Two ranges → two starts + two ends
        starts = [c for c in claims if c.feature == "overlap_start"]
        ends = [c for c in claims if c.feature == "overlap_end"]
        assert [c.value for c in starts] == [0.5, 5.4]
        assert [c.value for c in ends] == [3.1, 7.2]

    def test_non_sfs_tag_skipped(self):
        # overlap_segments has sfs_key=None (scored via IoU, separately).
        # The TaggedClaimParser expands it into overlap_start/overlap_end claim
        # pairs, NOT a generic feature claim. So the parser yields overlap claims,
        # not nothing — exercise that path explicitly.
        text = "<f_overlap_segments>Overlap segments at 0.5-1.0s</f>"
        claims = self.parser.parse(text)
        # Should produce overlap_start + overlap_end (NOT a generic feature claim).
        feats = {c.feature for c in claims}
        assert feats == {"overlap_start", "overlap_end"}

    def test_malformed_unmatched_close_skipped(self):
        # Open tag without close → no span produced
        claims = self.parser.parse("<f_snr>The SNR is 15.10 dB")
        assert claims == []

    def test_full_paragraph_round_trips_to_scorer(self):
        """End-to-end: tagged text → parser → SFSScorer with within-tolerance GT."""
        text = (
            "<f_snr>The SNR is 15.10 dB</f> and "
            "<f_srmr>the SRMR is 5.17</f>. "
            "<f_overlap_segments>Overlap segments are present at 0.5-3.1s, 5.4-7.2s</f>."
        )
        claims = self.parser.parse(text)
        scorer = SFSScorer()
        gt = {
            "snr": 15.0,            # within ±2 dB
            "srmr": 5.0,            # within ±0.5
            "overlap_segments": [(0.5, 3.1), (5.4, 7.2)],
        }
        result = scorer.score(claims, gt)
        assert result["precision"] == 1.0
        assert result["recall"] == 1.0


class TestHybridClaimParser:
    """Hybrid prefers tagged parsing; falls back to the legacy regex parser
    on untagged text so old (Phase-2) outputs still score."""

    def setup_method(self):
        self.parser = HybridClaimParser()

    def test_tagged_takes_precedence(self):
        text = "<f_snr>The SNR is 15.10 dB</f>"
        claims = self.parser.parse(text)
        assert len(claims) == 1 and claims[0].feature == "snr"

    def test_falls_back_to_regex_on_untagged(self):
        text = "The SNR is 28 dB and the duration is 4.2 s."
        claims = self.parser.parse(text)
        features = {c.feature for c in claims}
        assert "snr" in features
        assert "duration_sec" in features
