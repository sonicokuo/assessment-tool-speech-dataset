"""Tests for ClaimParser.parse_hedges (F20) — per-feature hedge detection with
overlap ATTRIBUTION, feeding the observability-gated RL reward (F19).

Contract under test (src/eval/sfs.py, ADD-only):
  - parse_hedges(text) -> {feature: {"attributed": bool}} for the ill-posed
    features only (f0_mean, f0_sd, jitter, shimmer, hnr — a pitch hedge covers
    BOTH pitch scalars, mirroring AbstentionDetector).
  - a hedge is ATTRIBUTED when the SAME SENTENCE names overlap as the cause
    ("during overlap", "overlapping speech", "because ... overlap",
    "due to ... overlap"); overlap mentioned in a DIFFERENT sentence does not
    attribute (sentence-scoped).
  - numeric-claim parsing is UNCHANGED: a hedge and a numeric claim in the same
    text are both extracted.

Torch-free: eval/sfs.py and its import chain are stdlib-only.
"""

from eval.sfs import ClaimParser, HybridClaimParser


PARSER = ClaimParser()


def hedges(text):
    return PARSER.parse_hedges(text)


# ── Attributed hedges (overlap named as cause, same sentence) ─────────────────
class TestAttributedHedges:
    def test_f0_attributed_overlapping_speech(self):
        h = hedges("Due to overlapping speech, the F0 cannot be reliably measured.")
        assert h["f0_mean"] == {"attributed": True}
        assert h["f0_sd"] == {"attributed": True}

    def test_pitch_noun_covers_both_pitch_slots(self):
        h = hedges("The pitch is unreliable during overlap.")
        assert set(h) == {"f0_mean", "f0_sd"}
        assert h["f0_mean"]["attributed"] is True
        assert h["f0_sd"]["attributed"] is True

    def test_jitter_attributed_because_overlap(self):
        h = hedges("Jitter cannot be reliably estimated because the speakers overlap.")
        assert h == {"jitter": {"attributed": True}}

    def test_shimmer_attributed_due_to_overlap(self):
        h = hedges("The shimmer is unreliable due to the heavy overlap.")
        assert h == {"shimmer": {"attributed": True}}

    def test_hnr_attributed_during_overlapping_speech(self):
        h = hedges("The HNR cannot be measured during overlapping speech.")
        assert h == {"hnr": {"attributed": True}}

    def test_fundamental_frequency_noun(self):
        h = hedges(
            "The fundamental frequency is not reliably measurable due to overlap.")
        assert h["f0_mean"]["attributed"] is True
        assert h["f0_sd"]["attributed"] is True

    def test_case_insensitive(self):
        h = hedges("the f0 CANNOT BE RELIABLY MEASURED DUE TO OVERLAP.")
        assert h["f0_mean"]["attributed"] is True

    def test_multi_feature_single_attributed_sentence(self):
        h = hedges(
            "Due to the overlapping speech, the F0, jitter, shimmer and HNR "
            "cannot be reliably measured."
        )
        assert set(h) == {"f0_mean", "f0_sd", "jitter", "shimmer", "hnr"}
        assert all(v["attributed"] for v in h.values())


# ── Generic hedges (no overlap cause in the hedge sentence) ───────────────────
class TestGenericHedges:
    def test_f0_generic_unreliable(self):
        h = hedges("The F0 is unreliable.")
        assert h["f0_mean"] == {"attributed": False}
        assert h["f0_sd"] == {"attributed": False}

    def test_jitter_omitted(self):
        h = hedges("The jitter was omitted.")
        assert h == {"jitter": {"attributed": False}}

    def test_shimmer_not_reliably_measurable(self):
        h = hedges("Shimmer is not reliably measurable.")
        assert h == {"shimmer": {"attributed": False}}

    def test_hnr_could_not_be_estimated(self):
        h = hedges("The HNR could not be estimated.")
        assert h == {"hnr": {"attributed": False}}

    def test_overlap_in_other_sentence_does_not_attribute(self):
        # Sentence-scoped: the overlap mention lives in a DIFFERENT sentence
        # from the hedge, so the hedge stays generic.
        h = hedges("There is heavy overlap in this clip. The F0 is unreliable.")
        assert h["f0_mean"] == {"attributed": False}
        assert h["f0_sd"] == {"attributed": False}


# ── Absent (no hedge at all) ──────────────────────────────────────────────────
class TestAbsentHedges:
    def test_numeric_text_has_no_hedges(self):
        text = ("The SNR is 16.10 dB. The F0 mean is 121.00 Hz. "
                "The jitter is 1.20 percent.")
        assert hedges(text) == {}

    def test_recoverable_feature_hedge_not_tracked(self):
        # SNR is not an ill-posed feature — "SNR is unreliable" is NOT a
        # hedge claim the parser tracks (the reward scores it as an
        # unparseable mention instead).
        assert hedges("The SNR is unreliable.") == {}

    def test_overlap_mention_without_hedge_cue(self):
        assert hedges("The overlap ratio is 0.80.") == {}

    def test_empty_text(self):
        assert hedges("") == {}


# ── Per-feature isolation ─────────────────────────────────────────────────────
class TestPerFeatureIsolation:
    def test_hedging_jitter_does_not_mark_shimmer(self):
        h = hedges("The jitter is unreliable during overlap. "
                   "The shimmer is 5.00 percent.")
        assert "jitter" in h
        assert "shimmer" not in h

    def test_attribution_tracked_per_sentence(self):
        h = hedges("The jitter is unreliable due to overlap. "
                   "The shimmer is unreliable.")
        assert h["jitter"]["attributed"] is True
        assert h["shimmer"]["attributed"] is False

    def test_attributed_wins_over_generic_across_sentences(self):
        # Same feature hedged twice — once generic, once attributed. The
        # attributed flag must OR across sentences.
        h = hedges("The F0 is unreliable. "
                   "Indeed, the F0 cannot be measured due to the overlap.")
        assert h["f0_mean"]["attributed"] is True


# ── Numeric parsing unchanged alongside hedges ────────────────────────────────
class TestNumericParsingUnchanged:
    def test_hedge_plus_numeric_claim_both_extracted(self):
        text = ("Due to the overlapping speech, the F0 cannot be reliably "
                "measured. The SNR is 16.10 dB.")
        h = hedges(text)
        assert h["f0_mean"]["attributed"] is True

        claims = PARSER.parse(text)
        by_feature = {c.feature: c.value for c in claims}
        assert by_feature.get("snr") == 16.10
        # The hedged pitch has NO numeric claim.
        assert "f0_mean" not in by_feature

    def test_hedge_and_numeric_in_same_sentence(self):
        text = ("The F0 cannot be reliably measured due to overlap, "
                "but the SNR is 16.10 dB.")
        h = hedges(text)
        assert h["f0_mean"]["attributed"] is True
        by_feature = {c.feature: c.value for c in PARSER.parse(text)}
        assert by_feature.get("snr") == 16.10

    def test_numeric_value_and_hedge_for_same_feature_coexist(self):
        # Pathological: a number AND a hedge for the same feature. The parser
        # reports both independently; precedence is the REWARD's business.
        text = "The F0 mean is 121.00 Hz, though the F0 is unreliable during overlap."
        h = hedges(text)
        assert h["f0_mean"]["attributed"] is True
        by_feature = {c.feature: c.value for c in PARSER.parse(text)}
        assert by_feature.get("f0_mean") == 121.00


# ── HybridClaimParser passthrough ─────────────────────────────────────────────
class TestHybridParserPassthrough:
    def test_hybrid_delegates_parse_hedges(self):
        text = "The jitter is unreliable due to overlap."
        assert HybridClaimParser().parse_hedges(text) == PARSER.parse_hedges(text)
