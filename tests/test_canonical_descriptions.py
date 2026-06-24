"""CPU unit tests for scripts/build_canonical_descriptions.py.

The canonical builder co-designs its output with the SFS HybridClaimParser:
every REPORTED feature must round-trip through the parser, abstained features
must be ABSENT (no number) under heavy overlap and PRESENT under low overlap,
the clause order is fixed, the output is deterministic given the filename, and
no value is fabricated when GT is missing.

Run (from repo root, with src/ + scripts/ on the path via conftest + the
sys.path insert below):
    python -m pytest tests/test_canonical_descriptions.py -v
"""
from __future__ import annotations

import os
import sys

# conftest.py adds src/ to the path; add scripts/ for the builder under test.
sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts")
)

import pytest  # noqa: E402

from sfs import AbstentionDetector, HybridClaimParser  # noqa: E402

from build_canonical_descriptions import (  # noqa: E402
    ABSTAIN_UNDER_OVERLAP,
    ALWAYS_REPORT,
    FEATURE_ORDER,
    build_canonical_description,
    _stable_choice,
)


# ── Fixtures: a low-overlap row (everything observable) and a high-overlap row ──
# Values cover the canonical NUMBER FORMATTING precisions so we can assert
# round-trip equality after formatting.
def _clean_features_entry():
    return {
        "snr_db": 15.63,
        "srmr": 5.16,
        "praat_speaking_rate_syl_sec": 6.311,
        "praat_articulation_rate_syl_sec": 5.2,
        "praat_pause_count": 3,
        "praat_pause_rate_per_min": 18.5,
    }


def _clean_f0_entry():
    return {"f0_mean_hz": 202.89, "f0_sd_hz": 10.74, "clean_voiced_frac": 0.5}


def _row(overlap_vad, fname="spk_clip.wav"):
    """A CSV row dict. hnr / jitter / shimmer / overlap_ratio come from here."""
    return {
        "filename": fname,
        "overlap_ratio": str(overlap_vad),
        "overlap_ratio_vad": str(overlap_vad),
        "hnr_db": "12.40",
        "jitter_local_pct": "1.23",
        "shimmer_pct": "4.56",
        # CSV fallbacks (clean_features should win over these when present):
        "snr_db": "99.99",
        "srmr": "0.0001",
    }


LOW_OVERLAP = 0.12   # < 0.5 threshold -> nothing abstained
HIGH_OVERLAP = 0.8162  # >= 0.5 threshold -> pitch / voice-quality abstained

# The 12 canonical features and the GT value each clause should carry when fully
# observable (clean GT preferred; hnr/jitter/shimmer/overlap from the CSV).
EXPECTED_VALUES = {
    "snr": 15.63,
    "srmr": 5.16,
    "hnr": 12.40,
    "f0_mean": 202.89,
    "f0_sd": 10.74,
    "jitter": 1.23,
    "shimmer": 4.56,
    "speaking_rate": 6.311,
    "articulation_rate": 5.2,
    "pause_count": 3,
    "pause_rate": 18.5,
    "overlap_ratio": 0.8162,  # overridden per-test from the row
}


def _build(overlap, fname="spk_clip.wav", threshold=0.5,
           clean_features=True, clean_f0=True):
    row = _row(overlap, fname)
    cf = {fname: _clean_features_entry()} if clean_features else None
    cf0 = {fname: _clean_f0_entry()} if clean_f0 else None
    stem = os.path.splitext(fname)[0]
    return build_canonical_description(
        row, stem, fname, cf, cf0, overlap_threshold=threshold,
    )


# ── 1. Every REPORTED feature round-trips through HybridClaimParser ──────────
def test_all_twelve_roundtrip_low_overlap():
    text = _build(LOW_OVERLAP)
    claims = HybridClaimParser().parse(text)
    got = {c.feature: c.value for c in claims}

    # All 12 canonical features present.
    for short_name, *_ in FEATURE_ORDER:
        assert short_name in got, f"{short_name} missing from parsed claims: {got}"

    # Values round-trip within formatting precision.
    expected = dict(EXPECTED_VALUES)
    expected["overlap_ratio"] = LOW_OVERLAP
    for feat, val in expected.items():
        assert abs(got[feat] - val) < 1e-4, (
            f"{feat}: parsed {got[feat]} != expected {val}"
        )


def test_each_clause_parses_in_isolation():
    """Defensive: every canonical clause parses on its own too (not relying on
    sentence context)."""
    parser = HybridClaimParser()
    text = _build(LOW_OVERLAP)
    # Split into sentences and confirm each feature-bearing sentence yields >=1 claim.
    n_feature_sentences = 0
    for sent in text.split(". "):
        claims = parser.parse(sent)
        # Intro / hedge sentences yield 0 claims; that's fine.
        if claims:
            n_feature_sentences += 1
    # All 12 features are reported, so we expect a healthy number of claim-bearing
    # sentences (each canonical clause is its own sentence).
    assert n_feature_sentences >= 12


# ── 2. Abstention: abstained features ABSENT under high overlap ──────────────
def test_abstained_features_absent_under_high_overlap():
    text = _build(HIGH_OVERLAP)
    claims = HybridClaimParser().parse(text)
    got = {c.feature for c in claims}

    # None of the abstain-under-overlap features carry a number.
    for feat in ABSTAIN_UNDER_OVERLAP:
        assert feat not in got, (
            f"abstained feature {feat} should NOT have a parsed number; got {got}"
        )

    # All ALWAYS-REPORT features are still present with correct values.
    parsed = {c.feature: c.value for c in claims}
    for feat in ALWAYS_REPORT:
        assert feat in parsed, f"always-report feature {feat} missing under overlap"
    assert abs(parsed["snr"] - 15.63) < 1e-4
    assert abs(parsed["overlap_ratio"] - HIGH_OVERLAP) < 1e-4


def test_abstention_detector_fires_under_overlap():
    """The grouped hedge must be recognized by the SFS AbstentionDetector as a
    calibrated pitch abstention."""
    text = _build(HIGH_OVERLAP)
    abstained = AbstentionDetector().detect(text)
    assert "f0_mean" in abstained
    assert "f0_sd" in abstained


def test_abstained_features_present_under_low_overlap():
    text = _build(LOW_OVERLAP)
    claims = HybridClaimParser().parse(text)
    got = {c.feature for c in claims}
    for feat in ABSTAIN_UNDER_OVERLAP:
        assert feat in got, (
            f"under low overlap, {feat} should be REPORTED as a number; got {got}"
        )
    # And no hedge fires when everything is observable.
    assert AbstentionDetector().detect(text) == set()


def test_single_grouped_hedge_sentence():
    """All abstained features share ONE hedge sentence (not five)."""
    text = _build(HIGH_OVERLAP)
    # The hedge sentence names F0, jitter, shimmer, HNR together.
    assert text.count("cannot be reliably estimated") == 1
    assert "(F0, jitter, shimmer, HNR)" in text


# ── 3. Fixed slot ORDER ──────────────────────────────────────────────────────
def test_fixed_clause_order_low_overlap():
    text = _build(LOW_OVERLAP).lower()  # connective glue lowercases a leading "The"
    # The canonical anchor phrases, in fixed order.
    anchors = [
        "the snr is",
        "the srmr is",
        "the hnr is",
        "the f0 mean is",
        "the f0 standard deviation sd is",
        "the jitter is",
        "the shimmer is",
        "the speaking rate is",
        "the articulation rate is",
        "the pause count is",
        "the pause rate is",
        "the overlap ratio is",
    ]
    positions = [text.find(a) for a in anchors]
    assert all(pos >= 0 for pos in positions), f"missing anchor in: {text}"
    assert positions == sorted(positions), (
        f"clauses out of fixed order: {list(zip(anchors, positions))}"
    )


def test_order_preserved_under_abstention():
    """Reported features keep fixed relative order; the hedge sits where the
    first abstained feature (f0_mean) would have been."""
    text = _build(HIGH_OVERLAP).lower()  # connective glue may lowercase "The"
    # SRMR (reported) precedes the hedge; speaking rate (reported) follows it.
    p_srmr = text.find("the srmr is")
    p_hedge = text.find("cannot be reliably estimated")
    p_srate = text.find("the speaking rate is")
    p_overlap = text.find("the overlap ratio is")
    assert -1 < p_srmr < p_hedge < p_srate < p_overlap


# ── 4. Determinism given the filename ────────────────────────────────────────
def test_deterministic_same_filename():
    a = _build(LOW_OVERLAP, fname="abc_def.wav")
    b = _build(LOW_OVERLAP, fname="abc_def.wav")
    assert a == b


def test_connective_variety_across_filenames():
    """Different filenames may select different connectives (the variety knob),
    but the FACTS and order are identical across clips with the same values."""
    texts = {_build(LOW_OVERLAP, fname=f"clip_{i}.wav") for i in range(40)}
    # The connective hash should produce at least 2 distinct intros across 40
    # filenames (otherwise variety is broken).
    intros = {t.split(".")[0] for t in texts}
    assert len(intros) >= 2, f"connective variety collapsed: {intros}"


def test_stable_choice_is_pure():
    # No global state: repeated calls identical, range respected.
    for n in (1, 3, 4, 7):
        vals = {_stable_choice("clip_x", "intro", n) for _ in range(5)}
        assert len(vals) == 1
        assert 0 <= next(iter(vals)) < n


# ── 5. No fabricated values when GT missing ──────────────────────────────────
def test_missing_gt_omits_clause_not_fabricated():
    """If clean_f0 has no entry for the clip (and overlap is low so F0 would be
    reported), the F0 clauses are OMITTED, not invented."""
    fname = "no_f0.wav"
    row = _row(LOW_OVERLAP, fname)
    cf = {fname: _clean_features_entry()}
    cf0 = {}  # no F0 entry for this clip
    text = build_canonical_description(
        row, os.path.splitext(fname)[0], fname, cf, cf0, overlap_threshold=0.5,
    )
    claims = {c.feature for c in HybridClaimParser().parse(text)}
    assert "f0_mean" not in claims
    assert "f0_sd" not in claims
    # The recoverable features are still all present.
    for feat in ALWAYS_REPORT:
        assert feat in claims
    # And no F0 phrasing leaked in.
    assert "The F0 mean is" not in text


def test_missing_recoverable_value_omitted():
    """A NaN / empty CSV+clean value omits that one clause; the rest stand."""
    fname = "no_srmr.wav"
    row = _row(LOW_OVERLAP, fname)
    row["srmr"] = "nan"
    cf_entry = _clean_features_entry()
    del cf_entry["srmr"]  # remove clean SRMR too -> truly missing
    cf = {fname: cf_entry}
    cf0 = {fname: _clean_f0_entry()}
    text = build_canonical_description(
        row, os.path.splitext(fname)[0], fname, cf, cf0, overlap_threshold=0.5,
    )
    claims = {c.feature for c in HybridClaimParser().parse(text)}
    assert "srmr" not in claims
    # SNR (the clause before SRMR) and HNR (after) are still present.
    assert "snr" in claims
    assert "hnr" in claims


def test_clean_features_preferred_over_csv():
    """When clean_features has a value it WINS over the (mixture) CSV column."""
    fname = "pref.wav"
    row = _row(LOW_OVERLAP, fname)  # CSV snr_db = "99.99"
    cf = {fname: _clean_features_entry()}  # clean snr_db = 15.63
    cf0 = {fname: _clean_f0_entry()}
    text = build_canonical_description(
        row, os.path.splitext(fname)[0], fname, cf, cf0, overlap_threshold=0.5,
    )
    parsed = {c.feature: c.value for c in HybridClaimParser().parse(text)}
    assert abs(parsed["snr"] - 15.63) < 1e-4  # clean value, not 99.99


# ── 6. Number formatting precision ───────────────────────────────────────────
def test_number_formatting_precision():
    fname = "fmt.wav"
    row = _row(LOW_OVERLAP, fname)
    cf = {fname: _clean_features_entry()}
    cf0 = {fname: _clean_f0_entry()}
    text = build_canonical_description(
        row, os.path.splitext(fname)[0], fname, cf, cf0, overlap_threshold=0.5,
    ).lower()  # connective glue may lowercase a clause-leading "The"
    # srmr {:.4f}, snr {:.2f}, speaking_rate {:.3f}, pause_count {:d}, overlap {:.4f}
    assert "the srmr is 5.1600." in text
    assert "the snr is 15.63 db." in text
    assert "the speaking rate is 6.311 syl/sec." in text
    assert "the pause count is 3." in text
    assert f"the overlap ratio is {LOW_OVERLAP:.4f}." in text


def test_partition_is_exhaustive_and_disjoint():
    """Sanity: the 12 canonical features partition cleanly into abstain / always."""
    all_feats = {short for short, *_ in FEATURE_ORDER}
    assert len(all_feats) == 12
    assert (ABSTAIN_UNDER_OVERLAP | ALWAYS_REPORT) == all_feats
    assert not (ABSTAIN_UNDER_OVERLAP & ALWAYS_REPORT)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
