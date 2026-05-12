"""Tests for the feature_tags module — tag rendering, span iteration, and
value extraction. These are the helpers the verbalizer, the tagged SFS parser,
and the eventual explainability code all depend on, so the contract here is
load-bearing."""

import pytest

from feature_tags import (
    CLOSE_TAG,
    FEATURE_TAGS,
    SPECIAL_TOKENS,
    TAG_BY_NAME,
    TAG_BY_TOKEN,
    build_cover_lines,
    extract_overlap_segments,
    extract_value,
    iter_tagged_spans,
    render_tagged_span,
    strip_tags,
)


class TestModuleConstants:
    def test_special_tokens_include_features_plus_sections_plus_shared_closes(self):
        # SPECIAL_TOKENS now covers both layers: section opens + </sec> + feature opens + </f>.
        # Catalog: 9 feature opens, 6 section opens, 2 shared closes = 17.
        from section_tags import SECTION_TAGS
        n_sections = len(SECTION_TAGS)
        n_features = len(FEATURE_TAGS)
        # 2 shared closes (</sec>, </f>)
        assert len(SPECIAL_TOKENS) == n_sections + n_features + 2
        assert CLOSE_TAG in SPECIAL_TOKENS

    def test_catalog_has_9_tags(self):
        # 2026-05-12 EMNLP rework: 8 scalar features + overlap_segments span set,
        # distributed across the 6 sections in src/section_tags.py.
        # If this count changes, double-check the section catalog still maps cleanly.
        assert len(FEATURE_TAGS) == 9

    def test_catalog_includes_expected_features(self):
        names = {ft.name for ft in FEATURE_TAGS}
        expected = {"snr", "srmr", "f0_mean", "f0_sd",
                    "speaking_rate", "pause_count", "pause_rate",
                    "overlap_ratio", "overlap_segments"}
        assert names == expected

    def test_dropped_tags_not_present(self):
        # Tags excluded on 2026-05-12 when realigning to the section catalog.
        # If any reappear, the paper's "one figure per quality dimension"
        # claim breaks because their attention story collides with a kept one.
        dropped = {"duration", "sample_rate", "silence_ratio",
                   "hnr", "jitter", "shimmer", "articulation_rate"}
        names = {ft.name for ft in FEATURE_TAGS}
        assert dropped.isdisjoint(names)

    def test_tag_names_unique(self):
        names = [ft.name for ft in FEATURE_TAGS]
        assert len(names) == len(set(names))

    def test_lookup_maps_match_table(self):
        for ft in FEATURE_TAGS:
            assert TAG_BY_NAME[ft.name] is ft
            assert TAG_BY_TOKEN[ft.tag] is ft

    def test_open_tags_follow_naming_convention(self):
        for ft in FEATURE_TAGS:
            assert ft.tag.startswith("<f_") and ft.tag.endswith(">")


class TestRenderTaggedSpan:
    def test_round_trip_snr(self):
        out = render_tagged_span("snr", "The SNR is 15.1 dB")
        assert out == "<f_snr>The SNR is 15.1 dB</f>"

    def test_unknown_name_raises(self):
        with pytest.raises(KeyError):
            render_tagged_span("not_a_feature", "body")


class TestIterTaggedSpans:
    def test_single_span(self):
        spans = list(iter_tagged_spans("<f_snr>x</f>"))
        assert len(spans) == 1
        assert spans[0].feature.name == "snr"
        assert spans[0].body == "x"

    def test_multiple_spans_in_order(self):
        text = "<f_snr>a</f> and <f_srmr>b</f>."
        names = [s.feature.name for s in iter_tagged_spans(text)]
        assert names == ["snr", "srmr"]

    def test_char_spans_locate_outer_brackets(self):
        text = "prefix <f_snr>body</f> suffix"
        span = next(iter_tagged_spans(text))
        assert text[span.char_start:span.char_end] == "<f_snr>body</f>"

    def test_no_spans_returns_empty(self):
        assert list(iter_tagged_spans("no tags here")) == []

    def test_unmatched_open_yields_nothing(self):
        # No </f> close → no well-formed span.
        assert list(iter_tagged_spans("<f_snr>dangling")) == []


class TestExtractValue:
    def test_extracts_first_number(self):
        assert extract_value("The SNR is 15.10 dB") == 15.10

    def test_handles_integer(self):
        assert extract_value("The pause count is 3") == 3.0

    def test_handles_zero(self):
        assert extract_value("the silence ratio is 0.0") == 0.0

    def test_no_number_returns_none(self):
        assert extract_value("no digits in this body") is None


class TestExtractOverlapSegments:
    def test_single_range(self):
        assert extract_overlap_segments("present at 0.5-3.1s") == [(0.5, 3.1)]

    def test_multiple_ranges(self):
        out = extract_overlap_segments("at 0.5-3.1s, 5.4-7.2s, and 9.0-10.0s")
        assert out == [(0.5, 3.1), (5.4, 7.2), (9.0, 10.0)]

    def test_handles_to_phrasing(self):
        # "X to Y s" should also match (the regex accepts both "-" and "to").
        assert extract_overlap_segments("0.5 to 3.1 s") == [(0.5, 3.1)]

    def test_skips_inverted_range(self):
        # end <= start is dropped silently.
        assert extract_overlap_segments("5.0-3.0s") == []


class TestStripTags:
    def test_removes_all_tags(self):
        text = "<f_snr>The SNR is 15.10 dB</f> and <f_srmr>SRMR is 5.17</f>."
        assert strip_tags(text) == "The SNR is 15.10 dB and SRMR is 5.17."

    def test_strip_removes_section_tags_too(self):
        text = "<sec_noise>noise body</sec> <sec_overlap>overlap body</sec>"
        assert strip_tags(text) == "noise body overlap body"

    def test_passthrough_no_tags(self):
        assert strip_tags("plain text") == "plain text"


class TestBuildCoverLines:
    def test_only_present_features_emit_lines(self):
        lines = build_cover_lines({"snr": "15.10", "srmr": "5.17"})
        assert len(lines) == 2
        assert any("<f_snr>" in l for l in lines)
        assert any("<f_srmr>" in l for l in lines)

    def test_skips_missing(self):
        lines = build_cover_lines({})
        assert lines == []
