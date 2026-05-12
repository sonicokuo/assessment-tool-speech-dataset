"""Section + feature tag SoT for AQUA-NL's evidence-grounded reasoning design (EMNLP rework, Path 3).

The output of the model is a hierarchy:

    <sec_NAME>
      free prose ... <f_FEATURE>natural-language claim with value</f> ... more prose ...
    </sec>
    <sec_NAME2>
      ...
    </sec>

Two tag layers serve two purposes:
  1. SECTION tags (<sec_*>) anchor cross-attention to the spectrogram encoder. At each
     <sec_*> open in the generated sequence, SectionQueryHead fires: a learnable query
     for that section cross-attends to the spec patches, producing one attention map per
     section per clip. These attention maps are the evidence overlays for the paper figure.

  2. INNER FEATURE tags (<f_*>) anchor SFS metric parsing. Each <f_FEATURE>...</f> span is
     parsed by TaggedClaimParser into (feature, value) pairs which are scored against
     Praat ground truth via SFSScorer.TOLERANCES.

The two layers are independent: section tags don't affect SFS (parser ignores them);
inner feature tags don't affect cross-attention (only section opens trigger the hook).

Catalog (6 sections × 1-2 scalars each + 1 span set):

    <sec_noise>      → snr
    <sec_reverb>     → srmr
    <sec_pitch>      → f0_mean, f0_sd
    <sec_tempo>      → speaking_rate
    <sec_pauses>     → pause_count, pause_rate
    <sec_overlap>    → overlap_ratio, overlap_segments

Trimmed from the broader 22-feature catalog on 2026-05-12 to one figure per quality
dimension. See scratch/EMNLP_REWORK_PLAN_20260511.md.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterator, Sequence


# ── Section tags ──────────────────────────────────────────────────────────
@dataclass(frozen=True)
class SectionTag:
    """One quality-dimension section.

    Attributes:
        name:           short canonical name, also the section_idx key.
        tag:            the open special token (1 vocab token after add_tokens).
        feature_names:  inner <f_*> feature names that appear inside this section
                        (in canonical order). Used by the verbalizer to know which
                        sub-feature spans to weave into the section body.
        display_name:   human-readable label (for verbalizer prompts and figure titles).
    """

    name: str
    tag: str
    feature_names: tuple[str, ...]
    display_name: str


SECTION_CLOSE_TAG: str = "</sec>"


SECTION_TAGS: tuple[SectionTag, ...] = (
    SectionTag(
        name="noise", tag="<sec_noise>",
        feature_names=("snr",),
        display_name="noise",
    ),
    SectionTag(
        name="reverb", tag="<sec_reverb>",
        feature_names=("srmr",),
        display_name="reverberation",
    ),
    SectionTag(
        name="pitch", tag="<sec_pitch>",
        feature_names=("f0_mean", "f0_sd"),
        display_name="pitch",
    ),
    SectionTag(
        name="tempo", tag="<sec_tempo>",
        feature_names=("speaking_rate",),
        display_name="tempo",
    ),
    SectionTag(
        name="pauses", tag="<sec_pauses>",
        feature_names=("pause_count", "pause_rate"),
        display_name="pause structure",
    ),
    SectionTag(
        name="overlap", tag="<sec_overlap>",
        feature_names=("overlap_ratio", "overlap_segments"),
        display_name="speaker overlap",
    ),
)

SECTION_BY_NAME: dict[str, SectionTag] = {s.name: s for s in SECTION_TAGS}
SECTION_BY_TOKEN: dict[str, SectionTag] = {s.tag: s for s in SECTION_TAGS}
SECTION_IDX_BY_NAME: dict[str, int] = {s.name: i for i, s in enumerate(SECTION_TAGS)}
N_SECTIONS: int = len(SECTION_TAGS)  # 6


# ── Inner feature tags ──────────────────────────────────────────────────────────
@dataclass(frozen=True)
class FeatureTag:
    """One numerical-claim feature inside a section.

    Attributes:
        name:          canonical short name (matches src/feature_set.py for the
                       8 SFS-scored scalars + the overlap_segments span set).
        tag:           the open special token.
        csv_col:       column in the features CSV (None for derived features).
        sfs_key:       key in SFSScorer.TOLERANCES (None if not scored — currently only
                       overlap_segments is scored via IoU, separately).
        unit:          display unit string.
        display_name:  human-readable label.
        template:      verbalizer phrasing for the body; `{value}` is the formatted value.
        section:       the parent section's `name`.
    """

    name: str
    tag: str
    csv_col: str | None
    sfs_key: str | None
    unit: str
    display_name: str
    template: str
    section: str


FEATURE_CLOSE_TAG: str = "</f>"


FEATURE_TAGS: tuple[FeatureTag, ...] = (
    FeatureTag(
        name="snr", tag="<f_snr>",
        csv_col="snr_db", sfs_key="snr",
        unit="dB", display_name="signal-to-noise ratio",
        template="the signal-to-noise ratio SNR is {value} dB",
        section="noise",
    ),
    FeatureTag(
        name="srmr", tag="<f_srmr>",
        csv_col="srmr", sfs_key="srmr",
        unit="", display_name="SRMR",
        template="the SRMR is {value}",
        section="reverb",
    ),
    FeatureTag(
        name="f0_mean", tag="<f_f0_mean>",
        csv_col="f0_mean_hz", sfs_key="f0_mean",
        unit="Hz", display_name="F0 mean",
        template="the F0 mean is {value} Hz",
        section="pitch",
    ),
    FeatureTag(
        name="f0_sd", tag="<f_f0_sd>",
        csv_col="f0_sd_hz", sfs_key="f0_sd",
        unit="Hz", display_name="F0 standard deviation",
        template="the F0 standard deviation SD is {value} Hz",
        section="pitch",
    ),
    FeatureTag(
        name="speaking_rate", tag="<f_speaking_rate>",
        csv_col="praat_speaking_rate_syl_sec", sfs_key="speaking_rate",
        unit="syl/sec", display_name="speaking rate",
        template="the speaking rate is {value} syl/sec",
        section="tempo",
    ),
    FeatureTag(
        name="pause_count", tag="<f_pause_count>",
        csv_col="praat_pause_count", sfs_key="pause_count",
        unit="", display_name="pause count",
        template="the pause count is {value}",
        section="pauses",
    ),
    FeatureTag(
        name="pause_rate", tag="<f_pause_rate>",
        csv_col="praat_pause_rate_per_min", sfs_key="pause_rate",
        unit="per min", display_name="pause rate",
        template="the pause rate is {value} per min",
        section="pauses",
    ),
    FeatureTag(
        name="overlap_ratio", tag="<f_overlap_ratio>",
        csv_col="overlap_ratio", sfs_key="overlap_ratio",
        unit="", display_name="overlap ratio",
        template="the overlap ratio is {value}",
        section="overlap",
    ),
    FeatureTag(
        name="overlap_segments", tag="<f_overlap_segments>",
        csv_col="overlap_segments", sfs_key=None,  # span set, scored by IoU separately
        unit="s", display_name="overlap segments",
        template="overlap segments are present at {value}",
        section="overlap",
    ),
)

FEATURE_BY_NAME: dict[str, FeatureTag] = {f.name: f for f in FEATURE_TAGS}
FEATURE_BY_TOKEN: dict[str, FeatureTag] = {f.tag: f for f in FEATURE_TAGS}
N_FEATURE_TAGS: int = len(FEATURE_TAGS)  # 9


# ── Range marker tokens ─────────────────────────────────────────────────────
# Inside multi-value spans (currently only <f_overlap_segments>) each value is
# wrapped in <r>…</r>. This lets the inference-time section-query hook fire
# once per value while keeping the parent span's natural-language flow:
#     "overlap segments at <r>0.5-1.0s</r>, <r>3.0-4.5s</r>, and <r>7.0-9.0s</r>"
# These markers are stripped at display time (strip_all_tags below) so the
# rendered prose reads as plain comma list.
RANGE_OPEN_TAG: str = "<r>"
RANGE_CLOSE_TAG: str = "</r>"


# ── Vocabulary registration ──────────────────────────────────────────────────
# All the tokens we add via tokenizer.add_tokens(..., special_tokens=False).
# `special_tokens=False` is deliberate: we want skip_special_tokens=True to KEEP
# these in the decoded string so SFS can parse them and we can find section
# positions for cross-attention. The same trade-off was already documented for
# the inline-tag design; section tags inherit it.
SPECIAL_TOKENS: list[str] = (
    [s.tag for s in SECTION_TAGS]
    + [SECTION_CLOSE_TAG]
    + [f.tag for f in FEATURE_TAGS]
    + [FEATURE_CLOSE_TAG]
    + [RANGE_OPEN_TAG, RANGE_CLOSE_TAG]
)
# 6 + 1 + 9 + 1 + 2 = 19 new tokens


# ── Section parsing ──────────────────────────────────────────────────────────
_SECTION_OPENS_RE = "|".join(re.escape(s.tag) for s in SECTION_TAGS)
_SECTION_SPAN_RE = re.compile(
    rf"({_SECTION_OPENS_RE})(.*?){re.escape(SECTION_CLOSE_TAG)}",
    re.DOTALL,
)


@dataclass(frozen=True)
class SectionSpan:
    """One <sec_NAME>…</sec> span extracted from generated text."""

    section: SectionTag
    body: str
    char_start: int  # inclusive — at the opening `<` of `<sec_NAME>`
    char_end: int    # exclusive — just past the closing `</sec>`


def iter_section_spans(text: str) -> Iterator[SectionSpan]:
    """Yield every well-formed `<sec_NAME>…</sec>` span in `text`, left-to-right."""
    for m in _SECTION_SPAN_RE.finditer(text):
        tag_literal = m.group(1)
        body = m.group(2)
        sec = SECTION_BY_TOKEN.get(tag_literal)
        if sec is None:
            continue
        yield SectionSpan(
            section=sec, body=body,
            char_start=m.start(), char_end=m.end(),
        )


# ── Feature parsing (inner) ──────────────────────────────────────────────────────────
_FEATURE_OPENS_RE = "|".join(re.escape(f.tag) for f in FEATURE_TAGS)
_FEATURE_SPAN_RE = re.compile(
    rf"({_FEATURE_OPENS_RE})(.*?){re.escape(FEATURE_CLOSE_TAG)}",
    re.DOTALL,
)


@dataclass(frozen=True)
class FeatureSpan:
    """One inner `<f_NAME>…</f>` span extracted from generated text."""

    feature: FeatureTag
    body: str
    char_start: int
    char_end: int


def iter_feature_spans(text: str) -> Iterator[FeatureSpan]:
    """Yield every well-formed `<f_NAME>…</f>` span in `text`, left-to-right."""
    for m in _FEATURE_SPAN_RE.finditer(text):
        tag_literal = m.group(1)
        body = m.group(2)
        ft = FEATURE_BY_TOKEN.get(tag_literal)
        if ft is None:
            continue
        yield FeatureSpan(
            feature=ft, body=body,
            char_start=m.start(), char_end=m.end(),
        )


# ── Value extraction helpers ──────────────────────────────────────────────
# Match a decimal number not glued to a letter on its left ("F0" → don't grab the 0).
# Negative lookbehind for a letter; positive lookahead for end-of-string or non-digit.
_NUMBER_RE = re.compile(r"(?<![A-Za-z])(-?\d+\.?\d*)")
_RANGE_RE = re.compile(r"(\d+\.?\d*)\s*(?:-|to)\s*(\d+\.?\d*)\s*s")


def extract_value(body: str) -> float | None:
    """First decimal value found in `body`, or None."""
    m = _NUMBER_RE.search(body)
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


def extract_overlap_segments(body: str) -> list[tuple[float, float]]:
    """Parse `X-Ys` ranges from an `<f_overlap_segments>` body."""
    out: list[tuple[float, float]] = []
    for m in _RANGE_RE.finditer(body):
        try:
            s, e = float(m.group(1)), float(m.group(2))
        except ValueError:
            continue
        if e > s:
            out.append((s, e))
    return out


# ── Render helpers ──────────────────────────────────────────────────────────
def render_feature_span(name: str, body: str) -> str:
    """Wrap `body` in the open/close tag for the named inner feature."""
    if name not in FEATURE_BY_NAME:
        raise KeyError(f"Unknown feature '{name}'. Known: {sorted(FEATURE_BY_NAME)}")
    return f"{FEATURE_BY_NAME[name].tag}{body}{FEATURE_CLOSE_TAG}"


def render_section_span(name: str, body: str) -> str:
    """Wrap `body` in the open/close tag for the named section."""
    if name not in SECTION_BY_NAME:
        raise KeyError(f"Unknown section '{name}'. Known: {sorted(SECTION_BY_NAME)}")
    return f"{SECTION_BY_NAME[name].tag}{body}{SECTION_CLOSE_TAG}"


def strip_all_tags(text: str) -> str:
    """Remove every section + feature tag from `text`.

    Used by BLEU/ROUGE/BERTScore so text-similarity isn't inflated by the
    model's ability to copy tags. SFS uses the structured spans separately.
    """
    parts = SPECIAL_TOKENS
    pattern = re.compile("|".join(re.escape(p) for p in parts))
    return pattern.sub("", text)


# ── Verbalizer helpers ──────────────────────────────────────────────────────
def build_section_body(section_name: str, feature_values: dict[str, str]) -> str:
    """Build the body of one section by concatenating its inner <f_*> spans.

    Args:
        section_name: e.g. "pitch"
        feature_values: dict mapping inner-feature short_name → formatted value
                        (e.g. {"f0_mean": "152.46", "f0_sd": "38.10"})

    Returns:
        A string like "<f_f0_mean>the F0 mean is 152.46 Hz</f> and "
                      "<f_f0_sd>the F0 standard deviation SD is 38.10 Hz</f>"
        suitable for splicing into a <sec_pitch>…</sec> wrapper.

    Missing features (not in `feature_values`) are skipped; the verbalizer is told
    in its prompt not to invent values.
    """
    sec = SECTION_BY_NAME[section_name]
    spans: list[str] = []
    for fname in sec.feature_names:
        if fname not in feature_values:
            continue
        ft = FEATURE_BY_NAME[fname]
        body = ft.template.format(value=feature_values[fname])
        spans.append(render_feature_span(fname, body))
    return " and ".join(spans) if spans else ""


# ── Convenience for the model side ──────────────────────────────────────────
def section_open_token_ids(tokenizer) -> dict[str, int]:
    """Return a dict mapping section name → token id.

    Used by inference.py to detect when a section open has just been generated.
    Call AFTER tokenizer.add_tokens(SPECIAL_TOKENS, ...).
    """
    return {s.name: tokenizer.convert_tokens_to_ids(s.tag) for s in SECTION_TAGS}


def section_name_by_token_id(tokenizer) -> dict[int, str]:
    """Inverse of section_open_token_ids — token id → section name."""
    return {tokenizer.convert_tokens_to_ids(s.tag): s.name for s in SECTION_TAGS}
