"""Compatibility shim — redirects to src/section_tags.py.

The inline-tag design (8 flat <f_*> tags) was an intermediate step on the way
to the EMNLP-rework section-based design (6 <sec_*> tags + 9 <f_*> inner
feature tags). Existing modules (src/sfs.py, src/train.py, src/inference.py,
scripts/feature_verbalization.py, tests/) import symbols by their original
names from this module; this file re-exports them from section_tags so callers
don't need to know about the rename.

New code should import directly from src/section_tags.py.
"""

from section_tags import (  # noqa: F401
    FEATURE_TAGS,
    FEATURE_CLOSE_TAG as CLOSE_TAG,
    FEATURE_BY_NAME as TAG_BY_NAME,
    FEATURE_BY_TOKEN as TAG_BY_TOKEN,
    SPECIAL_TOKENS,
    extract_overlap_segments,
    extract_value,
    iter_feature_spans as iter_tagged_spans,
    render_feature_span as render_tagged_span,
    strip_all_tags as strip_tags,
)


def build_cover_lines(values: dict) -> list[str]:
    """Legacy helper used by the inline-tag verbalizer prompt.

    For the section-based EMNLP design use section_tags.build_section_body instead;
    this function is preserved so the existing inline-tag verbalizer still runs.
    """
    from section_tags import FEATURE_BY_NAME, render_feature_span
    lines: list[str] = []
    for ft in FEATURE_TAGS:
        if ft.name not in values:
            continue
        body = ft.template.format(value=values[ft.name])
        lines.append(render_feature_span(ft.name, body))
    return lines


# Re-export the TaggedSpan dataclass under its old name for any test code
# that uses it directly. Same shape, just under section_tags' FeatureSpan now.
from section_tags import FeatureSpan as TaggedSpan  # noqa: E402, F401
