#!/usr/bin/env python3
"""Build a descriptions.partN.json deterministically from the per-clip feature
CSVs, with no LLM in the loop. Replaces the entire verbalization pipeline
(sanity_and_verbalize.sh -> feature_verbalization.py -> merge -> fix) for the
structural-correctness path.

Why
---
gemma4:e2b (and similar small LLMs), even at temperature=0 with HARD RULES,
paraphrase ~14% of section-tagged spans into untagged plain prose. The lost
tags silently break the section-query attention hook and the SFS parser.
Post-processing cannot recover what the LLM chose not to tag. The same row
data run through `_build_section_bodies` produces 100% structural integrity
in seconds with no GPU.

Overlap-column source-of-truth
------------------------------
This builder PREFERS the VAD-derived columns produced by
`scripts/fix_overlap_csv.py`:

    overlap_segments_vad  (string of '<a>-<b>;<c>-<d>...' in sample indices)
    overlap_ratio_vad     (float)

If those columns are missing on a row it falls back to the original
overlap_segments / overlap_ratio columns (which on Libri2Mix were
pyannote-derived, so the values are circular for GT). The fallback emits one
warning per CSV so it's visible.

The intended design (per CLAUDE.md) is:
  - overlap_info channels for model INPUT  = pyannote-on-mix  (preserved as
    the original overlap_segments / overlap_ratio columns, read by
    src/preprocess.py)
  - description GT for SFS                 = VAD-on-s1/s2     (new
    *_vad columns, read by this builder)

Per-part slicing matches scripts/run_section_verbalization.sh exactly:
    Part 1: train-100[0:4634]     dev[0:1000]      test[0:1000]
    Part 2: train-100[4634:9268]  dev[1000:2000]   test[1000:2000]
    Part 3: train-100[9268:13902] dev[2000:3000]   test[2000:3000]

Usage
-----
    # 1) Part 1 (default)
    python scripts/build_descriptions_deterministic.py --part 1

    # 2) all 19 900 clips in one combined JSON (no part split)
    python scripts/build_descriptions_deterministic.py --all \
        --output $SHARED/data/descriptions_tagged.json

The default features-dir is $SHARED/data/features_pyannote and the default
output is $SHARED/data/descriptions.part{N}.json.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))

from feature_verbalization import _build_section_bodies  # noqa: E402
from section_tags import SECTION_TAGS, render_section_span  # noqa: E402


PART_SLICES = {
    1: {"train-100": (0,    4634),  "dev": (0,    1000), "test": (0,    1000)},
    2: {"train-100": (4634, 9268),  "dev": (1000, 2000), "test": (1000, 2000)},
    3: {"train-100": (9268, 13902), "dev": (2000, 3000), "test": (2000, 3000)},
}


def _clean_overlap_segments(raw: str, dur_sec: float, sr: int = 16000) -> str:
    """Drop fully-OOB segments, clamp partial-OOB, swap reversed. Returns the
    cleaned `start-end;...` string in sample indices.

    Defensive code: with the upstream feature_extractor_mix patch the *_vad
    columns are already clean, but the old overlap_segments column may still
    contain bogus pyannote ranges and we want the fallback path to be safe."""
    raw = (raw or "").strip()
    if not raw:
        return ""
    if dur_sec <= 0:
        return ""
    max_samples = int(dur_sec * sr)
    out = []
    for seg in raw.split(";"):
        seg = seg.strip()
        if not seg:
            continue
        try:
            a_s, b_s = seg.split("-", 1)
            a, b = int(a_s), int(b_s)
        except (ValueError, IndexError):
            continue
        if b < a:
            a, b = b, a
        if a >= max_samples:
            continue
        if b > max_samples:
            b = max_samples
        if b <= a:
            continue
        out.append(f"{a}-{b}")
    return ";".join(out)


def _prepare_row_for_build(row: dict, fallback_warned: dict) -> dict:
    """Return a row dict with overlap_segments / overlap_ratio set to the
    canonical VAD values when available, else the cleaned pyannote values.

    `fallback_warned` is a mutable dict {bool} the caller passes in to receive
    a one-shot 'fell back to pyannote on row X' warning."""
    out = dict(row)
    try:
        dur_sec = float(row.get("duration_sec", "") or 0)
    except (TypeError, ValueError):
        dur_sec = 0.0
    try:
        sr = int(row.get("sample_rate_hz", "16000") or 16000)
    except (TypeError, ValueError):
        sr = 16000

    vad_segs = (row.get("overlap_segments_vad") or "").strip()
    vad_ratio = (row.get("overlap_ratio_vad") or "").strip()
    if vad_segs or vad_ratio:
        out["overlap_segments"] = _clean_overlap_segments(vad_segs, dur_sec, sr)
        if vad_ratio:
            out["overlap_ratio"] = vad_ratio
        out["__used_vad__"] = True
    else:
        # Fall back to pyannote with a defensive clamp.
        pyann_segs = (row.get("overlap_segments") or "").strip()
        out["overlap_segments"] = _clean_overlap_segments(pyann_segs, dur_sec, sr)
        out["__used_vad__"] = False
        if not fallback_warned["warned"]:
            print(
                f"  [warn] overlap_segments_vad missing on row "
                f"'{row.get('filename', '?')}'; falling back to (clamped) "
                f"pyannote-on-mix values. Run scripts/fix_overlap_csv.py "
                f"first for the architecturally correct GT.",
                file=sys.stderr,
            )
            fallback_warned["warned"] = True
    return out


# Matches:
#   opens:  <sec_NAME>, <f_NAME>, <r>
#   closes: </sec>,     </f>,     </r>
# Note the closing tags don't carry the _NAME suffix (the catalog uses one
# shared close per category), so the regex has to handle both forms.
_TAG_STRIP_RE = re.compile(r"<sec_\w+>|</sec>|<f_\w+>|</f>|<r>|</r>")


def _strip_tags(s: str) -> str:
    """Remove every <sec_*>, </sec>, <f_*>, </f>, <r>, </r> tag from s, leaving
    the natural-language content. Used by --untagged to produce a target that
    contains the same factual claims but no special tokens for the model to
    learn."""
    return _TAG_STRIP_RE.sub("", s)


def build_description(row: dict, fallback_warned: dict | None = None,
                      untagged: bool = False) -> str:
    """Compose a description for one CSV row.

    By default emits the tagged section format. With ``untagged=True`` the
    same factual content is emitted as plain prose (every <sec_*>, <f_*>,
    <r> tag stripped), which is the format used for the post-hoc attention
    extraction path — the LM trains on standard prose and per-section
    attention is recovered at inference by parsing the prose for section
    spans and aggregating the LM's native attention over those spans.

    The trailing 'F0 and formant estimates are unreliable during overlap
    windows.' sentence is appended when overlap_ratio > 0.
    """
    fallback_warned = fallback_warned if fallback_warned is not None else {"warned": False}
    row = _prepare_row_for_build(row, fallback_warned)
    bodies = _build_section_bodies(row)
    intro = ""
    dur = row.get("duration_sec")
    if dur not in (None, "", "nan", "NaN"):
        try:
            intro = f"The recording is {float(dur):.3f} s long. "
        except (TypeError, ValueError):
            intro = ""
    sections = []
    for sec in SECTION_TAGS:
        body = bodies.get(sec.name)
        if not body:
            continue
        sections.append(render_section_span(sec.name, body))
    if not intro and not sections:
        return ""
    text = intro + ". ".join(sections) + ("." if sections else "")
    try:
        ratio = float(row.get("overlap_ratio", "") or 0)
    except (TypeError, ValueError):
        ratio = 0.0
    if ratio > 0:
        text += " F0 and formant estimates are unreliable during overlap windows."
    text = text.strip()
    if untagged:
        text = _strip_tags(text)
    return text


def _iter_split(csv_path: Path, start: int, end: int):
    """Yield rows in Python-slice semantics rows[start:end].

    NOTE: PART_SLICES values are (start, end) like CSV[start:end], NOT
    (offset, count). The previous implementation treated the second value
    as a count, so Part 2's (4634, 9268) was iterating rows
    [4634:4634+9268] = [4634:13902] and consuming Part 3's train clips.
    """
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i < start:
                continue
            if i >= end:
                break
            yield row


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    grp = p.add_mutually_exclusive_group(required=True)
    grp.add_argument("--part", type=int, choices=[1, 2, 3])
    grp.add_argument("--all",  action="store_true",
                     help="build a single JSON over every row in every split")
    p.add_argument("--features-dir", type=Path,
                   default=Path(os.environ.get("SHARED",
                                "/ocean/projects/cis260125p/shared"))
                   / "data" / "features_pyannote")
    p.add_argument("--output", type=Path, default=None,
                   help="output JSON (default: $SHARED/data/descriptions.part{N}.json)")
    p.add_argument("--untagged", action="store_true",
                   help="Emit untagged prose (strip every <sec_*>, <f_*>, <r> "
                        "tag from the target). The factual content is "
                        "identical; only the special-token wrappers are "
                        "removed. Use this for the post-hoc attention path "
                        "where the LM trains on standard prose and per-section "
                        "attention is recovered at inference via the LM's "
                        "native attention layers.")
    args = p.parse_args()

    if not args.features_dir.is_dir():
        print(f"ERROR: features dir {args.features_dir} not found", file=sys.stderr)
        return 2

    out_path = args.output or (args.features_dir.parent
                               / (f"descriptions.part{args.part}.json"
                                  if args.part is not None
                                  else "descriptions_tagged.json"))

    if args.all:
        splits = {
            "train-100": (0, 10**9),
            "dev":       (0, 10**9),
            "test":      (0, 10**9),
        }
    else:
        splits = {k: v for k, v in PART_SLICES[args.part].items()}

    fallback_warned = {"warned": False}
    out: dict[str, str] = {}
    n_empty = n_used_vad = n_used_fallback = 0

    for split_name, (start, end) in splits.items():
        csv_path = args.features_dir / f"{split_name}.csv"
        if not csv_path.exists():
            print(f"ERROR: {csv_path} not found", file=sys.stderr)
            return 2
        n_rows = 0
        for row in _iter_split(csv_path, start, end):
            fname = (row.get("filename") or "").strip()
            stem = os.path.splitext(fname)[0]
            if not stem:
                continue
            text = build_description(row, fallback_warned, untagged=args.untagged)
            out[stem] = text
            n_rows += 1
            if not text:
                n_empty += 1
            # peek at the marker the builder stashed onto the row dict; we
            # tagged it on a COPY but the call returns text, so re-derive:
            if (row.get("overlap_segments_vad") or row.get("overlap_ratio_vad")):
                n_used_vad += 1
            else:
                n_used_fallback += 1
        print(f"  {split_name:9s}: {n_rows} rows  (start={start} end={end})")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    tmp.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    tmp.replace(out_path)
    print()
    print(f"wrote {out_path}")
    print(f"  entries           : {len(out)}")
    print(f"  used VAD overlap  : {n_used_vad}")
    print(f"  used pyannote     : {n_used_fallback}")
    print(f"  empty (no usable) : {n_empty}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
