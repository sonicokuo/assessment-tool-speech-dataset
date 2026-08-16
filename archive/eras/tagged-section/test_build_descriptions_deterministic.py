#!/usr/bin/env python3
"""Tests for build_descriptions_deterministic.py.

Run from repo root:
    python scripts/test_build_descriptions_deterministic.py
Exit code 0 = all pass, 1 = any fail.
"""
from __future__ import annotations

import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from build_descriptions_deterministic import (  # noqa: E402
    build_description,
    _clean_overlap_segments,
    _prepare_row_for_build,
    _iter_split,
    PART_SLICES,
)

# Reuse the fix_descriptions audit for "is this a clean span?" checks.
from fix_descriptions import audit, fix_all  # noqa: E402


PASS = FAIL = 0


def expect(label, got, want):
    global PASS, FAIL
    ok = got == want
    PASS += int(ok); FAIL += int(not ok)
    print(f"  {'ok  ' if ok else 'FAIL'} {label}")
    if not ok:
        print(f"       want={want!r}")
        print(f"       got ={got!r}")


def expect_true(label, cond):
    global PASS, FAIL
    PASS += int(bool(cond)); FAIL += int(not cond)
    print(f"  {'ok  ' if cond else 'FAIL'} {label}")


def section(label):
    print(f"\n== {label} ==")


DENSE_PYANNOTE_ROW = {
    "filename": "fake.wav",
    "duration_sec": "3.92",
    "sample_rate_hz": "16000",
    "snr_db": "13.17",
    "srmr": "5.3646",
    "f0_mean_hz": "152.48",
    "f0_sd_hz": "62.88",
    "praat_speaking_rate_syl_sec": "6.888",
    "praat_pause_count": "1",
    "praat_pause_rate_per_min": "15.306",
    "overlap_ratio": "0.3333",
    "overlap_segments": "80000-120000",   # 5.0-7.5s in pyannote (OOB for 3.92s)
}

DENSE_VAD_ROW = dict(DENSE_PYANNOTE_ROW)
DENSE_VAD_ROW["overlap_segments_vad"] = "16000-48000"  # 1.0-3.0s, valid
DENSE_VAD_ROW["overlap_ratio_vad"]    = "0.5102"


def main() -> int:
    print("================ TESTS ================")

    section("_clean_overlap_segments")
    expect("empty stays empty",
           _clean_overlap_segments("", 5.0, 16000), "")
    expect("kept when within bounds",
           _clean_overlap_segments("16000-48000", 5.0, 16000), "16000-48000")
    expect("clamped at end",
           _clean_overlap_segments("16000-96000", 5.0, 16000), "16000-80000")
    expect("dropped when fully OOB",
           _clean_overlap_segments("80000-120000", 3.92, 16000), "")
    expect("reversed swapped",
           _clean_overlap_segments("48000-16000", 5.0, 16000), "16000-48000")
    expect("multi: keep valid, drop OOB",
           _clean_overlap_segments("16000-32000;120000-160000", 5.0, 16000),
           "16000-32000")
    expect("dur<=0 -> empty",
           _clean_overlap_segments("16000-32000", 0.0, 16000), "")

    section("_prepare_row_for_build prefers VAD over pyannote")
    warn = {"warned": False}
    prepared = _prepare_row_for_build(DENSE_VAD_ROW, warn)
    expect("uses VAD segments string",
           prepared["overlap_segments"], "16000-48000")
    expect("uses VAD ratio string",
           prepared["overlap_ratio"], "0.5102")
    expect_true("__used_vad__ set True",  prepared["__used_vad__"] is True)
    expect_true("no fallback warning",    warn["warned"] is False)

    section("_prepare_row_for_build falls back when VAD absent")
    warn = {"warned": False}
    prepared2 = _prepare_row_for_build(DENSE_PYANNOTE_ROW, warn)
    expect("fallback drops OOB pyannote range",
           prepared2["overlap_segments"], "")
    expect("fallback keeps pyannote ratio (no override)",
           prepared2["overlap_ratio"], "0.3333")
    expect_true("__used_vad__ set False", prepared2["__used_vad__"] is False)
    expect_true("fallback warning printed", warn["warned"] is True)

    section("build_description structural integrity (VAD row)")
    out = build_description(DENSE_VAD_ROW)
    expect_true("starts with duration intro",
                out.startswith("The recording is 3.920 s long. "))
    for sec_name in ("noise", "reverb", "pitch", "tempo", "pauses", "overlap"):
        expect_true(f"contains <sec_{sec_name}>", f"<sec_{sec_name}>" in out)
    expect_true("every <sec_*> has a matching </sec>",
                out.count("</sec>") == out.count("<sec_"))
    expect_true("VAD range rendered as <r>1.0-3.0s</r>", "<r>1.0-3.0s</r>" in out)
    expect_true("no OOB pyannote range leaked", "<r>5.0-7.5s</r>" not in out)
    expect_true("trailing F0-unreliable sentence appended",
                "F0 and formant estimates are unreliable" in out)
    expect("audit clean", audit(out), [])
    expect("fix_all idempotent", fix_all(out), out)

    section("build_description from a pyannote-only row drops OOB")
    out2 = build_description(DENSE_PYANNOTE_ROW)
    expect_true("OOB pyannote range dropped",
                "<r>5.0-7.5s</r>" not in out2)
    expect_true("sections still present",
                out2.count("<sec_") == 6)
    # With OOB segment dropped and no other overlap, <sec_overlap> may still
    # appear because overlap_ratio (0.3333) is > 0 — but overlap_segments
    # body should be absent.
    expect_true("overlap_segments span omitted when no valid ranges",
                "<f_overlap_segments>" not in out2)
    expect("audit clean even on fallback", audit(out2), [])

    section("build_description: intro-only when no feature sections")
    just_dur = {"filename": "q.wav", "duration_sec": "1.0"}
    expect("intro-only output",
           build_description(just_dur), "The recording is 1.000 s long.")
    truly_empty = {"filename": "r.wav"}
    expect("nothing at all -> empty string",
           build_description(truly_empty), "")

    section("PART_SLICES match run_section_verbalization.sh")
    expect("part 1", PART_SLICES[1],
           {"train-100": (0,    4634), "dev": (0,    1000), "test": (0,    1000)})
    expect("part 2", PART_SLICES[2],
           {"train-100": (4634, 9268), "dev": (1000, 2000), "test": (1000, 2000)})
    expect("part 3", PART_SLICES[3],
           {"train-100": (9268, 13902), "dev": (2000, 3000), "test": (2000, 3000)})

    section("_iter_split uses Python-slice (start, end) semantics, not (offset, count)")
    import tempfile, csv as _csv
    from pathlib import Path
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False)
    w = _csv.writer(tmp)
    w.writerow(["filename"])
    for i in range(100):
        w.writerow([f"clip_{i:03d}.wav"])
    tmp.close()
    try:
        rows_a = list(_iter_split(Path(tmp.name), 10, 20))
        expect("len rows[10:20] == 10", len(rows_a), 10)
        expect("first row is clip_010", rows_a[0]["filename"], "clip_010.wav")
        expect("last row is clip_019",  rows_a[-1]["filename"], "clip_019.wav")
        # boundary that exposed the original bug: (4634, 9268) yielding 9268 rows
        rows_b = list(_iter_split(Path(tmp.name), 50, 70))
        expect("len rows[50:70] == 20", len(rows_b), 20)
        rows_c = list(_iter_split(Path(tmp.name), 90, 200))
        expect("end past CSV is fine (clamped)", len(rows_c), 10)
    finally:
        os.unlink(tmp.name)

    section("PART_SLICES collectively cover all rows exactly once")
    # The three parts should partition each split (train/dev/test) with no
    # overlap and no gap (modulo train-100's actual length of 13900 not 13902).
    for split_name, csv_len in (("train-100", 13900), ("dev", 3000), ("test", 3000)):
        covered = set()
        overlap_pairs = []
        for part_n in (1, 2, 3):
            start, end = PART_SLICES[part_n][split_name]
            new = set(range(start, min(end, csv_len)))
            if covered & new:
                overlap_pairs.append((part_n, sorted(covered & new)[:3]))
            covered |= new
        expect_true(f"{split_name}: no part overlaps another  (overlaps={overlap_pairs})",
                    not overlap_pairs)
        expect(f"{split_name}: parts cover all {csv_len} rows",
               len(covered), csv_len)

    section("overlap_ratio > 0 but empty segments still trips F0 trailing")
    edge = {"filename": "e.wav", "duration_sec": "2.0",
            "overlap_ratio": "0.1", "overlap_segments": "", "snr_db": "10"}
    out3 = build_description(edge)
    expect_true("F0 trailing appended", "F0 and formant estimates" in out3)

    print()
    print(f"================ {PASS} passed, {FAIL} failed ================")
    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
