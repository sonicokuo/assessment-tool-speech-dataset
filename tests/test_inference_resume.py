"""Tests for src/eval/inference_resume.py (F12, 2026-07-16) — torch-free.

Guards the resume-merge bug: inference.py's auto-resume skips filenames already
present in inference_results.json; without a checkpoint fingerprint, evaluating
two different checkpoints into the same directory silently merges generations
from different weights. These tests cover the fingerprint function and the pure
resume-decision logic (no torch, no model load).

Runnable under pytest or directly: python3 tests/test_inference_resume.py
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from eval.inference_resume import (
    FINGERPRINT_KEY,
    checkpoint_fingerprint,
    fingerprint_from_parts,
    fingerprint_suffixed_path,
    resolve_resume_output_path,
    results_fingerprints,
)


# ── Fingerprint core ─────────────────────────────────────────────────────────
def test_different_blobs_different_fingerprints():
    fp_a = fingerprint_from_parts(b"checkpoint-A-weights", 1000, 1720000000.0)
    fp_b = fingerprint_from_parts(b"checkpoint-B-weights", 1000, 1720000000.0)
    assert fp_a != fp_b


def test_same_blob_same_fingerprint():
    fp1 = fingerprint_from_parts(b"identical-bytes", 4096, 1720000123.456)
    fp2 = fingerprint_from_parts(b"identical-bytes", 4096, 1720000123.456)
    assert fp1 == fp2


def test_size_and_mtime_participate():
    head = b"same-first-megabyte"
    base = fingerprint_from_parts(head, 1000, 1720000000.0)
    assert fingerprint_from_parts(head, 2000, 1720000000.0) != base   # size differs
    assert fingerprint_from_parts(head, 1000, 1720009999.0) != base   # mtime differs


def test_fingerprint_is_short_hex():
    fp = fingerprint_from_parts(b"x", 1, 1.0)
    assert len(fp) == 12
    int(fp, 16)  # must be valid hex


def test_file_fingerprint_matches_parts_and_detects_content_change():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "best.pt")
        with open(path, "wb") as f:
            f.write(b"weights-v1")
        fp1 = checkpoint_fingerprint(path)
        st = os.stat(path)
        assert fp1 == fingerprint_from_parts(b"weights-v1", st.st_size, st.st_mtime)
        # Overwrite with different bytes -> different fingerprint.
        with open(path, "wb") as f:
            f.write(b"weights-v2")
        assert checkpoint_fingerprint(path) != fp1


# ── Entry-fingerprint extraction ─────────────────────────────────────────────
def test_results_fingerprints_extraction():
    entries = [
        {"filename": "a.pt", FINGERPRINT_KEY: "aaaaaaaaaaaa"},
        {"filename": "b.pt", FINGERPRINT_KEY: "aaaaaaaaaaaa"},
        {"filename": "legacy.pt"},  # pre-F12 entry, no fingerprint
    ]
    assert results_fingerprints(entries) == frozenset({"aaaaaaaaaaaa"})
    assert results_fingerprints([]) == frozenset()
    assert results_fingerprints(None) == frozenset()
    assert results_fingerprints([{"filename": "legacy.pt"}]) == frozenset()


# ── Resume decision ──────────────────────────────────────────────────────────
DEFAULT = "/ckpts/run1/inference_results.json"


def test_mismatch_diverts_to_fingerprint_suffixed_file():
    path, resume_ok = resolve_resume_output_path(
        DEFAULT, "bbbbbbbbbbbb", frozenset({"aaaaaaaaaaaa"})
    )
    assert not resume_ok
    assert path == "/ckpts/run1/inference_results.bbbbbbbbbbbb.json"
    assert path != DEFAULT


def test_match_resumes_default_file():
    path, resume_ok = resolve_resume_output_path(
        DEFAULT, "aaaaaaaaaaaa", frozenset({"aaaaaaaaaaaa"})
    )
    assert resume_ok
    assert path == DEFAULT


def test_absent_or_legacy_file_resumes_default():
    # No existing file / legacy file without fingerprints -> empty set -> default.
    path, resume_ok = resolve_resume_output_path(DEFAULT, "cccccccccccc", frozenset())
    assert resume_ok
    assert path == DEFAULT


def test_mixed_fingerprints_still_refuse():
    # A file already contaminated with two checkpoints must not be appended to
    # even if one of them matches the current checkpoint.
    path, resume_ok = resolve_resume_output_path(
        DEFAULT, "aaaaaaaaaaaa", frozenset({"aaaaaaaaaaaa", "bbbbbbbbbbbb"})
    )
    assert not resume_ok
    assert path == "/ckpts/run1/inference_results.aaaaaaaaaaaa.json"


def test_suffixed_path_keeps_extension():
    assert (
        fingerprint_suffixed_path("/x/inference_results.json", "deadbeef1234")
        == "/x/inference_results.deadbeef1234.json"
    )


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except Exception as e:  # noqa: BLE001 — report and continue
                failures += 1
                print(f"FAIL {name}: {e}")
    raise SystemExit(1 if failures else 0)
