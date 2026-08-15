"""Checkpoint-fingerprinted resume logic for src/inference.py (F12, 2026-07-16).

Problem: inference.py auto-resumes by skipping filenames already present in
inference_results.json in dirname(--checkpoint). If two DIFFERENT checkpoints are
ever evaluated into the same directory (best.pt overwritten by a later epoch,
last.pt vs best.pt, a re-trained run reusing the folder), generations produced by
different weights silently merge into one results file and every aggregate
computed from it is a chimera.

Fix: fingerprint the checkpoint file at load (sha256 of its first 1 MiB + file
size + mtime, hex[:12]); stamp every result entry with `_checkpoint_fingerprint`;
on resume, refuse to append to a results file whose entries carry a DIFFERENT
fingerprint and divert to `inference_results.<fingerprint>.json` instead.

This module is deliberately torch-free (stdlib only) so the logic is unit-testable
without the training environment (tests/test_inference_resume.py).
"""
from __future__ import annotations

import hashlib
import os

# How much of the checkpoint file participates in the content hash. 1 MiB reads in
# ~ms even on cold NFS and is enough to distinguish real checkpoints (headers +
# first tensor shards differ across weights), while size+mtime guard the tail.
FINGERPRINT_HEAD_BYTES: int = 1024 * 1024

# Key stamped into every entry of inference_results.json.
FINGERPRINT_KEY: str = "_checkpoint_fingerprint"


def fingerprint_from_parts(head: bytes, size: int, mtime: float) -> str:
    """Pure fingerprint core: sha256(first-bytes + size + mtime), hex[:12].

    Split out from checkpoint_fingerprint() so it is testable on raw byte blobs
    without touching the filesystem.
    """
    h = hashlib.sha256()
    h.update(head)
    h.update(f"|size={size}|mtime={mtime:.3f}".encode("ascii"))
    return h.hexdigest()[:12]


def checkpoint_fingerprint(path: str | os.PathLike) -> str:
    """Short fingerprint of a checkpoint file (first 1 MiB + size + mtime, hex[:12])."""
    p = os.fspath(path)
    st = os.stat(p)
    with open(p, "rb") as f:
        head = f.read(FINGERPRINT_HEAD_BYTES)
    return fingerprint_from_parts(head, st.st_size, st.st_mtime)


def results_fingerprints(entries) -> frozenset[str]:
    """Set of checkpoint fingerprints found in already-written result entries.

    Legacy results (written before F12) carry no fingerprint and contribute
    nothing to the set — they resume as before (unverifiable, not refused).
    """
    fps = set()
    for e in entries or ():
        if isinstance(e, dict) and FINGERPRINT_KEY in e:
            fps.add(str(e[FINGERPRINT_KEY]))
    return frozenset(fps)


def fingerprint_suffixed_path(default_path: str, fingerprint: str) -> str:
    """inference_results.json -> inference_results.<fingerprint>.json."""
    base, ext = os.path.splitext(default_path)
    return f"{base}.{fingerprint}{ext}"


def resolve_resume_output_path(
    default_path: str,
    current_fingerprint: str,
    existing_fingerprints,
) -> tuple[str, bool]:
    """Decide where this run may write its results.

    Args:
        default_path: the canonical results path (save_dir/inference_results.json).
        current_fingerprint: fingerprint of the checkpoint being evaluated.
        existing_fingerprints: fingerprints found in the file at default_path
            (empty when the file is absent or is a legacy no-fingerprint file).

    Returns:
        (output_path, resume_ok):
          - resume_ok True  -> output_path == default_path; existing entries were
            written by this same checkpoint (or are legacy/absent) and may be
            appended to.
          - resume_ok False -> the existing file holds generations from a DIFFERENT
            checkpoint; output_path is the fingerprint-suffixed sibling and the
            default file must not be touched.
    """
    others = {fp for fp in existing_fingerprints if fp != current_fingerprint}
    if others:
        return fingerprint_suffixed_path(default_path, current_fingerprint), False
    return default_path, True
