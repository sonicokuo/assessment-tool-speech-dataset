#!/usr/bin/env python3
"""Refresh ONLY the `overlap_info` and `overlap_segments` keys in every .pt
under a processed split directory, using the current (post-de49d6a) clamping
build_overlap_info().

Why this script exists
----------------------
The May-12 .pt files under $SHARED/data/processed_pyannote/{train,val,test}
were preprocessed before commit de49d6a, which clamps out-of-clip overlap
endpoints inside build_overlap_info(). Partial-OOB pyannote segments (e.g.
5.0-7.5 s on a 3.92 s clip) leaked inflated `segment_duration_s` values into
overlap_info[:, 1] of those files. The fix is purely in build_overlap_info —
no other tensor in the .pt is affected. So instead of re-running preprocess.py
(which would wipe the BEATs cache, costing ~4-6 h to rebuild), we surgically
refresh only the two affected keys per-clip, in place, atomic.

Source-of-truth for overlap segments
------------------------------------
This script reads the CSV's `overlap_segments` and `overlap_ratio` columns
(the pyannote-on-mix values) — NOT `overlap_segments_vad`. Per CLAUDE.md:
the model's overlap_info INPUT channels come from pyannote-on-mix (this
script's domain). The VAD-on-stems columns are GT for descriptions / SFS,
which is `descriptions.json`'s domain, not this one.

Guarantees per clip
-------------------
  - The set of keys in the cached dict is identical before and after.
  - audio_features, beats_patches, beats_grid_meta, filename keys keep their
    exact Python object identity (no reload, no copy).
  - overlap_info is float32 of shape (T, 4) where T = audio_features.shape[0].
  - Write is atomic via .tmp + os.replace, identical pattern to
    preprocess_beats.py:137-139, so a SIGKILL cannot corrupt a .pt.
  - If the .pt has no matching CSV row, the file is NOT modified, the failure
    is logged, and the run exits with a nonzero return code.

Usage
-----
    # Show side-by-side stats on 5 random clips, no writes
    python scripts/refresh_overlap_info.py \\
        --pt_dir       $SHARED/data/processed_pyannote/train \\
        --features_csv $SHARED/data/features_pyannote/train-100.csv \\
        --dry-run

    # Full run (writes back, atomic per clip)
    python scripts/refresh_overlap_info.py \\
        --pt_dir       $SHARED/data/processed_pyannote/train \\
        --features_csv $SHARED/data/features_pyannote/train-100.csv
"""
from __future__ import annotations

import argparse
import csv
import os
import random
import sys
from pathlib import Path

import torch

HERE = Path(__file__).parent.resolve()
sys.path.insert(0, str(HERE.parent / "src"))

# build_overlap_info already has the post-de49d6a clamping; we re-derive
# overlap_info / overlap_segments through it.
from preprocess import build_overlap_info  # noqa: E402

SAMPLE_RATE = 16000


# ---------------------------------------------------------------------------
# CSV loader
# ---------------------------------------------------------------------------
def load_overlap_map(features_csv: Path) -> dict:
    """Return {filename: (segs_str, ratio)} keyed by the CSV's `filename` col.

    Reads `overlap_segments` (pyannote, NOT *_vad) and `overlap_ratio`. Empty
    / 'nan' / 'n/a' values are normalized to empty string / 0.0 — matching
    src/preprocess.py::load_overlap_map.
    """
    mapping = {}
    with open(features_csv) as f:
        for row in csv.DictReader(f):
            segs = (row.get("overlap_segments") or "").strip()
            if segs.lower() in ("", "nan", "n/a"):
                segs = ""
            ratio_raw = (row.get("overlap_ratio") or "").strip()
            try:
                ratio = float(ratio_raw)
            except ValueError:
                ratio = 0.0
            mapping[row["filename"]] = (segs, ratio)
    return mapping


# ---------------------------------------------------------------------------
# Per-clip refresh
# ---------------------------------------------------------------------------
def _lookup_filename(cached: dict, pt_path: Path) -> str:
    """Prefer the .pt's stored `filename` key; fall back to <basename>.wav."""
    fn = cached.get("filename")
    if isinstance(fn, str) and fn:
        return fn
    return pt_path.stem + ".wav"


def refresh_one_pt(pt_path: Path, overlap_map: dict,
                   sample_rate: int = SAMPLE_RATE,
                   write: bool = True) -> str:
    """Refresh one .pt. Returns a status string:
        'changed'    : new overlap_info differs from old, file rewritten
        'unchanged'  : new == old, no write performed
        'no_csv'     : no matching CSV row, file NOT modified (caller errors)
    """
    cached = torch.load(pt_path, weights_only=False)
    keys_before = sorted(cached.keys())

    fname = _lookup_filename(cached, pt_path)
    if fname not in overlap_map:
        return "no_csv"

    segs_str, ratio = overlap_map[fname]
    T = cached["audio_features"].shape[0]
    new_info, new_segs = build_overlap_info(segs_str, ratio, T,
                                            sample_rate=sample_rate)

    # dtype / shape sanity
    assert new_info.dtype == torch.float32, \
        f"new overlap_info dtype {new_info.dtype}, expected float32"
    assert tuple(new_info.shape) == (T, 4), \
        f"new overlap_info shape {tuple(new_info.shape)}, expected ({T}, 4)"

    # Capture references for identity preservation (the untouched keys).
    af = cached["audio_features"]
    bp = cached.get("beats_patches")
    bm = cached.get("beats_grid_meta")
    fn = cached.get("filename")
    old_info = cached["overlap_info"]

    same_info = (isinstance(old_info, torch.Tensor)
                 and old_info.shape == new_info.shape
                 and old_info.dtype == new_info.dtype
                 and torch.equal(old_info, new_info))
    same_segs = (cached.get("overlap_segments") == new_segs)
    if same_info and same_segs:
        return "unchanged"

    # Mutate the only two keys we own.
    cached["overlap_info"] = new_info
    cached["overlap_segments"] = new_segs

    # Identity preserved for the untouched keys.
    assert cached["audio_features"] is af, \
        "audio_features identity broken (should be same Python object)"
    if bp is not None:
        assert cached["beats_patches"] is bp, "beats_patches identity broken"
    if bm is not None:
        assert cached["beats_grid_meta"] is bm, "beats_grid_meta identity broken"
    if fn is not None:
        assert cached["filename"] is fn, "filename identity broken"

    # Key set unchanged (no key added, no key dropped).
    keys_after = sorted(cached.keys())
    assert keys_after == keys_before, (
        f"key set changed!\n  before: {keys_before}\n  after:  {keys_after}"
    )

    if write:
        # Atomic: .tmp + os.replace, same pattern as preprocess_beats.py L137-139.
        tmp = str(pt_path) + ".tmp"
        torch.save(cached, tmp)
        os.replace(tmp, pt_path)
    return "changed"


# ---------------------------------------------------------------------------
# Whole-split runners
# ---------------------------------------------------------------------------
def _stat_col1(t: torch.Tensor):
    """(max, nonzero count, distinct value count) for overlap_info[:, 1]."""
    col1 = t[:, 1]
    nonzero = int((col1 != 0).sum().item())
    # round to 4 decimals to count "distinct" without float-dust explosion
    distinct = len({round(v, 4) for v in col1.tolist()})
    return float(col1.max().item()), nonzero, distinct


def run_dry(pt_dir: Path, features_csv: Path, n_sample: int = 5,
            seed: int = 0) -> int:
    """Side-by-side compare overlap_info[:, 1] stats on n_sample random clips.
    Does not write back. Returns 0."""
    overlap_map = load_overlap_map(features_csv)
    pt_files = sorted(p for p in pt_dir.iterdir() if p.suffix == ".pt")
    if not pt_files:
        print(f"  no .pt files in {pt_dir}")
        return 0
    rng = random.Random(seed)
    sample = rng.sample(pt_files, min(n_sample, len(pt_files)))

    print(f"=== --dry-run on {len(sample)} random clips (seed={seed}) ===")
    print(f"     no files written; overlap_info[:, 1] = segment_duration_s\n")
    n_changed = n_unchanged = n_no_csv = 0
    for pt_path in sample:
        cached = torch.load(pt_path, weights_only=False)
        T = cached["audio_features"].shape[0]
        dur_sec = T * 320 / SAMPLE_RATE
        fname = _lookup_filename(cached, pt_path)
        row = overlap_map.get(fname)
        if row is None:
            print(f"  {fname:55s}  NO CSV ROW")
            n_no_csv += 1
            continue
        segs_str, ratio = row
        old_info = cached["overlap_info"]
        new_info, _ = build_overlap_info(segs_str, ratio, T, sample_rate=SAMPLE_RATE)
        o_max, o_nz, o_d = _stat_col1(old_info)
        n_max, n_nz, n_d = _stat_col1(new_info)
        changed = not (
            old_info.shape == new_info.shape
            and old_info.dtype == new_info.dtype
            and torch.equal(old_info, new_info)
        )
        if changed: n_changed += 1
        else:       n_unchanged += 1
        tag = "CHANGED  " if changed else "unchanged"
        print(f"  {fname}")
        print(f"    T={T}  clip_dur={dur_sec:.3f}s  segs={segs_str!r}  {tag}")
        print(f"    old col1: max={o_max:.4f}  nonzero={o_nz:4d}  distinct={o_d}")
        print(f"    new col1: max={n_max:.4f}  nonzero={n_nz:4d}  distinct={n_d}\n")

    print(f"=== dry-run summary: changed={n_changed}  unchanged={n_unchanged}  no_csv={n_no_csv} ===")
    return 0


def run_full(pt_dir: Path, features_csv: Path,
             sample_rate: int = SAMPLE_RATE,
             heartbeat_every: int = 1000) -> int:
    """Refresh every .pt under pt_dir in place. Returns nonzero if any .pt
    had no matching CSV row, otherwise 0."""
    pt_dir = Path(pt_dir)
    features_csv = Path(features_csv)
    overlap_map = load_overlap_map(features_csv)
    pt_files = sorted(p for p in pt_dir.iterdir() if p.suffix == ".pt")
    n_total = len(pt_files)
    print(f"=== refreshing overlap_info in {n_total} .pt files under {pt_dir} ===")
    print(f"    CSV: {features_csv}  (rows: {len(overlap_map)})\n")

    n_processed = n_changed = n_unchanged = n_no_csv = 0
    sample_changed: list[str] = []
    sample_no_csv:  list[str] = []
    for i, pt_path in enumerate(pt_files):
        status = refresh_one_pt(pt_path, overlap_map, sample_rate=sample_rate)
        if status == "changed":
            n_changed += 1
            if len(sample_changed) < 5: sample_changed.append(pt_path.stem)
        elif status == "unchanged":
            n_unchanged += 1
        elif status == "no_csv":
            n_no_csv += 1
            if len(sample_no_csv) < 5: sample_no_csv.append(pt_path.stem)
        n_processed += 1
        if (i + 1) % heartbeat_every == 0:
            print(f"  [{i+1}/{n_total}] processed={n_processed} "
                  f"changed={n_changed} unchanged={n_unchanged} no_csv={n_no_csv}")

    print(f"\n=== summary ===")
    print(f"  total           : {n_total}")
    print(f"  processed       : {n_processed}")
    print(f"  changed (wrote) : {n_changed}")
    print(f"  unchanged       : {n_unchanged}")
    print(f"  no csv row      : {n_no_csv}")
    if sample_changed:
        print(f"  sample changed stems  : {sample_changed}")
    if sample_no_csv:
        print(f"  sample no_csv stems   : {sample_no_csv}")
    if n_no_csv:
        print(f"\nERROR: {n_no_csv} .pt file(s) had no matching CSV row. Those files were NOT modified.",
              file=sys.stderr)
        return 1
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--pt_dir", required=True,
                    help="Directory containing the .pt files to refresh.")
    ap.add_argument("--features_csv", required=True,
                    help="features_pyannote/<split>.csv with overlap_segments + overlap_ratio columns.")
    ap.add_argument("--dry-run", action="store_true", dest="dry_run",
                    help="Process N random clips and print before/after stats; "
                         "do NOT write back. See --dry-run-n.")
    ap.add_argument("--dry-run-n", type=int, default=5, dest="dry_run_n",
                    help="Number of clips to sample for --dry-run (default: 5).")
    ap.add_argument("--seed", type=int, default=0,
                    help="Random seed for --dry-run sampling (default: 0).")
    args = ap.parse_args(argv)

    pt_dir = Path(args.pt_dir)
    csv_path = Path(args.features_csv)
    if not pt_dir.is_dir():
        print(f"ERROR: --pt_dir not a directory: {pt_dir}", file=sys.stderr)
        return 2
    if not csv_path.is_file():
        print(f"ERROR: --features_csv not a file: {csv_path}", file=sys.stderr)
        return 2

    if args.dry_run:
        return run_dry(pt_dir, csv_path, n_sample=args.dry_run_n, seed=args.seed)
    return run_full(pt_dir, csv_path)


if __name__ == "__main__":
    raise SystemExit(main())
