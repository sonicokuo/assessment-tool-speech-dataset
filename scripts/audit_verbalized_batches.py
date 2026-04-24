#!/usr/bin/env python3
"""
Scan verbalized outputs for [ERROR] rows, gaps, and tail-missing ranges.

For each split (train-100 / dev / test), the script auto-detects which layout is
on disk and audits accordingly:

  1. Consolidated single file at top of verbalized_dir:
        <split>.csv
  2. Batched files at the top level:
        <split>_<start>_<end>.csv
  3. Batched files under a sibling subdir:
        <split>_batches/<split>_<start>_<end>.csv

Preference order: single file > top-level batches > batches subdir. If both a
single file and batches exist, the single file is treated as the source of
truth and the batches are reported as "also present" (won't cause double
counting).

Usage:
    python scripts/audit_verbalized_batches.py <verbalized_dir>
    python scripts/audit_verbalized_batches.py <verbalized_dir> --split test
"""
import csv
import re
import sys
from pathlib import Path

csv.field_size_limit(10_000_000)

EXPECTED = {"train-100": 13900, "dev": 3000, "test": 3000}


def count_rows_and_errors(path: Path) -> tuple[int, int, list[str]]:
    """Return (total_rows, error_rows, list_of_errored_filenames)."""
    n_rows = n_err = 0
    errored = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            n_rows += 1
            if (row.get("quality_description") or "").startswith("[ERROR]"):
                n_err += 1
                errored.append(row.get("filename", "?"))
    return n_rows, n_err, errored


def find_batches(vdir: Path, split: str) -> list[tuple[int, int, Path]]:
    """Find all `<split>_<start>_<end>.csv` files at top level OR in `<split>_batches/`."""
    pattern = re.compile(rf"^{re.escape(split)}_(\d+)_(\d+)\.csv$")
    dirs_to_search = [vdir, vdir / f"{split}_batches"]
    batches = []
    for d in dirs_to_search:
        if not d.is_dir():
            continue
        for path in d.iterdir():
            m = pattern.match(path.name)
            if m:
                batches.append((int(m.group(1)), int(m.group(2)), path))
    batches.sort()
    return batches


def audit_single_file(split: str, path: Path, exp: int) -> bool:
    """Audit a consolidated single-file split. Returns True if clean."""
    n_rows, n_err, errored = count_rows_and_errors(path)
    status = "✅" if n_rows == exp and n_err == 0 else "⚠️"
    print(f"{split:10s}: {status}  {n_rows} rows, {n_err} errors  "
          f"(expected {exp})  [single file: {path.name}]")
    if n_err:
        print(f"    {n_err} [ERROR] rows — rerun the split:")
        print(f"    python scripts/feature_verbalization.py "
              f"--input $SHARED/data/features/{split}.csv "
              f"--output $SHARED/data/verbalized/{split}.csv")
        for fn in errored[:5]:
            print(f"      - {fn}")
        if len(errored) > 5:
            print(f"      - ... ({len(errored) - 5} more)")
    if n_rows != exp:
        print(f"    row count mismatch — split may be incomplete")
    return n_rows == exp and n_err == 0


def audit_batched(split: str, batches: list[tuple[int, int, Path]], exp: int) -> bool:
    """Audit a list of batch files for a split. Returns True if clean."""
    total_rows = total_err = 0
    last_end = 0
    gaps = []
    bad = []
    for start, end, path in batches:
        if start > last_end:
            gaps.append((last_end, start))
        last_end = max(last_end, end)
        n_rows, n_err, _ = count_rows_and_errors(path)
        total_rows += n_rows
        total_err += n_err
        if n_err > 0:
            bad.append((start, end, n_err, path.name))

    tail_missing = exp - last_end
    clean = total_err == 0 and tail_missing == 0 and not gaps
    status = "✅" if clean else "⚠️"
    print(f"{split:10s}: {status}  {total_rows} rows in {len(batches)} batches, "
          f"{total_err} errors  (expected {exp})  [batched]")

    if gaps:
        for g in gaps:
            print(f"    gap: rows {g[0]}-{g[1]} never verbalized — "
                  f"run: ./scripts/run_batch.sh {g[0]} {g[1]}")
    if tail_missing > 0:
        print(f"    tail missing: rows {last_end}-{exp} — "
              f"run: ./scripts/run_batch.sh {last_end} {exp}")
    if bad:
        print("    error batches — rerun these:")
        for start, end, n_err, name in bad:
            print(f"      ./scripts/run_batch.sh {start} {end}   # "
                  f"{n_err} errors in {name}")
    return clean


def audit_split(vdir: Path, split: str) -> bool:
    """Dispatch to single-file or batched audit based on what exists on disk."""
    exp = EXPECTED[split]
    single_path = vdir / f"{split}.csv"
    batches = find_batches(vdir, split)

    if single_path.exists():
        clean = audit_single_file(split, single_path, exp)
        if batches:
            print(f"    (also found {len(batches)} batch files; single file takes precedence)")
        return clean

    if batches:
        return audit_batched(split, batches, exp)

    print(f"{split:10s}: ❌  MISSING  (expected {exp} rows)")
    return False


def main():
    import argparse
    ap = argparse.ArgumentParser(
        description="Audit verbalized outputs for [ERROR] rows and gaps."
    )
    ap.add_argument("verbalized_dir", help="Directory containing verbalized CSVs")
    ap.add_argument("--split", choices=["dev", "test", "train-100", "all"], default="all",
                    help="Audit only one split (default: all)")
    args = ap.parse_args()

    vdir = Path(args.verbalized_dir)
    if not vdir.is_dir():
        print(f"Not a directory: {vdir}")
        sys.exit(1)

    print(f"Auditing {vdir}\n")
    splits_to_audit = ["dev", "test", "train-100"] if args.split == "all" else [args.split]
    for split in splits_to_audit:
        audit_split(vdir, split)


if __name__ == "__main__":
    main()
