#!/usr/bin/env python3
"""
Scan verbalized outputs for [ERROR] rows across all splits:
  - train-100 batched files:  train-100_{start}_{end}.csv
  - single-file splits:       dev.csv, test.csv

Usage:
    python scripts/audit_verbalized_batches.py <verbalized_dir>

Prints per-split summary + rerun commands for the batched train-100
errors and a flag for dev/test errors (which need to be re-run via
the non-batched verbalization command).
"""
import csv
import re
import sys
from pathlib import Path

csv.field_size_limit(10_000_000)

BATCH_RE = re.compile(r"train-100_(\d+)_(\d+)\.csv$")
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


def audit_train100(vdir: Path) -> list[tuple[int, int, int, str]]:
    """Return list of (start, end, n_errors, path) for each bad train-100 batch."""
    bad = []
    total_rows = total_err = 0
    last_end = 0
    gaps = []

    batches = []
    for path in vdir.glob("train-100_*_*.csv"):
        m = BATCH_RE.search(path.name)
        if m:
            batches.append((int(m.group(1)), int(m.group(2)), path))
    batches.sort()

    for start, end, path in batches:
        if start > last_end:
            gaps.append((last_end, start))
        last_end = max(last_end, end)
        n_rows, n_err, _ = count_rows_and_errors(path)
        total_rows += n_rows
        total_err += n_err
        if n_err > 0:
            bad.append((start, end, n_err, path.name))

    tail_missing = EXPECTED["train-100"] - last_end

    status = "✅" if total_err == 0 and tail_missing == 0 and not gaps else "⚠️"
    print(f"train-100 : {status}  {total_rows} rows in {len(batches)} batches, "
          f"{total_err} errors  (expected {EXPECTED['train-100']})")

    if gaps:
        for g in gaps:
            print(f"    gap: rows {g[0]}-{g[1]} never verbalized — run: ./scripts/run_batched.sh {g[0]} {g[1]}")
    if tail_missing > 0:
        print(f"    tail missing: rows {last_end}-{EXPECTED['train-100']} — run: ./scripts/run_batched.sh {last_end} {EXPECTED['train-100']}")
    if bad:
        print("    error batches — rerun these:")
        for start, end, n_err, name in bad:
            print(f"      ./scripts/run_batched.sh {start} {end}   # {n_err} errors in {name}")
    return bad


def audit_single(vdir: Path, split: str) -> bool:
    """Audit a single-file split (dev.csv / test.csv). Returns True if clean."""
    path = vdir / f"{split}.csv"
    if not path.exists():
        print(f"{split:10s}: ❌  MISSING  (expected {EXPECTED[split]} rows)")
        return False
    n_rows, n_err, errored = count_rows_and_errors(path)
    exp = EXPECTED[split]
    status = "✅" if n_rows == exp and n_err == 0 else "⚠️"
    print(f"{split:10s}: {status}  {n_rows} rows, {n_err} errors  (expected {exp})")
    if n_err:
        print(f"    {n_err} [ERROR] rows — rerun the whole split:")
        print(f"    python scripts/feature_verbalization.py --input $SHARED/data/features/{split}.csv --output $SHARED/data/verbalized/{split}.csv")
        for fn in errored[:5]:
            print(f"      - {fn}")
        if len(errored) > 5:
            print(f"      - ... ({len(errored) - 5} more)")
    if n_rows != exp:
        print(f"    row count mismatch — split may be incomplete")
    return n_rows == exp and n_err == 0


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/audit_verbalized_batches.py <verbalized_dir>")
        sys.exit(1)

    vdir = Path(sys.argv[1])
    if not vdir.is_dir():
        print(f"Not a directory: {vdir}")
        sys.exit(1)

    print(f"Auditing {vdir}\n")
    audit_single(vdir, "dev")
    audit_single(vdir, "test")
    audit_train100(vdir)


if __name__ == "__main__":
    main()
