#!/usr/bin/env python3
"""
Scan verbalized batch files for [ERROR] rows and emit a list of
(start, end) ranges that need to be re-run through verbalization.

Usage:
    python scripts/audit_verbalized_batches.py <verbalized_dir>

Prints:
  - summary (total batches, batches with errors, total error rows)
  - one rerun command per bad batch
  - the final `cat` list so you can inspect it
"""
import csv
import re
import sys
from pathlib import Path

csv.field_size_limit(10_000_000)

BATCH_RE = re.compile(r"train-100_(\d+)_(\d+)\.csv$")


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/audit_verbalized_batches.py <verbalized_dir>")
        sys.exit(1)

    vdir = Path(sys.argv[1])
    bad = []          # list of (start, end, n_errors, path)
    total_rows = 0
    total_errors = 0
    total_batches = 0

    for path in sorted(vdir.glob("train-100_*_*.csv")):
        m = BATCH_RE.search(path.name)
        if not m:
            continue
        start, end = int(m.group(1)), int(m.group(2))
        total_batches += 1
        n_err = 0
        n_rows = 0
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                n_rows += 1
                if (row.get("quality_description") or "").startswith("[ERROR]"):
                    n_err += 1
        total_rows += n_rows
        total_errors += n_err
        if n_err > 0:
            bad.append((start, end, n_err, path.name))

    print(f"Scanned {total_batches} batches, {total_rows} rows total")
    print(f"Error rows: {total_errors}")
    print(f"Batches with ≥1 error: {len(bad)}")
    print()

    if bad:
        print("=== Rerun these ranges ===")
        for start, end, n_err, name in bad:
            print(f"./scripts/run_batched.sh {start} {end}   # {n_err} errors in {name}")


if __name__ == "__main__":
    main()
