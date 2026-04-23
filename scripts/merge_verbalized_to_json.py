#!/usr/bin/env python3
"""
Merge all train-100/dev/test verbalized CSVs into a single descriptions.json.

Skips rows where quality_description starts with [ERROR]. Handles both the
batched `train-100_START_END.csv` files and single-file splits
(`dev.csv`, `test.csv`).

Usage:
    python scripts/merge_verbalized_to_json.py \\
        --verbalized_dir /ocean/projects/cis260125p/shared/data/verbalized \\
        --output         /ocean/projects/cis260125p/shared/data/descriptions.json

    # Optional:
    --allow_errors   keep [ERROR] rows in the output (default: skip)
    --require_all    exit non-zero if any batch has errors
"""
import argparse
import csv
import json
import re
import sys
from pathlib import Path

csv.field_size_limit(10_000_000)

BATCH_RE = re.compile(r"train-100_(\d+)_(\d+)\.csv$")


def iter_rows(vdir: Path):
    """Yield (filename_stem, quality_description, is_error) for every row
    across dev.csv, test.csv, and all train-100_*_*.csv batches."""
    paths = []
    for name in ("dev.csv", "test.csv", "train-100.csv"):
        p = vdir / name
        if p.exists():
            paths.append(p)
    paths.extend(sorted(vdir.glob("train-100_*_*.csv")))

    for path in paths:
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                fname = row.get("filename", "")
                qd = row.get("quality_description", "") or ""
                stem = Path(fname).stem
                is_error = qd.startswith("[ERROR]")
                yield stem, qd, is_error, path.name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verbalized_dir", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--allow_errors", action="store_true",
                    help="Include [ERROR] rows in the output (default skips them)")
    ap.add_argument("--require_all", action="store_true",
                    help="Exit non-zero if any error rows are present")
    args = ap.parse_args()

    descriptions = {}
    n_total = n_error = n_dup = 0
    error_sources = []

    for stem, qd, is_error, src in iter_rows(Path(args.verbalized_dir)):
        n_total += 1
        if is_error:
            n_error += 1
            error_sources.append((stem, src))
            if not args.allow_errors:
                continue
        if stem in descriptions:
            n_dup += 1
        descriptions[stem] = qd

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(descriptions, f, ensure_ascii=False, indent=2)

    print(f"Rows processed: {n_total}")
    print(f"Error rows:     {n_error}  ({'kept' if args.allow_errors else 'skipped'})")
    print(f"Duplicates:     {n_dup}  (later rows overwrote earlier ones)")
    print(f"Entries in JSON: {len(descriptions)}")
    print(f"Written to:     {args.output}")

    if args.require_all and n_error > 0:
        print("\nError rows still present; rerun them before shipping.", file=sys.stderr)
        for stem, src in error_sources[:10]:
            print(f"  {stem}  ← {src}", file=sys.stderr)
        if len(error_sources) > 10:
            print(f"  ... ({len(error_sources) - 10} more)", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
