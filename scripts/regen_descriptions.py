#!/usr/bin/env python3
"""Regenerate descriptions for clips that fix_descriptions.py flagged as
structurally bad (A3 orphan <f_*>, A5 unbalanced <r>, A6 empty, Ax
unbalanced <f>). Reads the current JSON, finds the affected stems'
rows in the source feature CSVs, calls Ollama via the existing
verbalizer, runs the output through fix_descriptions.fix_all, and
writes the JSON back in place.

Prereqs:
  - Ollama serve up at localhost:11434
  - gemma4:e2b available (auto-loads on first call)
  - $SHARED env set if your features CSVs live there (the --features-dir
    arg defaults to $SHARED/data/features_pyannote)

Usage:
    python scripts/regen_descriptions.py --input <part1.json>
    python scripts/regen_descriptions.py --input <part2.json> --features-dir ...
    python scripts/regen_descriptions.py --input <X.json> --dry-run
"""
import argparse
import csv
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).parent.resolve()
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "src"))

from feature_verbalization import generate_quality_description_sectioned  # noqa: E402
from fix_descriptions import audit, fix_all  # noqa: E402

STRUCTURAL = {
    "A3_orphan_f_outside_section",
    "A5_unbalanced_r",
    "A6_empty",
    "Ax_unbalanced_f",
}


def main():
    default_features = f"{os.environ.get('SHARED', '/ocean/projects/cis260125p/shared')}/data/features_pyannote"
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--input", required=True, help="descriptions JSON (rewritten in place)")
    ap.add_argument("--features-dir", default=default_features,
                    help="dir containing train-100.csv, dev.csv, test.csv")
    ap.add_argument("--dry-run", action="store_true",
                    help="list affected stems and exit (no Ollama calls)")
    args = ap.parse_args()

    data = json.loads(Path(args.input).read_text())
    n = len(data)
    needs_regen = [
        stem for stem, text in data.items()
        if any(code in STRUCTURAL for code in audit(text))
    ]
    print(f"input: {args.input}  ({n} entries)")
    print(f"clips needing regeneration: {len(needs_regen)}")

    if args.dry_run:
        for s in needs_regen[:30]:
            print(f"  {s}")
        if len(needs_regen) > 30:
            print(f"  ... ({len(needs_regen) - 30} more)")
        return

    if not needs_regen:
        print("nothing to do")
        return

    # Build stem -> row lookup across all 3 feature CSVs
    fdir = Path(args.features_dir)
    needed = set(needs_regen)
    stem_to_row = {}
    for csv_path in [fdir / "train-100.csv", fdir / "dev.csv", fdir / "test.csv"]:
        if not csv_path.exists():
            print(f"WARN: missing {csv_path}")
            continue
        with open(csv_path, newline="") as f:
            for row in csv.DictReader(f):
                stem = Path(row.get("filename", "")).stem
                if stem in needed:
                    stem_to_row[stem] = row
    missing = needed - set(stem_to_row)
    if missing:
        print(f"WARN: {len(missing)} stems not in any feature CSV; will be skipped")

    n_ok = n_ollama_err = n_still_bad = n_skip = 0
    for i, stem in enumerate(needs_regen, 1):
        if stem not in stem_to_row:
            print(f"[{i}/{len(needs_regen)}] {stem}: SKIP (no source row)")
            n_skip += 1
            continue
        print(f"[{i}/{len(needs_regen)}] {stem} ...", end="", flush=True)
        new_text = generate_quality_description_sectioned(stem_to_row[stem])
        if new_text.startswith("[ERROR]"):
            print(f" ollama-err: {new_text[:120]}")
            n_ollama_err += 1
            continue
        cleaned = fix_all(new_text)
        leftover = [c for c in audit(cleaned) if c in STRUCTURAL]
        if leftover:
            print(f" still problematic: {leftover}")
            n_still_bad += 1
        else:
            print(" ok")
            n_ok += 1
        data[stem] = cleaned  # save even when imperfect; better than orphan

    Path(args.input).write_text(json.dumps(data, ensure_ascii=False, indent=2))
    print()
    print(f"regen complete: ok={n_ok}  ollama-err={n_ollama_err}  "
          f"still-bad={n_still_bad}  skipped={n_skip}")
    print(f"updated: {args.input}")


if __name__ == "__main__":
    main()
