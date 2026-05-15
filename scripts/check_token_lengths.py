#!/usr/bin/env python3
"""Report the Qwen tokenizer length distribution for a descriptions JSON.

Diagnoses whether `max_target_length` in the training config is too tight.
Targets longer than max_target_length get the LAST tokens truncated before
EOS is appended (in src/train.py::_tokenize_with_eos), which means the model
never learns to generate the late-sequence content (typically the
<sec_overlap> block and the trailing F0-unreliable sentence).

Usage:
    python scripts/check_token_lengths.py
    python scripts/check_token_lengths.py --descriptions $SHARED/data/descriptions.json
    python scripts/check_token_lengths.py --tokenizer Qwen/Qwen3-1.7B --sample 500
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--descriptions", default=None,
                   help="Path to descriptions.json. Defaults to "
                        "$SHARED/data/descriptions.json if $SHARED is set, "
                        "otherwise ./descriptions.json.")
    p.add_argument("--tokenizer", default="Qwen/Qwen3-1.7B",
                   help="HF tokenizer id. Default: Qwen/Qwen3-1.7B.")
    p.add_argument("--sample", type=int, default=0,
                   help="If > 0, sample this many entries (random.seed(0)) "
                        "for a faster check. Default: 0 = all.")
    p.add_argument("--caps", type=int, nargs="+",
                   default=[224, 256, 320, 384, 448, 512],
                   help="max_target_length caps to estimate truncation impact at. "
                        "Default: 224 256 320 384 448 512.")
    args = p.parse_args()

    if args.descriptions is None:
        shared = os.environ.get("SHARED")
        cand = (Path(shared) / "data" / "descriptions.json") if shared else None
        if cand and cand.exists():
            args.descriptions = str(cand)
        elif Path("descriptions.json").exists():
            args.descriptions = "descriptions.json"
        else:
            print("ERROR: --descriptions not given and could not locate a default. "
                  "Pass --descriptions <path>.", file=sys.stderr)
            return 2

    print(f"loading tokenizer: {args.tokenizer}")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer)

    print(f"loading descriptions: {args.descriptions}")
    with open(args.descriptions) as f:
        d = json.load(f)
    print(f"  {len(d):,} entries")

    if args.sample > 0 and args.sample < len(d):
        import random
        rng = random.Random(0)
        keys = rng.sample(list(d.keys()), args.sample)
        d = {k: d[k] for k in keys}
        print(f"  sampling {len(d):,} for speed")

    # Tokenize without add_special_tokens (matches _tokenize_with_eos in train.py).
    lens = [len(tok.encode(v, add_special_tokens=False)) for v in d.values()]
    lens.sort()
    n = len(lens)

    print()
    print(f"=== token-length distribution (Qwen3 BPE, no special tokens) ===")
    print(f"  min     : {lens[0]}")
    print(f"  p50     : {lens[n // 2]}")
    print(f"  mean    : {sum(lens) / n:.1f}")
    print(f"  p90     : {lens[int(n * 0.90)]}")
    print(f"  p95     : {lens[int(n * 0.95)]}")
    print(f"  p99     : {lens[int(n * 0.99)]}")
    print(f"  p99.9   : {lens[int(n * 0.999)]}")
    print(f"  max     : {lens[-1]}")
    print()

    print(f"=== truncation impact per cap (target gets truncated before EOS) ===")
    print(f"  An entry is truncated iff its token count >= max_target_length (the")
    print(f"  cap reserves slot N-1 for content, slot N for EOS via _tokenize_with_eos).")
    print()
    print(f"  {'cap':>6}  {'truncated':>12}  {'%':>8}")
    for cap in args.caps:
        n_trunc = sum(1 for L in lens if L >= cap)
        pct = 100.0 * n_trunc / n
        marker = "  <-- current YAML setting" if cap == 224 else ""
        print(f"  {cap:>6}  {n_trunc:>12,}  {pct:>7.2f}%{marker}")

    print()
    smallest_safe = next((cap for cap in sorted(args.caps)
                          if all(L < cap for L in lens)), None)
    if smallest_safe is None:
        print(f"  no cap in {args.caps} covers all entries; longest is {lens[-1]} tokens.")
        print(f"  set max_target_length >= {lens[-1] + 8} to be safe.")
    else:
        print(f"  smallest cap that covers EVERY description without truncation: "
              f"{smallest_safe}")
        print(f"  recommended max_target_length (smallest power-of-32 above max): "
              f"{((lens[-1] + 8 + 31) // 32) * 32}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
