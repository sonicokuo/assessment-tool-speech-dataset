#!/usr/bin/env python3
"""Overlap-hedging qualitative comparison.

Direct evidence for paper contribution #1 (overlap-aware quality description):
shows that the model emits the "F0 and formant estimates are unreliable during
overlap windows" hedging sentence on high-overlap clips but not on low-overlap
clips. If the model has learned the conditional hedge, the hedge_rate should
climb monotonically with overlap_ratio.

Usage
-----
    python scripts/overlap_hedging_compare.py \
      --inference_results $SHARED/checkpoints/v7_lora_8b/inference_results.json \
      --output            $SHARED/checkpoints/v7_lora_8b/overlap_hedging.md
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from sfs import ClaimParser  # noqa: E402


# Hedging sentence the deterministic builder emits when overlap_ratio > 0.
# Matches loosely (case-insensitive, allow "F0" or "fundamental frequency",
# variants of "unreliable" / "noisy") so model paraphrases still register.
_HEDGE_RE = re.compile(
    r"(?:F0|fundamental\s+frequency|pitch).*?(?:unreliable|inaccurate|"
    r"affected|degraded|noisy|unstable).*?overlap",
    re.IGNORECASE | re.DOTALL,
)


def has_hedge(text: str) -> bool:
    """True if the generation contains an overlap-aware hedging sentence."""
    return _HEDGE_RE.search(text) is not None


def overlap_ratio_from_target(target: str) -> float | None:
    """Pull the overlap_ratio out of the target prose. Returns None if absent."""
    parser = ClaimParser()
    for claim in parser.parse(target):
        if claim.feature == "overlap_ratio":
            return claim.value
    return None


# Bucket boundaries chosen to give roughly equal-size buckets on Libri2Mix
# (which has heavy mass at 0.5-0.9 overlap because mix_clean concatenates two
# speakers). Adjust if your distribution looks different.
BUCKETS = [
    ("none",    lambda r: r == 0.0),
    ("low",     lambda r: 0.0 < r < 0.10),
    ("mid",     lambda r: 0.10 <= r < 0.50),
    ("high",    lambda r: 0.50 <= r < 0.80),
    ("v_high",  lambda r: r >= 0.80),
]


def aggregate(results: list[dict]) -> dict[str, dict]:
    """Tally hedge presence per overlap-ratio bucket.

    Returns: {bucket_name: {"n": int, "n_hedged": int, "examples": list}}
    """
    agg: dict[str, dict] = {name: {"n": 0, "n_hedged": 0, "examples": []}
                            for name, _ in BUCKETS}
    for entry in results:
        target = entry.get("target") or ""
        generated = entry.get("generated") or ""
        ratio = overlap_ratio_from_target(target)
        if ratio is None:
            continue  # can't bucket without GT overlap
        hedged = has_hedge(generated)

        for name, pred in BUCKETS:
            if pred(ratio):
                agg[name]["n"] += 1
                if hedged:
                    agg[name]["n_hedged"] += 1
                # Keep at most 2 example clip filenames per bucket for the table
                if len(agg[name]["examples"]) < 2:
                    agg[name]["examples"].append({
                        "filename": entry.get("filename", "?"),
                        "overlap_ratio": ratio,
                        "hedged": hedged,
                        "generated_excerpt": generated[:200],
                    })
                break
    return agg


def render_markdown(agg: dict[str, dict], total_clips: int) -> str:
    lines = []
    lines.append("# Overlap-aware hedging — empirical evidence\n")
    lines.append(
        "Tests whether the model emits the **\"F0/formant estimates are "
        "unreliable during overlap windows\"** hedge in proportion to the "
        "ground-truth overlap ratio. If the hedge is overlap-aware (not "
        "vestigial), `hedge_rate` should climb monotonically with the bucket.\n"
    )
    lines.append(f"Total scorable clips: {sum(b['n'] for b in agg.values())} / {total_clips}\n")
    lines.append("| Overlap bucket | n | n hedged | hedge_rate |")
    lines.append("|---|---:|---:|---:|")
    for name, _ in BUCKETS:
        b = agg[name]
        rate = b["n_hedged"] / b["n"] if b["n"] else 0.0
        lines.append(f"| {name} | {b['n']} | {b['n_hedged']} | {rate:.3f} |")
    lines.append("")
    lines.append("## Concrete examples per bucket\n")
    for name, _ in BUCKETS:
        b = agg[name]
        if not b["examples"]:
            continue
        lines.append(f"### {name} (n={b['n']}, hedge_rate={b['n_hedged'] / max(1, b['n']):.3f})")
        for ex in b["examples"]:
            lines.append(f"- **{ex['filename']}** | overlap_ratio={ex['overlap_ratio']:.3f} | hedged={ex['hedged']}")
            lines.append(f"  > {ex['generated_excerpt']}...")
        lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append("- Monotonically increasing `hedge_rate` across buckets → model has learned overlap-conditional hedging.")
    lines.append("- Flat or random pattern → model emits the hedge spuriously or never; the conditional learning failed.")
    lines.append("- Hedge present on `none` bucket → false positives (over-hedging).")
    lines.append("- Hedge absent on `v_high` bucket → under-hedging on the cases that matter most.")
    return "\n".join(lines)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--inference_results", type=Path, required=True,
                   help="Path to inference_results.json produced by src/inference.py")
    p.add_argument("--output", type=Path, default=None,
                   help="Markdown output path (default: alongside the inference_results)")
    args = p.parse_args()

    if not args.inference_results.is_file():
        print(f"ERROR: {args.inference_results} not found", file=sys.stderr)
        return 2

    results = json.loads(args.inference_results.read_text())
    print(f"Loaded {len(results)} clip results from {args.inference_results}")

    agg = aggregate(results)
    md = render_markdown(agg, total_clips=len(results))

    out_path = args.output or args.inference_results.with_name("overlap_hedging.md")
    out_path.write_text(md)
    print(f"Wrote table → {out_path}")
    print()
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
