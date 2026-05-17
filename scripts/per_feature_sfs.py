#!/usr/bin/env python3
"""Per-feature SFS breakdown for paper analysis.

Consumes the inference_results.json produced by src/inference.py and computes
per-feature precision / recall / F1 — the table that goes into the paper's
'failure-mode analysis' section.

Why this exists
---------------
inference.py already reports an aggregate SFS (precision, recall, F1) plus a
per-feature "accuracy" (correct/mentioned). But aggregate hides which features
the model handles well (SNR, duration) vs which it doesn't (F0 SD, pause_rate),
and the per-feature accuracy is only the precision side — recall is missing.

This script adds:
  - **recall**: of clips where feature X was in GT, how often the model emitted
    a CORRECT claim for X. The denominator comes from re-parsing each target.
  - **mention_rate**: how often the model talked about X at all when X was in
    GT (separates "didn't mention" from "got value wrong").
  - **F1** per feature.

Usage
-----
    python scripts/per_feature_sfs.py \
      --inference_results $SHARED/checkpoints/v7_lora_8b/inference_results.json \
      --output            $SHARED/checkpoints/v7_lora_8b/per_feature_sfs.md
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from sfs import ClaimParser  # noqa: E402


def aggregate(results: list[dict]) -> dict[str, dict]:
    """Aggregate per-feature counts across all clips.

    For each feature `f`, accumulates:
        n_in_gt:    clips where target contained a claim for `f`
        n_mentioned_when_in_gt: clips where (target has `f`) AND (gen has `f`)
        n_correct:  clips where (gen has `f`) AND (claim was within tolerance)

    Returns a dict {feature: {n_in_gt, n_mentioned_when_in_gt, n_correct}}.
    """
    parser = ClaimParser()
    agg: dict[str, dict[str, int]] = defaultdict(
        lambda: {"n_in_gt": 0, "n_mentioned_when_in_gt": 0, "n_correct": 0}
    )

    for entry in results:
        target = entry.get("target") or ""
        # Re-parse target to learn which features GT contained.
        gt_features: set[str] = {c.feature for c in parser.parse(target)}
        # Overlap GT is span-list, scored as overlap_span by SFSScorer; treat it
        # as a "feature" in the table for parity with per_feature entries.
        if any(f in gt_features for f in ("overlap_start", "overlap_end")):
            gt_features.add("overlap_span")
            gt_features.discard("overlap_start")
            gt_features.discard("overlap_end")

        # Tally GT presence
        for f in gt_features:
            agg[f]["n_in_gt"] += 1

        # Tally mentions + correctness from the per_feature scoring entries
        # (only scored when feature was in both gen AND gt — exactly what we want
        # for "mentioned when in gt"). Dedup per-clip in case the model emitted
        # multiple claims for the same feature (e.g., multiple overlap spans).
        mentioned_this_clip: set[str] = set()
        correct_this_clip: set[str] = set()
        for pf in entry.get("per_feature", []):
            f = pf["feature"]
            mentioned_this_clip.add(f)
            if pf.get("correct"):
                correct_this_clip.add(f)
        for f in mentioned_this_clip:
            agg[f]["n_mentioned_when_in_gt"] += 1
        for f in correct_this_clip:
            agg[f]["n_correct"] += 1

    return agg


def compute_metrics(stats: dict[str, int]) -> dict[str, float]:
    """precision / recall / F1 / mention_rate from raw counts.

    precision     = n_correct / n_mentioned_when_in_gt  (of mentions, how often right)
    recall        = n_correct / n_in_gt                 (of GT, how often correctly stated)
    mention_rate  = n_mentioned_when_in_gt / n_in_gt    (of GT, how often talked about at all)
    """
    in_gt = stats["n_in_gt"]
    mentioned = stats["n_mentioned_when_in_gt"]
    correct = stats["n_correct"]
    precision = correct / mentioned if mentioned else 0.0
    recall = correct / in_gt if in_gt else 0.0
    mention_rate = mentioned / in_gt if in_gt else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return {
        "n_in_gt": in_gt,
        "n_mentioned": mentioned,
        "n_correct": correct,
        "mention_rate": mention_rate,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def render_markdown(agg: dict[str, dict], total_clips: int) -> str:
    """Format aggregated stats as a markdown table sorted by F1 descending."""
    rows = [
        (feature, compute_metrics(stats)) for feature, stats in agg.items()
    ]
    rows.sort(key=lambda r: -r[1]["f1"])

    lines = []
    lines.append("# Per-feature SFS breakdown\n")
    lines.append(f"Total clips scored: {total_clips}\n")
    # Flag features that look like they have a parser-coverage problem (high mention
    # rate but n_in_gt=0 is impossible by construction; n_in_gt=0 with no mentions
    # is genuinely missing). The interesting case: n_in_gt much smaller than
    # total_clips for a feature you'd expect in every target — points to a parser
    # regex that doesn't catch the target's phrasing.
    suspicious = [f for f, m in rows if 0 < m["n_in_gt"] < 0.1 * total_clips and m["n_mentioned"] > m["n_in_gt"]]
    if suspicious:
        lines.append(f"⚠ Likely parser-coverage issues (much lower n_in_gt than expected): {', '.join(suspicious)}")
        lines.append("")
    lines.append("| Feature | n in GT | n mentioned | n correct | mention rate | precision | recall | F1 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for feature, m in rows:
        lines.append(
            f"| {feature} | {m['n_in_gt']} | {m['n_mentioned']} | {m['n_correct']} | "
            f"{m['mention_rate']:.3f} | {m['precision']:.3f} | {m['recall']:.3f} | {m['f1']:.3f} |"
        )
    lines.append("")
    lines.append("**Columns:**")
    lines.append("- `n in GT` — clips whose target text mentions this feature (denominator for recall/mention_rate)")
    lines.append("- `n mentioned` — clips where the model emitted a claim for this feature (denominator for precision)")
    lines.append("- `n correct` — of those mentions, how many were within the SFS tolerance for that feature")
    lines.append("- `mention rate = n_mentioned / n_in_gt` — coverage independent of correctness")
    lines.append("- `precision = n_correct / n_mentioned` — accuracy given the model talked about it")
    lines.append("- `recall = n_correct / n_in_gt` — overall correctness across GT")
    lines.append("- `F1` — harmonic mean of precision and recall")
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

    out_path = args.output or args.inference_results.with_name("per_feature_sfs.md")
    out_path.write_text(md)
    print(f"Wrote table → {out_path}")

    # Also dump raw counts as JSON for downstream plotting / paper tables.
    json_path = out_path.with_suffix(".json")
    json_path.write_text(json.dumps(
        {f: compute_metrics(agg[f]) for f in agg}, indent=2,
    ))
    print(f"Wrote raw counts → {json_path}")
    print()
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
