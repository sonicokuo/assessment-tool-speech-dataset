#!/usr/bin/env python3
"""SFS-vs-LLM-judge correlation study — validates the SFS metric.

When you propose a new automatic metric for a paper, reviewers want to
know it correlates with an independent faithfulness judgement. The
reported rho=0.69 result was produced with an LLM judge (Claude), NOT
human raters. This script supports a two-phase workflow:

  PHASE 1 (--mode prepare):
    Sample N clips from inference_results.json, write a rating-ready CSV
    with one row per clip: [filename, target, generated, sfs_f1, judge_rating].
    The judge_rating column is blank. An LLM judge (Claude) fills it in
    with a 1-5 faithfulness score using the rubric printed by the script.

  PHASE 2 (--mode analyze):
    Read the filled CSV. Compute Spearman + Pearson correlation between
    SFS-F1 and the judge rating. Plot a scatter and save a markdown
    summary suitable for the paper's "metric validity" subsection.

NOTE: do NOT tune the SFS tolerances to maximize this correlation. The
judge is itself an LLM, so optimizing tolerances against it is circular.
Tolerances must be set from first principles (estimator disagreement /
measurement-noise floors), independent of the judge.

Rating rubric (printed by --mode prepare):
  5 — Every numerical claim is correct and the prose accurately describes
      what would be in the audio. No hallucinations.
  4 — All major claims are correct; minor numerical slips or omissions
      that don't change the qualitative assessment.
  3 — Roughly correct on half of features. Some salient errors but
      generally pointed in the right direction.
  2 — More wrong than right. Several major numerical errors and/or
      off-topic content.
  1 — Largely unrelated to the recording. Hallucinated values, wrong
      topic, or degenerate output.

Usage
-----
    # Phase 1 — prepare 50 clips for rating
    python scripts/sfs_judge_correlation.py --mode prepare \
      --inference_results $SHARED/checkpoints/v7_lora_8b/inference_results.json \
      --output_csv        $SHARED/checkpoints/v7_lora_8b/judge_ratings.csv \
      --n 50 --seed 42

    # Fill the judge_rating column (integer 1-5) with the LLM judge's scores

    # Phase 2 — analyze
    python scripts/sfs_judge_correlation.py --mode analyze \
      --rated_csv  $SHARED/checkpoints/v7_lora_8b/judge_ratings.csv \
      --output_md  $SHARED/checkpoints/v7_lora_8b/sfs_judge_correlation.md
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


RUBRIC = """\
Rate the FAITHFULNESS of each generated description on a 1-5 scale:
  5 — Every numerical claim correct; no hallucinations.
  4 — All major claims correct; minor numerical slips that don't change
      the qualitative assessment.
  3 — Roughly half right. Salient errors but pointed in the right
      direction.
  2 — More wrong than right; several major numerical errors and/or
      off-topic content.
  1 — Largely unrelated to the recording. Hallucinations or degenerate.

Compare 'target' (ground truth) to 'generated' (model output). Fill the
'judge_rating' column with an integer 1-5.
"""


def prepare(args) -> int:
    """Sample N clips from inference_results.json and write a rating CSV."""
    import random
    results = json.loads(args.inference_results.read_text())
    scored = [r for r in results if r.get("target") and "sfs_f1" in r]
    if len(scored) < args.n:
        print(f"WARN: only {len(scored)} scored clips available, want {args.n}")
    random.Random(args.seed).shuffle(scored)
    sample = scored[:args.n]
    # Stratify by sfs_f1 quartiles so the judge sees a range of quality
    sample.sort(key=lambda r: r["sfs_f1"])
    print(RUBRIC)
    print(f"Writing {len(sample)} clips to {args.output_csv}")
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["filename", "sfs_f1", "target", "generated", "judge_rating"])
        for r in sample:
            w.writerow([
                r["filename"],
                f"{r['sfs_f1']:.3f}",
                r["target"],
                r["generated"],
                "",  # to be filled by the LLM judge
            ])
    print(f"Done. Fill the 'judge_rating' column (1-5) with the LLM judge's "
          f"scores, then run --mode analyze.")
    return 0


def analyze(args) -> int:
    """Load the rated CSV and compute correlation statistics."""
    try:
        import scipy.stats as st
    except ImportError:
        print("ERROR: scipy not available. Install: pip install scipy")
        return 2

    rows = list(csv.DictReader(args.rated_csv.open()))
    rated = []
    for r in rows:
        try:
            sfs = float(r["sfs_f1"])
            judge = float(r["judge_rating"])
            rated.append({"filename": r["filename"], "sfs": sfs, "judge": judge,
                          "target": r.get("target", ""), "generated": r.get("generated", "")})
        except (ValueError, KeyError):
            continue
    if len(rated) < 5:
        print(f"ERROR: only {len(rated)} clips with valid judge_rating. Need ≥5.")
        return 2
    sfs = [r["sfs"] for r in rated]
    judge = [r["judge"] for r in rated]
    n = len(rated)

    spearman = st.spearmanr(sfs, judge)
    pearson = st.pearsonr(sfs, judge)

    # Per-bin means: bucket SFS into thirds, report mean judge rating per third
    sorted_idx = sorted(range(n), key=lambda i: sfs[i])
    third = n // 3
    bins = [sorted_idx[:third], sorted_idx[third:2*third], sorted_idx[2*third:]]
    bin_stats = []
    for i, idxs in enumerate(bins):
        if not idxs:
            continue
        bin_sfs = [sfs[j] for j in idxs]
        bin_judge = [judge[j] for j in idxs]
        bin_stats.append({
            "name": ["low SFS", "mid SFS", "high SFS"][i],
            "n": len(idxs),
            "sfs_range": (min(bin_sfs), max(bin_sfs)),
            "mean_judge": sum(bin_judge) / len(bin_judge),
        })

    md = [
        "# SFS-vs-LLM-judge correlation",
        "",
        "Ratings were produced by an LLM judge (Claude), not human raters.",
        "",
        f"**N rated clips:** {n}",
        "",
        "## Correlation",
        "",
        f"- **Spearman ρ** = {spearman.statistic:+.3f}  (p = {spearman.pvalue:.4f})",
        f"- **Pearson  r** = {pearson.statistic:+.3f}  (p = {pearson.pvalue:.4f})",
        "",
        "## Mean judge rating per SFS tercile",
        "",
        "| Bucket | n | SFS range | Mean judge rating |",
        "|---|---:|---|---:|",
    ]
    for b in bin_stats:
        md.append(f"| {b['name']} | {b['n']} | {b['sfs_range'][0]:.3f}–{b['sfs_range'][1]:.3f} | {b['mean_judge']:.2f} |")
    md.extend([
        "",
        "## Interpretation",
        "",
        "- Spearman ρ measures monotonic agreement (rank-based, robust to outliers).",
        "  Above 0.4 = meaningful; above 0.6 = strong correlation; above 0.8 = very strong.",
        "- Pearson r assumes a linear relationship.",
        "- The terciles table shows whether the metric tracks the judge's rating",
        "  even at the coarse high/mid/low level — a strict monotone increase in",
        "  mean judge rating across terciles is the cleanest qualitative evidence.",
        "- The judge is an LLM, so SFS tolerances must NOT be tuned to maximize this",
        "  correlation (that would be circular). Tolerances come from first principles.",
    ])

    if args.output_md:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_md.write_text("\n".join(md))
        print(f"Saved → {args.output_md}\n")
    print("\n".join(md))
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=["prepare", "analyze"], required=True)
    p.add_argument("--inference_results", type=Path,
                   help="(prepare) Path to inference_results.json")
    p.add_argument("--output_csv", type=Path,
                   help="(prepare) Where to write the rating-ready CSV")
    p.add_argument("--n", type=int, default=50,
                   help="(prepare) Number of clips to sample (default 50)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--rated_csv", type=Path,
                   help="(analyze) Path to the filled-in CSV")
    p.add_argument("--output_md", type=Path,
                   help="(analyze) Optional output markdown path")
    args = p.parse_args()

    if args.mode == "prepare":
        if not args.inference_results or not args.output_csv:
            print("ERROR: --mode prepare requires --inference_results and --output_csv")
            return 2
        return prepare(args)
    if args.mode == "analyze":
        if not args.rated_csv:
            print("ERROR: --mode analyze requires --rated_csv")
            return 2
        return analyze(args)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
