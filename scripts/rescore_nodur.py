#!/usr/bin/env python3
"""Re-score an existing inference_results.json against the no-duration target.

Generation is greedy and target-independent, so the generated text in
inference_results.json is final. Only the ground truth and the BLEU/ROUGE
reference change, so we can recompute every metric WITHOUT re-running the model.

Fixes the train/inference target skew: inference scored against
descriptions_untagged_noseg.json, which still contains a leading duration
sentence ("The recording is X s long."), even though the model was trained on
the no-duration target. That skew (a) put duration_sec in the SFS recall
denominator where it can never be matched, depressing recall/F1 for every
variant, and (b) left an unmatched duration sentence in the BLEU/ROUGE/BERTScore
reference. Re-scoring against the no-duration target removes both, raising every
variant's numbers consistently and honestly.

Reuses the same ClaimParser / SFSScorer / compute_generation_metrics that
inference.py uses, and the same per-clip averaging, so the output is directly
comparable to a fresh inference run.

Usage
-----
    python scripts/rescore_nodur.py \
      --inference_results $SHARED/checkpoints/v9_film_attn/inference_results.json \
      --descriptions      $SHARED/data/descriptions_untagged_noseg_nodur.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from sfs import ClaimParser, SFSScorer          # noqa: E402
from text_metrics import compute_generation_metrics  # noqa: E402


def stem_of(filename: str) -> str:
    return os.path.splitext(os.path.basename(filename))[0]


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inference_results", required=True,
                    help="A variant's inference_results.json (has per-clip 'generated').")
    ap.add_argument("--descriptions", required=True,
                    help="No-duration descriptions JSON {stem: target_text} = GT + BLEU reference.")
    ap.add_argument("--output_summary", default=None,
                    help="Where to write the corrected summary "
                         "(default: inference_summary_rescored.json next to the input).")
    ap.add_argument("--no_bertscore", action="store_true",
                    help="Skip BERTScore (faster).")
    args = ap.parse_args()

    results = json.loads(open(args.inference_results).read())
    desc = json.loads(open(args.descriptions).read())   # {stem: target_text}

    parser = ClaimParser()
    scorer = SFSScorer()

    per_clip = []
    feature_correct: dict[str, int] = {}
    feature_total: dict[str, int] = {}
    paired: list[tuple[str, str]] = []
    n_missing = 0

    for e in results:
        gen = e.get("generated", "")
        target = desc.get(stem_of(e["filename"]))
        if target is None:
            n_missing += 1
            continue
        ground_truth = {c.feature: c.value for c in parser.parse(target)}
        if not ground_truth:
            continue
        r = scorer.score(parser.parse(gen), ground_truth)
        per_clip.append(r)
        for feat in r["per_feature"]:
            name = feat["feature"]
            feature_total[name] = feature_total.get(name, 0) + 1
            if feat["correct"]:
                feature_correct[name] = feature_correct.get(name, 0) + 1
        paired.append((gen, target))

    summary: dict = {
        "source": os.path.abspath(args.inference_results),
        "target": os.path.abspath(args.descriptions),
        "n_samples": len(results),
        "n_scored": len(per_clip),
    }
    if per_clip:
        summary["sfs_precision"] = sum(r["precision"] for r in per_clip) / len(per_clip)
        summary["sfs_recall"] = sum(r["recall"] for r in per_clip) / len(per_clip)
        summary["sfs_f1"] = sum(r["f1"] for r in per_clip) / len(per_clip)
        summary["per_feature_accuracy"] = {
            n: {"correct": feature_correct.get(n, 0),
                "total": feature_total[n],
                "accuracy": feature_correct.get(n, 0) / feature_total[n]}
            for n in sorted(feature_total)
        }
    if paired:
        hyps, refs = zip(*paired)
        gm = compute_generation_metrics(list(hyps), list(refs),
                                        use_bertscore=not args.no_bertscore)
        summary["gen_metrics"] = {**gm, "n_paired": len(paired)}

    out = args.output_summary or os.path.join(
        os.path.dirname(args.inference_results), "inference_summary_rescored.json")
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Re-scored {len(per_clip)} clips ({n_missing} missing a target in the descriptions file)")
    if per_clip:
        print(f"  SFS   P={summary['sfs_precision']:.4f}  "
              f"R={summary['sfs_recall']:.4f}  F1={summary['sfs_f1']:.4f}")
    if "gen_metrics" in summary:
        g = summary["gen_metrics"]
        print(f"  BLEU={g.get('bleu')}  ROUGE-L={g.get('rouge_l')}  BERTScore={g.get('bertscore_f1')}")
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
