"""Merge inference_results.json from range-sharded inference runs and recompute
the aggregate summary EXACTLY as src/inference.py does (mean of per-clip P/R/F1,
summed per-feature counts, corpus BLEU/ROUGE-L/BERTScore over present pairs).

Used when a single test set is split across concurrent --start/--end shards
(separate save_dirs) for throughput. Writes the merged inference_results.json +
inference_summary.json to --output_dir.

Usage:
  python scripts/merge_shard_results.py \
    --results A/inference_results.json B/inference_results.json \
    --output_dir A_merged
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from text_metrics import compute_generation_metrics  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", nargs="+", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--test_dir", default="")
    args = ap.parse_args()

    merged, seen = [], set()
    for path in args.results:
        for e in json.load(open(path)):
            fn = e.get("filename")
            if fn in seen:
                continue
            seen.add(fn)
            merged.append(e)
    print(f"merged {len(merged)} unique clips from {len(args.results)} shards")

    scored = [e for e in merged if "sfs_f1" in e]
    summary = {"test_dir": args.test_dir, "n_samples": len(merged)}
    if scored:
        summary["sfs_precision"] = sum(e.get("sfs_precision", 0.0) for e in scored) / len(scored)
        summary["sfs_recall"] = sum(e.get("sfs_recall", 0.0) for e in scored) / len(scored)
        summary["sfs_f1"] = sum(e.get("sfs_f1", 0.0) for e in scored) / len(scored)
        summary["n_scored"] = len(scored)
        fc, ft = {}, {}
        for e in scored:
            for feat in e.get("per_feature", []):
                name = feat["feature"]
                ft[name] = ft.get(name, 0) + 1
                if feat["correct"]:
                    fc[name] = fc.get(name, 0) + 1
        summary["per_feature_accuracy"] = {
            n: {"correct": fc.get(n, 0), "total": ft[n], "accuracy": fc.get(n, 0) / ft[n]}
            for n in sorted(ft)
        }

    paired = [(e["generated"], e.get("target", "")) for e in merged if e.get("target")]
    if paired:
        hyps, refs = zip(*paired)
        gm = compute_generation_metrics(list(hyps), list(refs), use_bertscore=True)
        summary["gen_metrics"] = {**gm, "n_paired": len(paired)}

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "inference_results.json"), "w") as f:
        json.dump(merged, f, indent=2)
    with open(os.path.join(args.output_dir, "inference_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    sys.exit(main())
