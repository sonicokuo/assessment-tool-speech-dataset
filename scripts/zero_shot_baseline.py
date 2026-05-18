#!/usr/bin/env python3
"""Zero-shot LM baseline — no audio prefix, no adapter, no LoRA.

Runs the bare Qwen3-8B (instruct-tuned) on each test clip with just the
prose prompt — NO audio conditioning. Generates descriptions, scores them
with SFS against the same ground truth. Reveals the floor that any audio
adapter must beat. Reviewers will ask "is the model just memorizing the
training distribution?" — this baseline is the answer.

Output format matches src/inference.py's inference_results.json so the
existing analysis scripts (per_feature_sfs.py, overlap_hedging_compare.py)
can run on it unchanged.

Usage
-----
    python scripts/zero_shot_baseline.py --config configs/config.psc.emnlp.yaml \
      --test_dir   $SHARED/data/processed_pyannote/test \
      --output_dir $SHARED/checkpoints/zero_shot_baseline/ \
      --top_k 1

(Note: --test_dir is read for filenames only — the audio is ignored.)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))


def main() -> int:
    import torch
    import yaml
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from sfs import HybridClaimParser, SFSScorer
    from text_metrics import compute_generation_metrics

    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--test_dir", type=Path, required=True,
                   help="Preprocessed .pt directory — used only for filenames "
                        "(and target_text via descriptions_path) to mirror inference.py")
    p.add_argument("--output_dir", type=Path, required=True)
    p.add_argument("--lm_name", type=str, default=None,
                   help="Override LM name (default: read from config)")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end", type=int, default=None)
    p.add_argument("--max_new_tokens", type=int, default=384)
    p.add_argument("--top_k", type=int, default=1)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top_p", type=float, default=1.0)
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    config = yaml.safe_load(args.config.read_text())
    lm_name = args.lm_name or config["lm_name"]
    print(f"[zero-shot] LM = {lm_name}")
    print(f"[zero-shot] NO audio adapter, NO LoRA, NO checkpoint — pure pretrained LM")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")

    tokenizer = AutoTokenizer.from_pretrained(lm_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    llm = AutoModelForCausalLM.from_pretrained(
        lm_name, torch_dtype=torch.bfloat16, device_map={"": device},
    )
    llm.eval()

    # Same prose prompt the trained model sees at inference.
    prompt_str = config.get("prompt_prose") or config["prompt"]
    print(f"[prompt-prose] {prompt_str!r}")
    prompt_ids = tokenizer(prompt_str, return_tensors="pt").input_ids.to(device)

    # Load descriptions JSON for targets / SFS scoring.
    descriptions: dict[str, str] = {}
    desc_path = config.get("descriptions_path")
    if desc_path and os.path.exists(desc_path):
        descriptions = json.loads(Path(desc_path).read_text())

    # Walk the test_dir for filenames (audio not loaded — zero-shot).
    pt_files = sorted([f for f in os.listdir(args.test_dir) if f.endswith(".pt")])
    end_idx = args.end if args.end is not None else len(pt_files)
    pt_files = pt_files[args.start:end_idx]
    print(f"[clips] processing {len(pt_files)} clips (indices {args.start}:{end_idx})")

    output_path = args.output_dir / "inference_results.json"
    summary_path = args.output_dir / "inference_summary.json"

    # Resume support — skip clips already in the results file.
    all_outputs: list = []
    done = set()
    if output_path.exists():
        all_outputs = json.loads(output_path.read_text())
        done = {e["filename"] for e in all_outputs if "filename" in e}
        print(f"[resume] {len(done)} clips already done")

    parser = HybridClaimParser()
    scorer = SFSScorer()

    t0 = time.time()
    for i, pt_name in enumerate(pt_files):
        stem = os.path.splitext(pt_name)[0]
        filename = stem + ".wav"
        if filename in done:
            continue

        # Greedy generation from the prompt alone — no audio context whatsoever
        with torch.no_grad():
            out = llm.generate(
                input_ids=prompt_ids,
                max_new_tokens=args.max_new_tokens,
                do_sample=(args.top_k != 1 or args.top_p < 1.0),
                temperature=args.temperature,
                top_k=args.top_k if args.top_k > 0 else None,
                top_p=args.top_p,
                pad_token_id=tokenizer.pad_token_id,
            )
        generated_ids = out[0, prompt_ids.shape[1]:]
        generated = tokenizer.decode(generated_ids, skip_special_tokens=True)

        entry = {"filename": filename, "generated": generated}
        target = descriptions.get(stem)
        if target:
            entry["target"] = target
            gt_claims = parser.parse(target)
            ground_truth = {c.feature: c.value for c in gt_claims}
            claims = parser.parse(generated)
            result = scorer.score(claims, ground_truth)
            entry["sfs_precision"] = result["precision"]
            entry["sfs_recall"] = result["recall"]
            entry["sfs_f1"] = result["f1"]
            entry["per_feature"] = result["per_feature"]

        all_outputs.append(entry)

        if (i + 1) % 25 == 0:
            tmp = output_path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(all_outputs, indent=2))
            tmp.replace(output_path)
            elapsed = time.time() - t0
            print(f"  {i+1}/{len(pt_files)} done  ({elapsed:.1f}s elapsed)")

    # Final flush
    tmp = output_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(all_outputs, indent=2))
    tmp.replace(output_path)

    # Aggregate summary — same shape as src/inference.py's summary
    scored = [e for e in all_outputs if "sfs_f1" in e]
    summary: dict = {
        "lm_name": lm_name,
        "n_samples": len(all_outputs),
        "n_scored": len(scored),
        "baseline": "zero-shot pretrained LM (no audio prefix)",
    }
    if scored:
        summary["sfs_precision"] = sum(e["sfs_precision"] for e in scored) / len(scored)
        summary["sfs_recall"] = sum(e["sfs_recall"] for e in scored) / len(scored)
        summary["sfs_f1"] = sum(e["sfs_f1"] for e in scored) / len(scored)
        print(f"\n{'='*60}")
        print(f"Zero-shot baseline ({lm_name})")
        print(f"  SFS Precision: {summary['sfs_precision']:.4f}")
        print(f"  SFS Recall:    {summary['sfs_recall']:.4f}")
        print(f"  SFS F1:        {summary['sfs_f1']:.4f}")
        print(f"  Scored:        {len(scored)} / {len(all_outputs)}")
        print(f"{'='*60}")

    paired = [(e["generated"], e.get("target", "")) for e in all_outputs if e.get("target")]
    if paired:
        hyps, refs = zip(*paired)
        gen_metrics = compute_generation_metrics(
            list(hyps), list(refs),
            use_bertscore=config.get("use_bertscore", False),
        )
        summary["gen_metrics"] = gen_metrics
        if gen_metrics.get("bleu") is not None:
            print(f"  BLEU-4:        {gen_metrics['bleu']:.2f}")
        if gen_metrics.get("rouge_l") is not None:
            print(f"  ROUGE-L (F1):  {gen_metrics['rouge_l']:.4f}")

    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\nResults: {output_path}")
    print(f"Summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
