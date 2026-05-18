#!/usr/bin/env python3
"""Faithfulness study — mask the top-K attended audio frames per section,
re-run inference, measure SFS drop.

This is the causal companion to scripts/extract_attention.py. The attention
maps from extract_attention.py are *correlational* — they show where the
model looked, but not whether what it looked at actually drove the answer.
This script proves causality: if you mask the attended region for a section
and the SFS for that section drops more than under random masking, the
attention was causally responsible.

Algorithm
---------
For each clip with an attention JSON, and for each section in that JSON:

  1. Identify the top-K attended prefix-token indices (K configurable,
     default 5 ≈ 0.8 seconds of audio).
  2. Map prefix-token indices → audio-feature frame indices via the
     adapter's 8x compression: prefix_token_i ↔ wavlm_frames [8i, 8i+8).
  3. Zero out the corresponding rows of audio_features.
  4. Re-run inference with the masked audio features.
  5. Parse the new generation and score the corresponding section's SFS
     features (e.g., for section=noise, score 'snr'; for section=pitch,
     score 'f0_mean' + 'f0_sd'; etc.).
  6. Compare per-section SFS to:
        a. Original (unmasked) SFS — the masking should HURT
        b. Random-masking baseline (same K, random indices) — the
           attention-guided masking should hurt MORE than random

Output: per-clip JSON {section: {original_sfs, attn_masked_sfs, random_masked_sfs}}
Aggregate markdown table with mean drops per section.

Usage
-----
    python scripts/faithfulness_study.py \
      --config configs/config.psc.emnlp.yaml \
      --checkpoint $SHARED/checkpoints/v7_lora_8b/best.pt \
      --test_dir   $SHARED/data/processed_pyannote/test \
      --attention_dir $SHARED/checkpoints/v7_lora_8b/attention/ \
      --output     $SHARED/checkpoints/v7_lora_8b/faithfulness.json \
      --top_k_mask 5 \
      --random_seed 42
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

# Per-section SFS features — what to score after masking that section
SECTION_TO_SFS_FEATURES = {
    "noise":   ["snr"],
    "reverb":  ["srmr"],
    "pitch":   ["f0_mean", "f0_sd"],
    "tempo":   ["speaking_rate", "articulation_rate"],
    "pauses":  ["pause_count", "pause_rate"],
    "overlap": ["overlap_ratio"],
}


def _topk_indices(attn_vec, k: int) -> list[int]:
    """Top-k indices of a 1D attention vector (list or np array)."""
    import numpy as np
    a = np.asarray(attn_vec, dtype=float)
    if k >= len(a):
        return list(range(len(a)))
    return sorted(np.argsort(-a)[:k].tolist())


def _random_indices(P: int, k: int, rng) -> list[int]:
    """Random k distinct indices from [0, P)."""
    import numpy as np
    if k >= P:
        return list(range(P))
    return sorted(rng.choice(P, size=k, replace=False).tolist())


def _mask_audio_features(audio_features, prefix_token_indices: list[int]):
    """Zero out audio_features rows corresponding to the given prefix tokens.
    1 prefix token = 8 WavLM frames (8x adapter compression)."""
    import torch
    af = audio_features.clone()
    for ti in prefix_token_indices:
        start = ti * 8
        end = start + 8
        af[start:min(end, af.shape[0])] = 0.0
    return af


def _score_section(generated: str, target: str, section: str, parser, scorer) -> float:
    """SFS F1 for a single section's features. Returns 0 if no claims match."""
    target_claims = parser.parse(target)
    gt = {c.feature: c.value for c in target_claims}
    # Restrict GT to features for this section
    gt_filtered = {f: gt[f] for f in SECTION_TO_SFS_FEATURES.get(section, []) if f in gt}
    if not gt_filtered:
        return None
    claims = parser.parse(generated)
    # Restrict claims to the same features so cross-section claims don't pollute
    claims_filtered = [c for c in claims if c.feature in gt_filtered]
    if not claims_filtered:
        return 0.0
    result = scorer.score(claims_filtered, gt_filtered)
    return result["f1"]


def main() -> int:
    import numpy as np
    import torch
    from sfs import HybridClaimParser, SFSScorer

    # Lazy-import inference machinery
    sys.path.insert(0, str(REPO / "scripts"))
    from extract_attention import load_config, load_model

    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--test_dir", type=Path, required=True)
    p.add_argument("--attention_dir", type=Path, required=True,
                   help="Directory of *_attention.json files from extract_attention.py")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--top_k_mask", type=int, default=5,
                   help="Number of prefix tokens to mask per section (default 5 "
                        "= ~0.8s of audio at 8x compression)")
    p.add_argument("--random_seed", type=int, default=42)
    p.add_argument("--max_new_tokens", type=int, default=384)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config, ck = load_config(args.config, args.checkpoint)
    adapter, llm, tokenizer = load_model(config, ck, device)
    prompt_str = config.get("prompt_prose") or config["prompt"]
    prompt_ids = tokenizer(prompt_str, return_tensors="pt").input_ids.to(device)

    parser = HybridClaimParser()
    scorer = SFSScorer()
    rng = np.random.default_rng(args.random_seed)

    # Load descriptions JSON for targets
    desc_path = config.get("descriptions_path")
    descriptions = json.loads(Path(desc_path).read_text()) if desc_path and os.path.exists(desc_path) else {}

    attention_jsons = sorted(args.attention_dir.glob("*_attention.json"))
    print(f"[clips] {len(attention_jsons)} attention JSONs")

    embed_layer = llm.get_input_embeddings()
    per_clip_results = []

    for jp in attention_jsons:
        att = json.loads(jp.read_text())
        fname = att["filename"]
        stem = os.path.splitext(fname)[0]
        pt_path = args.test_dir / f"{stem}.pt"
        if not pt_path.exists():
            print(f"  [skip] {fname} — no .pt")
            continue
        target = descriptions.get(stem)
        if not target:
            print(f"  [skip] {fname} — no target description")
            continue

        cached = torch.load(pt_path, weights_only=False)
        audio_features = cached["audio_features"]
        overlap_info = cached["overlap_info"]
        P = att["n_prefix_tokens"]
        original_generated = att["generated"]

        clip_record = {
            "filename": fname,
            "n_prefix_tokens": P,
            "top_k_mask": args.top_k_mask,
            "original_generated": original_generated,
            "sections": {},
        }

        for section, attn_vec in att["section_attentions"].items():
            # Pick the attended K and a random K
            top_idxs = _topk_indices(attn_vec, args.top_k_mask)
            rand_idxs = _random_indices(P, args.top_k_mask, rng)

            # Score original
            orig_score = _score_section(original_generated, target, section, parser, scorer)

            section_record = {
                "top_attended_prefix_tokens": top_idxs,
                "random_baseline_prefix_tokens": rand_idxs,
                "original_sfs_f1": orig_score,
            }

            # Re-run inference with attention-masked audio + score
            for label, mask_idxs in (("attn", top_idxs), ("random", rand_idxs)):
                masked_af = _mask_audio_features(audio_features, mask_idxs)
                gen_masked = _generate_with_features(
                    masked_af, overlap_info, adapter, llm, tokenizer,
                    prompt_ids, embed_layer, device,
                    max_new_tokens=args.max_new_tokens,
                )
                masked_score = _score_section(gen_masked, target, section, parser, scorer)
                section_record[f"{label}_masked_sfs_f1"] = masked_score
                section_record[f"{label}_masked_generated"] = gen_masked[:240]  # truncate for storage

            # Drop = original - attn_masked, baselined against random_masked
            if (orig_score is not None
                and section_record["attn_masked_sfs_f1"] is not None
                and section_record["random_masked_sfs_f1"] is not None):
                section_record["attn_drop"] = orig_score - section_record["attn_masked_sfs_f1"]
                section_record["random_drop"] = orig_score - section_record["random_masked_sfs_f1"]
                section_record["causal_delta"] = (
                    section_record["attn_drop"] - section_record["random_drop"]
                )

            clip_record["sections"][section] = section_record
            print(f"  {fname[:35]:35s} | {section:8s} "
                  f"orig={orig_score} attn_masked={section_record['attn_masked_sfs_f1']} "
                  f"rand_masked={section_record['random_masked_sfs_f1']}")

        per_clip_results.append(clip_record)

    # Aggregate
    summary = _aggregate(per_clip_results)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(
        {"per_clip": per_clip_results, "summary": summary}, indent=2,
    ))
    print(f"\nSaved → {args.output}")
    _print_summary_table(summary)
    return 0


def _generate_with_features(audio_features, overlap_info, adapter, llm, tokenizer,
                            prompt_ids, embed_layer, device, max_new_tokens: int) -> str:
    """Minimal greedy generation — mirrors src/inference.py without
    section_head / range marker logic (we're targeting the LoRA + no-sections
    headline path)."""
    import torch
    af = audio_features.unsqueeze(0).to(device).to(torch.bfloat16)
    oi = overlap_info.unsqueeze(0).to(device).to(torch.bfloat16)
    with torch.no_grad():
        out = adapter(af, oi)
    prefix_embeds = out[0] if isinstance(out, tuple) else out
    prompt_embeds = embed_layer(prompt_ids)
    inputs_embeds = torch.cat([prefix_embeds, prompt_embeds], dim=1)

    generated_ids: list[int] = []
    past_key_values = None
    with torch.no_grad():
        outputs = llm(inputs_embeds=inputs_embeds, use_cache=True)
    past_key_values = outputs.past_key_values
    next_token_id = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    generated_ids.append(next_token_id.item())

    for _ in range(max_new_tokens - 1):
        next_embeds = embed_layer(next_token_id)
        with torch.no_grad():
            outputs = llm(
                inputs_embeds=next_embeds,
                past_key_values=past_key_values,
                use_cache=True,
            )
        past_key_values = outputs.past_key_values
        next_token_id = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        tok = next_token_id.item()
        generated_ids.append(tok)
        if tok == tokenizer.eos_token_id:
            break

    return tokenizer.decode(generated_ids, skip_special_tokens=True)


def _aggregate(per_clip_results: list[dict]) -> dict:
    """Mean attn_drop, random_drop, causal_delta per section across clips."""
    from collections import defaultdict
    import statistics
    by_section: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {"attn_drop": [], "random_drop": [], "causal_delta": []}
    )
    for clip in per_clip_results:
        for section, rec in clip["sections"].items():
            for k in ("attn_drop", "random_drop", "causal_delta"):
                if k in rec and rec[k] is not None:
                    by_section[section][k].append(rec[k])
    summary = {}
    for section, lists in by_section.items():
        summary[section] = {
            "n": len(lists["attn_drop"]),
            "mean_attn_drop": (statistics.mean(lists["attn_drop"])
                               if lists["attn_drop"] else None),
            "mean_random_drop": (statistics.mean(lists["random_drop"])
                                 if lists["random_drop"] else None),
            "mean_causal_delta": (statistics.mean(lists["causal_delta"])
                                  if lists["causal_delta"] else None),
        }
    return summary


def _print_summary_table(summary: dict) -> None:
    print("\n" + "=" * 70)
    print(f"{'Section':10s} | {'n':>3s} | {'attn_drop':>10s} | "
          f"{'rand_drop':>10s} | {'causal_Δ':>10s}")
    print("-" * 70)
    for section, m in summary.items():
        a = m["mean_attn_drop"]; r = m["mean_random_drop"]; c = m["mean_causal_delta"]
        a_s = f"{a:.3f}" if a is not None else "  —"
        r_s = f"{r:.3f}" if r is not None else "  —"
        c_s = f"{c:.3f}" if c is not None else "  —"
        print(f"{section:10s} | {m['n']:>3d} | {a_s:>10s} | {r_s:>10s} | {c_s:>10s}")
    print("=" * 70)
    print("Positive causal_Δ = attention-masking hurt SFS more than random-masking")
    print("  → the attended region was causally responsible for that section's claim")


if __name__ == "__main__":
    raise SystemExit(main())
