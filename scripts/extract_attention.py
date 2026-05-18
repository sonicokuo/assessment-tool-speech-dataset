#!/usr/bin/env python3
"""Post-hoc per-section attention extraction from a trained checkpoint.

This is the paper's evidence path: train the model on plain untagged prose
(no section_head, no special tokens), then at inference time recover one
attention vector per section by parsing the generated prose for section
spans and aggregating the LM's NATIVE attention layers over those spans.

Algorithm
---------
  1. Load checkpoint exactly the way src/inference.py does (same adapter,
     same LoRA wrap, same LM weights). Force `attn_implementation="eager"`
     so the forward returns attention matrices (flash_attention_2 doesn't).
  2. For one clip:
        prefix_embeds = adapter(audio_features, overlap_info)   # (1, P, d_lm)
        prompt_embeds = embed(prompt_ids)                        # (1, Q, d_lm)
        inputs_embeds = concat([prefix_embeds, prompt_embeds])   # (1, P+Q, d_lm)
     P = number of audio prefix tokens (~clip_seconds * 6.25 at 8x compression)
     Q = number of prompt tokens (~7 for "Describe the quality of this recording.")
  3. Generate token by token with KV cache + output_attentions=True. At each
     step t, capture the attention from the new token's query position back
     to the audio prefix positions [0:P] across selected layers and all heads.
     Result: attn_per_step[t] shape (P,) — one weight per audio prefix token.
  4. Regex-match section spans in the decoded text:
        noise   → "...SNR is 13.17 dB..."
        reverb  → "...SRMR is 4.20..."
        pitch   → "...F0 mean is 150 Hz..."  (also F0 SD)
        tempo   → "...speaking rate is 6.0 syl/sec..."
        pauses  → "...pause count is 1..."
        overlap → "...overlap ratio is 0.7..."
     For each section's char range, map back to the generated token index
     range, and average attention over those tokens.
  5. Save per-section attention vectors as JSON next to the input audio
     (one file per clip).

Output JSON schema
------------------
    {
      "filename": "1089-134686-0000_121-127105-0031.wav",
      "generated": "The recording is 7.100 s long. ...",
      "n_prefix_tokens": 31,
      "prefix_token_stride_sec": 0.16,
      "audio_duration_sec": 4.96,
      "section_attentions": {
        "noise":   [0.012, 0.034, ..., 0.041],  # length P
        "reverb":  [0.008, 0.022, ...],
        ...
      }
    }

Caveats
-------
- `attn_implementation="eager"` is ~3× slower than flash_attention_2 and
  uses more memory. Fine for a few-clip paper-figure run; don't use it
  in the inference loop over the 3000-clip test set.
- Layer aggregation: top half of layers is used (more semantic per LLaVA
  / GroundLMM findings). Toggle via --layer_selection.
- Mode-collapsing models (e.g. v6 full-FT) may produce off-topic prose
  that the section regex won't match. In that case nothing is saved for
  the missing section and a warning is printed.

Usage
-----
    python scripts/extract_attention.py --config configs/config.psc.emnlp.yaml \
      --checkpoint $SHARED/checkpoints/v7_lora_8b/best.pt \
      --test_dir   $SHARED/data/processed_pyannote/test \
      --filenames  1089-134686-0000_121-127105-0031.wav,1089-134686-0000_260-123440-0016.wav \
      --output_dir $SHARED/checkpoints/v7_lora_8b/attention/
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

# Heavy imports (torch / transformers / peft / adapter) are done lazily inside
# the functions that need them, so this module can be imported for unit-level
# checks (regex / arg parsing) without the full ML stack installed.


# ── Section regex patterns ─────────────────────────────────────────────────
# Match the verbalizer's standard phrasings. Each pattern captures the full
# sentence (up to and including the period) so the char range covers all
# tokens in the section.
#
# `_NOT_END` matches any character that is not a sentence-terminating period
# (allows periods inside decimal numbers like "4.2927"). `_END` matches a
# period that genuinely ends a sentence (period not followed by a digit).
_NOT_END = r"(?:[^.]|\.(?=\d))"
_END = r"\.(?!\d)"
SECTION_PATTERNS = {
    "noise":   rf"{_NOT_END}*?(?:signal-to-noise|SNR){_NOT_END}*?{_END}",
    "reverb":  rf"{_NOT_END}*?SRMR{_NOT_END}*?{_END}",
    "pitch":   rf"{_NOT_END}*?F0\s+(?:mean|standard\s+deviation|SD){_NOT_END}*?{_END}",
    "tempo":   rf"{_NOT_END}*?(?:speaking|articulation)\s+rate{_NOT_END}*?{_END}",
    "pauses":  rf"{_NOT_END}*?pause\s+(?:count|rate){_NOT_END}*?{_END}",
    "overlap": rf"{_NOT_END}*?overlap\s+ratio{_NOT_END}*?{_END}",
}


def load_config(config_path: Path, checkpoint_path: Path) -> dict:
    """YAML config + checkpoint config overrides — mirrors src/inference.py.

    The checkpoint embeds the config it was trained with; that takes
    precedence over the YAML for model-shape fields (lm_name, lora_rank,
    tagged_mode, adapter_variant) so we always rebuild the LM in the exact
    shape the checkpoint expects.
    """
    import torch
    import yaml
    config = yaml.safe_load(config_path.read_text())
    ck = torch.load(checkpoint_path, weights_only=False, map_location="cpu")
    ck_config = ck.get("config", {})
    for key in ("lm_name", "lora_rank", "lora_alpha", "lora_targets",
                "lora_dropout", "tagged_mode", "adapter_variant"):
        if key in ck_config and ck_config[key] != config.get(key):
            print(f"[config] {key}: {config.get(key)!r} → {ck_config[key]!r} (from checkpoint)")
            config[key] = ck_config[key]
    return config, ck


def load_model(config: dict, ck: dict, device):
    """Adapter + LM + tokenizer, set up for attention extraction.

    Differs from src/inference.py only in `attn_implementation="eager"`
    (required so the forward pass returns attention matrices) and the
    absence of section_head loading (we're using post-hoc attention from
    the LM's native layers, not the section_head's cross-attention).
    """
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from adapter import build_adapter
    tokenizer = AutoTokenizer.from_pretrained(config["lm_name"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    llm = AutoModelForCausalLM.from_pretrained(
        config["lm_name"],
        torch_dtype=torch.bfloat16,
        device_map={"": device},
        attn_implementation="eager",   # required for output_attentions
    )

    full_ft = not bool(config.get("lora_rank"))
    if not full_ft:
        llm = get_peft_model(
            llm,
            LoraConfig(
                r=config["lora_rank"],
                lora_alpha=config["lora_alpha"],
                target_modules=config["lora_targets"],
                lora_dropout=config["lora_dropout"],
                bias="none",
                task_type="CAUSAL_LM",
            ),
        )
        print(f"[LoRA] rank={config['lora_rank']}")

    lm_hidden_size = llm.config.hidden_size
    adapter = (
        build_adapter(config["adapter_variant"], lm_dim=lm_hidden_size)
        .to(device)
        .to(torch.bfloat16)
    )
    adapter.load_state_dict(ck["adapter_state_dict"])
    llm_sd = ck.get("llm_state_dict") or ck["lora_state_dict"]
    llm.load_state_dict(llm_sd)
    adapter.eval()
    llm.eval()
    print(f"[ckpt] epoch={ck.get('epoch')}  val_sfs_f1={ck.get('best_val_sfs_f1', 'n/a')}")
    return adapter, llm, tokenizer


def extract_one_clip(
    sample: dict,
    adapter, llm, tokenizer,
    prompt_ids,
    device,
    max_new_tokens: int,
    layer_selection: str,
) -> dict:
    """Run generation + attention capture for a single clip.

    Returns a dict with the JSON schema documented in the module docstring.
    """
    import torch
    audio_features = sample["audio_features"].unsqueeze(0).to(device).to(torch.bfloat16)
    overlap_info = sample["overlap_info"].unsqueeze(0).to(device).to(torch.bfloat16)

    with torch.no_grad():
        out = adapter(audio_features, overlap_info)
    prefix_embeds = out[0] if isinstance(out, tuple) else out
    P = prefix_embeds.shape[1]

    embed_layer = llm.get_input_embeddings()
    prompt_embeds = embed_layer(prompt_ids)
    Q = prompt_embeds.shape[1]

    inputs_embeds = torch.cat([prefix_embeds, prompt_embeds], dim=1)

    # Step-by-step generation with attention capture. Greedy (top-1) for
    # deterministic paper-figure outputs.
    generated_ids: list[int] = []
    attn_per_step: list[torch.Tensor] = []   # each shape (P,)
    past_key_values = None

    # First forward processes prefix + prompt at once
    with torch.no_grad():
        outputs = llm(
            inputs_embeds=inputs_embeds,
            use_cache=True,
            output_attentions=True,
        )
    past_key_values = outputs.past_key_values
    next_token_id = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    generated_ids.append(next_token_id.item())

    # Capture attention at the very last query position (the one that
    # produced this first generated token). All_attentions is a tuple
    # of n_layers tensors each (B, n_heads, query_len, key_len).
    n_layers = len(outputs.attentions)
    if layer_selection == "top_half":
        selected = list(range(n_layers // 2, n_layers))
    elif layer_selection == "all":
        selected = list(range(n_layers))
    elif layer_selection == "last_quarter":
        selected = list(range(3 * n_layers // 4, n_layers))
    else:
        raise ValueError(f"Unknown layer_selection={layer_selection!r}")

    attn_per_step.append(
        _avg_attention_to_prefix(outputs.attentions, P, selected,
                                 query_idx=-1)
    )

    # Subsequent generation
    for _ in range(max_new_tokens - 1):
        next_embeds = embed_layer(next_token_id)
        with torch.no_grad():
            outputs = llm(
                inputs_embeds=next_embeds,
                past_key_values=past_key_values,
                use_cache=True,
                output_attentions=True,
            )
        past_key_values = outputs.past_key_values
        next_token_id = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        tok = next_token_id.item()
        generated_ids.append(tok)

        # At subsequent steps the query is just the new token (length 1).
        attn_per_step.append(
            _avg_attention_to_prefix(outputs.attentions, P, selected, query_idx=0)
        )

        if tok == tokenizer.eos_token_id:
            break

    text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    # Match section spans → token index ranges → average attention
    section_attentions: dict[str, list[float]] = {}
    for section_name, pattern in SECTION_PATTERNS.items():
        match = re.search(pattern, text, re.IGNORECASE)
        if not match:
            print(f"  [warn] no match for section={section_name}")
            continue
        # Char range → token range. We re-encode the prose prefix because
        # token boundaries inside the decoded text are not 1:1 with
        # generated_ids (BPE merging can stitch token boundaries).
        prefix_text = text[:match.start()]
        full_text = text[:match.end()]
        n_pre = len(tokenizer.encode(prefix_text, add_special_tokens=False))
        n_full = len(tokenizer.encode(full_text, add_special_tokens=False))
        # Clamp to attn_per_step length in case of off-by-one from BPE.
        start_tok = max(0, min(n_pre, len(attn_per_step) - 1))
        end_tok = max(start_tok + 1, min(n_full, len(attn_per_step)))
        span_attns = attn_per_step[start_tok:end_tok]
        if not span_attns:
            continue
        section_attentions[section_name] = (
            torch.stack(span_attns).mean(dim=0).cpu().float().tolist()
        )

    wavlm_frame_rate_hz = 50.0
    audio_duration_sec = sample["audio_features"].shape[0] / wavlm_frame_rate_hz
    return {
        "filename": sample["filename"],
        "generated": text,
        "n_prefix_tokens": P,
        "prefix_token_stride_sec": 0.16,   # 8 WavLM frames * 20ms
        "audio_duration_sec": audio_duration_sec,
        "n_selected_layers": len(selected),
        "selected_layers": selected,
        "section_attentions": section_attentions,
    }


def _avg_attention_to_prefix(
    layer_attentions, P: int, selected_layers: list[int], query_idx: int,
):
    """Average attention across heads + selected layers, restricted to the
    audio-prefix key positions [0:P]. Returns (P,) on the original device."""
    import torch
    per_layer = []
    for layer_idx in selected_layers:
        layer_attn = layer_attentions[layer_idx]   # (B=1, n_heads, query, key)
        # query_idx: -1 (last) for prefill, 0 for incremental generation
        attn_to_prefix = layer_attn[0, :, query_idx, :P]   # (n_heads, P)
        per_layer.append(attn_to_prefix.mean(dim=0))       # (P,) avg over heads
    return torch.stack(per_layer).mean(dim=0)              # (P,) avg over layers


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--test_dir", type=Path, required=True,
                   help="Dir of preprocessed .pt files (output of src/preprocess.py)")
    p.add_argument("--filenames", type=str, required=True,
                   help="Comma-separated .wav filenames to process (e.g., "
                        "'1089-134686-0000_121-127105-0031.wav,...')")
    p.add_argument("--output_dir", type=Path, required=True)
    p.add_argument("--max_new_tokens", type=int, default=384)
    p.add_argument("--layer_selection", default="top_half",
                   choices=["top_half", "all", "last_quarter"],
                   help="Which transformer layers to average attention over. "
                        "top_half is the LLaVA / GroundLMM default — upper layers "
                        "carry more semantic information.")
    args = p.parse_args()

    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")

    config, ck = load_config(args.config, args.checkpoint)
    adapter, llm, tokenizer = load_model(config, ck, device)

    prompt_str = config.get("prompt_prose") or config["prompt"]
    prompt_ids = tokenizer(prompt_str, return_tensors="pt").input_ids.to(device)
    print(f"[prompt-prose] {prompt_str!r}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    requested = [f.strip() for f in args.filenames.split(",") if f.strip()]
    for fname in requested:
        stem = os.path.splitext(fname)[0]
        pt_path = args.test_dir / f"{stem}.pt"
        if not pt_path.exists():
            print(f"  [skip] {fname} — not found at {pt_path}")
            continue
        print(f"\n=== {fname} ===")
        cached = torch.load(pt_path, weights_only=False)
        sample = {
            "audio_features": cached["audio_features"],
            "overlap_info": cached["overlap_info"],
            "filename": cached.get("filename", fname),
        }
        result = extract_one_clip(
            sample, adapter, llm, tokenizer, prompt_ids, device,
            max_new_tokens=args.max_new_tokens,
            layer_selection=args.layer_selection,
        )
        print(f"  generated: {result['generated'][:150]}...")
        print(f"  sections captured: {sorted(result['section_attentions'].keys())}")
        out_path = args.output_dir / f"{stem}__attention.json"
        out_path.write_text(json.dumps(result, indent=2))
        print(f"  saved → {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
