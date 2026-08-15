#!/usr/bin/env python3
"""branch_point_logits.py — WHY does the model silently drop 4 of 11 features? (plan Q4 step 1a)

THE FINDING THIS EXPLAINS
On CLEAN clips the targets state a value for all five voice features (hnr 100%, jitter 100%,
shimmer 100%, f0_mean 98.8%, f0_sd 98.8%), yet the model emits f0_mean 0.1%, f0_sd 0.1%,
shimmer 0.0%, hnr 0.0% — and jitter 94.2%. It emits the hedge sentence on only 5.8% of clean
clips, so it is NOT withholding-with-a-reason: it SILENTLY OMITS. Every data-side explanation is
eliminated by measurement (targets verified, trained from scratch, not truncation, clean clips'
overlap channel genuinely zero).

HYPOTHESIS H1 — ARGMAX COLLAPSE AT A BIMODAL BRANCH POINT
After the srmr clause the training distribution is ~50/50: the ill-posed block (clean half) vs the
hedge sentence (mixture half). If the LM-visible evidence at that token is weak, the highest-
probability SINGLE continuation can be the thing BOTH branches share downstream — the
speaking_rate clause — and greedy decoding converts a plurality into 100% (Holtzman et al.,
ICLR 2020, arXiv:1904.09751). That predicts exactly the observed third path, which appears in
NEITHER target branch.

WHAT THIS SCRIPT MEASURES
Teacher-force the canonical prefix up to the post-srmr branch point on CLEAN clips, then read the
next-token distribution. No generation — fast, and it isolates the decision from everything
downstream.

READING (pre-registered):
  P(hnr-opener) HIGH but not argmax   -> H1: greedy amplification. A DECODE-TIME fix suffices
                                         (sampling, or the slot decoder which teacher-forces every
                                         frame). No retrain needed.
  P(hnr-opener) NEAR ZERO             -> the learned prior itself collapsed; needs the retrain
                                         with sigma-gated targets (plan Q4 step 2).
  P(hedge-opener) high on CLEAN       -> the model believes clean clips are mixtures; that is an
                                         evidence problem, not a decoding one.

The jitter-vs-shimmer contrast is the scalpel: identical target rates, same hedge group, adjacent
clauses, opposite outcomes. Whatever separates them at this branch point is the mechanism.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from data.feature_set import SUPERVISED_FEATURES  # noqa: E402
from model.adapter import build_adapter           # noqa: E402
from slot_decode import slot_frames               # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--test_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=40, help="clean clips to probe")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    import yaml
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cfg_yaml = yaml.safe_load(open(a.config))
    ck = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    cfg = ck.get("config", {}) or {}
    zero_ovl = bool(cfg.get("zero_overlap_input", cfg_yaml.get("zero_overlap_input", False)))
    lm_name = cfg.get("lm_name") or cfg_yaml.get("lm_name")

    tok = AutoTokenizer.from_pretrained(lm_name)
    llm = AutoModelForCausalLM.from_pretrained(lm_name, dtype=torch.bfloat16).to(a.device).eval()
    if ck.get("lora_state_dict"):
        # LoRA weights live in the checkpoint; without them this probes the BASE model and the
        # whole measurement is meaningless.
        from peft import LoraConfig, get_peft_model
        lc = LoraConfig(r=cfg.get("lora_r", 16), lora_alpha=cfg.get("lora_alpha", 32),
                        target_modules=cfg.get("lora_target_modules",
                                               ["q_proj", "k_proj", "v_proj", "o_proj"]),
                        lora_dropout=0.0, task_type="CAUSAL_LM")
        llm = get_peft_model(llm, lc)
        missing, unexpected = llm.load_state_dict(ck["lora_state_dict"], strict=False)
        print(f"[lora] loaded (missing={len(list(missing))} unexpected={len(list(unexpected))})",
              flush=True)

    adapter = build_adapter(
        variant=cfg.get("adapter_variant", "attn-concat"),
        audio_dim=cfg.get("audio_dim", 1024), lm_dim=cfg.get("lm_dim", 4096),
        compression=cfg.get("compression", 8), aux_pool=cfg.get("aux_pool") or "mean",
        reliability_head=bool(cfg.get("reliability_head", False)))
    miss, unexp = adapter.load_state_dict(
        ck.get("adapter_state_dict") or ck.get("adapter") or {}, strict=False)
    if miss or unexp:
        print(f"[fatal] adapter state_dict mismatch missing={list(miss)} unexpected={list(unexp)}")
        return 1
    adapter.eval().to(a.device)
    print(f"[load] zero_overlap_input={zero_ovl}  adapter=EXACT MATCH", flush=True)

    frames = slot_frames()
    short = [f[0] for f in frames]
    # The branch point sits immediately AFTER the srmr clause: the clean branch continues into
    # the ill-posed block, the mixture branch emits the hedge instead.
    i_srmr = short.index("srmr")
    prefix_txt = ""
    for k in range(i_srmr + 1):
        s, fmt, f_prefix, f_suffix = frames[k]
        prefix_txt += ("" if k == 0 else " ") + f_prefix + "0.00" + f_suffix
    print(f"[branch] teacher-forced prefix ends: ...{prefix_txt[-60:]!r}", flush=True)

    # ⚠️ TWO DECISIONS, TWO TOKEN POSITIONS. The first attempt scored the FIRST token of each
    # candidate and got an identical 0.9526 for all of them, because EVERY feature clause opens
    # with the same word ("The ..."). That measured P(" The"), not which feature follows.
    #
    #   position 0 : hedge-vs-continue   ("Because ..." vs "The ...")   <- the branch itself
    #   position 1 : WHICH feature       ("HNR" / "F0" / "jitter" ...)  <- the discriminating token
    #
    # Both are needed: position 0 says whether the model wants to withhold at all, position 1
    # says which clause it picks once it has decided to continue.
    cands = {nm: frames[short.index(nm)][2] for nm in short}
    hedge_txt = "Because the speakers overlap heavily"
    hedge_id = tok(" " + hedge_txt, add_special_tokens=False)["input_ids"][0]
    continue_id = tok(" " + next(iter(cands.values())).strip(),
                      add_special_tokens=False)["input_ids"][0]

    second_id, shared_prefix_ids = {}, None
    for nm, txt in cands.items():
        ids = tok(" " + txt.strip(), add_special_tokens=False)["input_ids"]
        if len(ids) >= 2:
            second_id[nm] = ids[1]
            if shared_prefix_ids is None:
                shared_prefix_ids = ids[:1]
    if shared_prefix_ids is None:
        print("[fatal] could not derive the shared clause opener")
        return 1
    print(f"[tokens] shared opener={tok.decode(shared_prefix_ids)!r}  "
          f"hedge-opener={tok.decode([hedge_id])!r}", flush=True)

    prompt = cfg_yaml.get("prompt_prose") or cfg_yaml.get("prompt") or ""
    p_ids = tok(prompt, return_tensors="pt").input_ids.to(a.device)
    pre_ids = tok(prefix_txt, return_tensors="pt", add_special_tokens=False).input_ids.to(a.device)

    files = sorted(f for f in os.listdir(a.test_dir)
                   if f.endswith(".pt") and "_s1clean" in f)[: a.n]
    print(f"[data] probing {len(files)} CLEAN clips", flush=True)

    agg = {nm: [] for nm in second_id}
    rank_of = {nm: [] for nm in second_id}
    branch = {"hedge": [], "continue": []}
    recs = []
    open_ids = torch.tensor([shared_prefix_ids], device=a.device)
    emb = llm.get_input_embeddings()
    with torch.no_grad():
        for i, fn in enumerate(files):
            d = torch.load(os.path.join(a.test_dir, fn), map_location="cpu", weights_only=False)
            af = d["audio_features"].float().unsqueeze(0).to(a.device)
            oi = d["overlap_info"].float().unsqueeze(0).to(a.device)
            if zero_ovl:
                oi = torch.zeros_like(oi)
            # The adapter is fp32 (loaded from the checkpoint) while the LM is bf16. Cast the
            # adapter's OUTPUT, never its input — feeding bf16 into fp32 conv weights raises
            # "Input type (c10::BFloat16) and bias type (float) should be the same".
            out = adapter(af, oi)
            pre_emb = out[0] if isinstance(out, (tuple, list)) else out
            pre_emb = pre_emb.to(emb.weight.dtype)
            seq = torch.cat([pre_emb, emb(p_ids), emb(pre_ids)], dim=1)
            # POSITION 0 — the branch itself: withhold ("Because") vs continue ("The")
            logits0 = llm(inputs_embeds=seq).logits[0, -1, :].float()
            p0 = torch.softmax(logits0, dim=-1)
            branch["hedge"].append(float(p0[hedge_id]))
            branch["continue"].append(float(p0[continue_id]))
            # POSITION 1 — teacher-force the shared opener, then read WHICH feature follows
            seq2 = torch.cat([seq, emb(open_ids)], dim=1)
            logits1 = llm(inputs_embeds=seq2).logits[0, -1, :].float()
            probs = torch.softmax(logits1, dim=-1)
            order = torch.argsort(probs, descending=True)
            rank_lookup = {int(t): r for r, t in enumerate(order[:2000].tolist())}
            rec = {"clip": fn[:-3], "argmax_token": tok.decode([int(order[0])]),
                   "p_hedge": float(p0[hedge_id]), "p_continue": float(p0[continue_id])}
            for nm, tid in second_id.items():
                p = float(probs[tid])
                agg[nm].append(p)
                rank_of[nm].append(rank_lookup.get(tid, 9999))
                rec[nm] = p
            recs.append(rec)
            if (i + 1) % 10 == 0:
                print(f"  {i+1}/{len(files)}", flush=True)

    import numpy as np
    print(f"\n=== POSITION 0 — the branch: withhold vs continue (CLEAN clips) ===")
    print(f"   P(hedge opener)    mean {np.mean(branch['hedge']):.4f}")
    print(f"   P(clause opener)   mean {np.mean(branch['continue']):.4f}")
    print("   On clean clips a correct model CONTINUES. A high hedge probability here would mean")
    print("   the model mistakes clean clips for mixtures — an evidence problem, not decoding.\n")
    print(f"=== POSITION 1 — WHICH feature, after the shared opener is teacher-forced ===")
    print(f"{'continuation':<16}{'mean P':>10}{'median rank':>13}{'P>0.01':>9}   note")
    print("-" * 66)
    order_show = ["hnr", "f0_mean", "f0_sd", "jitter", "shimmer",
                  "speaking_rate", "pause_count"]
    for nm in order_show:
        if nm not in agg:
            continue
        P = np.array(agg[nm]); R = np.array(rank_of[nm])
        note = ""
        if nm in ("hnr", "shimmer", "f0_mean", "f0_sd"):
            note = "DROPPED in generation"
        elif nm == "jitter":
            note = "KEPT in generation (94.2%)"
        print(f"{nm:<16}{P.mean():10.4f}{np.median(R):13.0f}{(P > 0.01).mean():9.2f}   {note}")

    from collections import Counter
    print("\nargmax token at the branch point:")
    for t, c in Counter(r["argmax_token"] for r in recs).most_common(6):
        print(f"   {t!r:<20}{c:4d}/{len(recs)}")
    print("\nREAD: dropped-feature openers with HIGH P but poor rank -> greedy amplification (H1),")
    print("a DECODE-TIME fix suffices. Near-zero P -> the learned prior collapsed, needs retrain.")
    print("jitter vs shimmer is the matched contrast: same hedge group, opposite outcome.")
    json.dump(recs, open(a.out, "w"))
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
