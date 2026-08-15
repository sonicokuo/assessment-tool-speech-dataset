#!/usr/bin/env python3
"""temperature_redecode.py — is coverage collapse GREEDY AMPLIFICATION or a COLLAPSED PRIOR?

THE QUESTION (plan Q4 step 1b)
On clean clips the model emits f0_mean 0.1%, f0_sd 0.1%, shimmer 0.0%, hnr 0.0% while the targets
state values for all of them (98.8-100%), and jitter alone survives at 94.2%. It is a CLAUSE-SKIP,
not a hedge: P(hedge opener) is 0.0000 and only 5.8% of clean generations contain hedge phrasing.
Two explanations imply OPPOSITE fixes:

  GREEDY AMPLIFICATION — the dropped clauses have real probability mass, but a competing
      continuation wins the argmax on every clip, so greedy decoding turns a plurality into 100%
      (Holtzman et al., ICLR 2020, arXiv:1904.09751). **A DECODE-TIME fix suffices** — sampling, or
      the slot decoder, which teacher-forces every frame and cannot skip.
  COLLAPSED PRIOR — the mass is not there at all. **Needs the sigma-gated retrain** (Q4 step 2).

WHY SAMPLING AND NOT A LOGIT PROBE
A logit probe must construct the context to probe FROM, and three attempts at that failed on this
model: a dtype mismatch, a shared-token degeneracy (every clause opens with "The", so scoring first
tokens measured P(" The") identically for all features), and — decisively — an OFF-DISTRIBUTION
prefix: real generations carry a preamble and varied connectives, so a canonical-frame prefix
probes a context the model rarely sees and its argmax there contradicts actual behaviour.
Sampling constructs nothing. It runs the model's own generation process with the argmax constraint
relaxed, which is exactly the counterfactual the hypothesis names.

READING (pre-registered, before the run):
  emission of the 4 dropped features RISES materially at T=1.0  -> GREEDY AMPLIFICATION
  stays ~0 at T=1.0                                             -> COLLAPSED PRIOR
  jitter (the matched control, same hedge group, adjacent clause) should stay HIGH at both
  temperatures; if jitter also collapses under sampling the comparison is uninformative.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from model.adapter import build_adapter  # noqa: E402

PATTERNS = [("snr", r"SNR is"), ("srmr", r"SRMR is"), ("hnr", r"HNR is"),
            ("f0_mean", r"F0 mean is"), ("f0_sd", r"F0 standard deviation"),
            ("jitter", r"jitter is"), ("shimmer", r"shimmer is"),
            ("speaking_rate", r"speaking rate is"), ("pause_count", r"pause count is"),
            ("pause_rate", r"pause rate is"), ("overlap_ratio", r"overlap ratio is")]
HEDGE = re.compile(r"cannot be reliably estimated|not reported", re.I)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--test_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--temps", default="0.0,1.0")
    ap.add_argument("--max_new_tokens", type=int, default=320)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    import yaml
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cfg_yaml = yaml.safe_load(open(a.config))
    ck = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    cfg = ck.get("config", {}) or {}
    zero_ovl = bool(cfg.get("zero_overlap_input", cfg_yaml.get("zero_overlap_input", False)))
    lm_name = cfg.get("lm_name") or cfg_yaml.get("lm_name")

    tok = AutoTokenizer.from_pretrained(lm_name)
    llm = AutoModelForCausalLM.from_pretrained(lm_name, dtype=torch.bfloat16).to(a.device).eval()
    # ⚠️ USE THE CANONICAL LOADERS. The first version hand-rolled a LoraConfig with GUESSED
    # target_modules and called load_state_dict(..., strict=False) without checking the result.
    # Nothing loaded, the script silently probed the BASE model, and it produced a confident
    # "COLLAPSED PRIOR" verdict from generic prose containing no numeric claims at all. That is
    # the 4th occurrence of the load-with-strict=False-and-never-check bug class in this repo
    # (inference.py, reliability_head, run_m3b_deletion). Mirror inference.py exactly instead.
    from peft import LoraConfig, get_peft_model

    from data.ckpt_io import load_llm_state_dict
    from model.peft_config import lora_config_kwargs

    merged_cfg = {**cfg_yaml, **cfg}
    llm_sd = ck.get("llm_state_dict") or ck.get("lora_state_dict")
    if llm_sd:
        llm = get_peft_model(llm, LoraConfig(**lora_config_kwargs(merged_cfg)))
        _missing, _unexpected = load_llm_state_dict(
            llm, llm_sd, ckpt_format=ck.get("ckpt_format"))
        if _unexpected:
            print(f"[fatal] unexpected LLM keys: {list(_unexpected)[:5]}")
            return 1
        print(f"[lora] loaded (missing={len(list(_missing))} unexpected=0)", flush=True)
    else:
        print("[fatal] checkpoint has neither llm_state_dict nor lora_state_dict")
        return 1

    adapter = build_adapter(
        variant=cfg.get("adapter_variant", "attn-concat"),
        audio_dim=cfg.get("audio_dim", 1024), lm_dim=cfg.get("lm_dim", 4096),
        compression=cfg.get("compression", 8), aux_pool=cfg.get("aux_pool") or "mean",
        reliability_head=bool(cfg.get("reliability_head", False)))
    miss, unexp = adapter.load_state_dict(
        ck.get("adapter_state_dict") or ck.get("adapter") or {}, strict=False)
    if miss or unexp:
        print(f"[fatal] adapter mismatch missing={list(miss)} unexpected={list(unexp)}")
        return 1
    adapter.eval().to(a.device)
    print(f"[load] zero_overlap_input={zero_ovl}  adapter=EXACT MATCH", flush=True)

    prompt = cfg_yaml.get("prompt_prose") or cfg_yaml.get("prompt") or ""
    p_ids = tok(prompt, return_tensors="pt").input_ids.to(a.device)
    emb = llm.get_input_embeddings()
    temps = [float(t) for t in a.temps.split(",")]
    files = sorted(f for f in os.listdir(a.test_dir)
                   if f.endswith(".pt") and "_s1clean" in f)[: a.n]
    print(f"[data] {len(files)} CLEAN clips x temps {temps}", flush=True)

    pats = [(n, re.compile(p, re.I)) for n, p in PATTERNS]
    out, counts = [], {t: {n: 0 for n, _ in PATTERNS} for t in temps}
    hedge_ct = {t: 0 for t in temps}
    # POSITIVE CONTROL, checked after the first clip: a correctly-loaded model emits "The SNR
    # is <x> dB" on essentially every clip (measured 100% coverage). If it does not, the LoRA
    # did not take and every verdict below is about the BASE model — abort rather than print a
    # confident answer, which is exactly what the previous version did.
    control_ok = None

    for i, fn in enumerate(files):
        d = torch.load(os.path.join(a.test_dir, fn), map_location="cpu", weights_only=False)
        af = d["audio_features"].float().unsqueeze(0).to(a.device)
        oi = d["overlap_info"].float().unsqueeze(0).to(a.device)
        if zero_ovl:
            oi = torch.zeros_like(oi)
        with torch.no_grad():
            r = adapter(af, oi)                       # adapter is fp32; cast its OUTPUT only
            pre = (r[0] if isinstance(r, (tuple, list)) else r).to(emb.weight.dtype)
            seq = torch.cat([pre, emb(p_ids)], dim=1)
            rec = {"clip": fn[:-3]}
            for t in temps:
                gen = llm.generate(
                    inputs_embeds=seq, max_new_tokens=a.max_new_tokens,
                    do_sample=(t > 0), temperature=(t if t > 0 else None),
                    top_k=0 if t > 0 else None, top_p=1.0 if t > 0 else None,
                    pad_token_id=tok.eos_token_id)
                txt = tok.decode(gen[0], skip_special_tokens=True)
                rec[f"gen_T{t}"] = txt
                for nm, pt in pats:
                    if pt.search(txt):
                        counts[t][nm] += 1
                if HEDGE.search(txt):
                    hedge_ct[t] += 1
        out.append(rec)
        if control_ok is None:
            g0 = rec.get(f"gen_T{temps[0]}", "")
            control_ok = bool(re.search(r"SNR is", g0, re.I))
            print(f"[control] first clip emits an SNR clause: {control_ok}", flush=True)
            if not control_ok:
                print("[fatal] the model produced no SNR clause — the fine-tuned weights are "
                      "NOT active. Any coverage verdict from this run would describe the base "
                      "model. Sample of what it generated:")
                print(f"        {g0[:220]!r}")
                return 1
        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{len(files)}", flush=True)

    n = max(len(out), 1)
    print(f"\n{'feature':<16}" + "".join(f"{'T=' + str(t):>10}" for t in temps) + "   verdict")
    print("-" * (16 + 10 * len(temps) + 30))
    for nm, _ in PATTERNS:
        row = [100.0 * counts[t][nm] / n for t in temps]
        v = ""
        if nm in ("hnr", "f0_mean", "f0_sd", "shimmer"):
            # ⚠️ AN UNDEFINED VERDICT MUST NOT DEFAULT TO THE ALARMING ONE. The old form was
            # `GREEDY AMPLIFICATION if len(row) > 1 and row[-1] > 20 else COLLAPSED PRIOR`, so a
            # SINGLE-temperature run (--temps 0.0) could never satisfy `len(row) > 1` and printed
            # "COLLAPSED PRIOR — needs retrain" next to a measured 98.3% emission. That inverts
            # the reading of a successful run, and it is what the fw2 probe reported.
            # Emission level is decidable from one temperature; only the GREEDY-vs-PRIOR
            # DISCRIMINATION needs two, so say so instead of guessing.
            if row[-1] > 80.0:
                v = "EMITS — no collapse"
            elif len(row) > 1:
                v = ("GREEDY AMPLIFICATION — decode fix" if row[-1] > 20
                     else "COLLAPSED PRIOR — needs retrain")
            else:
                v = (f"LOW ({row[-1]:.1f}%) — rerun with --temps 0.0,1.0 to tell "
                     f"greedy-amplification from a collapsed prior")
        elif nm == "jitter":
            v = "matched control (must stay high)"
        print(f"{nm:<16}" + "".join(f"{x:9.1f}%" for x in row) + f"   {v}")
    print(f"{'hedge phrase':<16}" + "".join(f"{100.0 * hedge_ct[t] / n:9.1f}%" for t in temps))
    json.dump(out, open(a.out, "w"))
    print(f"\nwrote {a.out}  n={len(out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
