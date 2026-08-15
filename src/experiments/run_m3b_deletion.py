"""run_m3b_deletion.py — LM-level causal deletion for M3b's token-grounding SNR map.

The valid grounding test (head-level is circular; see token_grounding_validate.py): mask the
top-alpha audio *prefix* frames the LM conditions on, REGENERATE, and measure how much the
EMITTED SNR number moves vs masking the same count of random frames.  A model_rand head
(random weights) is the Adebayo sanity: a real map => trained top-k moves the number more than
random AND more than the random head does; an artifact => trained ~ random.

Reuses scripts/extract_attention.py's exact inference-faithful model load + greedy generation.

Usage:
  python src/run_m3b_deletion.py --config configs/config.m3b.yaml \
      --checkpoint .../M3b_snr_tokengnd_cur/best.pt --processed_dir data/processed_aug/test \
      --max_clips 800 --frac 0.2 --out .../snrdel_lm.json
"""
import argparse
import glob
import json
import re
import sys
from pathlib import Path

import torch

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")
from extract_attention import load_config                          # noqa: E402
from experiments.token_grounding_validate import (                              # noqa: E402
    load_token_grounding_head, randomized_head_like,
)


def build_model_slim(cfg, ck, device):
    """Inference-faithful load that handles SLIM checkpoints (LoRA/trainable only, base
    from HF) via ckpt_io.load_llm_state_dict — extract_attention.load_model does a strict
    load and breaks on M3b's slim llm_state_dict."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model
    from model.adapter import build_adapter
    from data.ckpt_io import load_llm_state_dict
    tok = AutoTokenizer.from_pretrained(cfg["lm_name"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    llm = AutoModelForCausalLM.from_pretrained(
        cfg["lm_name"], dtype=torch.bfloat16, device_map={"": device},
        attn_implementation="eager",
    )
    if cfg.get("lora_rank"):
        llm = get_peft_model(llm, LoraConfig(
            r=cfg["lora_rank"], lora_alpha=cfg["lora_alpha"],
            target_modules=cfg["lora_targets"], lora_dropout=cfg["lora_dropout"],
            bias="none", task_type="CAUSAL_LM"))
    # BUGFIX 2026-08-10, same class as inference.py's 2026-07-23 fix: without these
    # flags a reliability_head=true / compression=4 / linear_softmax checkpoint loads
    # into the WRONG module shapes, and strict=False silently leaves the aux head at
    # RANDOM init. Everything downstream then looks structured but means nothing.
    adapter = build_adapter(
        cfg["adapter_variant"],
        lm_dim=llm.config.hidden_size,
        reliability_head=bool(cfg.get("reliability_head", False)),
        compression=int(cfg.get("compression", 8)),
        aux_pool=str(cfg.get("aux_pool") or "mean"),
    ).to(device).to(torch.bfloat16)
    missing, unexpected = adapter.load_state_dict(ck["adapter_state_dict"], strict=False)
    _bad = [k for k in list(missing) + list(unexpected) if "head" in k or "regress" in k]
    if _bad:
        raise RuntimeError(
            f"adapter head weights did not load: {_bad} — reliability_head/aux_pool/"
            "compression mismatch between training and this script?"
        )
    llm_sd = ck.get("llm_state_dict") or ck["lora_state_dict"]
    load_llm_state_dict(llm, llm_sd, ckpt_format=ck.get("ckpt_format"))
    adapter.eval(); llm.eval()
    return adapter, llm, tok

SNR_RE = re.compile(r"([-+]?\d+(?:\.\d+)?)\s*dB")


def snr_of(text):
    m = SNR_RE.search(text or "")
    return float(m.group(1)) if m else None


@torch.no_grad()
def generate(llm, tokenizer, prefix, prompt_embeds, device, max_new=180, want_hidden=False):
    embed = llm.get_input_embeddings()
    inp = torch.cat([prefix, prompt_embeds], dim=1)
    ids, hs = [], []
    out = llm(inputs_embeds=inp, use_cache=True, output_hidden_states=want_hidden)
    pkv = out.past_key_values
    nxt = out.logits[:, -1, :].argmax(-1, keepdim=True)
    ids.append(nxt.item())
    if want_hidden:
        hs.append(out.hidden_states[-1][:, -1, :])
    for _ in range(max_new - 1):
        o = llm(inputs_embeds=embed(nxt), past_key_values=pkv, use_cache=True,
                output_hidden_states=want_hidden)
        pkv = o.past_key_values
        nxt = o.logits[:, -1, :].argmax(-1, keepdim=True)
        t = nxt.item(); ids.append(t)
        if want_hidden:
            hs.append(o.hidden_states[-1][:, -1, :])
        if t == tokenizer.eos_token_id:
            break
    return tokenizer.decode(ids, skip_special_tokens=True), ids, hs


def snr_token_step(tokenizer, ids, text):
    """Map the SNR number's char offset to the generation step that emitted it."""
    m = SNR_RE.search(text)
    if not m:
        return None
    char = m.start(1)
    for t in range(1, len(ids) + 1):
        if len(tokenizer.decode(ids[:t], skip_special_tokens=True)) >= char + 1:
            return t - 1
    return None


def mask_prefix(prefix, idx):
    """Occlude prefix frames with the MEAN prefix token (RISE-style neutral baseline), not
    zero — zeroing injects an out-of-distribution token the LM reads as broken signal, which
    biases the emitted SNR downward for ANY masking and hides the grounding signal."""
    p = prefix.clone()
    baseline = prefix.mean(dim=1, keepdim=True)          # (1, 1, d) = average audio frame
    p[0, idx, :] = baseline[0, 0, :]
    return p


@torch.no_grad()
def deletion_one(adapter, llm, tokenizer, head, rnd_head, sample, prompt_embeds,
                 device, frac, rng):
    af = sample["audio_features"].unsqueeze(0).to(device).to(torch.bfloat16)
    oi = sample["overlap_info"].unsqueeze(0).to(device).to(torch.bfloat16)
    out = adapter(af, oi)
    prefix = out[0] if isinstance(out, tuple) else out          # (1, P, d)
    P = prefix.shape[1]
    text, ids, hs = generate(llm, tokenizer, prefix, prompt_embeds, device, want_hidden=True)
    base = snr_of(text)
    step = snr_token_step(tokenizer, ids, text)
    if base is None or step is None or step >= len(hs):
        return None
    h = hs[step].unsqueeze(1).float()                           # (1, 1, d)
    k = max(1, int(round(frac * P)))

    def run(head_):
        _, alpha, _ = head_(h, prefix.float(), None)           # alpha (1,1,P)
        top = torch.topk(alpha[0, 0], k).indices
        td = snr_of(generate(llm, tokenizer, mask_prefix(prefix, top), prompt_embeds, device)[0])
        perm = torch.randperm(P, generator=rng)[:k]
        rd = snr_of(generate(llm, tokenizer, mask_prefix(prefix, perm), prompt_embeds, device)[0])
        dtop = abs(td - base) if td is not None else None
        drand = abs(rd - base) if rd is not None else None
        return dtop, drand

    dt, dr = run(head)
    rt, rr = run(rnd_head)
    return {"base": base, "dtop": dt, "drand": dr, "rnd_dtop": rt, "rnd_drand": rr, "P": P}


def summarize(rows):
    def wr(a, b):
        pairs = [(x, y) for r in rows if (x := r.get(a)) is not None and (y := r.get(b)) is not None]
        if not pairs:
            return float("nan"), float("nan"), 0
        win = sum(1 for x, y in pairs if x > y) / len(pairs)
        md = sum(x for x, _ in pairs) / len(pairs)
        return win, md, len(pairs)
    tw, tmd, n = wr("dtop", "drand")
    rw, rmd, _ = wr("rnd_dtop", "rnd_drand")
    return {"n": n, "trained_win_rate": tw, "trained_mean_drop_top": tmd,
            "model_rand_win_rate": rw, "model_rand_mean_drop_top": rmd}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--processed_dir", required=True)
    ap.add_argument("--max_clips", type=int, default=800)
    ap.add_argument("--frac", type=float, default=0.2)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    cfg, ck = load_config(Path(a.config), Path(a.checkpoint))
    adapter, llm, tokenizer = build_model_slim(cfg, ck, a.device)
    prompt_str = cfg.get("prompt_prose") or cfg["prompt"]
    prompt_ids = tokenizer(prompt_str, return_tensors="pt").input_ids.to(a.device)
    prompt_embeds = llm.get_input_embeddings()(prompt_ids)

    head = load_token_grounding_head(a.checkpoint, a.device).float()
    rnd_head = randomized_head_like(head, seed=0).float()
    rng = torch.Generator().manual_seed(0)

    pts = sorted(glob.glob(f"{a.processed_dir}/*.pt"))[: a.max_clips]
    rows, done = [], 0
    for i, p in enumerate(pts):
        try:
            s = torch.load(p, map_location="cpu", weights_only=False)
            r = deletion_one(adapter, llm, tokenizer, head, rnd_head, s, prompt_embeds,
                             a.device, a.frac, rng)
            if r is not None:
                rows.append(r); done += 1
        except Exception as e:
            print(f"  [skip] {Path(p).name}: {type(e).__name__}: {e}", flush=True)
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(pts)} clips ({done} scored)", flush=True)
    summ = summarize(rows)
    Path(a.out).write_text(json.dumps({"summary": summ, "frac": a.frac,
                                       "n_clips": len(rows)}, indent=2))
    print("SUMMARY:", json.dumps(summ))


if __name__ == "__main__":
    main()
