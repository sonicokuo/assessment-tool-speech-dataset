"""run_f0_localized_deletion.py — HEAD-AGNOSTIC localized-grounding test for f0.

SNR came out null on frame-deletion because it is global (no frame locus). f0 is the opposite:
it lives in the VOICED frames. This tests whether the model's emitted f0 is causally grounded
in those frames, WITHOUT any grounding head — by masking the physically-relevant (voiced) frames
and comparing to masking the same number of random frames.

Confound control: voiced-mask and random-mask both occlude the SAME count k with the SAME
mean-baseline, so the length / OOD occlusion artifact is IDENTICAL for both. The DIFFERENCE
(|Δf0| voiced vs |Δf0| random) isolates the f0-specific causal effect. If voiced-masking
perturbs the emitted f0 more than random-masking, f0 IS frame-grounded (the contrast SNR lacked).

Only runs on clips where the model actually ASSERTS an f0 (it hedges f0 under overlap on ~83% of
the all-overlap test set); the ~17% asserting clips are the self-selected confident subset where
f0 grounding is testable. Voiced frames from data/pitch_map_targets/<split>/<stem>.pt (f0_map_mask).

Usage:
  python src/run_f0_localized_deletion.py --config configs/config.m3b.yaml \
    --checkpoint .../M3b_snr_tokengnd_cur/best.pt --processed_dir data/processed_aug/test \
    --pitch_target_dir data/pitch_map_targets/test --max_clips 800 --frac 0.2 --out out.json
"""
import argparse
import glob
import json
import re
from pathlib import Path

import torch
import torch.nn.functional as F

import sys
sys.path.insert(0, "src")
from experiments.run_m3b_deletion import build_model_slim, generate, mask_prefix       # noqa: E402
from extract_attention import load_config                                  # noqa: E402

F0_RE = re.compile(r"[Ff]0 mean is ([0-9]+(?:\.[0-9]+)?)\s*[Hh]z")


def f0_of(text):
    m = F0_RE.search(text or "")
    return float(m.group(1)) if m else None


def load_voiced(path):
    """Per-frame voiced indicator (521,), 1 where f0 is present (f0_map_mask==1)."""
    x = torch.load(path, map_location="cpu", weights_only=False)
    return x["f0_map_mask"].float().flatten()


def pool_to_P(v, P):
    return F.adaptive_avg_pool1d(v.view(1, 1, -1), P).view(-1)


@torch.no_grad()
def one_clip(adapter, llm, tok, sample, voiced, prompt_embeds, device, frac, rng):
    af = sample["audio_features"].unsqueeze(0).to(device).to(torch.bfloat16)
    oi = sample["overlap_info"].unsqueeze(0).to(device).to(torch.bfloat16)
    out = adapter(af, oi)
    prefix = out[0] if isinstance(out, tuple) else out
    P = prefix.shape[1]
    base = f0_of(generate(llm, tok, prefix, prompt_embeds, device)[0])
    if base is None:
        return None                                    # model hedged f0 -> not testable
    vp = pool_to_P(voiced, P).to(device)
    k = max(1, int(round(frac * P)))
    vi = torch.topk(vp, k).indices                     # most-voiced prefix frames
    ri = torch.randperm(P, generator=rng)[:k].to(device)
    fv = f0_of(generate(llm, tok, mask_prefix(prefix, vi), prompt_embeds, device)[0])
    fr = f0_of(generate(llm, tok, mask_prefix(prefix, ri), prompt_embeds, device)[0])
    if fv is None or fr is None:
        return None
    return {"base": base, "d_voiced": abs(fv - base), "d_rand": abs(fr - base)}


def summarize(rows):
    n = len(rows)
    if not n:
        return {"n": 0}
    mv = sum(r["d_voiced"] for r in rows) / n
    mr = sum(r["d_rand"] for r in rows) / n
    win = sum(1 for r in rows if r["d_voiced"] > r["d_rand"]) / n
    return {"n": n, "mean_d_voiced": mv, "mean_d_rand": mr, "voiced_win_rate": win,
            "grounded": win > 0.5 and mv > mr}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--processed_dir", required=True)
    ap.add_argument("--pitch_target_dir", required=True)
    ap.add_argument("--max_clips", type=int, default=800)
    ap.add_argument("--frac", type=float, default=0.2)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    cfg, ck = load_config(Path(a.config), Path(a.checkpoint))
    adapter, llm, tok = build_model_slim(cfg, ck, a.device)
    prompt_str = cfg.get("prompt_prose") or cfg["prompt"]
    prompt_ids = tok(prompt_str, return_tensors="pt").input_ids.to(a.device)
    prompt_embeds = llm.get_input_embeddings()(prompt_ids)
    rng = torch.Generator().manual_seed(0)

    pts = sorted(glob.glob(f"{a.processed_dir}/*.pt"))[: a.max_clips]
    rows, asserted = [], 0
    for i, p in enumerate(pts):
        stem = Path(p).stem
        tgt = Path(a.pitch_target_dir) / f"{stem}.pt"
        if not tgt.exists():
            continue
        try:
            s = torch.load(p, map_location="cpu", weights_only=False)
            voiced = load_voiced(tgt)
            r = one_clip(adapter, llm, tok, s, voiced, prompt_embeds, a.device, a.frac, rng)
            if r is not None:
                rows.append(r); asserted += 1
        except Exception as e:
            print(f"  [skip] {stem}: {type(e).__name__}: {e}", flush=True)
        if (i + 1) % 40 == 0:
            print(f"  {i+1}/{len(pts)} scanned ({asserted} asserted f0)", flush=True)
    summ = summarize(rows)
    Path(a.out).write_text(json.dumps({"summary": summ, "frac": a.frac}, indent=2))
    print("SUMMARY:", json.dumps(summ))


if __name__ == "__main__":
    main()
