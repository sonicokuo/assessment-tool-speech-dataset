"""run_m3b_snr_consistency.py — DIRECTIONAL consistency test for SNR (better than plain
deletion for a global feature).

Plain deletion asks "did the number move" (sign-agnostic) — near-useless for SNR because
it's a whole-clip average.  This instead uses the per-frame SNR truth to make a SIGNED,
physically-checkable prediction and asks whether the LM's emitted SNR moves the RIGHT WAY:

  - occlude the NOISIEST prefix frames (lowest true per-frame SNR)  -> emitted SNR should go UP
  - occlude the CLEANEST prefix frames (highest true per-frame SNR) -> emitted SNR should go DOWN
  - occlude RANDOM frames                                           -> emitted SNR ~ unchanged

If the LM tracks that ordering (Δnoisy > Δrandom > Δclean), its SNR claim is causally
sensitive to the physically-correct frames.  Reuses the M3b deletion driver's model load +
generation.  Per-frame SNR truth from data/snr_map_targets_aug/<split>/<stem>.pt.

Usage:
  python src/run_m3b_snr_consistency.py --config configs/config.m3b.yaml \
    --checkpoint .../M3b_snr_tokengnd_cur/best.pt --processed_dir data/processed_aug/test \
    --snr_target_dir data/snr_map_targets_aug/test --max_clips 200 --frac 0.2 --out out.json
"""
import argparse
import glob
import json
from pathlib import Path

import torch
import torch.nn.functional as F

import sys
sys.path.insert(0, "src")
from experiments.run_m3b_deletion import build_model_slim, generate, snr_of, mask_prefix  # noqa: E402
from extract_attention import load_config                                    # noqa: E402


def load_per_frame_snr(path):
    """Per-frame SNR (521,) with invalid frames (snr_map_mask==0) set to NaN so they are
    neutral in pooling and never selected as noisiest/cleanest."""
    x = torch.load(path, map_location="cpu", weights_only=False)
    tgt = x["snr_map_target"].float().clone()
    m = x.get("snr_map_mask")
    if m is not None:
        tgt[m.float().flatten() < 0.5] = float("nan")
    return tgt.flatten()


def pool_to_P(per_frame, P):
    pf = torch.nan_to_num(per_frame, nan=float(torch.nanmean(per_frame)))
    return F.adaptive_avg_pool1d(pf.view(1, 1, -1), P).view(-1)


@torch.no_grad()
def one_clip(adapter, llm, tokenizer, sample, snr_pf, prompt_embeds, device, frac, rng):
    af = sample["audio_features"].unsqueeze(0).to(device).to(torch.bfloat16)
    oi = sample["overlap_info"].unsqueeze(0).to(device).to(torch.bfloat16)
    out = adapter(af, oi)
    prefix = out[0] if isinstance(out, tuple) else out
    P = prefix.shape[1]
    base = snr_of(generate(llm, tokenizer, prefix, prompt_embeds, device)[0])
    if base is None:
        return None
    snr_p = pool_to_P(snr_pf, P).to(device)
    k = max(1, int(round(frac * P)))

    noisy = torch.topk(-snr_p, k).indices            # lowest SNR frames
    clean = torch.topk(snr_p, k).indices             # highest SNR frames
    rand = torch.randperm(P, generator=rng)[:k].to(device)

    def emit(idx):
        return snr_of(generate(llm, tokenizer, mask_prefix(prefix, idx), prompt_embeds, device)[0])

    sn, sc, sr = emit(noisy), emit(clean), emit(rand)
    if None in (sn, sc, sr):
        return None
    return {"base": base, "d_noisy": sn - base, "d_clean": sc - base, "d_rand": sr - base}


def summarize(rows):
    n = len(rows)
    if not n:
        return {"n": 0}
    mn = sum(r["d_noisy"] for r in rows) / n
    mc = sum(r["d_clean"] for r in rows) / n
    mr = sum(r["d_rand"] for r in rows) / n
    # directional consistency: noisy-removal raises SNR more than random AND clean-removal lowers it
    consistent = sum(1 for r in rows if r["d_noisy"] > r["d_rand"] and r["d_clean"] < r["d_rand"]) / n
    monotonic = mn > mr > mc
    return {"n": n, "mean_d_noisy": mn, "mean_d_rand": mr, "mean_d_clean": mc,
            "directional_consistency_rate": consistent, "monotonic_ordering": monotonic}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--processed_dir", required=True)
    ap.add_argument("--snr_target_dir", required=True)
    ap.add_argument("--max_clips", type=int, default=200)
    ap.add_argument("--frac", type=float, default=0.2)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    cfg, ck = load_config(Path(a.config), Path(a.checkpoint))
    adapter, llm, tokenizer = build_model_slim(cfg, ck, a.device)
    prompt_str = cfg.get("prompt_prose") or cfg["prompt"]
    prompt_ids = tokenizer(prompt_str, return_tensors="pt").input_ids.to(a.device)
    prompt_embeds = llm.get_input_embeddings()(prompt_ids)
    rng = torch.Generator().manual_seed(0)

    pts = sorted(glob.glob(f"{a.processed_dir}/*.pt"))[: a.max_clips]
    rows, done = [], 0
    for i, p in enumerate(pts):
        stem = Path(p).stem
        tgt = Path(a.snr_target_dir) / f"{stem}.pt"
        if not tgt.exists():
            continue
        try:
            s = torch.load(p, map_location="cpu", weights_only=False)
            snr_pf = load_per_frame_snr(tgt)
            r = one_clip(adapter, llm, tokenizer, s, snr_pf, prompt_embeds, a.device, a.frac, rng)
            if r is not None:
                rows.append(r); done += 1
        except Exception as e:
            print(f"  [skip] {stem}: {type(e).__name__}: {e}", flush=True)
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(pts)} ({done} scored)", flush=True)
    summ = summarize(rows)
    Path(a.out).write_text(json.dumps({"summary": summ, "frac": a.frac}, indent=2))
    print("SUMMARY:", json.dumps(summ))


if __name__ == "__main__":
    main()
