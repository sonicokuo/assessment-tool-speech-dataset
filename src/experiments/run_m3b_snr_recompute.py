"""run_m3b_snr_recompute.py — RECOMPUTE (magnitude-aware) grounding test for M3b's SNR claim.

The directional consistency test (run_m3b_snr_consistency.py) only checks the SIGN of the
emitted SNR change. This goes further: it recomputes what the clip SNR *should become* after
masking a set of frames, and checks whether the LM's emitted SNR TRACKS that recomputed target
in magnitude (not just direction).

Under stationary additive noise, the SNR over a kept subset of frames is the power-domain mean
of their per-frame SNRs:
    SNR_kept = 10*log10( mean_{t in kept} 10^(SNR_t / 10) )
which is computable from the per-frame SNR truth alone (no stems needed).

For each clip we mask the noisiest / cleanest / random k% of prefix frames at several fractions,
regenerate, and collect pairs (d_recomp, d_emit):
    d_recomp = SNR_kept(masked) - SNR_kept(full)         # what physics says should happen
    d_emit   = emitted_SNR(masked) - emitted_SNR(base)    # what the LM actually did
Headline = Pearson r and OLS slope of d_emit on d_recomp. slope~1 & high r => the SNR claim is
causally grounded in the physically-correct frames (magnitude, not just sign). r~0 => not.

Usage:
  python src/run_m3b_snr_recompute.py --config configs/config.m3b.yaml \
    --checkpoint .../M3b_snr_tokengnd_cur/best.pt --processed_dir data/processed_aug/test \
    --snr_target_dir data/snr_map_targets_aug/test --max_clips 200 --fracs 0.15,0.3 --out out.json
  python src/run_m3b_snr_recompute.py --selftest   # verify the recompute formula, no model
"""
import argparse
import glob
import json
import math
from pathlib import Path

import torch

import sys
sys.path.insert(0, "src")
from experiments.run_m3b_deletion import build_model_slim, generate, snr_of              # noqa: E402
from experiments.run_m3b_snr_consistency import load_per_frame_snr, pool_to_P             # noqa: E402
from extract_attention import load_config                                    # noqa: E402

_LN10 = math.log(10.0)


def remove_prefix(prefix, remove_idx):
    """Physically REMOVE (drop) the selected prefix frames — matches the recompute target
    which is the SNR of the *remaining* frames. Mean-masking would not; it replaces frames
    with average audio, so the model never sees the clip as actually shorter/cleaner."""
    keep = torch.ones(prefix.shape[1], dtype=torch.bool, device=prefix.device)
    keep[remove_idx] = False
    return prefix[:, keep, :]


def recompute_snr(x):
    """Stationary-noise clip SNR (dB) over the frames in 1-D tensor x (per-frame SNR dB).
    = 10*log10(mean(10^(x/10))), via logsumexp for numerical stability."""
    return float(10.0 * (torch.logsumexp(x * _LN10 / 10.0, dim=0) - math.log(x.numel())) / _LN10)


@torch.no_grad()
def one_clip(adapter, llm, tok, sample, snr_pf, prompt_embeds, device, fracs, rng):
    af = sample["audio_features"].unsqueeze(0).to(device).to(torch.bfloat16)
    oi = sample["overlap_info"].unsqueeze(0).to(device).to(torch.bfloat16)
    out = adapter(af, oi)
    prefix = out[0] if isinstance(out, tuple) else out
    P = prefix.shape[1]
    base = snr_of(generate(llm, tok, prefix, prompt_embeds, device)[0])
    if base is None:
        return []
    snr_p = pool_to_P(snr_pf, P)                    # (P,) on cpu
    full = recompute_snr(snr_p)
    pts = []
    for f in fracs:
        k = max(1, int(round(f * P)))
        order_lo = torch.topk(-snr_p, k).indices    # noisiest (lowest SNR)
        order_hi = torch.topk(snr_p, k).indices     # cleanest (highest SNR)
        rand = torch.randperm(P, generator=rng)[:k]
        for cond, R in (("noisy", order_lo), ("clean", order_hi), ("rand", rand)):
            keep = torch.ones(P, dtype=torch.bool)
            keep[R] = False
            if keep.sum() < 1:
                continue
            d_recomp = recompute_snr(snr_p[keep]) - full
            em = snr_of(generate(llm, tok, remove_prefix(prefix, R.to(device)), prompt_embeds, device)[0])
            if em is None:
                continue
            pts.append({"cond": cond, "frac": f, "d_recomp": d_recomp, "d_emit": em - base})
    return pts


def summarize(pts):
    n = len(pts)
    if n < 2:
        return {"n": n}
    dr = [p["d_recomp"] for p in pts]
    de = [p["d_emit"] for p in pts]
    mr, me = sum(dr) / n, sum(de) / n
    cov = sum((a - mr) * (b - me) for a, b in zip(dr, de)) / n
    vr = sum((a - mr) ** 2 for a in dr) / n
    ve = sum((b - me) ** 2 for b in de) / n
    r = cov / math.sqrt(vr * ve) if vr > 0 and ve > 0 else float("nan")
    slope = cov / vr if vr > 0 else float("nan")
    by = {}
    for c in ("noisy", "clean", "rand"):
        cp = [p for p in pts if p["cond"] == c]
        if cp:
            by[c] = {"n": len(cp),
                     "mean_d_recomp": sum(p["d_recomp"] for p in cp) / len(cp),
                     "mean_d_emit": sum(p["d_emit"] for p in cp) / len(cp)}
    return {"n": n, "pearson_r": r, "slope_emit_on_recomp": slope,
            "mean_d_recomp": mr, "mean_d_emit": me, "by_cond": by}


def selftest():
    x = torch.tensor([20.0, 20.0, 0.0, 0.0])
    full = recompute_snr(x)                                   # 10log10((100+100+1+1)/4)=17.03
    hi = recompute_snr(x[:2])                                 # mask noisy -> [20,20] -> 20.0
    lo = recompute_snr(x[2:])                                 # mask clean -> [0,0]   -> 0.0
    print(f"full={full:.3f} (exp 17.033)  mask_noisy={hi:.3f} (exp 20.0)  mask_clean={lo:.3f} (exp 0.0)")
    ok = abs(full - 17.033) < 1e-2 and abs(hi - 20.0) < 1e-3 and abs(lo - 0.0) < 1e-3
    print("SELFTEST", "PASS" if ok else "FAIL")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--checkpoint")
    ap.add_argument("--processed_dir")
    ap.add_argument("--snr_target_dir")
    ap.add_argument("--max_clips", type=int, default=200)
    ap.add_argument("--fracs", default="0.15,0.3")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest(); return

    fracs = [float(x) for x in a.fracs.split(",")]
    cfg, ck = load_config(Path(a.config), Path(a.checkpoint))
    adapter, llm, tok = build_model_slim(cfg, ck, a.device)
    prompt_str = cfg.get("prompt_prose") or cfg["prompt"]
    prompt_ids = tok(prompt_str, return_tensors="pt").input_ids.to(a.device)
    prompt_embeds = llm.get_input_embeddings()(prompt_ids)
    rng = torch.Generator().manual_seed(0)

    pts = sorted(glob.glob(f"{a.processed_dir}/*.pt"))[: a.max_clips]
    all_pts, done = [], 0
    for i, p in enumerate(pts):
        stem = Path(p).stem
        tgt = Path(a.snr_target_dir) / f"{stem}.pt"
        if not tgt.exists():
            continue
        try:
            s = torch.load(p, map_location="cpu", weights_only=False)
            snr_pf = load_per_frame_snr(tgt)
            rows = one_clip(adapter, llm, tok, s, snr_pf, prompt_embeds, a.device, fracs, rng)
            all_pts.extend(rows)
            if rows:
                done += 1
        except Exception as e:
            print(f"  [skip] {stem}: {type(e).__name__}: {e}", flush=True)
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(pts)} ({done} clips, {len(all_pts)} points)", flush=True)
    summ = summarize(all_pts)
    Path(a.out).write_text(json.dumps({"summary": summ, "fracs": fracs, "clips": done}, indent=2))
    print("SUMMARY:", json.dumps(summ))


if __name__ == "__main__":
    main()
