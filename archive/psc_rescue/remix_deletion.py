"""Items 26 + 45 + 48 — IN-DISTRIBUTION causal grounding via re-mix substitution.

WHY THIS REPLACES THE OLD PROTOCOL. The recorded deletion null (trained_win_rate 0.325 vs
model_rand 0.45) is NOT a trustworthy negative for two reasons: it was measured on the
film-mamba M3b checkpoint, and it ZEROED frames -- inputs the model never saw in training
(Hase et al., NeurIPS 2021). Zeroing tests behaviour off the data manifold, so a null says
nothing about grounding.

THE FIX, available only because we hold the stems. Every mixture is s1 + s2 + noise, and
`audio_corrected/<split>-s1clean` holds the clean-s1 version of every clip. So "remove the
interference in window [a,b]" is a SPLICE:

    cf = mix.copy() ;  cf[a:b] = s1clean[a:b]

This is in-distribution BY CONSTRUCTION -- and more strongly than "similar to training data":
the fully-substituted endpoint IS a training clip (the `_s1clean` twins are in the dataset),
so a spliced clip interpolates between two points the model has actually seen.

THE TEST (item 45). For each clip and feature f, compare how much the prediction moves when
we clean the region the ATTRIBUTION says matters, versus a random region of identical length:

    win  <=>  |pred_f(cf_attributed) - pred_f(mix)|  >  |pred_f(cf_random) - pred_f(mix)|

Win rate > 0.5 means the attribution identifies causally-effective regions. This is the
headline causal number (item 48), measured on the ARM OF RECORD, in-distribution.

Length is matched exactly between the attributed and random regions, so the comparison cannot
be won by cleaning more audio.

Usage: remix_deletion.py <ckpt> <mix_dir> <s1clean_dir> <proc_dir> <out.json> [n_clips] [frac]
"""
from __future__ import annotations

import glob
import json
import os
import sys
import types

import numpy as np

try:
    raise ImportError
except Exception:
    _m = types.ModuleType("mamba_ssm")

    class _NoMamba:  # pragma: no cover
        def __init__(self, *a, **k):
            raise RuntimeError("Mamba unavailable")

    _m.Mamba = _NoMamba
    sys.modules["mamba_ssm"] = _m

import torch  # noqa: E402
import torchaudio  # noqa: E402

sys.path.insert(0, os.path.join(os.getcwd(), "src"))
from data.feature_set import FEATURE_NAMES  # noqa: E402
from model.adapter import build_adapter  # noqa: E402

CKPT, MIX_DIR, S1_DIR, PROC_DIR, OUT = sys.argv[1:6]
NCLIP = int(sys.argv[6]) if len(sys.argv) > 6 else 60
FRAC = float(sys.argv[7]) if len(sys.argv) > 7 else 0.25   # fraction of the clip to clean
SR = 16000

dev = "cuda" if torch.cuda.is_available() else "cpu"
rng = np.random.default_rng(0)

ck = torch.load(CKPT, map_location="cpu", weights_only=False)
cfg = ck["config"]
adapter = build_adapter(
    cfg["adapter_variant"], lm_dim=4096,
    reliability_head=bool(cfg.get("reliability_head", False)),
    compression=int(cfg.get("compression", 8)),
    aux_pool=str(cfg.get("aux_pool", "mean") or "mean"),
).to(dev).eval()
missing, _ = adapter.load_state_dict(ck["adapter_state_dict"], strict=False)
assert not [k for k in missing if "head" in k or "regress" in k], missing

from transformers import WavLMModel  # noqa: E402
wavlm = WavLMModel.from_pretrained("microsoft/wavlm-large").to(dev).eval()
for p in wavlm.parameters():
    p.requires_grad = False
print(f"[init] device={dev} pool={cfg.get('aux_pool')} frac={FRAC}", flush=True)


def load_wav(path: str) -> torch.Tensor:
    w, sr = torchaudio.load(path)
    if sr != SR:
        w = torchaudio.functional.resample(w, sr, SR)
    return w.mean(dim=0)


@torch.no_grad()
def predict(wav: torch.Tensor, overlap_info: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    """Return (pooled prediction (F,), per-frame z (N,F)) for a raw waveform."""
    feats = wavlm(wav.unsqueeze(0).to(dev)).last_hidden_state          # (1,T,1024)
    T = feats.shape[1]
    oi = overlap_info[:T].unsqueeze(0).to(dev).float()
    if oi.shape[1] < T:                                                # pad if shorter
        oi = torch.cat([oi, oi[:, -1:].expand(-1, T - oi.shape[1], -1)], dim=1)
    prefix = adapter.inner(feats.float(), oi)
    out = adapter(feats.float(), oi)[1]
    if isinstance(out, tuple):
        out = out[0]
    z = adapter.regress_head(prefix)
    z = z[0] if isinstance(z, tuple) else z
    return out[0].float().cpu().numpy(), z[0].float().cpu().numpy()


files = sorted(glob.glob(os.path.join(MIX_DIR, "*.wav")))
rng.shuffle(files)
wins = {f: [] for f in FEATURE_NAMES}
d_att = {f: [] for f in FEATURE_NAMES}
d_rnd = {f: [] for f in FEATURE_NAMES}
used = 0

for path in files:
    if used >= NCLIP:
        break
    stem = os.path.splitext(os.path.basename(path))[0]
    s1p = os.path.join(S1_DIR, f"{stem}_s1clean.wav")
    procp = os.path.join(PROC_DIR, f"{stem}.pt")
    if not (os.path.exists(s1p) and os.path.exists(procp)):
        continue
    d = torch.load(procp, map_location="cpu", weights_only=False)
    oi_full = d["overlap_info"].float()
    if float(oi_full[:, 0].mean()) < 0.05:      # need some overlap to be worth cleaning
        continue

    mix = load_wav(path)
    s1 = load_wav(s1p)
    n = min(len(mix), len(s1))
    mix, s1 = mix[:n], s1[:n]
    if n < SR:
        continue

    base, z_base = predict(mix, oi_full)
    N = z_base.shape[0]
    win_tok = max(1, int(round(FRAC * N)))
    samp_per_tok = n / N

    for fi, feat in enumerate(FEATURE_NAMES):
        # attributed region = contiguous window maximising |deviation| mass for THIS feature
        dev_map = np.abs(z_base[:, fi] - z_base[:, fi].mean())
        if not np.isfinite(dev_map).all() or dev_map.sum() <= 0:
            continue
        csum = np.concatenate([[0.0], np.cumsum(dev_map)])
        sums = csum[win_tok:] - csum[:-win_tok] if win_tok < len(csum) else np.array([csum[-1]])
        a_tok = int(np.argmax(sums))
        # random region of IDENTICAL length, so the test cannot be won by cleaning more audio
        r_tok = int(rng.integers(0, max(1, N - win_tok + 1)))

        outs = []
        for start in (a_tok, r_tok):
            lo = int(start * samp_per_tok)
            hi = min(n, int((start + win_tok) * samp_per_tok))
            cf = mix.clone()
            cf[lo:hi] = s1[lo:hi]                      # <-- the re-mix substitution
            outs.append(predict(cf, oi_full)[0][fi])

        da, dr = abs(outs[0] - base[fi]), abs(outs[1] - base[fi])
        if np.isfinite(da) and np.isfinite(dr):
            wins[feat].append(1.0 if da > dr else (0.5 if da == dr else 0.0))
            d_att[feat].append(da)
            d_rnd[feat].append(dr)

    used += 1
    if used % 10 == 0:
        print(f"  {used}/{NCLIP}", flush=True)

print(f"\nIN-DISTRIBUTION re-mix deletion, arm={os.path.basename(os.path.dirname(CKPT))}, "
      f"n={used} clips, window={FRAC:.0%} of clip\n")
print(f"{'feature':<15}{'win rate':>10}{'95% CI':>18}{'mean|d|att':>12}{'mean|d|rnd':>12}{'n':>6}")
print("-" * 74)
res = {}
for feat in FEATURE_NAMES:
    w = np.asarray(wins[feat], float)
    if len(w) < 8:
        continue
    wr = float(w.mean())
    bs = np.random.default_rng(1).integers(0, len(w), size=(2000, len(w)))
    ci = (float(np.percentile(w[bs].mean(1), 2.5)), float(np.percentile(w[bs].mean(1), 97.5)))
    ma, mr = float(np.mean(d_att[feat])), float(np.mean(d_rnd[feat]))
    star = "  <-- above chance" if ci[0] > 0.5 else ("  <-- BELOW chance" if ci[1] < 0.5 else "")
    print(f"{feat:<15}{wr:>10.3f}   [{ci[0]:.3f},{ci[1]:.3f}]{ma:>12.3f}{mr:>12.3f}{len(w):>6}{star}")
    res[feat] = {"win_rate": wr, "ci": ci, "mean_delta_attributed": ma,
                 "mean_delta_random": mr, "n": len(w)}

json.dump({"ckpt": CKPT, "n_clips": used, "frac": FRAC, "per_feature": res},
          open(OUT, "w"), indent=1)
print(f"\nwrote {OUT}")
print("Chance = 0.50. Region lengths are matched exactly, so a win cannot come from "
      "cleaning more audio. This SUPERSEDES the zeroed-frame protocol (0.325 vs 0.45).")
