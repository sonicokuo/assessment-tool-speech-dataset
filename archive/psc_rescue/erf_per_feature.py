"""Item 29 — per-feature EFFECTIVE RECEPTIVE FIELD, in seconds.

The claim this produces is a NUMBER, not a picture: "f0 evidence spans ~0.4 s, SNR spans the
clip". That is a quantitative, physics-checkable statement about how far each acoustic
quantity's evidence reaches, and it is far harder for a reviewer to wave away than a heatmap.
It also validates the maps independently: a LOCAL quantity (f0, jitter, pause onsets) must
show a narrow ERF and a GLOBAL one (snr, speaking_rate) a wide one. If that ordering comes out
right, the maps are tracking physics rather than an artifact.

PERTURBATION = RE-MIX SUBSTITUTION, not zeroing. We clean one 160 ms window by splicing in the
clean-s1 stem, exactly as in remix_deletion.py. Zeroing would put the input off the data
manifold (Hase et al., NeurIPS 2021) and measure the network's response to an impossible
signal; substitution keeps every probe in-distribution.

METRIC. Perturb window t0, measure the induced change in the per-frame head output z at every
token t, then summarise the spread two ways:
  * participation ratio  PR = (sum d)^2 / (N * sum d^2)  -> 1/N = one token, 1.0 = uniform.
  * centroid distance    mean |t - t0| weighted by d, converted to seconds.
Both are reported: PR is scale-free, the centroid is interpretable in seconds.

Usage: erf_per_feature.py <ckpt> <mix_dir> <s1clean_dir> <proc_dir> <out.json> [n_clips] [n_probes]
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
NCLIP = int(sys.argv[6]) if len(sys.argv) > 6 else 20
NPROBE = int(sys.argv[7]) if len(sys.argv) > 7 else 6
SR = 16000
TOKEN_S = 0.16  # one prefix token = 160 ms at 6.25 tok/s

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
print(f"[init] device={dev} nclip={NCLIP} nprobe={NPROBE}", flush=True)


def load_wav(p: str) -> torch.Tensor:
    w, sr = torchaudio.load(p)
    if sr != SR:
        w = torchaudio.functional.resample(w, sr, SR)
    return w.mean(dim=0)


@torch.no_grad()
def per_frame_z(wav: torch.Tensor, oi_full: torch.Tensor) -> np.ndarray:
    feats = wavlm(wav.unsqueeze(0).to(dev)).last_hidden_state
    T = feats.shape[1]
    oi = oi_full[:T].unsqueeze(0).to(dev).float()
    if oi.shape[1] < T:
        oi = torch.cat([oi, oi[:, -1:].expand(-1, T - oi.shape[1], -1)], dim=1)
    z = adapter.regress_head(adapter.inner(feats.float(), oi))
    z = z[0] if isinstance(z, tuple) else z
    return z[0].float().cpu().numpy()


pr = {f: [] for f in FEATURE_NAMES}
cen = {f: [] for f in FEATURE_NAMES}
files = sorted(glob.glob(os.path.join(MIX_DIR, "*.wav")))
rng.shuffle(files)
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
    mix, s1 = load_wav(path), load_wav(s1p)
    n = min(len(mix), len(s1))
    mix, s1 = mix[:n], s1[:n]
    if n < 2 * SR:
        continue

    z0 = per_frame_z(mix, oi_full)
    N = z0.shape[0]
    if N < 8:
        continue
    samp_per_tok = n / N

    # probe interior tokens only: edge tokens have truncated receptive fields and would
    # bias the spread downward for reasons that have nothing to do with the feature.
    lo_t, hi_t = max(1, N // 8), min(N - 2, N - N // 8)
    if hi_t <= lo_t:
        continue
    probes = rng.choice(np.arange(lo_t, hi_t), size=min(NPROBE, hi_t - lo_t), replace=False)

    for t0 in probes:
        lo = int(t0 * samp_per_tok)
        hi = min(n, int((t0 + 1) * samp_per_tok))
        cf = mix.clone()
        cf[lo:hi] = s1[lo:hi]                      # re-mix substitution, in-distribution
        z1 = per_frame_z(cf, oi_full)
        m = min(len(z0), len(z1))
        delta = np.abs(z1[:m] - z0[:m])             # (N, F)
        for fi, feat in enumerate(FEATURE_NAMES):
            dvec = delta[:, fi]
            s = dvec.sum()
            if not np.isfinite(s) or s <= 1e-12:
                continue
            p = dvec / s
            pr[feat].append(float(1.0 / (m * np.sum(p ** 2))))          # 1/N .. 1
            idx = np.arange(m)
            cen[feat].append(float(np.sum(p * np.abs(idx - t0)) * TOKEN_S))  # seconds
    used += 1
    if used % 5 == 0:
        print(f"  {used}/{NCLIP}", flush=True)

print(f"\nEFFECTIVE RECEPTIVE FIELD (re-mix substitution), "
      f"arm={os.path.basename(os.path.dirname(CKPT))}, n={used} clips\n")
print(f"{'feature':<15}{'PR (1/N..1)':>13}{'ERF span (s)':>14}{'n probes':>10}")
print("-" * 52)
res = {}
order = sorted(FEATURE_NAMES, key=lambda f: np.mean(cen[f]) if cen[f] else 1e9)
for feat in order:
    if len(cen[feat]) < 8:
        continue
    res[feat] = {"participation_ratio": float(np.mean(pr[feat])),
                 "erf_seconds": float(np.mean(cen[feat])), "n": len(cen[feat])}
    print(f"{feat:<15}{res[feat]['participation_ratio']:>13.3f}"
          f"{res[feat]['erf_seconds']:>14.3f}{res[feat]['n']:>10}")

json.dump({"ckpt": CKPT, "n_clips": used, "token_s": TOKEN_S, "per_feature": res},
          open(OUT, "w"), indent=1)
print(f"\nwrote {OUT}")
print("EXPECTED IF THE MAPS TRACK PHYSICS: LOCAL quantities (f0, jitter, shimmer, pause "
      "onsets) narrow; GLOBAL ones (snr, speaking_rate) wide. Sorted by span above.")
