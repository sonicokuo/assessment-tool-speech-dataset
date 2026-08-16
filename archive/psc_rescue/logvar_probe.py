"""GATE probe for the proposed fix `w_t ~ softmax(-logvar_t)` (EXPLAINABILITY_2026-08-07 s1.6).

Question: does the per-frame log-variance RISE under overlap? If it does not, sourcing the
pooling weights from it buys nothing and the fix should not be built.

Three quantities per feature, over ~12 clips, CPU only:
  Q1  corr(logvar_t, overlap_t)      > 0  => uncertainty rises under overlap  (fix viable)
  Q2  corr(logvar_t, |z_t|)               => is logvar just magnitude in disguise?
  Q3  corr(w_new, overlap) vs corr(w_cur, overlap), where
        w_cur = |z_t| / sum|z|              (what the model does NOW)
        w_new = softmax(-logvar_t)          (what the fix would do)
      For the ill-posed features (f0*, jitter, shimmer, hnr) w_new must be MORE NEGATIVE
      than w_cur, i.e. the new weighting must AVOID overlapped frames.
"""
import glob, os, sys, types
import numpy as np

try:
    raise ImportError
except Exception:
    m = types.ModuleType("mamba_ssm")
    class _NoMamba:
        def __init__(self, *a, **k): raise RuntimeError("no mamba")
    m.Mamba = _NoMamba
    sys.modules["mamba_ssm"] = m

import torch
sys.path.insert(0, os.path.join(os.getcwd(), "src"))
from model.adapter import build_adapter
from data.feature_set import FEATURE_NAMES

torch.manual_seed(0)
CKPT, TEST = sys.argv[1], sys.argv[2]
NCLIP = int(sys.argv[3]) if len(sys.argv) > 3 else 12

ck = torch.load(CKPT, map_location="cpu", weights_only=False)
cfg = ck["config"]
print("ckpt:", os.path.basename(os.path.dirname(CKPT)),
      "| aux_pool =", cfg.get("aux_pool"), "| reliability_head =", cfg.get("reliability_head"))
if not cfg.get("reliability_head"):
    print("!! reliability_head is FALSE -> no logvar_t exists -> FIX IS NOT APPLICABLE"); sys.exit(0)

adapter = build_adapter(cfg["adapter_variant"], lm_dim=4096,
                        reliability_head=True,
                        compression=int(cfg.get("compression", 8)),
                        aux_pool=str(cfg.get("aux_pool", "mean"))).eval()
missing, _ = adapter.load_state_dict(ck["adapter_state_dict"], strict=False)
assert not [k for k in missing if "head" in k or "regress" in k], missing

def pear(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    a, b = a[m], b[m]
    if len(a) < 6 or a.std() < 1e-9 or b.std() < 1e-9: return np.nan
    return float(np.corrcoef(a, b)[0, 1])

F = len(FEATURE_NAMES)
q1 = [[] for _ in range(F)]; q2 = [[] for _ in range(F)]
wc = [[] for _ in range(F)]; wn = [[] for _ in range(F)]
files = sorted(glob.glob(f"{TEST}/*.pt"))[:NCLIP * 3]
used = 0
with torch.no_grad():
    for f in files:
        if used >= NCLIP: break
        d = torch.load(f, map_location="cpu", weights_only=False)
        oi = d["overlap_info"].unsqueeze(0).float()
        if float(oi[0, :, 0].mean()) < 0.05: continue      # need overlap to correlate against
        af = d["audio_features"].unsqueeze(0).float()
        prefix = adapter.inner(af, oi)
        out = adapter.regress_head(prefix)
        if not isinstance(out, tuple):
            print("!! regress_head did not return (mean, logvar)"); sys.exit(0)
        z = out[0][0].float().numpy()                       # (N, F)
        lv = out[1][0].float().numpy()                      # (N, F)
        N = z.shape[0]
        ov = torch.nn.functional.adaptive_avg_pool1d(
            oi[0, :, 0][None, None, :], N)[0, 0].numpy()
        a = np.abs(z)
        w_cur = a / np.clip(a.sum(0, keepdims=True), 1e-6, None)
        e = np.exp(-(lv - lv.max(0, keepdims=True)))
        w_new = e / np.clip(e.sum(0, keepdims=True), 1e-12, None)
        for k in range(F):
            q1[k].append(pear(lv[:, k], ov)); q2[k].append(pear(lv[:, k], a[:, k]))
            wc[k].append(pear(w_cur[:, k], ov)); wn[k].append(pear(w_new[:, k], ov))
        used += 1

ILL = {"f0_mean", "f0_sd", "jitter", "shimmer", "hnr"}
print(f"\nclips used = {used}\n")
print(f"{'feature':<15}{'Q1 lv~ovl':>11}{'Q2 lv~|z|':>11}{'w_cur~ovl':>11}{'w_new~ovl':>11}{'verdict':>16}")
print("-" * 76)
nm = lambda x: float(np.nanmean(x)) if len(x) else np.nan
for k, name in enumerate(FEATURE_NAMES):
    a, b, c, e = nm(q1[k]), nm(q2[k]), nm(wc[k]), nm(wn[k])
    v = ""
    if name in ILL:
        v = "FIX HELPS" if (e < c - 0.05) else ("no change" if abs(e - c) <= 0.05 else "FIX HURTS")
    print(f"{name:<15}{a:>11.3f}{b:>11.3f}{c:>11.3f}{e:>11.3f}{v:>16}")
print("\nGATE: Q1>0 on ill-posed feats => uncertainty rises under overlap.")
print("      w_new more NEGATIVE than w_cur on ill-posed feats => the fix does what we want.")
