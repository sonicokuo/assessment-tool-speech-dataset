"""Items 26/45/48, CORRECTED — in-distribution causal grounding by ERROR REDUCTION.

WHY THE FIRST VERSION WAS ILL-POSED. `remix_deletion.py` scored a win when cleaning the
attributed region moved the PREDICTION more than cleaning a random region. But re-mix
substitution removes INTERFERENCE, not speaker 1's intrinsic voice — and it is VERIFIED on all
3000 test twins that 9 of 11 GT features are 100% IDENTICAL between a mixture and its
`_s1clean` twin (only srmr and overlap_ratio are per-signal, 0% identical). So for those 9
features the counterfactual's TRUE VALUE IS UNCHANGED, and a moving prediction is INSTABILITY,
not evidence of grounding. That test rewarded the wrong behaviour.

WHAT THE UNCHANGED TRUTH ACTUALLY BUYS US. Because the target is fixed, cleaning a region can
only ADD information (it removes the interference that was corrupting the estimate). So the
error must fall, and the meaningful question is WHERE it falls fastest:

    err(x) = |pred_f(x) - truth_f|            truth is the SAME for mix and counterfactual
    win   <=>  err(cf_attributed) < err(cf_random)

A win means the attribution located the region whose corruption was actually degrading the
estimate. That is a causal claim about evidence, tested in-distribution, with region lengths
matched exactly so it cannot be won by cleaning more audio.

SCOPE. Only the 9 identical-GT features are scored. srmr and overlap_ratio are EXCLUDED and
reported as such: their true value changes under substitution, so scoring them would need the
instrument re-run on every counterfactual clip (a much larger job).

Usage: remix_error.py <ckpt> <mix_dir> <s1clean_dir> <proc_dir> <csv> <out.json> [n] [frac]
"""
from __future__ import annotations

import csv as _csv
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
from data.feature_set import FEATURE_NAMES, SUPERVISED_FEATURES  # noqa: E402
from model.adapter import build_adapter  # noqa: E402

CKPT, MIX_DIR, S1_DIR, PROC_DIR, CSV_PATH, OUT = sys.argv[1:7]
NCLIP = int(sys.argv[7]) if len(sys.argv) > 7 else 120
FRAC = float(sys.argv[8]) if len(sys.argv) > 8 else 0.25
SR = 16000

# VERIFIED 2026-08-07 on all 3000 test twins: identical 100%, these two 0%.
PER_SIGNAL = {"srmr", "overlap_ratio"}
SCORED = [f for f in FEATURE_NAMES if f not in PER_SIGNAL]

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

col = {n: c for n, c, _ in SUPERVISED_FEATURES}
GT: dict[str, dict[str, float]] = {}
with open(CSV_PATH, newline="") as fh:
    for r in _csv.DictReader(fh):
        stem = os.path.splitext(os.path.basename(r.get("filename", "")))[0]
        d = {}
        for n in SCORED:
            try:
                v = float(r.get(col[n], ""))
            except (TypeError, ValueError):
                continue
            if np.isfinite(v):
                d[n] = v
        GT[stem] = d
print(f"[init] device={dev} frac={FRAC} scored={len(SCORED)} feats "
      f"(excluded per-signal: {sorted(PER_SIGNAL)})", flush=True)


def load_wav(p: str) -> torch.Tensor:
    w, sr = torchaudio.load(p)
    if sr != SR:
        w = torchaudio.functional.resample(w, sr, SR)
    return w.mean(dim=0)


@torch.no_grad()
def predict(wav: torch.Tensor, oi_full: torch.Tensor):
    feats = wavlm(wav.unsqueeze(0).to(dev)).last_hidden_state
    T = feats.shape[1]
    oi = oi_full[:T].unsqueeze(0).to(dev).float()
    if oi.shape[1] < T:
        oi = torch.cat([oi, oi[:, -1:].expand(-1, T - oi.shape[1], -1)], dim=1)
    prefix = adapter.inner(feats.float(), oi)
    out = adapter(feats.float(), oi)[1]
    out = out[0] if isinstance(out, tuple) else out
    z = adapter.regress_head(prefix)
    z = z[0] if isinstance(z, tuple) else z
    return out[0].float().cpu().numpy(), z[0].float().cpu().numpy()


wins = {f: [] for f in SCORED}
red_a = {f: [] for f in SCORED}
red_r = {f: [] for f in SCORED}
files = sorted(glob.glob(os.path.join(MIX_DIR, "*.wav")))
rng.shuffle(files)
used = 0

for path in files:
    if used >= NCLIP:
        break
    stem = os.path.splitext(os.path.basename(path))[0]
    s1p = os.path.join(S1_DIR, f"{stem}_s1clean.wav")
    procp = os.path.join(PROC_DIR, f"{stem}.pt")
    g = GT.get(stem, {})
    if not (os.path.exists(s1p) and os.path.exists(procp) and g):
        continue
    d = torch.load(procp, map_location="cpu", weights_only=False)
    oi_full = d["overlap_info"].float()
    if float(oi_full[:, 0].mean()) < 0.05:
        continue
    mix, s1 = load_wav(path), load_wav(s1p)
    n = min(len(mix), len(s1))
    mix, s1 = mix[:n], s1[:n]
    if n < SR:
        continue

    base, z0 = predict(mix, oi_full)
    N = z0.shape[0]
    win_tok = max(1, int(round(FRAC * N)))
    spt = n / N

    for fi, feat in enumerate(FEATURE_NAMES):
        if feat not in SCORED or feat not in g:
            continue
        truth = g[feat]
        dev_map = np.abs(z0[:, fi] - z0[:, fi].mean())
        if not np.isfinite(dev_map).all() or dev_map.sum() <= 0:
            continue
        cs = np.concatenate([[0.0], np.cumsum(dev_map)])
        sums = cs[win_tok:] - cs[:-win_tok] if win_tok < len(cs) else np.array([cs[-1]])
        a_tok = int(np.argmax(sums))
        r_tok = int(rng.integers(0, max(1, N - win_tok + 1)))

        errs = []
        for start in (a_tok, r_tok):
            lo, hi = int(start * spt), min(n, int((start + win_tok) * spt))
            cf = mix.clone()
            cf[lo:hi] = s1[lo:hi]
            errs.append(abs(float(predict(cf, oi_full)[0][fi]) - truth))

        e0 = abs(float(base[fi]) - truth)
        ea, er = errs
        if all(np.isfinite(x) for x in (e0, ea, er)):
            wins[feat].append(1.0 if ea < er else (0.5 if ea == er else 0.0))
            red_a[feat].append(e0 - ea)     # positive = cleaning HELPED
            red_r[feat].append(e0 - er)

    used += 1
    if used % 20 == 0:
        print(f"  {used}/{NCLIP}", flush=True)

print(f"\nERROR-REDUCTION re-mix grounding (truth is INVARIANT under substitution), "
      f"arm={os.path.basename(os.path.dirname(CKPT))}, n={used} clips, window={FRAC:.0%}\n")
print(f"{'feature':<15}{'win rate':>10}{'95% CI':>18}{'err_red att':>13}{'err_red rnd':>13}{'n':>6}")
print("-" * 76)
res = {}
for feat in SCORED:
    w = np.asarray(wins[feat], float)
    if len(w) < 8:
        continue
    bs = np.random.default_rng(1).integers(0, len(w), size=(2000, len(w)))
    ci = (float(np.percentile(w[bs].mean(1), 2.5)), float(np.percentile(w[bs].mean(1), 97.5)))
    ra, rr = float(np.mean(red_a[feat])), float(np.mean(red_r[feat]))
    star = "  <-- ABOVE chance" if ci[0] > 0.5 else ("  <-- below chance" if ci[1] < 0.5 else "")
    print(f"{feat:<15}{w.mean():>10.3f}   [{ci[0]:.3f},{ci[1]:.3f}]{ra:>13.4f}{rr:>13.4f}"
          f"{len(w):>6}{star}")
    # PER-CLIP ARRAYS (added 2026-08-08). The means alone cannot distinguish a genuine
    # heavy tail from a few outliers -- f0_mean showed err_red +0.526 attributed vs -0.061
    # random while its win rate was only 0.534, which means most clips are coin flips and a
    # minority carry the effect. Deciding which needs the raw per-clip values, so dump them.
    res[feat] = {"win_rate": float(w.mean()), "ci": ci, "err_reduction_attributed": ra,
                 "err_reduction_random": rr, "n": len(w),
                 "median_red_attributed": float(np.median(red_a[feat])),
                 "median_red_random": float(np.median(red_r[feat])),
                 "per_clip_red_attributed": [float(x) for x in red_a[feat]],
                 "per_clip_red_random": [float(x) for x in red_r[feat]],
                 "per_clip_win": [float(x) for x in wins[feat]]}

json.dump({"ckpt": CKPT, "n_clips": used, "frac": FRAC,
           "excluded_per_signal": sorted(PER_SIGNAL), "per_feature": res},
          open(OUT, "w"), indent=1)
print(f"\nwrote {OUT}")
print("Positive err_red = cleaning that region moved the estimate TOWARD truth. Win rate >0.5 "
      "= the attribution located the region whose corruption was degrading the estimate.")
