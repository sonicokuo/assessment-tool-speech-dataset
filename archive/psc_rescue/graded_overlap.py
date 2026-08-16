"""Item 7 — GRADED partial-overlap synthesis. Eval-only, no retraining.

THE PROBLEM THIS SOLVES. Abstention is currently only demonstrable as a BINARY contrast:
verified per split, 99.1% of TEST mixtures sit at overlap >= 0.5 and 100% of clean clips sit
at exactly 0. So "the model abstains when the quantity is unrecoverable" and "the model
abstains on mixtures" are OBSERVATIONALLY IDENTICAL on our data. A reviewer can dismiss the
lead contribution as trivial mixture-detection, and they would be right to.

THE FIX. We can synthesise the missing middle. We hold the mixture and the clean-s1 stem, so

    interferer = mix - s1clean

recovers everything that is NOT speaker 1 (second talker + noise). Re-inject it over a
controlled fraction p of the clip:

    synth(p) = s1clean + interferer * mask_p

That sweeps overlap continuously from 0 (= the clean twin, a real training clip) to 1 (= the
original mixture, also a real training clip). Every intermediate point is a genuine mixture of
the same two sources under the same generative process -- in-distribution by construction, with
both endpoints verified to be actual dataset clips.

WHAT IT MEASURES. Ground truth is INVARIANT for the 9 intrinsic features (VERIFIED on all 3000
test twins: snr, f0_mean, f0_sd, speaking_rate, pause_count, pause_rate, jitter, shimmer, hnr
are 100% identical mixture-vs-s1clean). So as p rises, the TRUE value does not move and any
change in the model's output is degradation caused by observability loss. That gives us, per
feature, two curves that the binary contrast cannot produce:

    error(p)  -- does accuracy degrade GRADUALLY with observability?
    sigma(p)  -- does the model's own uncertainty RISE with observability loss?

If sigma(p) tracks error(p) across the sweep, abstention is driven by observability, not by
mixture-detection. That is the claim contribution 2 actually needs.

Usage: graded_overlap.py <ckpt> <mix_dir> <s1clean_dir> <proc_dir> <csv> <out.json> [n_clips]
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
from data.feature_set import SUPERVISED_FEATURES, FEATURE_NAMES  # noqa: E402
from model.adapter import build_adapter  # noqa: E402
from preprocess import build_overlap_info  # noqa: E402  -- reuse the EXACT training convention

CKPT, MIX_DIR, S1_DIR, PROC_DIR, CSV_PATH, OUT = sys.argv[1:7]
NCLIP = int(sys.argv[7]) if len(sys.argv) > 7 else 60
# BLIND CONTROL: zero the overlap channels so the model is NOT TOLD where the overlap is.
# Without this control the graded logvar curve is uninterpretable — the model could simply be
# reading its own oracle input channel rather than INFERRING unrecoverability from audio.
BLIND = len(sys.argv) > 8 and sys.argv[8] == "blind"
SR = 16000
GRID = [0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0]

# VERIFIED on all 3000 test twins: these two are per-signal (0% identical), the rest are 100%.
PER_SIGNAL = {"srmr", "overlap_ratio"}
INVARIANT = [f for f in FEATURE_NAMES if f not in PER_SIGNAL]

dev = "cuda" if torch.cuda.is_available() else "cpu"
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
for p_ in wavlm.parameters():
    p_.requires_grad = False

col = {n: c for n, c, _ in SUPERVISED_FEATURES}
GT: dict[str, dict[str, float]] = {}
with open(CSV_PATH, newline="") as fh:
    for r in _csv.DictReader(fh):
        stem = os.path.splitext(os.path.basename(r.get("filename", "")))[0]
        d = {}
        for n in INVARIANT:
            try:
                v = float(r.get(col[n], ""))
            except (TypeError, ValueError):
                continue
            if np.isfinite(v):
                d[n] = v
        GT[stem] = d
print(f"[init] device={dev} grid={GRID} invariant_feats={len(INVARIANT)} "
      f"BLIND={BLIND}", flush=True)


def load_wav(p: str) -> torch.Tensor:
    w, sr = torchaudio.load(p)
    if sr != SR:
        w = torchaudio.functional.resample(w, sr, SR)
    return w.mean(dim=0)


@torch.no_grad()
def run(wav: torch.Tensor, segs_str: str):
    feats = wavlm(wav.unsqueeze(0).to(dev)).last_hidden_state
    T = feats.shape[1]
    oi, _ = build_overlap_info(segs_str, 0.0, T, SR)     # exact training convention
    out = adapter(feats.float(), oi.unsqueeze(0).to(dev))[1]
    if isinstance(out, tuple):
        mean, logvar = out[0][0].float().cpu().numpy(), out[1][0].float().cpu().numpy()
    else:
        mean, logvar = out[0].float().cpu().numpy(), None
    return mean, logvar


err = {f: {p: [] for p in GRID} for f in INVARIANT}
sig = {f: {p: [] for p in GRID} for f in INVARIANT}
files = sorted(glob.glob(os.path.join(MIX_DIR, "*.wav")))
np.random.default_rng(0).shuffle(files)
used = 0

for path in files:
    if used >= NCLIP:
        break
    stem = os.path.splitext(os.path.basename(path))[0]
    s1p = os.path.join(S1_DIR, f"{stem}_s1clean.wav")
    g = GT.get(stem, {})
    if not (os.path.exists(s1p) and g):
        continue
    mix, s1 = load_wav(path), load_wav(s1p)
    n = min(len(mix), len(s1))
    mix, s1 = mix[:n], s1[:n]
    if n < 2 * SR:
        continue
    interferer = mix - s1                       # everything that is NOT speaker 1

    for p in GRID:
        L = int(round(p * n))
        synth = s1.clone()
        segs = ""
        if L > 0:
            start = (n - L) // 2                # centred, so p is the only thing that varies
            synth[start:start + L] = s1[start:start + L] + interferer[start:start + L]
            segs = f"{start}-{start + L}"
        mean, logvar = run(synth, "" if BLIND else segs)
        for fi, feat in enumerate(FEATURE_NAMES):
            if feat not in INVARIANT or feat not in g:
                continue
            e = abs(float(mean[fi]) - g[feat])
            if np.isfinite(e):
                err[feat][p].append(e)
            if logvar is not None and np.isfinite(logvar[fi]):
                sig[feat][p].append(float(logvar[fi]))
    used += 1
    if used % 10 == 0:
        print(f"  {used}/{NCLIP}", flush=True)


def spearman(x, y):
    if len(x) < 4:
        return float("nan")
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    if rx.std() < 1e-9 or ry.std() < 1e-9:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


print(f"\nGRADED OVERLAP SWEEP [{'BLIND: overlap channels ZEROED' if BLIND else 'oracle overlap GIVEN'}] — arm={os.path.basename(os.path.dirname(CKPT))}, n={used} clips")
print("GT is INVARIANT across the sweep, so any movement is observability loss.\n")
hdr = "".join(f"{p:>8.3f}" for p in GRID)
print(f"{'feature':<15}{'curve':<7}{hdr}{'rho(p)':>9}")
print("-" * (22 + 8 * len(GRID) + 9))
res = {}
for feat in INVARIANT:
    em = [float(np.mean(err[feat][p])) if err[feat][p] else float("nan") for p in GRID]
    sm = [float(np.mean(sig[feat][p])) if sig[feat][p] else float("nan") for p in GRID]
    if not np.isfinite(em).all():
        continue
    r_e = spearman(GRID, em)
    print(f"{feat:<15}{'err':<7}" + "".join(f"{v:>8.3f}" for v in em) + f"{r_e:>9.3f}")
    r_s = float("nan")
    if np.isfinite(sm).all():
        r_s = spearman(GRID, sm)
        print(f"{'':<15}{'logvar':<7}" + "".join(f"{v:>8.3f}" for v in sm) + f"{r_s:>9.3f}")
    res[feat] = {"grid": GRID, "err": em, "logvar": sm,
                 "rho_err_vs_p": r_e, "rho_logvar_vs_p": r_s}

json.dump({"ckpt": CKPT, "n_clips": used, "grid": GRID,
           "excluded_per_signal": sorted(PER_SIGNAL), "per_feature": res},
          open(OUT, "w"), indent=1)
print(f"\nwrote {OUT}")
print("READ: rho(err vs p) > 0  => accuracy degrades GRADUALLY with observability, not just "
      "on/off.\n      rho(logvar vs p) > 0 => the model's uncertainty TRACKS observability "
      "loss.\n      Both positive = abstention is observability-driven, NOT mixture-detection.")
