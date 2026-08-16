"""P2 — SELECTIVE RISK / AURC, the metric contribution 2 actually leads with.

WHY THIS IS THE RIGHT METRIC. Every baseline we have built tonight -- mel+ridge, the layer
sweep, the tuned probe family -- has 100% COVERAGE BY CONSTRUCTION. None can withhold a claim.
So none can be scored on selective risk at all. Accuracy comparisons we now lose; this is the
axis where the comparison is not merely favourable but UNDEFINED for the baseline.

WHAT IS COMPUTED, per feature:
  risk(c)  = mean |pred - truth| over the c most-confident fraction of clips
  AURC     = mean risk over coverage in (0, 1]           (lower is better)
  ORACLE   = same curve but ordered by TRUE error -- the unreachable floor
  RANDOM   = same curve under random ordering -- what "no useful confidence" gives
  E-AURC   = AURC - ORACLE, the excess attributable to imperfect ranking
  gain     = (RANDOM - AURC) / RANDOM, i.e. how much the model's own uncertainty buys

CONFIDENCE SIGNAL = -logvar from the heteroscedastic head. Note it is applied PER FEATURE:
abstention here is per-claim, not per-clip, which is the whole point -- a clip may have a
recoverable SNR and an unrecoverable f0 simultaneously.

REPORT BOTH PANELS. §1.8 established that the headline ROBUST5 set and the ILL-POSED set behave
oppositely under anything that touches observability, so a single mean hides the effect. The
ill-posed panel is where abstention is supposed to pay.

Usage: aurc_eval.py <ckpt> <test_dir> <features_csv> <out.json> [n_clips]
"""
from __future__ import annotations

import csv
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

sys.path.insert(0, os.path.join(os.getcwd(), "src"))
from data.feature_set import FEATURE_NAMES, SUPERVISED_FEATURES  # noqa: E402
from eval.selection_metric import HEADLINE_FEATURES  # noqa: E402
from model.adapter import build_adapter  # noqa: E402

CKPT, TEST, CSV_PATH, OUT = sys.argv[1:5]
NCLIP = int(sys.argv[5]) if len(sys.argv) > 5 else 0
ILL = ["f0_mean", "f0_sd", "jitter", "shimmer", "hnr"]
COL = {n: c for n, c, _ in SUPERVISED_FEATURES}

dev = "cuda" if torch.cuda.is_available() else "cpu"
ck = torch.load(CKPT, map_location="cpu", weights_only=False)
cfg = ck["config"]
if not cfg.get("reliability_head"):
    print("!! no reliability head -> no per-feature uncertainty -> AURC undefined"); sys.exit(1)
adapter = build_adapter(
    cfg["adapter_variant"], lm_dim=4096, reliability_head=True,
    compression=int(cfg.get("compression", 8)),
    aux_pool=str(cfg.get("aux_pool", "mean") or "mean"),
).to(dev).eval()
missing, _ = adapter.load_state_dict(ck["adapter_state_dict"], strict=False)
assert not [k for k in missing if "head" in k or "regress" in k], missing

gt = {}
with open(CSV_PATH, newline="") as fh:
    for r in csv.DictReader(fh):
        stem = os.path.splitext(os.path.basename(r.get("filename", "")))[0]
        d = {}
        for n in FEATURE_NAMES:
            try:
                v = float(r.get(COL[n], ""))
            except (TypeError, ValueError):
                continue
            if np.isfinite(v):
                d[n] = v
        gt[stem] = d

files = sorted(glob.glob(os.path.join(TEST, "*.pt")))
if NCLIP:
    files = files[:NCLIP]
print(f"[init] {len(files)} clips  pool={cfg.get('aux_pool')}  dev={dev}", flush=True)

P, LV, Y = [], [], []
with torch.no_grad():
    for i, f in enumerate(files):
        d = torch.load(f, map_location="cpu", weights_only=False)
        stem = os.path.splitext(os.path.basename(d.get("filename", os.path.basename(f))))[0]
        g = gt.get(stem)
        if not g:
            continue
        out = adapter(d["audio_features"].unsqueeze(0).to(dev).float(),
                      d["overlap_info"].unsqueeze(0).to(dev).float())[1]
        mean, logvar = out
        P.append(mean[0].float().cpu().numpy())
        LV.append(logvar[0].float().cpu().numpy())
        Y.append([g.get(n, np.nan) for n in FEATURE_NAMES])
        if (i + 1) % 1000 == 0:
            print(f"  {i+1}/{len(files)}", flush=True)

P, LV, Y = np.stack(P), np.stack(LV), np.stack(Y)
print(f"[data] {P.shape[0]} scored clips", flush=True)


def aurc(err, conf):
    """Mean risk over all coverage levels, ordered by DESCENDING confidence."""
    order = np.argsort(-conf)
    e = err[order]
    return float(np.mean(np.cumsum(e) / np.arange(1, len(e) + 1)))


res = {}
print(f"\n{'feature':<15}{'AURC':>9}{'oracle':>9}{'random':>9}{'E-AURC':>9}{'gain%':>8}{'n':>7}")
print("-" * 66)
rng = np.random.default_rng(0)
for k, name in enumerate(FEATURE_NAMES):
    m = np.isfinite(Y[:, k]) & np.isfinite(P[:, k]) & np.isfinite(LV[:, k])
    if m.sum() < 100:
        continue
    err = np.abs(P[m, k] - Y[m, k])
    a_model = aurc(err, -LV[m, k])                      # low variance = high confidence
    a_oracle = aurc(err, -err)                          # unreachable floor
    a_rand = float(np.mean([aurc(err, rng.random(len(err))) for _ in range(20)]))
    gain = 100.0 * (a_rand - a_model) / a_rand if a_rand > 0 else np.nan
    res[name] = {"aurc": a_model, "oracle": a_oracle, "random": a_rand,
                 "e_aurc": a_model - a_oracle, "gain_pct": gain, "n": int(m.sum())}
    print(f"{name:<15}{a_model:>9.3f}{a_oracle:>9.3f}{a_rand:>9.3f}"
          f"{a_model-a_oracle:>9.3f}{gain:>8.1f}{int(m.sum()):>7}")

for panel, feats in (("ROBUST5", HEADLINE_FEATURES), ("ILL-POSED", ILL)):
    g = [res[f]["gain_pct"] for f in feats if f in res and np.isfinite(res[f]["gain_pct"])]
    if g:
        print(f"\n  {panel} mean AURC gain over random confidence: {np.mean(g):+.1f}%")
json.dump({"ckpt": CKPT, "per_feature": res}, open(OUT, "w"), indent=1)
print(f"\nwrote {OUT}")
print("A ridge/mel probe has 100% coverage BY CONSTRUCTION and CANNOT be scored on this axis.")
