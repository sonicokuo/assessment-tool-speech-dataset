"""Item 24 — aux-head-only readout, re-pooled four ways from ONE forward pass.

WHY THIS IS DECISIVE. The recorded test gap (lsm verified-slot 0.6927 vs mean-pool arms
0.7236/0.7246) was originally blamed on "selection overfitting on a 200-clip val set". That
was WRONG: val is FREE DECODE (parsed LM digits, train.py:2600-2609) and test is
VERIFIED-SLOT (aux-head scalar), so the two columns are different ESTIMATORS, not two splits.
The open question is therefore: is lsm's REPRESENTATION broken, or only its POOLING?

We answer it by applying the head PER FRAME once (giving z_t and logvar_t) and then pooling
those same per-frame outputs four different ways. Any accuracy difference between readouts is
attributable to POOLING ALONE, because the representation and the head weights are identical
across all four.

  native      — the pooling the checkpoint was actually trained with.
  mean        — plain unweighted mean. If lsm's representation is healthy, this RECOVERS
                accuracy and proves the deficit is pooling, not representation.
  obs         — softmax(-logvar_t): normalized inverse-variance / precision weighting.
                This is the proposed repair. Weight by how TRUSTWORTHY a frame is, not how
                BIG its value is.
  absz        — |z_t|-weighted (lsm's rule) applied to whatever checkpoint is loaded. On a
                mean-trained checkpoint this is the counterfactual "what would lsm pooling
                have cost this representation".

Padding is moot: batch size is 1, so every frame is real.

Usage: aux_repool.py <ckpt> <test_dir> <features_csv> <out.json> [n_clips]
"""
from __future__ import annotations

import csv
import glob
import json
import os
import sys
import types

import numpy as np

# mamba_ssm needs CUDA at import on some builds; the attn variants never touch it.
try:
    raise ImportError
except Exception:
    _m = types.ModuleType("mamba_ssm")

    class _NoMamba:  # pragma: no cover
        def __init__(self, *a, **k):
            raise RuntimeError("Mamba unavailable in this probe env")

    _m.Mamba = _NoMamba
    sys.modules["mamba_ssm"] = _m

import torch  # noqa: E402

sys.path.insert(0, os.path.join(os.getcwd(), "src"))
from data.feature_set import SUPERVISED_FEATURES, FEATURE_NAMES  # noqa: E402
from eval.selection_metric import HEADLINE_FEATURES  # noqa: E402
from model.adapter import build_adapter  # noqa: E402

CKPT, TEST, CSV_PATH, OUT = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
NCLIP = int(sys.argv[5]) if len(sys.argv) > 5 else 0
# ITEM 52: zero the 4 overlap channels at inference. `overlap_info` is derived from ORACLE
# VAD on the CLEAN STEMS, which do not exist at test time in any real deployment. If
# srcc_robust holds without it, the abstention claim needs NO oracle input at all.
# NOTE this is a LOWER BOUND: we are zeroing a channel the model was TRAINED with, which is
# off-distribution for it. A model trained without the channel should do at least as well.
ZERO_OVL = len(sys.argv) > 6 and sys.argv[6] == "zero_overlap"

dev = "cuda" if torch.cuda.is_available() else "cpu"
ck = torch.load(CKPT, map_location="cpu", weights_only=False)
cfg = ck["config"]
native_pool = str(cfg.get("aux_pool", "mean"))
has_rel = bool(cfg.get("reliability_head", False))
print(f"ckpt={os.path.basename(os.path.dirname(CKPT))} native_pool={native_pool} "
      f"reliability_head={has_rel} device={dev} ZERO_OVERLAP={ZERO_OVL}", flush=True)

adapter = build_adapter(
    cfg["adapter_variant"], lm_dim=4096,
    reliability_head=has_rel,
    compression=int(cfg.get("compression", 8)),
    aux_pool=native_pool,
).to(dev).eval()
missing, _ = adapter.load_state_dict(ck["adapter_state_dict"], strict=False)
assert not [k for k in missing if "head" in k or "regress" in k], missing

# ── instrument ground truth ────────────────────────────────────────────────────
csv_col = {name: col for name, col, _fmt in SUPERVISED_FEATURES}
gt: dict[str, dict[str, float]] = {}
with open(CSV_PATH, newline="") as fh:
    for row in csv.DictReader(fh):
        stem = os.path.splitext(os.path.basename(row.get("filename") or row.get("file") or ""))[0]
        if not stem:
            continue
        d = {}
        for name in FEATURE_NAMES:
            raw = row.get(csv_col[name], "")
            try:
                v = float(raw)
            except (TypeError, ValueError):
                continue
            if np.isfinite(v):
                d[name] = v
        gt[stem] = d
print(f"[GT] instrument CSV: {len(gt)} clips", flush=True)

READOUTS = ("native", "mean", "obs", "absz")
preds: dict[str, dict[str, dict[str, float]]] = {r: {} for r in READOUTS}

files = sorted(glob.glob(os.path.join(TEST, "*.pt")))
if NCLIP:
    files = files[:NCLIP]
print(f"[run] {len(files)} clips", flush=True)

with torch.no_grad():
    for i, f in enumerate(files):
        d = torch.load(f, map_location="cpu", weights_only=False)
        stem = os.path.splitext(os.path.basename(d.get("filename", os.path.basename(f))))[0]
        af = d["audio_features"].unsqueeze(0).to(dev).float()
        oi = d["overlap_info"].unsqueeze(0).to(dev).float()
        if ZERO_OVL:
            oi = torch.zeros_like(oi)

        prefix = adapter.inner(af, oi)                 # (1, N, lm_dim)
        out = adapter.regress_head(prefix)             # per-frame
        if isinstance(out, tuple):
            z_t, lv_t = out[0][0].float(), out[1][0].float()   # (N,F) each
        else:
            z_t, lv_t = out[0].float(), None

        # native = whatever the model itself emits (authoritative, not a restated formula)
        emitted = adapter(af, oi)[1]
        if isinstance(emitted, tuple):
            emitted = emitted[0]
        native = emitted[0].float()

        vals = {"native": native, "mean": z_t.mean(0)}

        w = z_t.abs()
        w = w / w.sum(0, keepdim=True).clamp(min=1e-6)
        vals["absz"] = (w * z_t).sum(0)

        if lv_t is not None:
            e = torch.exp(-(lv_t - lv_t.max(0, keepdim=True).values))
            e = e / e.sum(0, keepdim=True).clamp(min=1e-12)
            vals["obs"] = (e * z_t).sum(0)
        else:
            vals["obs"] = vals["mean"]

        for r in READOUTS:
            v = vals[r].cpu().numpy()
            preds[r][stem] = {n: float(v[k]) for k, n in enumerate(FEATURE_NAMES)}

        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{len(files)}", flush=True)


def srcc(a: list[float], b: list[float]) -> float:
    if len(a) < 8:
        return float("nan")
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    if ra.std() < 1e-9 or rb.std() < 1e-9:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


print(f"\n{'readout':<10}" + "".join(f"{n[:11]:>12}" for n in FEATURE_NAMES) + f"{'ROBUST5':>10}")
print("-" * (10 + 12 * len(FEATURE_NAMES) + 10))
summary = {}
for r in READOUTS:
    per = {}
    for name in FEATURE_NAMES:
        xs, ys = [], []
        for stem, pv in preds[r].items():
            g = gt.get(stem, {})
            if name in g and name in pv and np.isfinite(pv[name]):
                xs.append(pv[name]); ys.append(g[name])
        per[name] = srcc(xs, ys)
    rob = [per[n] for n in HEADLINE_FEATURES if np.isfinite(per.get(n, float("nan")))]
    rmean = float(np.mean(rob)) if rob else float("nan")
    summary[r] = {"per_feature": per, "srcc_robust": rmean, "n": len(preds[r])}
    print(f"{r:<10}" + "".join(f"{per[n]:>12.3f}" for n in FEATURE_NAMES) + f"{rmean:>10.4f}")

json.dump({"ckpt": CKPT, "native_pool": native_pool, "summary": summary},
          open(OUT, "w"), indent=1)
print(f"\nwrote {OUT}")
print("READ: native vs mean isolates POOLING from REPRESENTATION; "
      "obs is the proposed repair; absz is lsm's rule applied to this representation.")
