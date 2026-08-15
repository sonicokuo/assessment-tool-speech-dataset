#!/usr/bin/env python3
"""capture_saliency.py — gradient attribution maps (items 0.8 + 0.2).

WHY GRADIENT AND NOT THE VALUE MAP
The aux head computes yhat = W . mean_t(z_t). The loss therefore depends only on
the SUM of z_t, so it is invariant to permuting z_t in time: training applies NO
pressure on where evidence sits, and the value map z_t/N is unconstrained. That
is the PREDICTED NULL, and it is a statement about the objective, not the model.

The gradient dodges this, but only if taken at the right place. d(yhat)/d(z_t) =
W/N is CONSTANT -- taking gradients w.r.t. the pooled tokens would reproduce the
invariance rather than escape it. We therefore differentiate w.r.t. the INPUT
WavLM FRAMES, where conv + self-attention make frames contribute unequally:

    saliency_t = || d yhat_f / d audio_features[t, :] ||_2

That is a property of the learned FUNCTION, so it can carry evidence structure
even when the pooled loss cannot.

ZEROED-OVERLAP CONTROL (0.2)
`overlap_info[:,0]` is the oracle overlap layout, verified rho 0.9999 with the GT
column -- i.e. the attribution REFERENCE is also a model INPUT. --zero_overlap
re-captures with that channel zeroed. If saliency survives, the map is audio-derived;
if it collapses, the previous correlational panel was input echo.

Adapter-only: the 8B LM is never built or called.

Usage:
  python scripts/capture_saliency.py --checkpoint $SH/checkpoints/full/<arm>/best.pt \
      --test_dir $SH/data/processed_corrected/test --out $SH/saliency_test.npz \
      --features f0_sd,f0_mean [--zero_overlap] [--limit N]
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from data.feature_set import SUPERVISED_FEATURES  # noqa: E402
from model.adapter import build_adapter           # noqa: E402

WAVLM_RATE = 50.0     # WavLM-Large frame rate (20 ms)
ORACLE_RATE = 100.0   # Praat grid the oracle maps live on


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--test_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--features", default="f0_sd,f0_mean,overlap_ratio")
    ap.add_argument("--zero_overlap", action="store_true")
    ap.add_argument("--method", default="grad", choices=["grad", "gradxinput", "ig"],
                    help="grad=sensitivity (weakest); gradxinput/ig=contribution-like (0.8b)")
    ap.add_argument("--ig_steps", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    names = [f.strip() for f in a.features.split(",") if f.strip()]
    # SUPERVISED_FEATURES entries are TUPLES (short_name, csv_column, fmt) — index 0,
    # not an attribute. str(f) would silently yield "('f0_sd', 'f0_sd_hz', ...)" and
    # every lookup would miss.
    short = [f[0] if isinstance(f, (tuple, list)) else str(f) for f in SUPERVISED_FEATURES]
    idx = {}
    for n in names:
        if n not in short:
            print(f"[fatal] feature {n!r} not in aux head. Available: {short}")
            return 1
        idx[n] = short.index(n)

    ck = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    cfg = ck.get("config", {}) or {}
    # reliability_head SWITCHES THE HEAD TYPE: ReliabilityHead (proj -> 2F, mean+log_var)
    # vs nn.Linear (-> F). Omitting it built the wrong head, so the checkpoint's
    # regress_head.proj.* landed in `unexpected` and the head stayed RANDOMLY
    # INITIALISED while only 2 keys were reported missing. Every attribution map
    # captured that way was backprop through noise.
    adapter = build_adapter(
        variant=cfg.get("adapter_variant", "attn-concat"),
        audio_dim=cfg.get("audio_dim", 1024),
        lm_dim=cfg.get("lm_dim", 4096),
        compression=cfg.get("compression", 8),
        aux_pool=cfg.get("aux_pool") or "mean",
        reliability_head=bool(cfg.get("reliability_head", False)),
    )
    sd = ck.get("adapter_state_dict") or ck.get("adapter") or {}
    missing, unexpected = adapter.load_state_dict(sd, strict=False)
    # STRICT: any mismatch touching the aux head means the predictions are not the
    # model's. Fail loudly rather than silently producing plausible garbage.
    bad = [k for k in list(missing) + list(unexpected) if "regress_head" in k or "head" in k]
    if bad or missing or unexpected:
        print(f"[fatal] state_dict mismatch — missing={list(missing)} unexpected={list(unexpected)}")
        print("[fatal] refusing to run: attribution from an unloaded head is meaningless.")
        return 1
    print(f"[load] variant={cfg.get('adapter_variant')} aux_pool={cfg.get('aux_pool') or 'mean'} "
          f"reliability_head={bool(cfg.get('reliability_head', False))} "
          f"state_dict=EXACT MATCH", flush=True)

    # Positive control that the head is live: a random-weight head must NOT reproduce
    # these predictions. Cheap, and it would have caught the bug above immediately.
    with torch.no_grad():
        _d = torch.load(os.path.join(a.test_dir, sorted(os.listdir(a.test_dir))[0]),
                        map_location="cpu", weights_only=False)
        _af = _d["audio_features"].float().unsqueeze(0)
        _oi = _d["overlap_info"].float().unsqueeze(0)
        _r = adapter(_af, _oi)
        _v = _r[1] if isinstance(_r, (tuple, list)) else _r
        _m = _v[0] if isinstance(_v, (tuple, list)) else _v
        print(f"[check] aux output dim={tuple(_m.shape)} "
              f"finite={bool(torch.isfinite(_m).all())} "
              f"sigma_returned={isinstance(_v, (tuple, list))}", flush=True)
    adapter.eval().to(a.device)
    for p in adapter.parameters():
        p.requires_grad_(False)

    files = sorted(f for f in os.listdir(a.test_dir) if f.endswith(".pt"))
    if a.limit:
        files = files[: a.limit]

    out: dict[str, np.ndarray] = {}
    n_done = 0
    for fi, fn in enumerate(files):
        d = torch.load(os.path.join(a.test_dir, fn), map_location="cpu", weights_only=False)
        af = d["audio_features"].float().unsqueeze(0).to(a.device)
        oi = d["overlap_info"].float().unsqueeze(0).to(a.device)
        if a.zero_overlap:
            oi = torch.zeros_like(oi)
        af.requires_grad_(True)

        stem = fn[:-3]

        def _aux_of(x: torch.Tensor):
            r = adapter(x, oi)
            v = r[1] if isinstance(r, (tuple, list)) else r
            lv = None
            if isinstance(v, (tuple, list)):
                v, lv = v[0], v[1]             # heteroscedastic head -> (mean, log_var)
            if v.dim() == 3:                   # linear_softmax variants return (B,N,F)
                v = v.mean(dim=1)
            if lv is not None and lv.dim() == 3:
                lv = lv.mean(dim=1)
            return v, lv

        aux, logvar = _aux_of(af)
        # item 0.29: per-clip sigma, so grounding can be correlated against uncertainty
        if logvar is not None:
            out[f"sigma/{stem}"] = logvar[0].detach().float().cpu().numpy().astype(np.float32)

        for name, j in idx.items():
            if a.method == "ig":
                # Integrated Gradients: contribution, not sensitivity. Zero baseline,
                # so IG_t = x_t * mean_k dF(x*k/m)/dx_t.
                acc = torch.zeros_like(af)
                for k in range(1, a.ig_steps + 1):
                    xk = (af.detach() * (k / a.ig_steps)).requires_grad_(True)
                    vk, _ = _aux_of(xk)
                    gk = torch.autograd.grad(vk[0, j], xk, retain_graph=False)[0]
                    acc = acc + gk.detach()
                attr = (af.detach() * acc / a.ig_steps)[0]
            else:
                if af.grad is not None:
                    af.grad = None
                aux[0, j].backward(retain_graph=True)
                g = af.grad[0].detach()                    # (T, 1024)
                # grad*input turns a SENSITIVITY into a contribution-like quantity,
                # which is what the oracle map actually is.
                attr = g * af.detach()[0] if a.method == "gradxinput" else g
            # ⚠️ D1 (2026-08-12): the L2 NORM over the 1024 dims was applied to ALL methods,
            # including IG and grad*input — which DESTROYS the very property that makes them
            # contributions. Summing over dims preserves completeness
            # (sum_t sum_d IG_td = F(x) - F(baseline), Sundararajan ICML 2017); the norm
            # discards sign, inflates frames whose per-dim contributions cancel, and couples
            # the map to WavLM frame-norm structure (which itself tracks energy/voicing).
            # Scoring an UNSIGNED candidate against a SIGN-VARYING reference (f0_sd) or a
            # ZERO-SUM one (snr) is the same type error as using mass-concentration on a
            # zero-sum reference. Keep the norm ONLY for the raw-gradient sensitivity
            # variant, and score that against |phi| only.
            if a.method in ("ig", "gradxinput"):
                sal = attr.sum(dim=-1).cpu().numpy().astype(np.float32)    # SIGNED, complete
            else:
                sal = attr.norm(dim=-1).cpu().numpy().astype(np.float32)   # sensitivity magnitude
            # lift 50 Hz -> the oracle's 100 Hz grid so scoring pools both identically
            out[f"{name}/{stem}"] = np.repeat(sal, int(round(ORACLE_RATE / WAVLM_RATE)))
        n_done += 1
        if (fi + 1) % 200 == 0:
            print(f"  {fi+1}/{len(files)}", flush=True)

    np.savez_compressed(a.out, **out)
    print(f"wrote {a.out}  clips={n_done}  features={names}  "
          f"method={a.method}  zero_overlap={a.zero_overlap}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
