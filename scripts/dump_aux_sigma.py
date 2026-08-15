#!/usr/bin/env python3
"""dump_aux_sigma.py — per-clip aux predictions AND sigma, in the arm's own input condition.

WHY THIS EXISTS
Three separate jobs all need the same thing and none of them should wait 8-12 hours for a full
LM eval to produce it:

  1. `oracle_error_ceiling.py`  — needs per-clip |error|, i.e. clean predictions.
  2. sigma-gate tau calibration — needs per-clip sigma to pick per-feature thresholds on dev.
  3. risk-coverage / AURC curves — need (sigma, error) pairs; the curve is then a sweep, not a
     re-run.

It is an ADAPTER-ONLY forward pass: the 8B LM is never built. Minutes, not hours.

⚠️ THE INPUT CONDITION IS NOT OPTIONAL. `zero_overlap_input` is read from the CHECKPOINT's own
training config, because an arm trained with the channel zeroed never gave its
`OverlapEmbedding` Linear(4,32) any gradient — feeding it the oracle overlap at eval is
structured OOD noise through a random-init layer. That exact bug silently corrupted a 6000-clip
headline (4th occurrence of the unforwarded-config class), so this script resolves the flag
itself and PRINTS what it used rather than trusting a caller to pass it.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from data.feature_set import SUPERVISED_FEATURES  # noqa: E402
from model.adapter import build_adapter           # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--test_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dump_repr", action="store_true",
                    help="also save mean+std pooled ADAPTER prefix tokens as <out>_repr.npz")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    names = [f[0] if isinstance(f, (tuple, list)) else str(f) for f in SUPERVISED_FEATURES]
    ck = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    cfg = ck.get("config", {}) or {}
    dump_repr = bool(getattr(a, 'dump_repr', False))
    zero_ovl = bool(cfg.get("zero_overlap_input", False))

    adapter = build_adapter(
        variant=cfg.get("adapter_variant", "attn-concat"),
        audio_dim=cfg.get("audio_dim", 1024), lm_dim=cfg.get("lm_dim", 4096),
        compression=cfg.get("compression", 8), aux_pool=cfg.get("aux_pool") or "mean",
        reliability_head=bool(cfg.get("reliability_head", False)))
    miss, unexp = adapter.load_state_dict(
        ck.get("adapter_state_dict") or ck.get("adapter") or {}, strict=False)
    if miss or unexp:
        # A reliability_head checkpoint loaded into a plain nn.Linear leaves the head at RANDOM
        # INIT while reporting only two missing keys. Fail loudly; this has bitten twice.
        print(f"[fatal] state_dict mismatch missing={list(miss)} unexpected={list(unexp)}")
        return 1
    adapter.eval().to(a.device)
    print(f"[load] variant={cfg.get('adapter_variant')} "
          f"reliability_head={bool(cfg.get('reliability_head', False))} "
          f"zero_overlap_input={zero_ovl}  state_dict=EXACT MATCH", flush=True)

    files = sorted(f for f in os.listdir(a.test_dir) if f.endswith(".pt"))
    if a.limit:
        files = files[: a.limit]

    out = []
    reprs, repr_names = [], []
    with torch.no_grad():
        for i, fn in enumerate(files):
            d = torch.load(os.path.join(a.test_dir, fn), map_location="cpu", weights_only=False)
            af = d["audio_features"].float().unsqueeze(0).to(a.device)
            oi = d["overlap_info"].float().unsqueeze(0).to(a.device)
            if zero_ovl:
                oi = torch.zeros_like(oi)
            r = adapter(af, oi)
            v = r[1] if isinstance(r, (tuple, list)) else r
            lv = None
            if isinstance(v, (tuple, list)):
                v, lv = v[0], v[1]
            if v.dim() == 3:                      # linear_softmax variants return (B,N,F)
                v = v.mean(dim=1)
            if lv is not None and lv.dim() == 3:
                lv = lv.mean(dim=1)
            rec = {"filename": fn[:-3],
                   "aux_mean": v[0, :len(names)].float().cpu().tolist()}
            if dump_repr:
                # OUR ADAPTER REPRESENTATION (mean+std pooled prefix tokens). Needed because the
                # 3-arm abstention comparison built its features from the FROZEN WavLM tensor, so
                # arms B and C read the identical matrix and "B-C isolates our representation" was
                # false: only the appended point estimate differed. To compare our representation
                # against pooled frozen WavLM, the confidence model has to actually SEE this.
                pref = r[0] if isinstance(r, (tuple, list)) else r
                if torch.is_tensor(pref) and pref.dim() == 3:
                    reprs.append(torch.cat([pref[0].mean(0), pref[0].std(0)])
                                 .float().cpu().numpy())
                    repr_names.append(fn[:-3])
            if lv is not None:
                # store SIGMA directly (exp(0.5*log_var)); the gate thresholds sigma, and
                # storing it here means the risk-coverage curve never needs another forward pass
                rec["sigma"] = torch.exp(0.5 * lv[0, :len(names)]).float().cpu().tolist()
            out.append(rec)
            if (i + 1) % 1000 == 0:
                print(f"  {i+1}/{len(files)}", flush=True)

    json.dump(out, open(a.out, "w"))
    if dump_repr and reprs:
        import numpy as _np
        _np.savez_compressed(a.out.replace(".json", "") + "_repr.npz",
                             repr=_np.stack(reprs), filenames=_np.array(repr_names))
        print(f"[repr] wrote {a.out.replace('.json','')}_repr.npz  "
              f"{len(reprs)} x {reprs[0].shape[0]} (mean+std pooled prefix tokens)")
    has_sigma = sum(1 for r in out if "sigma" in r)
    print(f"\nwrote {a.out}  clips={len(out)}  with sigma={has_sigma}  "
          f"features={len(names)}  zero_overlap_input={zero_ovl}")
    if has_sigma:
        S = np.array([r["sigma"] for r in out if "sigma" in r])
        print("\nper-feature sigma (median / p10 / p90) — the range tau must be swept over:")
        for j, nm in enumerate(names):
            print(f"  {nm:<15}{np.median(S[:, j]):9.3f}{np.percentile(S[:, j], 10):9.3f}"
                  f"{np.percentile(S[:, j], 90):9.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
