#!/usr/bin/env python3
"""snr_gain_slopes.py — the SNR double dissociation. No attribution map required.

WHY THIS REPLACES THE MAP ROW
The exact influence map phi_t = (10/ln10)(a_t/A - b_t/B) is 87% explained by the plain
loudness envelope at our token rate (envelope null 0.586 vs ceiling 0.674, n=1488), because
WHAM noise is quasi-stationary so b_t/B is near-flat. Measured, not assumed. And NO map
definition escapes it: single-frame deletion reduces to phi to first order, the local-SNR
timeline is log-envelope plus a constant, and Shapley/IG collapse to the same object. The
degeneracy is in the DATA, not the functional.

But the question "does the model read the NOISE or just the LOUDNESS" is behavioural, and
this corpus answers it exactly. Because snr_db is realised by a noise gain, scaling any
component moves the TRUE label in CLOSED FORM:

    noise  x c   ->  snr_db - 20log10(c)      true slope vs 20log10(c) = -1
    speech x c   ->  snr_db + 20log10(c)      true slope = +1
    both   x c   ->  snr_db unchanged         true slope =  0

A loudness reader cannot produce this pattern: it has no way to give OPPOSITE signs to the
two single-component gains. The +1/-1 pair is the discriminative part; the global row is
weaker because a gain-normalising front end passes it trivially (checked and reported).

In-distribution by construction: the training set sampled SNR uniformly on [0,40] and
scaled WHAM, so a gain sweep re-runs the generative process at a different draw. Edits that
push the realised SNR outside [0,40] are SKIPPED rather than extrapolated.

Replay gotchas that silently break this (each cost a run): `_load_real_rir` takes channel 0
NOT the mean; it PEAK-NORMALISES the RIR; the RIR list excludes filenames containing
"noise". Simulated-RIR clips cannot be replayed and are skipped.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import scipy.signal as sps
import soundfile as sf
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from data.feature_set import SUPERVISED_FEATURES  # noqa: E402
from model.adapter import build_adapter           # noqa: E402

SR, HOP = 16000, 320


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--mix_dir", required=True)
    ap.add_argument("--wham_dir", required=True)
    ap.add_argument("--rir_glob", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--p_real", type=float, default=0.5)
    ap.add_argument("--db_steps", default="-6,-3,3,6")  # modest: fixed-norm edits change amplitude   # in dB; c = 10**(db/20)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    steps = [float(x) for x in a.db_steps.split(",")]
    short = [f[0] if isinstance(f, (tuple, list)) else str(f) for f in SUPERVISED_FEATURES]
    SNR_I = short.index("snr")

    from transformers import WavLMModel
    wavlm = WavLMModel.from_pretrained("microsoft/wavlm-large").to(a.device).eval()

    ck = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    cfg = ck.get("config", {}) or {}
    adapter = build_adapter(
        variant=cfg.get("adapter_variant", "attn-concat"),
        audio_dim=cfg.get("audio_dim", 1024), lm_dim=cfg.get("lm_dim", 4096),
        compression=cfg.get("compression", 8), aux_pool=cfg.get("aux_pool") or "mean",
        reliability_head=bool(cfg.get("reliability_head", False)))
    miss, unexp = adapter.load_state_dict(
        ck.get("adapter_state_dict") or ck.get("adapter") or {}, strict=False)
    if miss or unexp:
        print(f"[fatal] state_dict mismatch missing={list(miss)} unexpected={list(unexp)}")
        return 1
    print("[load] state_dict=EXACT MATCH", flush=True)
    adapter.eval().to(a.device)

    @torch.no_grad()
    def predict(wav):
        x = torch.from_numpy(wav.astype(np.float32)).unsqueeze(0).to(a.device)
        h = wavlm(x).last_hidden_state
        oi = torch.zeros(h.shape[1], 4).unsqueeze(0).to(a.device)   # audio-only condition
        r = adapter(h.float(), oi.float())
        v = r[1] if isinstance(r, (tuple, list)) else r
        if isinstance(v, (tuple, list)):
            v = v[0]
        if v.dim() == 3:
            v = v.mean(dim=1)
        return float(v[0, SNR_I])

    mixes = sorted(glob.glob(os.path.join(a.mix_dir, "*.wav")))
    real_rirs = [f for f in sorted(glob.glob(a.rir_glob))
                 if "noise" not in os.path.basename(f).lower()]
    wham_files = glob.glob(os.path.join(a.wham_dir, "*.wav"))
    print(f"mixes={len(mixes)} rirs={len(real_rirs)} wham={len(wham_files)}", flush=True)

    # ⚠️ FIXED NORMALISATION (2026-08-12). Re-peaking EACH edit made the two single-
    # component arms algebraically identical: normalize(c*xr + noise) == normalize(xr +
    # noise/c), because the leading scalar cancels under peak division. The data showed it
    # exactly -- noise(-6dB) and speech(+6dB) agreed to four decimals, i.e. the model saw
    # the SAME waveform. One experiment reported as two.
    # Normalising by the ORIGINAL clip's peak keeps the conditions distinct AND makes the
    # both-gain arm a genuine AMPLITUDE-INVARIANCE control instead of a tautology.
    def make_norm(y0):
        pk = float(np.max(np.abs(y0))) or 1.0
        return lambda y: 0.98 * y / pk

    out, n_sim = [], 0
    for idx, mp in enumerate(mixes):
        if len(out) >= a.n:
            break
        rng = np.random.default_rng(a.seed * 1_000_003 + idx)
        try:
            x, sr = sf.read(mp)
            x = (x.mean(1) if x.ndim > 1 else x).astype(np.float64)
            if not (len(real_rirs) and rng.random() < a.p_real):
                n_sim += 1
                continue
            h, hsr = sf.read(real_rirs[rng.integers(len(real_rirs))])
            if h.ndim > 1:
                h = h[:, 0]
            if hsr != sr:
                h = sps.resample_poly(h, sr, hsr)
            h = h.astype(np.float64)
            h = h / (np.max(np.abs(h)) or 1.0)
            xr = sps.fftconvolve(x, h)[:len(x)]
            snr0 = float(rng.uniform(0.0, 40.0))
            nz, nsr = sf.read(wham_files[rng.integers(len(wham_files))])
            nz = nz.mean(1) if nz.ndim > 1 else nz
            if nsr != sr:
                nz = sps.resample_poly(nz.astype(np.float64), sr, nsr)
            nz = nz.astype(np.float64)
            if len(nz) < len(xr):
                nz = np.tile(nz, int(np.ceil(len(xr) / max(len(nz), 1))))
            off = rng.integers(0, max(1, len(nz) - len(xr) + 1))
            nz = nz[off:off + len(xr)]
            ps, pn = np.mean(xr ** 2) + 1e-12, np.mean(nz ** 2) + 1e-12
            g = np.sqrt(ps / (pn * (10 ** (snr0 / 10.0))))
            noise = g * nz
        except Exception:                                   # noqa: BLE001
            continue

        norm = make_norm(xr + noise)          # peak of the UNEDITED clip, fixed for all arms
        rec = {"snr0": snr0, "base": predict(norm(xr + noise)),
               "noise": {}, "speech": {}, "both": {}}
        for db in steps:
            c = 10 ** (db / 20.0)
            # noise-only: true snr moves by -db ; speech-only: +db ; both: unchanged
            if 0.0 <= snr0 - db <= 40.0:
                rec["noise"][str(db)] = predict(norm(xr + c * noise))
            if 0.0 <= snr0 + db <= 40.0:
                rec["speech"][str(db)] = predict(norm(c * xr + noise))
            rec["both"][str(db)] = predict(norm(c * (xr + noise)))
        out.append(rec)
        if len(out) % 25 == 0:
            print(f"  {len(out)}/{a.n}", flush=True)

    json.dump(out, open(a.out, "w"))

    def slope(kind, sign):
        X, Y = [], []
        for r in out:
            for k, v in r[kind].items():
                X.append(sign * float(k)); Y.append(v - r["base"])
        if len(X) < 20:
            return None
        X, Y = np.array(X), np.array(Y)
        s = float(np.polyfit(X, Y, 1)[0])
        rng2 = np.random.default_rng(0)
        bs = [float(np.polyfit(X[i], Y[i], 1)[0]) for i in
              (rng2.integers(0, X.size, X.size) for _ in range(1000)) if X[i].std() > 1e-9]
        return s, float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)), X.size

    print(f"\n=== SNR GAIN-SLOPE DOUBLE DISSOCIATION  (n={len(out)} clips, "
          f"simulated-RIR skipped={n_sim}) ===")
    print("    a RATIO reader gives -1 / +1 / 0 ; a LOUDNESS reader cannot give opposite")
    print("    signs to the two single-component gains")
    # sign is applied to the x-axis, so a faithful model yields +1 on BOTH single-
    # component arms (my earlier "-1" target double-counted the sign and mislabelled
    # a correct result as off-target).
    for kind, sign, want in (("noise", -1.0, +1.0), ("speech", +1.0, +1.0), ("both", +1.0, 0.0)):
        r = slope(kind, sign)
        if r is None:
            print(f"    {kind:<8} too few"); continue
        s, lo, hi, n = r
        ok = "MATCHES" if lo <= want <= hi else "off-target"
        print(f"    {kind+'-gain':<12} slope {s:+.3f} [{lo:+.3f},{hi:+.3f}]  "
              f"true {want:+.0f}  {ok}   n={n}")
    print("\n    NOTE: a gain-normalising front end passes 'both' trivially; the +1/-1 pair")
    print("    is the discriminative evidence.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
