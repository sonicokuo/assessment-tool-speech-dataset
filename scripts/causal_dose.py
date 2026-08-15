#!/usr/bin/env python3
"""causal_dose.py — DOSE-RESPONSE causal test (protocol P1 v2, 2026-08-11).

WHY THIS REPLACES THE WINDOW-CONTRAST VERSION
The previous design contrasted a HIGH-overlap window against a LOW-overlap "control".
Measured, that control failed: the LOW window averaged **62.6% overlap**, with 87% of
clips above 50% and only 0.5% below 10% — because 99.1% of test mixtures sit above 0.5
overlap, so THE DATA CONTAINS NO CLEAN CONTROL WINDOW. Both conditions removed real
interference, the contrast collapsed, and only jitter survived the paired test.

A dose-response needs no clean window. Attenuate the interferer GLOBALLY by
`alpha in {0, 0.25, 0.5, 0.75, 1}`:

    edited(alpha) = s1clean + (1 - alpha) * s2          [exact: mix = s1clean + s2]

alpha=0 is the untouched mixture; alpha=1 IS the s1clean twin, which is IN-DISTRIBUTION
(19,900 such clips are in training). Mixing is linear, so every intermediate alpha lies
on the straight-line path between two real training points — the same path stem-path
Integrated Gradients would integrate over, so this doubles as P6.

TWO ENDPOINTS, by GT provenance (verified at source, not from docstrings):

  INVARIANT — speaking_rate, pause_count, pause_rate, jitter, shimmer, hnr
      GT is measured on the CLEAN S1 STEM, so it is FIXED for every alpha. The test is
      whether the ERROR falls MONOTONICALLY as interference is removed. Monotonicity
      across 5 doses is a far stronger signal than a 2-point contrast and needs no
      control window.

  SENSITIVE — f0_mean, f0_sd, srmr
      GT MOVES with alpha and must be RECOMPUTED by running the instrument on the edited
      audio. f0 = Praat on the edited mixture restricted to voiced AND NOT overlapped,
      with the OVERLAP MASK ITSELF UPDATED (attenuating s2 removes the overlap; holding
      the original windows fixed was the bug that made |dGT| = 0.00 and invalidated the
      earlier f0 arm). srmr = SRMR recomputed on the edited audio
      (`regenerate_corrected.py:113` computes it on the DEGRADED MIXTURE, not the stem —
      srmr was misclassified as invariant in the first version).

  snr is the sampled construction parameter and never moves; overlap_ratio comes from
  stem VAD and is reported for completeness only.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from data.f0_clean import parse_overlap_windows_samples  # noqa: E402
from data.feature_set import SUPERVISED_FEATURES         # noqa: E402
from model.adapter import build_adapter                  # noqa: E402

SR = 16000
HOP = 320
FRAME = 160          # 10 ms frames for the s2 activity envelope


def s2_active_mask(s2: np.ndarray, windows, alpha: float):
    """Overlap mask AFTER attenuating s2 by (1-alpha).

    A frame stays overlapped only while the attenuated interferer is still audible.
    The threshold is CALIBRATED PER CLIP so that alpha=0 reproduces the original
    windows; alpha=1 necessarily empties the mask. Without this the excluded frame set
    never changes and the f0 GT cannot move -- the exact bug in the first version.
    """
    n = len(s2)
    nf = max(1, n // FRAME)
    rms = np.sqrt((s2[: nf * FRAME].reshape(nf, FRAME) ** 2).mean(axis=1) + 1e-12)
    orig = np.zeros(nf, dtype=bool)
    for (s, e) in windows:
        orig[max(0, int(s * SR) // FRAME): min(nf, int(e * SR) // FRAME)] = True
    if not orig.any():
        return []
    thr = float(np.percentile(rms[orig], 10))          # calibrate on the original mask
    keep = orig & ((1.0 - alpha) * rms > thr)
    out, i = [], 0
    while i < nf:
        if keep[i]:
            j = i
            while j < nf and keep[j]:
                j += 1
            out.append((i * FRAME / SR, j * FRAME / SR))
            i = j
        else:
            i += 1
    return out


def f0_stats(wav: np.ndarray, windows):
    import parselmouth
    snd = parselmouth.Sound(wav.astype(np.float64), sampling_frequency=SR)
    p = snd.to_pitch(time_step=0.01, pitch_floor=75.0, pitch_ceiling=500.0)
    q = p.selected_array["frequency"].astype(float)
    t = np.asarray(p.xs(), dtype=float)
    keep = q > 0
    for (s, e) in windows:
        keep &= ~((t >= s) & (t < e))
    if keep.sum() < 3:
        return float("nan"), float("nan")
    v = q[keep]
    return float(v.mean()), float(v.std(ddof=1))


def srmr_of(wav: np.ndarray):
    try:
        from versa.utterance_metrics.srmr import srmr_metric
        r = srmr_metric(wav.astype(np.float64), SR, n_cochlear_filters=23,
                        low_freq=125, min_cf=4, max_cf=128, fast=True, norm=False)
        return float(r["srmr"] if isinstance(r, dict) else r)
    except Exception:
        return float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--features_csv", required=True)
    ap.add_argument("--audio_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--alphas", default="0,0.25,0.5,0.75,1.0")
    ap.add_argument("--hold_input", action="store_true", help="freeze overlap channel at alpha=0")
    ap.add_argument("--zero_input", action="store_true", help="zero the overlap channel at every dose")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    alphas = [float(x) for x in a.alphas.split(",")]
    short = [f[0] if isinstance(f, (tuple, list)) else str(f) for f in SUPERVISED_FEATURES]

    from transformers import WavLMModel
    wavlm = WavLMModel.from_pretrained("microsoft/wavlm-large").to(a.device).eval()

    ck = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    cfg = ck.get("config", {}) or {}
    adapter = build_adapter(
        variant=cfg.get("adapter_variant", "attn-concat"),
        audio_dim=cfg.get("audio_dim", 1024), lm_dim=cfg.get("lm_dim", 4096),
        compression=cfg.get("compression", 8), aux_pool=cfg.get("aux_pool") or "mean",
        reliability_head=bool(cfg.get("reliability_head", False)))
    missing, unexpected = adapter.load_state_dict(
        ck.get("adapter_state_dict") or ck.get("adapter") or {}, strict=False)
    if missing or unexpected:
        print(f"[fatal] state_dict mismatch missing={list(missing)} unexpected={list(unexpected)}")
        return 1
    print("[load] state_dict=EXACT MATCH", flush=True)
    adapter.eval().to(a.device)

    @torch.no_grad()
    def predict(wav, oi):
        x = torch.from_numpy(wav.astype(np.float32)).unsqueeze(0).to(a.device)
        h = wavlm(x).last_hidden_state
        T = h.shape[1]
        o = oi[:T] if oi.shape[0] >= T else torch.cat(
            [oi, oi.new_zeros(T - oi.shape[0], oi.shape[1])], 0)
        r = adapter(h.float(), o.unsqueeze(0).to(a.device).float())
        v = r[1] if isinstance(r, (tuple, list)) else r
        if isinstance(v, (tuple, list)):
            v = v[0]
        if v.dim() == 3:
            v = v.mean(dim=1)
        return v[0].float().cpu().numpy()

    rows = [r for r in csv.DictReader(open(a.features_csv))
            if "_s1clean" not in (r.get("filename") or "")]
    np.random.default_rng(0).shuffle(rows)
    clean_dir = a.audio_dir.rstrip("/") + "-s1clean"
    ovl_col = "overlap_segments_vad" if (rows and rows[0].get("overlap_segments_vad")) else "overlap_segments"

    out = []
    for r in rows:
        if len(out) >= a.n:
            break
        fn = r["filename"]
        p_mix, p_cln = os.path.join(a.audio_dir, fn), os.path.join(clean_dir, fn[:-4] + "_s1clean.wav")
        if not (os.path.exists(p_mix) and os.path.exists(p_cln)):
            continue
        mix, _ = sf.read(p_mix)
        s1c, _ = sf.read(p_cln)
        n = min(len(mix), len(s1c))
        mix, s1c = mix[:n], s1c[:n]
        s2 = mix - s1c
        win0 = parse_overlap_windows_samples(r.get(ovl_col) or "", SR)
        if not win0:
            continue

        rec = {"filename": fn, "doses": {}}
        oi0 = None
        for al in alphas:
            edited = s1c + (1.0 - al) * s2
            win = s2_active_mask(s2, win0, al)          # overlap mask UPDATED per dose
            T = n // HOP
            oi = torch.zeros(T, 4)
            for (s, e) in win:
                oi[max(0, int(s * SR) // HOP): min(T, int(e * SR) // HOP), 0] = 1.0
            # CONTROL (added 2026-08-11): --hold_input keeps the overlap channel at its
            # alpha=0 state, and --zero_input zeroes it. Rebuilding the channel per dose
            # feeds the model a CHANGED INPUT alongside the changed audio, so any response
            # is ambiguous between the two -- and for overlap_ratio it is simply the
            # rho-0.9999 input echo. Without this control the dose result is not causal
            # evidence about AUDIO.
            if oi0 is None:                      # must be captured BEFORE it is read
                oi0 = oi.clone()
            oi_use = oi0 if a.hold_input else (torch.zeros_like(oi) if a.zero_input else oi)
            pred = predict(edited, oi_use)
            m, sd = f0_stats(edited, win)
            rec["doses"][str(al)] = {
                "pred": {short[i]: float(pred[i]) for i in range(len(short))},
                "gt_f0_mean": None if not np.isfinite(m) else m,
                "gt_f0_sd": None if not np.isfinite(sd) else sd,
                "gt_srmr": srmr_of(edited),
                "ovl_frac": float(sum(e - s for s, e in win) * SR / max(n, 1)),
            }
        out.append(rec)
        if len(out) % 25 == 0:
            print(f"  {len(out)}/{a.n}", flush=True)

    json.dump(out, open(a.out, "w"))
    print(f"wrote {a.out}  n={len(out)}  alphas={alphas}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
