#!/usr/bin/env python3
"""causal_remix.py — in-distribution causal test, protocol P1+P2 (2026-08-10 rewrite).

WHAT WAS WRONG WITH THE OLD TEST (all four fixed here)
1. ENDPOINT: substitution CHANGES the ground truth (removing the interferer converts
   overlapped frames into kept frames), so scoring against the ORIGINAL clip's GT is
   invalid -- a PERFECT model tracks the NEW GT. We recompute GT on the edited audio.
2. FEATURE CHOICE: it was run on f0, where the model beats a constant predictor by 4%,
   so |dGT|/sigma_resid is far inside the noise and win-rate ~ 0.5 EVEN FOR A FAITHFUL
   MODEL. We test the features where the model actually has signal.
3. POWER: binary win-rate at n=120 has CI half-width ~0.09 (0.55 vs 0.50 needs n~780).
   We use a PAIRED CONTINUOUS endpoint.
4. SPLICING: waveform splices are detectable at 20 ms (cf. the PartialSpoof literature).
   The mixture is a LINEAR SUM, so we apply a smooth GAIN ENVELOPE to the s2 stem and
   re-sum -- continuous everywhere, exactly on-manifold.

THE DESIGN — one intervention, TWO opposite predictions
GT provenance decides what a faithful model must do (verified from source):
  * speaking_rate / pause_* / jitter / shimmer / hnr / srmr are measured on the CLEAN
    S1 STEM, so attenuating s2 CANNOT change their GT
        -> a faithful model must be INVARIANT.  Drift = reading the interferer.
  * f0_mean / f0_sd are measured on the MIXTURE outside overlap, so attenuating s2
    DOES change their GT
        -> a faithful model must be SENSITIVE, and track dGT.
  * snr is the sampled construction parameter and never moves.

HIGH vs LOW is the paired control: attenuating s2 inside the most-overlapped window
should matter; doing it inside the least-overlapped window removes almost nothing and
is the matched null. Interventions are >= 1 s because the measured ERF is ~1 s and
shorter edits blur into sub-noise.

Usage:
  python scripts/causal_remix.py --checkpoint <ckpt> --n 200 --seconds 1.5 --out remix.json
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
HOP = 320                       # WavLM-Large: one frame per 20 ms
RAMP_MS = 30                    # cosine ramp; avoids any detectable discontinuity

# GT provenance -> what a faithful model must do under an s2 edit.
# GT PROVENANCE, verified at source. srmr was MISCLASSIFIED here: regenerate_corrected.py:113
# recomputes it on `out_wav`, the DEGRADED MIXTURE ("SRMR on the DEGRADED audio ... = GT for
# srmr"), NOT on the clean stem. So attenuating s2 MOVES its GT and it belongs with f0.
INVARIANT = ("speaking_rate", "pause_count", "pause_rate", "jitter", "shimmer", "hnr")
SENSITIVE = ("f0_mean", "f0_sd", "srmr")
FIXED = ("snr", "overlap_ratio")


def gain_envelope(n: int, a: int, b: int) -> np.ndarray:
    """1 outside [a,b], 0 inside, with cosine ramps. Applied to the s2 STEM, never spliced."""
    g = np.ones(n, dtype=np.float64)
    r = max(1, int(RAMP_MS * SR / 1000))
    a, b = max(0, a), min(n, b)
    if b <= a:
        return g
    g[a:b] = 0.0
    lo0, lo1 = max(0, a - r), a
    if lo1 > lo0:
        g[lo0:lo1] = 0.5 * (1 + np.cos(np.linspace(0, np.pi, lo1 - lo0)))
    hi0, hi1 = b, min(n, b + r)
    if hi1 > hi0:
        g[hi0:hi1] = 0.5 * (1 - np.cos(np.linspace(0, np.pi, hi1 - hi0)))
    return g


def pick_windows(n: int, windows, seconds: float):
    """Most- and least-overlapped windows of the requested length (the paired control)."""
    L = int(seconds * SR)
    if n <= L:
        return (0, n), (0, n)
    ovl = np.zeros(n, dtype=np.float32)
    for (s, e) in windows:
        ovl[max(0, int(s * SR)):min(n, int(e * SR))] = 1.0
    cs = np.concatenate([[0.0], np.cumsum(ovl)])
    starts = np.arange(0, n - L, max(1, L // 4))
    mass = cs[starts + L] - cs[starts]
    return (int(starts[mass.argmax()]), int(starts[mass.argmax()]) + L), \
           (int(starts[mass.argmin()]), int(starts[mass.argmin()]) + L)


def f0_stats(wav: np.ndarray, windows):
    """Recompute the f0 GT the way the instrument does: Praat pitch on THIS audio,
    voiced frames OUTSIDE the overlap windows (src/data/f0_clean.py:69)."""
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--features_csv", required=True)
    ap.add_argument("--audio_dir", required=True)
    ap.add_argument("--clean_dir", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seconds", type=float, default=1.5)   # >= measured ERF (~1 s)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    short = [f[0] if isinstance(f, (tuple, list)) else str(f) for f in SUPERVISED_FEATURES]

    from transformers import WavLMModel
    wavlm = WavLMModel.from_pretrained("microsoft/wavlm-large").to(a.device).eval()

    ck = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    cfg = ck.get("config", {}) or {}
    adapter = build_adapter(
        variant=cfg.get("adapter_variant", "attn-concat"),
        audio_dim=cfg.get("audio_dim", 1024), lm_dim=cfg.get("lm_dim", 4096),
        compression=cfg.get("compression", 8), aux_pool=cfg.get("aux_pool") or "mean",
        reliability_head=bool(cfg.get("reliability_head", False)),
    )
    missing, unexpected = adapter.load_state_dict(
        ck.get("adapter_state_dict") or ck.get("adapter") or {}, strict=False)
    if missing or unexpected:
        print(f"[fatal] state_dict mismatch missing={list(missing)} unexpected={list(unexpected)}")
        return 1
    print(f"[load] reliability_head={bool(cfg.get('reliability_head'))} state_dict=EXACT MATCH", flush=True)
    adapter.eval().to(a.device)

    @torch.no_grad()
    def predict(wav: np.ndarray, ovl_info: torch.Tensor):
        x = torch.from_numpy(wav.astype(np.float32)).unsqueeze(0).to(a.device)
        h = wavlm(x).last_hidden_state                       # (1, T, 1024)
        T = h.shape[1]
        oi = ovl_info[:T] if ovl_info.shape[0] >= T else torch.cat(
            [ovl_info, ovl_info.new_zeros(T - ovl_info.shape[0], ovl_info.shape[1])], 0)
        r = adapter(h.float(), oi.unsqueeze(0).to(a.device).float())
        v = r[1] if isinstance(r, (tuple, list)) else r
        if isinstance(v, (tuple, list)):
            v = v[0]
        if v.dim() == 3:
            v = v.mean(dim=1)
        return v[0].float().cpu().numpy()

    rows = [r for r in csv.DictReader(open(a.features_csv))
            if "_s1clean" not in (r.get("filename") or "")]
    rng = np.random.default_rng(0)
    rng.shuffle(rows)
    clean_dir = a.clean_dir or (a.audio_dir.rstrip("/") + "-s1clean")
    ovl_col = "overlap_segments_vad" if (rows and rows[0].get("overlap_segments_vad")) else "overlap_segments"

    out = []
    for r in rows:
        if len(out) >= a.n:
            break
        fn = r["filename"]
        p_mix = os.path.join(a.audio_dir, fn)
        p_cln = os.path.join(clean_dir, fn[:-4] + "_s1clean.wav")
        if not (os.path.exists(p_mix) and os.path.exists(p_cln)):
            continue
        mix, _ = sf.read(p_mix)
        s1c, _ = sf.read(p_cln)
        n = min(len(mix), len(s1c))
        mix, s1c = mix[:n], s1c[:n]
        s2 = mix - s1c                                    # exact: mix = s1clean + s2
        win = parse_overlap_windows_samples(r.get(ovl_col) or "", SR)
        if not win:
            continue
        (ha, hb), (la, lb) = pick_windows(n, win, a.seconds)

        T = n // HOP
        oi = torch.zeros(T, 4)
        for (s, e) in win:
            oi[max(0, int(s * SR) // HOP): min(T, int(e * SR) // HOP), 0] = 1.0

        base = predict(mix, oi)
        # Record the BASE prediction, not just deltas. The endpoint that matters is
        # d(ERROR) = |yhat' - GT| - |yhat - GT|, and for the clean-stem features GT is
        # FIXED under an s2 edit -- so error change is computable only if the base
        # prediction is stored. |dyhat| alone cannot distinguish "moved toward truth"
        # (the model measures s1 and interference was degrading it) from "moved away"
        # (the model reads the interferer as signal), which are opposite verdicts.
        rec = {"filename": fn, "base": {short[i]: float(base[i]) for i in range(len(short))}}
        for tag, (wa, wb) in (("high", (ha, hb)), ("low", (la, lb))):
            edited = s1c + gain_envelope(n, wa, wb) * s2
            pred = predict(edited, oi)                     # overlap channel held fixed
            rec[f"d_{tag}"] = {short[i]: float(pred[i] - base[i]) for i in range(len(short))}
            m0, s0 = f0_stats(mix, win)
            m1, s1_ = f0_stats(edited, win)
            rec[f"dgt_{tag}"] = {"f0_mean": float(m1 - m0) if np.isfinite(m1 - m0) else None,
                                 "f0_sd": float(s1_ - s0) if np.isfinite(s1_ - s0) else None}
        out.append(rec)
        if len(out) % 25 == 0:
            print(f"  {len(out)}/{a.n}", flush=True)

    json.dump(out, open(a.out, "w"))
    print(f"wrote {a.out}  n={len(out)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
