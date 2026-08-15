#!/usr/bin/env python3
"""snr_triviality_panel.py — is our SRCC 0.948 on snr actually cheap? (item E1)

THE OBJECTION, stated concretely so we can answer it ourselves:
  "0.948 on a uniformly-sampled, construction-controlled 0-40 dB label is what
   WADA-SNR achieved in 2008 WITHOUT TRAINING, and your attribution map for
   stationary noise reduces to voice-activity detection."

This fits TRAINING-FREE estimators on the identical clips and reports them beside
the model. Every one of these is something a reviewer can implement in an hour:

  P90/P10      10*log10(P90/P10) of the frame-energy envelope. This is the repo's
               OWN RETIRED ground-truth estimator — the single most likely baseline
               a reviewer reaches for, because it is in our git history.
  VAD 2-point  noise floor from low-energy frames, speech level from high-energy
               frames, SNR = 10*log10((S-N)/N). Closed form, no training.
  WADA-SNR     Kim & Stern, Interspeech 2008. Amplitude-distribution based,
               training-free, and MOST accurate exactly in our regime (broadband,
               quasi-stationary noise, which WHAM is).
  RMS only     a single number per clip. If THIS correlates, the label leaks
               through absolute level and the task is not measurement at all.

Interpretation, decided in advance:
  * if any training-free estimator reaches ~0.9+, the snr accuracy claim is not
    evidence the model measures anything, and snr must move to a SUPPORTING role;
  * if `RMS only` correlates strongly, the pipeline's loudness handling leaks the
    label and snr should be dropped from the headline entirely.
"""
from __future__ import annotations

import argparse
import csv
import os

import numpy as np
import soundfile as sf

FRAME = 320          # 20 ms at 16 kHz


def frame_energy(x: np.ndarray) -> np.ndarray:
    n = len(x) // FRAME
    if n < 4:
        return np.array([])
    return (x[: n * FRAME].reshape(n, FRAME) ** 2).mean(axis=1) + 1e-12


def p90_p10(x: np.ndarray) -> float:
    e = frame_energy(x)
    if e.size < 4:
        return float("nan")
    return float(10 * np.log10(np.percentile(e, 90) / np.percentile(e, 10)))


def vad_two_point(x: np.ndarray) -> float:
    """Energy-THRESHOLD VAD, genuinely distinct from the percentile estimator.

    The first version reused the 10th/90th percentiles, so it was algebraically the
    same statistic as P90/P10 and both returned identical SRCC -- not an independent
    baseline. Here the speech/noise partition comes from a threshold relative to the
    frame-energy maximum (the standard -25 dB-below-peak rule Praat uses), so the
    partition is data-driven rather than a fixed quantile.
    """
    e = frame_energy(x)
    if e.size < 8:
        return float("nan")
    edb = 10 * np.log10(e)
    thr = edb.max() - 25.0
    sp, ns = e[edb > thr], e[edb <= thr]
    if sp.size < 3 or ns.size < 3:
        return float("nan")
    noise = float(ns.mean())
    speech = float(sp.mean())
    if noise <= 0 or speech <= noise:
        return float("nan")
    return float(10 * np.log10((speech - noise) / noise))


def wada_snr(x: np.ndarray) -> float:
    """WADA statistic (Kim & Stern, Interspeech 2008), RAW and uncalibrated.

    WADA maps `log(mean|x|) - mean(log|x|)` through a monotone table to dB. We report
    SPEARMAN, which is invariant to any monotone transform, so the calibration table
    is unnecessary -- and my first attempt returned nan for every clip because the
    statistic fell outside my transcribed table. Using the raw statistic removes a
    whole class of transcription error and measures exactly what SRCC would see.
    Returns the NEGATED statistic so it increases with SNR (the statistic itself
    decreases as noise rises).
    """
    a = np.abs(x[np.abs(x) > 1e-10])
    if a.size < 1000:
        return float("nan")
    v = float(np.log(a.mean()) - np.mean(np.log(a)))
    return -v if np.isfinite(v) else float("nan")


def _rank(x):
    o = np.argsort(x, kind="mergesort")
    r = np.empty(x.size, float)
    s = x[o]
    i = 0
    while i < x.size:
        j = i + 1
        while j < x.size and s[j] == s[i]:
            j += 1
        r[o[i:j]] = 0.5 * (i + j - 1)
        i = j
    return r


def spearman(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    a, b = a[m], b[m]
    if a.size < 5:
        return float("nan"), 0
    ra, rb = _rank(a) - _rank(a).mean(), _rank(b) - _rank(b).mean()
    d = float(np.sqrt((ra ** 2).sum() * (rb ** 2).sum()))
    return (float((ra * rb).sum() / d) if d > 0 else float("nan")), a.size


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features_csv", required=True)
    ap.add_argument("--audio_dir", required=True)
    ap.add_argument("--n", type=int, default=1500)
    a = ap.parse_args()

    rows = [r for r in csv.DictReader(open(a.features_csv))
            if "_s1clean" not in (r.get("filename") or "")]
    np.random.default_rng(0).shuffle(rows)

    gt, est = [], {k: [] for k in ("P90/P10", "VAD 2-point", "WADA-SNR", "RMS only")}
    for r in rows:
        if len(gt) >= a.n:
            break
        p = os.path.join(a.audio_dir, r["filename"])
        if not os.path.exists(p):
            continue
        try:
            g = float(r["snr_db"])
        except (TypeError, ValueError, KeyError):
            continue
        x, _ = sf.read(p)
        if x.ndim > 1:
            x = x.mean(axis=1)
        gt.append(g)
        est["P90/P10"].append(p90_p10(x))
        est["VAD 2-point"].append(vad_two_point(x))
        est["WADA-SNR"].append(wada_snr(x))
        est["RMS only"].append(float(10 * np.log10(np.sqrt((x ** 2).mean()) + 1e-12)))
        if len(gt) % 250 == 0:
            print(f"  {len(gt)}/{a.n}", flush=True)

    g = np.array(gt, float)
    print(f"\n=== TRAINING-FREE SNR ESTIMATORS vs the construction label (n={g.size}) ===")
    print("    our model's aux head: SRCC 0.948   <- the number under test")
    for k, v in est.items():
        rho, n = spearman(np.array(v, float), g)
        note = ""
        if np.isfinite(rho) and abs(rho) >= 0.90:
            note = "  <-- MATCHES THE MODEL: accuracy claim is not evidence of measurement"
        elif np.isfinite(rho) and abs(rho) >= 0.70:
            note = "  <-- strong; must be reported as a baseline row"
        print(f"    {k:<14} SRCC {rho:+.4f}   n={n}{note}")
    print("\n    'RMS only' correlating strongly would mean the label leaks through")
    print("    absolute level and snr should leave the headline entirely.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
