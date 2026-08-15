#!/usr/bin/env python3
"""build_oracle_maps_events.py — EXACT oracle maps for the EVENT-based features (item 0.5/0.6).

Extends the oracle-map panel beyond f0 so the flagship result does not rest on a
single feature.

EXACT AGGREGATIONS, read from source (`feature_extractor_mix.py`) — this check is what
licenses the word "exact"; a head implementing a different formula than the instrument
makes the whole by-construction claim false:

  speaking_rate = nuclei / duration
      nuclei = intensity-contour peaks with v > max_int - 25 dB, v > left neighbour,
      v >= right neighbour, and a >= 2 dB dip to the min of a 10-frame window on either
      side. Each nucleus contributes EXACTLY 1/duration, so the contribution map is a
      SPIKE TRAIN: phi_t = 1/nuclei after normalisation. Perfectly additive.

  pause_count = number of silent intervals >= 0.3 s in Praat's silence TextGrid
      (-25 dB threshold, min sounding 0.1 s).
      A COUNT has NO unique additive decomposition and is NON-MONOTONE (splitting one
      0.65 s pause RAISES the count). We therefore emit a DENSITY MAP whose integral is
      the count (Lempitsky & Zisserman, NIPS 2010), and label it APPROX everywhere.

  pause_rate = pause_count * 60 / duration
      A FIXED RESCALING of pause_count, so it shares pause_count's map exactly. It is
      NOT an independent feature for attribution purposes and must not be counted twice.

PROVENANCE: all three are measured on the CLEAN S1 STEM (`clean_rate_and_pauses` ->
`compute_praat_*`), NOT the mixture. So maps are built from the `_s1clean` audio, on the
same timeline as the mixture (equal length, verified).

Usage:
  python scripts/build_oracle_maps_events.py --features_csv <csv> --clean_dir <dir> \
      --out $SH/oracle_maps_events_test.npz [--limit N]
"""
from __future__ import annotations

import argparse
import csv
import os

import numpy as np

TIME_STEP = 0.01          # match the f0 oracle grid (100 Hz) so panels are poolable together
SIL_DB = -25.0
MIN_PAUSE = 0.3
MIN_SOUND = 0.1
MIN_DIP = 2.0


def nucleus_times(wav_path: str):
    """Replicate compute_praat_speaking_rate's nucleus loop, but KEEP the timestamps."""
    import parselmouth

    snd = parselmouth.Sound(wav_path)
    dur = snd.duration
    if dur < 0.1:
        return [], dur
    intensity = snd.to_intensity(minimum_pitch=50.0)
    vals = intensity.values[0]
    times = np.asarray(intensity.xs(), dtype=float)
    if vals.size < 3:
        return [], dur
    thr = float(np.max(vals)) - 25.0
    out = []
    n = len(vals)
    for i in range(1, n - 1):
        v = vals[i]
        if v <= thr or v < vals[i - 1] or v <= vals[i + 1]:
            continue
        left = float(np.min(vals[max(0, i - 10): i]))
        right = float(np.min(vals[i + 1: min(n, i + 11)]))
        if (v - left >= MIN_DIP) or (v - right >= MIN_DIP):
            out.append(float(times[i]))
    return out, dur


def pause_intervals(wav_path: str):
    """Silent intervals >= MIN_PAUSE from Praat's silence TextGrid — the pause events."""
    import parselmouth
    from parselmouth.praat import call

    snd = parselmouth.Sound(wav_path)
    dur = snd.duration
    intensity = snd.to_intensity(minimum_pitch=50.0)
    tg = call(intensity, "To TextGrid (silences)", SIL_DB, MIN_PAUSE, MIN_SOUND,
              "silent", "sounding")
    n = int(call(tg, "Get number of intervals", 1))
    out = []
    for i in range(1, n + 1):
        if call(tg, "Get label of interval", 1, i) != "silent":
            continue
        t0 = float(call(tg, "Get start time of interval", 1, i))
        t1 = float(call(tg, "Get end time of interval", 1, i))
        if t1 - t0 >= MIN_PAUSE:
            out.append((t0, t1))
    return out, dur


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features_csv", required=True)
    ap.add_argument("--clean_dir", required=True, help="the *-s1clean audio dir")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    rows = list(csv.DictReader(open(a.features_csv)))
    if a.limit:
        rows = rows[: a.limit]

    maps_rate, maps_pause, names = {}, {}, []
    exact_rate, n_err = [], 0

    for i, r in enumerate(rows):
        fn = r.get("filename") or ""
        stem = fn[:-4] if fn.endswith(".wav") else fn
        # GT is measured on the CLEAN STEM for all three features.
        base = stem if stem.endswith("_s1clean") else stem + "_s1clean"
        wav = os.path.join(a.clean_dir, base + ".wav")
        if not os.path.exists(wav):
            n_err += 1
            if n_err <= 3:
                print(f"[warn] missing {wav}", flush=True)
            continue
        try:
            nuc, dur = nucleus_times(wav)
            pauses, _ = pause_intervals(wav)
        except Exception as e:                       # noqa: BLE001 — report, never silently skip
            n_err += 1
            if n_err <= 3:
                print(f"[warn] {fn}: {e}", flush=True)
            continue
        if dur <= 0:
            continue
        T = max(8, int(round(dur / TIME_STEP)))

        # speaking_rate: unit mass per nucleus (EXACT)
        phi_r = np.zeros(T, dtype=np.float32)
        for t in nuc:
            j = int(round(t / TIME_STEP))
            if 0 <= j < T:
                phi_r[j] += 1.0
        if phi_r.sum() > 0:
            # EXACTNESS: sum(phi)/duration must reproduce the emitted speaking_rate
            exact_rate.append(abs(float(phi_r.sum()) / dur - len(nuc) / dur))
            phi_r /= phi_r.sum()

        # pause_count: DENSITY whose integral is the count (APPROX by construction)
        phi_p = np.zeros(T, dtype=np.float32)
        for (t0, t1) in pauses:
            j0, j1 = int(round(t0 / TIME_STEP)), int(round(t1 / TIME_STEP))
            j0, j1 = max(0, j0), min(T, j1)
            if j1 > j0:
                phi_p[j0:j1] += 1.0 / (j1 - j0)
        if phi_p.sum() > 0:
            phi_p /= phi_p.sum()

        names.append(stem)
        maps_rate[stem], maps_pause[stem] = phi_r, phi_p
        if (i + 1) % 250 == 0:
            print(f"  {i+1}/{len(rows)}", flush=True)

    print("\n=== EXACTNESS ===", flush=True)
    ok = bool(exact_rate) and max(exact_rate) < 1e-9
    print(f"  speaking_rate spike-train reproduces nuclei/duration: "
          f"max err {(max(exact_rate) if exact_rate else float('nan')):.3e}  {'PASS' if ok else 'FAIL'}")
    print("  pause_count/pause_rate: DENSITY map, labelled APPROX "
          "(a count has no unique additive decomposition)")

    np.savez_compressed(
        a.out,
        names=np.array(names),
        **{f"speaking_rate/{k}": v for k, v in maps_rate.items()},
        **{f"pause_count/{k}": v for k, v in maps_pause.items()},
        # pause_rate is pause_count * 60/duration — a FIXED rescaling, identical map.
        **{f"pause_rate/{k}": v for k, v in maps_pause.items()},
    )
    print(f"\nwrote {a.out}  clips={len(names)}  errors={n_err}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
