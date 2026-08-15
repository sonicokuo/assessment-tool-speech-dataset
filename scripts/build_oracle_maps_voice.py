#!/usr/bin/env python3
"""build_oracle_maps_voice.py — oracle maps for the VALUE-DEPENDENT features, plus a
boundary-mass variant for pause_count.

THE PREDICTION BEING TESTED
The trivial-null panel split our features cleanly, and the split is structural:

  SUPPORT-defined references (WHICH frames count) are trivially predictable, because
  energy and voicing are exactly what predict support:
      pause_count  -energy 0.576, voicing 0.676  vs ceiling 0.917  -> model 0.524 LOSES
      f0_mean      voicing 0.463                  vs ceiling 0.719  -> model 0.055 LOSES

  VALUE-dependent references (HOW MUCH each frame deviates) are NOT:
      f0_sd        all nulls |0.066|              vs ceiling 0.743  -> the only clean row

`hnr`, `jitter`, `shimmer` are all VALUE-dependent means, so they should behave like
f0_sd rather than like pause_count. That is a specific, falsifiable prediction about the
FEATURES rather than about the metric.

EXACT AGGREGATIONS, read from source (feature_extractor_mix.py):
  hnr      : to_harmonicity_cc, mean over frames with value != -200 (Praat's unvoiced
             sentinel). A MASKED MEAN, so the reweighting influence is the SIGNED
             deviation map  phi_t = m_t (h_t - v) / sum(m).
  jitter   : To PointProcess (periodic, cc) 75-500 Hz, then 'Get jitter (local)' =
             mean |T_i - T_{i+1}| / mean(T). A mean over PERIODS -> per-period deviation,
             aggregated to the 100 Hz frame grid.
  shimmer  : same point process, 'Get shimmer (local)' over amplitude ratios.

  pause_count BOUNDARY variant: a count's influence is concentrated at THRESHOLD
             CROSSINGS -- frames whose perturbation changes the count -- not spread
             uniformly over quiet interiors. Mass goes at interval EDGES, weighted by
             proximity to the 0.3 s minimum-duration rule (a pause near threshold is
             pivotal; a 2 s pause is not).
"""
from __future__ import annotations

import argparse
import csv
import os

import numpy as np

TIME_STEP = 0.01
MIN_PAUSE, SIL_DB, MIN_SOUND = 0.3, -25.0, 0.1
EDGE_S = 0.05          # 50 ms at each boundary carries the count's influence


def hnr_map(wav):
    """Signed deviation map for a masked mean over harmonicity frames."""
    import parselmouth
    snd = parselmouth.Sound(wav)
    h = snd.to_harmonicity_cc(time_step=TIME_STEP, minimum_pitch=75.0)
    vals = np.asarray(h.values[0], dtype=float)
    keep = vals != -200.0                      # Praat's unvoiced sentinel, matched exactly
    if keep.sum() < 3:
        return None, float("nan")
    v = float(vals[keep].mean())
    phi = np.zeros(vals.shape, dtype=float)
    n = int(keep.sum())
    phi[keep] = (vals[keep] - v) / n           # reweighting influence, zero-sum on support
    return phi, v


def period_maps(wav, n_frames):
    """Per-period jitter/shimmer deviations, aggregated to the frame grid."""
    import parselmouth
    from parselmouth.praat import call
    snd = parselmouth.Sound(wav)
    pp = call(snd, "To PointProcess (periodic, cc)", 75.0, 500.0)
    npt = int(call(pp, "Get number of points"))
    if npt < 6:
        return None, None, float("nan"), float("nan")
    t = np.array([call(pp, "Get time from index", i) for i in range(1, npt + 1)], dtype=float)
    T = np.diff(t)                              # period lengths
    ok = (T > 0.0001) & (T < 0.02)              # Praat's period floor/ceiling
    if ok.sum() < 4:
        return None, None, float("nan"), float("nan")
    dT = np.abs(np.diff(T))                     # |T_i - T_{i+1}| -> local jitter terms
    mT = float(T[ok].mean())
    jit = float(dT.mean() / mT) if mT > 0 else float("nan")
    # amplitude at each pulse -> shimmer terms
    x = snd.values[0]
    sr = int(round(1.0 / snd.time_step)) if snd.time_step else 16000
    idx = np.clip((t * sr).astype(int), 0, len(x) - 1)
    A = np.abs(x[idx])
    dA = np.abs(np.diff(A))
    mA = float(A.mean()) if A.size else 0.0
    shm = float(dA.mean() / mA) if mA > 0 else float("nan")

    def to_frames(term_times, terms, mean_val):
        phi = np.zeros(n_frames, dtype=float)
        if terms.size == 0 or not np.isfinite(mean_val) or mean_val <= 0:
            return phi
        dev = (terms / mean_val - terms.mean() / mean_val) / max(1, terms.size)
        for tt, d in zip(term_times, dev):
            j = int(round(tt / TIME_STEP))
            if 0 <= j < n_frames:
                phi[j] += d
        return phi

    return (to_frames(t[1:-1], dT, dT.mean() if dT.size else 0.0),
            to_frames(t[1:], dA, dA.mean() if dA.size else 0.0), jit, shm)


def pause_boundary_map(wav, n_frames):
    """Count influence at THRESHOLD CROSSINGS, weighted by proximity to the 0.3 s rule."""
    import parselmouth
    from parselmouth.praat import call
    snd = parselmouth.Sound(wav)
    inten = snd.to_intensity(minimum_pitch=50.0)
    tg = call(inten, "To TextGrid (silences)", SIL_DB, MIN_PAUSE, MIN_SOUND, "silent", "sounding")
    n = int(call(tg, "Get number of intervals", 1))
    phi = np.zeros(n_frames, dtype=float)
    tot = 0.0
    for i in range(1, n + 1):
        if call(tg, "Get label of interval", 1, i) != "silent":
            continue
        t0 = float(call(tg, "Get start time of interval", 1, i))
        t1 = float(call(tg, "Get end time of interval", 1, i))
        dur = t1 - t0
        if dur < MIN_PAUSE:
            continue
        # a pause NEAR the threshold is pivotal; a long one is not
        w = float(np.exp(-(dur - MIN_PAUSE) / MIN_PAUSE))
        for tt in (t0, t1):
            a = max(0, int((tt - EDGE_S) / TIME_STEP))
            b = min(n_frames, int((tt + EDGE_S) / TIME_STEP))
            if b > a:
                phi[a:b] += w / (b - a)
                tot += w
    if tot > 0:
        phi /= phi.sum() or 1.0
    return phi


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features_csv", required=True)
    ap.add_argument("--clean_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    rows = list(csv.DictReader(open(a.features_csv)))
    if a.limit:
        rows = rows[: a.limit]
    M = {k: {} for k in ("hnr", "jitter", "shimmer", "pause_count_boundary")}
    names, err_h, n_err = [], [], 0

    for i, r in enumerate(rows):
        fn = r.get("filename") or ""
        stem = fn[:-4] if fn.endswith(".wav") else fn
        base = stem if stem.endswith("_s1clean") else stem + "_s1clean"
        wav = os.path.join(a.clean_dir, base + ".wav")
        if not os.path.exists(wav):
            n_err += 1
            continue
        try:
            ph, hv = hnr_map(wav)
            if ph is None:
                continue
            nf = ph.size
            pj, ps, jv, sv = period_maps(wav, nf)
            pb = pause_boundary_map(wav, nf)
        except Exception as e:                              # noqa: BLE001
            n_err += 1
            if n_err <= 3:
                print(f"[warn] {stem}: {e}", flush=True)
            continue
        # EXACTNESS: the deviation map must be zero-sum on its support
        err_h.append(abs(float(ph.sum())))
        names.append(stem)
        M["hnr"][stem] = ph.astype(np.float32)
        if pj is not None:
            M["jitter"][stem] = pj.astype(np.float32)
            M["shimmer"][stem] = ps.astype(np.float32)
        M["pause_count_boundary"][stem] = pb.astype(np.float32)
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(rows)}", flush=True)

    ok = bool(err_h) and max(err_h) < 1e-9
    print(f"\n=== EXACTNESS ===\n  hnr deviation map zero-sum: max |sum phi| = "
          f"{(max(err_h) if err_h else float('nan')):.3e}  {'PASS' if ok else 'FAIL'}")
    print("  jitter/shimmer: per-period deviations aggregated to frames — APPROX at frame")
    print("  granularity (periods are 5-10 ms, below the grid), labelled as such")
    print("  pause_count_boundary: threshold-crossing variant, APPROX by construction")
    np.savez_compressed(a.out, names=np.array(names),
                        **{f"{k}/{s}": v for k, d in M.items() for s, v in d.items()})
    print(f"\nwrote {a.out}  clips={len(names)}  errors={n_err}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
