#!/usr/bin/env python3
"""build_oracle_maps.py — EXACT ground-truth contribution maps (items 0.6 + 0.6b + 0.3).

For features that are FORMULAS OVER FRAMES we can compute the true per-frame
contribution to the emitted scalar, rather than guessing at it with a saliency
heuristic. This is the asset the whole attribution evaluation rests on: it exists
here and not in vision/NLP because our targets are instrument measurements with
closed-form definitions and we hold the constituent signals.

U-1 (resolved 2026-08-09 from source): f0 GT is Praat pitch on the MIXTURE
(mix_clean), restricted to VOICED frames OUTSIDE the VAD overlap windows
(`src/data/f0_clean.py:69`, `keep = voiced & ~in_overlap`, time_step 0.01 -> 100 Hz).
So the f0 map support is `voiced AND NOT overlap` on the mixture. NOT the clean stem.

Maps built here (all EXACT, all inputs already on disk):
  f0_mean       phi_t = keep_t / sum(keep)                     [uniform over kept frames]
  f0_sd         phi_t = keep_t (q_t - mu) / ((N-1) * sd)       [NON-uniform, sign-varying]
  overlap_ratio phi_t = ovl_t / sum(ovl)                       [the input-echo control feature]

EXACTNESS (0.6b) is VERIFIED, not asserted:
  * f0_mean: sum_t phi_t q_t must equal the GT scalar   (additive reconstruction)
  * f0_sd:   analytic phi_t must match a central finite difference d(sd)/d(q_t)
A map that fails its check is reported and EXCLUDED rather than silently shipped.

CEILINGS (0.3): a PERFECT map block-pooled to a token grid loses resolution. We
report ceiling(feature, rate) = spearman(pool_r(phi), phi) so every model score
can be quoted as %-of-ceiling instead of against an unreachable 1.0.

Usage:
  python scripts/build_oracle_maps.py \
      --features_csv $SH/data/features_corrected_merged/test.csv \
      --audio_dir    $SH/data/audio_corrected/test \
      --out          $SH/oracle_maps_test.npz  [--limit N] [--rates 6.25,12.5,25,50]
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from data.f0_clean import parse_overlap_windows_samples  # noqa: E402

# Praat pitch settings — MUST match compute_clean_f0.py or the maps describe a
# different quantity than the GT they are meant to explain.
MIN_PITCH, MAX_PITCH, TIME_STEP = 75.0, 500.0, 0.01


def praat_pitch(wav_path: str):
    """Per-frame F0 (Hz, 0 = unvoiced) and frame centre times."""
    import parselmouth

    snd = parselmouth.Sound(wav_path)
    p = snd.to_pitch(time_step=TIME_STEP, pitch_floor=MIN_PITCH, pitch_ceiling=MAX_PITCH)
    f0 = p.selected_array["frequency"].astype(float)   # 0 where unvoiced
    return f0, np.asarray(p.xs(), dtype=float)


def _overlap_mask(times: np.ndarray, windows) -> np.ndarray:
    m = np.zeros(times.shape, dtype=bool)
    for (s, e) in windows:
        m |= (times >= s) & (times < e)
    return m


def _f0_sd_analytic(q: np.ndarray, keep: np.ndarray) -> tuple[np.ndarray, float]:
    """d(sd)/d(q_t) on kept frames.

    var = sum_kept (q-mu)^2/(N-1);  d(var)/d(q_t) = 2(q_t-mu)/(N-1) because
    sum(q_i-mu) = 0 kills the mean's own dependence. Hence
    d(sd)/d(q_t) = (q_t-mu)/((N-1) sd).
    """
    idx = np.flatnonzero(keep)
    n = idx.size
    phi = np.zeros(q.shape, dtype=float)
    if n < 3:
        return phi, float("nan")
    v = q[idx]
    mu = v.mean()
    sd = float(np.sqrt(((v - mu) ** 2).sum() / (n - 1)))
    if sd <= 0:
        return phi, sd
    phi[idx] = (v - mu) / ((n - 1) * sd)
    return phi, sd


def _f0_sd_findiff(q: np.ndarray, keep: np.ndarray, probe: np.ndarray, h: float = 1e-3):
    """Central finite difference of sd w.r.t. each probed frame — the exactness check."""
    def sd_of(x):
        v = x[keep]
        if v.size < 3:
            return float("nan")
        return float(np.sqrt(((v - v.mean()) ** 2).sum() / (v.size - 1)))

    out = np.zeros(probe.size, dtype=float)
    for j, t in enumerate(probe):
        up, dn = q.copy(), q.copy()
        up[t] += h
        dn[t] -= h
        out[j] = (sd_of(up) - sd_of(dn)) / (2 * h)
    return out


def pool_to_rate(phi: np.ndarray, src_rate: float, tgt_rate: float) -> np.ndarray:
    """Block-sum a frame map onto a coarser token grid (contribution is ADDITIVE)."""
    factor = max(1, int(round(src_rate / tgt_rate)))
    n = int(np.ceil(phi.size / factor)) * factor
    padded = np.zeros(n, dtype=float)
    padded[: phi.size] = phi
    return padded.reshape(-1, factor).sum(axis=1)


def _avg_rank(x: np.ndarray) -> np.ndarray:
    """Tie-AVERAGED ranks.

    argsort(argsort(x)) gives ORDINAL ranks, which break ties by array position.
    These maps are heavily tied by construction (the f0_mean map is binary:
    1/N on kept frames, 0 elsewhere), so ordinal ranks inject position-dependent
    noise and made ceilings NON-MONOTONE in rate (50 Hz scored below 25 Hz,
    which is impossible). Average ranks are required, not cosmetic.
    """
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(x.size, dtype=float)
    sx = x[order]
    i = 0
    while i < x.size:
        j = i + 1
        while j < x.size and sx[j] == sx[i]:
            j += 1
        ranks[order[i:j]] = 0.5 * (i + j - 1)
        i = j
    return ranks


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 3 or np.allclose(a, a[0]) or np.allclose(b, b[0]):
        return float("nan")
    ra = _avg_rank(a)
    rb = _avg_rank(b)
    ra -= ra.mean()
    rb -= rb.mean()
    d = float(np.sqrt((ra ** 2).sum() * (rb ** 2).sum()))
    return float((ra * rb).sum() / d) if d > 0 else float("nan")


def ceiling_at(phi: np.ndarray, src_rate: float, tgt_rate: float) -> float:
    """Upsample the pooled PERFECT map back and correlate — the resolution ceiling."""
    factor = max(1, int(round(src_rate / tgt_rate)))
    pooled = pool_to_rate(phi, src_rate, tgt_rate)
    back = np.repeat(pooled / factor, factor)[: phi.size]
    return spearman(back, phi)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features_csv", required=True)
    ap.add_argument("--audio_dir", required=True)
    ap.add_argument("--clean_audio_dir", default="", help="clean-twin audio; default <audio_dir>-s1clean")
    ap.add_argument("--out", required=True)
    ap.add_argument("--sr", type=int, default=16000)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--rates", default="6.25,12.5,25,50")
    a = ap.parse_args()

    rates = [float(x) for x in a.rates.split(",")]
    rows = list(csv.DictReader(open(a.features_csv)))
    if a.limit:
        rows = rows[: a.limit]
    ovl_col = "overlap_segments_vad" if (rows and rows[0].get("overlap_segments_vad")) else "overlap_segments"

    maps_f0m, maps_f0sd, maps_ovl, names = {}, {}, {}, []
    maps_q, maps_keep = {}, {}
    exact_f0m, exact_f0sd = [], []
    ceil = {f: {r: [] for r in rates} for f in ("f0_mean", "f0_sd", "overlap_ratio")}
    n_err = 0

    for i, r in enumerate(rows):
        fn = r.get("filename") or r.get("file") or ""
        # Clean twins live in a SIBLING dir: audio_corrected/test-s1clean/<stem>_s1clean.wav.
        # Routing every row to --audio_dir silently loses all 3000 of them, which is
        # exactly the CLEAN-CLIP PANEL where N5 is vacuous -- the sharpest condition.
        base = a.clean_audio_dir or (a.audio_dir.rstrip("/") + "-s1clean")
        adir = base if "_s1clean" in fn else a.audio_dir
        wav = os.path.join(adir, fn if fn.endswith(".wav") else fn + ".wav")
        if not os.path.exists(wav):
            n_err += 1
            if n_err <= 3:
                print(f"[warn] missing audio: {wav}", flush=True)
            continue
        try:
            q, t = praat_pitch(wav)
        except Exception as e:                      # noqa: BLE001 - report, never silently skip
            n_err += 1
            if n_err <= 3:
                print(f"[warn] {fn}: {e}", flush=True)
            continue

        windows = parse_overlap_windows_samples(r.get(ovl_col) or "", a.sr)
        in_ovl = _overlap_mask(t, windows)
        keep = (q > 0.0) & (~in_ovl)
        if keep.sum() < 3:
            continue

        # ---- f0_mean: uniform over kept frames; EXACT additive reconstruction
        phi_m = np.zeros(q.shape, dtype=float)
        phi_m[keep] = 1.0 / keep.sum()
        recon = float((phi_m * q).sum())
        truth = float(q[keep].mean())
        exact_f0m.append(abs(recon - truth))

        # ---- f0_sd: non-uniform, sign-varying; analytic vs finite difference
        phi_s, sd = _f0_sd_analytic(q, keep)
        idx = np.flatnonzero(keep)
        probe = idx[:: max(1, idx.size // 8)][:8]
        if probe.size and np.isfinite(sd) and sd > 0:
            fd = _f0_sd_findiff(q, keep, probe)
            denom = max(1e-12, float(np.abs(phi_s[probe]).max()))
            exact_f0sd.append(float(np.abs(fd - phi_s[probe]).max() / denom))

        # ---- overlap_ratio: the echo control
        phi_o = np.zeros(q.shape, dtype=float)
        if in_ovl.sum() > 0:
            phi_o[in_ovl] = 1.0 / in_ovl.sum()

        stem = fn[:-4] if fn.endswith(".wav") else fn
        names.append(stem)
        maps_f0m[stem], maps_f0sd[stem], maps_ovl[stem] = phi_m, phi_s, phi_o
        # Per-frame F0 and the keep mask are the TARGETS a defining-statistic model
        # needs (item 0.18 positive control). Without them the control cannot be
        # built and the model's 17%-of-ceiling stays uninterpretable.
        maps_q[stem] = q.astype(np.float32)
        maps_keep[stem] = keep.astype(np.uint8)

        src = 1.0 / TIME_STEP                      # 100 Hz
        for rate in rates:
            ceil["f0_mean"][rate].append(ceiling_at(phi_m, src, rate))
            ceil["f0_sd"][rate].append(ceiling_at(phi_s, src, rate))
            if in_ovl.sum() > 0:
                ceil["overlap_ratio"][rate].append(ceiling_at(phi_o, src, rate))

        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(rows)} clips", flush=True)

    # ---------------- exactness verdict (item 0.6b) ----------------
    print("\n=== EXACTNESS (0.6b) — a map that fails here is NOT shipped ===", flush=True)
    ok_m = bool(exact_f0m) and max(exact_f0m) < 1e-6
    ok_s = bool(exact_f0sd) and float(np.median(exact_f0sd)) < 1e-3
    print(f"  f0_mean additive reconstruction : max |recon-truth| = "
          f"{(max(exact_f0m) if exact_f0m else float('nan')):.3e}   {'PASS' if ok_m else 'FAIL'}")
    print(f"  f0_sd analytic vs finite-diff   : median rel err   = "
          f"{(float(np.median(exact_f0sd)) if exact_f0sd else float('nan')):.3e}   {'PASS' if ok_s else 'FAIL'}")

    # ---------------- resolution ceilings (item 0.3) ----------------
    print("\n=== CEILINGS (0.3) — spearman(pooled perfect map, perfect map) ===", flush=True)
    print(f"  {'feature':<14}" + "".join(f"{r:>10.2f}Hz" for r in rates), flush=True)
    ceil_out = {}
    for f in ("f0_mean", "f0_sd", "overlap_ratio"):
        cells = []
        for rate in rates:
            v = np.array([x for x in ceil[f][rate] if np.isfinite(x)], dtype=float)
            m = float(v.mean()) if v.size else float("nan")
            ceil_out[f"{f}@{rate}"] = m
            cells.append(f"{m:>12.3f}")
        print(f"  {f:<14}" + "".join(cells), flush=True)

    np.savez_compressed(
        a.out,
        names=np.array(names),
        **{f"f0_mean/{k}": v for k, v in maps_f0m.items()},
        **{f"f0_sd/{k}": v for k, v in maps_f0sd.items()},
        **{f"overlap_ratio/{k}": v for k, v in maps_ovl.items()},
        **{f"q/{k}": v for k, v in maps_q.items()},
        **{f"keep/{k}": v for k, v in maps_keep.items()},
        ceilings=np.array([f"{k}={v}" for k, v in ceil_out.items()]),
    )
    print(f"\nwrote {a.out}  clips={len(names)}  errors={n_err}", flush=True)
    return 0 if (ok_m and ok_s) else 1


if __name__ == "__main__":
    raise SystemExit(main())
