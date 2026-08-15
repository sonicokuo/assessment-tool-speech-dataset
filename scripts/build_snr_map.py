#!/usr/bin/env python3
"""build_snr_map.py — EXACT oracle contribution map for `snr` (SNR RE-INCLUSION, 2026-08-11).

WHY snr IS IN THE PANEL AFTER ALL
It was excluded on the grounds that "the label is a sampled construction parameter, so
the derivative is identically zero". That differentiates the STORED CSV VALUE instead of
the FUNCTION the label instantiates -- by that logic `f0_mean` is unmappable too, since
its CSV value also does not move when you perturb audio. `regenerate_corrected.py:76-86`
chooses the noise gain so the ACHIEVED ratio equals the sampled value:

    snr_db = 10*log10( sum_t a_t / sum_t b_t )        to ~4e-9 dB

with a_t the per-frame energy of the reverberant mixture and b_t that of the scaled
noise. That is a ratio of frame sums -- the same form we already accept for hnr. (WHAM!
and the DNS Challenge generate GT SNR exactly this way; the whole enhancement literature
treats these labels as functions of the signal.)

THE MAP. Raw partials are FLAT (10/(ln10 * A) and -10/(ln10 * B)), so the structure lives
in the REWEIGHTING influence function, exactly as planned for hnr. Perturbing frame t's
inclusion weight gives

    phi_t = (10/ln10) * ( a_t/A - b_t/B )

ZERO-SUM and SIGN-VARYING -- the same family as the f0_sd map. Scale-invariant, so the
0.98-peak normalisation is irrelevant.

NO REGENERATION. The lineage REPLAYS to within one 16-bit quantisation step (verified:
max|diff| = 3.052e-05 = 2^-15), so a_t and b_t are recomputed IN MEMORY for the clips the
model was actually evaluated on. Nothing is written to the dataset.

Three replay details that silently break everything if missed (all cost me a run):
  * `_load_real_rir` takes channel 0, NOT the channel mean;
  * it PEAK-NORMALISES the RIR (h / max|h|);
  * the RIR list EXCLUDES filenames containing "noise" (325 files, matching the log).

EXACTNESS (what licenses the word):
  1. label reconstruction  |10log10(A/B) - snr_db| < 1e-6 dB
  2. analytic phi vs central finite difference in frame-WEIGHT space
"""
from __future__ import annotations

import argparse
import csv
import glob
import os

import numpy as np
import scipy.signal as sps
import soundfile as sf

SR = 16000
TIME_STEP = 0.01                      # 100 Hz, matching the f0 oracle grid
FRAME = int(SR * TIME_STEP)           # 160 samples
LN10 = np.log(10.0)


def frame_energy(x: np.ndarray, nf: int) -> np.ndarray:
    """Non-overlapping rectangular frames -> a PARTITION, so sum(a_t) == sum(x**2) EXACTLY.

    The partition must COVER the signal. Using nf = len//FRAME silently drops up to 159
    tail samples from A and B, while the instrument's np.mean runs over the FULL array —
    that alone put label reconstruction at 1.0e-02 dB instead of the ~4e-9 the epsilons
    predict. Callers must pass nf = ceil(len/FRAME) and this pads the remainder.
    """
    need = nf * FRAME
    if len(x) < need:
        x = np.concatenate([x, np.zeros(need - len(x))])
    return (x[:need].reshape(nf, FRAME) ** 2).sum(axis=1)


def snr_phi(a: np.ndarray, b: np.ndarray):
    """phi_t = (10/ln10)(a_t/A - b_t/B); returns (phi, snr_db)."""
    A, B = a.sum(), b.sum()
    if A <= 0 or B <= 0:
        return None, float("nan")
    phi = (10.0 / LN10) * (a / A - b / B)
    return phi, float(10.0 * np.log10(A / B))


def findiff(a: np.ndarray, b: np.ndarray, probe: np.ndarray, h: float = 1e-5):
    """d/de of 10log10(sum (1+e*1_t) a / sum (1+e*1_t) b) — the exactness check."""
    out = np.zeros(probe.size)
    for j, t in enumerate(probe):
        wa = a.copy(); wb = b.copy()
        wa[t] *= (1 + h); wb[t] *= (1 + h)
        up = 10 * np.log10(wa.sum() / wb.sum())
        wa = a.copy(); wb = b.copy()
        wa[t] *= (1 - h); wb[t] *= (1 - h)
        dn = 10 * np.log10(wa.sum() / wb.sum())
        out[j] = (up - dn) / (2 * h)
    return out


def _avg_rank(x):
    o = np.argsort(x, kind="mergesort"); r = np.empty(x.size, float); s = x[o]; i = 0
    while i < x.size:
        j = i + 1
        while j < x.size and s[j] == s[i]:
            j += 1
        r[o[i:j]] = 0.5 * (i + j - 1); i = j
    return r


def spearman(u, v):
    n = min(u.size, v.size)
    u, v = u[:n], v[:n]
    if n < 3 or np.allclose(u, u[0]) or np.allclose(v, v[0]):
        return float("nan")
    ru, rv = _avg_rank(u) - _avg_rank(u).mean(), _avg_rank(v) - _avg_rank(v).mean()
    d = float(np.sqrt((ru ** 2).sum() * (rv ** 2).sum()))
    return float((ru * rv).sum() / d) if d > 0 else float("nan")


def ceiling(phi, rate):
    f = max(1, int(round((1.0 / TIME_STEP) / rate)))
    n = (phi.size // f) * f
    if n == 0:
        return float("nan")
    pooled = phi[:n].reshape(-1, f).sum(axis=1)
    return spearman(np.repeat(pooled / f, f), phi[:n])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mix_dir", required=True, help="Libri2Mix <split>/mix_clean (replay source)")
    ap.add_argument("--live_dir", required=True, help="audio_corrected/<split> (for the replay check)")
    ap.add_argument("--wham_dir", required=True)
    ap.add_argument("--rir_glob", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--p_real", type=float, default=0.5)
    ap.add_argument("--snr_lo", type=float, default=0.0)
    ap.add_argument("--snr_hi", type=float, default=40.0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--rates", default="6.25,12.5,25,50,100")
    a = ap.parse_args()

    rates = [float(x) for x in a.rates.split(",")]
    mixes = sorted(glob.glob(os.path.join(a.mix_dir, "*.wav")))
    real_rirs = [f for f in sorted(glob.glob(a.rir_glob))
                 if "noise" not in os.path.basename(f).lower()]
    wham_files = glob.glob(os.path.join(a.wham_dir, "*.wav"))     # UNSORTED, as generated
    print(f"mixes={len(mixes)}  real_rirs={len(real_rirs)}  wham={len(wham_files)}", flush=True)

    maps, names, env = {}, [], {}
    rec_err, fd_err, repl_err, n_sim, n_err = [], [], [], 0, 0
    ceil = {r: [] for r in rates}

    for idx, mp in enumerate(mixes):
        if a.limit and len(names) >= a.limit:
            break
        stem = os.path.basename(mp)[:-4]
        rng = np.random.default_rng(a.seed * 1_000_003 + idx)
        try:
            x, sr = sf.read(mp)
            x = (x.mean(1) if x.ndim > 1 else x).astype(np.float64)
            if not (len(real_rirs) and rng.random() < a.p_real):
                n_sim += 1                       # simulated RIR: needs pyroomacoustics replay
                continue
            hp = real_rirs[rng.integers(len(real_rirs))]
            h, hsr = sf.read(hp)
            if h.ndim > 1:
                h = h[:, 0]                      # channel 0, NOT the mean
            if hsr != sr:
                h = sps.resample_poly(h, sr, hsr)
            h = h.astype(np.float64)
            h = h / (np.max(np.abs(h)) or 1.0)   # PEAK-NORMALISED
            xr = sps.fftconvolve(x, h)[:len(x)]

            snr_db = float(rng.uniform(a.snr_lo, a.snr_hi))
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
            g = np.sqrt(ps / (pn * (10 ** (snr_db / 10.0))))
            noise = g * nz
        except Exception as e:                   # noqa: BLE001 - report, never silently skip
            n_err += 1
            if n_err <= 3:
                print(f"[warn] {stem}: {e}", flush=True)
            continue

        # replay check against the live file (16-bit, so the floor is 2^-15)
        lp = os.path.join(a.live_dir, stem + ".wav")
        if os.path.exists(lp):
            xd = xr + noise
            xd = 0.98 * xd / (np.max(np.abs(xd)) or 1.0)
            live, _ = sf.read(lp)
            m = min(len(xd), len(live))
            repl_err.append(float(np.max(np.abs(xd[:m] - live[:m]))))

        nf = int(np.ceil(len(xr) / FRAME))      # COVER the signal, do not truncate
        if nf < 8:
            continue
        aE, bE = frame_energy(xr, nf), frame_energy(noise, nf)
        phi, recon = snr_phi(aE, bE)
        if phi is None or not np.isfinite(recon):
            continue
        rec_err.append(abs(recon - snr_db))
        idxs = np.arange(nf)[:: max(1, nf // 8)][:8]
        fd = findiff(aE, bE, idxs)
        den = max(1e-12, float(np.abs(phi[idxs]).max()))
        fd_err.append(float(np.abs(fd - phi[idxs]).max() / den))

        names.append(stem)
        maps[stem] = phi.astype(np.float32)
        env[stem] = aE.astype(np.float32)        # input energy envelope -> the N5-style null
        for r in rates:
            ceil[r].append(ceiling(phi, r))
        if len(names) % 200 == 0:
            print(f"  {len(names)} clips", flush=True)

    print(f"\n=== REPLAY CHECK (16-bit floor = {2**-15:.3e}) ===")
    if repl_err:
        rp = np.array(repl_err)
        print(f"  max|replay - live| : median {np.median(rp):.3e}  max {rp.max():.3e}  "
              f"{'PASS (at quantisation floor)' if np.median(rp) < 4e-5 else 'FAIL'}  n={rp.size}")

    print("\n=== EXACTNESS ===")
    ok1 = bool(rec_err) and max(rec_err) < 1e-6
    ok2 = bool(fd_err) and float(np.median(fd_err)) < 1e-4
    print(f"  label reconstruction |10log10(A/B) - snr_db| : max {max(rec_err):.3e} dB  "
          f"{'PASS' if ok1 else 'FAIL'}")
    print(f"  analytic phi vs finite difference            : median {np.median(fd_err):.3e}  "
          f"{'PASS' if ok2 else 'FAIL'}")

    print("\n=== CEILINGS ===")
    print("  " + "".join(f"{r:>10.2f}Hz" for r in rates))
    print("  " + "".join(f"{np.nanmean(ceil[r]):>12.3f}" for r in rates))

    np.savez_compressed(a.out, names=np.array(names),
                        **{f"snr/{k}": v for k, v in maps.items()},
                        **{f"energy/{k}": v for k, v in env.items()})
    print(f"\nwrote {a.out}  clips={len(names)}  simulated-RIR skipped={n_sim}  errors={n_err}")
    return 0 if (ok1 and ok2) else 1


if __name__ == "__main__":
    raise SystemExit(main())
