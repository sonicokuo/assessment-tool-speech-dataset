#!/usr/bin/env python3
"""compute_pitch_map_targets.py — build the DENSE per-frame ORACLE F0 (pitch)-MAP
targets from the Libri2Mix CLEAN s1 stem.

WHY (the f0_mean rescue)
------------------------
The scalar f0_mean target REGRESSED under map-supervision because it is a single
clip-level number with little per-clip signal for a dense head to latch onto (same
pathology as the time-averaged SRMR map — not clip-discriminative). The per-FRAME F0
CONTOUR, by contrast, has real per-clip temporal variation: it traces the speaker's
intonation over time. Supervising the model on the dense oracle pitch contour gives
the audio→F0 pathway a frame-level gradient, and the scalar f0_mean can be read back
as the voiced-frame mean of the predicted contour (the CBM tie, mirroring the SNR
map's pooled-scalar tie).

The contour is computed on the CLEAN s1 stem (oracle pitch), NOT the mixture, so it is
the true single-speaker F0 even where the two speakers overlap — exactly where a model
that hears only the mix cannot recover F0 and should hedge. This makes the target a
clean teacher for the "F0 unreliable under overlap" behaviour.

EXTRACTOR
---------
Praat autocorrelation pitch via parselmouth (Sound.to_pitch_ac), the same estimator
family as the clean-stem scalar F0 GT (clean_f0_*.json). Pitch floor 75 Hz / ceiling
500 Hz (speech range). Praat is sampled on a fixed `time_step` so its frames land at
predictable times; we then RESAMPLE Praat's voiced/unvoiced contour onto the WavLM
50 Hz frame centres (frame t centre == (t + 0.5) * 320 / sr seconds) by nearest Praat
frame. Unvoiced Praat frames (F0 == 0 / NaN) → mask 0 at that WavLM frame.

TARGET REPRESENTATION
---------------------
Stored in TWO value forms (head picks one; both share the voiced mask):
  * f0_hz_target     (T, 1) float32 — F0 in Hz, 0.0 on unvoiced frames (mask 0).
  * f0_loghz_target  (T, 1) float32 — log10(F0 Hz) on voiced frames, 0.0 elsewhere.
                     log-Hz is the more stable regression target (pitch is perceived
                     ~log; matches the SRMR log-map convention).
  * f0_map_mask      (T, 1) float32 — 1.0 on voiced frames, 0.0 on unvoiced/silence.
Grid: T == audio_features.shape[0] (clip's exact WavLM frame count), 50 Hz, 20 ms/frame.

STEM RESOLUTION (mixture-only splits)
-------------------------------------
s1 = --stems_root/s1/<base>.wav. The processed dirs here (processed/{train,val,test})
are mixture clips whose stem is the base Libri2Mix id, so base == clip stem. (The
processed_aug suffix logic from compute_snr_map_targets is not needed for these dirs;
if an _s1clean / _augNp suffix is ever seen the base is stripped the same way so the
clean s1 contour is still found.)

Mirrors compute_srmr_maps.py: ONE small per-clip .pt + manifest.json, resume-safe.

Usage (one split):
  python scripts/compute_pitch_map_targets.py \
    --processed_dir $SHARED/data/processed/train \
    --stems_root    $SHARED/data/Libri2Mix/Libri2Mix/wav16k/min/train-100 \
    --output_dir    $SHARED/data/pitch_map_targets/train   [--limit N]
"""
import argparse
import json
import os
import re
import sys

import numpy as np  # noqa: E402
import torch  # noqa: E402
import soundfile as sf  # noqa: E402

WAVLM_HOP = 320           # samples @ 16 kHz → 50 Hz frame grid (preprocess.py)
SR = 16000
F0_FLOOR = 75.0           # Hz (speech range; matches clean-stem F0 GT family)
F0_CEIL = 500.0           # Hz
PRAAT_TIME_STEP = 0.005   # s — Praat pitch sampling step (finer than 20 ms WavLM hop)

_S1CLEAN_RE = re.compile(r"^(.*)_s1clean$")
_AUGNP_RE = re.compile(r"^(.*)_augNp\d+_\d+$")


def _read_mono(path: str):
    a, sr = sf.read(path, dtype="float32")
    if getattr(a, "ndim", 1) > 1:
        a = a.mean(axis=1)
    return a, sr


def _base_stem(stem: str) -> str:
    """Strip any processed_aug suffix to recover the BASE clip (whose s1 we read)."""
    m = _S1CLEAN_RE.match(stem)
    if m:
        return m.group(1)
    m = _AUGNP_RE.match(stem)
    if m:
        return m.group(1)
    return stem


def f0_contour_praat(x: np.ndarray, sr: int, T: int,
                     hop: int = WAVLM_HOP,
                     f0_floor: float = F0_FLOOR, f0_ceil: float = F0_CEIL,
                     time_step: float = PRAAT_TIME_STEP):
    """Per-WavLM-frame oracle F0 (Hz) + voiced mask from a clean mono signal via Praat.

    Returns (f0_hz, voiced) both (T,) float32 / float32 in {0,1}. F0 is 0.0 where
    unvoiced. The Praat pitch object is sampled at `time_step` then resampled to the
    WavLM frame centres ((t+0.5)*hop/sr seconds) by nearest Praat frame.
    """
    import parselmouth  # imported here so the module imports even if absent at parse-time

    f0_hz = np.zeros(int(T), dtype=np.float32)
    voiced = np.zeros(int(T), dtype=np.float32)
    if int(T) <= 0 or x.size < int(0.05 * sr):
        return f0_hz, voiced

    snd = parselmouth.Sound(x.astype(np.float64), sampling_frequency=float(sr))
    try:
        pitch = snd.to_pitch_ac(time_step=time_step, pitch_floor=f0_floor,
                                pitch_ceiling=f0_ceil)
    except Exception:
        # to_pitch_ac can reject very short / silent clips; fall back to default ac.
        try:
            pitch = snd.to_pitch(time_step=time_step,
                                 pitch_floor=f0_floor, pitch_ceiling=f0_ceil)
        except Exception:
            return f0_hz, voiced

    praat_t = pitch.xs()                                  # (P,) Praat frame times (s)
    praat_f0 = pitch.selected_array["frequency"]          # (P,) Hz, 0.0 = unvoiced
    if praat_t.size == 0:
        return f0_hz, voiced

    # WavLM frame-centre times, nearest Praat frame per WavLM frame.
    centres = (np.arange(int(T)) + 0.5) * hop / float(sr)
    idx = np.searchsorted(praat_t, centres)
    idx = np.clip(idx, 0, praat_t.size - 1)
    # refine to true nearest (searchsorted gives the right insertion point)
    left = np.clip(idx - 1, 0, praat_t.size - 1)
    pick_left = np.abs(praat_t[left] - centres) <= np.abs(praat_t[idx] - centres)
    idx = np.where(pick_left, left, idx)

    vals = praat_f0[idx].astype(np.float32)
    vmask = (vals > 0.0) & np.isfinite(vals) & (vals >= f0_floor) & (vals <= f0_ceil)
    f0_hz[vmask] = vals[vmask]
    voiced[vmask] = 1.0
    return f0_hz, voiced


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--processed_dir", required=True,
                    help="dir of processed .pt clips (drives filename list + per-clip T)")
    ap.add_argument("--stems_root", required=True,
                    help="split dir containing s1/ subdir (clean target stems)")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--sr", type=int, default=SR)
    ap.add_argument("--f0_floor", type=float, default=F0_FLOOR)
    ap.add_argument("--f0_ceil", type=float, default=F0_CEIL)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--flush_every", type=int, default=200)
    args = ap.parse_args()

    s1_dir = os.path.join(args.stems_root, "s1")
    if not os.path.isdir(s1_dir):
        print(f"ERROR: need s1 dir {s1_dir}", file=sys.stderr)
        return 2
    os.makedirs(args.output_dir, exist_ok=True)

    pts = sorted(f for f in os.listdir(args.processed_dir) if f.endswith(".pt"))
    if args.limit:
        pts = pts[: args.limit]

    manifest_path = os.path.join(args.output_dir, "manifest.json")
    manifest = {}
    if os.path.exists(manifest_path):
        try:
            manifest = json.load(open(manifest_path))
        except Exception:
            manifest = {}

    n_done = n_missing = 0
    voiced_sum = 0.0
    frame_n = 0
    f0_vals_sum = 0.0
    f0_vals_n = 0
    for i, ptname in enumerate(pts):
        cached = torch.load(os.path.join(args.processed_dir, ptname), weights_only=False)
        filename = cached.get("filename", os.path.splitext(ptname)[0] + ".wav")
        stem = os.path.splitext(filename)[0]
        base = _base_stem(stem)
        T = int(cached["audio_features"].shape[0])

        if filename in manifest and os.path.exists(
            os.path.join(args.output_dir, manifest[filename])
        ):
            continue

        s1_path = os.path.join(s1_dir, base + ".wav")
        if not os.path.exists(s1_path):
            if n_missing < 20:
                print(f"  [WARNING] missing s1 for {stem}: {s1_path}", flush=True)
            n_missing += 1
            continue
        try:
            s1, sr = _read_mono(s1_path)
        except Exception as e:  # noqa: BLE001
            print(f"  [WARNING] read failed {filename}: {e}")
            continue

        f0_hz, voiced = f0_contour_praat(
            s1, sr, T, hop=WAVLM_HOP, f0_floor=args.f0_floor, f0_ceil=args.f0_ceil,
        )
        f0_log = np.zeros_like(f0_hz)
        vm = voiced > 0.5
        f0_log[vm] = np.log10(np.clip(f0_hz[vm], 1e-3, None)).astype(np.float32)

        voiced_sum += float(voiced.sum())
        frame_n += int(T)
        f0_vals_sum += float(f0_hz[vm].sum())
        f0_vals_n += int(vm.sum())

        rec = {
            "filename": filename,
            "f0_hz_target": torch.from_numpy(f0_hz.reshape(T, 1)),       # (T, 1) Hz
            "f0_loghz_target": torch.from_numpy(f0_log.reshape(T, 1)),   # (T, 1) log10 Hz
            "f0_map_mask": torch.from_numpy(voiced.reshape(T, 1)),       # (T, 1) voiced
        }
        rel = f"{stem}.pt"
        torch.save(rec, os.path.join(args.output_dir, rel))
        manifest[filename] = rel
        n_done += 1
        if n_done % args.flush_every == 0:
            tmp = manifest_path + ".tmp"
            json.dump(manifest, open(tmp, "w"))
            os.replace(tmp, manifest_path)
            vf = (voiced_sum / frame_n) if frame_n else 0.0
            mf0 = (f0_vals_sum / f0_vals_n) if f0_vals_n else 0.0
            print(f"  {n_done} computed ({i+1}/{len(pts)} scanned) "
                  f"voiced_frac={vf:.3f} mean_voiced_f0={mf0:.1f}Hz", flush=True)

    tmp = manifest_path + ".tmp"
    json.dump(manifest, open(tmp, "w"))
    os.replace(tmp, manifest_path)

    voiced_frac = (voiced_sum / frame_n) if frame_n else 0.0
    mean_f0 = (f0_vals_sum / f0_vals_n) if f0_vals_n else 0.0
    print(f"done: {len(manifest)} pitch-map targets -> {args.output_dir}  "
          f"(computed {n_done}, missing-s1 {n_missing}, "
          f"voiced_frac={voiced_frac:.4f}, mean_voiced_f0={mean_f0:.2f}Hz)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
