#!/usr/bin/env python3
"""compute_srmr_maps_timeresolved.py — build the TIME-RESOLVED ORACLE 2D SRMR
modulation-energy targets from the Libri2Mix CLEAN s1 stem. This is the Q3 FIX.

THE PROBLEM IT FIXES
--------------------
The existing srmr_map target (src/srmr_maps.py::srmr_map_target, built by
scripts/compute_srmr_maps.py) is the TIME-AVERAGED 23x8 modulation-energy tensor:
it means over the frame axis before storing. Time-averaging destroys the per-clip
temporal structure, so the stored 23x8 field ends up nearly the same across clips
(clip-specificity gap ~0.01) and a head supervised on it cannot become
clip-discriminative — the Q3 negative result.

THE FIX
-------
Keep the TIME axis. srmr_maps.srmr_energy_tensor already returns the FULL
(n_acoustic=23, n_modulation=8, n_frames) energy tensor BEFORE the time-average; we
just stop averaging. To keep files small and the head's time grid fixed, the native
SRMR frame axis (~15 frames/s, 0.064 s hop on the 400 Hz envelope rate) is
DOWNSAMPLED to a coarse fixed time grid (default T_GRID=32 frames) by average-pooling
contiguous native-frame blocks. The per-clip temporal modulation profile is
preserved (clips with reverberant tails / overlap show the low→high modulation-energy
shift LOCALIZED in time), which is exactly the discriminative signal the averaged map
threw away.

STORED SHAPE + GRID
-------------------
  srmr_tr_logmap : (n_acoustic=23, n_modulation=8, T_grid) float32
                   per-(acoustic-band, modulation-band, coarse-time-bin) LOG10 energy
                   (log10(avg-pooled energy + floor)). LOG to tame the many-orders-of-
                   magnitude band spread (same convention as the averaged srmr_logmap).
  srmr_tr_avg    : (23, 8, T_grid) float32 — raw avg-pooled energy (linear), the field
                   whose TIME-MEAN aggregate reproduces the library scalar / GT.
  srmr_tr_mask   : (23, 8, T_grid) float32 — 1.0 on real (populated) time bins, 0.0 on
                   pad bins for clips shorter than T_grid native frames.
  srmr_scalar    : float — library SRMR scalar (from the time-mean of the field, exact).
  kstar          : int   — adaptive upper modulation band (SRMRpy 90%-bandwidth rule).
  n_frames_native: int   — native SRMR frame count before downsampling (diagnostic).
Axis order is (A, M, T) — acoustic, modulation, time — so a head can pool over T to
recover the averaged 23x8 map and over (low/high modulation bands) to recover the scalar.

CONSISTENCY WITH THE AVERAGED MAP
---------------------------------
Time-mean over the T_grid axis of srmr_tr_avg == the averaged srmr_avg (up to the
edge effect of average-pooling an unequal number of native frames into the last bin),
so srmr_scalar_from_avg(mean_T(srmr_tr_avg), kstar) reproduces the GT scalar. The
builder validates this with --validate (mean |tensor_agg - GT_scalar|).

s1-ONLY (no interferer): SRMR is a single-speaker reverberation property, computed on
the clean s1 stem for every clip suffix (same rationale as compute_srmr_maps.py).

Mirrors compute_srmr_maps.py: ONE small per-clip .pt + manifest.json, resume-safe.

Usage (one split):
  python scripts/compute_srmr_maps_timeresolved.py \
    --processed_dir $SHARED/data/processed/train \
    --stems_root    $SHARED/data/Libri2Mix/Libri2Mix/wav16k/min/train-100 \
    --output_dir    $SHARED/data/srmr_map_tr_targets/train \
    --gt_json       $SHARED/data/clean_features_train.json  [--t_grid 32] [--validate 20] [--limit N]
"""
import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import soundfile as sf  # noqa: E402

from srmr_maps import srmr_energy_tensor, srmr_scalar_from_avg  # noqa: E402

T_GRID_DEFAULT = 32       # coarse time bins (downsample target; ~25-50 is fine per spec)
LOG_FLOOR = 1e-8

_S1CLEAN_RE = re.compile(r"^(.*)_s1clean$")
_AUGNP_RE = re.compile(r"^(.*)_augNp\d+_\d+$")


def _read_mono(path: str):
    a, sr = sf.read(path, dtype="float32")
    if getattr(a, "ndim", 1) > 1:
        a = a.mean(axis=1)
    return a, sr


def _base_stem(stem: str) -> str:
    m = _S1CLEAN_RE.match(stem)
    if m:
        return m.group(1)
    m = _AUGNP_RE.match(stem)
    if m:
        return m.group(1)
    return stem


def downsample_time(energy: np.ndarray, t_grid: int):
    """Average-pool the time axis of (A, M, n_frames) energy to (A, M, t_grid).

    Returns (pooled, mask) where pooled is (A, M, t_grid) float64 and mask is
    (A, M, t_grid) float32 (1.0 on bins backed by >=1 native frame, 0.0 on pad bins).
    Native frames are split into t_grid contiguous blocks via linspace edges so each
    coarse bin covers a contiguous time span; short clips (n_frames < t_grid) populate
    the first n_frames bins (one native frame each) and zero-pad the rest (mask 0).
    """
    A, M, n = energy.shape
    pooled = np.zeros((A, M, t_grid), dtype=np.float64)
    mask = np.zeros((A, M, t_grid), dtype=np.float32)
    if n <= 0:
        return pooled, mask
    if n >= t_grid:
        edges = np.linspace(0, n, t_grid + 1).astype(int)
        for b in range(t_grid):
            a0, a1 = edges[b], max(edges[b] + 1, edges[b + 1])
            a1 = min(a1, n)
            if a1 > a0:
                pooled[:, :, b] = energy[:, :, a0:a1].mean(axis=2)
                mask[:, :, b] = 1.0
    else:
        # fewer native frames than bins: one native frame per leading bin, pad rest.
        pooled[:, :, :n] = energy
        mask[:, :, :n] = 1.0
    return pooled, mask


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--processed_dir", required=True,
                    help="dir of processed .pt clips (drives the filename list)")
    ap.add_argument("--stems_root", required=True,
                    help="split dir containing s1/ subdir (clean target stems)")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--gt_json", default=None,
                    help="clean_features_<split>.json for scalar-SRMR validation")
    ap.add_argument("--t_grid", type=int, default=T_GRID_DEFAULT,
                    help="coarse time bins to downsample the native SRMR frame axis to")
    ap.add_argument("--sr", type=int, default=16000)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--validate", type=int, default=0,
                    help="validate time-mean tensor-aggregate vs GT scalar on first N clips")
    ap.add_argument("--flush_every", type=int, default=100)
    ap.add_argument("--n_acoustic", type=int, default=23)
    ap.add_argument("--n_modulation", type=int, default=8)
    args = ap.parse_args()

    s1_dir = os.path.join(args.stems_root, "s1")
    if not os.path.isdir(s1_dir):
        print(f"ERROR: need s1 dir {s1_dir}", file=sys.stderr)
        return 2
    os.makedirs(args.output_dir, exist_ok=True)

    gt = {}
    if args.gt_json and os.path.exists(args.gt_json):
        gt = json.load(open(args.gt_json))

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

    n_done = n_missing = n_val = 0
    val_abs_err = []
    logmap_sum = 0.0
    logmap_n = 0
    for i, ptname in enumerate(pts):
        cached = torch.load(os.path.join(args.processed_dir, ptname), weights_only=False)
        filename = cached.get("filename", os.path.splitext(ptname)[0] + ".wav")
        stem = os.path.splitext(filename)[0]
        base = _base_stem(stem)

        already = (
            filename in manifest
            and os.path.exists(os.path.join(args.output_dir, manifest[filename]))
        )
        do_validate = args.validate and n_val < args.validate
        if already and not do_validate:
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

        energy, kstar = srmr_energy_tensor(
            s1.astype(np.float64), fs=sr,
            n_acoustic=args.n_acoustic, n_modulation=args.n_modulation,
        )                                                  # (A, M, n_frames_native)
        n_native = int(energy.shape[2])
        pooled, mask = downsample_time(energy, args.t_grid)        # (A, M, T_grid)
        # scalar from the time-mean of the field (exact library reduction).
        avg_full = np.mean(energy, axis=2)                         # (A, M)
        scalar = srmr_scalar_from_avg(avg_full, kstar)
        logmap = np.log10(np.clip(pooled, LOG_FLOOR, None)).astype(np.float32)

        if do_validate:
            # aggregate from the STORED downsampled field: mask-weighted time-mean.
            denom = mask.sum(axis=2, keepdims=True)
            denom[denom == 0] = 1.0
            tr_avg = (pooled * mask).sum(axis=2) / denom[:, :, 0]   # (A, M)
            agg = srmr_scalar_from_avg(tr_avg, kstar)
            gt_scalar = None
            for cand in (filename, base + ".wav", stem, base):
                if cand in gt and gt[cand].get("srmr") is not None:
                    gt_scalar = float(gt[cand]["srmr"]); break
            n_val += 1
            if gt_scalar is not None:
                err = abs(agg - gt_scalar)
                val_abs_err.append(err)
                print(f"  [VALIDATE] {stem[:38]:40s} tr_agg={agg:8.4f} "
                      f"GT={gt_scalar:8.4f} |err|={err:.4f} kstar={kstar} "
                      f"shape={logmap.shape} native_T={n_native}", flush=True)
            else:
                print(f"  [VALIDATE] {stem[:38]:40s} tr_agg={agg:8.4f} GT=NA "
                      f"shape={logmap.shape} native_T={n_native}", flush=True)

        if already:
            continue

        logmap_sum += float(logmap[mask > 0.5].sum())
        logmap_n += int((mask > 0.5).sum())

        rec = {
            "filename": filename,
            "srmr_tr_logmap": torch.from_numpy(np.asarray(logmap, dtype=np.float32)),  # (A,M,T)
            "srmr_tr_avg": torch.from_numpy(np.asarray(pooled, dtype=np.float32)),     # (A,M,T)
            "srmr_tr_mask": torch.from_numpy(np.asarray(mask, dtype=np.float32)),      # (A,M,T)
            "srmr_scalar": float(scalar),
            "kstar": int(kstar),
            "n_frames_native": int(n_native),
            "t_grid": int(args.t_grid),
        }
        rel = f"{stem}.pt"
        torch.save(rec, os.path.join(args.output_dir, rel))
        manifest[filename] = rel
        n_done += 1
        if n_done % args.flush_every == 0:
            tmp = manifest_path + ".tmp"
            json.dump(manifest, open(tmp, "w"))
            os.replace(tmp, manifest_path)
            print(f"  {n_done} computed ({i+1}/{len(pts)} scanned)", flush=True)

    tmp = manifest_path + ".tmp"
    json.dump(manifest, open(tmp, "w"))
    os.replace(tmp, manifest_path)

    if val_abs_err:
        arr = np.asarray(val_abs_err)
        print(f"[VALIDATE] {len(arr)} clips: mean |tr_agg - GT_scalar| = "
              f"{arr.mean():.5f} (max {arr.max():.5f})  "
              f"{'PASS' if arr.mean() < 1e-2 else 'CHECK'}")
    mean_logmap = (logmap_sum / logmap_n) if logmap_n else float("nan")
    print(f"done: {len(manifest)} time-resolved SRMR-map targets -> {args.output_dir}  "
          f"(computed {n_done}, missing-s1 {n_missing}, "
          f"shape ({args.n_acoustic},{args.n_modulation},{args.t_grid}), "
          f"mean_logmap={mean_logmap:.3f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
