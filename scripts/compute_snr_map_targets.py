#!/usr/bin/env python3
"""compute_snr_map_targets.py — build the ORACLE dense local-SNR-map targets
(the build-A targets) from the Libri2Mix clean s1/s2 stems.

For every processed .pt clip in --processed_dir it reads the clip's WavLM frame
count T (off audio_features) and the clean s1 / s2 stems, then computes:

  * per-FRAME instantaneous SNR timeline  (T,)  + s1-active mask (T,)
        SNR(i) = 10*log10( sum_frame s1^2 / sum_frame s2^2 )    on the 50 Hz WavLM grid
        (clean_features.snr_timeline_db — exact oracle since mix = s1 + s2)
  * optional per-T-F IRM grid (T_p, F_bins)  (clean_features.irm_grid)  with --irm

and writes ONE small per-clip target .pt per filename into --output_dir, plus a
manifest.json {filename: rel_path}. The dataset loads these lazily by filename
(PreprocessedDataset(snr_map_dir=...)), so the dense targets never bloat the main
processed .pt files and the feature is default-OFF (no dir → no targets → no-op).

The frame grid is matched to preprocess.py (WavLM-Large hop 320 @ 16 kHz → 50 Hz),
and the timeline is forced to the clip's exact WavLM T so it aligns frame-for-frame
with audio_features in collate.

Stems live next to mix_clean in the Libri2Mix tree:
  .../Libri2Mix/wav16k/min/<split>/{mix_clean,s1,s2}/<filename>.wav
Pass --stems_root pointing at the split dir that CONTAINS s1/ and s2/.

Usage:
  python scripts/compute_snr_map_targets.py \
    --processed_dir $SHARED/data/processed/train \
    --stems_root    $SHARED/data/Libri2Mix/Libri2Mix/wav16k/min/train-100 \
    --output_dir    $SHARED/data/snr_map_targets/train   [--irm] [--limit N]
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import soundfile as sf  # noqa: E402

from clean_features import snr_timeline_db, irm_grid  # noqa: E402

WAVLM_HOP = 320          # samples @ 16 kHz → 50 Hz frame grid (preprocess.py)
F_P_DEFAULT = 8          # BEATs frequency-patch count


def _read_mono(path: str):
    a, sr = sf.read(path, dtype="float32")
    if getattr(a, "ndim", 1) > 1:
        a = a.mean(axis=1)
    return a, sr


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--processed_dir", required=True,
                    help="dir of processed .pt clips (for the per-clip WavLM frame count T)")
    ap.add_argument("--stems_root", required=True,
                    help="split dir containing s1/ and s2/ subdirs")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--sr", type=int, default=16000)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--flush_every", type=int, default=200)
    ap.add_argument("--irm", action="store_true",
                    help="also compute the per-T-F IRM grid (needs --t_p_from_beats or a fixed t_p)")
    ap.add_argument("--irm_t_p", type=int, default=0,
                    help="fixed IRM time-patch count; 0 → round(T/20) (WavLM frames per BEATs time-patch)")
    ap.add_argument("--irm_f_bins", type=int, default=F_P_DEFAULT)
    args = ap.parse_args()

    s1_dir = os.path.join(args.stems_root, "s1")
    s2_dir = os.path.join(args.stems_root, "s2")
    if not os.path.isdir(s1_dir) or not os.path.isdir(s2_dir):
        print(f"ERROR: need both {s1_dir} and {s2_dir}", file=sys.stderr)
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
    for i, ptname in enumerate(pts):
        cached = torch.load(os.path.join(args.processed_dir, ptname), weights_only=False)
        filename = cached.get("filename", os.path.splitext(ptname)[0] + ".wav")
        stem = os.path.splitext(filename)[0]
        T = int(cached["audio_features"].shape[0])

        if filename in manifest and os.path.exists(
            os.path.join(args.output_dir, manifest[filename])
        ):
            continue

        s1_path = os.path.join(s1_dir, filename)
        s2_path = os.path.join(s2_dir, filename)
        if not (os.path.exists(s1_path) and os.path.exists(s2_path)):
            n_missing += 1
            continue
        try:
            s1, sr = _read_mono(s1_path)
            s2, _ = _read_mono(s2_path)
        except Exception as e:  # noqa: BLE001
            print(f"  [WARNING] read failed {filename}: {e}")
            continue

        timeline, active = snr_timeline_db(
            s1, s2, sr=sr, hop=WAVLM_HOP, n_frames=T,
        )
        rec = {
            "filename": filename,
            "snr_map_target": torch.from_numpy(np.asarray(timeline, dtype=np.float32)),  # (T,)
            "snr_map_mask": torch.from_numpy(np.asarray(active, dtype=np.float32)),       # (T,)
        }
        if args.irm:
            t_p = args.irm_t_p or max(1, int(round(T / 20.0)))
            irm = irm_grid(s1, s2, sr=sr, t_p=t_p, f_bins=args.irm_f_bins)
            rec["snr_irm_target"] = torch.from_numpy(np.asarray(irm, dtype=np.float32))   # (T_p, F)

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
    print(f"done: {len(manifest)} dense SNR-map targets -> {args.output_dir}  "
          f"(computed {n_done}, missing-stem {n_missing}, irm={'on' if args.irm else 'off'})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
