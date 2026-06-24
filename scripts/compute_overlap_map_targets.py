#!/usr/bin/env python3
"""compute_overlap_map_targets.py — build the DENSE per-frame OVERLAP-MAP targets
(a build-A style dense target) from the ORACLE overlap segments already cached in
each processed .pt clip.

WHAT THIS IS
------------
The model's overlap-aware hedging is supervised today only through the prose
(`overlap with a ratio of X`) and through the optional input overlap_info channels.
This target exposes the oracle overlap structure as a DENSE per-frame regression /
classification map at the WavLM 50 Hz grid, so a map-supervision head can learn
*where in time* the two speakers overlap (not just the clip-level ratio). The
per-frame overlap map is the natural 1-D dense companion to the local-SNR timeline
(snr_map_head) and the 2D SRMR field (srmr_map).

The target is OVERLAP FRACTION per WavLM frame (continuous in [0,1]):
    overlap_map[t] = (overlapped samples in frame t) / (frame length samples)
A frame fully inside an overlap segment is 1.0; a frame straddling a segment edge
is the partial fraction; a frame outside any segment is 0.0. This is a strictly
richer signal than a hard binary mask (it preserves the sub-frame segment edges),
and a head can threshold it back to binary trivially. The mask is ALL-VALID (every
WavLM frame has a well-defined overlap fraction, since overlap_segments is oracle).

SOURCE (no audio read needed — pure metadata rasterization)
-----------------------------------------------------------
Each processed .pt already carries the oracle overlap geometry from
feature_extractor_mix.py (Silero VAD on the clean s1/s2 stems), in TWO forms:
  * overlap_segments : list[(start_sec, end_sec)]   — the authoritative segment list
  * overlap_info     : (T, >=1) tensor, col 0 == is_overlap binary per WavLM frame
We rasterize `overlap_segments` to the per-frame fraction at the 50 Hz grid (hop ==
frame == 320 samples @ 16 kHz, the preprocess.py convention) so the target aligns
1:1 with audio_features / overlap_info. `overlap_info[:,0]` is used only as a
cross-check (binarized fraction must match it on the overwhelming majority of
frames); the fraction map is the stored target.

This mirrors compute_snr_map_targets.py / compute_srmr_maps.py: ONE small per-clip
.pt keyed by filename + a manifest.json {filename: rel_path}. The dataset loads them
lazily; absent dir → no targets → loss no-ops (default-OFF).

Output record (per clip):
    {
      "filename"           : str,
      "overlap_map_target" : (T, 1) float32 — per-frame overlap fraction in [0,1],
      "overlap_map_mask"   : (T, 1) float32 — all-ones (every frame valid),
    }
Grid: T == audio_features.shape[0] (clip's exact WavLM frame count), 50 Hz, 20 ms/frame.

Usage (one split):
  python scripts/compute_overlap_map_targets.py \
    --processed_dir $SHARED/data/processed/train \
    --output_dir    $SHARED/data/overlap_map_targets/train   [--limit N]
"""
import argparse
import json
import os
import sys

import numpy as np  # noqa: E402
import torch  # noqa: E402

WAVLM_HOP = 320            # samples @ 16 kHz → 50 Hz frame grid (preprocess.py)
WAVLM_FRAME = 320         # non-overlapping frame == hop on the WavLM grid
SR = 16000


def overlap_fraction_timeline(segments_sec, T, sr=SR, hop=WAVLM_HOP, frame=WAVLM_FRAME):
    """Rasterize oracle overlap segments to a per-frame overlap-fraction timeline.

    For each WavLM frame t covering samples [t*hop, t*hop+frame), compute the
    fraction of that span covered by the union of `segments_sec`. Returns a
    float32 (T,) array in [0, 1]. Pure interval arithmetic on a per-sample 0/1
    overlap indicator so straddling frames get their true partial fraction and
    multi-segment clips (common: up to 7 segments) are handled by the union.
    """
    frac = np.zeros(int(T), dtype=np.float32)
    if T <= 0 or not segments_sec:
        return frac
    clip_end_samp = int(T) * hop
    # Build a per-sample binary overlap indicator over the clip, then frame-pool.
    # (T*320 samples; for a 6 s clip that is ~96k bools — cheap and exact.)
    ind = np.zeros(clip_end_samp, dtype=np.float32)
    for (s_sec, e_sec) in segments_sec:
        a = int(round(float(s_sec) * sr))
        b = int(round(float(e_sec) * sr))
        a = max(0, min(a, clip_end_samp))
        b = max(0, min(b, clip_end_samp))
        if b > a:
            ind[a:b] = 1.0
    for t in range(int(T)):
        s = t * hop
        e = min(s + frame, clip_end_samp)
        if e > s:
            frac[t] = float(ind[s:e].mean())
    return frac


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--processed_dir", required=True,
                    help="dir of processed .pt clips (carry overlap_segments + audio_features)")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--sr", type=int, default=SR)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--flush_every", type=int, default=500)
    args = ap.parse_args()

    if not os.path.isdir(args.processed_dir):
        print(f"ERROR: processed_dir not found {args.processed_dir}", file=sys.stderr)
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

    n_done = n_skip = 0
    n_with_overlap = 0
    frac_sum = 0.0
    frac_n = 0
    bin_match = 0
    bin_total = 0
    for i, ptname in enumerate(pts):
        cached = torch.load(os.path.join(args.processed_dir, ptname), weights_only=False)
        filename = cached.get("filename", os.path.splitext(ptname)[0] + ".wav")
        stem = os.path.splitext(filename)[0]

        if filename in manifest and os.path.exists(
            os.path.join(args.output_dir, manifest[filename])
        ):
            continue

        T = int(cached["audio_features"].shape[0])
        segs = cached.get("overlap_segments", []) or []
        frac = overlap_fraction_timeline(segs, T, sr=args.sr)

        # cross-check against the cached binary is_overlap channel (overlap_info col 0)
        oi = cached.get("overlap_info", None)
        if oi is not None and getattr(oi, "shape", (0,))[0] == T and oi.shape[1] >= 1:
            cached_bin = (oi[:, 0].cpu().numpy() > 0.5).astype(np.float32)
            our_bin = (frac > 0.5).astype(np.float32)
            bin_match += int((cached_bin == our_bin).sum())
            bin_total += T

        target = frac.reshape(T, 1).astype(np.float32)            # (T, 1)
        mask = np.ones((T, 1), dtype=np.float32)                  # all-valid

        if frac.max() > 0:
            n_with_overlap += 1
        frac_sum += float(frac.sum())
        frac_n += int(T)

        rec = {
            "filename": filename,
            "overlap_map_target": torch.from_numpy(target),       # (T, 1)
            "overlap_map_mask": torch.from_numpy(mask),           # (T, 1)
        }
        rel = f"{stem}.pt"
        torch.save(rec, os.path.join(args.output_dir, rel))
        manifest[filename] = rel
        n_done += 1
        if n_done % args.flush_every == 0:
            tmp = manifest_path + ".tmp"
            json.dump(manifest, open(tmp, "w"))
            os.replace(tmp, manifest_path)
            print(f"  {n_done} computed ({i+1}/{len(pts)} scanned) "
                  f"with_overlap={n_with_overlap}", flush=True)

    tmp = manifest_path + ".tmp"
    json.dump(manifest, open(tmp, "w"))
    os.replace(tmp, manifest_path)

    mean_frac = (frac_sum / frac_n) if frac_n else 0.0
    bin_agree = (bin_match / bin_total) if bin_total else float("nan")
    print(f"done: {len(manifest)} overlap-map targets -> {args.output_dir}  "
          f"(computed {n_done}, skipped {n_skip}, "
          f"clips_with_overlap={n_with_overlap}, mean_overlap_frac={mean_frac:.4f}, "
          f"binarized-vs-cached agreement={bin_agree:.4f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
