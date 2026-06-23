#!/usr/bin/env python3
"""compute_srmr_maps.py — build the ORACLE 2D SRMR modulation-energy targets
(the build-A SRMR-map targets) from the Libri2Mix CLEAN s1 stem.

SRMR (Falk, Zheng & Chan, IEEE TASLP 2010) is the one AQUA-NL feature with a
genuinely time-frequency-2D dense target: the per-(acoustic-band x modulation-band)
modulation-energy tensor (default 23 gammatone bands x 8 modulation bands). The
scalar SRMR a quality model reports is its aggregate (low/high modulation-energy
ratio). This driver mirrors compute_snr_map_targets.py: for every processed .pt clip
in --processed_dir it locates the clip's CLEAN s1 stem, computes the 23x8 oracle
tensor (srmr_maps.srmr_map_target), and writes ONE small per-clip target .pt keyed
by filename + a manifest.json. The dataset loads these lazily by filename
(PreprocessedDataset(srmr_map_dir=...)); absent dir → no targets → loss no-ops
(default-OFF byte-identical).

WHY ONLY s1 (no interferer)
---------------------------
SRMR is a property of the SINGLE-SPEAKER signal's reverberation, not of the mixture
(clean_features.clean_srmr already computes the scalar GT on s1 alone). So the dense
2D target is computed on the clean s1 stem for EVERY clip suffix:
  * mixture (no suffix):  s1 = stems_root/s1/<base>.wav
  * <base>_s1clean:       input == s1, so same s1 stem (stems_root/s1/<base>.wav)
  * <base>_augNp{N}_00:   the noise augmentation does NOT touch s1's reverberation,
                          so the clean-s1 SRMR is the well-posed reverb target.
This means the SRMR target is per BASE clip; the per-suffix processed_aug expansion
reuses the same base-s1 SRMR (resolved here per .pt stem so the manifest stays 1:1
with processed_aug).

VALIDATION (--validate N)
-------------------------
Re-derives the scalar from the stored 23x8 tensor (srmr_scalar_from_avg) and compares
it to the clean_features GT json (--gt_json), reporting mean abs error over N clips.
Tensor-aggregate == library scalar == GT to 4 dp (the aggregate is the exact SRMRpy
reduction); reverberant/augmented clips show the expected low->high modulation-energy
shift (lower SRMR / more high-mod energy).

Resume-safe via manifest.json (flush_every).

Wav roots (confirm with the inventory):
  --stems_root  $SHARED/data/Libri2Mix/Libri2Mix/wav16k/min/<split>   (has s1/)
  --gt_json     $SHARED/data/clean_features_<split>.json              (scalar srmr GT)

Usage (one split):
  python scripts/compute_srmr_maps.py \
    --processed_dir $SHARED/data/processed_aug/val \
    --stems_root    $SHARED/data/Libri2Mix/Libri2Mix/wav16k/min/dev \
    --output_dir    $SHARED/data/srmr_map_targets/dev \
    --gt_json       $SHARED/data/clean_features_dev.json   [--validate 20] [--limit N]
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

from srmr_maps import srmr_map_target, srmr_scalar_from_avg  # noqa: E402

_S1CLEAN_RE = re.compile(r"^(.*)_s1clean$")
_AUGNP_RE = re.compile(r"^(.*)_augNp\d+_\d+$")


def _read_mono(path: str):
    a, sr = sf.read(path, dtype="float32")
    if getattr(a, "ndim", 1) > 1:
        a = a.mean(axis=1)
    return a, sr


def _base_stem(stem: str) -> str:
    """Strip the processed_aug suffix to recover the BASE clip (whose s1 we read)."""
    m = _S1CLEAN_RE.match(stem)
    if m:
        return m.group(1)
    m = _AUGNP_RE.match(stem)
    if m:
        return m.group(1)
    return stem


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
    ap.add_argument("--sr", type=int, default=16000)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--validate", type=int, default=0,
                    help="validate tensor-aggregate vs GT scalar on the first N clips, then continue")
    ap.add_argument("--flush_every", type=int, default=200)
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

        tgt = srmr_map_target(
            s1, fs=sr, n_acoustic=args.n_acoustic, n_modulation=args.n_modulation,
        )

        # ── validation: tensor-aggregate vs GT scalar ──
        if do_validate:
            agg = srmr_scalar_from_avg(tgt["srmr_avg"], tgt["kstar"])
            gt_scalar = None
            for cand in (filename, base + ".wav", stem, base):
                if cand in gt and gt[cand].get("srmr") is not None:
                    gt_scalar = float(gt[cand]["srmr"]); break
            n_val += 1
            if gt_scalar is not None:
                err = abs(agg - gt_scalar)
                val_abs_err.append(err)
                print(f"  [VALIDATE] {stem[:40]:42s} tensor_agg={agg:8.4f} "
                      f"GT_scalar={gt_scalar:8.4f} |err|={err:.4f} "
                      f"kstar={tgt['kstar']} shape={tgt['srmr_logmap'].shape}", flush=True)
            else:
                print(f"  [VALIDATE] {stem[:40]:42s} tensor_agg={agg:8.4f} "
                      f"GT_scalar=NA shape={tgt['srmr_logmap'].shape}", flush=True)

        if already:
            continue

        rec = {
            "filename": filename,
            "srmr_logmap": torch.from_numpy(np.asarray(tgt["srmr_logmap"], dtype=np.float32)),  # (A, M)
            "srmr_mask": torch.from_numpy(np.asarray(tgt["srmr_mask"], dtype=np.float32)),       # (A, M)
            "srmr_avg": torch.from_numpy(np.asarray(tgt["srmr_avg"], dtype=np.float32)),         # (A, M)
            "srmr_scalar": float(tgt["srmr_scalar"]),
            "kstar": int(tgt["kstar"]),
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
        print(f"[VALIDATE] {len(arr)} clips: mean |tensor_agg - GT_scalar| = "
              f"{arr.mean():.5f} (max {arr.max():.5f})  "
              f"{'PASS' if arr.mean() < 1e-2 else 'CHECK'}")
    print(f"done: {len(manifest)} SRMR-map targets -> {args.output_dir}  "
          f"(computed {n_done}, missing-s1 {n_missing}, "
          f"tensor {args.n_acoustic}x{args.n_modulation})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
