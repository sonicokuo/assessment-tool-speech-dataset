#!/usr/bin/env python3
"""Compute DENSE oracle SNR supervision targets from the clean Libri2Mix stems for
every clip in a features CSV, mirroring scripts/compute_clean_features.py.

For each clip it reads the CLEAN s1 (target) and s2 (interferer) stems and builds:

  (1) snr_timeline  (T,)         per-FRAME instantaneous target-vs-interferer SNR,
                                 10*log10((sum s1^2+eps)/(sum s2^2+eps)) on 320-sample
                                 (20 ms / 50 Hz) frames, clamped [-30, 40] dB. T equals
                                 the clip's WavLM frame count (n_samples // 320), so it
                                 aligns 1:1 with the model's WavLM frames / overlap_info.
                                 s1-active frames are flagged so silence is not read as a
                                 real local SNR.  [the TIME map — cleanest oracle]

  (2) irm_map      (T_p,F_bins)  per-T-F-bin Ideal Ratio Mask
                                 (|S1|^2/(|S1|^2+|S2|^2+eps))^0.5 in [0,1], STFT 25 ms /
                                 10 ms, freq-pooled to F_bins (default 8 = BEATs F_p) and
                                 time-pooled to T_p (default BEATs T_p). ~1 where s1
                                 dominates, ~0.5 where equal, ~0 where s2 dominates.
                                 [the T-F map — the literal separation target]

Pure math lives in src/snr_maps.py (numpy only, unit-testable). This driver only does
IO + a resume-safe loop, exactly like compute_clean_features.py.

The s1/s2 stems live next to mix_clean in the Libri2Mix tree:
  .../Libri2Mix/wav16k/min/<split>/{mix_clean,s1,s2}/<filename>.wav
Pass --stems_root pointing at the split dir that CONTAINS s1/ and s2/ (the same dir you
would pass to feature_extractor_mix as --libri2mix_root).

OUTPUT
------
By default a single .npz archive (compact for the (T_p,F_bins) maps + variable-length
timelines): each clip stored as `<filename>::snr_timeline`, `<filename>::s1_active`,
`<filename>::irm_map`; plus a `_meta` json blob with {filename: {t_p, f_bins, T}}.
Pass --json to instead write a {filename: {snr_timeline, irm_map, t_p, f_bins}} JSON
lookup (larger, human-inspectable). Both formats are resume-safe.

ALIGNMENT NOTE (frame rate vs T_p)
----------------------------------
snr_timeline is at the WavLM 50 Hz frame grid (T == n_samples//320) — it lines up 1:1
with overlap_info rows and the adapter's pre-conv WavLM frames. irm_map is pooled to the
BEATs patch grid (T_p, F_bins). If you cache BEATs patches at preprocessing time the
EXACT T_p is known; pass it per-clip via --t_p_from_beats <dir> to re-pool onto the
cached grid. Without that flag T_p defaults to beats_t_p(n_samples) (the documented BEATs
fbank math, T_p ~= 31 for a 5 s clip). The map is cheap to re-pool, so a default-grid
extraction here can be re-pooled to the real cached T_p inside the dataset loader.

Usage:
  python scripts/compute_snr_maps.py \
    --features_csv $SHARED/data/features_pyannote/test.csv \
    --stems_root   $SHARED/data/Libri2Mix/Libri2Mix/wav16k/min/test \
    --output       $SHARED/data/snr_maps_test.npz   [--f_bins 8] [--limit N] [--json]

DO NOT run the full extraction from this docstring blindly; the FULL command is at the
bottom of this file and in the subagent report.
"""
import argparse
import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import soundfile as sf  # noqa: E402
from snr_maps import (  # noqa: E402
    snr_timeline_from_stems,
    irm_map_from_stems,
    F_BINS_DEFAULT,
)


def _read_mono(path: str):
    a, sr = sf.read(path, dtype="float32")
    if getattr(a, "ndim", 1) > 1:
        a = a.mean(axis=1)
    return a, sr


# ── NPZ store (resume-safe) ──────────────────────────────────────────────────
def _load_npz(path: str):
    """Return (data_dict, meta_dict). data_dict maps array keys -> np.ndarray."""
    if not os.path.exists(path):
        return {}, {}
    try:
        z = np.load(path, allow_pickle=True)
        data = {k: z[k] for k in z.files if k != "_meta"}
        meta = {}
        if "_meta" in z.files:
            meta = json.loads(str(z["_meta"].item()))
        return data, meta
    except Exception:
        return {}, {}


def _save_npz(path: str, data: dict, meta: dict):
    tmp = path + ".tmp.npz"
    np.savez_compressed(tmp, _meta=np.array(json.dumps(meta)), **data)
    os.replace(tmp, path)


def _save_json(path: str, out: dict):
    tmp = path + ".tmp"
    json.dump(out, open(tmp, "w"))
    os.replace(tmp, path)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--features_csv", required=True)
    ap.add_argument("--stems_root", required=True,
                    help="split dir containing s1/ and s2/ subdirs (== feature_extractor_mix --libri2mix_root)")
    ap.add_argument("--output", required=True, help=".npz (default) or .json with --json")
    ap.add_argument("--f_bins", type=int, default=F_BINS_DEFAULT,
                    help="frequency bins for the IRM map (default 8 = BEATs F_p)")
    ap.add_argument("--t_p", type=int, default=0,
                    help="fixed T_p for every clip's IRM map (0 = per-clip BEATs T_p)")
    ap.add_argument("--json", action="store_true",
                    help="write a JSON lookup instead of the compact .npz archive")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--flush_every", type=int, default=200)
    args = ap.parse_args()

    s1_dir = os.path.join(args.stems_root, "s1")
    s2_dir = os.path.join(args.stems_root, "s2")
    if not os.path.isdir(s1_dir):
        print(f"ERROR: s1 dir not found: {s1_dir}", file=sys.stderr)
        return 2
    if not os.path.isdir(s2_dir):
        print(f"ERROR: s2 dir not found: {s2_dir} (interferer required for SNR maps)",
              file=sys.stderr)
        return 2

    rows = list(csv.DictReader(open(args.features_csv)))
    if args.limit:
        rows = rows[: args.limit]

    use_json = args.json
    if use_json:
        out = {}
        if os.path.exists(args.output):
            try:
                out = json.load(open(args.output))
            except Exception:
                out = {}
        done_keys = set(out.keys())
    else:
        data, meta = _load_npz(args.output)
        done_keys = set(meta.keys())

    t_p_arg = args.t_p if args.t_p > 0 else None
    n_done = n_missing = 0
    for i, r in enumerate(rows):
        fn = (r.get("filename") or "").strip()
        if not fn or fn in done_keys:
            continue
        s1_path = os.path.join(s1_dir, fn)
        s2_path = os.path.join(s2_dir, fn)
        if not (os.path.exists(s1_path) and os.path.exists(s2_path)):
            n_missing += 1
            continue
        try:
            s1, sr = _read_mono(s1_path)
            s2, _ = _read_mono(s2_path)
        except Exception as e:  # noqa: BLE001
            print(f"  [WARNING] could not read stems for {fn}: {e}")
            continue

        tl = snr_timeline_from_stems(s1, s2)
        irm = irm_map_from_stems(s1, s2, t_p=t_p_arg, f_bins=args.f_bins)

        if use_json:
            out[fn] = {
                "snr_timeline": [round(float(v), 3) for v in tl["snr_timeline"]],
                "s1_active": [bool(v) for v in tl["s1_active"]],
                "irm_map": [[round(float(v), 4) for v in row] for row in irm["irm_map"]],
                "irm_active": [[bool(v) for v in row] for row in irm["irm_active"]],
                "t_p": irm["t_p"],
                "f_bins": irm["f_bins"],
            }
        else:
            data[f"{fn}::snr_timeline"] = tl["snr_timeline"].astype(np.float32)
            data[f"{fn}::s1_active"] = tl["s1_active"].astype(np.bool_)
            data[f"{fn}::irm_map"] = irm["irm_map"].astype(np.float32)
            data[f"{fn}::irm_active"] = irm["irm_active"].astype(np.bool_)
            meta[fn] = {"t_p": int(irm["t_p"]), "f_bins": int(irm["f_bins"]),
                        "T": int(tl["snr_timeline"].shape[0])}

        done_keys.add(fn)
        n_done += 1
        if n_done % args.flush_every == 0:
            if use_json:
                _save_json(args.output, out)
            else:
                _save_npz(args.output, data, meta)
            print(f"  {n_done} computed ({i+1}/{len(rows)} scanned)", flush=True)

    if use_json:
        _save_json(args.output, out)
        n_total = len(out)
    else:
        _save_npz(args.output, data, meta)
        n_total = len(meta)
    print(f"done: {n_total} clips with dense SNR maps -> {args.output}  "
          f"(computed {n_done}, missing-stem {n_missing}, "
          f"format={'json' if use_json else 'npz'}, f_bins={args.f_bins})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# FULL EXTRACTION (do not run from the docstring; run deliberately on a compute node):
#   for split in train-100 dev test; do
#     python scripts/compute_snr_maps.py \
#       --features_csv $SHARED/data/features_pyannote/${split}.csv \
#       --stems_root   $SHARED/data/Libri2Mix/Libri2Mix/wav16k/min/${split} \
#       --output       $SHARED/data/snr_maps_${split}.npz --f_bins 8
#   done
