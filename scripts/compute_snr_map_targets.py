#!/usr/bin/env python3
"""compute_snr_map_targets.py — build the ORACLE dense local-SNR-map targets
(the build-A targets) from the Libri2Mix clean s1 stem and the ACTUAL input
waveform the model hears.

For every processed .pt clip in --processed_dir it reads the clip's WavLM frame
count T (off audio_features) and the relevant wavs, then computes:

  * per-FRAME instantaneous SNR timeline  (T,)  + s1-active mask (T,)
        SNR(i) = 10*log10( sum_frame s1^2 / sum_frame (input - s1)^2 )
        on the 50 Hz WavLM grid (clean_features.snr_timeline_db).
  * optional per-T-F IRM grid (T_p, F_bins)  (clean_features.irm_grid)  with --irm
    (IRM is built from s1 vs the interferer = input - s1).

and writes ONE small per-clip target .pt per filename into --output_dir, plus a
manifest.json {filename: rel_path}. The dataset loads these lazily by filename
(PreprocessedDataset(snr_map_dir=...)), so the dense targets never bloat the main
processed .pt files and the feature is default-OFF (no dir → no targets → no-op).

The frame grid is matched to preprocess.py (WavLM-Large hop 320 @ 16 kHz → 50 Hz),
and the timeline is forced to the clip's exact WavLM T so it aligns frame-for-frame
with audio_features in collate.

TWO MODES
---------
default (mixture-only, LEGACY)
  Interferer = s2 stem directly. Requires --stems_root with s1/ and s2/. Reproduces
  the original behaviour byte-for-byte (s1-vs-s2 speaker SNR over mix_clean clips).

--general (the non-overlap + noise fix, REQUIRED for processed_aug)
  The interferer is derived per clip SUFFIX as (INPUT_AUDIO - s1), where INPUT_AUDIO
  is the ACTUAL waveform the model hears for that .pt:
    * mixture (no suffix):  input = mix_clean/<base>.wav  (= s1+s2)
                            → interferer = s2  → speaker-vs-speaker SNR (legacy match).
    * <base>_s1clean:       input = s1clean_wav_links/<base>_s1clean.wav (== s1)
                            → interferer ≈ 0 → clean (+40 dB everywhere, all-active).
    * <base>_augNp{N}_00:   input = aug_noise_wav/<base>_augNp{N}_00.wav
                            → interferer = WHAM noise → speech-vs-noise SNR (~N dB).
  s1 is ALWAYS the clean target stem (mixture/augNp read it from --stems_root/s1;
  s1clean reads its own input which equals s1). The general path needs NO s2 stem
  (interferer is computed from input - s1), so it works for the noise/clean variants
  the legacy s2-only path cannot.

Per-frame SNR clamp [-30, 40] dB (task spec), s1-active mask (frame s1 energy within
40 dB of the clip's peak s1 frame). Resume-safe via manifest.json.

Wav roots (confirm with the inventory):
  --stems_root      $SHARED/data/Libri2Mix/Libri2Mix/wav16k/min/<split>   (has s1/ s2/ mix_clean/)
  --s1clean_dir     $SHARED/data/s1clean_wav_links                        (<base>_s1clean.wav)
  --aug_noise_dir   $SHARED/data/aug_noise_wav                            (<base>_augNp{N}_00.wav)

Usage (legacy mixture-only):
  python scripts/compute_snr_map_targets.py \
    --processed_dir $SHARED/data/processed/train \
    --stems_root    $SHARED/data/Libri2Mix/Libri2Mix/wav16k/min/train-100 \
    --output_dir    $SHARED/data/snr_map_targets/train   [--irm] [--limit N]

Usage (general, processed_aug 41700: mixture + s1clean + augNp):
  python scripts/compute_snr_map_targets.py --general \
    --processed_dir $SHARED/data/processed_aug/train \
    --stems_root    $SHARED/data/Libri2Mix/Libri2Mix/wav16k/min/train-100 \
    --s1clean_dir   $SHARED/data/s1clean_wav_links \
    --aug_noise_dir $SHARED/data/aug_noise_wav \
    --output_dir    $SHARED/data/snr_map_targets_aug/train   [--limit N]
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

from clean_features import snr_timeline_db, irm_grid  # noqa: E402

WAVLM_HOP = 320          # samples @ 16 kHz → 50 Hz frame grid (preprocess.py)
F_P_DEFAULT = 8          # BEATs frequency-patch count
SNR_CLAMP = (-30.0, 40.0)  # task-spec per-frame clamp

_S1CLEAN_RE = re.compile(r"^(.*)_s1clean$")
_AUGNP_RE = re.compile(r"^(.*)_augNp\d+_\d+$")


def _read_mono(path: str):
    a, sr = sf.read(path, dtype="float32")
    if getattr(a, "ndim", 1) > 1:
        a = a.mean(axis=1)
    return a, sr


def _classify(stem: str):
    """Return (kind, base_stem) where kind in {mixture, s1clean, augNp}."""
    m = _S1CLEAN_RE.match(stem)
    if m:
        return "s1clean", m.group(1)
    m = _AUGNP_RE.match(stem)
    if m:
        return "augNp", m.group(1)
    return "mixture", stem


def _resolve_general(stem, args):
    """For the actual processed_aug .pt stem, return (s1_path, input_path, kind).

    s1_path = the CLEAN target stem (numerator). input_path = the waveform the model
    actually hears (denominator interferer = input - s1). For mixture clips both come
    from the Libri2Mix tree; s1clean/augNp pull the single-speaker / noisy input from
    their dedicated dirs while s1 stays the Libri2Mix s1 stem.
    """
    kind, base = _classify(stem)
    s1_path = os.path.join(args.stems_root, "s1", base + ".wav")
    if kind == "mixture":
        input_path = os.path.join(args.stems_root, "mix_clean", base + ".wav")
    elif kind == "s1clean":
        input_path = os.path.join(args.s1clean_dir, stem + ".wav")
    else:  # augNp
        input_path = os.path.join(args.aug_noise_dir, stem + ".wav")
    return s1_path, input_path, kind


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--processed_dir", required=True,
                    help="dir of processed .pt clips (for the per-clip WavLM frame count T)")
    ap.add_argument("--stems_root", required=True,
                    help="split dir containing s1/ (and mix_clean/, s2/) subdirs")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--general", action="store_true",
                    help="GENERAL per-suffix targets (mixture/s1clean/augNp); interferer = input - s1")
    ap.add_argument("--s1clean_dir", default=None,
                    help="[general] dir of <base>_s1clean.wav inputs (== s1)")
    ap.add_argument("--aug_noise_dir", default=None,
                    help="[general] dir of <base>_augNp{N}_00.wav noisy inputs")
    ap.add_argument("--sr", type=int, default=16000)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--flush_every", type=int, default=200)
    ap.add_argument("--irm", action="store_true",
                    help="also compute the per-T-F IRM grid")
    ap.add_argument("--irm_t_p", type=int, default=0,
                    help="fixed IRM time-patch count; 0 → round(T/20)")
    ap.add_argument("--irm_f_bins", type=int, default=F_P_DEFAULT)
    args = ap.parse_args()

    s1_dir = os.path.join(args.stems_root, "s1")
    s2_dir = os.path.join(args.stems_root, "s2")
    if args.general:
        if not os.path.isdir(s1_dir):
            print(f"ERROR: need s1 dir {s1_dir}", file=sys.stderr)
            return 2
        if not args.s1clean_dir or not os.path.isdir(args.s1clean_dir):
            print(f"ERROR: --general needs --s1clean_dir (got {args.s1clean_dir})", file=sys.stderr)
            return 2
        if not args.aug_noise_dir or not os.path.isdir(args.aug_noise_dir):
            print(f"ERROR: --general needs --aug_noise_dir (got {args.aug_noise_dir})", file=sys.stderr)
            return 2
    else:
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
    kind_counts = {"mixture": 0, "s1clean": 0, "augNp": 0}
    for i, ptname in enumerate(pts):
        cached = torch.load(os.path.join(args.processed_dir, ptname), weights_only=False)
        filename = cached.get("filename", os.path.splitext(ptname)[0] + ".wav")
        stem = os.path.splitext(filename)[0]
        T = int(cached["audio_features"].shape[0])

        if filename in manifest and os.path.exists(
            os.path.join(args.output_dir, manifest[filename])
        ):
            continue

        if args.general:
            s1_path, input_path, kind = _resolve_general(stem, args)
            if not (os.path.exists(s1_path) and os.path.exists(input_path)):
                if n_missing < 20:
                    print(f"  [WARNING] missing wav ({kind}) s1={os.path.exists(s1_path)} "
                          f"input={os.path.exists(input_path)} :: {stem}", flush=True)
                n_missing += 1
                continue
            try:
                s1, sr = _read_mono(s1_path)
                inp, _ = _read_mono(input_path)
            except Exception as e:  # noqa: BLE001
                print(f"  [WARNING] read failed {filename}: {e}")
                continue
            n = min(len(s1), len(inp))
            interferer = (inp[:n].astype(np.float64) - s1[:n].astype(np.float64))
            s1 = s1[:n]
        else:
            kind = "mixture"
            s1_path = os.path.join(s1_dir, filename)
            s2_path = os.path.join(s2_dir, filename)
            if not (os.path.exists(s1_path) and os.path.exists(s2_path)):
                n_missing += 1
                continue
            try:
                s1, sr = _read_mono(s1_path)
                interferer, _ = _read_mono(s2_path)
            except Exception as e:  # noqa: BLE001
                print(f"  [WARNING] read failed {filename}: {e}")
                continue

        timeline, active = snr_timeline_db(
            s1, interferer, sr=sr, hop=WAVLM_HOP, n_frames=T,
        )
        timeline = np.clip(np.asarray(timeline, dtype=np.float32), SNR_CLAMP[0], SNR_CLAMP[1])
        rec = {
            "filename": filename,
            "snr_map_target": torch.from_numpy(np.asarray(timeline, dtype=np.float32)),  # (T,)
            "snr_map_mask": torch.from_numpy(np.asarray(active, dtype=np.float32)),       # (T,)
        }
        if args.irm:
            t_p = args.irm_t_p or max(1, int(round(T / 20.0)))
            irm = irm_grid(s1, interferer, sr=sr, t_p=t_p, f_bins=args.irm_f_bins)
            rec["snr_irm_target"] = torch.from_numpy(np.asarray(irm, dtype=np.float32))   # (T_p, F)

        rel = f"{stem}.pt"
        torch.save(rec, os.path.join(args.output_dir, rel))
        manifest[filename] = rel
        n_done += 1
        kind_counts[kind] = kind_counts.get(kind, 0) + 1
        if n_done % args.flush_every == 0:
            tmp = manifest_path + ".tmp"
            json.dump(manifest, open(tmp, "w"))
            os.replace(tmp, manifest_path)
            print(f"  {n_done} computed ({i+1}/{len(pts)} scanned) "
                  f"mix={kind_counts['mixture']} s1clean={kind_counts['s1clean']} "
                  f"augNp={kind_counts['augNp']}", flush=True)

    tmp = manifest_path + ".tmp"
    json.dump(manifest, open(tmp, "w"))
    os.replace(tmp, manifest_path)
    print(f"done: {len(manifest)} dense SNR-map targets -> {args.output_dir}  "
          f"(computed {n_done}, missing-wav {n_missing}, "
          f"mix={kind_counts['mixture']} s1clean={kind_counts['s1clean']} augNp={kind_counts['augNp']}, "
          f"irm={'on' if args.irm else 'off'}, mode={'general' if args.general else 'legacy'})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
