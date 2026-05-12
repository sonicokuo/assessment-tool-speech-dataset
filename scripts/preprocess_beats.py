"""Cache BEATs patch embeddings into the per-clip .pt files.

The section-query path in src/train.py reads `beats_patches` directly from
the dataset's .pt files (cached path, `beats_cached: true` in config).
This script populates that key for every clip in a given split:

  for each .pt file in pt_dir:
    - look up the corresponding .wav in audio_dir (by filename stem)
    - load mono 16 kHz waveform
    - run BEATs once
    - write `beats_patches` and `beats_grid_meta` into the .pt file

Run once per split on PSC after WavLM preprocessing is done. Idempotent —
clips that already have `beats_patches` are skipped unless `--overwrite` is set.

Usage:
    python scripts/preprocess_beats.py \\
        --audio_dir /ocean/projects/cis260125p/shared/data/Libri2Mix/wav16k/min/train-100/mix_clean \\
        --pt_dir    /ocean/projects/cis260125p/shared/data/processed_pyannote/train \\
        --checkpoint_name BEATs_iter3_plus_AS2M.pt    # on lpepino/beats_ckpts HF repo

Expected time: ~3-4 hours for 13k clips on a single A100, ~12-16 hours on CPU.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import soundfile as sf
import torch

# Add src/ to path so we can import spec_encoder.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from spec_encoder import SpecEncoder  # noqa: E402


SAMPLE_RATE = 16000


def find_waveform_path(audio_dir: str, stem: str) -> str | None:
    """Locate the .wav (or .flac) corresponding to a .pt's stem."""
    for ext in (".wav", ".flac"):
        path = os.path.join(audio_dir, stem + ext)
        if os.path.exists(path):
            return path
    return None


def load_waveform(path: str) -> torch.Tensor:
    """Load a mono 16 kHz waveform as a float32 tensor of shape (n_samples,)."""
    audio, sr = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=-1)
    if sr != SAMPLE_RATE:
        raise ValueError(
            f"{path}: expected {SAMPLE_RATE} Hz, got {sr} Hz. "
            f"Resample upstream — BEATs is fixed to 16 kHz mono."
        )
    return torch.from_numpy(audio)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio_dir", required=True,
                        help="Directory of source .wav files (e.g., Libri2Mix mix_clean).")
    parser.add_argument("--pt_dir", required=True,
                        help="Directory of preprocessed .pt files (output of src/preprocess.py).")
    parser.add_argument("--checkpoint_name", default="BEATs_iter3_plus_AS2M.pt",
                        help="BEATs checkpoint filename on lpepino/beats_ckpts. "
                             "Use BEATs_iter3.pt for the SSL-only variant.")
    parser.add_argument("--checkpoint_path", default=None,
                        help="Local path to a BEATs checkpoint (skip HF Hub download).")
    parser.add_argument("--overwrite", action="store_true",
                        help="Re-encode clips that already have beats_patches.")
    parser.add_argument("--device", default=None,
                        help="cuda / cpu / mps. Default: cuda if available, else cpu.")
    parser.add_argument("--limit", type=int, default=0,
                        help="Process at most N files (for smoke-testing).")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[preprocess_beats] device={device}")
    print(f"[preprocess_beats] audio_dir={args.audio_dir}")
    print(f"[preprocess_beats] pt_dir={args.pt_dir}")

    # Load BEATs once.
    encoder = SpecEncoder(
        model_name="beats",
        checkpoint_name=args.checkpoint_name,
        checkpoint_path=args.checkpoint_path,
        freeze=True,
    ).to(device)
    print(f"[preprocess_beats] BEATs loaded; d_out={encoder.d_out}, freq_dim={encoder._freq_dim}")

    pt_files = sorted([f for f in os.listdir(args.pt_dir) if f.endswith(".pt")])
    if args.limit:
        pt_files = pt_files[: args.limit]
    print(f"[preprocess_beats] {len(pt_files)} .pt files to scan")

    n_processed = n_skipped = n_missing = 0
    t_start = time.time()

    for i, pt_name in enumerate(pt_files):
        pt_path = os.path.join(args.pt_dir, pt_name)
        cached = torch.load(pt_path, weights_only=False)

        if "beats_patches" in cached and not args.overwrite:
            n_skipped += 1
            continue

        stem = os.path.splitext(pt_name)[0]
        # Some .pt files store filename with extension; honour it when present.
        wav_stem = os.path.splitext(cached.get("filename", pt_name))[0]
        wav_path = find_waveform_path(args.audio_dir, wav_stem)
        if wav_path is None:
            print(f"  [WARN] no .wav found for {wav_stem} in {args.audio_dir}; skipping")
            n_missing += 1
            continue

        try:
            waveform = load_waveform(wav_path).unsqueeze(0)  # (1, n_samples)
            patches, grid = encoder(waveform.to(device))
            # Store as (n_patches, d_patch) — collate_fn stacks across batch.
            cached["beats_patches"] = patches.squeeze(0).detach().cpu()
            cached["beats_grid_meta"] = {
                "n_patches": grid.n_patches,
                "d_patch": grid.d_patch,
                "time_dim": grid.time_dim,
                "freq_dim": grid.freq_dim,
                "backend": grid.backend,
            }
            # Atomic write: tmp + rename so a SIGKILL mid-process doesn't corrupt the .pt.
            tmp = pt_path + ".tmp"
            torch.save(cached, tmp)
            os.replace(tmp, pt_path)
            n_processed += 1
        except Exception as e:
            print(f"  [ERROR] {pt_name}: {type(e).__name__}: {e}")
            n_missing += 1
            continue

        if (i + 1) % 100 == 0:
            elapsed = time.time() - t_start
            rate = (n_processed + n_skipped) / elapsed
            remaining = (len(pt_files) - i - 1) / rate if rate > 0 else float("inf")
            print(f"  [{i+1}/{len(pt_files)}] processed={n_processed} skipped={n_skipped} "
                  f"missing={n_missing} rate={rate:.1f}/s ETA={remaining/60:.1f}min")

    elapsed = time.time() - t_start
    print(f"\n[preprocess_beats] done in {elapsed/60:.1f} min")
    print(f"  processed: {n_processed}")
    print(f"  skipped (already had patches): {n_skipped}")
    print(f"  missing / errored:             {n_missing}")


if __name__ == "__main__":
    main()
