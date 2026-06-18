#!/usr/bin/env python3
"""
augment_noise.py — controlled-synthesis noise augmentation with EXACT SNR ground truth.

Paper contribution / motivation
--------------------------------
The real Libri2Mix data covers a narrow SNR band (roughly 15-20 dB in practice).
A speech-quality-description model trained only on that band never sees clean or
very noisy recordings, so its SNR claims do not generalize.

This script synthesizes unlimited training data spanning the FULL SNR range by
mixing a clean speech clip with a noise clip at a KNOWN target SNR. The trick:

    if we mix at a target SNR by construction, the SNR ground truth is EXACT
    (it is the injection parameter, not an estimate).

Crucially, *additive* noise (no convolution / no reverb) does NOT change the
clean source's pitch, speaking rate, pauses, jitter/shimmer, SRMR, overlap, etc.
Those features are intrinsic to the clean source signal. So for a noise-augmented
clip:

    snr_db        = the injected target SNR
    all other     = the clean source's measured features, passed through unchanged

This gives us per-clip rows with one exactly-known label (SNR) and a full set of
otherwise-valid SP features, at any SNR we choose.

Mixing math
-----------
Power is mean-square: P(x) = mean(x**2).
Given a target SNR in dB,

    target_snr_db = 10 * log10( P_speech / P_noise_scaled )

We scale the noise by a constant alpha so that P_noise_scaled = alpha**2 * P_noise:

    target_snr_db = 10 * log10( P_speech / (alpha**2 * P_noise) )
    => alpha**2 = P_speech / (P_noise * 10**(target_snr_db/10))
    => alpha    = sqrt( P_speech / (P_noise * 10**(target_snr_db/10)) )

The mixture is  mix = speech + alpha * noise.  By construction
snr_db(speech, alpha*noise) == target_snr_db exactly (up to float precision).

Clipping is handled by peak-normalizing the mixture ONLY when |mix| > 1, which
preserves the SNR exactly (it scales speech and noise by the same factor, so the
*ratio* of powers — and therefore the SNR — is invariant). See `mix_at_snr`.
"""

from __future__ import annotations

import argparse
import copy
import csv
import os
from typing import Dict, List, Optional

import numpy as np

# Name of the SNR column in the feature CSV (see src/feature_extractor_mix.py COLUMN_ORDER).
SNR_COLUMN = "snr_db"
FILENAME_COLUMN = "filename"
# Floor on noise/speech power to avoid divide-by-zero on silent inputs.
_POWER_EPS = 1e-12


# ──────────────────────────────────────────────────────────────────────────────
# Core signal math (pure functions — no IO, no global state)
# ──────────────────────────────────────────────────────────────────────────────
def power(x: np.ndarray) -> float:
    """Signal power = mean of squares. Returns a Python float."""
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return 0.0
    return float(np.mean(x ** 2))


def snr_db(speech: np.ndarray, noise: np.ndarray) -> float:
    """
    SNR in dB between a speech signal and a noise signal:

        10 * log10( mean(speech**2) / mean(noise**2) )

    This is the verification function used by the tests: after `mix_at_snr`
    scales the noise, snr_db(speech, scaled_noise) must equal the target.
    """
    p_s = power(speech)
    p_n = power(noise)
    return 10.0 * np.log10(p_s / max(p_n, _POWER_EPS))


def fit_noise_length(noise: np.ndarray, length: int) -> np.ndarray:
    """
    Tile (if shorter) or crop (if longer) the noise to exactly `length` samples.
    Tiling is deterministic (np.tile from the start), so output is reproducible.
    """
    noise = np.asarray(noise, dtype=np.float64).reshape(-1)
    if noise.size == 0:
        # Degenerate: no noise content. Return zeros so power is 0 (handled by eps).
        return np.zeros(length, dtype=np.float64)
    if noise.size < length:
        reps = int(np.ceil(length / noise.size))
        noise = np.tile(noise, reps)
    return noise[:length].copy()


def noise_scale_for_snr(speech: np.ndarray, noise: np.ndarray, target_snr_db: float) -> float:
    """
    Return alpha such that snr_db(speech, alpha * noise) == target_snr_db.

        alpha = sqrt( P_speech / (P_noise * 10**(target_snr_db/10)) )
    """
    p_s = power(speech)
    p_n = max(power(noise), _POWER_EPS)
    ratio = 10.0 ** (target_snr_db / 10.0)
    return float(np.sqrt(p_s / (p_n * ratio)))


def mix_at_snr(
    speech: np.ndarray,
    noise: np.ndarray,
    target_snr_db: float,
    prevent_clipping: bool = True,
):
    """
    Mix `speech` and `noise` at an exact `target_snr_db` and return the mixture.

    Steps
    -----
    1. Fit the noise to the speech length (tile if shorter, crop if longer).
    2. Scale the noise by alpha so the speech-vs-scaled-noise SNR == target.
    3. mix = speech + alpha * noise.
    4. If `prevent_clipping` and peak(|mix|) > 1, peak-normalize the WHOLE
       mixture by the same factor. This scales speech and noise identically,
       so the power ratio (and hence the SNR) is unchanged — the injected SNR
       remains exact. We only normalize down (never amplify), so quiet mixes
       are left alone.

    Returns
    -------
    mix : np.ndarray (float64), length == len(speech)
        The mixture signal.
    info : dict
        {'alpha': scale applied to noise,
         'scaled_noise': alpha*noise after any clip-norm,
         'speech': speech after any clip-norm,
         'clip_norm': the clip-normalization factor applied (1.0 if none),
         'realized_snr_db': measured SNR of the returned speech vs scaled_noise}
    """
    speech = np.asarray(speech, dtype=np.float64).reshape(-1)
    if speech.size == 0:
        raise ValueError("speech signal is empty")

    noise_fit = fit_noise_length(noise, speech.size)
    alpha = noise_scale_for_snr(speech, noise_fit, target_snr_db)
    scaled_noise = alpha * noise_fit
    mix = speech + scaled_noise

    clip_norm = 1.0
    if prevent_clipping:
        peak = float(np.max(np.abs(mix))) if mix.size else 0.0
        if peak > 1.0:
            clip_norm = 1.0 / peak
            mix = mix * clip_norm
            speech = speech * clip_norm
            scaled_noise = scaled_noise * clip_norm

    info = {
        "alpha": alpha,
        "scaled_noise": scaled_noise,
        "speech": speech,
        "clip_norm": clip_norm,
        "realized_snr_db": snr_db(speech, scaled_noise),
    }
    return mix, info


# ──────────────────────────────────────────────────────────────────────────────
# Exact-ground-truth feature row construction
# ──────────────────────────────────────────────────────────────────────────────
def _suffix_filename(filename: str, target_snr_db: float) -> str:
    """
    Make the augmented filename unique by inserting `_augN<snr>` before the
    extension, e.g. 'clip.wav' + 12.0 dB -> 'clip_augN12.00.wav'. The SNR is
    formatted with sign and 2 decimals so positive/negative variants are
    distinct and filesystem-safe (minus sign is fine; we use 'm' for clarity).
    """
    base, ext = os.path.splitext(filename)
    snr_tag = f"{target_snr_db:+.2f}".replace("+", "p").replace("-", "m").replace(".", "_")
    return f"{base}_augN{snr_tag}{ext}"


def augment_feature_row(clean_row: Dict[str, str], target_snr_db: float) -> Dict[str, str]:
    """
    Return a COPY of `clean_row` (a feature-CSV row dict) representing the
    noise-augmented clip:

      * `snr_db`   overwritten with the injected target (string, 2 decimals).
      * `filename` suffixed with `_augN<snr>` so it is unique.
      * every other column passed through UNCHANGED (additive noise does not
        affect F0, speaking rate, pauses, SRMR, overlap, jitter/shimmer, ...).

    Works on whatever columns the row has (the real CSV carries extra trailing
    columns like overlap_segments_vad beyond COLUMN_ORDER), so it is
    column-agnostic by design.
    """
    row = copy.deepcopy(clean_row)
    row[SNR_COLUMN] = f"{round(float(target_snr_db), 2):.2f}"
    if FILENAME_COLUMN in row and row[FILENAME_COLUMN]:
        row[FILENAME_COLUMN] = _suffix_filename(str(row[FILENAME_COLUMN]), target_snr_db)
    return row


# ──────────────────────────────────────────────────────────────────────────────
# CLI plumbing
# ──────────────────────────────────────────────────────────────────────────────
def _parse_snr_grid(s: str) -> List[float]:
    return [float(tok) for tok in s.split(",") if tok.strip() != ""]


def _load_feature_rows(csv_path: str):
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = [dict(r) for r in reader]
    return fieldnames, rows


def _list_noise_files(noise_dir: str) -> List[str]:
    exts = {".wav", ".flac", ".ogg", ".mp3"}
    files = [
        os.path.join(noise_dir, fn)
        for fn in sorted(os.listdir(noise_dir))
        if os.path.splitext(fn)[1].lower() in exts
    ]
    if not files:
        raise FileNotFoundError(f"no noise audio files found in {noise_dir}")
    return files


def run_cli(args: argparse.Namespace) -> None:
    # soundfile is imported lazily (only the CLI needs wav IO); the pure
    # functions + tests never import it, matching src/preprocess.py's pattern.
    import soundfile as sf

    rng = np.random.default_rng(args.seed)
    snr_grid = _parse_snr_grid(args.snr_grid)
    fieldnames, rows = _load_feature_rows(args.clean_features_csv)
    if SNR_COLUMN not in fieldnames:
        raise ValueError(f"clean CSV missing required column '{SNR_COLUMN}'")

    noise_files = _list_noise_files(args.noise_dir)
    os.makedirs(args.out_wav_dir, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)) or ".", exist_ok=True)

    if args.limit is not None:
        rows = rows[: args.limit]

    out_rows: List[Dict[str, str]] = []
    n_written = 0
    n_skipped = 0

    for clip_idx, clean_row in enumerate(rows):
        clean_name = clean_row.get(FILENAME_COLUMN, "")
        clean_wav_path = os.path.join(args.clean_wav_dir, clean_name)
        if not os.path.isfile(clean_wav_path):
            print(f"  [skip] missing clean wav: {clean_wav_path}")
            n_skipped += 1
            continue

        speech, sr = sf.read(clean_wav_path, dtype="float32")
        if speech.ndim > 1:  # downmix to mono
            speech = speech.mean(axis=1)
        speech = speech.astype(np.float64)

        # Choose per_clip SNR points (without replacement when possible).
        k = min(args.per_clip, len(snr_grid))
        chosen_snr_idx = rng.choice(len(snr_grid), size=k, replace=False)

        for snr_pos, snr_idx in enumerate(chosen_snr_idx):
            target_snr = snr_grid[int(snr_idx)]
            # Deterministic noise selection: index by (clip_idx + snr_idx).
            noise_path = noise_files[(clip_idx + int(snr_idx)) % len(noise_files)]
            noise, n_sr = sf.read(noise_path, dtype="float32")
            if noise.ndim > 1:
                noise = noise.mean(axis=1)
            noise = noise.astype(np.float64)

            mix, info = mix_at_snr(speech, noise, target_snr, prevent_clipping=True)

            new_row = augment_feature_row(clean_row, target_snr)
            out_name = new_row[FILENAME_COLUMN]
            out_path = os.path.join(args.out_wav_dir, out_name)
            # update filepath column if present
            if "filepath" in new_row:
                new_row["filepath"] = out_path
            sf.write(out_path, mix.astype(np.float32), sr)
            out_rows.append(new_row)
            n_written += 1

        if (clip_idx + 1) % 100 == 0:
            print(f"  [{clip_idx + 1}/{len(rows)}] written={n_written} skipped={n_skipped}")

    with open(args.out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in out_rows:
            writer.writerow({k: r.get(k, "") for k in fieldnames})

    print(f"Done. Wrote {n_written} augmented clips ({n_skipped} clips skipped) -> {args.out_csv}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Controlled-synthesis noise augmentation with exact SNR ground truth."
    )
    p.add_argument("--clean_wav_dir", required=True, help="dir of clean source .wav files")
    p.add_argument("--clean_features_csv", required=True, help="feature CSV for the clean clips")
    p.add_argument("--noise_dir", required=True, help="dir of noise .wav files")
    p.add_argument("--out_wav_dir", required=True, help="output dir for augmented mixtures")
    p.add_argument("--out_csv", required=True, help="output augmented features CSV")
    p.add_argument(
        "--snr_grid",
        default="0,5,10,15,20,30",
        help="comma-separated target SNRs in dB (default: 0,5,10,15,20,30)",
    )
    p.add_argument("--per_clip", type=int, default=1, help="random SNR points per clip (default 1)")
    p.add_argument("--limit", type=int, default=None, help="process only the first N clean clips")
    p.add_argument("--seed", type=int, default=1234, help="numpy RNG seed (reproducibility)")
    return p


def main(argv: Optional[List[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    run_cli(args)


if __name__ == "__main__":
    main()
