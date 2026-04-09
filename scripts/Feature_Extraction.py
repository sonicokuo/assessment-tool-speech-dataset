"""
Speech Feature Extractor - Step 1 of Speech Quality Assessment Pipeline
========================================================================
Extracts the following features from audio files:
  - duration (seconds)
  - sample_rate (Hz)
  - overlap_ratio  (via pyannote-audio)
  - snr_db         (signal-to-noise ratio, estimated via waveform energy)
  - silence_ratio  (fraction of frames below energy threshold)

Output: features.csv

Dependencies:
    pip install torchaudio pyannote.audio pandas numpy torch

Usage:
    python feature_extractor.py --audio_dir ./audio_samples --output features.csv
"""

import os
import argparse
import warnings
import numpy as np
import pandas as pd
import torch
import torchaudio

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# 1. Duration & Sample Rate
# ─────────────────────────────────────────────

def get_duration_and_sr(wav_path: str) -> dict:
    """Return duration in seconds and sample rate."""
    info = torchaudio.info(wav_path)
    duration = info.num_frames / info.sample_rate
    return {
        "duration_sec": round(duration, 3),
        "sample_rate_hz": info.sample_rate,
    }


# ─────────────────────────────────────────────
# 2. SNR Estimation (waveform-based, no reference)
# ─────────────────────────────────────────────

def estimate_snr(waveform: torch.Tensor, frame_len: int = 2048, hop_len: int = 512) -> float:
    """
    Estimate SNR using the percentile method:
      - Top 10% energy frames  → signal power estimate
      - Bottom 10% energy frames → noise power estimate
    Returns SNR in dB. Returns NaN if audio is silent.
    """
    waveform = waveform.mean(dim=0)  # mono
    frames = waveform.unfold(0, frame_len, hop_len)
    energies = (frames ** 2).mean(dim=1).numpy()

    if energies.max() < 1e-10:
        return float("nan")  # silent file

    noise_power  = np.percentile(energies, 10)
    signal_power = np.percentile(energies, 90)

    if noise_power < 1e-10:
        noise_power = 1e-10  # avoid log(0)

    snr_db = 10 * np.log10(signal_power / noise_power)
    return round(float(snr_db), 2)


# ─────────────────────────────────────────────
# 3. Silence Ratio
# ─────────────────────────────────────────────

def compute_silence_ratio(
    waveform: torch.Tensor,
    sr: int,
    frame_len_ms: int = 30,
    threshold_db: float = -40.0,
) -> float:
    """
    Fraction of 30 ms frames whose RMS energy is below threshold_db.
    """
    waveform = waveform.mean(dim=0)
    frame_len = int(sr * frame_len_ms / 1000)
    if frame_len == 0 or waveform.numel() < frame_len:
        return float("nan")

    frames = waveform.unfold(0, frame_len, frame_len)
    rms = (frames ** 2).mean(dim=1).sqrt()
    ref = rms.max().item()
    if ref < 1e-10:
        return 1.0  # fully silent

    rms_db = 20 * torch.log10(rms / ref + 1e-10)
    silence_ratio = (rms_db < threshold_db).float().mean().item()
    return round(silence_ratio, 4)


# ─────────────────────────────────────────────
# 4. Overlap Speech Ratio  (via pyannote)
# ─────────────────────────────────────────────

def load_overlap_pipeline():
    """
    Load pyannote overlapped-speech-detection pipeline.
    Requires a Hugging Face token with access to pyannote models.
    Set env var: HF_TOKEN=your_token
    Or pass --hf_token on the command line.
    """
    try:
        from pyannote.audio import Pipeline
        hf_token = os.environ.get("HF_TOKEN", None)
        pipeline = Pipeline.from_pretrained(
            "pyannote/overlapped-speech-detection",
            use_auth_token=hf_token,
        )
        return pipeline
    except Exception as e:
        print(f"[WARNING] Could not load pyannote pipeline: {e}")
        print("          overlap_ratio will be set to NaN.")
        return None


def compute_overlap_ratio(wav_path: str, pipeline, duration_sec: float) -> float:
    """
    Run pyannote overlapped-speech-detection; return fraction of audio
    where overlap was detected.
    """
    if pipeline is None or duration_sec == 0:
        return float("nan")
    try:
        output = pipeline(wav_path)
        overlap_duration = sum(
            segment.end - segment.start
            for segment, _, label in output.itertracks(yield_label=True)
            if label == "OVERLAP"
        )
        return round(overlap_duration / duration_sec, 4)
    except Exception as e:
        print(f"[WARNING] Pyannote failed on {wav_path}: {e}")
        return float("nan")


# ─────────────────────────────────────────────
# 5. Main extractor
# ─────────────────────────────────────────────

SUPPORTED_EXTENSIONS = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}


def extract_features(wav_path: str, overlap_pipeline) -> dict:
    """Extract all features for a single audio file."""
    filename = os.path.basename(wav_path)
    print(f"  Processing: {filename}")

    result = {"filename": filename, "filepath": wav_path}

    # --- Duration & SR ---
    try:
        meta = get_duration_and_sr(wav_path)
        result.update(meta)
    except Exception as e:
        print(f"    [ERROR] Could not read file: {e}")
        result.update({"duration_sec": float("nan"), "sample_rate_hz": float("nan")})
        return result

    # --- Load waveform for energy-based features ---
    try:
        waveform, sr = torchaudio.load(wav_path)
    except Exception as e:
        print(f"    [ERROR] Could not load waveform: {e}")
        result.update({
            "snr_db": float("nan"),
            "silence_ratio": float("nan"),
            "overlap_ratio": float("nan"),
        })
        return result

    # --- SNR ---
    result["snr_db"] = estimate_snr(waveform)

    # --- Silence Ratio ---
    result["silence_ratio"] = compute_silence_ratio(waveform, sr)

    # --- Overlap Ratio ---
    result["overlap_ratio"] = compute_overlap_ratio(
        wav_path, overlap_pipeline, result.get("duration_sec", 0)
    )

    return result


# ─────────────────────────────────────────────
# 6. CLI
# ─────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Speech Feature Extractor")
    parser.add_argument(
        "--audio_dir", type=str, default="./audio_samples",
        help="Directory containing audio files"
    )
    parser.add_argument(
        "--output", type=str, default="features.csv",
        help="Output CSV path"
    )
    parser.add_argument(
        "--hf_token", type=str, default=None,
        help="Hugging Face token for pyannote (or set HF_TOKEN env var)"
    )
    parser.add_argument(
        "--no_overlap", action="store_true",
        help="Skip overlap detection (faster, no pyannote needed)"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Set HF token if provided via CLI
    if args.hf_token:
        os.environ["HF_TOKEN"] = args.hf_token

    # Collect audio files
    audio_dir = args.audio_dir
    if not os.path.isdir(audio_dir):
        print(f"[ERROR] Directory not found: {audio_dir}")
        return

    audio_files = sorted([
        os.path.join(audio_dir, f)
        for f in os.listdir(audio_dir)
        if os.path.splitext(f)[1].lower() in SUPPORTED_EXTENSIONS
    ])

    if not audio_files:
        print(f"[ERROR] No audio files found in: {audio_dir}")
        return

    print(f"\nFound {len(audio_files)} audio file(s) in '{audio_dir}'")

    # Load pyannote pipeline (once, shared across files)
    overlap_pipeline = None
    if not args.no_overlap:
        print("Loading pyannote overlap detection pipeline...")
        overlap_pipeline = load_overlap_pipeline()

    # Extract features
    print("\nExtracting features...\n" + "-" * 50)
    records = []
    for wav_path in audio_files:
        record = extract_features(wav_path, overlap_pipeline)
        records.append(record)

    # Save to CSV
    df = pd.DataFrame(records, columns=[
        "filename", "filepath",
        "duration_sec", "sample_rate_hz",
        "snr_db", "silence_ratio", "overlap_ratio",
    ])

    df.to_csv(args.output, index=False)
    print("\n" + "=" * 50)
    print(f"Done! Features saved to: {args.output}")
    print(f"Total files processed: {len(df)}")
    print("\nPreview:")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()