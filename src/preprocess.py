"""
Preprocess audio files: extract WavLM features + Pyannote overlap info.

Usage:
    # Dataset with official splits:
    python src/preprocess.py --audio_dir ./data/raw/ami/train --output_dir ./data/processed/train
    python src/preprocess.py --audio_dir ./data/raw/ami/val   --output_dir ./data/processed/val
    python src/preprocess.py --audio_dir ./data/raw/ami/test  --output_dir ./data/processed/test

    # Dataset without splits (train.py splits automatically):
    python src/preprocess.py --audio_dir ./data/raw/all --output_dir ./data/processed
"""

import argparse
import os

import torch
import torchaudio
from transformers import WavLMModel

from feature_extractor import extract_overlap_info, load_overlap_pipeline


def preprocess(audio_dir: str, output_dir: str, sample_rate: int = 16000):
    os.makedirs(output_dir, exist_ok=True)

    # Load frozen models once
    device = torch.device(
        "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    )

    wavlm = WavLMModel.from_pretrained("microsoft/wavlm-large").to(device).eval()  # type: ignore
    for p in wavlm.parameters():
        p.requires_grad = False

    pipeline = load_overlap_pipeline()

    # Process each audio file
    audio_files = sorted([file for file in os.listdir(audio_dir) if file.endswith((".wav", ".flac", ".mp3"))])

    print(f"Processing {len(audio_files)} files........")

    for i, filename in enumerate(audio_files):
        wav_path = os.path.join(audio_dir, filename)
        stem = os.path.splitext(filename)[0]

        # Load and resample to target sample rate
        waveform, sr = torchaudio.load(wav_path)
        if sr != sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, sample_rate)
        waveform = waveform.mean(dim=0)  # wavlm expects mono

        # WavLM features
        with torch.no_grad():
            audio_features = waveform.unsqueeze(0).to(device)
            audio_features = wavlm(audio_features).last_hidden_state.cpu()
            audio_features = audio_features.squeeze(0)  # (T, 1024)

        # Pyannote overlap
        overlap_info, segments = extract_overlap_info(wav_path, pipeline, sample_rate)

        # Align overlap_info length to WavLM output
        T = audio_features.shape[0]
        overlap_info = overlap_info[:T]
        if overlap_info.shape[0] < T:
            pad = torch.zeros(T - overlap_info.shape[0], overlap_info.shape[1])
            overlap_info = torch.cat([overlap_info, pad], dim=0)

        # Save
        torch.save(
            {
                "audio_features": audio_features,  # (T, 1024)
                "overlap_info": overlap_info,       # (T, 2)
                "overlap_segments": segments,
                "filename": filename,
            },
            os.path.join(output_dir, f"{stem}.pt"),
        )

        if (i + 1) % 100 == 0:
            print(f"\t{i + 1}/{len(audio_files)} done")

    print(f"Done. Saved {len(audio_files)} files to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./data/processed")
    args = parser.parse_args()

    preprocess(args.audio_dir, args.output_dir)
