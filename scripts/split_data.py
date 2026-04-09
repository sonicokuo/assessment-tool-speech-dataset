"""
Split audio files into train/val/test directories.

Groups by speaker ID (first part of LibriSpeech filename, e.g. "1089" from "1089-134686-0000.flac")
to prevent data leakage — all clips from the same speaker go into the same split.

Usage:
    python split_data.py --audio_dir ./data/raw --output_dir ./data/raw
    python split_data.py --audio_dir ./data/raw --output_dir ./data/raw --train 0.8 --val 0.1 --test 0.1
"""

import argparse
import os
import random
import shutil
from collections import defaultdict


def get_speaker_id(filename: str) -> str:
    """Extract speaker ID from filename. e.g. '1089-134686-0000.flac' -> '1089'"""
    return filename.split("-")[0]


def split_data(
    audio_dir: str,
    output_dir: str,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> None:
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"

    # Collect audio files
    extensions = {".wav", ".flac", ".mp3", ".ogg"}
    audio_files = sorted([
        f for f in os.listdir(audio_dir)
        if os.path.splitext(f)[1].lower() in extensions
    ])
    assert len(audio_files) > 0, f"No audio files found in {audio_dir}"

    # Group by speaker
    speaker_files = defaultdict(list)
    for f in audio_files:
        speaker_id = get_speaker_id(f)
        speaker_files[speaker_id].append(f)

    speakers = sorted(speaker_files.keys())
    print(f"Found {len(audio_files)} files from {len(speakers)} speakers")

    # Split speakers (not files) to prevent leakage
    random.seed(seed)
    random.shuffle(speakers)

    n = len(speakers)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    train_speakers = speakers[:train_end]
    val_speakers = speakers[train_end:val_end]
    test_speakers = speakers[val_end:]

    splits = {
        "train": train_speakers,
        "val": val_speakers,
        "test": test_speakers,
    }

    # Copy files into split directories
    for split_name, split_speakers in splits.items():
        split_dir = os.path.join(output_dir, split_name)
        os.makedirs(split_dir, exist_ok=True)

        count = 0
        for speaker in split_speakers:
            for f in speaker_files[speaker]:
                src = os.path.join(audio_dir, f)
                dst = os.path.join(split_dir, f)
                shutil.copy2(src, dst)
                count += 1

        print(f"  {split_name}: {len(split_speakers)} speakers, {count} files -> {split_dir}")

    print(f"\nDone. Total: {len(audio_files)} files split into {output_dir}/{{train,val,test}}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio_dir", type=str, required=True, help="Directory with all audio files")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for splits")
    parser.add_argument("--train", type=float, default=0.8)
    parser.add_argument("--val", type=float, default=0.1)
    parser.add_argument("--test", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    split_data(args.audio_dir, args.output_dir, args.train, args.val, args.test, args.seed)
