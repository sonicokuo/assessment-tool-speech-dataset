"""Shared dataset and collate utilities for the speech quality pipeline."""

import json
import os

import torch
from torch.utils.data import Dataset


class PreprocessedDataset(Dataset):
    """Loads pre-computed WavLM features + overlap info from .pt files."""

    def __init__(self, data_dir: str, descriptions_path: str = None):
        self.data_dir = data_dir
        self.files = sorted([f for f in os.listdir(data_dir) if f.endswith(".pt")])

        self.descriptions = None
        if descriptions_path and os.path.exists(descriptions_path):
            with open(descriptions_path) as f:
                self.descriptions = json.load(f)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        cached = torch.load(
            os.path.join(self.data_dir, self.files[idx]),
            weights_only=False,
        )
        stem = os.path.splitext(self.files[idx])[0]

        result = {
            "audio_features": cached["audio_features"],
            "overlap_info": cached["overlap_info"],
            "filename": cached.get("filename", self.files[idx]),
            "overlap_segments": cached.get("overlap_segments", []),
        }

        if self.descriptions and stem in self.descriptions:
            result["target_text"] = self.descriptions[stem]

        return result


def collate_fn(batch):
    """Pad variable-length audio features and overlap info to the longest in the batch."""
    audio_features = [item["audio_features"] for item in batch]
    overlap_info = [item["overlap_info"] for item in batch]
    target_text = [item["target_text"] for item in batch]

    max_len = max(f.shape[0] for f in audio_features)
    B = len(batch)
    audio_dim = audio_features[0].shape[-1]
    overlap_dim = overlap_info[0].shape[-1]

    audio_padded = torch.zeros(B, max_len, audio_dim)
    overlap_padded = torch.zeros(B, max_len, overlap_dim)

    for i, (af, oi) in enumerate(zip(audio_features, overlap_info)):
        audio_padded[i, : af.shape[0]] = af
        overlap_padded[i, : oi.shape[0]] = oi

    return {
        "audio_features": audio_padded,
        "overlap_info": overlap_padded,
        "target_text": target_text,
    }
