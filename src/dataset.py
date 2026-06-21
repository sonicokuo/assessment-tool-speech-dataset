"""Shared dataset and collate utilities for the speech quality pipeline."""

import csv
import json
import os
import sys

import torch
from torch.utils.data import Dataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from feature_set import N_FEATURES, build_nums_target, extract_scalars


class PreprocessedDataset(Dataset):
    """Loads pre-computed WavLM features + overlap info from .pt files.

    Optionally loads ground-truth scalar features and the bare-numbers target string
    from a feature CSV (for B-full multi-task training and the auxiliary regression head).
    """

    def __init__(
        self,
        data_dir: str,
        descriptions_path: str | None = None,
        features_csv: str | None = None,
    ):
        self.data_dir = data_dir
        self.files = sorted([f for f in os.listdir(data_dir) if f.endswith(".pt")])

        self.descriptions = None
        if descriptions_path and os.path.exists(descriptions_path):
            with open(descriptions_path) as f:
                self.descriptions = json.load(f)

        # features_csv → per-clip GT scalars and bare-numbers target string for B-full.
        # Map filename → CSV row dict. Keys missing from the map fall back to
        # "all-zero scalars + all-False mask + empty nums target", and the loss code
        # in train.py automatically skips contributions from clips with no scalars.
        self.feature_csv_map: dict[str, dict] = {}
        if features_csv and os.path.exists(features_csv):
            with open(features_csv) as f:
                for row in csv.DictReader(f):
                    self.feature_csv_map[row["filename"]] = row

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        cached = torch.load(
            os.path.join(self.data_dir, self.files[idx]),
            weights_only=False,
        )
        stem = os.path.splitext(self.files[idx])[0]
        filename = cached.get("filename", self.files[idx])

        result = {
            "audio_features": cached["audio_features"],
            "overlap_info": cached["overlap_info"],
            "filename": filename,
            "overlap_segments": cached.get("overlap_segments", []),
        }

        # Pass through BEATs patch embeddings if they were precomputed in this .pt
        # (added by scripts/preprocess_beats.py for the EMNLP rework's section path).
        # Absent on legacy .pt files — train.py falls back to online encoding or skips
        # the section-query path depending on config.
        if "beats_patches" in cached:
            result["beats_patches"] = cached["beats_patches"]

        if self.descriptions and stem in self.descriptions:
            result["target_text"] = self.descriptions[stem]

        # B-full / aux-head scalars (only when features_csv was provided)
        if self.feature_csv_map:
            row = self.feature_csv_map.get(filename)
            if row is not None:
                scalars, mask = extract_scalars(row)
                result["gt_scalars"] = scalars              # (13,)
                result["gt_mask"] = mask                    # (13,) bool
                result["target_nums"] = build_nums_target(row)
            else:
                # Filename not in CSV — emit zero scalars + all-False mask so the loss
                # contributes nothing for this clip. Empty nums target → train.py skips.
                result["gt_scalars"] = torch.zeros(N_FEATURES, dtype=torch.float32)
                result["gt_mask"] = torch.zeros(N_FEATURES, dtype=torch.bool)
                result["target_nums"] = ""

        return result


def collate_fn(batch):
    """Pad variable-length audio features and overlap info to the longest in the batch.

    If samples carry gt_scalars / gt_mask / target_nums (B-full path), stack/collect them.
    """
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

    out = {
        "audio_features": audio_padded,
        "overlap_info": overlap_padded,
        "target_text": target_text,
        # Per-clip UNPADDED WavLM frame counts → clip duration (n_frames / 50 Hz)
        # for the overlap-map supervision time-mask; cheap and back-compatible.
        "audio_lens": torch.tensor([f.shape[0] for f in audio_features], dtype=torch.long),
    }

    # Oracle overlap spans [(start_s, end_s), ...] in SECONDS, one list per clip.
    # Carried as a plain list (variable length, NOT stacked) so the overlap-map
    # supervision in decoupled_grounding_loss_term can build a per-clip time target.
    # Present on every item (defaults to [] in the dataset), so always emitted.
    if "overlap_segments" in batch[0]:
        out["overlap_segments"] = [item.get("overlap_segments", []) for item in batch]

    # BEATs patches — variable patch count per clip because Libri2Mix durations
    # vary. Zero-pad to the batch max and emit a (B, P_max) bool mask that's
    # True at padded positions; SectionQueryHead's cross-attention uses that
    # mask to fill those positions with -inf before the softmax, so attention
    # never lands on padding.
    if all("beats_patches" in item for item in batch):
        patches_list = [item["beats_patches"] for item in batch]
        d_patch = patches_list[0].shape[1]
        max_len = max(p.shape[0] for p in patches_list)
        patches_padded = torch.zeros(B, max_len, d_patch, dtype=patches_list[0].dtype)
        patches_mask = torch.zeros(B, max_len, dtype=torch.bool)
        for i, p in enumerate(patches_list):
            L = p.shape[0]
            patches_padded[i, :L] = p
            patches_mask[i, L:] = True   # True = padded (excluded from softmax)
        out["beats_patches"] = patches_padded
        out["beats_patches_mask"] = patches_mask

    # B-full extras — present only when the dataset was constructed with features_csv.
    if "gt_scalars" in batch[0]:
        out["gt_scalars"] = torch.stack([item["gt_scalars"] for item in batch], dim=0)
        out["gt_mask"] = torch.stack([item["gt_mask"] for item in batch], dim=0)
        out["target_nums"] = [item["target_nums"] for item in batch]

    return out
