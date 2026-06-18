"""Regression test for collate_fn beats_patches handling.

Bug: collate_fn decided whether to gather `beats_patches` from ONLY batch[0].
A shuffled batch that mixes clips WITH cached BEATs (original Libri2Mix .pt) and
clips WITHOUT (augmented clips from preprocess.py) would KeyError when batch[0]
happened to have beats but a later item did not. Fix: gather beats only when ALL
items have them (untagged runs don't need beats anyway).
"""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from dataset import collate_fn  # noqa: E402


def _item(beats=False):
    it = {
        "audio_features": torch.randn(10, 1024),
        "overlap_info": torch.zeros(10, 4),
        "target_text": "the snr is 10 dB",
        "filename": "clip.wav",
        "overlap_segments": [],
    }
    if beats:
        it["beats_patches"] = torch.randn(5, 768)
    return it


def test_collate_mixed_beats_does_not_crash():
    # first item HAS beats, second does NOT — the exact crash configuration.
    out = collate_fn([_item(beats=True), _item(beats=False)])
    assert "beats_patches" not in out  # skipped because not all items have them


def test_collate_no_beats_skips():
    out = collate_fn([_item(beats=False), _item(beats=False)])
    assert "beats_patches" not in out


def test_collate_all_beats_collated():
    out = collate_fn([_item(beats=True), _item(beats=True)])
    assert "beats_patches" in out
    assert out["beats_patches"].shape[0] == 2
    assert "beats_patches_mask" in out


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
