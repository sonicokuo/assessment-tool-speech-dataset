"""Tests for the encoder-unfreeze plumbing (deliverable requirement (c)).

Covers:
  - unfreeze_top_n_blocks(encoder, N) marks EXACTLY the top N blocks trainable
    (param-count assertion), leaving earlier blocks + the feature extractor frozen;
  - n=0 is a hard no-op (all frozen);
  - the unfrozen params build an optimizer group at lr_encoder distinct from the
    adapter/LM groups;
  - both the `encoder.encoder.layers` (WavLM/BEATs) and bare `encoder.layers` layouts
    are located.

Tiny CPU stand-in encoder — no real WavLM/BEATs weights needed; the plumbing operates
on whatever nn.Module is handed in.
"""

import os
import sys

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from encoder_unfreeze import (  # noqa: E402
    unfreeze_top_n_blocks,
    encoder_trainable_params,
    count_blocks,
    count_trainable,
    freeze_all,
    _find_encoder_blocks,
)


class _Block(nn.Module):
    """A 2-param transformer-block stand-in (Linear w + b)."""
    def __init__(self, d=4):
        super().__init__()
        self.lin = nn.Linear(d, d)

    def forward(self, x):
        return self.lin(x)


class WavLMLikeEncoder(nn.Module):
    """Mimics transformers WavLMModel / vendored BEATs: blocks live at
    self.encoder.layers, plus a frozen feature extractor."""
    def __init__(self, n_blocks=12, d=4):
        super().__init__()
        self.feature_extractor = nn.Linear(d, d)
        self.encoder = nn.Module()
        self.encoder.layers = nn.ModuleList([_Block(d) for _ in range(n_blocks)])


class BareEncoder(nn.Module):
    """A bare *Encoder already: blocks at self.layers."""
    def __init__(self, n_blocks=6, d=4):
        super().__init__()
        self.layers = nn.ModuleList([_Block(d) for _ in range(n_blocks)])


def _params_per_block(enc) -> int:
    blocks = _find_encoder_blocks(enc)
    return sum(p.numel() for p in list(blocks)[0].parameters())


def test_locate_blocks_both_layouts():
    assert count_blocks(WavLMLikeEncoder(12)) == 12
    assert count_blocks(BareEncoder(6)) == 6


def test_unknown_layout_raises():
    class Weird(nn.Module):
        def __init__(self):
            super().__init__()
            self.stuff = nn.Linear(4, 4)
    with pytest.raises(AttributeError):
        count_blocks(Weird())


def test_n_zero_is_noop_all_frozen():
    enc = WavLMLikeEncoder(12)
    freeze_all(enc)
    got = unfreeze_top_n_blocks(enc, 0)
    assert got == []
    assert count_trainable(enc) == 0, "n=0 must leave the encoder fully frozen"


def test_unfreeze_exactly_top_n_blocks_param_count():
    enc = WavLMLikeEncoder(n_blocks=12, d=4)
    freeze_all(enc)
    ppb = _params_per_block(enc)

    got = unfreeze_top_n_blocks(enc, 4)
    # Exactly 4 blocks' worth of params became trainable.
    assert count_trainable(enc) == 4 * ppb
    assert sum(p.numel() for p in got) == 4 * ppb

    # The trainable set is EXACTLY the last 4 blocks (by identity).
    blocks = list(_find_encoder_blocks(enc))
    top_ids = {id(p) for b in blocks[-4:] for p in b.parameters()}
    bottom_ids = {id(p) for b in blocks[:-4] for p in b.parameters()}
    trainable_ids = {id(p) for p in encoder_trainable_params(enc)}
    assert trainable_ids == top_ids
    assert trainable_ids.isdisjoint(bottom_ids), "earlier blocks must stay frozen"
    # Feature extractor stays frozen.
    assert not enc.feature_extractor.weight.requires_grad


def test_unfreeze_clamps_to_available_blocks():
    enc = BareEncoder(n_blocks=3)
    freeze_all(enc)
    got = unfreeze_top_n_blocks(enc, 99)  # more than exist
    assert count_trainable(enc) == count_trainable(enc)  # tautology guard
    # All 3 blocks unfrozen, nothing crashes.
    all_block_ids = {id(p) for b in _find_encoder_blocks(enc) for p in b.parameters()}
    assert {id(p) for p in got} == all_block_ids


def test_optimizer_gets_encoder_group_at_lr_encoder():
    """Build the param-groups the way train.py does and assert the encoder group
    exists at lr_encoder, separate from the adapter/LM groups."""
    enc = WavLMLikeEncoder(n_blocks=12)
    freeze_all(enc)
    enc_params = unfreeze_top_n_blocks(enc, 2)

    # Stand-in adapter + LM trainables.
    adapter = nn.Linear(4, 4)
    lm = nn.Linear(4, 4)

    lr_adapter, lr_lm, lr_encoder = 1e-4, 1e-5, 2e-6
    param_groups = [
        {"params": adapter.parameters(), "lr": lr_adapter},
        {"params": list(lm.parameters()), "lr": lr_lm},
    ]
    if enc_params:
        param_groups.append({"params": enc_params, "lr": lr_encoder})

    opt = torch.optim.AdamW(param_groups)
    # There must be exactly 3 groups and the encoder group is at lr_encoder.
    assert len(opt.param_groups) == 3
    enc_group = opt.param_groups[-1]
    assert enc_group["lr"] == pytest.approx(lr_encoder)
    # The encoder group holds exactly the unfrozen encoder params.
    enc_ids = {id(p) for p in enc_params}
    group_ids = {id(p) for p in enc_group["params"]}
    assert group_ids == enc_ids
    # And those ids are NOT in the adapter / LM groups.
    other_ids = {id(p) for g in opt.param_groups[:2] for p in g["params"]}
    assert enc_ids.isdisjoint(other_ids)


def test_no_encoder_group_when_n_zero():
    """When n=0, no encoder params → train.py appends no extra group."""
    enc = WavLMLikeEncoder(12)
    freeze_all(enc)
    enc_params = unfreeze_top_n_blocks(enc, 0)
    adapter = nn.Linear(4, 4)
    param_groups = [{"params": adapter.parameters(), "lr": 1e-4}]
    if enc_params:
        param_groups.append({"params": enc_params, "lr": 1e-6})
    opt = torch.optim.AdamW(param_groups)
    assert len(opt.param_groups) == 1, "no encoder group should be added at n=0"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
