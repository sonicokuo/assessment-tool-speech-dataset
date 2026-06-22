"""Encoder-unfreeze plumbing: selectively fine-tune the top-N transformer blocks of
a frozen SSL audio frontend (WavLM or BEATs).

Motivation (capacity experiment, Q4): the production path runs WavLM (or BEATs) FROZEN
and caches its features, so the adapter only ever sees a fixed representation. Several
SSL-for-quality results show that unfreezing the top encoder blocks and fine-tuning
them with a SMALL learning rate improves downstream quality prediction (UTMOS,
arXiv:2204.02152; wav2vec2 quality assessment, arXiv:2204.02135). This module makes
that an additive, config-gated capability without disturbing the default frozen path.

Design:
  - `unfreeze_top_n_blocks(encoder, n)` flips requires_grad=True on exactly the top N
    transformer blocks (and only those) of the encoder, leaving every earlier block,
    the feature extractor / patch embedding, and projections frozen. n=0 is a hard
    no-op: nothing is touched, the encoder stays fully frozen and byte-identical to
    the default.
  - `encoder_trainable_params(encoder)` returns the now-trainable params so train.py
    can put them in their OWN optimizer param-group at `lr_encoder` (a small LR,
    distinct from the adapter / LM-LoRA groups).
  - `count_trainable(...)` / `count_blocks(...)` are tiny helpers for the param-count
    assertions in the tests and the startup summary.

Both encoders expose their transformer stack as a `nn.ModuleList` at
`model.encoder.layers` (verified: transformers WavLMEncoder.layers and the vendored
BEATs TransformerEncoder.layers). We locate that list generically so the same code
unfreezes either backbone.

IMPORTANT: nothing here imports torch's heavy model classes or downloads weights — it
operates on whatever nn.Module is handed in, so it is unit-testable on CPU with a tiny
stand-in encoder.

GPU-memory note: unfreezing top-N blocks adds N blocks' worth of trainable params AND
their activation/gradient memory and an optimizer-state slot per param. WavLM-Large
blocks are ~12.6M params each; unfreezing 4 adds ~50M trainable params plus their Adam
moments (~2x) and the encoder-forward activations must now be retained for backprop
(they are not, in the frozen/cached path). Expect a multi-GB step-memory increase per
unfrozen block; budget for it (smaller batch / more grad-accum) before enabling.
"""

from __future__ import annotations

import torch.nn as nn


def _find_encoder_blocks(encoder: nn.Module) -> nn.ModuleList:
    """Return the ModuleList of transformer blocks for a WavLM/BEATs-style encoder.

    Tries, in order:
      1. encoder.encoder.layers   (transformers WavLMModel, vendored BEATs model)
      2. encoder.layers           (a bare *Encoder already, or a test stand-in)

    Raises a clear error if neither exists so a backbone with a different layout fails
    loudly at setup rather than silently unfreezing nothing.
    """
    inner = getattr(encoder, "encoder", None)
    if inner is not None and isinstance(getattr(inner, "layers", None), nn.ModuleList):
        return inner.layers
    if isinstance(getattr(encoder, "layers", None), nn.ModuleList):
        return encoder.layers
    raise AttributeError(
        "Could not locate the transformer block ModuleList on the encoder. "
        "Expected `encoder.encoder.layers` (WavLM/BEATs) or `encoder.layers`. "
        f"Got encoder of type {type(encoder).__name__}."
    )


def count_blocks(encoder: nn.Module) -> int:
    """Number of transformer blocks in the encoder's stack."""
    return len(_find_encoder_blocks(encoder))


def freeze_all(encoder: nn.Module) -> None:
    """Set requires_grad=False on every encoder parameter (the default state)."""
    for p in encoder.parameters():
        p.requires_grad_(False)


def unfreeze_top_n_blocks(encoder: nn.Module, n: int) -> list[nn.Parameter]:
    """Unfreeze exactly the top `n` transformer blocks of `encoder`.

    "Top" = the last N entries of the block ModuleList (closest to the output), which
    are the most task-specific blocks and the standard choice for partial SSL
    fine-tuning. Everything else (earlier blocks, feature extractor / patch embedding,
    layer norms, projections) is left frozen.

    Args:
        encoder: the frontend module (WavLM / BEATs / a stand-in with .encoder.layers
                 or .layers).
        n:       number of top blocks to unfreeze. n <= 0 is a hard no-op (nothing is
                 touched). n is clamped to the number of available blocks.

    Returns:
        The list of nn.Parameters that were flipped to requires_grad=True (empty when
        n <= 0). train.py puts these in the `lr_encoder` optimizer group.
    """
    if n is None or n <= 0:
        return []

    blocks = _find_encoder_blocks(encoder)
    n_blocks = len(blocks)
    n_eff = min(int(n), n_blocks)

    trainable: list[nn.Parameter] = []
    # Top N = the last n_eff blocks.
    for block in list(blocks)[n_blocks - n_eff :]:
        for p in block.parameters():
            p.requires_grad_(True)
            trainable.append(p)
    return trainable


def encoder_trainable_params(encoder: nn.Module) -> list[nn.Parameter]:
    """Every encoder parameter currently requiring grad (after an unfreeze call).

    train.py calls this to build the encoder optimizer group. Returns a fresh list so
    the caller can dedupe by id() against the adapter / LM groups.
    """
    return [p for p in encoder.parameters() if p.requires_grad]


def count_trainable(encoder: nn.Module) -> int:
    """Total number of trainable (requires_grad) scalar parameters in the encoder."""
    return sum(p.numel() for p in encoder.parameters() if p.requires_grad)
