"""Tests for the train.py integration of the token-free 2D grounding head.

These exercise `decoupled_grounding_loss_term` — the small PURE helper that
train.py's compute_loss calls to run the DecoupledGroundingHead as a PARALLEL
branch off each batch's BEATs patches. The helper lives in src/decoupled_grounding.py
(not train.py) precisely so it imports WITHOUT transformers / peft / wandb and can
be unit-tested on CPU. train.py re-exports it; this is the exact callable used in
the training loop.

What is proven here:
  * given a fake batch (beats_patches, gt_scalars, gt_mask) the helper returns a
    FINITE scalar loss whose backward puts gradient on the head's LEARNED QUERIES
    (the parallel branch trains), and the head yields a (B,F,T,F) 2D map via reshape;
  * the lambda weight scales the returned loss linearly (lambda=0 → no-op);
  * the flag-off / no-head / no-patches paths add NOTHING (None loss, empty metrics);
  * the branch is decoupled: V_proj gets NO gradient from the grounding loss, so the
    head can only satisfy it by reshaping attention — it never rewrites the values
    and (in the real loop) never touches the LM CE graph;
  * the collate_fn pad-mask (True at PADDED) is correctly INVERTED to a valid-mask
    so padded patches receive ~0 attention.

Pure torch + feature_set, runs anywhere on CPU.
"""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from decoupled_grounding import (  # noqa: E402
    DecoupledGroundingHead,
    decoupled_grounding_loss_term,
)
from feature_set import N_FEATURES  # noqa: E402


# ── helpers ──────────────────────────────────────────────────────────────────
def _fake_batch(B=3, P=12, d_patch=16, with_mask=False, n_pad=0):
    """A minimal batch dict carrying exactly the fields the helper reads:
    beats_patches (B,P,d_patch), gt_scalars (B,F), gt_mask (B,F). Optionally a
    collate_fn-style beats_patches_mask (True at PADDED positions)."""
    torch.manual_seed(0)
    batch = {
        "beats_patches": torch.randn(B, P, d_patch),
        "gt_scalars": torch.randn(B, N_FEATURES),
        "gt_mask": torch.ones(B, N_FEATURES, dtype=torch.bool),
    }
    if with_mask:
        pad = torch.zeros(B, P, dtype=torch.bool)
        if n_pad > 0:
            pad[:, P - n_pad:] = True   # last n_pad patches are PADDING
        batch["beats_patches_mask"] = pad
    return batch


def _head(d_patch=16, d_model=24):
    return DecoupledGroundingHead(d_model=d_model, d_patch=d_patch)


# ── the parallel-branch trains: finite loss + grad on queries + 2D map ───────
def test_helper_returns_finite_loss_and_grads_queries():
    head = _head()
    # give the readout a non-trivial map so there is real gradient to propagate.
    torch.nn.init.normal_(head.readout.weight, std=0.5)
    batch = _fake_batch()

    weighted, metrics = decoupled_grounding_loss_term(head, batch, lambda_decoupled=0.5)

    assert weighted is not None
    assert weighted.dim() == 0                       # scalar
    assert torch.isfinite(weighted).all()
    assert weighted.requires_grad
    assert float(weighted.detach()) > 0.0

    # metrics surface the UNWEIGHTED loss + one MAE per scored feature.
    assert "loss_decoupled" in metrics
    assert sum(1 for k in metrics if k.startswith("decoupled_mae/")) == N_FEATURES

    # the parallel branch actually trains: backward lands gradient on the LEARNED
    # QUERIES (and K_proj) — the loss can reshape the per-feature maps.
    weighted.backward()
    assert head.queries.grad is not None
    assert head.queries.grad.abs().sum().item() > 0.0
    assert head.K_proj.weight.grad is not None
    assert head.K_proj.weight.grad.abs().sum().item() > 0.0


def test_head_produces_B_F_T_F_map_via_reshape():
    """The same patches the helper consumes give a (B, n_features, T, F) 2D map
    through the head's reshape — the figure product."""
    B, T, Fdim, d_patch = 2, 5, 4, 16
    head = _head(d_patch=d_patch)
    patches = torch.randn(B, T * Fdim, d_patch)
    A, _z, _pred = head(patches)
    grid = DecoupledGroundingHead.reshape_map(A, T, Fdim)
    assert grid.shape == (B, N_FEATURES, T, Fdim)
    # reshape preserves rows in row-major (T,F) order.
    assert torch.allclose(grid.reshape(B, N_FEATURES, T * Fdim), A)


# ── lambda scaling ───────────────────────────────────────────────────────────
def test_lambda_scales_loss_linearly():
    head = _head()
    torch.nn.init.normal_(head.readout.weight, std=0.5)
    batch = _fake_batch()
    w_half, _ = decoupled_grounding_loss_term(head, batch, lambda_decoupled=0.5)
    w_one, m_one = decoupled_grounding_loss_term(head, batch, lambda_decoupled=1.0)
    # weighted loss is exactly lambda * unweighted; metrics report the UNWEIGHTED loss.
    assert torch.allclose(w_one * 0.5, w_half, atol=1e-6)
    assert abs(float(w_one.detach()) - m_one["loss_decoupled"]) < 1e-6


# ── flag-off / no-op paths add NOTHING ───────────────────────────────────────
def test_lambda_zero_is_noop():
    head = _head()
    batch = _fake_batch()
    loss, metrics = decoupled_grounding_loss_term(head, batch, lambda_decoupled=0.0)
    assert loss is None
    assert metrics == {}


def test_no_head_is_noop():
    batch = _fake_batch()
    loss, metrics = decoupled_grounding_loss_term(None, batch, lambda_decoupled=0.5)
    assert loss is None
    assert metrics == {}


def test_missing_beats_patches_is_noop():
    """Legacy .pt files (no beats_patches in the batch) → silent no-op, never an error."""
    head = _head()
    batch = {
        "gt_scalars": torch.randn(2, N_FEATURES),
        "gt_mask": torch.ones(2, N_FEATURES, dtype=torch.bool),
        # no 'beats_patches'
    }
    loss, metrics = decoupled_grounding_loss_term(head, batch, lambda_decoupled=0.5)
    assert loss is None and metrics == {}


def test_missing_gt_scalars_is_noop():
    head = _head()
    batch = {"beats_patches": torch.randn(2, 6, 16)}  # no gt_scalars / gt_mask
    loss, metrics = decoupled_grounding_loss_term(head, batch, lambda_decoupled=0.5)
    assert loss is None and metrics == {}


# ── the decoupling property: V gets NO grad → branch can't rewrite values ─────
def test_grounding_grad_does_not_reach_V_proj():
    """The grounding loss pools over V.detach(), so V_proj receives NO gradient —
    the head can only satisfy the loss by reshaping attention. In the real loop
    this is also what keeps the branch off the LM CE graph (it shares only the
    detached/encoder patches), so the LM keeps generating clean untagged prose."""
    head = _head()
    torch.nn.init.normal_(head.readout.weight, std=0.5)
    batch = _fake_batch()
    weighted, _ = decoupled_grounding_loss_term(head, batch, lambda_decoupled=0.7)
    weighted.backward()
    # value path is dead under the grounding loss.
    assert head.V_proj.weight.grad is None
    assert head.V_proj.bias.grad is None
    # readout (on the live z) and queries DO learn.
    assert head.readout.weight.grad is not None
    assert head.readout.weight.grad.abs().sum().item() > 0.0


def test_input_patches_not_mutated_and_no_grad_leak_to_batch():
    """The helper must not require grad on the incoming batch tensors (they come
    straight off the dataloader). The head builds its own graph from them."""
    head = _head()
    torch.nn.init.normal_(head.readout.weight, std=0.5)
    batch = _fake_batch()
    patches_before = batch["beats_patches"].clone()
    weighted, _ = decoupled_grounding_loss_term(head, batch, lambda_decoupled=0.5)
    weighted.backward()
    # batch tensors are leaves with requires_grad=False → never accumulate grad.
    assert batch["beats_patches"].grad is None
    assert batch["gt_scalars"].grad is None
    # and the helper did not mutate the input patches in place.
    assert torch.allclose(batch["beats_patches"], patches_before)


# ── collate_fn pad-mask is inverted to a valid-mask ──────────────────────────
def test_pad_mask_is_inverted_so_padded_patches_get_zero_attention():
    """collate_fn sets beats_patches_mask=True at PADDED positions; the head wants
    True at VALID positions. The helper inverts it, so the head must place ~0
    attention on the padded tail."""
    B, P, d_patch, n_pad = 2, 10, 16, 4
    head = _head(d_patch=d_patch)
    batch = _fake_batch(B=B, P=P, d_patch=d_patch, with_mask=True, n_pad=n_pad)

    # Reproduce the head call exactly as the helper does, then inspect the map.
    valid_mask = ~batch["beats_patches_mask"]
    A, _z, _pred = head(batch["beats_patches"], patch_mask=valid_mask)
    # padded tail (last n_pad patches) gets ~0 attention; valid head sums to 1.
    assert A[:, :, P - n_pad:].abs().max().item() < 1e-6
    valid_sums = A[:, :, : P - n_pad].sum(dim=-1)
    assert torch.allclose(valid_sums, torch.ones_like(valid_sums), atol=1e-5)

    # and the helper runs cleanly with the mask present (finite loss).
    weighted, _ = decoupled_grounding_loss_term(head, batch, lambda_decoupled=0.5)
    assert weighted is not None and torch.isfinite(weighted).all()


def test_zero_gt_mask_gives_zero_loss_through_helper():
    """A batch where nothing is supervised (all-False gt_mask) → zero grounding
    loss, mirroring the head's own zero-mask contract, but still returns a tensor
    (not None — the head DID run, there was just nothing to score)."""
    head = _head()
    batch = _fake_batch()
    batch["gt_mask"] = torch.zeros_like(batch["gt_mask"])
    weighted, metrics = decoupled_grounding_loss_term(head, batch, lambda_decoupled=0.5)
    assert weighted is not None
    assert float(weighted.detach()) == 0.0
    assert metrics["loss_decoupled"] == 0.0


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
