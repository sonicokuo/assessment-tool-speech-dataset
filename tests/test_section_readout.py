"""Tests for src/section_readout.py — the attention-grounding readout head.

This module had ZERO tests despite being gradient-correctness-critical (it is
the only thing that supervises the section attention maps). The central claim is
that regressing each section's scalar from z = alpha · V.detach() puts gradient
on the ATTENTION (alpha) but not on the patch values (V) — that is what forces
the map onto evidence rather than letting V smuggle the feature. These tests
prove exactly that, plus routing, masking, dispatch, and the dynamic path.

Pure torch + section_tags + feature_set (no transformers/mamba), runs anywhere.
"""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from section_readout import (  # noqa: E402
    SectionReadoutHead,
    build_section_feature_mask,
    section_readout_loss,
    query_section_indices,
    warmup_lambda,
)


def test_warmup_lambda_ramps_then_holds():
    # warmup over 3 epochs: 0 at epoch 0, target at epoch 3, holds after.
    assert warmup_lambda(0.05, 0, 3) == 0.0
    assert abs(warmup_lambda(0.05, 1, 3) - 0.05 / 3) < 1e-9
    assert abs(warmup_lambda(0.05, 2, 3) - 0.05 * 2 / 3) < 1e-9
    assert warmup_lambda(0.05, 3, 3) == 0.05
    assert warmup_lambda(0.05, 9, 3) == 0.05      # held at target after warmup


def test_warmup_lambda_disabled_returns_target():
    assert warmup_lambda(0.5, 0, 0) == 0.5        # no warmup -> target immediately
    assert warmup_lambda(0.5, 0, -1) == 0.5
from feature_set import N_FEATURES, SUPERVISED_FEATURES  # noqa: E402
from section_tags import N_SECTIONS, SECTION_TAGS  # noqa: E402


# ── routing matrix ───────────────────────────────────────────────────────────
def test_route_matrix_shape_and_dtype():
    R = build_section_feature_mask()
    assert R.shape == (N_SECTIONS, N_FEATURES)
    assert R.dtype == torch.bool


def test_route_matrix_matches_catalog():
    """R[s,f] iff feature f is named by section s in the catalog."""
    R = build_section_feature_mask()
    feat_idx = {name: i for i, (name, _c, _f) in enumerate(SUPERVISED_FEATURES)}
    for s_idx, sec in enumerate(SECTION_TAGS):
        expected = {feat_idx[f] for f in sec.feature_names if f in feat_idx}
        got = {j for j in range(N_FEATURES) if bool(R[s_idx, j])}
        assert got == expected, f"section {sec.name}: {got} != {expected}"


def test_route_each_scalar_routed_exactly_once():
    # In the EMNLP catalog every one of the 8 scalars belongs to exactly one
    # section, so the matrix has exactly N_FEATURES True entries.
    R = build_section_feature_mask()
    assert int(R.sum()) == N_FEATURES
    # and each feature column has exactly one section owner
    assert torch.all(R.sum(dim=0) == 1)


# ── forward shape ────────────────────────────────────────────────────────────
def test_forward_shape_and_upcasts_bf16():
    head = SectionReadoutHead(d_v=16)
    z = torch.randn(3, 16, dtype=torch.bfloat16)
    out = head(z)
    assert out.shape == (3, N_FEATURES)
    assert out.dtype == head.mlp[0].weight.dtype   # up-cast to head's float32


# ── THE grounding-gradient claim (static mode) ───────────────────────────────
def test_loss_static_gradient_flows_to_alpha_not_V():
    torch.manual_seed(0)
    B, P, d_v = 2, 5, 16
    head = SectionReadoutHead(d_v=d_v)
    # Give the readout a non-trivial output map so there is real gradient.
    torch.nn.init.normal_(head.mlp[-1].weight, std=0.5)

    alpha_raw = torch.randn(B, N_SECTIONS, P, requires_grad=True)
    alpha = torch.softmax(alpha_raw, dim=-1)
    V = torch.randn(B, P, d_v, requires_grad=True)
    gt = torch.randn(B, N_FEATURES)
    mask = torch.ones(B, N_FEATURES, dtype=torch.bool)

    loss, mae = head.loss_static(alpha, V, gt, mask)
    assert loss.requires_grad and float(loss.detach()) > 0.0
    loss.backward()

    # alpha (the attention) receives gradient — the loss can move the map.
    assert alpha_raw.grad is not None
    assert alpha_raw.grad.abs().sum().item() > 0.0
    # V (the patch encodings) receives NONE — detached, so the head cannot
    # satisfy the loss by rewriting V; it must move alpha. This is the fix.
    assert V.grad is None
    assert mae.shape == (N_FEATURES,)


def test_loss_static_zero_mask_is_zero_loss():
    head = SectionReadoutHead(d_v=8)
    alpha = torch.softmax(torch.randn(2, N_SECTIONS, 4), dim=-1)
    V = torch.randn(2, 4, 8)
    gt = torch.randn(2, N_FEATURES)
    mask = torch.zeros(2, N_FEATURES, dtype=torch.bool)   # nothing supervised
    loss, _ = head.loss_static(alpha, V, gt, mask)
    assert loss.item() == 0.0


# ── dynamic mode ─────────────────────────────────────────────────────────────
def test_loss_dynamic_grounds_alpha_and_drops_skipped_queries():
    torch.manual_seed(1)
    Nq, B, P, d_v = 3, 2, 4, 8
    head = SectionReadoutHead(d_v=d_v)
    torch.nn.init.normal_(head.mlp[-1].weight, std=0.5)

    alpha_raw = torch.randn(Nq, P, requires_grad=True)
    alpha = torch.softmax(alpha_raw, dim=-1)
    V = torch.randn(B, P, d_v, requires_grad=True)
    batch_idx = torch.tensor([0, 1, 0])
    qsi = torch.tensor([0, -1, 2])        # middle query is <r> → must be dropped
    gt = torch.randn(B, N_FEATURES)
    mask = torch.ones(B, N_FEATURES, dtype=torch.bool)

    loss, mae = head.loss_dynamic(alpha, V, batch_idx, qsi, gt, mask)
    assert loss is not None
    loss.backward()

    assert V.grad is None                                  # V detached
    assert alpha_raw.grad is not None
    assert alpha_raw.grad[1].abs().sum().item() == 0.0     # the -1 query got no gradient
    assert alpha_raw.grad[0].abs().sum().item() > 0.0      # a kept query did


def test_loss_dynamic_all_skipped_returns_none():
    head = SectionReadoutHead(d_v=8)
    alpha = torch.softmax(torch.randn(2, 4), dim=-1)
    V = torch.randn(1, 4, 8)
    loss, mae = head.loss_dynamic(
        alpha, V, torch.tensor([0, 0]), torch.tensor([-1, -1]),
        torch.randn(1, N_FEATURES), torch.ones(1, N_FEATURES, dtype=torch.bool),
    )
    assert loss is None and mae is None


# ── dispatch entry point ─────────────────────────────────────────────────────
def test_section_readout_loss_noop_when_inputs_missing():
    assert section_readout_loss(None, None, None) == (None, {})
    head = SectionReadoutHead(d_v=8)
    ctx = {"readout_head": head, "readout_alpha": torch.rand(2, N_SECTIONS, 4),
           "V": torch.rand(2, 4, 8), "mode": "static"}
    # gt_scalars None → no-op
    assert section_readout_loss(ctx, None, torch.ones(2, N_FEATURES, dtype=torch.bool)) == (None, {})
    # missing alpha → no-op
    ctx2 = {"readout_head": head, "V": torch.rand(2, 4, 8), "mode": "static"}
    assert section_readout_loss(ctx2, torch.randn(2, N_FEATURES),
                                torch.ones(2, N_FEATURES, dtype=torch.bool)) == (None, {})


def test_section_readout_loss_static_returns_named_metrics():
    head = SectionReadoutHead(d_v=8)
    ctx = {
        "readout_head": head,
        "readout_alpha": torch.softmax(torch.randn(2, N_SECTIONS, 4), dim=-1),
        "V": torch.randn(2, 4, 8),
        "mode": "static",
    }
    gt = torch.randn(2, N_FEATURES)
    mask = torch.ones(2, N_FEATURES, dtype=torch.bool)
    loss, metrics = section_readout_loss(ctx, gt, mask)
    assert loss is not None
    assert set(metrics.keys()) == {name for name, _c, _f in SUPERVISED_FEATURES}


# ── query → section idx mapping ──────────────────────────────────────────────
def test_query_section_indices_maps_and_defaults_negative():
    fired = torch.tensor([10, 11, 99])
    out = query_section_indices(fired, {10: 0, 11: 3})
    assert out.tolist() == [0, 3, -1]   # 99 is unknown (e.g. <r>) → -1


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
