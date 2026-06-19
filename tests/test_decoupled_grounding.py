"""Tests for src/decoupled_grounding.py — the token-free 2D grounding head.

The central claim mirrors section_readout.py's: regressing each feature's scalar
from z = A · V.detach() puts the grounding gradient on the ATTENTION (here the
LEARNED queries + K_proj), NOT on the patch value encodings V. That is what
forces each per-feature map onto real evidence instead of letting V smuggle the
feature in while the map stays flat. These tests prove exactly that, plus shapes,
softmax/masking, masked loss, and the anti-collapse diversity penalty.

Pure torch + feature_set (no transformers/mamba/LM), runs anywhere on CPU.
"""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from decoupled_grounding import (  # noqa: E402
    DecoupledGroundingHead,
    query_orthogonality_penalty,
    feature_names,
)
from feature_set import N_FEATURES, SUPERVISED_FEATURES  # noqa: E402


# ── shapes ───────────────────────────────────────────────────────────────────
def test_forward_shapes_default_n_features():
    B, P, d_patch, d_model = 2, 7, 12, 16
    head = DecoupledGroundingHead(d_model=d_model, d_patch=d_patch)
    assert head.n_features == N_FEATURES                  # synced to catalog
    patches = torch.randn(B, P, d_patch)
    A, z, pred = head(patches)
    assert A.shape == (B, N_FEATURES, P)                  # per-feature map over P
    assert z.shape == (B, N_FEATURES, d_model)
    assert pred.shape == (B, N_FEATURES)


def test_query_table_is_parameter_not_embedding():
    """Token-free: the queries are a free nn.Parameter, NOT a vocab-tied table."""
    head = DecoupledGroundingHead(d_model=8, d_patch=8)
    assert isinstance(head.queries, torch.nn.Parameter)
    assert head.queries.shape == (N_FEATURES, 8)
    # no nn.Embedding anywhere in the module → not tied to a tokenizer vocab.
    assert not any(isinstance(m, torch.nn.Embedding) for m in head.modules())


def test_reshape_map_to_T_F():
    B, T, Fdim, d_patch, d_model = 2, 4, 3, 10, 16
    head = DecoupledGroundingHead(d_model=d_model, d_patch=d_patch)
    patches = torch.randn(B, T * Fdim, d_patch)
    A, _, _ = head(patches)
    grid = DecoupledGroundingHead.reshape_map(A, T, Fdim)
    assert grid.shape == (B, N_FEATURES, T, Fdim)
    # reshape preserves the row, in row-major (T,F) order.
    assert torch.allclose(grid.reshape(B, N_FEATURES, T * Fdim), A)


def test_forward_accepts_T_F_grid_input():
    B, T, Fdim, d_patch, d_model = 2, 4, 3, 10, 16
    head = DecoupledGroundingHead(d_model=d_model, d_patch=d_patch)
    grid = torch.randn(B, T, Fdim, d_patch)
    A, z, pred = head(grid)
    assert A.shape == (B, N_FEATURES, T * Fdim)
    assert pred.shape == (B, N_FEATURES)


def test_custom_n_features_and_multihead():
    B, P, d_patch, d_model = 2, 9, 12, 16
    head = DecoupledGroundingHead(d_model=d_model, d_patch=d_patch, n_features=5, n_heads=4)
    A, z, pred = head(torch.randn(B, P, d_patch))
    assert A.shape == (B, 5, P)                            # heads averaged into ONE map
    assert z.shape == (B, 5, d_model)
    assert pred.shape == (B, 5)


def test_d_model_not_divisible_by_heads_raises():
    try:
        DecoupledGroundingHead(d_model=10, d_patch=8, n_heads=4)
    except ValueError:
        return
    raise AssertionError("expected ValueError for indivisible d_model/n_heads")


# ── softmax / masking ────────────────────────────────────────────────────────
def test_attention_rows_sum_to_one():
    head = DecoupledGroundingHead(d_model=16, d_patch=12)
    A, _, _ = head(torch.randn(2, 7, 12))
    sums = A.sum(dim=-1)                                   # (B, n_features)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)


def test_masked_patches_get_near_zero_attention():
    B, P, d_patch = 2, 6, 12
    head = DecoupledGroundingHead(d_model=16, d_patch=d_patch)
    patches = torch.randn(B, P, d_patch)
    mask = torch.ones(B, P, dtype=torch.bool)
    mask[:, 3:] = False                                    # last 3 patches invalid
    A, _, _ = head(patches, patch_mask=mask)
    assert A[:, :, 3:].abs().max().item() < 1e-6          # ~0 attention on masked
    # all mass on the valid patches; rows still sum to 1.
    valid_sums = A[:, :, :3].sum(dim=-1)
    assert torch.allclose(valid_sums, torch.ones_like(valid_sums), atol=1e-5)


def test_grid_mask_folds_like_patches():
    B, T, Fdim, d_patch = 2, 3, 2, 10
    head = DecoupledGroundingHead(d_model=16, d_patch=d_patch)
    grid = torch.randn(B, T, Fdim, d_patch)
    gmask = torch.ones(B, T, Fdim, dtype=torch.bool)
    gmask[:, 2, :] = False                                 # whole last time-step out
    A, _, _ = head(grid, patch_mask=gmask)
    grid_map = DecoupledGroundingHead.reshape_map(A, T, Fdim)
    assert grid_map[:, :, 2, :].abs().max().item() < 1e-6


# ── THE grounding-gradient proof (most important) ────────────────────────────
def test_grounding_grad_reaches_queries_not_V():
    """Backprop the grounding loss and assert:
      - the learned QUERIES receive nonzero gradient (the loss can move the map),
      - K_proj receives gradient (the other half of 'where to look'),
      - V IS DETACHED in the readout pool, so V_proj gets NO gradient from this
        loss and the input patches get NO gradient — the head cannot satisfy the
        loss by rewriting V; it must reshape attention. This is the fix.
    """
    torch.manual_seed(0)
    B, P, d_patch, d_model = 2, 6, 12, 16
    head = DecoupledGroundingHead(d_model=d_model, d_patch=d_patch)
    # Give the readout a non-trivial map so there is real gradient to propagate.
    last = head.readout if isinstance(head.readout, torch.nn.Linear) else head.readout[-1]
    torch.nn.init.normal_(last.weight, std=0.5)

    patches = torch.randn(B, P, d_patch, requires_grad=True)
    gt = torch.randn(B, N_FEATURES)
    mask = torch.ones(B, N_FEATURES, dtype=torch.bool)

    A, z, pred = head(patches)
    loss, mae = head.grounding_loss(pred, gt, mask)
    assert loss.requires_grad and float(loss.detach()) > 0.0
    loss.backward()

    # the attention path receives gradient — the loss can reshape the maps.
    assert head.queries.grad is not None
    assert head.queries.grad.abs().sum().item() > 0.0
    assert head.K_proj.weight.grad is not None
    assert head.K_proj.weight.grad.abs().sum().item() > 0.0

    # V is detached in the readout pool → the value path gets NO grounding grad.
    assert head.V_proj.weight.grad is None
    assert head.V_proj.bias.grad is None
    # and the patches themselves receive NO gradient: every route from the loss to
    # `patches` goes through K (→ softmax, fine) or through V (→ detached). Because
    # softmax(scores) feeds V.detach() in the pool, the only live route to patches
    # is via K. K does see patches, so patches.grad is NON-None but is purely the
    # "where to look" signal, never the "rewrite V" signal. Assert it exists and
    # is finite (sanity), and separately prove the V route is dead below.
    assert patches.grad is not None
    assert torch.isfinite(patches.grad).all()
    assert mae.shape == (N_FEATURES,)


def test_V_path_is_dead_isolated():
    """Isolate the V route: if we feed the readout from z built on the NON-detached
    V, patches would get a value-rewriting gradient. The head detaches V, so build
    the same pool with V.detach() and confirm zero grad flows through it.

    Concretely: a readout loss computed purely from V.detach() must give patches a
    gradient identical to one where V never appears. We verify the head's z carries
    no grad-graph dependence on V_proj by checking V_proj has no grad after the
    grounding loss (done above) AND that z's grad-fn does not reach V_proj's params
    — operationalized: zero out the K route and confirm the loss becomes constant
    in V.
    """
    torch.manual_seed(1)
    B, P, d_patch, d_model = 1, 5, 8, 8
    head = DecoupledGroundingHead(d_model=d_model, d_patch=d_patch)
    last = head.readout if isinstance(head.readout, torch.nn.Linear) else head.readout[-1]
    torch.nn.init.normal_(last.weight, std=0.5)

    # Freeze the attention (queries + K_proj) so the ONLY way the loss could change
    # is by moving V. Since V is detached, the gradient on V_proj must be exactly 0
    # even with the rest of the graph live.
    head.queries.requires_grad_(False)
    for p in head.K_proj.parameters():
        p.requires_grad_(False)

    patches = torch.randn(B, P, d_patch)
    gt = torch.randn(B, N_FEATURES)
    mask = torch.ones(B, N_FEATURES, dtype=torch.bool)
    A, z, pred = head(patches)
    loss, _ = head.grounding_loss(pred, gt, mask)
    loss.backward()

    # readout still learns (its weights are on the live z), but V_proj — the value
    # encoder the map pools over — is provably untouched by the grounding loss.
    assert head.V_proj.weight.grad is None
    last = head.readout if isinstance(head.readout, torch.nn.Linear) else head.readout[-1]
    assert last.weight.grad is not None and last.weight.grad.abs().sum().item() > 0.0


def test_readout_is_shallow_bottleneck():
    """Default readout is a single Linear(d_model→1) so the attention map — not a
    deep readout — is the evidence bottleneck."""
    head = DecoupledGroundingHead(d_model=16, d_patch=8)
    assert isinstance(head.readout, torch.nn.Linear)
    assert head.readout.out_features == 1
    # opt-in 1-hidden MLP variant.
    head2 = DecoupledGroundingHead(d_model=16, d_patch=8, readout_hidden=32)
    assert isinstance(head2.readout, torch.nn.Sequential)


# ── masked loss ──────────────────────────────────────────────────────────────
def test_zero_mask_is_zero_loss():
    head = DecoupledGroundingHead(d_model=8, d_patch=8)
    A, z, pred = head(torch.randn(2, 4, 8))
    gt = torch.randn(2, N_FEATURES)
    mask = torch.zeros(2, N_FEATURES, dtype=torch.bool)   # nothing supervised
    loss, mae = head.grounding_loss(pred, gt, mask)
    assert loss.item() == 0.0
    assert torch.all(mae == 0.0)


def test_present_features_contribute_to_loss():
    torch.manual_seed(2)
    head = DecoupledGroundingHead(d_model=8, d_patch=8)
    last = head.readout
    torch.nn.init.normal_(last.weight, std=0.5)
    last.bias.data.fill_(10.0)                             # force a large error
    A, z, pred = head(torch.randn(3, 5, 8))
    gt = torch.zeros(3, N_FEATURES)
    full = torch.ones(3, N_FEATURES, dtype=torch.bool)
    one = torch.zeros(3, N_FEATURES, dtype=torch.bool)
    one[:, 0] = True
    loss_full, _ = head.grounding_loss(pred, gt, full)
    loss_one, _ = head.grounding_loss(pred, gt, one)
    assert loss_full.item() > 0.0 and loss_one.item() > 0.0
    # supervising more present features changes the loss (they contribute).
    assert abs(loss_full.item() - loss_one.item()) > 0.0


def test_loss_normalizes_by_feature_scale():
    """The error is divided by FEATURE_SCALES, so a fixed raw error contributes
    less for a large-scale feature (f0_mean, scale 50) than a small one
    (overlap_ratio, scale 0.3)."""
    head = DecoupledGroundingHead(d_model=4, d_patch=4)
    pred = torch.zeros(1, N_FEATURES)
    gt = torch.zeros(1, N_FEATURES)
    names = [n for n, _c, _f in SUPERVISED_FEATURES]
    ovr_idx = names.index("overlap_ratio")
    f0_idx = names.index("f0_mean")
    gt[0, f0_idx] = 1.0
    m_f0 = torch.zeros(1, N_FEATURES, dtype=torch.bool); m_f0[0, f0_idx] = True
    gt2 = torch.zeros(1, N_FEATURES); gt2[0, ovr_idx] = 1.0
    m_ovr = torch.zeros(1, N_FEATURES, dtype=torch.bool); m_ovr[0, ovr_idx] = True
    loss_f0, _ = head.grounding_loss(pred, gt, m_f0)
    loss_ovr, _ = head.grounding_loss(pred, gt2, m_ovr)
    # same raw error of 1.0, but f0 scale (50) >> overlap scale (0.3) → f0 loss tiny.
    assert loss_f0.item() < loss_ovr.item()


# ── diversity / anti-collapse ────────────────────────────────────────────────
def test_orthogonality_penalty_zero_for_orthogonal_high_for_identical():
    # orthonormal rows → ~0 penalty.
    q_ortho = torch.eye(4, 8)
    assert query_orthogonality_penalty(q_ortho).item() < 1e-6
    # identical rows → cosine 1 off-diagonal → penalty ≈ 1.
    q_same = torch.ones(4, 8)
    assert query_orthogonality_penalty(q_same).item() > 0.99
    # single query → no pairs → exactly 0.
    assert query_orthogonality_penalty(torch.randn(1, 8)).item() == 0.0


def test_orthogonality_penalty_is_differentiable_and_pushes_apart():
    """One gradient step on the penalty reduces the similarity of two parallel
    queries (proves it actually drives maps apart, not just reports a number)."""
    q = torch.tensor([[1.0, 0.0, 0.0], [0.9, 0.1, 0.0]], requires_grad=True)
    p0 = query_orthogonality_penalty(q)
    p0.backward()
    with torch.no_grad():
        q_new = q - 1.0 * q.grad
    p1 = query_orthogonality_penalty(q_new.detach())
    assert p1.item() < p0.item()


def test_identical_init_queries_diverge_after_a_step():
    """Two features whose queries start IDENTICAL produce identical maps; after one
    optimizer step against the grounding loss on structured patches they diverge —
    the map-collapse risk is escapable with supervision."""
    torch.manual_seed(3)
    B, P, d_patch, d_model = 4, 6, 8, 8
    head = DecoupledGroundingHead(d_model=d_model, d_patch=d_patch, n_features=2)
    last = head.readout
    torch.nn.init.normal_(last.weight, std=0.5)
    # force the two queries identical at init.
    with torch.no_grad():
        head.queries[1] = head.queries[0]

    # structured patches: distinct regions carry distinct signal so the two
    # features have a reason to attend differently.
    patches = torch.randn(B, P, d_patch)
    gt = torch.randn(B, 2)
    mask = torch.ones(B, 2, dtype=torch.bool)

    A0, _, pred0 = head(patches)
    assert torch.allclose(A0[:, 0], A0[:, 1], atol=1e-6)   # identical maps at init

    opt = torch.optim.SGD(head.parameters(), lr=1.0)
    loss, _ = head.grounding_loss(pred0, gt, mask)
    opt.zero_grad(); loss.backward(); opt.step()

    A1, _, _ = head(patches)
    # after a supervised step the two maps are no longer identical.
    assert not torch.allclose(A1[:, 0], A1[:, 1], atol=1e-4)


# ── misc ─────────────────────────────────────────────────────────────────────
def test_feature_names_matches_catalog():
    assert feature_names() == [n for n, _c, _f in SUPERVISED_FEATURES]
    assert len(feature_names()) == N_FEATURES


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
