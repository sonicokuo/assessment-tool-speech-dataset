"""Tests for DIRECT overlap-feature 2D-map supervision (the strongest grounding claim).

The decoupled grounding head only WEAKLY supervises the overlap map: it regresses the
overlap_ratio SCALAR from the pooled z, and a diffuse map predicts that scalar as well
as a sharp one. This module tests the ADDITIVE segmentation-style loss in
src/decoupled_grounding.py that forces the overlap query's map to land on the actual
overlapped time region, using the oracle `overlap_segments` already in every .pt.

What is proven (matches the task spec a-e):
  (a) the time-mask builder — overlap_time_target — maps seconds→patch-bin correctly:
      a clip with [(1.0, 2.0)] and duration 4 s, T_p=8 lights the bins covering 1-2 s
      and zeros elsewhere; empty segments → all-zero; multi-segment is correct.
  (b) the overlap-map loss DECREASES under gradient descent on a toy map, and the
      map-vs-target IoU RISES, as the map is pulled onto the target region.
  (c) lambda_overlap_map=0 is an EXACT no-op (loss tensor unchanged) in BOTH modes.
  (d) the loss works for BOTH a softmax map and a bottleneck keep-prob map, with grad
      reaching the head's queries in each.
  (e) a NO-overlap (clean) clip is driven toward an EMPTY overlap map.

Pure torch + feature_set — runs on CPU anywhere.
"""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from decoupled_grounding import (  # noqa: E402
    DecoupledGroundingHead,
    decoupled_grounding_loss_term,
    overlap_time_target,
    overlap_time_activation,
    overlap_map_loss_from_map,
    soft_dice_loss,
    overlap_ratio_index,
    F_P_DEFAULT,
    WAVLM_FRAME_RATE_HZ,
)
from feature_set import N_FEATURES  # noqa: E402


# ════════════════════════════════════════════════════════════════════════════════
# (a) TIME-MASK BUILDER — seconds → patch-time-bin
# ════════════════════════════════════════════════════════════════════════════════
def test_time_target_single_segment_lights_correct_bins():
    """[(1.0,2.0)] over a 4 s clip with T_p=8 → each bin covers 0.5 s, so bins 2,3
    (covering [1.0,1.5),[1.5,2.0)) are positive, all others zero."""
    tgt = overlap_time_target([(1.0, 2.0)], duration_sec=4.0, t_p=8, soft=False)
    assert tgt.shape == (8,)
    expected = torch.zeros(8)
    expected[2] = 1.0   # [1.0, 1.5)
    expected[3] = 1.0   # [1.5, 2.0)
    assert torch.equal(tgt, expected), f"got {tgt.tolist()}"


def test_time_target_soft_is_covered_fraction():
    """Soft target = fraction of each 0.5 s bin covered by the span. [(1.25,2.0)]:
    bin 2 = [1.0,1.5) is half covered (0.25/0.5=0.5), bin 3 = [1.5,2.0) fully (1.0)."""
    tgt = overlap_time_target([(1.25, 2.0)], duration_sec=4.0, t_p=8, soft=True)
    assert abs(tgt[2].item() - 0.5) < 1e-6
    assert abs(tgt[3].item() - 1.0) < 1e-6
    assert tgt[0].item() == 0.0 and tgt[4].item() == 0.0


def test_time_target_empty_segments_is_all_zero():
    """A clip with NO overlap → all-zero target (the map should be empty)."""
    tgt = overlap_time_target([], duration_sec=4.0, t_p=8, soft=True)
    assert torch.count_nonzero(tgt).item() == 0
    # and None / 0-duration also degrade to all-zero rather than erroring.
    assert torch.count_nonzero(overlap_time_target([(1.0, 2.0)], 0.0, 8)).item() == 0


def test_time_target_multi_segment_is_correct():
    """Two disjoint spans light their respective bins (and nothing in between)."""
    # 8 s clip, T_p=8 → bins are 1 s wide. [(0,1)] → bin 0; [(5,7)] → bins 5,6.
    tgt = overlap_time_target([(0.0, 1.0), (5.0, 7.0)], duration_sec=8.0, t_p=8, soft=False)
    expected = torch.zeros(8)
    expected[0] = 1.0
    expected[5] = 1.0
    expected[6] = 1.0
    assert torch.equal(tgt, expected), f"got {tgt.tolist()}"


def test_time_target_half_open_intersection_matches_grounding_metrics():
    """A bin counts as positive iff its [t0,t1) intersects a span — the same half-open
    rule as grounding_metrics._time_bin_in_windows. A span ending exactly on a bin
    boundary does NOT light the next bin."""
    # 4 s, T_p=4 → 1 s bins. [(1.0, 2.0)] → bin 1 only (boundary at 2.0 excludes bin 2).
    tgt = overlap_time_target([(1.0, 2.0)], duration_sec=4.0, t_p=4, soft=False)
    assert tgt.tolist() == [0.0, 1.0, 0.0, 0.0]


def test_time_target_uses_50hz_duration_convention():
    """Duration is n_wavlm_frames / 50; this test documents the exact constant so the
    train-time builder (audio_lens / 50) and the eval builder agree."""
    assert WAVLM_FRAME_RATE_HZ == 50.0
    # 200 frames → 4.0 s. Span [(1,2)] with T_p=8 → bins 2,3 as in the single-seg test.
    dur = 200 / WAVLM_FRAME_RATE_HZ
    assert dur == 4.0


# ════════════════════════════════════════════════════════════════════════════════
# frequency marginalization + Dice primitives
# ════════════════════════════════════════════════════════════════════════════════
def test_time_activation_marginalizes_over_frequency_time_major():
    """The flat map is TIME-MAJOR (idx = t*F_P + f). Marginalizing over F_P gives the
    per-time activation. Build a map that is 1 on time-bin 2 (all freqs), 0 elsewhere."""
    F_p = F_P_DEFAULT
    T_p = 4
    grid = torch.zeros(1, T_p, F_p)
    grid[0, 2, :] = 1.0
    flat = grid.reshape(1, T_p * F_p)
    act = overlap_time_activation(flat, f_p=F_p, reduce="mean")
    assert act.shape == (1, T_p)
    assert act[0].tolist() == [0.0, 0.0, 1.0, 0.0]


def test_soft_dice_zero_when_pred_matches_target():
    pred = torch.tensor([[0.0, 1.0, 1.0, 0.0]])
    tgt = torch.tensor([[0.0, 1.0, 1.0, 0.0]])
    loss = soft_dice_loss(pred, tgt)
    assert loss.item() < 1e-3


def test_soft_dice_high_when_pred_disjoint_from_target():
    pred = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
    tgt = torch.tensor([[0.0, 0.0, 1.0, 1.0]])
    loss = soft_dice_loss(pred, tgt)
    assert loss.item() > 0.5


def test_soft_dice_empty_vs_empty_is_zero():
    """A clean clip (all-zero target) with an all-zero pred → ~0 Dice loss (the eps
    smoothing makes 0/0 resolve to a matched-empty, the clean-clip ideal)."""
    pred = torch.zeros(1, 6)
    tgt = torch.zeros(1, 6)
    assert soft_dice_loss(pred, tgt).item() < 1e-3


# ════════════════════════════════════════════════════════════════════════════════
# (b) THE LOSS DECREASES UNDER GD AND IoU RISES
# ════════════════════════════════════════════════════════════════════════════════
def test_overlap_map_loss_decreases_and_iou_rises_under_gd():
    """Optimize a toy per-time activation toward the target region: the soft-Dice loss
    falls and the map-vs-target IoU climbs (the map is pulled onto the overlap)."""
    torch.manual_seed(0)
    F_p = F_P_DEFAULT
    T_p = 8
    # Target: overlap in bins 2-3 (e.g. [(1,2)] over 4 s).
    target = overlap_time_target([(1.0, 2.0)], 4.0, T_p, soft=True).unsqueeze(0)  # (1,T_p)
    has = torch.ones(1)
    # A learnable flat map (logits → sigmoid keep-probs), init slightly OFF the region.
    logits = torch.zeros(1, T_p * F_p, requires_grad=True)
    opt = torch.optim.Adam([logits], lr=0.2)

    def step_loss():
        mp = torch.sigmoid(logits)               # (1, P) in [0,1]
        loss, metrics = overlap_map_loss_from_map(mp, target, has, f_p=F_p)
        return loss, metrics

    loss0, m0 = step_loss()
    iou0 = m0["overlap_map_iou"]
    for _ in range(300):
        opt.zero_grad()
        loss, _ = step_loss()
        loss.backward()
        opt.step()
    loss1, m1 = step_loss()
    iou1 = m1["overlap_map_iou"]

    assert loss1.item() < loss0.item() - 0.1, f"loss did not drop: {loss0.item()}→{loss1.item()}"
    assert iou1 > iou0, f"IoU did not rise: {iou0}→{iou1}"
    assert iou1 > 0.8, f"final IoU should be high, got {iou1}"
    # the optimized map concentrates on bins 2,3 (marginalized over frequency).
    act = overlap_time_activation(torch.sigmoid(logits).detach(), f_p=F_p)[0]
    assert act[2] > 0.5 and act[3] > 0.5
    assert act[0] < 0.5 and act[7] < 0.5


# ════════════════════════════════════════════════════════════════════════════════
# (e) NO-OVERLAP CLIP DRIVEN TOWARD AN EMPTY MAP
# ════════════════════════════════════════════════════════════════════════════════
def test_clean_clip_driven_toward_empty_map():
    """A non-overlap clip (has_overlap=0, all-zero target) — minimizing the loss drives
    the overlap map's total activation DOWN (toward empty), supporting the hedging
    story (no overlap → near-empty overlap map)."""
    torch.manual_seed(1)
    F_p = F_P_DEFAULT
    T_p = 8
    target = torch.zeros(1, T_p)        # clean clip
    has = torch.zeros(1)                # no overlap
    logits = torch.full((1, T_p * F_p), 2.0, requires_grad=True)  # start HOT (~0.88)
    opt = torch.optim.Adam([logits], lr=0.2)

    act0 = overlap_time_activation(torch.sigmoid(logits).detach(), f_p=F_p).sum().item()
    for _ in range(200):
        opt.zero_grad()
        mp = torch.sigmoid(logits)
        loss, _ = overlap_map_loss_from_map(mp, target, has, f_p=F_p)
        loss.backward()
        opt.step()
    act1 = overlap_time_activation(torch.sigmoid(logits).detach(), f_p=F_p).sum().item()

    assert act1 < act0 - 0.5, f"clean-clip activation did not fall: {act0}→{act1}"
    assert act1 < 0.5, f"clean-clip map should be near-empty, total act={act1}"


# ════════════════════════════════════════════════════════════════════════════════
# integration through decoupled_grounding_loss_term — both modes + no-op
# ════════════════════════════════════════════════════════════════════════════════
def _batch_with_overlap(B=4, T_p=8, d_patch=16):
    """A batch the helper reads, with overlap_segments + audio_lens so the overlap-map
    supervision fires. P = T_p * F_P. Clips alternate overlap / clean."""
    F_p = F_P_DEFAULT
    P = T_p * F_p
    torch.manual_seed(0)
    batch = {
        "beats_patches": torch.randn(B, P, d_patch),
        "gt_scalars": torch.randn(B, N_FEATURES),
        "gt_mask": torch.ones(B, N_FEATURES, dtype=torch.bool),
        # 4 s clips = 200 WavLM frames; even idx have a [(1,2)] overlap, odd are clean.
        "audio_lens": torch.full((B,), 200, dtype=torch.long),
        "overlap_segments": [
            [(1.0, 2.0)] if (b % 2 == 0) else [] for b in range(B)
        ],
    }
    return batch


def _head(d_patch=16, d_model=24, grounding_mode="softmax"):
    return DecoupledGroundingHead(
        d_model=d_model, d_patch=d_patch, grounding_mode=grounding_mode,
    )


# ── (c) lambda_overlap_map = 0 → EXACT no-op, BOTH modes ─────────────────────────
def test_lambda_overlap_map_zero_is_exact_noop_softmax():
    head = _head(grounding_mode="softmax")
    torch.nn.init.normal_(head.readout_weight, std=0.5)
    batch = _batch_with_overlap()
    base, base_m = decoupled_grounding_loss_term(head, batch, lambda_decoupled=0.5)
    sup, sup_m = decoupled_grounding_loss_term(
        head, batch, lambda_decoupled=0.5, lambda_overlap_map=0.0,
    )
    # identical loss value, and NO overlap-map metric was added.
    assert torch.allclose(base, sup, atol=0.0)
    assert "loss_overlap_map" not in sup_m
    assert sup_m.keys() == base_m.keys()


def test_lambda_overlap_map_zero_is_exact_noop_bottleneck():
    head = _head(grounding_mode="bottleneck")
    head.eval()  # deterministic concrete sample so the comparison is exact
    batch = _batch_with_overlap()
    base, base_m = decoupled_grounding_loss_term(head, batch, lambda_decoupled=0.5)
    sup, sup_m = decoupled_grounding_loss_term(
        head, batch, lambda_decoupled=0.5, lambda_overlap_map=0.0,
    )
    assert torch.allclose(base, sup, atol=0.0)
    assert "loss_overlap_map" not in sup_m


def test_missing_overlap_segments_is_noop_even_when_lambda_positive():
    """A batch with NO overlap_segments (legacy) → the supervision is skipped but the
    rest of the term runs unchanged; no overlap-map metric, finite loss."""
    head = _head()
    torch.nn.init.normal_(head.readout_weight, std=0.5)
    batch = _batch_with_overlap()
    del batch["overlap_segments"]
    base, _ = decoupled_grounding_loss_term(head, batch, lambda_decoupled=0.5)
    sup, sup_m = decoupled_grounding_loss_term(
        head, batch, lambda_decoupled=0.5, lambda_overlap_map=0.5,
    )
    assert torch.allclose(base, sup, atol=0.0)
    assert "loss_overlap_map" not in sup_m


# ── (d) works for BOTH softmax and bottleneck maps; grad reaches the queries ──────
def test_overlap_map_supervision_adds_loss_and_grads_queries_softmax():
    head = _head(grounding_mode="softmax")
    torch.nn.init.normal_(head.readout_weight, std=0.5)
    batch = _batch_with_overlap()
    base, _ = decoupled_grounding_loss_term(head, batch, lambda_decoupled=0.5)
    sup, sup_m = decoupled_grounding_loss_term(
        head, batch, lambda_decoupled=0.5, lambda_overlap_map=0.7,
    )
    # the supervised loss is strictly larger (a positive Dice term was added).
    assert float(sup.detach()) > float(base.detach())
    assert "loss_overlap_map" in sup_m and sup_m["loss_overlap_map"] > 0.0
    assert "overlap_map_iou" in sup_m
    # gradient from the overlap-map term reaches the LEARNED queries.
    sup.backward()
    assert head.queries.grad is not None
    assert head.queries.grad.abs().sum().item() > 0.0


def test_overlap_map_supervision_grads_queries_bottleneck():
    head = _head(grounding_mode="bottleneck")
    head.train()  # stochastic concrete; grad still flows through the keep-prob
    batch = _batch_with_overlap()
    sup, sup_m = decoupled_grounding_loss_term(
        head, batch, lambda_decoupled=0.5, lambda_overlap_map=0.7,
    )
    assert "loss_overlap_map" in sup_m
    assert torch.isfinite(sup).all()
    sup.backward()
    # bottleneck uses the differentiable keep-PROB (hard_concrete_keepprob) off the
    # stashed logit, so the overlap-map grad lands on queries / K_proj.
    assert head.queries.grad is not None
    assert head.queries.grad.abs().sum().item() > 0.0


def test_overlap_map_grad_does_not_reach_V_proj():
    """The overlap-map term operates on the attention/keep map only (softmax row or
    keep-prob), never on V — so V_proj stays grad-free, preserving the decoupling
    property (the map can only move, never rewrite the values)."""
    head = _head(grounding_mode="softmax")
    torch.nn.init.normal_(head.readout_weight, std=0.5)
    batch = _batch_with_overlap()
    sup, _ = decoupled_grounding_loss_term(
        head, batch, lambda_decoupled=0.5, lambda_overlap_map=0.7,
    )
    sup.backward()
    assert head.V_proj.weight.grad is None
    assert head.V_proj.bias.grad is None


def test_only_overlap_feature_map_is_supervised():
    """The supervision touches ONLY the overlap query's map. We verify the helper
    selects overlap_ratio's index and that the integration's positive Dice term is
    computed from that row (sanity: index resolves and is in range)."""
    idx = overlap_ratio_index()
    assert 0 <= idx < N_FEATURES
    # overlap_ratio is the catalog's localizable, oracle-GT feature.
    from decoupled_grounding import feature_names
    assert feature_names()[idx] == "overlap_ratio"


def test_reduce_max_also_works():
    """The frequency marginalization supports 'max' as well as 'mean'."""
    head = _head()
    torch.nn.init.normal_(head.readout_weight, std=0.5)
    batch = _batch_with_overlap()
    sup, sup_m = decoupled_grounding_loss_term(
        head, batch, lambda_decoupled=0.5, lambda_overlap_map=0.5,
        overlap_map_reduce="max",
    )
    assert "loss_overlap_map" in sup_m
    assert torch.isfinite(sup).all()


# ════════════════════════════════════════════════════════════════════════════════
# ADVERSARIAL — PADDED-DURATION TIME-BIN BUG (variable-length batch)
# ════════════════════════════════════════════════════════════════════════════════
# In decoupled_grounding_loss_term the time-target is built with
#     t_p = P // F_P_DEFAULT
# where P is the BATCH-PADDED BEATs patch count, but bin_dur = per-clip-duration / t_p
# uses each clip's OWN (unpadded) duration. For a SHORT clip in a batch padded to a
# longer max, this spreads the clip's duration across the padded t_p (too many bins),
# so the overlap span lands on the WRONG bins — and partly on padding bins where the
# map activation is forced to ~0 (softmax -inf / bottleneck keep-prob 0). The supervised
# region therefore disagrees with the per-clip valid-t_p region the eval (grounding_
# validate.iou_time, which runs the head on the UNPADDED clip) scores against.
#
# The existing tests never hit this because every clip in _batch_with_overlap shares one
# T_p and there is no padding. These tests use a MIXED-DURATION batch.
def test_padded_batch_target_lands_on_valid_region_softmax():
    """Short clip (4 s, valid t_p=10) padded into a batch whose max t_p=25: the loss's
    target for the [1,2] s overlap must fall inside the clip's VALID time-bins [0,10),
    not in the padding region. Currently it lands on bins 6-12 (padded t_p=25), three of
    which (10,11,12) are PADDING — the overlap map can never activate there."""
    F_p = F_P_DEFAULT
    d_patch = 16
    valid_tp0, P0 = 10, 10 * F_p           # 4 s clip
    valid_tp1, P1 = 25, 25 * F_p           # 10 s clip (sets batch max)
    Pmax = P1
    B = 2

    patches = torch.randn(B, Pmax, d_patch)
    pad_mask = torch.zeros(B, Pmax, dtype=torch.bool)
    pad_mask[0, P0:] = True                # clip0 valid only in first 80 patches
    gt = torch.zeros(B, N_FEATURES)
    gtm = torch.ones(B, N_FEATURES, dtype=torch.bool)

    batch = {
        "beats_patches": patches,
        "beats_patches_mask": pad_mask,
        "gt_scalars": gt,
        "gt_mask": gtm,
        "audio_lens": torch.tensor([int(4.0 * WAVLM_FRAME_RATE_HZ),    # 200
                                    int(10.0 * WAVLM_FRAME_RATE_HZ)]),  # 500
        "overlap_segments": [[(1.0, 2.0)], []],
    }

    # Rebuild EXACTLY what the loss term builds for clip0 (padded t_p), and what the
    # eval path would build (valid t_p), then assert the supervised positive bins fall
    # inside clip0's valid region [0, valid_tp0).
    head = DecoupledGroundingHead(d_model=24, d_patch=d_patch, grounding_mode="softmax")
    ovl_idx = overlap_ratio_index()
    A, _z, _ps = head(patches, patch_mask=~pad_mask)
    P = A.shape[-1]
    t_p_used = P // F_p                                       # 25 (padded — the bug)
    tgt_used = overlap_time_target([(1.0, 2.0)], 4.0, t_p_used, soft=False)
    pos_bins = tgt_used.nonzero().flatten().tolist()
    in_padding = [b for b in pos_bins if b >= valid_tp0]
    assert not in_padding, (
        f"PADDED-DURATION BUG: overlap-map target bins {pos_bins} for a 4 s clip use "
        f"padded t_p={t_p_used}; bins {in_padding} fall in the padding region "
        f"(>= valid t_p {valid_tp0}) where the map activation is forced to ~0. "
        f"Correct (valid t_p={valid_tp0}) bins would be "
        f"{overlap_time_target([(1.0,2.0)], 4.0, valid_tp0, soft=False).nonzero().flatten().tolist()}."
    )


def test_softmax_clean_clip_term_has_no_effect_mass_conserved():
    """In softmax mode the per-time activation sums to 1/F_P (mass conservation), so the
    clean-clip 'drive map to empty' Dice term is a near-constant with no useful gradient,
    and the logged overlap_map_iou (abs thresh 0.5) is identically 0 because activation
    never exceeds 1/F_P=0.125. Both make the softmax ablation's overlap-map signal
    degenerate."""
    F_p = F_P_DEFAULT
    P = 8 * F_p
    diffuse = torch.softmax(torch.randn(1, P), dim=-1)
    peaked = torch.softmax(torch.randn(1, P) * 6, dim=-1)
    a_d = overlap_time_activation(diffuse, f_p=F_p)
    a_p = overlap_time_activation(peaked, f_p=F_p)
    zero = torch.zeros_like(a_d)
    # The clean-clip term should DISTINGUISH a diffuse map from a peaked one if it is to
    # provide gradient toward 'empty'. Under softmax both have identical total mass, so
    # the Dice-vs-zero is identical → no signal. This asserts the term IS useful;
    # currently it is not (the two are equal), so the assert FAILS.
    d_diffuse = float(soft_dice_loss(a_d, zero))
    d_peaked = float(soft_dice_loss(a_p, zero))
    assert abs(d_diffuse - d_peaked) > 1e-3, (
        f"softmax mass conservation: clean-clip Dice is constant "
        f"({d_diffuse:.4f} == {d_peaked:.4f}) regardless of map shape → no gradient to "
        f"empty the map; also act.max={float(a_p.max()):.3f} < 0.5 so logged "
        f"overlap_map_iou (abs thresh 0.5) is identically 0 in softmax mode."
    )


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
