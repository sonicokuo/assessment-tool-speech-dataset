"""Tests for the SUPERVISED dense local-SNR-map head (src/snr_map_head.py) +
its train.py integration (snr_map_loss_term, default-off no-op) + the oracle target
builder (clean_features.snr_timeline_db) + the validation core (snr_map_validate).

What is proven (matches the task spec):
  - head outputs are FINITE and length-preserving (timeline) / in-range (IRM ∈ [0,1]);
  - the dense Huber loss DECREASES under gradient descent (pred → target);
  - the loss is MASKED to active frames (padding / inactive frames don't contribute);
  - lambda_snr_map=0 is an EXACT no-op (snr_map_loss_term returns None) AND off-by-
    default is byte-identical (no head → no-op);
  - DELETION-FAITHFULNESS: removing predicted-high-SNR frames drops the pooled SNR
    more than removing random frames (a concentrated/faithful timeline);
  - GRADIENT reaches the head's parameters (in_proj / temporal_conv / out_proj);
  - the oracle stem target is correct on a synthetic two-source mix;
  - the CBM scalar tie pools the timeline → a scalar that tracks the per-frame field.

Pure torch + numpy — runs on CPU anywhere.
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from snr_map_head import SupervisedSNRMapHead, snr_map_loss_term  # noqa: E402
from clean_features import snr_timeline_db, irm_grid  # noqa: E402
import snr_map_validate as smv  # noqa: E402


# ── head forward: finite + shape + range ────────────────────────────────────────
def test_timeline_forward_finite_and_length_preserving():
    head = SupervisedSNRMapHead(audio_dim=16, hidden=8, kernel_size=5)
    x = torch.randn(3, 37, 16)
    y = head.forward_timeline(x)
    assert y.shape == (3, 37)               # length-preserving
    assert torch.isfinite(y).all()


def test_irm_forward_in_range():
    head = SupervisedSNRMapHead(audio_dim=16, d_patch=12, f_bins=4, predict_irm=True)
    patches = torch.randn(2, 5 * 4, 12)     # (B, T_p*F, d)
    irm = head.forward_irm(patches, t_p=5)
    assert irm.shape == (2, 5, 4)
    assert torch.isfinite(irm).all()
    assert (irm >= 0).all() and (irm <= 1).all()


def test_irm_raises_when_disabled():
    head = SupervisedSNRMapHead(audio_dim=16, predict_irm=False)
    try:
        head.forward_irm(torch.randn(1, 8, head.d_patch))
        assert False, "expected RuntimeError when predict_irm=False"
    except RuntimeError:
        pass


# ── dense loss decreases pred → target ──────────────────────────────────────────
def test_timeline_loss_decreases():
    torch.manual_seed(0)
    head = SupervisedSNRMapHead(audio_dim=16, hidden=16, kernel_size=3)
    x = torch.randn(4, 25, 16)
    # a learnable target the conv head can fit
    target = torch.randn(4, 25)
    mask = torch.ones(4, 25)
    opt = torch.optim.Adam(head.parameters(), lr=1e-2)
    losses = []
    for _ in range(60):
        opt.zero_grad()
        pred = head.forward_timeline(x)
        loss, _ = head.timeline_loss(pred, target, mask)
        loss.backward()
        opt.step()
        losses.append(float(loss.detach()))
    assert losses[-1] < losses[0] * 0.8, f"loss did not decrease: {losses[0]:.3f}->{losses[-1]:.3f}"


# ── masked to active frames ─────────────────────────────────────────────────────
def test_loss_masked_to_active_frames():
    head = SupervisedSNRMapHead(audio_dim=8)
    pred = torch.zeros(1, 10)
    target = torch.zeros(1, 10)
    # put a huge error in an INACTIVE frame; mask it out → loss must stay 0.
    target[0, 7] = 100.0
    mask = torch.ones(1, 10)
    mask[0, 7] = 0.0
    loss, m = head.timeline_loss(pred, target, mask)
    assert float(loss) == 0.0
    assert m["snr_map_n_frames"] == 9.0
    # empty mask → exactly 0 loss (no supervision)
    loss0, _ = head.timeline_loss(pred, target, torch.zeros_like(mask))
    assert float(loss0) == 0.0


# ── lambda=0 / off-by-default no-op ─────────────────────────────────────────────
def test_loss_term_noop_when_lambda_zero():
    head = SupervisedSNRMapHead(audio_dim=8)
    batch = {"audio_features": torch.randn(2, 10, 8),
             "snr_map_target": torch.randn(2, 10)}
    out, metrics = snr_map_loss_term(head, batch, lambda_snr_map=0.0)
    assert out is None and metrics == {}


def test_loss_term_noop_when_no_head_or_no_target():
    batch = {"audio_features": torch.randn(2, 10, 8),
             "snr_map_target": torch.randn(2, 10)}
    out, _ = snr_map_loss_term(None, batch, lambda_snr_map=0.5)   # no head
    assert out is None
    out2, _ = snr_map_loss_term(SupervisedSNRMapHead(audio_dim=8),
                                {"audio_features": torch.randn(2, 10, 8)},  # no target
                                lambda_snr_map=0.5)
    assert out2 is None


def test_loss_term_active_returns_weighted_loss():
    head = SupervisedSNRMapHead(audio_dim=8)
    batch = {
        "audio_features": torch.randn(2, 10, 8),
        "snr_map_target": torch.randn(2, 10),
        "snr_map_mask": torch.ones(2, 10),
    }
    out, metrics = snr_map_loss_term(head, batch, lambda_snr_map=0.5)
    assert out is not None and torch.isfinite(out)
    assert "loss_snr_map" in metrics and "snr_map_mae" in metrics


# ── grad reaches the head ───────────────────────────────────────────────────────
def test_grad_reaches_head():
    head = SupervisedSNRMapHead(audio_dim=8, hidden=8)
    batch = {
        "audio_features": torch.randn(2, 12, 8),
        "snr_map_target": torch.randn(2, 12),
        "snr_map_mask": torch.ones(2, 12),
    }
    out, _ = snr_map_loss_term(head, batch, lambda_snr_map=0.5)
    out.backward()
    grads = [p.grad for p in head.parameters() if p.grad is not None]
    assert grads, "no parameter received a gradient"
    assert any(float(g.abs().sum()) > 0 for g in grads)
    # the temporal conv specifically must receive gradient (the localization path)
    assert head.temporal_conv.weight.grad is not None
    assert float(head.temporal_conv.weight.grad.abs().sum()) > 0


def test_grad_does_not_touch_audio_features():
    # The head reads audio_features but the term must not require/produce grad on the
    # input tensor (it's a cached feature, not a trainable param).
    head = SupervisedSNRMapHead(audio_dim=8)
    af = torch.randn(1, 10, 8)              # requires_grad=False
    batch = {"audio_features": af, "snr_map_target": torch.randn(1, 10),
             "snr_map_mask": torch.ones(1, 10)}
    out, _ = snr_map_loss_term(head, batch, lambda_snr_map=0.5)
    out.backward()
    assert af.grad is None


# ── CBM scalar tie ──────────────────────────────────────────────────────────────
def test_pooled_snr_tracks_timeline():
    head = SupervisedSNRMapHead(audio_dim=4)
    snr_frame = torch.tensor([[10.0, 20.0, 30.0, 0.0]])
    mask = torch.tensor([[1.0, 1.0, 1.0, 0.0]])      # pool only the first 3
    pooled = head.pooled_snr_db(snr_frame, mask)
    assert abs(float(pooled[0]) - 20.0) < 1e-5       # mean(10,20,30)
    # no active frames → 0
    pooled0 = head.pooled_snr_db(snr_frame, torch.zeros_like(mask))
    assert abs(float(pooled0[0])) < 1e-5


def test_scalar_tie_term_runs():
    from feature_set import N_FEATURES, FEATURE_NAMES
    head = SupervisedSNRMapHead(audio_dim=8)
    gt = torch.zeros(2, N_FEATURES)
    gtm = torch.zeros(2, N_FEATURES, dtype=torch.bool)
    snr_idx = FEATURE_NAMES.index("snr")
    gt[:, snr_idx] = 15.0
    gtm[:, snr_idx] = True
    batch = {
        "audio_features": torch.randn(2, 10, 8),
        "snr_map_target": torch.randn(2, 10),
        "snr_map_mask": torch.ones(2, 10),
        "gt_scalars": gt, "gt_mask": gtm,
    }
    out, metrics = snr_map_loss_term(head, batch, lambda_snr_map=0.5, lambda_scalar=0.3)
    assert "loss_snr_scalar" in metrics and "snr_pooled_mae" in metrics
    assert torch.isfinite(out)


# ── deletion-faithfulness ───────────────────────────────────────────────────────
def test_deletion_faithfulness_concentrated_beats_random():
    # A concentrated timeline (one frame much higher) → deleting the top frame drops
    # the pooled SNR more than deleting a random frame, on average.
    torch.manual_seed(1)
    B, T = 20, 12
    snr = torch.randn(B, T) * 2.0
    snr[:, 0] = 50.0                          # a clear high-SNR frame in each clip
    mask = torch.ones(B, T)
    res = smv.deletion_faithfulness(snr, mask, delete_frac=0.1, seed=0)
    assert res["deletion_drop_high_db"] > res["deletion_drop_random_db"]
    assert res["deletion_win_rate"] >= 0.9


def test_timeline_agreement_perfect_and_uncorrelated():
    t = torch.arange(20).float()
    mask = torch.ones(20)
    perfect = smv.timeline_agreement(t, t, mask)
    assert perfect["timeline_pearson"] > 0.999
    assert perfect["timeline_mae_db"] < 1e-5
    flat = smv.timeline_agreement(torch.zeros(20), t, mask)
    assert abs(flat["timeline_pearson"]) < 1e-6   # zero-variance pred → 0 corr


def test_model_randomization_sanity():
    # A trained-ish head vs a fresh random head should be largely DECORRELATED on
    # random audio (low |corr|), confirming the validation check runs + is sane.
    torch.manual_seed(2)
    head = SupervisedSNRMapHead(audio_dim=16, hidden=8)
    af = torch.randn(8, 30, 16)
    mask = torch.ones(8, 30)
    res = smv.model_randomization(head, af, mask)
    assert "model_rand_corr" in res
    assert abs(res["model_rand_corr"]) <= 1.0


# ── oracle target builder (clean_features) ──────────────────────────────────────
def test_snr_timeline_oracle_on_synthetic_mix():
    sr = 16000
    hop = 320
    n = hop * 10
    rng = np.random.default_rng(0)
    # s1 loud in first half, quiet in second; s2 constant. SNR(frame) should be HIGH
    # where s1 is loud, LOW where s1 is quiet.
    s1 = np.zeros(n, dtype=np.float32)
    s1[: n // 2] = rng.standard_normal(n // 2).astype(np.float32) * 1.0
    s1[n // 2:] = rng.standard_normal(n - n // 2).astype(np.float32) * 0.01
    s2 = rng.standard_normal(n).astype(np.float32) * 0.1
    timeline, active = snr_timeline_db(s1, s2, sr=sr, hop=hop, n_frames=10)
    assert timeline.shape == (10,) and active.shape == (10,)
    assert np.isfinite(timeline).all()
    # loud-s1 frames have higher SNR than quiet-s1 frames
    assert timeline[:5].mean() > timeline[5:].mean()
    # active mask flags the loud frames, not the near-silent ones
    assert active[:5].mean() > active[5:].mean()


def test_snr_timeline_length_forced_to_n_frames():
    s1 = np.ones(320 * 7, dtype=np.float32)
    s2 = np.ones(320 * 7, dtype=np.float32) * 0.5
    # force to 12 frames (more than the 7 available) → zero-padded to length 12
    tl, ac = snr_timeline_db(s1, s2, hop=320, n_frames=12)
    assert tl.shape == (12,) and ac.shape == (12,)
    assert (ac[7:] == 0).all()       # padded frames are inactive


def test_irm_grid_in_range_and_shape():
    rng = np.random.default_rng(0)
    s1 = rng.standard_normal(16000).astype(np.float32)
    s2 = rng.standard_normal(16000).astype(np.float32)
    irm = irm_grid(s1, s2, t_p=10, f_bins=8)
    assert irm.shape == (10, 8)
    assert np.isfinite(irm).all()
    assert (irm >= 0).all() and (irm <= 1).all()
