"""Tests for the TRUE-2D SRMR modulation-energy map feature.

Covers (per the task spec):
  - srmr_maps.srmr_map_target tensor SHAPE (23x8) and RANGE (finite log-energy map);
  - the AGGREGATE read back from the stored tensor matches the scalar SRMR the same
    module reports (srmr_scalar_from_avg(avg, kstar) == target['srmr_scalar']) — i.e.
    the scalar is read THROUGH the dense field (the CBM property), exact for whichever
    path (SRMRpy internals or the self-contained fallback) is available;
  - a REVERBERANT signal shows the expected low->high modulation-energy shift (lower
    SRMR than its anechoic source) — the physics the 2D field is meant to localize;
  - the SupervisedSRMRMapHead forward is finite + correct shape; pooled_srmr is finite;
  - lambda_srmr_map=0 is an EXACT no-op (srmr_map_loss_term returns None) AND off-by-
    default is byte-identical (no head -> no-op, missing target -> no-op);
  - the 2D map Huber DECREASES under gradient descent (pred -> target), masked.

Pure numpy + torch — runs on CPU anywhere. The SRMR signal path uses SRMRpy when
present and a self-contained gammatone+modulation-filterbank fallback otherwise, so
these tests pass with or without the SRMRpy / gammatone packages installed.
"""
import ast
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import srmr_maps as sm  # noqa: E402
from snr_map_head import (  # noqa: E402
    SupervisedSRMRMapHead, srmr_map_loss_term,
    SRMR_N_ACOUSTIC, SRMR_N_MODULATION,
)


def _tone(fs=16000, dur=1.5, f0=140.0, n_harm=6, seed=0):
    """A voiced-like harmonic test signal (clean, anechoic)."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(fs * dur)) / fs
    x = np.zeros_like(t)
    for k in range(1, n_harm + 1):
        x += (1.0 / k) * np.sin(2 * np.pi * f0 * k * t)
    # gentle amplitude modulation (syllabic ~3 Hz) so there's low-modulation energy
    x *= (0.6 + 0.4 * np.sin(2 * np.pi * 3.0 * t))
    x += 0.01 * rng.standard_normal(x.shape)
    return (x / (np.abs(x).max() + 1e-9)).astype(np.float32)


def _reverberate(x, fs=16000, rt60=0.6, seed=1):
    """Convolve with a synthetic exponentially-decaying noise RIR (smears modulation)."""
    rng = np.random.default_rng(seed)
    n = int(rt60 * fs)
    rir = rng.standard_normal(n) * np.exp(-6.9 * np.arange(n) / n)  # -60 dB at rt60
    rir[0] += 1.0  # direct path
    y = np.convolve(x, rir)[: len(x)]
    return (y / (np.abs(y).max() + 1e-9)).astype(np.float32)


# ── tensor shape + range ────────────────────────────────────────────────────────
def test_target_shape_and_range():
    x = _tone()
    tgt = sm.srmr_map_target(x, fs=16000)
    assert tgt["srmr_logmap"].shape == (SRMR_N_ACOUSTIC, SRMR_N_MODULATION)  # 23 x 8
    assert tgt["srmr_avg"].shape == (SRMR_N_ACOUSTIC, SRMR_N_MODULATION)
    assert tgt["srmr_mask"].shape == (SRMR_N_ACOUSTIC, SRMR_N_MODULATION)
    assert np.isfinite(tgt["srmr_logmap"]).all()
    assert (tgt["srmr_avg"] >= 0).all()            # energy is non-negative
    assert np.isfinite(tgt["srmr_scalar"]) and tgt["srmr_scalar"] > 0
    assert 5 <= tgt["kstar"] <= SRMR_N_MODULATION


def test_empty_signal_is_safe():
    tgt = sm.srmr_map_target(np.zeros(8, dtype=np.float32), fs=16000)
    assert tgt["srmr_logmap"].shape == (SRMR_N_ACOUSTIC, SRMR_N_MODULATION)
    assert (tgt["srmr_mask"] == 0).all()           # all-zero mask -> no supervision


# ── aggregate read THROUGH the tensor matches the scalar SRMR (CBM property) ─────
def test_aggregate_matches_scalar():
    x = _tone(seed=3)
    tgt = sm.srmr_map_target(x, fs=16000)
    agg = sm.srmr_scalar_from_avg(tgt["srmr_avg"], tgt["kstar"])
    # the stored scalar IS this aggregate (read through the dense field), exact.
    assert abs(agg - tgt["srmr_scalar"]) < 1e-4


# ── reverberation shifts energy low->high modulation bands (lowers SRMR) ─────────
def test_reverberation_lowers_srmr():
    clean = _tone(seed=5)
    rev = _reverberate(clean, rt60=0.7, seed=6)
    s_clean = sm.srmr_map_target(clean, fs=16000)["srmr_scalar"]
    s_rev = sm.srmr_map_target(rev, fs=16000)["srmr_scalar"]
    assert np.isfinite(s_clean) and np.isfinite(s_rev)
    # reverberation smears the envelope -> more high-modulation energy -> lower SRMR.
    assert s_rev < s_clean, f"expected reverb SRMR {s_rev:.3f} < clean {s_clean:.3f}"


# ── head forward: finite + shape; pooled scalar finite ──────────────────────────
def test_head_forward_shape_and_pool():
    head = SupervisedSRMRMapHead(audio_dim=16, n_acoustic=23, n_modulation=8, hidden=8)
    x = torch.randn(4, 29, 16)
    y = head.forward(x)
    assert y.shape == (4, 23, 8)
    assert torch.isfinite(y).all()
    pooled = head.pooled_srmr(y, kstar=8)
    assert pooled.shape == (4,)
    assert torch.isfinite(pooled).all()


def test_head_respects_audio_lens_pooling():
    head = SupervisedSRMRMapHead(audio_dim=12, hidden=8)
    x = torch.randn(2, 20, 12)
    lens = torch.tensor([20, 5])
    y = head.forward(x, audio_lens=lens)
    assert y.shape == (2, 23, 8) and torch.isfinite(y).all()


# ── lambda_srmr_map = 0 is an EXACT no-op; missing head / target no-op ───────────
def test_loss_term_noop_when_lambda_zero():
    head = SupervisedSRMRMapHead(audio_dim=16, hidden=8)
    batch = {
        "audio_features": torch.randn(2, 10, 16),
        "srmr_map_target": torch.randn(2, 23, 8),
    }
    loss, metrics = srmr_map_loss_term(head, batch, lambda_srmr_map=0.0)
    assert loss is None and metrics == {}


def test_loss_term_noop_when_no_head():
    batch = {"audio_features": torch.randn(2, 10, 16), "srmr_map_target": torch.randn(2, 23, 8)}
    loss, metrics = srmr_map_loss_term(None, batch, lambda_srmr_map=0.3)
    assert loss is None and metrics == {}


def test_loss_term_noop_when_no_target():
    head = SupervisedSRMRMapHead(audio_dim=16, hidden=8)
    batch = {"audio_features": torch.randn(2, 10, 16)}  # no srmr_map_target
    loss, metrics = srmr_map_loss_term(head, batch, lambda_srmr_map=0.3)
    assert loss is None and metrics == {}


# ── the 2D map loss DECREASES pred -> target (masked Huber learns) ───────────────
def test_map_loss_decreases():
    torch.manual_seed(0)
    head = SupervisedSRMRMapHead(audio_dim=16, n_acoustic=23, n_modulation=8, hidden=32)
    audio = torch.randn(3, 12, 16)
    target = torch.randn(3, 23, 8)
    batch = {"audio_features": audio, "srmr_map_target": target,
             "audio_lens": torch.tensor([12, 12, 12])}
    opt = torch.optim.Adam(head.parameters(), lr=0.05)

    loss0 = None
    for step in range(60):
        opt.zero_grad()
        loss, metrics = srmr_map_loss_term(head, batch, lambda_srmr_map=1.0)
        assert loss is not None
        assert "loss_srmr_map" in metrics and "srmr_map_mae" in metrics
        if loss0 is None:
            loss0 = float(loss.item())
        loss.backward()
        # gradient reaches the head's params
        assert head.out_proj.weight.grad is not None
        opt.step()
    loss_final = float(loss.item())
    assert loss_final < loss0 * 0.5, f"loss did not decrease: {loss0:.4f} -> {loss_final:.4f}"


def test_map_loss_masked():
    """Bands with mask=0 must not contribute to the loss/MAE."""
    head = SupervisedSRMRMapHead(audio_dim=8, hidden=8)
    audio = torch.randn(1, 6, 8)
    target = torch.zeros(1, 23, 8)
    mask = torch.zeros(1, 23, 8)  # fully masked -> loss must be ~0 regardless of pred
    batch = {"audio_features": audio, "srmr_map_target": target, "srmr_map_mask": mask}
    loss, metrics = srmr_map_loss_term(head, batch, lambda_srmr_map=1.0)
    assert loss is not None
    assert float(loss.item()) == 0.0
    assert metrics["srmr_map_mae"] == 0.0


# ── source files parse (ast) ─────────────────────────────────────────────────────
def test_source_files_parse():
    here = os.path.dirname(__file__)
    for rel in ("../src/srmr_maps.py", "../scripts/compute_srmr_maps.py",
                "../src/snr_map_head.py"):
        path = os.path.join(here, rel)
        with open(path) as f:
            ast.parse(f.read())
