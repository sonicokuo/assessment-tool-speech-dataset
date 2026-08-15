"""Tests for H2 (Stirn stop-gradient in the NLL) and B3 (two-mask hedging).

Torch-dependent -> runs on PSC (or any env with torch). Validates the two behaviours the
audit flagged as easy to get wrong:
  - H2: with stop_grad_mean=True the NLL trains ONLY log_var, never the mean.
  - B3: the hedge mask is ILL_POSED-only (never a recoverable/counting feature), matches the
        canonical overlap>=0.5 rule, hedges the nums target, and leaves the PRESENCE mask
        (which the NLL uses) hedge-unaware.
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from model.reliability_head import heteroscedastic_nll  # noqa: E402
from data.feature_set import (  # noqa: E402
    FEATURE_NAMES,
    HEDGE_OVERLAP_TAU,
    ILL_POSED_UNDER_OVERLAP_FEATURES,
    N_FEATURES,
    RECOVERABLE_FEATURES,
    build_nums_target,
    extract_scalars,
    hedge_mask,
)

HEAVY = {  # overlap >= 0.5 -> ill-posed features abstained
    "snr_db": 15.66, "srmr": 5.15, "f0_mean_hz": 152.46, "f0_sd_hz": 53.18,
    "praat_speaking_rate_syl_sec": 5.61, "praat_pause_count": 3,
    "praat_pause_rate_per_min": 5.317, "overlap_ratio": 0.79,
    "jitter_local_pct": 2.7732, "shimmer": 14.1259, "hnr": 8.34,
}
LIGHT = {**HEAVY, "overlap_ratio": 0.30}   # below tau -> nothing hedged


# ── H2 Stirn ───────────────────────────────────────────────────────────────
def test_stirn_stops_gradient_to_mean():
    mean = torch.randn(4, N_FEATURES, requires_grad=True)
    log_var = torch.zeros(4, N_FEATURES, requires_grad=True)
    target = torch.randn(4, N_FEATURES)
    loss = heteroscedastic_nll(mean, log_var, target, stop_grad_mean=True)
    loss.backward()
    # mean has NO gradient path through the NLL; log_var does.
    assert mean.grad is None or mean.grad.abs().sum().item() == 0.0
    assert log_var.grad is not None and log_var.grad.abs().sum().item() > 0.0


def test_no_stirn_trains_mean():
    mean = torch.randn(4, N_FEATURES, requires_grad=True)
    log_var = torch.zeros(4, N_FEATURES, requires_grad=True)
    target = torch.randn(4, N_FEATURES)
    loss = heteroscedastic_nll(mean, log_var, target, stop_grad_mean=False)
    loss.backward()
    assert mean.grad is not None and mean.grad.abs().sum().item() > 0.0


def test_stirn_loss_value_matches_plain():
    # detaching the mean must not change the forward VALUE, only the gradient.
    torch.manual_seed(0)
    mean = torch.randn(3, N_FEATURES)
    log_var = torch.randn(3, N_FEATURES) * 0.1
    target = torch.randn(3, N_FEATURES)
    a = heteroscedastic_nll(mean, log_var, target, stop_grad_mean=True)
    b = heteroscedastic_nll(mean, log_var, target, stop_grad_mean=False)
    assert torch.allclose(a, b)


# ── B3 hedge mask ────────────────────────────────────────────────────────────
def test_hedge_mask_illposed_only_under_heavy_overlap():
    m = hedge_mask(HEAVY)
    for i, name in enumerate(FEATURE_NAMES):
        if name in ILL_POSED_UNDER_OVERLAP_FEATURES:
            assert bool(m[i]) is True, f"{name} should be hedged under heavy overlap"
        else:
            assert bool(m[i]) is False, f"recoverable/counting {name} must NEVER be hedged"


def test_hedge_mask_never_touches_recoverable():
    # invariant across overlap levels: no recoverable feature is ever hedged
    for row in (HEAVY, LIGHT, {"overlap_ratio": 1.0}, {}):
        m = hedge_mask(row)
        for i, name in enumerate(FEATURE_NAMES):
            if name in RECOVERABLE_FEATURES:
                assert bool(m[i]) is False


def test_hedge_mask_off_below_tau_and_missing():
    assert not hedge_mask(LIGHT).any()
    assert not hedge_mask({}).any()                      # overlap missing -> no hedge
    assert HEDGE_OVERLAP_TAU == 0.5


# ── B3 nums target + presence mask ──────────────────────────────────────────
def test_build_nums_target_hedges_illposed_keeps_recoverable():
    out = build_nums_target(HEAVY, hedge=True)
    for f in ("f0_mean", "f0_sd", "jitter", "shimmer", "hnr"):
        assert f"{f}=na" in out, f"{f} should be 'na' under heavy overlap"
    assert "snr=15.66" in out                            # recoverable kept
    assert "speaking_rate=5.61" in out   # F15: constant two-decimal surface form
    assert "pause_count=3" in out
    # without hedge, the ill-posed values are emitted
    out2 = build_nums_target(HEAVY, hedge=False)
    assert "f0_mean=152.46" in out2 and "hnr=8.34" in out2


def test_presence_mask_is_hedge_unaware():
    # extract_scalars still marks PRESENT (measured) slots True even under heavy overlap,
    # so the NLL keeps the hard f0/voice pairs. B3 must not change this.
    scalars, mask = extract_scalars(HEAVY)
    for f in ("f0_mean", "jitter", "hnr", "snr"):
        idx = FEATURE_NAMES.index(f)
        assert bool(mask[idx]) is True, f"{f} present value -> presence mask True (hedge-unaware)"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
