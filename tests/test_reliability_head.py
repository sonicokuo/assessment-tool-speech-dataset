"""Tests for the heteroscedastic reliability/abstention head + its NLL.

Covers (deliverable requirement (a)):
  - the head outputs (B, F) mean + (B, F) sigma > 0;
  - the NLL is finite, decreases when the mean approaches the GT, and the head
    LEARNS to lower σ on a learnable feature and raise it on a pure-noise feature
    (variance calibration);
  - the NLL is masked by GT presence (absent slots contribute nothing).

Pure-torch, tiny shapes, CPU. No transformers/mamba needed for this file.
"""

import os
import sys
import types
import importlib.machinery

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

# mamba_ssm is CUDA-only; stub it so adapter.build_adapter imports on CPU.
if "mamba_ssm" not in sys.modules:
    _stub = types.ModuleType("mamba_ssm")
    _stub.__spec__ = importlib.machinery.ModuleSpec("mamba_ssm", loader=None)
    _stub.Mamba = object
    sys.modules["mamba_ssm"] = _stub

from reliability_head import (  # noqa: E402
    ReliabilityHead,
    heteroscedastic_nll,
    LOGVAR_MIN,
    LOGVAR_MAX,
)
from feature_set import N_FEATURES, FEATURE_SCALES  # noqa: E402


def test_head_output_shapes_and_positive_sigma():
    head = ReliabilityHead(d_in=16, n_features=N_FEATURES)
    pooled = torch.randn(5, 16)
    mean, log_var = head(pooled)
    assert mean.shape == (5, N_FEATURES)
    assert log_var.shape == (5, N_FEATURES)
    sigma = ReliabilityHead.sigma(log_var)
    assert sigma.shape == (5, N_FEATURES)
    assert torch.all(sigma > 0), "sigma must be strictly positive"
    # log_var is clamped into the documented stability band.
    assert torch.all(log_var >= LOGVAR_MIN - 1e-6)
    assert torch.all(log_var <= LOGVAR_MAX + 1e-6)


def test_logvar_bias_inits_to_zero():
    """At init the log-var rows' bias is 0 (σ²=1 in normalized units) so the head
    starts at the plain-MSE operating point."""
    head = ReliabilityHead(d_in=8, n_features=N_FEATURES)
    bias = head.proj.bias.detach()
    assert torch.allclose(bias[N_FEATURES:], torch.zeros(N_FEATURES), atol=1e-7)


def test_nll_finite_and_decreases_as_mean_approaches_gt():
    F = N_FEATURES
    target = torch.randn(3, F)
    log_var = torch.zeros(3, F)  # σ²=1
    scales = torch.ones(F)

    far = target + 5.0            # large error
    near = target + 0.01         # tiny error
    nll_far = heteroscedastic_nll(far, log_var, target, scales=scales)
    nll_near = heteroscedastic_nll(near, log_var, target, scales=scales)

    assert torch.isfinite(nll_far) and torch.isfinite(nll_near)
    assert nll_near < nll_far, "NLL must drop as the mean approaches GT"


def test_nll_masked_by_presence():
    F = N_FEATURES
    mean = torch.randn(2, F)
    log_var = torch.zeros(2, F)
    target = torch.randn(2, F)

    full_mask = torch.ones(2, F, dtype=torch.bool)
    zero_mask = torch.zeros(2, F, dtype=torch.bool)

    nll_full = heteroscedastic_nll(mean, log_var, target, mask=full_mask)
    nll_zero = heteroscedastic_nll(mean, log_var, target, mask=zero_mask)

    assert nll_full > 0.0
    # Nothing present → masked sum is 0 / clamp(denom) = 0.
    assert float(nll_zero) == pytest.approx(0.0, abs=1e-7)

    # Masking one column out changes the average (it's a mean over present slots).
    half_mask = full_mask.clone()
    half_mask[:, 0] = False
    nll_half = heteroscedastic_nll(mean, log_var, target, mask=half_mask)
    assert torch.isfinite(nll_half)


def test_optimal_sigma_for_constant_error_matches_closed_form():
    """For a FIXED error e (mean fixed, only log_var trainable), the NLL is minimized
    at σ² = e², i.e. log_var* = 2*log|e|. Verify the head's variance channel calibrates
    to the noise level via gradient descent."""
    torch.manual_seed(0)
    F = N_FEATURES
    target = torch.zeros(1, F)
    fixed_error = 2.0
    mean = torch.full((1, F), fixed_error)  # constant error of 2.0, not trained
    scales = torch.ones(F)

    log_var = torch.zeros(1, F, requires_grad=True)
    opt = torch.optim.Adam([log_var], lr=0.1)
    for _ in range(2000):
        opt.zero_grad()
        loss = heteroscedastic_nll(mean, log_var.clamp(LOGVAR_MIN, LOGVAR_MAX),
                                   target, scales=scales)
        loss.backward()
        opt.step()

    expected = 2.0 * torch.log(torch.tensor(fixed_error))  # log(e^2)
    assert torch.allclose(log_var.detach().mean(), expected, atol=0.1), (
        f"log_var converged to {log_var.detach().mean():.3f}, expected {expected:.3f}"
    )


def test_head_raises_sigma_on_noise_feature_lowers_on_signal_feature():
    """End-to-end calibration: train a head where feature 0 is a clean linear function
    of the input and feature 1 is pure label noise. After training, σ(noise) > σ(signal):
    the head learned to flag the unpredictable feature as unreliable."""
    torch.manual_seed(0)
    d_in = 8
    head = ReliabilityHead(d_in=d_in, n_features=2)
    opt = torch.optim.Adam(head.parameters(), lr=0.02)

    W = torch.randn(d_in)  # the true signal direction for feature 0
    scales = torch.ones(2)
    for _ in range(800):
        x = torch.randn(64, d_in)
        y0 = x @ W                         # learnable
        y1 = torch.randn(64)               # pure noise, unlearnable from x
        target = torch.stack([y0, y1], dim=1)
        mean, log_var = head(x)
        opt.zero_grad()
        loss = heteroscedastic_nll(mean, log_var, target, scales=scales)
        loss.backward()
        opt.step()

    with torch.no_grad():
        x = torch.randn(256, d_in)
        _, log_var = head(x)
        sigma = ReliabilityHead.sigma(log_var).mean(dim=0)
    assert sigma[1] > sigma[0], (
        f"noise-feature sigma {sigma[1]:.3f} should exceed signal-feature sigma {sigma[0]:.3f}"
    )


def test_nll_gradients_flow_to_both_mean_and_logvar():
    head = ReliabilityHead(d_in=8, n_features=N_FEATURES)
    x = torch.randn(4, 8)
    mean, log_var = head(x)
    target = torch.randn(4, N_FEATURES)
    scales = torch.tensor(FEATURE_SCALES)
    loss = heteroscedastic_nll(mean, log_var, target, scales=scales)
    loss.backward()
    g = head.proj.weight.grad
    assert g is not None
    # Gradient must reach BOTH the mean rows [:F] and the log-var rows [F:].
    assert g[:N_FEATURES].abs().sum() > 0, "no gradient to the mean rows"
    assert g[N_FEATURES:].abs().sum() > 0, "no gradient to the log-var rows"


def test_build_adapter_reliability_flag_default_off_is_plain_linear():
    """build_adapter default (reliability_head=False) keeps the plain Linear mean head,
    and forward returns (prefix, tensor) — byte-identical signature to before."""
    from adapter import build_adapter, AdapterWithAuxHead
    import torch.nn as nn

    a = build_adapter("concat-only", lm_dim=8, n_aux_features=N_FEATURES)
    assert isinstance(a, AdapterWithAuxHead)
    assert not a.reliability_head
    assert isinstance(a.regress_head, nn.Linear)
    audio = torch.randn(2, 64, 1024)
    overlap = torch.randn(2, 64, 4)
    prefix, scalar_pred = a(audio, overlap)
    assert torch.is_tensor(scalar_pred)
    assert scalar_pred.shape == (2, N_FEATURES)


def test_build_adapter_reliability_flag_on_returns_mean_logvar():
    """build_adapter(reliability_head=True) installs ReliabilityHead and forward returns
    (prefix, (mean, log_var)) each (B, F), with sigma > 0."""
    from adapter import build_adapter, AdapterWithAuxHead

    a = build_adapter("concat-only", lm_dim=8, n_aux_features=N_FEATURES,
                      reliability_head=True)
    assert isinstance(a, AdapterWithAuxHead)
    assert a.reliability_head
    assert isinstance(a.regress_head, ReliabilityHead)
    audio = torch.randn(2, 64, 1024)
    overlap = torch.randn(2, 64, 4)
    prefix, out = a(audio, overlap)
    assert isinstance(out, tuple) and len(out) == 2
    mean, log_var = out
    assert mean.shape == (2, N_FEATURES)
    assert log_var.shape == (2, N_FEATURES)
    assert torch.all(ReliabilityHead.sigma(log_var) > 0)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
