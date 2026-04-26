"""Diagnostic test: confirm the FiLMConditioning identity-init bug at adapter.py:80-83.

Hypothesis: FiLM gamma is initialized to weight=0, bias=1, so
    gamma(overlap_embed) = 0 * overlap_embed + 1 = 1   (constant, ignores overlap)
    beta(overlap_embed)  = 0 * overlap_embed + 0 = 0
    → forward = 1 * audio + 0 = audio                  (pure identity)

Consequence: at step 0, FiLM gives zero gradient to overlap_embed. The only path for
overlap to influence FiLM is via gamma.bias and beta.bias, both shared across clips —
so all clips look identical to FiLM until gamma.weight drifts away from zero.

Compare to ConcatOnlyAdapter, which routes overlap_info directly into a Linear+MLP,
giving full gradient flow on step 0. This explains why concat > film-mamba in the
test results (P=0.46/F1=0.55 vs P=0.37/F1=0.47).

We reimplement the FiLM class verbatim from src/adapter.py:74-89 here so the test
runs without mamba_ssm (not installed on Mac). Read both side-by-side: any change
to either definition should be mirrored.

Run: python -m pytest tests/test_film_init_diagnostic.py -v -s
"""

import torch
import torch.nn as nn


class FiLMConditioning(nn.Module):
    """VERBATIM copy of src/adapter.py::FiLMConditioning (lines 74-89, commit-current).

    DO NOT modify this without mirroring in src/adapter.py. The point of this file
    is to test that production code's behavior is what we think it is.
    """

    def __init__(self, lm_dim: int = 1024, overlap_dim: int = 32):
        super().__init__()
        self.gamma = nn.Linear(overlap_dim, lm_dim)
        self.beta = nn.Linear(overlap_dim, lm_dim)

        nn.init.zeros_(self.gamma.weight)   # ← BUG: zeros out overlap signal at init
        nn.init.ones_(self.gamma.bias)
        nn.init.zeros_(self.beta.weight)    # ← also zero, so beta starts at 0 regardless of overlap
        nn.init.zeros_(self.beta.bias)

    def forward(self, audio: torch.Tensor, overlap_embed: torch.Tensor) -> torch.Tensor:
        gamma = self.gamma(overlap_embed)
        beta = self.beta(overlap_embed)
        return gamma * audio + beta


def test_film_init_weights_match_production():
    """Direct read of the gamma/beta weights at init."""
    film = FiLMConditioning(lm_dim=64, overlap_dim=32)

    assert torch.allclose(film.gamma.weight, torch.zeros_like(film.gamma.weight))
    assert torch.allclose(film.gamma.bias, torch.ones_like(film.gamma.bias))
    assert torch.allclose(film.beta.weight, torch.zeros_like(film.beta.weight))
    assert torch.allclose(film.beta.bias, torch.zeros_like(film.beta.bias))

    print("\n[CONFIRMED] FiLM init at adapter.py:80-83:")
    print(f"  gamma.weight = zeros        (shape {tuple(film.gamma.weight.shape)})")
    print(f"  gamma.bias   = ones         (shape {tuple(film.gamma.bias.shape)})")
    print(f"  beta.weight  = zeros        (shape {tuple(film.beta.weight.shape)})")
    print(f"  beta.bias    = zeros        (shape {tuple(film.beta.bias.shape)})")


def test_film_forward_ignores_overlap_at_init():
    """At init, FiLM's output is identical regardless of what overlap_embed it sees."""
    torch.manual_seed(0)
    film = FiLMConditioning(lm_dim=64, overlap_dim=32)
    film.eval()

    audio = torch.randn(2, 10, 64)
    overlap_a = torch.randn(2, 10, 32)
    overlap_b = torch.randn(2, 10, 32) * 100  # very different overlap signal

    with torch.no_grad():
        out_a = film(audio, overlap_a)
        out_b = film(audio, overlap_b)

    assert torch.allclose(out_a, out_b, atol=1e-7), \
        f"Bug not reproduced: ||out_a - out_b|| = {(out_a - out_b).abs().max():.2e}"
    assert torch.allclose(out_a, audio, atol=1e-7), \
        "FiLM(audio, anything) at init should equal audio (pure identity)"

    print(f"\n[CONFIRMED] FiLM forward at init is pure identity:")
    print(f"  ||out_a - out_b|| = {(out_a - out_b).abs().max().item():.2e}  (should be 0)")
    print(f"  ||out_a - audio|| = {(out_a - audio).abs().max().item():.2e}  (should be 0)")


def test_overlap_grad_through_film_is_zero_at_init():
    """At init, gradient of FiLM output w.r.t. overlap_embed is zero.

    Chain rule:
        d(gamma * audio + beta) / d(overlap_embed)
            = audio * gamma.weight^T + beta.weight^T
            = audio * 0           + 0
            = 0
    """
    torch.manual_seed(0)
    film = FiLMConditioning(lm_dim=64, overlap_dim=32)

    audio = torch.randn(2, 10, 64)
    overlap = torch.randn(2, 10, 32, requires_grad=True)
    loss = film(audio, overlap).sum()
    loss.backward()

    grad_norm = overlap.grad.abs().max().item()
    assert grad_norm < 1e-7, \
        f"d(loss)/d(overlap) should be ~0 at init due to gamma.weight=zeros; got {grad_norm}"
    print(f"\n[CONFIRMED] At init, ||d(loss)/d(overlap)|| through FiLM = {grad_norm:.2e}")
    print("  Overlap signal contributes ZERO gradient through FiLM until gamma.weight drifts.")


def test_proposed_fix_admits_overlap_gradient():
    """Sanity-check the proposed fix: small-random gamma.weight init.

    With gamma.weight ~ N(0, std), the chain rule gives:
        d(loss)/d(overlap) = audio * gamma.weight^T  ≠  0
    So overlap gradient flows from step 0.

    Two candidate fixes (both keep the residual identity property at expectation):
      1. nn.init.normal_(gamma.weight, 0, 0.01) + ones(gamma.bias)
      2. nn.init.kaiming_uniform_(gamma.weight) + zero out + ones bias
                  with a small scale
    Either approach lets overlap signal flow on step 0.
    """
    torch.manual_seed(0)
    film = FiLMConditioning(lm_dim=64, overlap_dim=32)
    nn.init.normal_(film.gamma.weight, mean=0.0, std=0.01)   # PROPOSED FIX

    audio = torch.randn(2, 10, 64)
    overlap = torch.randn(2, 10, 32, requires_grad=True)
    loss = film(audio, overlap).sum()
    loss.backward()

    grad_norm = overlap.grad.abs().max().item()
    assert grad_norm > 1e-3, \
        f"After fix, overlap gradient should be substantial; got {grad_norm}"

    # Also confirm forward is approximately identity (residual property preserved at scale)
    film.eval()
    with torch.no_grad():
        out = film(audio, overlap)
    deviation = (out - audio).abs().mean().item()

    print(f"\n[FIX] gamma.weight ~ N(0, 0.01):")
    print(f"  overlap gradient at init: {grad_norm:.4f}   (was ~0)")
    print(f"  mean |FiLM(x) - x|:        {deviation:.4f}   (still ~0, residual preserved)")


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v", "-s"])
